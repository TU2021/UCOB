# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_micro_batch_with_logits(
        self,
        micro_batch,
        temperature,
        kl_topk_k: int = None,
        kl_topk_indices: torch.Tensor = None,
        calculate_entropy=False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass that also exposes logits gathered on a top-k support.

        ``kl_topk_k`` asks this model to choose its own top-k tokens. Passing
        ``kl_topk_indices`` gathers this model's logits on another view's top-k
        support, which is what DSDAR Teacher-TopK uses during actor update.
        """
        if kl_topk_k is None and kl_topk_indices is None:
            raise ValueError("Must provide either kl_topk_k or kl_topk_indices")
        if kl_topk_k is not None and kl_topk_indices is not None:
            raise ValueError("kl_topk_k and kl_topk_indices are mutually exclusive")
        if self.use_fused_kernels:
            raise NotImplementedError("Teacher-TopK needs raw logits; disable use_fused_kernels.")

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)

                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=False,
                )
                need_logsumexp = not _as_bool(self.config.get("sdar_topk_norm_to_one", True), default=True)
                if need_logsumexp:
                    logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1)
                else:
                    logsumexp_rmpad = logits_rmpad.new_zeros(logits_rmpad.shape[0])
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                if kl_topk_k == -1:
                    logits_k_rmpad = logits_rmpad
                    topk_indices_rmpad = None
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices_rmpad = logits_rmpad.topk(kl_topk_k, dim=-1)
                    logits_k_rmpad = logits_rmpad.gather(-1, topk_indices_rmpad)
                else:
                    logits_k_rmpad = None
                    topk_indices_rmpad = None

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    logsumexp_rmpad = gather_outpus_and_unpad(logsumexp_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if logits_k_rmpad is not None:
                        logits_k_rmpad = gather_outpus_and_unpad(logits_k_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    else:
                        logits_rmpad = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if topk_indices_rmpad is not None:
                        topk_indices_rmpad = gather_outpus_and_unpad(topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                if kl_topk_k is None and kl_topk_indices is not None:
                    assert kl_topk_indices.shape[0] == batch_size, (
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    )
                    assert kl_topk_indices.shape[1] == response_length, (
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"
                    )

                    total_nnz = logsumexp_rmpad.shape[0]
                    inverse_indices = torch.full((batch_size * seqlen,), -1, dtype=torch.long, device=logsumexp_rmpad.device)
                    inverse_indices[indices] = torch.arange(total_nnz, device=logsumexp_rmpad.device)

                    response_start = seqlen - response_length - 1
                    batch_offsets = torch.arange(batch_size, device=logsumexp_rmpad.device) * seqlen
                    response_offsets = torch.arange(response_length, device=logsumexp_rmpad.device)
                    flattened_response_pos = batch_offsets.unsqueeze(1) + response_start + response_offsets.unsqueeze(0)
                    rmpad_response_pos = inverse_indices[flattened_response_pos]
                    valid_mask = rmpad_response_pos >= 0
                    safe_rmpad_pos = rmpad_response_pos.clamp(min=0)

                    k = kl_topk_indices.shape[-1]
                    rmpad_pos_expanded = safe_rmpad_pos.unsqueeze(-1).expand(-1, -1, k)
                    logits_k = logits_rmpad[rmpad_pos_expanded.reshape(-1), kl_topk_indices.reshape(-1)]
                    logits_k = logits_k.reshape(batch_size, response_length, k)
                    logsumexp = logsumexp_rmpad[safe_rmpad_pos]
                    logits_k = logits_k * valid_mask.unsqueeze(-1)
                    logsumexp = logsumexp * valid_mask

                    full_log_probs = pad_input(log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                    if calculate_entropy:
                        full_entropy = pad_input(entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}
                else:
                    full_log_probs = pad_input(log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logits_k = pad_input(logits_k_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logsumexp = pad_input(logsumexp_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                    logits_k = full_logits_k[:, -response_length - 1 : -1, :]
                    logsumexp = full_logsumexp.squeeze(-1)[:, -response_length - 1 : -1]
                    if calculate_entropy:
                        full_entropy = pad_input(entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

                    if kl_topk_k == -1:
                        kl_inputs = {"logits_k": logits_k, "topk_indices": None, "logsumexp": logsumexp}
                    else:
                        full_topk_indices = pad_input(topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]
                        kl_inputs = {"logits_k": logits_k, "topk_indices": topk_indices, "logsumexp": logsumexp}

            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]
                log_probs = logprobs_from_logits(logits=logits, labels=micro_batch["responses"], inplace_backward=False)
                need_logsumexp = not _as_bool(self.config.get("sdar_topk_norm_to_one", True), default=True)
                if need_logsumexp:
                    logsumexp = torch.logsumexp(logits, dim=-1)
                else:
                    logsumexp = logits.new_zeros(logits.shape[:-1])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)

                if kl_topk_k == -1:
                    kl_inputs = {"logits_k": logits, "topk_indices": None, "logsumexp": logsumexp}
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices = logits.topk(kl_topk_k, dim=-1)
                    logits_k = logits.gather(-1, topk_indices)
                    kl_inputs = {"logits_k": logits_k, "topk_indices": topk_indices, "logsumexp": logsumexp}
                else:
                    assert kl_topk_indices is not None
                    assert kl_topk_indices.shape[0] == batch_size, (
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    )
                    assert kl_topk_indices.shape[1] == response_length, (
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"
                    )
                    logits_k = logits.gather(-1, kl_topk_indices)
                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}

            assert kl_inputs["logsumexp"].shape == log_probs.shape
            assert kl_inputs["logits_k"].shape[0] == log_probs.shape[0]
            assert kl_inputs["logits_k"].shape[1] == log_probs.shape[1]
            if kl_inputs["topk_indices"] is not None:
                assert kl_inputs["topk_indices"].shape == kl_inputs["logits_k"].shape

            return entropy, log_probs, kl_inputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_with_logits(self, data: DataProto, kl_topk_k: int) -> tuple:
        """Compute sampled log-probs plus logits on this model's top-k support."""
        assert kl_topk_k == -1 or kl_topk_k > 0, (
            f"kl_topk_k must be -1 (full logits) or >0 (top-k), got {kl_topk_k}"
        )
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        calculate_entropy = _as_bool(data.meta_info.get("calculate_entropy", True), default=True)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        kl_inputs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    micro_batch=micro_batch,
                    temperature=temperature,
                    kl_topk_k=kl_topk_k,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            kl_inputs_lst.append(kl_inputs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        output_kl_inputs = {
            "logits_k": torch.concat([item["logits_k"] for item in kl_inputs_lst], dim=0),
            "logsumexp": torch.concat([item["logsumexp"] for item in kl_inputs_lst], dim=0),
        }
        if kl_topk_k > 0:
            output_kl_inputs["topk_indices"] = torch.concat(
                [item["topk_indices"] for item in kl_inputs_lst],
                dim=0,
            )
        else:
            output_kl_inputs["topk_indices"] = None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=log_probs.device)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
            output_kl_inputs["logits_k"] = output_kl_inputs["logits_k"][revert_indices]
            output_kl_inputs["logsumexp"] = output_kl_inputs["logsumexp"][revert_indices]
            if kl_topk_k > 0:
                output_kl_inputs["topk_indices"] = output_kl_inputs["topk_indices"][revert_indices]

        return log_probs, entropys, output_kl_inputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        use_loss_mask = multi_turn or _as_bool(self.config.get("action_loss_mask_enable", False))

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if use_loss_mask:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.get("use_sdl_loss", False) or self.config.get("use_sdar_loss", False):
            select_keys.append("teacher_log_probs")
        if self.config.get("use_sdar_loss", False) and "sdar_gate_sign" in data.batch.keys():
            select_keys.append("sdar_gate_sign")
        if self.config.get("use_sdar_loss", False) and "sdar_loss_weight" in data.batch.keys():
            select_keys.append("sdar_loss_weight")
        if self.config.get("use_sdar_loss", False) and "sdar_direction_id" in data.batch.keys():
            select_keys.append("sdar_direction_id")
        ucob_cbsd_enabled = _as_bool(self.config.get("ucob_cbsd_enable", False), default=False)
        ucob_cbsd_loss_type = str(self.config.get("ucob_cbsd_loss_type", "nll")).strip().lower()
        ucob_cbsd_topk_opd_enabled = (
            ucob_cbsd_enabled
            and ucob_cbsd_loss_type in {"topk_opd", "topk-opd", "topk"}
        )
        ucob_cbsd_sampled_opd_enabled = (
            ucob_cbsd_enabled
            and ucob_cbsd_loss_type == "opd"
        )
        if ucob_cbsd_enabled and "ucob_cbsd_loss_mask" in data.batch.keys():
            select_keys.append("ucob_cbsd_loss_mask")
            if "ucob_cbsd_loss_weight" in data.batch.keys():
                select_keys.append("ucob_cbsd_loss_weight")
            if (
                ucob_cbsd_sampled_opd_enabled
                and "ucob_cbsd_ref_log_probs" in data.batch.keys()
            ):
                select_keys.append("ucob_cbsd_ref_log_probs")
            if (
                ucob_cbsd_topk_opd_enabled
                and "ucob_cbsd_ref_topk_indices" in data.batch.keys()
                and "ucob_cbsd_ref_logits_k" in data.batch.keys()
                and "ucob_cbsd_ref_logsumexp" in data.batch.keys()
                and "ucob_cbsd_ref_log_probs" in data.batch.keys()
            ):
                select_keys.extend(
                    [
                        "ucob_cbsd_ref_topk_indices",
                        "ucob_cbsd_ref_logits_k",
                        "ucob_cbsd_ref_logsumexp",
                        "ucob_cbsd_ref_log_probs",
                    ]
                )
        sdar_topk_enabled = _as_bool(self.config.get("sdar_topk_enable", False))
        sdar_topk_action_mask_enabled = _as_bool(self.config.get("sdar_topk_action_mask_enable", False))
        sdar_topk_gate_enabled = _as_bool(self.config.get("sdar_topk_gate_enable", True), default=True)
        if self.config.get("use_sdar_loss", False) and sdar_topk_enabled and "dsdar_ref_topk_indices" in data.batch.keys():
            select_keys.extend(["dsdar_ref_logits_k", "dsdar_ref_logsumexp", "dsdar_ref_topk_indices"])
            if sdar_topk_action_mask_enabled and "action_token_mask" in data.batch.keys():
                select_keys.append("action_token_mask")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if use_loss_mask:
                        base_response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        base_response_mask = attention_mask[:, -response_length:]
                    response_mask = base_response_mask
                    cbsd_loss_mask = None
                    cbsd_loss_weight = None
                    cbsd_raw_full_tokens = None
                    cbsd_raw_response_tokens = None
                    if ucob_cbsd_enabled and "ucob_cbsd_loss_mask" in data:
                        raw_cbsd_loss_mask = data["ucob_cbsd_loss_mask"].to(
                            device=base_response_mask.device,
                            dtype=torch.float32,
                        )
                        cbsd_raw_full_tokens = raw_cbsd_loss_mask.sum()
                        if raw_cbsd_loss_mask.shape[-1] == attention_mask.shape[-1]:
                            cbsd_loss_mask = raw_cbsd_loss_mask[:, -response_length:]
                        else:
                            cbsd_loss_mask = raw_cbsd_loss_mask
                        cbsd_raw_response_tokens = cbsd_loss_mask.sum()
                        cbsd_loss_mask = cbsd_loss_mask * attention_mask[:, -response_length:].to(
                            dtype=cbsd_loss_mask.dtype
                        )
                        cbsd_row_mask = (cbsd_loss_mask.sum(dim=-1, keepdim=True) > 0).to(
                            dtype=torch.float32
                        )
                        response_mask = base_response_mask * (1.0 - cbsd_row_mask)
                        if "ucob_cbsd_loss_weight" in data:
                            cbsd_loss_weight = data["ucob_cbsd_loss_weight"].to(
                                device=base_response_mask.device,
                                dtype=torch.float32,
                            )
                            if cbsd_loss_weight.dim() == 1:
                                cbsd_loss_weight = cbsd_loss_weight.unsqueeze(-1)
                            if cbsd_loss_weight.shape[-1] != response_length:
                                cbsd_loss_weight = cbsd_loss_weight[:, -response_length:]
                    topk_response_mask = response_mask
                    if sdar_topk_action_mask_enabled and "action_token_mask" in data:
                        action_token_mask = data["action_token_mask"][:, -response_length:].to(
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        )
                        topk_response_mask = response_mask * action_token_mask

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    has_dsdar_topk = (
                        self.config.get("use_sdar_loss", False)
                        and sdar_topk_enabled
                        and "dsdar_ref_topk_indices" in data.keys()
                        and "sdar_direction_id" in data.keys()
                    )
                    has_cbsd_topk_opd = (
                        ucob_cbsd_topk_opd_enabled
                        and "ucob_cbsd_ref_topk_indices" in data.keys()
                        and "ucob_cbsd_ref_logits_k" in data.keys()
                        and "ucob_cbsd_ref_logsumexp" in data.keys()
                        and "ucob_cbsd_ref_log_probs" in data.keys()
                    )
                    has_cbsd_sampled_opd = (
                        ucob_cbsd_sampled_opd_enabled
                        and "ucob_cbsd_ref_log_probs" in data.keys()
                    )
                    if has_dsdar_topk and has_cbsd_topk_opd:
                        raise ValueError("DSDAR top-k and UCOB cbsd top-k OPD cannot be enabled in the same actor batch.")
                    if has_dsdar_topk:
                        entropy, log_prob, dsdar_topk_inputs = self._forward_micro_batch_with_logits(
                            micro_batch=data,
                            temperature=temperature,
                            kl_topk_indices=data["dsdar_ref_topk_indices"],
                            calculate_entropy=calculate_entropy,
                        )
                        cbsd_topk_inputs = None
                    elif has_cbsd_topk_opd:
                        entropy, log_prob, cbsd_topk_inputs = self._forward_micro_batch_with_logits(
                            micro_batch=data,
                            temperature=temperature,
                            kl_topk_indices=data["ucob_cbsd_ref_topk_indices"],
                            calculate_entropy=calculate_entropy,
                        )
                        dsdar_topk_inputs = None
                    else:
                        dsdar_topk_inputs = None
                        cbsd_topk_inputs = None
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if ucob_cbsd_enabled and cbsd_loss_mask is not None:
                        if cbsd_loss_weight is None:
                            cbsd_loss_weight = torch.ones_like(cbsd_loss_mask)
                        cbsd_coef = float(self.config.get("ucob_cbsd_coef", 0.05))
                        cbsd_extra_metrics = {}
                        if has_cbsd_topk_opd and cbsd_topk_inputs is not None:
                            from verl.trainer.ppo.sdar_utils import compute_dsdar_topk_loss

                            cbsd_loss, cbsd_extra_metrics = compute_dsdar_topk_loss(
                                target_logits_k=cbsd_topk_inputs["logits_k"],
                                target_logsumexp=cbsd_topk_inputs["logsumexp"],
                                reference_logits_k=data["ucob_cbsd_ref_logits_k"],
                                reference_logsumexp=data["ucob_cbsd_ref_logsumexp"],
                                response_mask=cbsd_loss_mask,
                                target_log_probs=log_prob,
                                reference_log_probs=data["ucob_cbsd_ref_log_probs"],
                                gate_enable=_as_bool(
                                    self.config.get("ucob_cbsd_gate_enable", True),
                                    default=True,
                                ),
                                gate_beta=float(self.config.get("ucob_cbsd_gate_beta", 5.0)),
                                norm_to_one=_as_bool(
                                    self.config.get("ucob_cbsd_topk_norm_to_one", True),
                                    default=True,
                                ),
                                clip_log_ratio=_as_bool(
                                    self.config.get("ucob_cbsd_topk_clip_log_ratio", False),
                                    default=False,
                                ),
                                loss_weight=cbsd_loss_weight,
                                loss_agg_mode=loss_agg_mode,
                            )
                        elif has_cbsd_sampled_opd:
                            from verl.trainer.ppo.sdar_utils import compute_sdar_loss

                            cbsd_loss, cbsd_extra_metrics = compute_sdar_loss(
                                target_log_probs=log_prob,
                                reference_log_probs=data["ucob_cbsd_ref_log_probs"],
                                response_mask=cbsd_loss_mask,
                                loss_weight=cbsd_loss_weight,
                                gate_enable=_as_bool(
                                    self.config.get("ucob_cbsd_gate_enable", True),
                                    default=True,
                                ),
                                gate_beta=float(self.config.get("ucob_cbsd_gate_beta", 5.0)),
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            cbsd_loss_mat = -log_prob * cbsd_loss_weight
                            cbsd_loss = agg_loss(
                                loss_mat=cbsd_loss_mat,
                                loss_mask=cbsd_loss_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                        policy_loss = policy_loss + cbsd_loss * cbsd_coef
                        with torch.no_grad():
                            valid_response_tokens = attention_mask[:, -response_length:].to(dtype=cbsd_loss_mask.dtype)
                            valid_count = valid_response_tokens.sum().clamp(min=1.0)
                            cbsd_tokens = cbsd_loss_mask.sum()
                            cbsd_rows = (cbsd_loss_mask.sum(dim=-1) > 0).to(dtype=cbsd_loss_mask.dtype)
                            cbsd_row_count = cbsd_rows.sum()
                            raw_full_count = (
                                cbsd_raw_full_tokens
                                if cbsd_raw_full_tokens is not None
                                else torch.zeros((), device=cbsd_loss_mask.device)
                            )
                            raw_response_count = (
                                cbsd_raw_response_tokens
                                if cbsd_raw_response_tokens is not None
                                else torch.zeros((), device=cbsd_loss_mask.device)
                            )
                        append_to_dict(
                            metrics,
                            {
                                "ucob/cbsd_loss": cbsd_loss.detach().item(),
                                "ucob/cbsd_coef": cbsd_coef,
                                "ucob/cbsd_token_fraction": (cbsd_tokens / valid_count).item(),
                                "ucob/cbsd_row_fraction": cbsd_rows.float().mean().item(),
                                "ucob/cbsd_raw_full_token_count_max": raw_full_count.item(),
                                "ucob/cbsd_raw_response_token_count_max": raw_response_count.item(),
                                "ucob/cbsd_active_token_count_max": cbsd_tokens.item(),
                                "ucob/cbsd_active_row_count_max": cbsd_row_count.item(),
                            },
                        )
                        if cbsd_extra_metrics:
                            if has_cbsd_topk_opd:
                                metric_prefix = "ucob/cbsd_topk_opd"
                                metric_key = "sdar/topk"
                            else:
                                metric_prefix = "ucob/cbsd_sampled_opd"
                                metric_key = "sdar"
                            append_to_dict(
                                metrics,
                                {
                                    key.replace(metric_key, metric_prefix): value
                                    for key, value in cbsd_extra_metrics.items()
                                },
                            )

                    if self.config.get("use_sdl_loss", False):
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        from verl.trainer.ppo.sdar_utils import compute_dsdar_topk_loss, compute_sdar_loss
                        reference_log_probs = data["teacher_log_probs"]
                        gate_delta_sign = data.get("sdar_gate_sign", None)
                        sdar_loss_weight = data.get("sdar_loss_weight", None)
                        sdar_direction_id = data.get("sdar_direction_id", None)
                        sampled_sdar_loss_weight = sdar_loss_weight
                        if has_dsdar_topk and sdar_direction_id is not None:
                            correction_mask = ((sdar_direction_id == 1) | (sdar_direction_id == 2)).to(dtype=response_mask.dtype)
                            if sampled_sdar_loss_weight is None:
                                sampled_sdar_loss_weight = 1.0 - correction_mask
                            else:
                                sampled_sdar_loss_weight = sampled_sdar_loss_weight * (1.0 - correction_mask)
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            target_log_probs=log_prob,
                            reference_log_probs=reference_log_probs,
                            response_mask=response_mask,
                            gate_delta_sign=gate_delta_sign,
                            loss_weight=sampled_sdar_loss_weight,
                            direction_ids=sdar_direction_id,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef
                        if has_dsdar_topk:
                            topk_loss, topk_metrics = compute_dsdar_topk_loss(
                                target_logits_k=dsdar_topk_inputs["logits_k"],
                                target_logsumexp=dsdar_topk_inputs["logsumexp"],
                                reference_logits_k=data["dsdar_ref_logits_k"],
                                reference_logsumexp=data["dsdar_ref_logsumexp"],
                                response_mask=topk_response_mask,
                                target_log_probs=log_prob,
                                reference_log_probs=reference_log_probs,
                                gate_delta_sign=gate_delta_sign,
                                loss_weight=sdar_loss_weight,
                                direction_ids=sdar_direction_id,
                                gate_enable=sdar_topk_gate_enabled,
                                gate_beta=self.config.get("sdar_gate_beta", 5.0),
                                norm_to_one=_as_bool(self.config.get("sdar_topk_norm_to_one", True), default=True),
                                clip_log_ratio=_as_bool(self.config.get("sdar_topk_clip_log_ratio", False), default=False),
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss + topk_loss * sdar_coef
                            metrics.update(topk_metrics)


                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
