"""Main entry point for UCOB training.

UCOB collects mixed skill/no-skill rollouts, groups records by task and
anchor state, and applies CBSD from the higher-return context view to the
opposite view while evolving the dual-granularity skill memory.
"""

import copy
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from agent_system.multi_turn_rollout.utils import filter_group_data, to_list_of_dict, torch_to_numpy


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _plain_cfg(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        converted = OmegaConf.to_container(value, resolve=True)
    except Exception:
        converted = value
    return dict(converted or {}) if isinstance(converted, dict) else {}


def _merge_cfg(*configs) -> dict:
    merged = {}
    for cfg in configs:
        merged.update(_plain_cfg(cfg))
    return merged


def _legacy_cbsd_cfg(config) -> dict:
    legacy = _plain_cfg(config.algorithm.get("dmgigpo", {}))
    if not legacy:
        return {}
    mapped = {}
    for key, value in legacy.items():
        if key.startswith("xctx_"):
            mapped[key[len("xctx_"):]] = value
        else:
            mapped[key] = value
    return mapped


def _ucob_cfg(config) -> dict:
    return _plain_cfg(config.algorithm.get("ucob", {}))


def _cbsd_cfg(config) -> dict:
    ucob_cfg = _ucob_cfg(config)
    return _merge_cfg(_legacy_cbsd_cfg(config), ucob_cfg.get("cbsd", {}))


def _skill_memory_cfg(config) -> dict:
    ucob_cfg = _ucob_cfg(config)
    return _merge_cfg(
        config.algorithm.get("dynamic_memory", {}),
        ucob_cfg.get("skill_memory", {}),
        ucob_cfg.get("reflection", {}),
    )


_ACTION_SPAN_PATTERNS = [
    re.compile(r"<action\b[^>]*>(.*?)</action>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<code\b[^>]*>(.*?)</code>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<search\b[^>]*>(.*?)</search>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[action>(.*?)</action>\]?", re.IGNORECASE | re.DOTALL),
    re.compile(r"action>(.*?)</action>", re.IGNORECASE | re.DOTALL),
]
_UNCLOSED_ACTION_SPAN_PATTERN = re.compile(r"<action\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
_UNCLOSED_CODE_SPAN_PATTERN = re.compile(r"<code\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
_UNCLOSED_SEARCH_SPAN_PATTERN = re.compile(r"<search\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
_UNCLOSED_ANSWER_SPAN_PATTERN = re.compile(r"<answer\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
_WEBSHOP_ACTION_PATTERN = re.compile(r"(?:search|click)\[[^\]\n]*\]", re.IGNORECASE)


def _extract_search_env_kwargs(non_tensor_batch):
    tools_kwargs = non_tensor_batch.get("tools_kwargs")
    if tools_kwargs is None:
        return None

    env_kwargs = []
    for item in tools_kwargs:
        create_kwargs = None
        if isinstance(item, dict):
            create_kwargs = item.get("search", {}).get("create_kwargs")
        env_kwargs.append(create_kwargs)

    if not env_kwargs or any(value is None for value in env_kwargs):
        return None
    return np.array(env_kwargs, dtype=object)


@hydra.main(config_path="config", config_name="ucob_trainer", version_base=None)
def main(config):
    run_ucob(config)


def run_ucob(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = UCOBTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class UCOBTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        dsdar_cfg = config.algorithm.get("dsdar", config.algorithm.get("sdar", {}))
        cbsd_cfg = _cbsd_cfg(config)
        skill_memory_cfg = _skill_memory_cfg(config)
        teacher_loss_enable = _as_bool(dsdar_cfg.get("teacher_loss_enable", True), default=True)
        with open_dict(config):
            config.algorithm.dynamic_memory = OmegaConf.create(skill_memory_cfg)
            config.actor_rollout_ref.actor.use_sdar_loss = teacher_loss_enable
            config.actor_rollout_ref.actor.sdar_loss_coef = dsdar_cfg.get("sdar_coef", 0.1) if teacher_loss_enable else 0.0
            config.actor_rollout_ref.actor.sdar_gate_beta = dsdar_cfg.get("gate_beta", 5.0)
            config.actor_rollout_ref.actor.sdar_topk_enable = _as_bool(dsdar_cfg.get("topk_enable", False))
            config.actor_rollout_ref.actor.sdar_topk_k = dsdar_cfg.get("topk_k", 32)
            config.actor_rollout_ref.actor.sdar_topk_norm_to_one = dsdar_cfg.get("topk_norm_to_one", True)
            config.actor_rollout_ref.actor.sdar_topk_clip_log_ratio = dsdar_cfg.get("topk_clip_log_ratio", False)
            config.actor_rollout_ref.actor.sdar_topk_gate_enable = _as_bool(dsdar_cfg.get("topk_gate_enable", True))
            config.actor_rollout_ref.actor.sdar_topk_action_mask_enable = _as_bool(
                dsdar_cfg.get("topk_action_mask_enable", False)
            )
            config.actor_rollout_ref.actor.action_loss_mask_enable = _as_bool(
                dsdar_cfg.get("action_loss_mask_enable", False)
            )
            config.actor_rollout_ref.actor.ucob_cbsd_enable = _as_bool(
                cbsd_cfg.get("enable", False),
                default=False,
            )
            config.actor_rollout_ref.actor.ucob_cbsd_coef = cbsd_cfg.get("coef", 0.05)
            config.actor_rollout_ref.actor.ucob_cbsd_weight_max = cbsd_cfg.get("weight_max", 2.0)
            config.actor_rollout_ref.actor.ucob_cbsd_loss_type = cbsd_cfg.get("loss_type", "nll")
            config.actor_rollout_ref.actor.ucob_cbsd_topk_k = cbsd_cfg.get("topk_k", 32)
            config.actor_rollout_ref.actor.ucob_cbsd_gate_enable = _as_bool(
                cbsd_cfg.get("gate_enable", True),
                default=True,
            )
            config.actor_rollout_ref.actor.ucob_cbsd_gate_beta = cbsd_cfg.get("gate_beta", 5.0)
            config.actor_rollout_ref.actor.ucob_cbsd_topk_norm_to_one = _as_bool(
                cbsd_cfg.get("topk_norm_to_one", True),
                default=True,
            )
            config.actor_rollout_ref.actor.ucob_cbsd_topk_clip_log_ratio = _as_bool(
                cbsd_cfg.get("topk_clip_log_ratio", False),
                default=False,
            )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        skills_dir = dsdar_cfg.get("skills_dir", "skills/alfworld")
        skill_all = dsdar_cfg.get("skill_all", False)
        rollout_mode = dsdar_cfg.get("rollout_mode", "mixed")
        adaptive_enable = dsdar_cfg.get("adaptive_enable", False)
        action_loss_mask_enable = _as_bool(dsdar_cfg.get("action_loss_mask_enable", False))
        rlrt_enable = dsdar_cfg.get("rlrt_enable", False)
        correct_coef = dsdar_cfg.get("correct_coef", 1.0)
        rlrt_coef = dsdar_cfg.get("rlrt_coef", 1.0)
        aux_target_branch = dsdar_cfg.get("aux_target_branch", "both")
        topk_enable = dsdar_cfg.get("topk_enable", False)
        topk_k = dsdar_cfg.get("topk_k", 32)
        topk_norm_to_one = dsdar_cfg.get("topk_norm_to_one", True)
        topk_clip_log_ratio = dsdar_cfg.get("topk_clip_log_ratio", False)
        topk_gate_enable = _as_bool(dsdar_cfg.get("topk_gate_enable", True))
        topk_action_mask_enable = _as_bool(dsdar_cfg.get("topk_action_mask_enable", False))
        force_skill_from_no_skill = _as_bool(dsdar_cfg.get("force_skill_from_no_skill", False), default=False)
        bad_memory_unidirectional = _as_bool(dsdar_cfg.get("bad_memory_unidirectional", True), default=True)
        dynamic_memory_cfg = _skill_memory_cfg(config)
        dynamic_memory_enable = _as_bool(dynamic_memory_cfg.get("enable", False))
        if dynamic_memory_enable:
            skill_provider = None
            print("[UCOB] Skill memory enabled; fixed skillbank initialization is skipped.")
        else:
            from verl.trainer.ppo.rlsd_utils import SkillProvider

            skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
            print(f"[UCOB] Loaded fixed skills from {skills_dir}")
            print(f"[UCOB] Available skills: {list(skill_provider.skill_contents.keys())}")
            print(f"[UCOB] Task-to-skill mapping: {skill_provider.task_to_skill}")
        print(f"[UCOB] legacy_sdar_coef: {config.actor_rollout_ref.actor.sdar_loss_coef}")
        print(f"[UCOB] legacy_teacher_loss_enable: {teacher_loss_enable}")
        print(f"[UCOB] legacy_gate_beta: {config.actor_rollout_ref.actor.sdar_gate_beta}")
        print(f"[UCOB] rollout_mode: {rollout_mode}")
        print(f"[UCOB] legacy_adaptive_enable: {adaptive_enable}")
        print(f"[UCOB] action_loss_mask_enable: {action_loss_mask_enable}")
        print(f"[UCOB] legacy_rlrt_enable: {rlrt_enable}")
        print(f"[UCOB] legacy_correct_coef: {correct_coef}")
        print(f"[UCOB] legacy_rlrt_coef: {rlrt_coef}")
        print(f"[UCOB] legacy_aux_target_branch: {aux_target_branch}")
        print(f"[UCOB] legacy_topk_enable: {topk_enable}")
        print(f"[UCOB] legacy_topk_k: {topk_k}")
        print(f"[UCOB] legacy_topk_norm_to_one: {topk_norm_to_one}")
        print(f"[UCOB] legacy_topk_clip_log_ratio: {topk_clip_log_ratio}")
        print(f"[UCOB] legacy_topk_gate_enable: {topk_gate_enable}")
        print(f"[UCOB] legacy_topk_action_mask_enable: {topk_action_mask_enable}")
        print(f"[UCOB] legacy_force_skill_from_no_skill: {force_skill_from_no_skill}")
        print(f"[UCOB] legacy_bad_memory_unidirectional: {bad_memory_unidirectional}")

        traj_collector = UCOBTrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            skill_provider=skill_provider,
        )

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.ucob_ray_trainer import UCOBRayTrainer

        class UCOBTrainer(UCOBRayTrainer):
            def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
                if "dsdar_teacher_prompts" not in batch.batch.keys():
                    return super()._compute_teacher_log_probs(batch)

                response_length = batch.batch["responses"].size(1)
                teacher_prompts = batch.batch["dsdar_teacher_prompts"]
                teacher_prompt_mask = batch.batch["dsdar_teacher_prompt_attention_mask"]
                response_mask = batch.batch["attention_mask"][:, -response_length:]

                teacher_input_ids = torch.cat([teacher_prompts, batch.batch["responses"]], dim=-1)
                teacher_attention_mask = torch.cat([teacher_prompt_mask, response_mask], dim=-1)
                teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

                teacher_batch = DataProto.from_dict(
                    tensors={
                        "input_ids": teacher_input_ids,
                        "attention_mask": teacher_attention_mask,
                        "position_ids": teacher_position_ids,
                        "responses": batch.batch["responses"],
                    }
                )
                dsdar_cfg = self.config.algorithm.get("dsdar", self.config.algorithm.get("sdar", {}))
                if _as_bool(dsdar_cfg.get("topk_enable", False)):
                    teacher_batch.meta_info["kl_topk_k"] = int(dsdar_cfg.get("topk_k", 32))
                    teacher_batch.meta_info["calculate_entropy"] = False
                    teacher_output = self.actor_rollout_wg.compute_log_prob_with_logits(teacher_batch)
                    batch.batch["dsdar_ref_logits_k"] = teacher_output.batch["actor_logits_k"]
                    batch.batch["dsdar_ref_logsumexp"] = teacher_output.batch["actor_logsumexp"]
                    batch.batch["dsdar_ref_topk_indices"] = teacher_output.batch["actor_topk_indices"]
                else:
                    teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
                return teacher_output.batch["old_log_probs"]

            def _prepare_state_value_gen_batch(self, test_data) -> DataProto:
                from verl import DataProto

                test_batch = DataProto.from_single_dict(test_data)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "state_value_eval": True,
                }
                return test_gen_batch

            def _collect_motivation_rollout_rows(
                self,
                test_gen_batch: DataProto,
                *,
                rollout_mode: str,
                run_id: str,
                tasks_per_rollout: int,
                rollout_group_n: int,
                val_env_batch_size: int,
                task_key_offset: int,
                task_key_suffix: str = "",
            ) -> list[dict]:
                from verl.trainer.ucob_state_value_eval import extract_turn_rows_from_rollout

                traj_collector = self.traj_collector
                use_skill = rollout_mode == "skill"
                rollout_skill_mask_value = use_skill
                rows: list[dict] = []
                batch_len = len(test_gen_batch)

                for task_start in range(0, batch_len, tasks_per_rollout):
                    task_end = min(task_start + tasks_per_rollout, batch_len)
                    task_chunk = test_gen_batch.select_idxs(np.arange(task_start, task_end))
                    chunk_size = len(task_chunk)
                    task_keys = np.array(
                        [
                            f"val_task_{task_key_offset + task_start + i}{task_key_suffix}"
                            for i in range(chunk_size)
                        ],
                        dtype=object,
                    )

                    rollout_batch = task_chunk.repeat(
                        repeat_times=rollout_group_n,
                        interleave=True,
                    )
                    if len(rollout_batch) > val_env_batch_size:
                        raise ValueError(
                            "state_value_eval rollout batch exceeds val env capacity: "
                            f"{len(rollout_batch)} > {val_env_batch_size}."
                        )

                    eval_task_keys = np.repeat(task_keys, rollout_group_n)
                    sample_indices = np.tile(np.arange(rollout_group_n), chunk_size)
                    rollout_skill_mask = np.full(len(rollout_batch), rollout_skill_mask_value, dtype=bool)

                    (
                        total_batch_list,
                        total_episode_rewards,
                        total_episode_lengths,
                        total_success,
                        total_traj_uid,
                        total_tool_callings,
                    ) = traj_collector.vanilla_multi_turn_loop(
                        gen_batch=rollout_batch,
                        actor_rollout_wg=self.actor_rollout_wg,
                        envs=self.val_envs,
                        rollout_skill_mask=rollout_skill_mask,
                        is_train=False,
                    )
                    del total_episode_lengths, total_tool_callings
                    step_returns_by_traj = traj_collector._compute_memory_step_returns(total_batch_list)
                    terminal_success = total_success.get("success_rate")
                    rows.extend(
                        extract_turn_rows_from_rollout(
                            total_batch_list,
                            step_returns_by_traj,
                            total_episode_rewards,
                            terminal_success,
                            eval_task_keys=eval_task_keys,
                            sample_indices=sample_indices,
                            rollout_mode=rollout_mode,
                            run_id=run_id,
                            traj_uids=total_traj_uid,
                        )
                    )
                return rows

            def collect_state_value_eval(self) -> dict:
                """Run separate skill/no-skill rollouts and export motivation CSVs."""
                from verl.trainer.ucob_state_value_eval import write_motivation_csvs

                sve_cfg = self.config.trainer.get("state_value_eval", {}) or {}
                run_id = str(sve_cfg.get("run_id", self.config.trainer.experiment_name))
                output_dir = str(
                    sve_cfg.get(
                        "output_dir",
                        f"outputs/motivation_eval/{run_id}",
                    )
                )
                gap_margin = float(sve_cfg.get("gap_margin", 0.05))
                repeat_times = int(self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1))
                rollout_group_n = int(self.config.env.rollout.n)
                val_env_batch_size = int(self.config.data.val_batch_size)
                if rollout_group_n <= 1:
                    raise ValueError(
                        "state_value_eval requires env.rollout.n > 1 for multi-sample per task."
                    )
                if val_env_batch_size % rollout_group_n != 0:
                    raise ValueError(
                        "state_value_eval requires data.val_batch_size to be divisible by env.rollout.n; "
                        f"got val_batch_size={val_env_batch_size}, rollout.n={rollout_group_n}."
                    )
                tasks_per_rollout = val_env_batch_size // rollout_group_n

                all_rows: list[dict] = []
                task_key_offset = 0
                for test_data in self.val_dataloader:
                    probe_batch = self._prepare_state_value_gen_batch(test_data)
                    batch_len = len(probe_batch)
                    del probe_batch
                    for repeat_idx in range(repeat_times):
                        repeat_suffix = f"_r{repeat_idx}" if repeat_times > 1 else ""
                        for rollout_mode in ("skill", "no_skill"):
                            test_gen_batch = self._prepare_state_value_gen_batch(test_data)
                            all_rows.extend(
                                self._collect_motivation_rollout_rows(
                                    test_gen_batch,
                                    rollout_mode=rollout_mode,
                                    run_id=run_id,
                                    tasks_per_rollout=tasks_per_rollout,
                                    rollout_group_n=rollout_group_n,
                                    val_env_batch_size=val_env_batch_size,
                                    task_key_offset=task_key_offset,
                                    task_key_suffix=repeat_suffix,
                                )
                            )
                    task_key_offset += batch_len

                if not all_rows:
                    raise RuntimeError(
                        "state_value_eval produced no turn rows; check val data, env.rollout.n, and memory config."
                    )

                csv_paths = write_motivation_csvs(
                    all_rows,
                    run_id=run_id,
                    output_dir=output_dir,
                    gap_margin=gap_margin,
                )
                if int(csv_paths["num_valid_turn_summary_rows"]) <= 0:
                    raise RuntimeError(
                        "state_value_eval produced no valid uid+turn pairs with both skill and no_skill branches."
                    )

                skill_rows = [row for row in all_rows if row["rollout_mode"] == "skill"]
                no_skill_rows = [row for row in all_rows if row["rollout_mode"] == "no_skill"]

                metrics = {
                    "state_value_eval/run_id": run_id,
                    "state_value_eval/output_dir": output_dir,
                    "state_value_eval/num_turn_rows": float(len(all_rows)),
                    "state_value_eval/num_turn_rows_skill": float(len(skill_rows)),
                    "state_value_eval/num_turn_rows_no_skill": float(len(no_skill_rows)),
                    "state_value_eval/num_turn_summary_rows": float(csv_paths["num_turn_summary_rows"]),
                    "state_value_eval/num_valid_turn_summary_rows": float(csv_paths["num_valid_turn_summary_rows"]),
                    "state_value_eval/num_traj_summary_rows": float(csv_paths["num_traj_summary_rows"]),
                    "state_value_eval/turn_rows_csv": str(csv_paths["turn_rows_csv"]),
                    "state_value_eval/turn_summary_csv": str(csv_paths["turn_summary_csv"]),
                    "state_value_eval/traj_summary_csv": str(csv_paths["traj_summary_csv"]),
                }
                print(
                    "[state_value_eval] wrote "
                    f"{len(all_rows)} turn rows, "
                    f"{csv_paths['num_turn_summary_rows']} turn-summary rows "
                    f"({csv_paths['num_valid_turn_summary_rows']} with valid pairs) "
                    f"to {output_dir}"
                )
                return metrics

        trainer = UCOBTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
        )
        original_validate = trainer._validate
        state_value_eval_cfg = config.trainer.get("state_value_eval", {}) or {}
        state_value_eval_enable = _as_bool(state_value_eval_cfg.get("enable", False), default=False)

        if state_value_eval_enable:
            def state_value_validate():
                return trainer.collect_state_value_eval()

            trainer._validate = state_value_validate
        else:
            def dual_validate():
                traj_collector.eval_inject_skill = True
                traj_collector.dynamic_memory_eval_stats_history = []
                metrics_with_skill = original_validate()
                metrics_with_skill.update(
                    trainer._dynamic_memory_stats_metrics(
                        trainer._aggregate_dynamic_memory_stats(
                            getattr(traj_collector, "dynamic_memory_eval_stats_history", [])
                        ),
                        prefix="val",
                    )
                )
                with_prefix = "val_with_skill"
                prefixed_with = {
                    f"{with_prefix}/{k.replace('val/', '')}": v
                    for k, v in metrics_with_skill.items()
                }

                traj_collector.eval_inject_skill = False
                metrics_no_skill = original_validate()
                no_prefix = "val_no_skill"
                prefixed_no = {
                    f"{no_prefix}/{k.replace('val/', '')}": v
                    for k, v in metrics_no_skill.items()
                }

                combined = {}
                combined.update(prefixed_with)
                combined.update(prefixed_no)
                combined.update(metrics_with_skill)
                return combined

            trainer._validate = dual_validate
        trainer.init_workers()
        trainer.fit()


class UCOBTrajectoryCollector:
    """
    Dual-view rollout collector.

    During training:
      1. Build both no-skill and skill-augmented prompt batches.
      2. Within each rollout group, sample half from no-skill and half from skill.
      3. Store the opposite-view prompt so the trainer can compute cross-view
         teacher/reference log-probabilities on the same response.

    During validation, ``eval_inject_skill`` controls whether rollout sees
    skill text, enabling paired with-skill/no-skill evaluation.
    """

    def __init__(self, config, tokenizer, processor=None, skill_provider=None):
        from agent_system.multi_turn_rollout import TrajectoryCollector

        self._collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.skill_provider = skill_provider
        self.eval_inject_skill = False
        dsdar_cfg = config.algorithm.get("dsdar", config.algorithm.get("sdar", {}))
        self.action_loss_mask_enable = _as_bool(dsdar_cfg.get("action_loss_mask_enable", False))
        self.topk_action_mask_enable = _as_bool(dsdar_cfg.get("topk_action_mask_enable", False))
        self.teacher_loss_enable = _as_bool(dsdar_cfg.get("teacher_loss_enable", True), default=True)
        cbsd_cfg = _cbsd_cfg(config)
        self.cbsd_enable = _as_bool(cbsd_cfg.get("enable", False), default=False)
        self.cbsd_action_mask_enable = _as_bool(cbsd_cfg.get("action_mask_enable", True), default=True)
        self.cbsd_margin = float(cbsd_cfg.get("margin", 0.05))
        self.cbsd_tau = max(float(cbsd_cfg.get("tau", 0.2)), 1e-6)
        self.cbsd_weight_max = float(cbsd_cfg.get("weight_max", 2.0))
        self.cbsd_force_source_teacher = _as_bool(
            cbsd_cfg.get("force_source_teacher", False),
            default=False,
        )
        self.cbsd_force_gap = float(cbsd_cfg.get("force_gap", self.cbsd_tau))
        self.cbsd_max_pairs_per_group = int(cbsd_cfg.get("max_pairs_per_group", 2))
        self.cbsd_loss_type = str(cbsd_cfg.get("loss_type", "nll")).strip().lower()
        self.cbsd_opd_granularity = str(cbsd_cfg.get("opd_granularity", "turn")).strip().lower()
        if self.cbsd_opd_granularity not in {"turn", "trajectory", "hybrid"}:
            raise ValueError(
                "Unsupported algorithm.ucob.cbsd.opd_granularity="
                f"{self.cbsd_opd_granularity!r}; choose 'turn', 'trajectory', or 'hybrid'."
            )
        self.cbsd_opd_scope = str(cbsd_cfg.get("opd_scope", "action")).strip().lower()
        if self.cbsd_opd_scope not in {"action", "response", "full_response"}:
            raise ValueError(
                "Unsupported algorithm.ucob.cbsd.opd_scope="
                f"{self.cbsd_opd_scope!r}; choose 'action' or 'response'."
            )
        self.cbsd_action_weight = float(cbsd_cfg.get("action_weight", 1.0))
        self.cbsd_max_turns_per_trajectory = int(cbsd_cfg.get("max_turns_per_trajectory", 0))
        self.cbsd_source_branches = self._parse_branch_set(
            cbsd_cfg.get("source_branches", "good,none")
        )
        self.cbsd_target_policy = str(cbsd_cfg.get("target_policy", "opposite")).strip().lower()
        analysis_cfg = config.algorithm.get("analysis", {})
        if not isinstance(analysis_cfg, dict):
            analysis_cfg = OmegaConf.to_container(analysis_cfg, resolve=True)
        case_debug_env = str(os.environ.get("UCOB_CASE_STUDY_DEBUG", "") or "").strip().lower()
        self.case_study_debug_enable = _as_bool(
            analysis_cfg.get("case_study_debug_enable", False),
            default=False,
        ) or case_debug_env in {"1", "true", "yes", "on"}
        self.case_study_debug_dir = str(
            os.environ.get(
                "UCOB_CASE_STUDY_DEBUG_DIR",
                analysis_cfg.get("case_study_debug_dir", "outputs/case_study_debug"),
            )
        )
        self.case_study_debug_max_per_step = max(
            int(analysis_cfg.get("case_study_debug_max_per_step", 8) or 8),
            0,
        )
        self.case_study_debug_min_gap = float(
            analysis_cfg.get("case_study_debug_min_gap", self.cbsd_margin)
        )
        self.case_study_debug_include_prompts = _as_bool(
            analysis_cfg.get("case_study_debug_include_prompts", False),
            default=False,
        )
        self.case_study_debug_include_full_trajectory = _as_bool(
            analysis_cfg.get("case_study_debug_include_full_trajectory", False),
            default=False,
        )
        self.case_study_written_memories = []
        if self.case_study_debug_enable:
            os.makedirs(self.case_study_debug_dir, exist_ok=True)
            print(
                "[UCOB][case_study] enabled: "
                f"dir={self.case_study_debug_dir}, "
                f"max_per_step={self.case_study_debug_max_per_step}, "
                f"min_gap={self.case_study_debug_min_gap}"
            )
        dynamic_memory_cfg = _skill_memory_cfg(config)
        self.dynamic_memory_enabled = _as_bool(dynamic_memory_cfg.get("enable", False))
        self.dynamic_memory_provider = None
        self.last_dynamic_memory_stats = {}
        self.dynamic_memory_eval_stats_history = []
        self.dynamic_memory_state_aware_retrieval = _as_bool(
            dynamic_memory_cfg.get("state_aware_retrieval", False),
            default=False,
        )
        self.dynamic_memory_step_reflection_enable = _as_bool(
            dynamic_memory_cfg.get("step_reflection_enable", False),
            default=False,
        )
        self.dynamic_memory_step_reflection_per_group = int(dynamic_memory_cfg.get("step_reflection_per_group", 1))
        self.dynamic_memory_step_reflection_min_gap = float(dynamic_memory_cfg.get("step_reflection_min_gap", 0.05))
        self.dynamic_memory_step_utility_tau = float(dynamic_memory_cfg.get("step_utility_tau", 1.0))
        self.dynamic_memory_max_context_tokens = int(
            dynamic_memory_cfg.get(
                "max_context_tokens",
                dynamic_memory_cfg.get("max_skill_context_tokens", 1024),
            )
        )
        self.dynamic_memory_reflection_max_prompt_tokens = int(
            dynamic_memory_cfg.get("reflection_max_prompt_tokens", 4096)
        )
        self.dynamic_memory_prompt_template_reserve_tokens = int(
            dynamic_memory_cfg.get("prompt_template_reserve_tokens", 128)
        )
        self.current_training_step = 0
        self.total_training_steps = 0
        self.reflection_train_enable = _as_bool(dynamic_memory_cfg.get("reflection_train_enable", False), default=False)
        self.reflection_train_source = str(
            dynamic_memory_cfg.get("reflection_train_source", "current_step_used_memory")
        ).strip().lower()
        self.reflection_train_warmup_steps = int(
            dynamic_memory_cfg.get(
                "reflection_train_warmup_steps",
                dynamic_memory_cfg.get("reflection_train_warmup", 0),
            )
        )
        self.reflection_train_max_task_samples_per_step = int(
            dynamic_memory_cfg.get("reflection_train_max_task_samples_per_step", 16)
        )
        self.reflection_train_max_state_samples_per_step = int(
            dynamic_memory_cfg.get("reflection_train_max_state_samples_per_step", 16)
        )
        self.reflection_train_min_lifetime_count_task = int(
            dynamic_memory_cfg.get("reflection_train_min_lifetime_count_task", 1)
        )
        self.reflection_train_min_lifetime_count_state = int(
            dynamic_memory_cfg.get("reflection_train_min_lifetime_count_state", 10)
        )
        self.reflection_train_min_abs_utility = float(
            dynamic_memory_cfg.get("reflection_train_min_abs_utility", 0.02)
        )
        self.reflection_reward_mode = str(dynamic_memory_cfg.get("reflection_reward_mode", "batch_zscore")).strip().lower()
        self.reflection_advantage_group_by = str(
            dynamic_memory_cfg.get("reflection_advantage_group_by", "memory_pool")
        ).strip().lower()
        if self.reflection_reward_mode not in {"batch_zscore", "zscore"}:
            raise ValueError(
                "Unsupported algorithm.ucob.skill_memory.reflection_reward_mode="
                f"{self.reflection_reward_mode!r}; currently use 'batch_zscore'."
            )
        if self.reflection_advantage_group_by not in {"memory_pool", "pool", "task_state"}:
            raise ValueError(
                "Unsupported algorithm.ucob.skill_memory.reflection_advantage_group_by="
                f"{self.reflection_advantage_group_by!r}; currently use 'memory_pool'."
            )
        self.reflection_advantage_eps = float(dynamic_memory_cfg.get("reflection_advantage_eps", 0.05))
        self.reflection_advantage_clip = float(dynamic_memory_cfg.get("reflection_advantage_clip", 2.0))
        self.reflection_train_response_length = int(
            dynamic_memory_cfg.get(
                "reflection_train_response_length",
                dynamic_memory_cfg.get(
                    "reflection_max_response_length",
                    config.actor_rollout_ref.rollout.get("response_length", config.data.get("max_response_length", 512)),
                ),
            )
        )
        counterfactual_mode = str(dynamic_memory_cfg.get("counterfactual_mode", "no_memory")).strip().lower()
        counterfactual_aliases = {
            "none": "no_memory",
            "no": "no_memory",
            "no_skill": "no_memory",
            "no-memory": "no_memory",
            "no_memory": "no_memory",
            "bad": "bad_memory",
            "bad_skill": "bad_memory",
            "bad-memory": "bad_memory",
            "bad_memory": "bad_memory",
        }
        self.dynamic_memory_counterfactual_mode = counterfactual_aliases.get(counterfactual_mode, counterfactual_mode)
        if self.dynamic_memory_counterfactual_mode not in {"no_memory", "bad_memory"}:
            raise ValueError(
                "Unsupported algorithm.ucob.skill_memory.counterfactual_mode="
                f"{counterfactual_mode!r}; choose 'no_memory' or 'bad_memory'."
            )
        self.reflection_per_group = int(dynamic_memory_cfg.get("reflection_per_group", 0))
        self.reflection_selection = str(dynamic_memory_cfg.get("reflection_selection", "success_failure")).lower()
        self.reflection_max_response_length = dynamic_memory_cfg.get("reflection_max_response_length", None)
        self.reflection_temperature = dynamic_memory_cfg.get("reflection_temperature", None)
        self.reflection_top_p = dynamic_memory_cfg.get("reflection_top_p", None)
        self.reflection_top_k = dynamic_memory_cfg.get("reflection_top_k", None)
        self.reflection_backend = str(
            dynamic_memory_cfg.get(
                "reflection_backend",
                os.environ.get("DYNAMIC_MEMORY_REFLECTION_BACKEND", "vllm"),
            )
        ).strip().lower()
        self.reflection_api_model = dynamic_memory_cfg.get(
            "reflection_api_model",
            os.environ.get(
                "DYNAMIC_MEMORY_REFLECTION_API_MODEL",
                os.environ.get("DYNAMIC_MEMORY_REFLECTION_MODEL", ""),
            ),
        )
        self.reflection_api_url = dynamic_memory_cfg.get(
            "reflection_api_url",
            os.environ.get("DYNAMIC_MEMORY_REFLECTION_API_URL", ""),
        )
        self.reflection_api_base_url = dynamic_memory_cfg.get(
            "reflection_api_base_url",
            os.environ.get(
                "DYNAMIC_MEMORY_REFLECTION_API_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", ""),
            ),
        )
        self.reflection_api_key = dynamic_memory_cfg.get(
            "reflection_api_key",
            os.environ.get(
                "DYNAMIC_MEMORY_REFLECTION_API_KEY",
                os.environ.get("OPENAI_API_KEY", ""),
            ),
        )
        self.reflection_api_timeout = int(dynamic_memory_cfg.get(
            "reflection_api_timeout",
            os.environ.get("DYNAMIC_MEMORY_REFLECTION_API_TIMEOUT", 120),
        ))
        self.reflection_api_max_workers = int(dynamic_memory_cfg.get(
            "reflection_api_max_workers",
            os.environ.get("DYNAMIC_MEMORY_REFLECTION_API_MAX_WORKERS", 8),
        ))
        self.reflection_api_retries = int(dynamic_memory_cfg.get(
            "reflection_api_retries",
            os.environ.get("DYNAMIC_MEMORY_REFLECTION_API_RETRIES", 2),
        ))
        if self.dynamic_memory_enabled:
            env_name = str(config.env.env_name).lower()
            if (
                "alfworld" not in env_name
                and "webshop" not in env_name
                and "sokoban" not in env_name
                and "appworld" not in env_name
                and "search" not in env_name
            ):
                raise ValueError(
                    "algorithm.ucob.skill_memory.enable=true currently supports only ALFWorld, WebShop, Sokoban, AppWorld, and Search."
                )
            dynamic_memory_cfg = dict(dynamic_memory_cfg)
            dynamic_memory_cfg.setdefault("env_name", env_name)
            if not dynamic_memory_cfg.get("skill_mapping_dir"):
                if "alfworld" in env_name:
                    dynamic_memory_cfg["skill_mapping_dir"] = "skills/alfworld"
                elif "webshop" in env_name:
                    dynamic_memory_cfg["skill_mapping_dir"] = "skills/webshop"
                elif "sokoban" in env_name:
                    dynamic_memory_cfg["skill_mapping_dir"] = "skills/sokoban"
                elif "appworld" in env_name:
                    dynamic_memory_cfg["skill_mapping_dir"] = "skills/appworld"
                else:
                    dynamic_memory_cfg["skill_mapping_dir"] = "skills/search"
            from agent_system.memory import DynamicMemoryProvider

            self.dynamic_memory_provider = DynamicMemoryProvider(dynamic_memory_cfg)
            print(f"[UCOB] Skill memory file: {self.dynamic_memory_provider.memory.filepath}")
            print(f"[UCOB] Skill memory size: {len(self.dynamic_memory_provider.memory)}")
            print(
                "[UCOB] Skill memory format: "
                f"{self.dynamic_memory_provider.skill_format}, "
                f"category_filter={self.dynamic_memory_provider.category_filter}"
            )
            print(f"[UCOB] Skill memory counterfactual mode: {self.dynamic_memory_counterfactual_mode}")
            print(f"[UCOB] Skill memory reflection backend: {self.reflection_backend}")
            print(
                "[UCOB] Skill memory reflection train: "
                f"{self.reflection_train_enable}, warmup_steps={self.reflection_train_warmup_steps}"
            )
            print(f"[UCOB] Skill memory state-aware retrieval: {self.dynamic_memory_state_aware_retrieval}")
            print(f"[UCOB] Skill memory step reflection: {self.dynamic_memory_step_reflection_enable}")
            print(
                "[UCOB] Skill memory retrieval: "
                f"dual_pool={self.dynamic_memory_provider.dual_pool_retrieval}, "
                f"pool_mode={self.dynamic_memory_provider.memory_pool_mode}, "
                f"top_k={self.dynamic_memory_provider.top_k}, "
                f"task_top_k={self.dynamic_memory_provider.task_top_k}, "
                f"state_top_k={self.dynamic_memory_provider.state_top_k}"
            )
            print(
                "[UCOB] Skill memory utility: "
                f"mode={self.dynamic_memory_provider.utility_update}, "
                f"task_beta={self.dynamic_memory_provider.task_utility_beta}, "
                f"state_beta={self.dynamic_memory_provider.state_utility_beta}, "
                f"initial={self.dynamic_memory_provider.initial_utility_score}"
            )
            print(
                "[UCOB] Skill memory length caps: "
                f"context_tokens={self.dynamic_memory_max_context_tokens}, "
                f"reflection_prompt_tokens={self.dynamic_memory_reflection_max_prompt_tokens}, "
                f"template_reserve_tokens={self.dynamic_memory_prompt_template_reserve_tokens}, "
                f"data_max_prompt_length={self.config.data.max_prompt_length}, "
                f"reflection_max_response_length={self.reflection_max_response_length}"
            )
        dsdar_cfg = config.algorithm.get("dsdar", config.algorithm.get("sdar", {}))
        rollout_mode = str(dsdar_cfg.get("rollout_mode", "mixed"))
        aliases = {"no_skill": "no_skill_only", "skill": "skill_only", "half": "mixed"}
        self.rollout_mode = aliases.get(rollout_mode, rollout_mode)
        valid_modes = {"no_skill_only", "skill_only", "mixed"}
        if self.rollout_mode not in valid_modes:
            raise ValueError(f"Unsupported UCOB rollout_mode={rollout_mode!r}; choose one of {sorted(valid_modes)}")

    @staticmethod
    def _parse_branch_set(value) -> set:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        else:
            raw_items = list(value)
        aliases = {
            "no": "none",
            "no_memory": "none",
            "no-memory": "none",
            "no_skill": "none",
            "no-skill": "none",
            "skill": "good",
            "memory": "good",
            "good_memory": "good",
            "good-memory": "good",
            "bad_memory": "bad",
            "bad-memory": "bad",
        }
        branches = set()
        for item in raw_items:
            branch = str(item).strip().lower()
            if not branch:
                continue
            branches.add(aliases.get(branch, branch))
        return branches

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        self.current_training_step = int(current_step)
        self.total_training_steps = int(total_steps)
        if self.dynamic_memory_provider is not None:
            self.dynamic_memory_provider.set_training_progress(current_step=current_step, total_steps=total_steps)

    def _reflection_train_metadata_active(self) -> bool:
        return (
            self.reflection_train_enable
            and self.dynamic_memory_enabled
            and self.dynamic_memory_provider is not None
            and self.reflection_backend in {"vllm", "rollout", ""}
        )

    def _reflection_train_active(self) -> bool:
        if not self._reflection_train_metadata_active():
            return False
        warmup_steps = int(self.reflection_train_warmup_steps)
        return warmup_steps <= 0 or int(self.current_training_step) > warmup_steps

    def _reflection_training_metadata(self, prompt: str, response: str, pool: str) -> dict:
        return {
            "reflection_train_backend": "vllm_self_reflection",
            "reflection_prompt": str(prompt or ""),
            "reflection_response": str(response or ""),
            "reflection_train_pool": str(pool or "task"),
            "reflection_created_step": int(self.current_training_step),
        }

    def _inject_skills_into_obs(self, obs):
        if self.skill_provider is None:
            return obs

        skill_obs = copy.copy(obs)
        obs_texts = obs.get("text", None)
        if obs_texts is None:
            return skill_obs

        skill_texts = list(obs_texts)
        for i, text in enumerate(skill_texts):
            if text is None:
                continue
            skill_text = self.skill_provider.get_privileged_info_from_prompt(text)
            if skill_text:
                skill_texts[i] = f"[Privileged Skill Information]\n{skill_text}\n\n{text}"
        skill_obs["text"] = skill_texts
        return skill_obs

    @staticmethod
    def _dynamic_memory_use_instruction() -> str:
        return (
            "Before choosing an action, review the skills above. If a skill applies to the current task "
            "and observation, use its principle to guide your reasoning and action. Ignore skills that do not apply."
        )

    def _truncate_text_by_tokens(self, text: str, max_tokens: int, mode: str = "right") -> str:
        text = str(text or "")
        max_tokens = int(max_tokens)
        if max_tokens <= 0 or not text:
            return text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text

        marker = "\n...[truncated to fit context budget]...\n"
        marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
        budget = max(max_tokens - len(marker_ids), 1)
        if mode == "middle" and budget > 8:
            left = max(int(budget * 0.6), 1)
            right = max(budget - left, 1)
            kept_ids = token_ids[:left] + marker_ids + token_ids[-right:]
        elif mode == "left":
            kept_ids = marker_ids + token_ids[-budget:]
        else:
            kept_ids = token_ids[:budget] + marker_ids
        return self.tokenizer.decode(kept_ids, skip_special_tokens=False)

    def _text_token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))

    def _effective_prompt_text_budget(self, requested_tokens: int) -> int:
        requested_tokens = int(requested_tokens)
        data_prompt_length = int(getattr(self.config.data, "max_prompt_length", 0) or 0)
        reserve = max(int(self.dynamic_memory_prompt_template_reserve_tokens), 0)
        if data_prompt_length > 0:
            requested_tokens = min(requested_tokens, max(data_prompt_length - reserve, 1))
        return max(requested_tokens, 0)

    def _truncate_memory_context(self, memory_context: str, base_text: str = "") -> str:
        max_tokens = self._effective_prompt_text_budget(self.dynamic_memory_max_context_tokens)
        if base_text:
            instruction_tokens = self._text_token_len(self._dynamic_memory_use_instruction())
            base_tokens = self._text_token_len(base_text)
            data_prompt_length = int(getattr(self.config.data, "max_prompt_length", 0) or 0)
            reserve = max(int(self.dynamic_memory_prompt_template_reserve_tokens), 0)
            if data_prompt_length > 0:
                available_tokens = data_prompt_length - reserve - base_tokens - instruction_tokens - 16
                max_tokens = min(max_tokens, max(available_tokens, 0))
        if max_tokens <= 0:
            return ""
        return self._truncate_text_by_tokens(
            memory_context,
            max_tokens=max_tokens,
            mode="right",
        )

    def _truncate_reflection_obs_for_vllm(self, obs: dict) -> dict:
        max_tokens = self._effective_prompt_text_budget(self.dynamic_memory_reflection_max_prompt_tokens)
        if max_tokens <= 0 or not isinstance(obs, dict) or not obs.get("text"):
            return obs
        truncated = copy.copy(obs)
        text_values = list(obs.get("text", []))
        truncated_texts = [
            self._truncate_text_by_tokens(text, max_tokens=max_tokens, mode="middle")
            for text in text_values
        ]
        truncated["text"] = truncated_texts
        if "anchor" in truncated and truncated.get("anchor") is not None:
            anchors = list(truncated.get("anchor", []))
            truncated["anchor"] = [
                truncated_texts[i] if i < len(truncated_texts) else anchor
                for i, anchor in enumerate(anchors)
            ]
        return truncated

    def _insert_retrieved_memory_context(self, text: str, memory_context: str) -> str:
        memory_block = f"{memory_context}\n{self._dynamic_memory_use_instruction()}"
        turn_marker = "Now it's your turn"
        marker_index = text.find(turn_marker)
        if marker_index >= 0:
            context_prefix = text[:marker_index].rstrip()
            action_suffix = text[marker_index:].lstrip()
            return f"{context_prefix}\n\n{memory_block}\n\n{action_suffix}"
        return f"{memory_block}\n\n{text}"

    def _inject_retrieved_memory_into_obs(self, obs, retrieved_memory_by_env):
        memory_obs = copy.copy(obs)
        obs_texts = obs.get("text", None)
        if obs_texts is None:
            return memory_obs

        memory_texts = list(obs_texts)
        for i, text in enumerate(memory_texts):
            if text is None:
                continue
            retrieved = retrieved_memory_by_env[i] if i < len(retrieved_memory_by_env) else []
            memory_context = self.dynamic_memory_provider.format_retrieved(retrieved)
            if memory_context:
                memory_context = self._truncate_memory_context(memory_context, base_text=text)
                if memory_context:
                    memory_texts[i] = self._insert_retrieved_memory_context(text, memory_context)
        memory_obs["text"] = memory_texts
        return memory_obs

    def _get_terminal_infos(self, total_infos, total_batch_list):
        terminal_infos = []
        for env_infos, env_batches in zip(total_infos, total_batch_list):
            terminal_info = env_infos[-1] if env_infos else {}
            for step_idx in reversed(range(len(env_batches))):
                if env_batches[step_idx].get("active_masks", False):
                    terminal_info = env_infos[step_idx]
                    break
            terminal_infos.append(terminal_info)
        return terminal_infos

    def _get_terminal_successes(self, envs, terminal_infos, episode_rewards):
        if hasattr(envs, "get_terminal_successes"):
            return envs.get_terminal_successes(terminal_infos)
        return [float(reward) > 0.0 for reward in episode_rewards]

    def _build_context_obs(self, obs, retrieved_memory_by_env):
        if self.dynamic_memory_enabled:
            return self._inject_retrieved_memory_into_obs(obs, retrieved_memory_by_env)
        return self._inject_skills_into_obs(obs)

    def _dynamic_memory_pool_enabled(self, pool: str) -> bool:
        if self.dynamic_memory_provider is None:
            return False
        allows_pool = getattr(self.dynamic_memory_provider, "allows_pool", None)
        if callable(allows_pool):
            return bool(allows_pool(pool))
        return True

    @staticmethod
    def _stringify_anchor(anchor) -> str:
        if anchor is None:
            return ""
        if isinstance(anchor, bytes):
            return anchor.decode("utf-8", errors="ignore")
        if isinstance(anchor, (list, tuple)):
            return " ".join(str(item) for item in anchor)
        return str(anchor)

    def _get_state_contexts(self, obs, batch_size: int) -> list:
        anchors = obs.get("anchor", None)
        if anchors is None:
            anchors = obs.get("text", None)
        if anchors is None:
            return [""] * batch_size
        return [
            self._stringify_anchor(anchors[i] if i < len(anchors) else "")
            for i in range(batch_size)
        ]

    @staticmethod
    def _retrieved_memory_ids(retrieved) -> list:
        return [str(item.get("memory_id", "")) for item in retrieved if item.get("memory_id", "")]

    @staticmethod
    def _retrieved_memory_ids_by_pool(retrieved, pool: str) -> list:
        target_pool = str(pool or "").strip().lower()
        memory_ids = []
        for item in retrieved:
            memory_id = str(item.get("memory_id", "") or "")
            if not memory_id:
                continue
            item_pool = str(item.get("memory_pool", "") or "").strip().lower()
            if not item_pool:
                item_pool = "state" if item.get("memory_type") == "state_skill" else "task"
            if item_pool == target_pool:
                memory_ids.append(memory_id)
        return memory_ids

    @staticmethod
    def _coerce_memory_id_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return [str(memory_id) for memory_id in value if str(memory_id)]
        if isinstance(value, str):
            return [value] if value else []
        return []

    def _memory_ids_from_step_by_pool(self, step: dict, pool: str) -> list:
        target_pool = str(pool or "").strip().lower()
        explicit_key = "dynamic_memory_state_ids" if target_pool == "state" else "dynamic_memory_task_ids"
        explicit_ids = self._coerce_memory_id_list(step.get(explicit_key, []))
        if explicit_ids:
            return explicit_ids

        memory_ids = self._coerce_memory_id_list(step.get("dynamic_memory_ids", []))
        if not memory_ids or self.dynamic_memory_provider is None:
            return []
        filtered = []
        for memory_id in memory_ids:
            memory_pool = self.dynamic_memory_provider.memory_pool_by_id(memory_id)
            if memory_pool == target_pool:
                filtered.append(memory_id)
        return filtered

    @staticmethod
    def _dynamic_memory_item_pool(item: dict) -> str:
        item_pool = str(item.get("memory_pool", "") or "").strip().lower()
        if item_pool in {"task", "state"}:
            return item_pool
        memory_type = str(item.get("memory_type", "") or "").strip().lower()
        return "state" if memory_type in {"state_skill", "step_skill"} else "task"

    @staticmethod
    def _dynamic_memory_float(item: dict, key: str, default: float = 0.0) -> float:
        try:
            value = float(item.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(value):
            return float(default)
        return value

    def _dynamic_memory_bank_size_stats(self) -> dict:
        memory = getattr(self.dynamic_memory_provider, "memory", None)
        data = getattr(memory, "data", []) if memory is not None else []
        task_count = 0
        state_count = 0
        for item in data:
            if self._dynamic_memory_item_pool(item) == "state":
                state_count += 1
            else:
                task_count += 1
        return {
            "dynamic_memory_bank_size_total": float(len(data)),
            "dynamic_memory_bank_size_task": float(task_count),
            "dynamic_memory_bank_size_state": float(state_count),
        }

    def _dynamic_memory_bank_utility_stats(self) -> dict:
        memory = getattr(self.dynamic_memory_provider, "memory", None)
        data = getattr(memory, "data", []) if memory is not None else []
        initial_utility = float(getattr(self.dynamic_memory_provider, "initial_utility_score", 0.0) or 0.0)
        threshold = initial_utility + 1e-8
        if not data:
            return {
                "dynamic_memory_bank_utility_mean": 0.0,
                "dynamic_memory_task_bank_utility_mean": 0.0,
                "dynamic_memory_state_bank_utility_mean": 0.0,
                "dynamic_memory_bank_utility_p25": 0.0,
                "dynamic_memory_bank_utility_p50": 0.0,
                "dynamic_memory_bank_utility_p75": 0.0,
                "dynamic_memory_task_bank_utility_p25": 0.0,
                "dynamic_memory_task_bank_utility_p50": 0.0,
                "dynamic_memory_task_bank_utility_p75": 0.0,
                "dynamic_memory_state_bank_utility_p25": 0.0,
                "dynamic_memory_state_bank_utility_p50": 0.0,
                "dynamic_memory_state_bank_utility_p75": 0.0,
                "dynamic_memory_useful_skill_count": 0.0,
                "dynamic_memory_task_useful_skill_count": 0.0,
                "dynamic_memory_state_useful_skill_count": 0.0,
                "dynamic_memory_nonpositive_skill_count": 0.0,
                "dynamic_memory_task_nonpositive_skill_count": 0.0,
                "dynamic_memory_state_nonpositive_skill_count": 0.0,
                "dynamic_memory_useful_skill_ratio": 0.0,
                "dynamic_memory_task_useful_skill_ratio": 0.0,
                "dynamic_memory_state_useful_skill_ratio": 0.0,
            }

        def _pool(item: dict) -> str:
            return self._dynamic_memory_item_pool(item)

        def _utility(item: dict) -> float:
            return self._dynamic_memory_float(item, "utility_score", default=0.0)

        def _is_useful(item: dict) -> bool:
            return _utility(item) > threshold

        def _stats(items: list[dict]) -> tuple[float, float, float, float]:
            if not items:
                return 0.0, 0.0, 0.0, 0.0
            values = np.asarray([_utility(item) for item in items], dtype=np.float32)
            if values.size == 0:
                return 0.0, 0.0, 0.0, 0.0
            return (
                float(np.mean(values)),
                float(np.percentile(values, 25)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 75)),
            )

        task_items = [item for item in data if _pool(item) == "task"]
        state_items = [item for item in data if _pool(item) == "state"]
        useful_items = [item for item in data if _is_useful(item)]
        task_useful_items = [item for item in task_items if _is_useful(item)]
        state_useful_items = [item for item in state_items if _is_useful(item)]
        task_nonpositive_items = [item for item in task_items if not _is_useful(item)]
        state_nonpositive_items = [item for item in state_items if not _is_useful(item)]
        bank_mean, bank_p25, bank_p50, bank_p75 = _stats(list(data))
        task_mean, task_p25, task_p50, task_p75 = _stats(task_items)
        state_mean, state_p25, state_p50, state_p75 = _stats(state_items)
        bank_size = float(len(data))
        task_bank_size = float(len(task_items))
        state_bank_size = float(len(state_items))
        return {
            "dynamic_memory_bank_utility_mean": bank_mean,
            "dynamic_memory_task_bank_utility_mean": task_mean,
            "dynamic_memory_state_bank_utility_mean": state_mean,
            "dynamic_memory_bank_utility_p25": bank_p25,
            "dynamic_memory_bank_utility_p50": bank_p50,
            "dynamic_memory_bank_utility_p75": bank_p75,
            "dynamic_memory_task_bank_utility_p25": task_p25,
            "dynamic_memory_task_bank_utility_p50": task_p50,
            "dynamic_memory_task_bank_utility_p75": task_p75,
            "dynamic_memory_state_bank_utility_p25": state_p25,
            "dynamic_memory_state_bank_utility_p50": state_p50,
            "dynamic_memory_state_bank_utility_p75": state_p75,
            "dynamic_memory_useful_skill_count": float(len(useful_items)),
            "dynamic_memory_task_useful_skill_count": float(len(task_useful_items)),
            "dynamic_memory_state_useful_skill_count": float(len(state_useful_items)),
            "dynamic_memory_nonpositive_skill_count": float(len(data) - len(useful_items)),
            "dynamic_memory_task_nonpositive_skill_count": float(len(task_nonpositive_items)),
            "dynamic_memory_state_nonpositive_skill_count": float(len(state_nonpositive_items)),
            "dynamic_memory_useful_skill_ratio": float(len(useful_items) / bank_size) if bank_size > 0 else 0.0,
            "dynamic_memory_task_useful_skill_ratio": float(len(task_useful_items) / task_bank_size) if task_bank_size > 0 else 0.0,
            "dynamic_memory_state_useful_skill_ratio": float(len(state_useful_items) / state_bank_size) if state_bank_size > 0 else 0.0,
        }

    @staticmethod
    def _new_dynamic_memory_retrieval_stats() -> dict:
        return {
            "count": [],
            "task_count": [],
            "state_count": [],
            "utility": [],
            "task_utility": [],
            "state_utility": [],
            "relevance": [],
            "task_relevance": [],
            "state_relevance": [],
            "score": [],
            "task_score": [],
            "state_score": [],
        }

    @staticmethod
    def _new_dynamic_memory_efficiency_stats() -> dict:
        # Efficiency analysis only: these values are logged, never fed back into training.
        return {
            "retrieval_time_s": 0.0,
            "task_retrieval_time_s": 0.0,
            "state_retrieval_time_s": 0.0,
            "skill_generation_time_s": 0.0,
            "task_skill_generation_time_s": 0.0,
            "state_skill_generation_time_s": 0.0,
            "skill_write_time_s": 0.0,
            "task_skill_write_time_s": 0.0,
            "state_skill_write_time_s": 0.0,
            "base_prompt_tokens": [],
            "skill_prompt_tokens": [],
            "memory_context_tokens": [],
        }

    def _timed_retrieve_dynamic_memory_for_step(self, efficiency_stats: dict | None, *args, **kwargs):
        pool = str(kwargs.get("pool", "all") or "all").strip().lower()
        start_time = time.perf_counter()
        result = self._retrieve_dynamic_memory_for_step(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        if efficiency_stats is not None:
            efficiency_stats["retrieval_time_s"] = float(efficiency_stats.get("retrieval_time_s", 0.0)) + elapsed
            if pool == "task":
                efficiency_stats["task_retrieval_time_s"] = (
                    float(efficiency_stats.get("task_retrieval_time_s", 0.0)) + elapsed
                )
            elif pool == "state":
                efficiency_stats["state_retrieval_time_s"] = (
                    float(efficiency_stats.get("state_retrieval_time_s", 0.0)) + elapsed
                )
        return result

    def _update_dynamic_memory_retrieval_stats(
        self,
        stats: dict,
        retrieved_memory_by_env,
        rollout_skill_mask,
        active_masks=None,
    ) -> None:
        if stats is None:
            return
        batch_size = min(len(retrieved_memory_by_env), len(rollout_skill_mask))
        if active_masks is not None:
            batch_size = min(batch_size, len(active_masks))
        for i in range(batch_size):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if not bool(rollout_skill_mask[i]):
                continue
            retrieved = list(retrieved_memory_by_env[i] or [])
            task_items = [item for item in retrieved if self._dynamic_memory_item_pool(item) == "task"]
            state_items = [item for item in retrieved if self._dynamic_memory_item_pool(item) == "state"]
            stats["count"].append(float(len(retrieved)))
            stats["task_count"].append(float(len(task_items)))
            stats["state_count"].append(float(len(state_items)))
            for item in retrieved:
                pool = self._dynamic_memory_item_pool(item)
                for key in ("utility", "relevance", "score"):
                    source_key = "utility_score" if key == "utility" else key
                    value = self._dynamic_memory_float(item, source_key, default=0.0)
                    stats[key].append(value)
                    stats[f"{pool}_{key}"].append(value)

    def _update_dynamic_memory_prompt_token_stats(
        self,
        efficiency_stats: dict | None,
        base_obs,
        skill_obs,
        retrieved_memory_by_env,
        rollout_skill_mask,
        active_masks=None,
    ) -> None:
        if efficiency_stats is None:
            return
        base_texts = base_obs.get("text", None) if isinstance(base_obs, dict) else None
        skill_texts = skill_obs.get("text", None) if isinstance(skill_obs, dict) else None
        if base_texts is None or skill_texts is None:
            return
        batch_size = min(len(base_texts), len(skill_texts), len(rollout_skill_mask))
        if active_masks is not None:
            batch_size = min(batch_size, len(active_masks))
        instruction_tokens = self._text_token_len(self._dynamic_memory_use_instruction())
        for i in range(batch_size):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if not bool(rollout_skill_mask[i]):
                continue
            base_text = base_texts[i]
            skill_text = skill_texts[i]
            if base_text is None or skill_text is None:
                continue
            base_tokens = float(self._text_token_len(base_text))
            skill_tokens = float(self._text_token_len(skill_text))
            memory_context_tokens = max(skill_tokens - base_tokens - float(instruction_tokens), 0.0)
            efficiency_stats["base_prompt_tokens"].append(base_tokens)
            efficiency_stats["skill_prompt_tokens"].append(skill_tokens)
            efficiency_stats["memory_context_tokens"].append(memory_context_tokens)

    @staticmethod
    def _finalize_dynamic_memory_retrieval_stats(stats: dict) -> dict:
        def _mean(values) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "dynamic_memory_retrieved_count_mean": _mean(stats.get("count", [])),
            "dynamic_memory_task_retrieved_count_mean": _mean(stats.get("task_count", [])),
            "dynamic_memory_state_retrieved_count_mean": _mean(stats.get("state_count", [])),
            "dynamic_memory_retrieved_utility_mean": _mean(stats.get("utility", [])),
            "dynamic_memory_task_retrieved_utility_mean": _mean(stats.get("task_utility", [])),
            "dynamic_memory_state_retrieved_utility_mean": _mean(stats.get("state_utility", [])),
            "dynamic_memory_retrieved_relevance_mean": _mean(stats.get("relevance", [])),
            "dynamic_memory_task_retrieved_relevance_mean": _mean(stats.get("task_relevance", [])),
            "dynamic_memory_state_retrieved_relevance_mean": _mean(stats.get("state_relevance", [])),
            "dynamic_memory_retrieved_score_mean": _mean(stats.get("score", [])),
            "dynamic_memory_task_retrieved_score_mean": _mean(stats.get("task_score", [])),
            "dynamic_memory_state_retrieved_score_mean": _mean(stats.get("state_score", [])),
        }

    @staticmethod
    def _finalize_dynamic_memory_efficiency_stats(stats: dict) -> dict:
        def _mean(values) -> float:
            return float(np.mean(values)) if values else 0.0

        base_mean = _mean(stats.get("base_prompt_tokens", []))
        skill_mean = _mean(stats.get("skill_prompt_tokens", []))
        return {
            "dynamic_memory_retrieval_time_s": float(stats.get("retrieval_time_s", 0.0) or 0.0),
            "dynamic_memory_task_retrieval_time_s": float(stats.get("task_retrieval_time_s", 0.0) or 0.0),
            "dynamic_memory_state_retrieval_time_s": float(stats.get("state_retrieval_time_s", 0.0) or 0.0),
            "dynamic_memory_skill_generation_time_s": float(stats.get("skill_generation_time_s", 0.0) or 0.0),
            "dynamic_memory_task_skill_generation_time_s": float(stats.get("task_skill_generation_time_s", 0.0) or 0.0),
            "dynamic_memory_state_skill_generation_time_s": float(stats.get("state_skill_generation_time_s", 0.0) or 0.0),
            "dynamic_memory_skill_write_time_s": float(stats.get("skill_write_time_s", 0.0) or 0.0),
            "dynamic_memory_task_skill_write_time_s": float(stats.get("task_skill_write_time_s", 0.0) or 0.0),
            "dynamic_memory_state_skill_write_time_s": float(stats.get("state_skill_write_time_s", 0.0) or 0.0),
            "dynamic_memory_base_prompt_tokens_mean": base_mean,
            "dynamic_memory_skill_prompt_tokens_mean": skill_mean,
            "dynamic_memory_memory_context_tokens_mean": _mean(stats.get("memory_context_tokens", [])),
            "dynamic_memory_prompt_token_overhead_ratio": (
                float((skill_mean - base_mean) / base_mean) if base_mean > 0.0 else 0.0
            ),
        }

    @staticmethod
    def _dynamic_memory_write_stats(task_stats: dict, state_stats: dict) -> dict:
        known_reasons = [
            "parse_failed",
            "success_mismatch",
            "empty_skill",
            "empty_lesson",
            "add_failed",
        ]

        def _pool_stats(prefix: str, stats: dict) -> dict:
            stats = stats or {}
            attempted = int(stats.get("attempted", 0) or 0)
            added = int(stats.get("added", 0) or 0)
            reasons = stats.get("reasons", {}) or {}
            result = {
                f"dynamic_memory_{prefix}_write_attempted": float(attempted),
                f"dynamic_memory_{prefix}_write_added": float(added),
                f"dynamic_memory_{prefix}_write_success_rate": float(added / attempted) if attempted > 0 else 0.0,
            }
            known_total = 0
            for reason in known_reasons:
                count = int(reasons.get(reason, 0) or 0)
                known_total += count
                result[f"dynamic_memory_{prefix}_write_reason_{reason}"] = float(count)
            result[f"dynamic_memory_{prefix}_write_reason_other"] = float(max((attempted - added) - known_total, 0))
            return result

        task_attempted = int((task_stats or {}).get("attempted", 0) or 0)
        task_added = int((task_stats or {}).get("added", 0) or 0)
        state_attempted = int((state_stats or {}).get("attempted", 0) or 0)
        state_added = int((state_stats or {}).get("added", 0) or 0)
        total_attempted = task_attempted + state_attempted
        total_added = task_added + state_added

        result = {}
        result.update(_pool_stats("task", task_stats))
        result.update(_pool_stats("state", state_stats))
        result["dynamic_memory_write_added_total"] = float(total_added)
        result["dynamic_memory_write_success_rate_total"] = (
            float(total_added / total_attempted) if total_attempted > 0 else 0.0
        )
        return result

    @staticmethod
    def _attach_dynamic_memory_scalar_success(success: dict, stats: dict, batch_size: int) -> None:
        for key, value in stats.items():
            success[key] = np.array([float(value)] * batch_size, dtype=np.float32)

    def _dynamic_memory_update_split_counts(self, memory_ids) -> dict:
        counts = {"task_updates": 0, "state_updates": 0}
        if self.dynamic_memory_provider is None:
            return counts
        for memory_id in memory_ids:
            memory_pool = self.dynamic_memory_provider.memory_pool_by_id(memory_id)
            if memory_pool == "state":
                counts["state_updates"] += 1
            elif memory_pool == "task":
                counts["task_updates"] += 1
        return counts

    def _dynamic_memory_retrieved_update_stats(self, retrieved_by_env, rollout_skill_mask) -> dict:
        memory_ids = []
        batch_size = min(len(retrieved_by_env), len(rollout_skill_mask))
        for i in range(batch_size):
            if not bool(rollout_skill_mask[i]):
                continue
            for item in retrieved_by_env[i] or []:
                memory_id = str(item.get("memory_id", "") or "")
                if memory_id:
                    memory_ids.append(memory_id)
        split_counts = self._dynamic_memory_update_split_counts(memory_ids)
        return {
            "groups": sum(1 for i in range(batch_size) if bool(rollout_skill_mask[i])),
            "updates": len(memory_ids),
            **split_counts,
        }

    def _dynamic_memory_relative_outcome_update_stats(self, retrieved_by_env, rollout_skill_mask) -> dict:
        memory_ids = []
        batch_size = min(len(retrieved_by_env), len(rollout_skill_mask))
        group_n = max(int(self.config.env.rollout.n), 1)
        usable_groups = 0
        for start in range(0, batch_size, group_n):
            group_indices = list(range(start, min(start + group_n, batch_size)))
            no_skill_indices = [idx for idx in group_indices if not bool(rollout_skill_mask[idx])]
            skill_indices = [idx for idx in group_indices if bool(rollout_skill_mask[idx])]
            if not no_skill_indices or not skill_indices:
                continue
            usable_groups += 1
            seen_memory_ids = set()
            for idx in skill_indices:
                for item in retrieved_by_env[idx] or []:
                    memory_id = str(item.get("memory_id", "") or "")
                    if memory_id and memory_id not in seen_memory_ids:
                        memory_ids.append(memory_id)
                        seen_memory_ids.add(memory_id)
        split_counts = self._dynamic_memory_update_split_counts(memory_ids)
        return {
            "groups": usable_groups,
            "updates": len(memory_ids),
            **split_counts,
        }

    @staticmethod
    def _discounted_returns_from_trajectory_steps(steps, gamma: float) -> np.ndarray:
        returns = np.zeros(len(steps), dtype=np.float32)
        running_return = 0.0
        for step_idx in reversed(range(len(steps))):
            step = steps[step_idx]
            if not bool(step.get("active_masks", True)):
                returns[step_idx] = 0.0
                continue
            reward_value = step.get("rewards", 0.0)
            if hasattr(reward_value, "item"):
                reward_value = reward_value.item()
            running_return = float(reward_value) + (float(gamma) * running_return)
            returns[step_idx] = running_return
        return returns

    def _compute_memory_step_returns(self, total_batch_list) -> list:
        gamma = float(self.config.algorithm.get("gamma", 1.0))
        return [
            self._discounted_returns_from_trajectory_steps(steps, gamma=gamma)
            for steps in total_batch_list
        ]

    @staticmethod
    def _opposite_cbsd_branch(branch: str) -> str:
        return "none" if branch == "good" else "good"

    def _cbsd_effective_teaching_gap(self, raw_gap: float) -> float | None:
        raw_gap = float(raw_gap)
        if self.cbsd_force_source_teacher:
            return max(raw_gap, float(self.cbsd_force_gap), 1e-6)
        if raw_gap <= self.cbsd_margin:
            return None
        return raw_gap

    @staticmethod
    def _clone_row_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        return torch.as_tensor(value).detach().clone()

    @staticmethod
    def _repeat_like_row(tensor: torch.Tensor, rows: int) -> torch.Tensor:
        row = torch.zeros_like(tensor[:1])
        if rows <= 1:
            return row[:rows]
        return row.repeat((rows,) + (1,) * (row.dim() - 1))

    @staticmethod
    def _empty_non_tensor_like(rows: int, reference: np.ndarray) -> np.ndarray:
        shape = (rows,) + tuple(reference.shape[1:])
        values = np.empty(shape, dtype=object)
        values[...] = ""
        return values

    @staticmethod
    def _align_non_tensor_ndim(values: np.ndarray, rows: int, reference: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=object)
        if values.ndim == reference.ndim:
            return values
        if values.ndim == 1 and values.shape[0] == rows and reference.ndim > 1:
            target_shape = (rows,) + (1,) * (reference.ndim - 1)
            if values.size == int(np.prod(target_shape)):
                return values.reshape(target_shape)
        return values

    @staticmethod
    def _concat_aligned_batches(left: DataProto, right: DataProto) -> DataProto:
        if right is None or len(right) == 0:
            return left
        if left is None or len(left) == 0:
            return right

        left_tensors = {key: value for key, value in left.batch.items()}
        right_tensors = {key: value for key, value in right.batch.items()}
        for key in sorted(set(left_tensors.keys()) | set(right_tensors.keys())):
            if key not in left_tensors:
                left_tensors[key] = UCOBTrajectoryCollector._repeat_like_row(right_tensors[key], len(left))
            if key not in right_tensors:
                right_tensors[key] = UCOBTrajectoryCollector._repeat_like_row(left_tensors[key], len(right))

        left_non_tensors = dict(left.non_tensor_batch)
        right_non_tensors = dict(right.non_tensor_batch)
        for key in sorted(set(left_non_tensors.keys()) | set(right_non_tensors.keys())):
            if key not in left_non_tensors:
                left_non_tensors[key] = UCOBTrajectoryCollector._empty_non_tensor_like(
                    len(left),
                    np.asarray(right_non_tensors[key], dtype=object),
                )
            if key not in right_non_tensors:
                right_non_tensors[key] = UCOBTrajectoryCollector._empty_non_tensor_like(
                    len(right),
                    np.asarray(left_non_tensors[key], dtype=object),
                )
            left_non_tensors[key] = UCOBTrajectoryCollector._align_non_tensor_ndim(
                left_non_tensors[key],
                len(left),
                np.asarray(right_non_tensors[key], dtype=object),
            )
            right_non_tensors[key] = UCOBTrajectoryCollector._align_non_tensor_ndim(
                right_non_tensors[key],
                len(right),
                np.asarray(left_non_tensors[key], dtype=object),
            )

        left_aligned = DataProto.from_dict(
            tensors=left_tensors,
            non_tensors=left_non_tensors,
            meta_info=left.meta_info,
        )
        right_aligned = DataProto.from_dict(
            tensors=right_tensors,
            non_tensors=right_non_tensors,
            meta_info=left.meta_info,
        )
        return DataProto.concat([left_aligned, right_aligned])

    @staticmethod
    def _strip_cbsd_record_keys(total_batch_list) -> None:
        keys = [
            "ucob_cbsd_alt_prompt_ids",
            "ucob_cbsd_alt_prompt_attention_mask",
            "ucob_cbsd_alt_branch",
            "ucob_cbsd_action_token_mask",
            "ucob_cbsd_action_parse_success",
        ]
        for steps in total_batch_list:
            for step in steps:
                for key in keys:
                    step.pop(key, None)

    def _cbsd_branch_for_step(self, step: dict) -> str:
        branch = str(step.get("dynamic_memory_branch", "") or "").strip().lower()
        if branch in {"good", "bad", "none"}:
            return branch
        mode = str(step.get("dsdar_rollout_mode", "") or "").strip().lower()
        return "good" if mode == "skill" else "none"

    def _cbsd_prompt_for_branch(self, step: dict, branch: str, response_length: int):
        current_branch = self._cbsd_branch_for_step(step)
        if branch == current_branch:
            prompt_ids = self._clone_row_tensor(step["prompts"])
            prompt_attention = self._clone_row_tensor(step["attention_mask"][:-response_length])
            return prompt_ids, prompt_attention

        alt_branch = str(step.get("ucob_cbsd_alt_branch", "") or "").strip().lower()
        if branch == alt_branch and "ucob_cbsd_alt_prompt_ids" in step:
            return (
                self._clone_row_tensor(step["ucob_cbsd_alt_prompt_ids"]),
                self._clone_row_tensor(step["ucob_cbsd_alt_prompt_attention_mask"]),
            )
        return None

    def _cbsd_action_loss_mask(self, source_step: dict, response_length: int):
        response_mask = self._clone_row_tensor(source_step["attention_mask"][-response_length:]).to(dtype=torch.long)
        if not self.cbsd_action_mask_enable:
            return response_mask
        if "ucob_cbsd_action_token_mask" not in source_step:
            return torch.zeros_like(response_mask)
        full_action_mask = self._clone_row_tensor(source_step["ucob_cbsd_action_token_mask"])
        action_mask = full_action_mask[-response_length:].to(dtype=response_mask.dtype)
        return action_mask * response_mask

    @staticmethod
    def _case_study_safe_text(value, max_chars: int = 2000) -> str:
        if value is None:
            return ""
        if isinstance(value, np.ndarray):
            value = value.tolist()
        text = " ".join(str(value).strip().split())
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "...[truncated]"
        return text

    def _case_study_memory_lookup(self) -> dict:
        provider = getattr(self, "dynamic_memory_provider", None)
        memory = getattr(provider, "memory", None)
        data = getattr(memory, "data", []) if memory is not None else []
        lookup = {}
        for item in data or []:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", "") or "")
            if memory_id:
                lookup[memory_id] = item
        return lookup

    def _case_study_memory_summary(self, item: dict | None, extra: dict | None = None) -> dict:
        item = item or {}
        extra = extra or {}
        pool = str(
            item.get("memory_pool", "")
            or item.get("pool", "")
            or extra.get("memory_pool", "")
            or extra.get("pool", "")
            or ""
        ).strip().lower()
        if not pool:
            memory_type = str(item.get("memory_type", "") or extra.get("memory_type", "") or "").strip().lower()
            pool = "state" if memory_type in {"state_skill", "step_skill"} else "task"
        return {
            "memory_id": str(item.get("memory_id", "") or extra.get("memory_id", "") or ""),
            "pool": pool,
            "title": self._case_study_safe_text(item.get("title", ""), 300),
            "principle": self._case_study_safe_text(item.get("principle", item.get("reflection", "")), 900),
            "when_to_apply": self._case_study_safe_text(item.get("when_to_apply", ""), 600),
            "utility_score": self._dynamic_memory_float(item, "utility_score", default=0.0) if item else 0.0,
            "relevance": self._dynamic_memory_float(extra, "relevance", default=0.0) if extra else 0.0,
            "score": self._dynamic_memory_float(extra, "score", default=0.0) if extra else 0.0,
        }

    def _case_study_retrieved_debug_items(self, retrieved) -> list:
        if not self.case_study_debug_enable:
            return []
        return [self._case_study_memory_summary(item, extra=item) for item in (retrieved or []) if isinstance(item, dict)]

    def _case_study_memory_ids_from_step(self, step: dict) -> list:
        return self._coerce_memory_id_list(step.get("dynamic_memory_ids", []))

    def _case_study_retrieved_skills(self, step: dict, memory_lookup: dict) -> list:
        debug_items = step.get("dynamic_memory_retrieved_debug", [])
        if isinstance(debug_items, np.ndarray):
            debug_items = debug_items.tolist()
        if isinstance(debug_items, (list, tuple)) and debug_items:
            return [item for item in debug_items if isinstance(item, dict)]
        return [
            self._case_study_memory_summary(memory_lookup.get(memory_id))
            for memory_id in self._case_study_memory_ids_from_step(step)
        ]

    def _case_study_written_memories_for_ids(self, memory_ids: set[str]) -> list:
        if not memory_ids:
            return []
        return [
            item for item in getattr(self, "case_study_written_memories", [])
            if str(item.get("memory_id", "") or "") in memory_ids
        ]

    def _case_study_memory_id_set(self) -> set[str]:
        return set(self._case_study_memory_lookup().keys())

    def _case_study_capture_new_memories(self, before_ids: set[str]) -> list:
        if not self.case_study_debug_enable:
            return []
        lookup = self._case_study_memory_lookup()
        new_ids = sorted(set(lookup.keys()) - set(before_ids or set()))
        return [self._case_study_memory_summary(lookup[memory_id]) for memory_id in new_ids]

    def _case_study_decode_prompt(self, step: dict) -> str:
        prompts = step.get("prompts", None)
        if prompts is None:
            return ""
        if isinstance(prompts, torch.Tensor):
            token_ids = prompts.detach().cpu().tolist()
        elif isinstance(prompts, np.ndarray):
            token_ids = prompts.tolist()
        else:
            token_ids = prompts
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return str(prompts)

    def _case_study_step_record(
        self,
        *,
        uid: str,
        anchor_obs: str,
        records: list,
        source_record: dict | None,
        target_record: dict | None,
        source_branch: str,
        target_branch: str,
        return_gap: float,
        ignored_reason: str,
        memory_lookup: dict,
        total_batch_list=None,
    ) -> dict:
        source_record = source_record or {}
        target_record = target_record or {}
        source_step = source_record.get("step", {}) or {}
        target_step = target_record.get("step", {}) or {}
        skill_record = next((item for item in records if item.get("branch") == "good"), None)
        noskill_record = next((item for item in records if item.get("branch") == "none"), None)
        skill_step = (skill_record or source_record if source_branch == "good" else target_record).get("step", {}) or {}
        noskill_step = (noskill_record or source_record if source_branch == "none" else target_record).get("step", {}) or {}
        source_return = float(source_record.get("return", 0.0) or 0.0)
        target_return = float(target_record.get("return", 0.0) or 0.0)
        teacher_direction = "ignored"
        if ignored_reason == "none":
            if source_branch == "good" and target_branch == "none":
                teacher_direction = "skill_to_noskill"
            elif source_branch == "none" and target_branch == "good":
                teacher_direction = "noskill_to_skill"
        skill_memory_ids = self._case_study_memory_ids_from_step(skill_step)
        task_memory_ids = self._memory_ids_from_step_by_pool(skill_step, "task")
        state_memory_ids = self._memory_ids_from_step_by_pool(skill_step, "state")
        record = {
            "global_step": int(self.current_training_step),
            "env_name": str(self.config.env.env_name),
            "uid": str(uid),
            "anchor_obs": self._case_study_safe_text(anchor_obs, 2000),
            "task": self._case_study_safe_text(source_step.get("raw_prompt", target_step.get("raw_prompt", "")), 2000),
            "source_branch": str(source_branch),
            "target_branch": str(target_branch),
            "teacher_direction": teacher_direction,
            "return_gap": float(return_gap),
            "source_return": source_return,
            "target_return": target_return,
            "skill_branch_action": self._case_study_safe_text(self._decode_step_response(skill_step), 2000),
            "noskill_branch_action": self._case_study_safe_text(self._decode_step_response(noskill_step), 2000),
            "skill_branch_traj_uid": str(skill_step.get("traj_uid", "")),
            "noskill_branch_traj_uid": str(noskill_step.get("traj_uid", "")),
            "skill_branch_step_idx": int((skill_record or {}).get("step_idx", -1)),
            "noskill_branch_step_idx": int((noskill_record or {}).get("step_idx", -1)),
            "skill_memory_ids": skill_memory_ids,
            "task_memory_ids": task_memory_ids,
            "state_memory_ids": state_memory_ids,
            "retrieved_skills": self._case_study_retrieved_skills(skill_step, memory_lookup),
            "written_skills": self._case_study_written_memories_for_ids(set(skill_memory_ids)),
            "step_written_skills": getattr(self, "case_study_written_memories", []),
            "ignored_reason": ignored_reason,
        }
        if self.case_study_debug_include_prompts:
            record["source_prompt"] = self._case_study_safe_text(self._case_study_decode_prompt(source_step), 4000)
            record["target_prompt"] = self._case_study_safe_text(self._case_study_decode_prompt(target_step), 4000)
        if self.case_study_debug_include_full_trajectory:
            skill_traj_idx = int((skill_record or {}).get("traj_idx", -1))
            noskill_traj_idx = int((noskill_record or {}).get("traj_idx", -1))
            record["skill_branch_traj_idx"] = skill_traj_idx
            record["noskill_branch_traj_idx"] = noskill_traj_idx
            total_batch_list = total_batch_list or []
            record["skill_branch_trajectory"] = [
                self._case_study_safe_text(self._decode_step_response(step), 2000)
                for step in (
                    total_batch_list[skill_traj_idx]
                    if 0 <= skill_traj_idx < len(total_batch_list)
                    else []
                )
            ]
            record["noskill_branch_trajectory"] = [
                self._case_study_safe_text(self._decode_step_response(step), 2000)
                for step in (
                    total_batch_list[noskill_traj_idx]
                    if 0 <= noskill_traj_idx < len(total_batch_list)
                    else []
                )
            ]
        return record

    def _case_study_write_records(self, records: list[dict]) -> None:
        if not self.case_study_debug_enable or not records:
            return
        os.makedirs(self.case_study_debug_dir, exist_ok=True)
        path = os.path.join(self.case_study_debug_dir, f"step_{int(self.current_training_step):06d}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for record in records[: self.case_study_debug_max_per_step]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_cbsd_auxiliary_batch(self, total_batch_list, step_returns_by_traj) -> DataProto | None:
        if not self.cbsd_enable:
            self.last_cbsd_analysis_stats = {
                "analysis/mech/raw_pair_count": 0.0,
                "analysis/mech/activated_pair_count": 0.0,
                "analysis/mech/margin_filtered_count": 0.0,
                "analysis/mech/margin_filtered_ratio": 0.0,
                "analysis/mech/activated_pair_ratio": 0.0,
                "analysis/mech/ignored_pair_count": 0.0,
                "analysis/mech/ignored_pair_ratio": 0.0,
                "analysis/mech/skill_to_noskill_pair_count": 0.0,
                "analysis/mech/noskill_to_skill_pair_count": 0.0,
                "analysis/mech/skill_to_noskill_pair_ratio": 0.0,
                "analysis/mech/noskill_to_skill_pair_ratio": 0.0,
                "analysis/mech/append_failed_count": 0.0,
            }
            return None

        use_response_scope = (
            self.cbsd_loss_type in {"opd", "topk_opd", "topk-opd", "topk"}
            and self.cbsd_opd_scope in {"response", "full_response"}
        )
        grouped_records = {}
        records_by_uid = {}
        cbsd_analysis_stats = {
            "raw_pair_count": 0.0,
            "activated_pair_count": 0.0,
            "margin_filtered_count": 0.0,
            "append_failed_count": 0.0,
            "skill_to_noskill_pair_count": 0.0,
            "noskill_to_skill_pair_count": 0.0,
        }
        case_study_records = []
        case_study_memory_lookup = self._case_study_memory_lookup() if self.case_study_debug_enable else {}

        def maybe_add_case_study_record(
            *,
            uid: str,
            anchor_obs: str,
            records: list,
            source_record: dict | None,
            target_record: dict | None,
            source_branch: str,
            target_branch: str,
            return_gap: float,
            ignored_reason: str,
        ) -> None:
            if not self.case_study_debug_enable:
                return
            if len(case_study_records) >= self.case_study_debug_max_per_step:
                return
            if abs(float(return_gap)) < float(self.case_study_debug_min_gap):
                return
            case_study_records.append(
                self._case_study_step_record(
                    uid=uid,
                    anchor_obs=anchor_obs,
                    records=records,
                    source_record=source_record,
                    target_record=target_record,
                    source_branch=source_branch,
                    target_branch=target_branch,
                    return_gap=float(return_gap),
                    ignored_reason=ignored_reason,
                    memory_lookup=case_study_memory_lookup,
                    total_batch_list=total_batch_list,
                )
            )

        def record_activated_pair(source_branch: str, target_branch: str) -> None:
            source_branch = str(source_branch or "").strip().lower()
            target_branch = str(target_branch or "").strip().lower()
            if source_branch == "good" and target_branch == "none":
                cbsd_analysis_stats["skill_to_noskill_pair_count"] += 1.0
            elif source_branch == "none" and target_branch == "good":
                cbsd_analysis_stats["noskill_to_skill_pair_count"] += 1.0

        def finalize_cbsd_analysis_stats() -> dict[str, float]:
            raw_pair_count = float(cbsd_analysis_stats.get("raw_pair_count", 0.0) or 0.0)
            skill_to_noskill_count = float(cbsd_analysis_stats.get("skill_to_noskill_pair_count", 0.0) or 0.0)
            noskill_to_skill_count = float(cbsd_analysis_stats.get("noskill_to_skill_pair_count", 0.0) or 0.0)
            activated_pair_count = skill_to_noskill_count + noskill_to_skill_count
            ignored_pair_count = max(raw_pair_count - activated_pair_count, 0.0)
            denom = max(raw_pair_count, 1.0)
            return {
                "analysis/mech/raw_pair_count": raw_pair_count,
                "analysis/mech/activated_pair_count": activated_pair_count,
                "analysis/mech/margin_filtered_count": float(
                    cbsd_analysis_stats.get("margin_filtered_count", 0.0) or 0.0
                ),
                "analysis/mech/margin_filtered_ratio": float(
                    float(cbsd_analysis_stats.get("margin_filtered_count", 0.0) or 0.0) / denom
                ),
                "analysis/mech/activated_pair_ratio": float(activated_pair_count / denom),
                "analysis/mech/ignored_pair_count": ignored_pair_count,
                "analysis/mech/ignored_pair_ratio": float(ignored_pair_count / denom),
                "analysis/mech/skill_to_noskill_pair_count": skill_to_noskill_count,
                "analysis/mech/noskill_to_skill_pair_count": noskill_to_skill_count,
                "analysis/mech/skill_to_noskill_pair_ratio": float(skill_to_noskill_count / denom),
                "analysis/mech/noskill_to_skill_pair_ratio": float(noskill_to_skill_count / denom),
                "analysis/mech/append_failed_count": float(cbsd_analysis_stats.get("append_failed_count", 0.0) or 0.0),
            }
        for traj_idx, steps in enumerate(total_batch_list):
            if traj_idx >= len(step_returns_by_traj):
                continue
            returns = step_returns_by_traj[traj_idx]
            for step_idx, step in enumerate(steps):
                if step_idx >= len(returns) or not bool(step.get("active_masks", True)):
                    continue
                if not bool(step.get("is_action_valid", True)):
                    continue
                if "responses" not in step or "prompts" not in step or "attention_mask" not in step:
                    continue
                branch = self._cbsd_branch_for_step(step)
                response_length = int(step["responses"].shape[-1])
                response_mask = self._clone_row_tensor(step["attention_mask"][-response_length:]).to(dtype=torch.long)
                if response_mask.sum().item() <= 0:
                    continue
                action_mask = self._cbsd_action_loss_mask(step, response_length=response_length)
                if not use_response_scope and action_mask.sum().item() <= 0:
                    continue
                group_key = (
                    str(step.get("uid", "")),
                    self._stringify_anchor(step.get("anchor_obs", "")),
                )
                record = {
                    "uid": str(group_key[0]),
                    "anchor_obs": str(group_key[1]),
                    "traj_idx": traj_idx,
                    "step_idx": step_idx,
                    "step": step,
                    "branch": branch,
                    "return": float(returns[step_idx]),
                    "action_mask": action_mask,
                    "response_mask": response_mask,
                }
                grouped_records.setdefault(group_key, []).append(record)
                records_by_uid.setdefault(str(group_key[0]), []).append(record)

        aux_rows = []
        aux_meta = []

        def append_aux_row(winner, target_record, target_branch: str, gap: float, uid: str, granularity: str) -> bool:
            response_length = int(winner["step"]["responses"].shape[-1])
            target_prompt_pair = self._cbsd_prompt_for_branch(
                target_record["step"],
                target_branch,
                response_length=response_length,
            )
            if target_prompt_pair is None:
                return False
            reference_prompt_pair = self._cbsd_prompt_for_branch(
                winner["step"],
                winner["branch"],
                response_length=response_length,
            )
            if reference_prompt_pair is None:
                return False

            prompt_ids, prompt_attention = target_prompt_pair
            ref_prompt_ids, ref_prompt_attention = reference_prompt_pair
            responses = self._clone_row_tensor(winner["step"]["responses"])
            response_mask = self._clone_row_tensor(winner["step"]["attention_mask"][-response_length:]).to(
                dtype=prompt_attention.dtype
            )
            if response_mask.sum().item() <= 0:
                return False

            action_mask = self._cbsd_action_loss_mask(winner["step"], response_length=response_length).to(
                dtype=prompt_attention.dtype
            )
            if use_response_scope:
                loss_token_mask = response_mask
            else:
                if action_mask.sum().item() <= 0:
                    return False
                loss_token_mask = action_mask * response_mask
            if loss_token_mask.sum().item() <= 0:
                return False

            input_ids = torch.cat([prompt_ids, responses], dim=-1)
            attention_mask = torch.cat([prompt_attention, response_mask], dim=-1)
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

            full_loss_mask = torch.cat([torch.zeros_like(prompt_attention), loss_token_mask], dim=-1)
            weight = min(max(float(gap) / self.cbsd_tau, 0.0), self.cbsd_weight_max)
            loss_weight = torch.full(
                (response_length,),
                float(weight),
                dtype=torch.float32,
            )
            if use_response_scope and self.cbsd_action_weight != 1.0 and action_mask.sum().item() > 0:
                action_weight = action_mask.to(dtype=torch.float32).clamp(min=0.0, max=1.0)
                loss_weight = loss_weight * (1.0 + ((float(self.cbsd_action_weight) - 1.0) * action_weight))

            aux_rows.append(
                {
                    "prompts": prompt_ids,
                    "responses": responses,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "ucob_cbsd_loss_mask": full_loss_mask,
                    "ucob_cbsd_loss_weight": loss_weight,
                    "ucob_cbsd_ref_prompt_ids": ref_prompt_ids,
                    "ucob_cbsd_ref_prompt_attention_mask": ref_prompt_attention,
                }
            )
            aux_meta.append(
                {
                    "ucob_cbsd_is_aux": True,
                    "ucob_cbsd_source_branch": winner["branch"],
                    "ucob_cbsd_target_branch": target_branch,
                    "ucob_cbsd_return_gap": float(gap),
                    "ucob_cbsd_granularity": granularity,
                    "uid": str(uid),
                    "traj_uid": f"cbsd_{uuid.uuid4().hex[:12]}",
                    "dsdar_rollout_mode": "cbsd",
                    "dynamic_memory_branch": "cbsd",
                    "is_action_valid": True,
                    "active_masks": True,
                }
            )
            return True

        uids_with_turn_aux = set()
        if self.cbsd_opd_granularity in {"turn", "hybrid"}:
            for group_key, records in grouped_records.items():
                sources = [record for record in records if record["branch"] in self.cbsd_source_branches]
                if len(sources) < 1 or len(records) < 1:
                    continue
                winner = max(sources, key=lambda item: item["return"])
                target_branch = self._opposite_cbsd_branch(winner["branch"])
                if self.cbsd_target_policy not in {"opposite", "all"}:
                    target_branch = self.cbsd_target_policy

                group_returns = np.asarray([record["return"] for record in records], dtype=np.float32)
                group_mean = float(np.mean(group_returns)) if len(group_returns) else winner["return"]
                target_records = [
                    record
                    for record in records
                    if record is not winner and self._cbsd_prompt_for_branch(
                        record["step"],
                        target_branch,
                        int(winner["step"]["responses"].shape[-1]),
                    ) is not None
                ]
                if not target_records:
                    target_records = [winner]

                scored_targets = []
                for target in target_records:
                    cbsd_analysis_stats["raw_pair_count"] += 1.0
                    baseline_return = target["return"] if target is not winner else group_mean
                    raw_gap = float(winner["return"] - baseline_return)
                    gap = self._cbsd_effective_teaching_gap(raw_gap)
                    if gap is None:
                        cbsd_analysis_stats["margin_filtered_count"] += 1.0
                        maybe_add_case_study_record(
                            uid=str(group_key[0]),
                            anchor_obs=str(group_key[1]),
                            records=records,
                            source_record=winner,
                            target_record=target,
                            source_branch=winner["branch"],
                            target_branch=target_branch,
                            return_gap=raw_gap,
                            ignored_reason="margin_filtered",
                        )
                        continue
                    scored_targets.append((gap, target))
                scored_targets.sort(key=lambda item: item[0], reverse=True)
                if self.cbsd_max_pairs_per_group > 0:
                    scored_targets = scored_targets[: self.cbsd_max_pairs_per_group]

                for gap, target in scored_targets:
                    if append_aux_row(
                        winner=winner,
                        target_record=target,
                        target_branch=target_branch,
                        gap=gap,
                        uid=str(group_key[0]),
                        granularity="turn",
                    ):
                        uids_with_turn_aux.add(str(group_key[0]))
                        record_activated_pair(winner["branch"], target_branch)
                        maybe_add_case_study_record(
                            uid=str(group_key[0]),
                            anchor_obs=str(group_key[1]),
                            records=records,
                            source_record=winner,
                            target_record=target,
                            source_branch=winner["branch"],
                            target_branch=target_branch,
                            return_gap=gap,
                            ignored_reason="none",
                        )
                    else:
                        cbsd_analysis_stats["append_failed_count"] += 1.0
                        maybe_add_case_study_record(
                            uid=str(group_key[0]),
                            anchor_obs=str(group_key[1]),
                            records=records,
                            source_record=winner,
                            target_record=target,
                            source_branch=winner["branch"],
                            target_branch=target_branch,
                            return_gap=gap,
                            ignored_reason="missing_prompt",
                        )

        if self.cbsd_opd_granularity in {"trajectory", "hybrid"}:
            for uid, records in records_by_uid.items():
                if self.cbsd_opd_granularity == "hybrid" and uid in uids_with_turn_aux:
                    continue
                traj_groups = {}
                for record in records:
                    traj_groups.setdefault(int(record["traj_idx"]), []).append(record)
                traj_summaries = []
                for traj_idx, traj_records in traj_groups.items():
                    traj_records = sorted(traj_records, key=lambda item: int(item["step_idx"]))
                    if not traj_records:
                        continue
                    branch = traj_records[0]["branch"]
                    traj_return = float(traj_records[0]["return"])
                    traj_summaries.append(
                        {
                            "traj_idx": traj_idx,
                            "branch": branch,
                            "return": traj_return,
                            "records": traj_records,
                        }
                    )
                source_trajs = [item for item in traj_summaries if item["branch"] in self.cbsd_source_branches]
                if not source_trajs:
                    continue
                winner_traj = max(source_trajs, key=lambda item: item["return"])
                target_branch = self._opposite_cbsd_branch(winner_traj["branch"])
                if self.cbsd_target_policy not in {"opposite", "all"}:
                    target_branch = self.cbsd_target_policy
                target_trajs = [item for item in traj_summaries if item is not winner_traj and item["branch"] == target_branch]
                if target_trajs:
                    baseline_return = max(item["return"] for item in target_trajs)
                else:
                    other_returns = [item["return"] for item in traj_summaries if item is not winner_traj]
                    baseline_return = float(np.mean(other_returns)) if other_returns else float(winner_traj["return"])
                raw_gap = float(winner_traj["return"] - baseline_return)
                gap = self._cbsd_effective_teaching_gap(raw_gap)
                if gap is None:
                    cbsd_analysis_stats["raw_pair_count"] += 1.0
                    cbsd_analysis_stats["margin_filtered_count"] += 1.0
                    maybe_add_case_study_record(
                        uid=uid,
                        anchor_obs=str(winner_traj["records"][0].get("anchor_obs", "")) if winner_traj["records"] else "",
                        records=records,
                        source_record=winner_traj["records"][0] if winner_traj["records"] else None,
                        target_record=None,
                        source_branch=winner_traj["branch"],
                        target_branch=target_branch,
                        return_gap=raw_gap,
                        ignored_reason="margin_filtered",
                    )
                    continue
                cbsd_analysis_stats["raw_pair_count"] += 1.0

                winner_records = [
                    record
                    for record in winner_traj["records"]
                    if self._cbsd_prompt_for_branch(
                        record["step"],
                        target_branch,
                        int(record["step"]["responses"].shape[-1]),
                    ) is not None
                ]
                max_turns = int(self.cbsd_max_turns_per_trajectory)
                if max_turns > 0:
                    winner_records = winner_records[:max_turns]
                appended_for_pair = False
                for winner in winner_records:
                    if append_aux_row(
                        winner=winner,
                        target_record=winner,
                        target_branch=target_branch,
                        gap=gap,
                        uid=uid,
                        granularity="trajectory",
                    ):
                        appended_for_pair = True
                    else:
                        cbsd_analysis_stats["append_failed_count"] += 1.0
                if appended_for_pair:
                    record_activated_pair(winner_traj["branch"], target_branch)
                    maybe_add_case_study_record(
                        uid=uid,
                        anchor_obs=str(winner_records[0].get("anchor_obs", "")) if winner_records else "",
                        records=records,
                        source_record=winner_records[0] if winner_records else None,
                        target_record=winner_records[0] if winner_records else None,
                        source_branch=winner_traj["branch"],
                        target_branch=target_branch,
                        return_gap=gap,
                        ignored_reason="none",
                    )

        finalized_cbsd_analysis_stats = finalize_cbsd_analysis_stats()
        self.last_cbsd_analysis_stats = finalized_cbsd_analysis_stats
        self._case_study_write_records(case_study_records)
        if not aux_rows:
            return None

        tensors = {key: torch.stack([row[key] for row in aux_rows], dim=0) for key in aux_rows[0].keys()}
        non_tensors = {
            key: np.array([meta.get(key, "") for meta in aux_meta], dtype=object)
            for key in aux_meta[0].keys()
        }
        aux_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        aux_batch.meta_info.update(finalized_cbsd_analysis_stats)
        return aux_batch

    def _update_step_relative_memory_utility(self, total_batch_list, step_returns_by_traj) -> dict:
        if self.dynamic_memory_provider is None:
            return {"groups": 0, "updates": 0, "task_updates": 0, "state_updates": 0}

        grouped_steps = {}
        for traj_idx, steps in enumerate(total_batch_list):
            if traj_idx >= len(step_returns_by_traj):
                continue
            returns = step_returns_by_traj[traj_idx]
            for step_idx, step in enumerate(steps):
                if step_idx >= len(returns) or not bool(step.get("active_masks", True)):
                    continue
                memory_ids = self._coerce_memory_id_list(step.get("dynamic_memory_ids", []))
                group_key = (
                    str(step.get("uid", "")),
                    self._stringify_anchor(step.get("anchor_obs", "")),
                )
                grouped_steps.setdefault(group_key, []).append(
                    {
                        "return": float(returns[step_idx]),
                        "memory_ids": memory_ids,
                        "branch": str(step.get("dynamic_memory_branch", "")),
                    }
                )

        updates = []
        usable_groups = 0
        tau = max(float(self.dynamic_memory_step_utility_tau), 1e-6)
        for group_steps in grouped_steps.values():
            if len(group_steps) < 2:
                continue
            returns = np.asarray([item["return"] for item in group_steps], dtype=np.float32)
            mean_return = float(np.mean(returns))
            usable_groups += 1
            for item in group_steps:
                if item.get("branch") != "good" or not item.get("memory_ids"):
                    continue
                relative_delta = (float(item["return"]) - mean_return) / tau
                score = 0.5 + (0.5 * relative_delta)
                score = max(0.0, min(1.0, score))
                for memory_id in item["memory_ids"]:
                    updates.append((memory_id, score))

        task_updates = 0
        state_updates = 0
        for memory_id, _ in updates:
            memory_pool = self.dynamic_memory_provider.memory_pool_by_id(memory_id)
            if memory_pool == "state":
                state_updates += 1
            elif memory_pool == "task":
                task_updates += 1
        updated = self.dynamic_memory_provider.update_utility_by_id_scores(updates) if updates else 0
        return {
            "groups": usable_groups,
            "updates": updated,
            "task_updates": task_updates,
            "state_updates": state_updates,
        }

    def _update_baseline_relative_memory_utility(
        self,
        total_batch_list,
        terminal_successes,
        rollout_skill_mask,
    ) -> dict:
        if self.dynamic_memory_provider is None:
            return {"groups": 0, "updates": 0, "task_updates": 0, "state_updates": 0}

        batch_size = min(len(total_batch_list), len(terminal_successes), len(rollout_skill_mask))
        if batch_size <= 0:
            return {"groups": 0, "updates": 0, "task_updates": 0, "state_updates": 0}

        successes = np.asarray([1.0 if terminal_successes[i] else 0.0 for i in range(batch_size)], dtype=np.float32)
        skill_mask = np.asarray(rollout_skill_mask[:batch_size], dtype=bool)
        group_n = max(int(self.config.env.rollout.n), 1)
        task_updates = []
        state_updates = []
        usable_groups = 0

        for start in range(0, batch_size, group_n):
            end = min(start + group_n, batch_size)
            group_indices = np.arange(start, end)
            good_indices = group_indices[skill_mask[start:end]]
            baseline_indices = group_indices[~skill_mask[start:end]]
            if len(good_indices) == 0:
                continue

            baseline_success = float(np.mean(successes[baseline_indices])) if len(baseline_indices) > 0 else 0.0
            good_success = float(np.mean(successes[good_indices]))
            task_credit = good_success - baseline_success
            usable_groups += 1

            task_memory_ids = set()
            for traj_idx in good_indices:
                for step in total_batch_list[int(traj_idx)]:
                    if not bool(step.get("active_masks", True)):
                        continue
                    if str(step.get("dynamic_memory_branch", "")) != "good":
                        continue
                    task_memory_ids.update(self._memory_ids_from_step_by_pool(step, "task"))
            task_updates.extend((memory_id, task_credit) for memory_id in sorted(task_memory_ids))

            for traj_idx in good_indices:
                state_credit = float(successes[int(traj_idx)] - baseline_success)
                state_memory_ids = set()
                for step in total_batch_list[int(traj_idx)]:
                    if not bool(step.get("active_masks", True)):
                        continue
                    if str(step.get("dynamic_memory_branch", "")) != "good":
                        continue
                    state_memory_ids.update(self._memory_ids_from_step_by_pool(step, "state"))
                state_updates.extend((memory_id, state_credit) for memory_id in sorted(state_memory_ids))

        task_updated = (
            self.dynamic_memory_provider.update_utility_by_id_scores(
                task_updates,
                beta=self.dynamic_memory_provider.task_utility_beta,
            )
            if task_updates
            else 0
        )
        state_updated = (
            self.dynamic_memory_provider.update_utility_by_id_scores(
                state_updates,
                beta=self.dynamic_memory_provider.state_utility_beta,
            )
            if state_updates
            else 0
        )
        return {
            "groups": usable_groups,
            "updates": int(task_updated) + int(state_updated),
            "task_updates": int(task_updated),
            "state_updates": int(state_updated),
        }

    def _retrieve_dynamic_memory_for_step(
        self,
        tasks,
        state_contexts,
        is_train: bool,
        memory_counterfactual_mode: str,
        pool: str = "all",
    ):
        pool = str(pool or "all").strip().lower()
        if pool in {"task", "state"}:
            if not self._dynamic_memory_pool_enabled(pool):
                empty = [[] for _ in range(len(tasks))]
                return empty, empty
            provider = self.dynamic_memory_provider
            is_eval = not is_train
            if pool == "task":
                top_k = provider.eval_task_top_k if is_eval else provider.task_top_k
            else:
                top_k = provider.eval_state_top_k if is_eval else provider.state_top_k
            if memory_counterfactual_mode == "bad_memory":
                return provider.memory.retrieve_good_bad_many(
                    tasks,
                    top_k=top_k,
                    filter_type=provider.filter_type,
                    state_contexts=state_contexts,
                    pool=pool,
                )
            retrieved = provider.memory.retrieve_many(
                tasks,
                top_k=top_k,
                filter_type=provider.filter_type,
                state_contexts=state_contexts,
                pool=pool,
            )
            return retrieved, [[] for _ in range(len(tasks))]

        if memory_counterfactual_mode == "bad_memory":
            return self.dynamic_memory_provider.retrieve_good_bad_for_tasks(
                tasks,
                is_eval=not is_train,
                state_contexts=state_contexts,
            )
        retrieved = self.dynamic_memory_provider.retrieve_for_tasks(
            tasks,
            is_eval=not is_train,
            state_contexts=state_contexts,
        )
        return retrieved, [[] for _ in range(len(tasks))]

    def _mix_dataproto_by_mask(self, left: DataProto, right: DataProto, right_mask: np.ndarray) -> DataProto:
        right_mask = np.asarray(right_mask, dtype=bool)
        tensors = {}
        if left.batch is not None:
            for key in left.batch.keys():
                left_tensor = left.batch[key]
                right_tensor = right.batch[key]
                mask = torch.as_tensor(right_mask, device=left_tensor.device)
                view_shape = [mask.shape[0]] + [1] * (left_tensor.dim() - 1)
                tensors[key] = torch.where(mask.view(*view_shape), right_tensor, left_tensor)

        non_tensors = {}
        for key, left_value in left.non_tensor_batch.items():
            right_value = right.non_tensor_batch[key]
            non_tensors[key] = np.array(
                [right_value[i] if right_mask[i] else left_value[i] for i in range(len(right_mask))],
                dtype=object,
            )

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=left.meta_info)

    def _build_output_with_prompt(self, prompt_batch_input: DataProto, policy_output: DataProto) -> DataProto:
        responses = policy_output.batch["responses"]
        response_length = responses.size(1)

        prompt_ids = prompt_batch_input.batch["input_ids"]
        prompt_mask = prompt_batch_input.batch["attention_mask"]
        response_mask = policy_output.batch["attention_mask"][:, -response_length:]

        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask)

        tensors = {
            "prompts": prompt_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if "loss_mask" in policy_output.batch.keys():
            tensors["loss_mask"] = torch.cat(
                [torch.zeros_like(prompt_mask), policy_output.batch["loss_mask"][:, -response_length:]],
                dim=-1,
            )

        return DataProto.from_dict(tensors=tensors, meta_info=policy_output.meta_info)

    def _encode_reflection_prompt(self, prompt: str):
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        chat = [{"content": str(prompt or ""), "role": "user"}]
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        return input_ids[0], attention_mask[0]

    def _encode_reflection_response(self, response: str):
        response_ids = self.tokenizer.encode(str(response or ""), add_special_tokens=False)
        max_response_length = max(int(self.reflection_train_response_length), 1)
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]
        if not response_ids:
            return None, None
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        valid_length = len(response_ids)
        padded_ids = response_ids + [pad_token_id] * (max_response_length - valid_length)
        response_mask = [1] * valid_length + [0] * (max_response_length - valid_length)
        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(response_mask, dtype=torch.long)

    @staticmethod
    def _memory_pool_from_item(item: dict) -> str:
        pool = str(item.get("reflection_train_pool", "") or "").strip().lower()
        if pool in {"task", "state"}:
            return pool
        memory_type = str(item.get("memory_type", "") or "").strip().lower()
        return "state" if memory_type == "state_skill" else "task"

    def _build_reflection_train_row(self, record: dict):
        try:
            prompt_ids, prompt_attention = self._encode_reflection_prompt(record["prompt"])
            responses, response_mask = self._encode_reflection_response(record["response"])
        except Exception as exc:
            print(f"[UCOB][skill_memory] skip reflection train row: {exc}")
            return None
        if responses is None or response_mask is None or response_mask.sum().item() <= 0:
            return None
        max_model_len = int(self.config.actor_rollout_ref.rollout.get("max_model_len", 0) or 0)
        if max_model_len > 0:
            tensor_seq_len = int(prompt_ids.numel() + responses.numel())
            if tensor_seq_len > max_model_len:
                print(
                    "[UCOB][skill_memory] skip reflection train row: "
                    f"sequence_length={tensor_seq_len} is larger than max_model_len={max_model_len}."
                )
                return None

        attention_mask = torch.cat([prompt_attention, response_mask.to(dtype=prompt_attention.dtype)], dim=-1)
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
        advantage = torch.full(
            responses.shape,
            float(record["advantage"]),
            dtype=torch.float32,
        ) * response_mask.to(dtype=torch.float32)
        return {
            "prompts": prompt_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "advantages": advantage,
            "returns": advantage.clone(),
            "dynamic_memory_reflection_train_mask": torch.ones((), dtype=torch.float32),
            "dynamic_memory_reflection_advantage": torch.tensor(float(record["advantage"]), dtype=torch.float32),
            "dynamic_memory_reflection_utility": torch.tensor(float(record["utility"]), dtype=torch.float32),
            "dynamic_memory_reflection_step_count": torch.tensor(float(record["step_count"]), dtype=torch.float32),
        }

    def _collect_current_step_memory_counts(self, total_batch_list) -> dict:
        counts = {"task": {}, "state": {}}
        for steps in total_batch_list:
            for step in steps:
                if not bool(step.get("active_masks", True)):
                    continue
                if str(step.get("dynamic_memory_branch", "")) != "good":
                    continue
                for pool in ("task", "state"):
                    for memory_id in self._memory_ids_from_step_by_pool(step, pool):
                        pool_counts = counts.setdefault(pool, {})
                        pool_counts[memory_id] = pool_counts.get(memory_id, 0) + 1
        return counts

    def _select_reflection_train_records_for_pool(self, pool: str, memory_counts: dict) -> list:
        if self.dynamic_memory_provider is None:
            return []
        pool = str(pool or "task").strip().lower()
        max_samples = (
            self.reflection_train_max_state_samples_per_step
            if pool == "state"
            else self.reflection_train_max_task_samples_per_step
        )
        min_lifetime_count = (
            self.reflection_train_min_lifetime_count_state
            if pool == "state"
            else self.reflection_train_min_lifetime_count_task
        )
        if max_samples <= 0:
            return []

        candidates = []
        for memory_id, step_count in memory_counts.items():
            item = self.dynamic_memory_provider.get_by_id(memory_id)
            if not item:
                continue
            if self._memory_pool_from_item(item) != pool:
                continue
            if item.get("reflection_train_backend") != "vllm_self_reflection":
                continue
            prompt = str(item.get("reflection_prompt", "") or "")
            response = str(item.get("reflection_response", "") or "")
            if not prompt.strip() or not response.strip():
                continue
            lifetime_count = int(item.get("count", 1))
            if lifetime_count < min_lifetime_count:
                continue
            utility = float(item.get("utility_score", 0.0))
            if abs(utility) < self.reflection_train_min_abs_utility:
                continue
            candidates.append(
                {
                    "memory_id": memory_id,
                    "pool": pool,
                    "prompt": prompt,
                    "response": response,
                    "utility": max(-1.0, min(1.0, utility)),
                    "step_count": int(step_count),
                    "lifetime_count": lifetime_count,
                }
            )

        candidates.sort(
            key=lambda item: (
                int(item["step_count"]),
                abs(float(item["utility"])),
                int(item["lifetime_count"]),
            ),
            reverse=True,
        )
        candidates = candidates[:max_samples]
        if len(candidates) < 2:
            return []

        rewards = np.asarray([item["utility"] for item in candidates], dtype=np.float32)
        mean = float(np.mean(rewards))
        std = max(float(np.std(rewards)), float(self.reflection_advantage_eps))
        advantage_clip = float(self.reflection_advantage_clip)
        for item in candidates:
            advantage = (float(item["utility"]) - mean) / std
            if advantage_clip > 0:
                advantage = max(-advantage_clip, min(advantage_clip, advantage))
            item["advantage"] = float(advantage)
        return [item for item in candidates if abs(float(item.get("advantage", 0.0))) > 0.0]

    def _build_dynamic_memory_reflection_train_batch(self, total_batch_list) -> DataProto | None:
        if not self._reflection_train_active():
            return None
        if self.reflection_train_source != "current_step_used_memory":
            return None
        memory_counts = self._collect_current_step_memory_counts(total_batch_list)
        records = []
        if self._dynamic_memory_pool_enabled("task"):
            records.extend(self._select_reflection_train_records_for_pool("task", memory_counts.get("task", {})))
        if self._dynamic_memory_pool_enabled("state"):
            records.extend(self._select_reflection_train_records_for_pool("state", memory_counts.get("state", {})))
        if not records:
            return None

        rows = []
        meta = []
        for record in records:
            row = self._build_reflection_train_row(record)
            if row is None:
                continue
            rows.append(row)
            meta.append(
                {
                    "uid": f"reflection_{record['pool']}",
                    "traj_uid": f"reflection_{record['memory_id']}_{uuid.uuid4().hex[:8]}",
                    "dsdar_rollout_mode": "reflection_train",
                    "dynamic_memory_branch": "reflection_train",
                    "dynamic_memory_reflection_memory_id": record["memory_id"],
                    "dynamic_memory_reflection_pool": record["pool"],
                    "dynamic_memory_reflection_is_aux": True,
                    "is_action_valid": True,
                    "active_masks": True,
                }
            )

        if not rows:
            return None

        tensors = {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0].keys()}
        non_tensors = {
            key: np.array([item.get(key, "") for item in meta], dtype=object)
            for key in meta[0].keys()
        }
        batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        batch.meta_info = dict(getattr(self.config.actor_rollout_ref.rollout, "meta_info", {}) or {})
        task_rows = sum(1 for item in meta if item["dynamic_memory_reflection_pool"] == "task")
        state_rows = sum(1 for item in meta if item["dynamic_memory_reflection_pool"] == "state")
        print(
            "[UCOB][skill_memory] reflection train rows: "
            f"total={len(rows)}, task={task_rows}, state={state_rows}"
        )
        return batch

    @staticmethod
    def _find_action_spans(text: str):
        spans = []
        for pattern in _ACTION_SPAN_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            match = _UNCLOSED_ACTION_SPAN_PATTERN.search(text)
            if match is not None:
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            match = _UNCLOSED_CODE_SPAN_PATTERN.search(text)
            if match is not None:
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            match = _UNCLOSED_SEARCH_SPAN_PATTERN.search(text)
            if match is not None:
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            match = _UNCLOSED_ANSWER_SPAN_PATTERN.search(text)
            if match is not None:
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            spans.extend(match.span() for match in _WEBSHOP_ACTION_PATTERN.finditer(text))
        if not spans:
            return []

        spans = sorted(spans)
        merged = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _overlaps_action_span(start: int, end: int, spans) -> bool:
        if end <= start:
            return False
        return any(start < span_end and end > span_start for span_start, span_end in spans)

    def _build_action_token_mask(
        self,
        responses: torch.Tensor,
        response_texts,
        response_mask: torch.Tensor,
    ):
        action_mask = torch.zeros_like(response_mask, dtype=response_mask.dtype, device=response_mask.device)
        parsed_rows = torch.zeros(response_mask.size(0), dtype=torch.bool, device=response_mask.device)

        for row_idx, response_text in enumerate(response_texts):
            spans = self._find_action_spans(response_text)
            if not spans:
                continue

            valid_positions = torch.nonzero(response_mask[row_idx].bool(), as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            marked = False
            try:
                encoded = self.tokenizer(
                    response_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = encoded.get("offset_mapping", None)
            except Exception:
                offsets = None

            if offsets is not None:
                limit = min(len(offsets), valid_positions.numel())
                for token_offset_idx in range(limit):
                    start, end = offsets[token_offset_idx]
                    if self._overlaps_action_span(int(start), int(end), spans):
                        action_mask[row_idx, valid_positions[token_offset_idx]] = 1
                        marked = True

            if marked:
                parsed_rows[row_idx] = True
                continue

            valid_ids = responses[row_idx, valid_positions].detach().cpu().tolist()
            prev_text = ""
            for token_offset_idx in range(len(valid_ids)):
                current_text = self.tokenizer.decode(valid_ids[: token_offset_idx + 1], skip_special_tokens=True)
                start, end = len(prev_text), len(current_text)
                prev_text = current_text
                if self._overlaps_action_span(start, end, spans):
                    action_mask[row_idx, valid_positions[token_offset_idx]] = 1
                    marked = True

            if marked:
                parsed_rows[row_idx] = True

        return action_mask, parsed_rows

    def _attach_action_loss_mask(self, batch: DataProto, response_texts) -> None:
        if (
            not self.action_loss_mask_enable
            and not self.topk_action_mask_enable
            and not (self.cbsd_enable and self.cbsd_action_mask_enable)
        ):
            return

        responses = batch.batch["responses"]
        response_length = responses.size(1)
        attention_mask = batch.batch["attention_mask"]
        prompt_mask = attention_mask[:, :-response_length]
        response_mask = attention_mask[:, -response_length:]
        action_mask, parsed_rows = self._build_action_token_mask(
            responses=responses,
            response_texts=response_texts,
            response_mask=response_mask,
        )
        full_action_mask = torch.cat(
            [torch.zeros_like(prompt_mask), action_mask.to(dtype=attention_mask.dtype)],
            dim=-1,
        )
        if self.action_loss_mask_enable:
            batch.batch["loss_mask"] = full_action_mask
        if self.topk_action_mask_enable:
            batch.batch["action_token_mask"] = full_action_mask
        if self.cbsd_enable and self.cbsd_action_mask_enable:
            batch.batch["ucob_cbsd_action_token_mask"] = full_action_mask
        batch.non_tensor_batch["action_loss_mask_parse_success"] = parsed_rows.detach().cpu().numpy().astype(bool)
        batch.non_tensor_batch["topk_action_mask_parse_success"] = parsed_rows.detach().cpu().numpy().astype(bool)
        batch.non_tensor_batch["ucob_cbsd_action_parse_success"] = parsed_rows.detach().cpu().numpy().astype(bool)

    def _reflection_sampling_meta(self):
        meta = {}
        if self.reflection_max_response_length is not None:
            response_length = int(self.reflection_max_response_length)
            meta["response_length"] = response_length
            meta["max_tokens"] = response_length
        if self.reflection_temperature is not None:
            meta["temperature"] = float(self.reflection_temperature)
        if self.reflection_top_p is not None:
            meta["top_p"] = float(self.reflection_top_p)
        if self.reflection_top_k is not None:
            meta["top_k"] = int(self.reflection_top_k)
        return meta

    def _reflection_api_endpoint(self) -> str:
        if self.reflection_api_url:
            return str(self.reflection_api_url).strip()

        base_url = str(self.reflection_api_base_url or "").strip().rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    @staticmethod
    def _reflection_ssl_context():
        try:
            import certifi
        except ImportError:
            return None
        return ssl.create_default_context(cafile=certifi.where())

    def _call_reflection_api_once(self, prompt: str) -> str:
        endpoint = self._reflection_api_endpoint()
        if not endpoint:
            raise ValueError(
                "External reflection API requires DYNAMIC_MEMORY_REFLECTION_API_URL "
                "or DYNAMIC_MEMORY_REFLECTION_API_BASE_URL/OPENAI_BASE_URL."
            )
        if not self.reflection_api_model:
            raise ValueError("External reflection API requires DYNAMIC_MEMORY_REFLECTION_API_MODEL.")
        if not self.reflection_api_key:
            raise ValueError("External reflection API requires DYNAMIC_MEMORY_REFLECTION_API_KEY/OPENAI_API_KEY.")

        payload = {
            "model": str(self.reflection_api_model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.reflection_temperature) if self.reflection_temperature is not None else 1.0,
            "top_p": float(self.reflection_top_p) if self.reflection_top_p is not None else 1.0,
            "max_tokens": int(self.reflection_max_response_length) if self.reflection_max_response_length is not None else 2048,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.reflection_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ssl_context = self._reflection_ssl_context()
        if ssl_context is not None:
            response = urllib.request.urlopen(
                request,
                timeout=self.reflection_api_timeout,
                context=ssl_context,
            )
        else:
            response = urllib.request.urlopen(request, timeout=self.reflection_api_timeout)
        with response:
            response_body = response.read().decode("utf-8")

        data = json.loads(response_body)
        choices = data.get("choices", [])
        if not choices:
            return ""
        first_choice = choices[0]
        message = first_choice.get("message", {})
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message.get("content", ""))
        if first_choice.get("text") is not None:
            return str(first_choice.get("text", ""))
        return ""

    def _call_reflection_api(self, prompt: str) -> str:
        last_error = None
        for attempt in range(max(self.reflection_api_retries, 0) + 1):
            try:
                return self._call_reflection_api_once(prompt)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.reflection_api_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        print(f"[UCOB][skill_memory] external reflection API failed: {last_error}")
        return ""

    def _generate_external_api_reflections(self, prompts: list) -> list:
        if not prompts:
            return []
        if not self._reflection_api_endpoint():
            raise ValueError(
                "External reflection API requires DYNAMIC_MEMORY_REFLECTION_API_URL "
                "or DYNAMIC_MEMORY_REFLECTION_API_BASE_URL/OPENAI_BASE_URL."
            )
        if not self.reflection_api_model:
            raise ValueError("External reflection API requires DYNAMIC_MEMORY_REFLECTION_API_MODEL.")
        if not self.reflection_api_key:
            raise ValueError("External reflection API requires DYNAMIC_MEMORY_REFLECTION_API_KEY/OPENAI_API_KEY.")

        max_workers = max(1, min(int(self.reflection_api_max_workers), len(prompts)))
        if max_workers == 1:
            return [self._call_reflection_api(prompt) for prompt in prompts]

        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_reflection_api, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"[UCOB][skill_memory] external reflection API future failed: {exc}")
                    results[idx] = ""
        return results

    def _select_reflection_indices(self, successes, batch_size: int) -> list:
        if self.reflection_per_group <= 0:
            return list(range(batch_size))

        group_n = int(self.config.env.rollout.n)
        if group_n <= 0:
            group_n = batch_size

        selected = []
        for start in range(0, batch_size, group_n):
            group_indices = list(range(start, min(start + group_n, batch_size)))
            group_selected = []
            if self.reflection_selection == "success_failure":
                success_indices = [idx for idx in group_indices if bool(successes[idx])]
                failure_indices = [idx for idx in group_indices if not bool(successes[idx])]
                if success_indices:
                    group_selected.append(success_indices[0])
                if len(group_selected) < self.reflection_per_group and failure_indices:
                    group_selected.append(failure_indices[0])
                for idx in group_indices:
                    if len(group_selected) >= self.reflection_per_group:
                        break
                    if idx not in group_selected:
                        group_selected.append(idx)
            else:
                group_selected = group_indices[: self.reflection_per_group]
            selected.extend(group_selected[: self.reflection_per_group])

        return selected

    @staticmethod
    def _select_reflection_obs(reflection_obs, selected_indices):
        selected_set = list(selected_indices)
        selected_obs = dict(reflection_obs)
        for key in ("text", "anchor", "tasks", "task_categories", "trajectories", "successes"):
            value = reflection_obs.get(key, None)
            if isinstance(value, list):
                selected_obs[key] = [value[i] for i in selected_set]
        return selected_obs

    def _use_vllm_skill_reflection_prompt(self) -> bool:
        return self.reflection_backend in {"vllm", "rollout", ""}

    def _build_pairwise_reflection_prompt(self, task: str, success_trajectory: str, failure_trajectory: str) -> str:
        env_name = str(self.config.env.env_name).lower()
        use_skill_reflection = (
            self.dynamic_memory_provider is not None
            and getattr(self.dynamic_memory_provider, "skill_format", "") == "claude_style"
        )
        use_vllm_skill_prompt = use_skill_reflection and self._use_vllm_skill_reflection_prompt()
        if "alfworld" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.alfworld import ALFWORLD_VLLM_TASK_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.alfworld import ALFWORLD_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.alfworld import ALFWORLD_REFLECT_TEMPLATE as reflect_template
        elif "webshop" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.webshop import WEBSHOP_VLLM_TASK_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.webshop import WEBSHOP_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.webshop import WEBSHOP_REFLECT_TEMPLATE as reflect_template
        elif "sokoban" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.sokoban import SOKOBAN_VLLM_TASK_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.sokoban import SOKOBAN_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.sokoban import SOKOBAN_REFLECT_TEMPLATE as reflect_template
        elif "appworld" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.appworld import APPWORLD_VLLM_TASK_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.appworld import APPWORLD_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.appworld import APPWORLD_REFLECT_TEMPLATE as reflect_template
        elif "search" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.search import SEARCH_VLLM_TASK_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.search import SEARCH_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.search import SEARCH_REFLECT_TEMPLATE as reflect_template
        else:
            raise ValueError(
                "Skill memory group reflection is currently implemented only for ALFWorld, WebShop, Sokoban, AppWorld, and Search."
            )

        if str(success_trajectory or "").strip():
            reference_trajectory = (
                "Reference Successful Trajectory from the same rollout group (for comparison):\n"
                f"{success_trajectory}"
            )
        else:
            reference_trajectory = "No successful attempts available for reference."
        current_trajectory = (
            "Failed Trajectory from the same rollout group. This failed trajectory is the attempt being evaluated:\n"
            f"{failure_trajectory}"
        )
        format_kwargs = dict(
            task_description=task,
            success="unsuccessfully",
            reference_trajectory=reference_trajectory,
            current_trajectory=current_trajectory,
        )
        if use_skill_reflection:
            format_kwargs["task_category"] = self.dynamic_memory_provider.infer_task_category(task)
        return reflect_template.format(**format_kwargs)

    def _rollout_group_success_metadata(self, selected_idx: int, successes, rollout_skill_mask) -> dict:
        if successes is None or rollout_skill_mask is None:
            return {}
        try:
            batch_size = min(len(successes), len(rollout_skill_mask))
        except TypeError:
            return {}
        if selected_idx < 0 or selected_idx >= batch_size:
            return {}

        group_n = max(int(self.config.env.rollout.n), 1)
        group_start = (int(selected_idx) // group_n) * group_n
        group_end = min(group_start + group_n, batch_size)
        group_indices = list(range(group_start, group_end))
        skill_indices = [idx for idx in group_indices if bool(rollout_skill_mask[idx])]
        no_skill_indices = [idx for idx in group_indices if not bool(rollout_skill_mask[idx])]

        def _success_count(indices):
            return int(sum(1 for idx in indices if bool(successes[idx])))

        def _success_rate(indices):
            if not indices:
                return None
            return float(_success_count(indices) / len(indices))

        skill_success_count = _success_count(skill_indices)
        no_skill_success_count = _success_count(no_skill_indices)
        metadata = {
            "rollout_group_start": int(group_start),
            "rollout_group_end": int(group_end),
            "rollout_group_size": int(len(group_indices)),
            "rollout_group_success_rate": _success_rate(group_indices),
            "skill_rollout_count": int(len(skill_indices)),
            "skill_rollout_success_count": int(skill_success_count),
            "skill_rollout_success_rate": _success_rate(skill_indices),
            "no_skill_rollout_count": int(len(no_skill_indices)),
            "no_skill_rollout_success_count": int(no_skill_success_count),
            "no_skill_rollout_success_rate": _success_rate(no_skill_indices),
            "selected_rollout_is_skill": bool(rollout_skill_mask[selected_idx]),
            "selected_rollout_success": bool(successes[selected_idx]),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _attach_rollout_branch_success_rates(self, success: dict) -> dict:
        if not isinstance(success, dict):
            return success
        if "success_rate" not in success or "skill_rollout_mask" not in success:
            return success

        try:
            successes = np.asarray(success["success_rate"], dtype=np.float32)
            skill_mask = np.asarray(success["skill_rollout_mask"], dtype=np.float32).astype(bool)
        except Exception:
            return success

        if len(successes) == 0 or len(successes) != len(skill_mask):
            return success

        no_skill_mask = np.logical_not(skill_mask)

        def _masked_rate(mask: np.ndarray) -> float:
            if not np.any(mask):
                return float("nan")
            return float(np.mean(successes[mask]))

        batch_size = len(successes)
        success["skill_rollout_success_rate"] = np.full(
            batch_size,
            _masked_rate(skill_mask),
            dtype=np.float32,
        )
        success["no_skill_rollout_success_rate"] = np.full(
            batch_size,
            _masked_rate(no_skill_mask),
            dtype=np.float32,
        )
        success["skill_rollout_count"] = np.full(
            batch_size,
            float(np.sum(skill_mask)),
            dtype=np.float32,
        )
        success["no_skill_rollout_count"] = np.full(
            batch_size,
            float(np.sum(no_skill_mask)),
            dtype=np.float32,
        )
        return success

    def _build_group_task_reflection_obs(
        self,
        gen_batch: DataProto,
        reflection_obs: dict,
        rollout_skill_mask=None,
    ) -> tuple:
        tasks = reflection_obs.get("tasks", [])
        task_categories = reflection_obs.get("task_categories", [])
        trajectories = reflection_obs.get("trajectories", [])
        successes = reflection_obs.get("successes", [])
        batch_size = len(tasks)
        if batch_size == 0:
            return [], {}, [], [], [], []

        group_n = int(self.config.env.rollout.n)
        if group_n <= 0:
            group_n = batch_size

        selected_indices = []
        selected_texts = []
        selected_tasks = []
        selected_task_categories = []
        selected_trajectories = []
        selected_successes = []
        selected_metadata = []

        for start in range(0, batch_size, group_n):
            group_indices = list(range(start, min(start + group_n, batch_size)))
            success_indices = [idx for idx in group_indices if bool(successes[idx])]
            failure_indices = [idx for idx in group_indices if not bool(successes[idx])]

            if success_indices and failure_indices:
                success_idx = success_indices[0]
                failure_idx = failure_indices[0]
                task = tasks[failure_idx]
                success_trajectory = trajectories[success_idx] if success_idx < len(trajectories) else ""
                failure_trajectory = trajectories[failure_idx] if failure_idx < len(trajectories) else ""
                combined_trajectory = (
                    "Reference Successful Trajectory from the same rollout group:\n"
                    f"{success_trajectory}\n\n"
                    "Failed Trajectory from the same rollout group:\n"
                    f"{failure_trajectory}"
                )
                selected_indices.append(failure_idx)
                selected_texts.append(self._build_pairwise_reflection_prompt(task, success_trajectory, failure_trajectory))
                selected_tasks.append(task)
                if failure_idx < len(task_categories):
                    selected_task_categories.append(task_categories[failure_idx])
                elif self.dynamic_memory_provider is not None:
                    selected_task_categories.append(self.dynamic_memory_provider.infer_task_category(task))
                selected_trajectories.append(combined_trajectory)
                selected_successes.append(False)
                selected_metadata.append(
                    self._rollout_group_success_metadata(failure_idx, successes, rollout_skill_mask)
                )
            elif failure_indices:
                failure_idx = failure_indices[0]
                task = tasks[failure_idx]
                failure_trajectory = trajectories[failure_idx] if failure_idx < len(trajectories) else ""
                selected_indices.append(failure_idx)
                selected_texts.append(self._build_pairwise_reflection_prompt(task, "", failure_trajectory))
                selected_tasks.append(task)
                if failure_idx < len(task_categories):
                    selected_task_categories.append(task_categories[failure_idx])
                elif self.dynamic_memory_provider is not None:
                    selected_task_categories.append(self.dynamic_memory_provider.infer_task_category(task))
                selected_trajectories.append(failure_trajectory)
                selected_successes.append(False)
                selected_metadata.append(
                    self._rollout_group_success_metadata(failure_idx, successes, rollout_skill_mask)
                )

        if not selected_indices:
            return [], {}, [], [], [], []

        selected_reflection_obs = dict(reflection_obs)
        selected_reflection_obs["text"] = selected_texts
        selected_reflection_obs["anchor"] = selected_texts.copy()
        selected_reflection_obs["image"] = None
        selected_reflection_obs["tasks"] = selected_tasks
        selected_reflection_obs["task_categories"] = selected_task_categories
        selected_reflection_obs["trajectories"] = selected_trajectories
        selected_reflection_obs["successes"] = selected_successes
        selected_gen_batch = gen_batch.select_idxs(selected_indices)
        return (
            selected_indices,
            selected_reflection_obs,
            selected_tasks,
            selected_trajectories,
            selected_successes,
            selected_metadata,
        )

    def _decode_step_response(self, step: dict) -> str:
        responses = step.get("responses", None)
        if responses is None:
            return ""
        if isinstance(responses, torch.Tensor):
            token_ids = responses.detach().cpu().tolist()
        elif isinstance(responses, np.ndarray):
            token_ids = responses.tolist()
        else:
            token_ids = responses
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return str(responses)

    def _build_step_contrast_reflection_prompt(
        self,
        task: str,
        anchor_obs: str,
        better_action: str,
        better_return: float,
        worse_action: str,
        worse_return: float,
    ) -> str:
        env_name = str(self.config.env.env_name).lower()
        use_skill_reflection = (
            self.dynamic_memory_provider is not None
            and getattr(self.dynamic_memory_provider, "skill_format", "") == "claude_style"
        )
        use_vllm_skill_prompt = use_skill_reflection and self._use_vllm_skill_reflection_prompt()
        if "alfworld" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.alfworld import ALFWORLD_VLLM_STATE_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.alfworld import ALFWORLD_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.alfworld import ALFWORLD_REFLECT_TEMPLATE as reflect_template
        elif "webshop" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.webshop import WEBSHOP_VLLM_STATE_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.webshop import WEBSHOP_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.webshop import WEBSHOP_REFLECT_TEMPLATE as reflect_template
        elif "sokoban" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.sokoban import SOKOBAN_VLLM_STATE_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.sokoban import SOKOBAN_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.sokoban import SOKOBAN_REFLECT_TEMPLATE as reflect_template
        elif "appworld" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.appworld import APPWORLD_VLLM_STATE_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.appworld import APPWORLD_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.appworld import APPWORLD_REFLECT_TEMPLATE as reflect_template
        elif "search" in env_name:
            if use_vllm_skill_prompt:
                from agent_system.environments.prompts.search import SEARCH_VLLM_STATE_SKILL_REFLECT_TEMPLATE as reflect_template
            elif use_skill_reflection:
                from agent_system.environments.prompts.search import SEARCH_SKILL_REFLECT_TEMPLATE as reflect_template
            else:
                from agent_system.environments.prompts.search import SEARCH_REFLECT_TEMPLATE as reflect_template
        else:
            raise ValueError(
                "Skill memory step reflection is currently implemented only for ALFWorld, WebShop, Sokoban, AppWorld, and Search."
            )

        reference_trajectory = (
            "Reference Better Step from the same rollout group and same observation:\n"
            f"Observation: {anchor_obs}\n"
            f"Action: {better_action}\n"
            f"Discounted return after this step: {better_return:.4f}"
        )
        current_trajectory = (
            "Worse Step from the same rollout group and same observation. "
            "This worse step is the attempt being evaluated:\n"
            f"Observation: {anchor_obs}\n"
            f"Action: {worse_action}\n"
            f"Discounted return after this step: {worse_return:.4f}"
        )
        format_kwargs = dict(
            task_description=task,
            success="unsuccessfully",
            reference_trajectory=reference_trajectory,
            current_trajectory=current_trajectory,
        )
        if use_skill_reflection:
            format_kwargs["task_category"] = self.dynamic_memory_provider.infer_task_category(task)
        return reflect_template.format(**format_kwargs)

    def _select_step_contrast_reflection_records(
        self,
        tasks,
        total_batch_list,
        step_returns_by_traj,
        terminal_successes=None,
        rollout_skill_mask=None,
    ) -> list:
        grouped_steps = {}
        for traj_idx, steps in enumerate(total_batch_list):
            if traj_idx >= len(step_returns_by_traj):
                continue
            returns = step_returns_by_traj[traj_idx]
            for step_idx, step in enumerate(steps):
                if step_idx >= len(returns) or not bool(step.get("active_masks", True)):
                    continue
                anchor_obs = self._stringify_anchor(step.get("anchor_obs", ""))
                if not anchor_obs:
                    continue
                uid = str(step.get("uid", ""))
                key = (uid, anchor_obs)
                grouped_steps.setdefault(key, []).append(
                    {
                        "traj_idx": traj_idx,
                        "step_idx": step_idx,
                        "return": float(returns[step_idx]),
                        "step": step,
                    }
                )

        candidates_by_uid = {}
        for (uid, anchor_obs), records in grouped_steps.items():
            if len(records) < 2:
                continue
            records = sorted(records, key=lambda item: item["return"])
            worst = records[0]
            best = records[-1]
            gap = float(best["return"] - worst["return"])
            if gap < self.dynamic_memory_step_reflection_min_gap:
                continue
            traj_idx = int(worst["traj_idx"])
            task = tasks[traj_idx] if traj_idx < len(tasks) else ""
            better_action = self._decode_step_response(best["step"])
            worse_action = self._decode_step_response(worst["step"])
            prompt = self._build_step_contrast_reflection_prompt(
                task=task,
                anchor_obs=anchor_obs,
                better_action=better_action,
                better_return=float(best["return"]),
                worse_action=worse_action,
                worse_return=float(worst["return"]),
            )
            record = {
                "uid": uid,
                "selected_idx": traj_idx,
                "task": task,
                "prompt": prompt,
                "trajectory": (
                    "Better step:\n"
                    f"Observation: {anchor_obs}\nAction: {better_action}\nReturn: {float(best['return']):.4f}\n\n"
                    "Worse step:\n"
                    f"Observation: {anchor_obs}\nAction: {worse_action}\nReturn: {float(worst['return']):.4f}"
                ),
                "metadata": {
                    "memory_type": "state_skill",
                    "source_anchor_obs": anchor_obs,
                    "state_summary": anchor_obs,
                    "better_action": better_action,
                    "worse_action": worse_action,
                    "step_return_gap": gap,
                    "better_step_return": float(best["return"]),
                    "worse_step_return": float(worst["return"]),
                },
            }
            record["metadata"].update(
                self._rollout_group_success_metadata(traj_idx, terminal_successes, rollout_skill_mask)
            )
            candidates_by_uid.setdefault(uid, []).append(record)

        selected = []
        per_group = max(int(self.dynamic_memory_step_reflection_per_group), 0)
        if per_group <= 0:
            return selected
        for uid in sorted(candidates_by_uid):
            candidates = sorted(
                candidates_by_uid[uid],
                key=lambda item: item["metadata"]["step_return_gap"],
                reverse=True,
            )
            selected.extend(candidates[:per_group])
        return selected

    def _generate_dynamic_memory_step_reflections(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        tasks,
        total_batch_list,
        step_returns_by_traj,
        terminal_successes=None,
        rollout_skill_mask=None,
        efficiency_stats: dict | None = None,
    ):
        if (
            not self.dynamic_memory_enabled
            or self.dynamic_memory_provider is None
            or not self.dynamic_memory_step_reflection_enable
            or not self._dynamic_memory_pool_enabled("state")
        ):
            return np.zeros(len(gen_batch.batch), dtype=np.float32), {"attempted": 0, "added": 0, "reasons": {}}

        records = self._select_step_contrast_reflection_records(
            tasks=tasks,
            total_batch_list=total_batch_list,
            step_returns_by_traj=step_returns_by_traj,
            terminal_successes=terminal_successes,
            rollout_skill_mask=rollout_skill_mask,
        )
        if not records:
            return np.zeros(len(gen_batch.batch), dtype=np.float32), {"attempted": 0, "added": 0, "reasons": {}}

        selected_indices = [record["selected_idx"] for record in records]
        selected_reflection_obs = {
            "text": [record["prompt"] for record in records],
            "anchor": [record["prompt"] for record in records],
            "image": None,
        }
        selected_gen_batch = gen_batch.select_idxs(selected_indices)

        generation_start = time.perf_counter()
        if self.reflection_backend in {"external_api", "api", "openai"}:
            reflection_texts = self._generate_external_api_reflections(selected_reflection_obs.get("text", []))
        elif self.reflection_backend in {"vllm", "rollout", ""}:
            selected_reflection_obs = self._truncate_reflection_obs_for_vllm(selected_reflection_obs)
            reflection_batch = self._collector.preprocess_batch(gen_batch=selected_gen_batch, obs=selected_reflection_obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in reflection_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in reflection_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in reflection_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")

            reflection_batch_input = reflection_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            reflection_batch_input.meta_info = dict(gen_batch.meta_info)
            reflection_batch_input.meta_info.update(self._reflection_sampling_meta())

            reflection_batch_input_padded, pad_size = pad_dataproto_to_divisor(
                reflection_batch_input,
                actor_rollout_wg.world_size,
            )
            reflection_output_padded = actor_rollout_wg.generate_sequences(reflection_batch_input_padded)
            reflection_output = unpad_dataproto(reflection_output_padded, pad_size=pad_size)
            reflection_texts = self.tokenizer.batch_decode(
                reflection_output.batch["responses"],
                skip_special_tokens=True,
            )
        else:
            raise ValueError(
                f"Unsupported algorithm.ucob.skill_memory.reflection_backend={self.reflection_backend!r}; "
                "choose 'vllm' or 'external_api'."
            )
        generation_elapsed = time.perf_counter() - generation_start
        if efficiency_stats is not None:
            efficiency_stats["skill_generation_time_s"] = (
                float(efficiency_stats.get("skill_generation_time_s", 0.0)) + generation_elapsed
            )
            efficiency_stats["state_skill_generation_time_s"] = (
                float(efficiency_stats.get("state_skill_generation_time_s", 0.0)) + generation_elapsed
            )

        write_success = np.zeros(len(gen_batch.batch), dtype=np.float32)
        reason_counts = {}
        prompt_texts = selected_reflection_obs.get("text", [])
        write_start = time.perf_counter()
        for i, (record, reflection_text) in enumerate(zip(records, reflection_texts)):
            metadata = dict(record["metadata"])
            if self._reflection_train_metadata_active():
                metadata.update(
                    self._reflection_training_metadata(
                        prompt=prompt_texts[i] if i < len(prompt_texts) else record.get("prompt", ""),
                        response=reflection_text,
                        pool="state",
                    )
                )
            added, reason = self.dynamic_memory_provider.add_from_reflection_output(
                task_description=record["task"],
                reflection_output=reflection_text,
                trajectory=record["trajectory"],
                actual_success=False,
                metadata=metadata,
                save=False,
            )
            original_idx = int(record["selected_idx"])
            if original_idx < len(write_success):
                write_success[original_idx] = 1.0 if added else 0.0
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reflection_texts:
            self.dynamic_memory_provider.save()
        write_elapsed = time.perf_counter() - write_start
        if efficiency_stats is not None:
            efficiency_stats["skill_write_time_s"] = (
                float(efficiency_stats.get("skill_write_time_s", 0.0)) + write_elapsed
            )
            efficiency_stats["state_skill_write_time_s"] = (
                float(efficiency_stats.get("state_skill_write_time_s", 0.0)) + write_elapsed
            )

        stats = {
            "attempted": len(reflection_texts),
            "added": int(write_success.sum()),
            "selected": len(records),
            "reasons": reason_counts,
            "memory_size": len(self.dynamic_memory_provider.memory),
        }
        print(f"[UCOB][skill_memory] step reflection write stats: {stats}")
        return write_success, stats

    def _generate_dynamic_memory_reflections(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs,
        terminal_infos,
        episode_lengths,
        rollout_skill_mask=None,
        efficiency_stats: dict | None = None,
    ):
        if not self.dynamic_memory_enabled or self.dynamic_memory_provider is None:
            return np.zeros(len(gen_batch.batch), dtype=np.float32), {}
        if not self._dynamic_memory_pool_enabled("task"):
            return np.zeros(len(gen_batch.batch), dtype=np.float32), {"attempted": 0, "added": 0, "reasons": {}}
        if not hasattr(envs, "build_reflection_observations"):
            raise ValueError(
                "Skill memory reflection is currently implemented only for ALFWorld, WebShop, Sokoban, AppWorld, and Search."
            )

        reflection_obs = envs.build_reflection_observations(
            infos=terminal_infos,
            dynamic_memory_provider=self.dynamic_memory_provider,
            episode_lengths=episode_lengths,
        )
        tasks = reflection_obs.get("tasks", envs.get_task_descriptions())
        trajectories = reflection_obs.get("trajectories", [])
        successes = reflection_obs.get("successes", self._get_terminal_successes(envs, terminal_infos, episode_lengths))
        if self.reflection_selection == "group_task":
            (
                selected_indices,
                selected_reflection_obs,
                selected_tasks,
                selected_trajectories,
                selected_successes,
                selected_metadata,
            ) = self._build_group_task_reflection_obs(
                gen_batch=gen_batch,
                reflection_obs=reflection_obs,
                rollout_skill_mask=rollout_skill_mask,
            )
        else:
            selected_indices = self._select_reflection_indices(successes=successes, batch_size=len(tasks))
            selected_gen_batch = gen_batch.select_idxs(selected_indices) if selected_indices else None
            selected_reflection_obs = self._select_reflection_obs(reflection_obs, selected_indices) if selected_indices else {}
            selected_tasks = [tasks[i] for i in selected_indices]
            selected_trajectories = [trajectories[i] if i < len(trajectories) else "" for i in selected_indices]
            selected_successes = [bool(successes[i]) for i in selected_indices]
            selected_metadata = [
                self._rollout_group_success_metadata(i, successes, rollout_skill_mask)
                for i in selected_indices
            ]

        if not selected_indices:
            return np.zeros(len(gen_batch.batch), dtype=np.float32), {"attempted": 0, "added": 0, "reasons": {}}

        if self.reflection_selection == "group_task":
            selected_gen_batch = gen_batch.select_idxs(selected_indices)

        generation_start = time.perf_counter()
        if self.reflection_backend in {"external_api", "api", "openai"}:
            reflection_texts = self._generate_external_api_reflections(selected_reflection_obs.get("text", []))
        elif self.reflection_backend in {"vllm", "rollout", ""}:
            selected_reflection_obs = self._truncate_reflection_obs_for_vllm(selected_reflection_obs)
            reflection_batch = self._collector.preprocess_batch(gen_batch=selected_gen_batch, obs=selected_reflection_obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in reflection_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in reflection_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in reflection_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")

            reflection_batch_input = reflection_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            reflection_batch_input.meta_info = dict(gen_batch.meta_info)
            reflection_batch_input.meta_info.update(self._reflection_sampling_meta())

            reflection_batch_input_padded, pad_size = pad_dataproto_to_divisor(
                reflection_batch_input,
                actor_rollout_wg.world_size,
            )
            reflection_output_padded = actor_rollout_wg.generate_sequences(reflection_batch_input_padded)
            reflection_output = unpad_dataproto(reflection_output_padded, pad_size=pad_size)
            reflection_texts = self.tokenizer.batch_decode(
                reflection_output.batch["responses"],
                skip_special_tokens=True,
            )
        else:
            raise ValueError(
                f"Unsupported algorithm.ucob.skill_memory.reflection_backend={self.reflection_backend!r}; "
                "choose 'vllm' or 'external_api'."
            )
        generation_elapsed = time.perf_counter() - generation_start
        if efficiency_stats is not None:
            efficiency_stats["skill_generation_time_s"] = (
                float(efficiency_stats.get("skill_generation_time_s", 0.0)) + generation_elapsed
            )
            efficiency_stats["task_skill_generation_time_s"] = (
                float(efficiency_stats.get("task_skill_generation_time_s", 0.0)) + generation_elapsed
            )

        write_success = np.zeros(len(gen_batch.batch), dtype=np.float32)
        reason_counts = {}
        write_start = time.perf_counter()
        for i, reflection_text in enumerate(reflection_texts):
            original_idx = selected_indices[i]
            trajectory = selected_trajectories[i]
            actual_success = selected_successes[i]
            metadata = dict(selected_metadata[i]) if i < len(selected_metadata) and selected_metadata[i] else {}
            if self._reflection_train_metadata_active():
                prompt_texts = selected_reflection_obs.get("text", [])
                metadata.update(
                    self._reflection_training_metadata(
                        prompt=prompt_texts[i] if i < len(prompt_texts) else "",
                        response=reflection_text,
                        pool="task",
                    )
                )
            added, reason = self.dynamic_memory_provider.add_from_reflection_output(
                task_description=selected_tasks[i],
                reflection_output=reflection_text,
                trajectory=trajectory,
                actual_success=actual_success,
                metadata=metadata,
                save=False,
            )
            write_success[original_idx] = 1.0 if added else 0.0
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reflection_texts:
            self.dynamic_memory_provider.save()
        write_elapsed = time.perf_counter() - write_start
        if efficiency_stats is not None:
            efficiency_stats["skill_write_time_s"] = (
                float(efficiency_stats.get("skill_write_time_s", 0.0)) + write_elapsed
            )
            efficiency_stats["task_skill_write_time_s"] = (
                float(efficiency_stats.get("task_skill_write_time_s", 0.0)) + write_elapsed
            )

        stats = {
            "attempted": len(reflection_texts),
            "added": int(write_success.sum()),
            "selected": len(selected_indices),
            "reasons": reason_counts,
            "memory_size": len(self.dynamic_memory_provider.memory),
        }
        self.last_dynamic_memory_stats = stats
        print(f"[UCOB][skill_memory] reflection write stats: {stats}")
        return write_success, stats

    def _rollout_skill_mask(self, batch_size: int) -> np.ndarray:
        if self.rollout_mode == "no_skill_only":
            return np.zeros(batch_size, dtype=bool)
        if self.rollout_mode == "skill_only":
            return np.ones(batch_size, dtype=bool)

        group_n = int(self.config.env.rollout.n)
        if group_n <= 1:
            return np.arange(batch_size) % 2 == 1

        no_skill_per_group = max(1, group_n // 2)
        if group_n % 2 != 0:
            print(f"[UCOB] Warning: env.rollout.n={group_n} is odd; using {no_skill_per_group} no-skill rows per group.")
        return (np.arange(batch_size) % group_n) >= no_skill_per_group

    def preprocess_single_sample(self, item, gen_batch, obs):
        return self._collector.preprocess_single_sample(item, gen_batch, obs)

    def preprocess_batch(self, gen_batch, obs):
        return self._collector.preprocess_batch(gen_batch, obs)

    def gather_rollout_data(self, *args, **kwargs):
        return self._collector.gather_rollout_data(*args, **kwargs)

    def _assign_step_credit(self, total_batch_list, total_episode_rewards):
        if getattr(self._collector, "enable_credit_assignment", False):
            return self._collector.credit_assignment(
                total_batch_list,
                getattr(self._collector, "step_gamma", 0.95),
            )
        return self._collector.no_credit_assignment(total_batch_list, total_episode_rewards)

    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs, rollout_skill_mask=None, is_train: bool = True):
        batch_size = len(gen_batch.batch)
        if self.case_study_debug_enable:
            self.case_study_written_memories = []
        if rollout_skill_mask is None:
            rollout_skill_mask = self._rollout_skill_mask(batch_size)
        else:
            rollout_skill_mask = np.asarray(rollout_skill_mask, dtype=bool)
            assert rollout_skill_mask.shape[0] == batch_size

        env_kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)
        if env_kwargs is None:
            env_kwargs = _extract_search_env_kwargs(gen_batch.non_tensor_batch)
        obs, infos = envs.reset(kwargs=env_kwargs)

        length_obs = len(obs["text"]) if obs["text"] is not None else len(obs["image"])
        assert len(gen_batch.batch) == length_obs, (
            f"gen_batch size {len(gen_batch.batch)} does not match obs size {length_obs}"
        )

        retrieved_memory_by_env = [[] for _ in range(batch_size)]
        bad_retrieved_memory_by_env = [[] for _ in range(batch_size)]
        task_retrieved_memory_by_env = [[] for _ in range(batch_size)]
        bad_task_retrieved_memory_by_env = [[] for _ in range(batch_size)]
        state_contexts = [""] * batch_size
        dynamic_memory_retrieval_stats = (
            self._new_dynamic_memory_retrieval_stats() if self.dynamic_memory_enabled else None
        )
        dynamic_memory_efficiency_stats = (
            self._new_dynamic_memory_efficiency_stats() if self.dynamic_memory_enabled else None
        )
        memory_counterfactual_mode = "no_memory"
        if is_train and self.dynamic_memory_enabled:
            memory_counterfactual_mode = self.dynamic_memory_counterfactual_mode
        if self.dynamic_memory_enabled:
            if not hasattr(envs, "get_task_descriptions"):
                raise ValueError(
                    "Skill memory retrieval is currently implemented only for ALFWorld, WebShop, Sokoban, AppWorld, and Search."
                )
            tasks = envs.get_task_descriptions()
            if len(tasks) != batch_size:
                raise ValueError(f"Expected {batch_size} task descriptions, got {len(tasks)}.")
            if not self.dynamic_memory_state_aware_retrieval:
                retrieved_memory_by_env, bad_retrieved_memory_by_env = self._timed_retrieve_dynamic_memory_for_step(
                    dynamic_memory_efficiency_stats,
                    tasks=tasks,
                    state_contexts=None,
                    is_train=is_train,
                    memory_counterfactual_mode=memory_counterfactual_mode,
                )
            elif self.dynamic_memory_provider.dual_pool_retrieval:
                task_retrieved_memory_by_env, bad_task_retrieved_memory_by_env = self._timed_retrieve_dynamic_memory_for_step(
                    dynamic_memory_efficiency_stats,
                    tasks=tasks,
                    state_contexts=None,
                    is_train=is_train,
                    memory_counterfactual_mode=memory_counterfactual_mode,
                    pool="task",
                )

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            if self.dynamic_memory_enabled and self.dynamic_memory_state_aware_retrieval:
                state_contexts = self._get_state_contexts(obs, batch_size=batch_size)
                if self.dynamic_memory_provider.dual_pool_retrieval:
                    state_retrieved_memory_by_env, bad_state_retrieved_memory_by_env = self._timed_retrieve_dynamic_memory_for_step(
                        dynamic_memory_efficiency_stats,
                        tasks=tasks,
                        state_contexts=state_contexts,
                        is_train=is_train,
                        memory_counterfactual_mode=memory_counterfactual_mode,
                        pool="state",
                    )
                    retrieved_memory_by_env = [
                        list(task_items) + list(state_items)
                        for task_items, state_items in zip(task_retrieved_memory_by_env, state_retrieved_memory_by_env)
                    ]
                    bad_retrieved_memory_by_env = [
                        list(task_items) + list(state_items)
                        for task_items, state_items in zip(
                            bad_task_retrieved_memory_by_env,
                            bad_state_retrieved_memory_by_env,
                        )
                    ]
                else:
                    retrieved_memory_by_env, bad_retrieved_memory_by_env = self._timed_retrieve_dynamic_memory_for_step(
                        dynamic_memory_efficiency_stats,
                        tasks=tasks,
                        state_contexts=state_contexts,
                        is_train=is_train,
                        memory_counterfactual_mode=memory_counterfactual_mode,
                    )
            if self.dynamic_memory_enabled:
                self._update_dynamic_memory_retrieval_stats(
                    stats=dynamic_memory_retrieval_stats,
                    retrieved_memory_by_env=retrieved_memory_by_env,
                    rollout_skill_mask=rollout_skill_mask,
                    active_masks=active_masks,
                )

            plain_batch = self._collector.preprocess_batch(gen_batch=gen_batch, obs=obs)
            good_memory_obs = self._build_context_obs(obs, retrieved_memory_by_env)
            if self.dynamic_memory_enabled:
                self._update_dynamic_memory_prompt_token_stats(
                    efficiency_stats=dynamic_memory_efficiency_stats,
                    base_obs=obs,
                    skill_obs=good_memory_obs,
                    retrieved_memory_by_env=retrieved_memory_by_env,
                    rollout_skill_mask=rollout_skill_mask,
                    active_masks=active_masks,
                )
            good_memory_batch = self._collector.preprocess_batch(
                gen_batch=gen_batch,
                obs=good_memory_obs,
            )
            if memory_counterfactual_mode == "bad_memory":
                bad_memory_batch = self._collector.preprocess_batch(
                    gen_batch=gen_batch,
                    obs=self._build_context_obs(obs, bad_retrieved_memory_by_env),
                )
            else:
                bad_memory_batch = plain_batch

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in plain_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in plain_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in plain_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")

            plain_batch_input = plain_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            good_memory_batch_input = good_memory_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=[
                    key for key in non_tensor_batch_keys_to_pop if key in good_memory_batch.non_tensor_batch
                ],
            )
            if memory_counterfactual_mode == "bad_memory":
                bad_memory_batch_input = bad_memory_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=[
                        key for key in non_tensor_batch_keys_to_pop if key in bad_memory_batch.non_tensor_batch
                    ],
                )
            else:
                bad_memory_batch_input = plain_batch_input
            plain_batch_input.meta_info = gen_batch.meta_info
            good_memory_batch_input.meta_info = gen_batch.meta_info
            bad_memory_batch_input.meta_info = gen_batch.meta_info

            student_batch = self._mix_dataproto_by_mask(bad_memory_batch, good_memory_batch, rollout_skill_mask)
            student_batch_input = self._mix_dataproto_by_mask(
                bad_memory_batch_input,
                good_memory_batch_input,
                rollout_skill_mask,
            )
            teacher_batch_input = self._mix_dataproto_by_mask(
                good_memory_batch_input,
                bad_memory_batch_input,
                rollout_skill_mask,
            )

            student_batch_input_padded, pad_size = pad_dataproto_to_divisor(
                student_batch_input,
                actor_rollout_wg.world_size,
            )
            policy_output_padded = actor_rollout_wg.generate_sequences(student_batch_input_padded)
            policy_output = unpad_dataproto(policy_output_padded, pad_size=pad_size)

            student_output = self._build_output_with_prompt(student_batch_input, policy_output)
            response_length = student_output.batch["responses"].size(1)
            if self.cbsd_enable:
                alt_prompt_input = self._mix_dataproto_by_mask(
                    good_memory_batch_input,
                    plain_batch_input,
                    rollout_skill_mask,
                )
                student_output.batch["ucob_cbsd_alt_prompt_ids"] = alt_prompt_input.batch["input_ids"]
                student_output.batch["ucob_cbsd_alt_prompt_attention_mask"] = alt_prompt_input.batch["attention_mask"]
                student_output.non_tensor_batch["ucob_cbsd_alt_branch"] = np.array(
                    ["none" if use_skill else "good" for use_skill in rollout_skill_mask],
                    dtype=object,
                )
            gate_sign = np.where(rollout_skill_mask, -1.0, 1.0).astype(np.float32)
            if self.teacher_loss_enable:
                student_output.batch["dsdar_teacher_prompts"] = teacher_batch_input.batch["input_ids"]
                student_output.batch["dsdar_teacher_prompt_attention_mask"] = teacher_batch_input.batch["attention_mask"]
                student_output.batch["sdar_gate_sign"] = torch.as_tensor(gate_sign).unsqueeze(-1).expand(-1, response_length)

            student_batch.non_tensor_batch["uid"] = uid_batch
            student_batch.non_tensor_batch["traj_uid"] = traj_uid
            student_batch.non_tensor_batch["dsdar_rollout_mode"] = np.array(
                ["skill" if use_skill else "no_skill" for use_skill in rollout_skill_mask],
                dtype=object,
            )
            if self.dynamic_memory_enabled:
                student_batch.non_tensor_batch["dynamic_memory_used"] = rollout_skill_mask.astype(bool)
                student_batch.non_tensor_batch["dynamic_memory_good_used"] = rollout_skill_mask.astype(bool)
                student_batch.non_tensor_batch["dynamic_memory_bad_used"] = np.logical_and(
                    np.logical_not(rollout_skill_mask),
                    memory_counterfactual_mode == "bad_memory",
                )
                student_batch.non_tensor_batch["dynamic_memory_branch"] = np.array(
                    [
                        "good" if use_skill else ("bad" if memory_counterfactual_mode == "bad_memory" else "none")
                        for use_skill in rollout_skill_mask
                    ],
                    dtype=object,
                )
                student_batch.non_tensor_batch["dynamic_memory_retrieved_count"] = np.array(
                    [len(items) for items in retrieved_memory_by_env],
                    dtype=np.float32,
                )
                student_batch.non_tensor_batch["dynamic_memory_ids"] = np.array(
                    [
                        self._retrieved_memory_ids(retrieved_memory_by_env[i]) if rollout_skill_mask[i] else []
                        for i in range(batch_size)
                    ],
                    dtype=object,
                )
                student_batch.non_tensor_batch["dynamic_memory_task_ids"] = np.array(
                    [
                        self._retrieved_memory_ids_by_pool(retrieved_memory_by_env[i], "task")
                        if rollout_skill_mask[i]
                        else []
                        for i in range(batch_size)
                    ],
                    dtype=object,
                )
                student_batch.non_tensor_batch["dynamic_memory_state_ids"] = np.array(
                    [
                        self._retrieved_memory_ids_by_pool(retrieved_memory_by_env[i], "state")
                        if rollout_skill_mask[i]
                        else []
                        for i in range(batch_size)
                    ],
                    dtype=object,
                )
                if self.case_study_debug_enable:
                    student_batch.non_tensor_batch["dynamic_memory_retrieved_debug"] = np.array(
                        [
                            self._case_study_retrieved_debug_items(retrieved_memory_by_env[i])
                            if rollout_skill_mask[i]
                            else []
                            for i in range(batch_size)
                        ],
                        dtype=object,
                    )
                student_batch.non_tensor_batch["dynamic_memory_task_retrieved_count"] = np.array(
                    [
                        len(self._retrieved_memory_ids_by_pool(retrieved_memory_by_env[i], "task"))
                        for i in range(batch_size)
                    ],
                    dtype=np.float32,
                )
                student_batch.non_tensor_batch["dynamic_memory_state_retrieved_count"] = np.array(
                    [
                        len(self._retrieved_memory_ids_by_pool(retrieved_memory_by_env[i], "state"))
                        for i in range(batch_size)
                    ],
                    dtype=np.float32,
                )
                student_batch.non_tensor_batch["dynamic_memory_state_context"] = np.array(
                    state_contexts,
                    dtype=object,
                )
                student_batch.non_tensor_batch["dynamic_memory_bad_retrieved_count"] = np.array(
                    [len(items) for items in bad_retrieved_memory_by_env],
                    dtype=np.float32,
                )
            student_output.non_tensor_batch.update(student_batch.non_tensor_batch)
            batch = student_output

            text_actions = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            self._attach_action_loss_mask(batch, text_actions)
            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if "is_action_valid" in infos[0]:
                batch.non_tensor_batch["is_action_valid"] = np.array(
                    [info["is_action_valid"] for info in infos],
                    dtype=bool,
                )
            else:
                batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)

            if "tool_calling" in infos[0]:
                tool_callings[active_masks] += np.array(
                    [info["tool_calling"] for info in infos],
                    dtype=np.float32,
                )[active_masks]

            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, (
                f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            )
            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

            batch_list = to_list_of_dict(batch)
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

            if is_done.all():
                break

        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        success["skill_rollout_mask"] = rollout_skill_mask.astype(np.float32)

        if self.dynamic_memory_enabled and is_train:
            terminal_infos = self._get_terminal_infos(total_infos=total_infos, total_batch_list=total_batch_list)
            terminal_successes = self._get_terminal_successes(envs, terminal_infos, episode_rewards)
            memory_step_returns = self._compute_memory_step_returns(total_batch_list)
            utility_update_mode = getattr(self.dynamic_memory_provider, "utility_update", "")
            step_utility_stats = {"groups": 0, "updates": 0, "task_updates": 0, "state_updates": 0}
            if utility_update_mode == "step_relative":
                step_utility_stats = self._update_step_relative_memory_utility(
                    total_batch_list=total_batch_list,
                    step_returns_by_traj=memory_step_returns,
                )
            elif utility_update_mode in {"baseline_relative", "d2skill", "d2skill_relative"}:
                step_utility_stats = self._update_baseline_relative_memory_utility(
                    total_batch_list=total_batch_list,
                    terminal_successes=terminal_successes,
                    rollout_skill_mask=rollout_skill_mask,
                )
            elif utility_update_mode == "relative_outcome":
                step_utility_stats = self._dynamic_memory_relative_outcome_update_stats(
                    retrieved_by_env=retrieved_memory_by_env,
                    rollout_skill_mask=rollout_skill_mask,
                )
                updated = self.dynamic_memory_provider.update_retrieved_utilities_relative(
                    retrieved_by_env=retrieved_memory_by_env,
                    successes=terminal_successes,
                    skill_mask=rollout_skill_mask,
                    group_n=int(self.config.env.rollout.n),
                )
                step_utility_stats["updates"] = int(updated)
                if int(updated) <= 0:
                    step_utility_stats["task_updates"] = 0
                    step_utility_stats["state_updates"] = 0
            else:
                utility_updates = []
                for i, (retrieved, use_memory) in enumerate(zip(retrieved_memory_by_env, rollout_skill_mask)):
                    if use_memory and retrieved:
                        utility_updates.append((retrieved, terminal_successes[i]))
                step_utility_stats = self._dynamic_memory_retrieved_update_stats(
                    retrieved_by_env=retrieved_memory_by_env,
                    rollout_skill_mask=rollout_skill_mask,
                )
                if utility_updates:
                    updated = self.dynamic_memory_provider.update_retrieved_utilities(utility_updates)
                    step_utility_stats["updates"] = int(updated)
                    if int(updated) <= 0:
                        step_utility_stats["task_updates"] = 0
                        step_utility_stats["state_updates"] = 0

            memory_write_success = np.zeros(batch_size, dtype=np.float32)
            step_memory_write_success = np.zeros(batch_size, dtype=np.float32)
            task_reflection_stats = {"attempted": 0, "added": 0, "reasons": {}}
            step_reflection_stats = {"attempted": 0, "added": 0, "reasons": {}}
            case_study_memory_ids_before_write = (
                self._case_study_memory_id_set() if self.case_study_debug_enable else set()
            )
            if is_train:
                memory_write_success, task_reflection_stats = self._generate_dynamic_memory_reflections(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                    terminal_infos=terminal_infos,
                    episode_lengths=episode_lengths,
                    rollout_skill_mask=rollout_skill_mask,
                    efficiency_stats=dynamic_memory_efficiency_stats,
                )
                step_memory_write_success, step_reflection_stats = self._generate_dynamic_memory_step_reflections(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    tasks=tasks,
                    total_batch_list=total_batch_list,
                    step_returns_by_traj=memory_step_returns,
                    terminal_successes=terminal_successes,
                    rollout_skill_mask=rollout_skill_mask,
                    efficiency_stats=dynamic_memory_efficiency_stats,
                )
            if self.case_study_debug_enable:
                self.case_study_written_memories = self._case_study_capture_new_memories(
                    case_study_memory_ids_before_write
                )
            success["dynamic_memory_write_success_rate"] = memory_write_success
            success["dynamic_memory_step_write_success_rate"] = step_memory_write_success
            success["dynamic_memory_retrieved_count"] = np.array(
                [len(items) for items in retrieved_memory_by_env],
                dtype=np.float32,
            )
            success["dynamic_memory_used_rate"] = rollout_skill_mask.astype(np.float32)
            success["dynamic_memory_good_used_rate"] = rollout_skill_mask.astype(np.float32)
            success["dynamic_memory_bad_used_rate"] = np.logical_and(
                np.logical_not(rollout_skill_mask),
                memory_counterfactual_mode == "bad_memory",
            ).astype(np.float32)
            success["dynamic_memory_bad_retrieved_count"] = np.array(
                [len(items) for items in bad_retrieved_memory_by_env],
                dtype=np.float32,
            )
            success["dynamic_memory_step_utility_groups"] = np.array(
                [step_utility_stats.get("groups", 0)] * batch_size,
                dtype=np.float32,
            )
            success["dynamic_memory_step_utility_updates"] = np.array(
                [step_utility_stats.get("updates", 0)] * batch_size,
                dtype=np.float32,
            )
            success["dynamic_memory_task_utility_updates"] = np.array(
                [step_utility_stats.get("task_updates", 0)] * batch_size,
                dtype=np.float32,
            )
            success["dynamic_memory_state_utility_updates"] = np.array(
                [step_utility_stats.get("state_updates", 0)] * batch_size,
                dtype=np.float32,
            )
            success["dynamic_memory_step_reflection_attempted"] = np.array(
                [step_reflection_stats.get("attempted", 0)] * batch_size,
                dtype=np.float32,
            )
            success["dynamic_memory_step_reflection_added"] = np.array(
                [step_reflection_stats.get("added", 0)] * batch_size,
                dtype=np.float32,
            )
            dynamic_memory_analysis_stats = {}
            dynamic_memory_analysis_stats.update(self._dynamic_memory_bank_size_stats())
            dynamic_memory_analysis_stats.update(self._dynamic_memory_bank_utility_stats())
            dynamic_memory_analysis_stats.update(
                self._finalize_dynamic_memory_retrieval_stats(dynamic_memory_retrieval_stats or {})
            )
            dynamic_memory_analysis_stats.update(
                self._finalize_dynamic_memory_efficiency_stats(dynamic_memory_efficiency_stats or {})
            )
            dynamic_memory_analysis_stats.update(
                self._dynamic_memory_write_stats(task_reflection_stats, step_reflection_stats)
            )
            dynamic_memory_analysis_stats.update(
                {
                    "dynamic_memory_utility_update_groups": float(step_utility_stats.get("groups", 0)),
                    "dynamic_memory_utility_updates_total": float(step_utility_stats.get("updates", 0)),
                    "dynamic_memory_task_utility_updates": float(step_utility_stats.get("task_updates", 0)),
                    "dynamic_memory_state_utility_updates": float(step_utility_stats.get("state_updates", 0)),
                }
            )
            self.last_dynamic_memory_stats = dynamic_memory_analysis_stats
            self._attach_dynamic_memory_scalar_success(
                success=success,
                stats=dynamic_memory_analysis_stats,
                batch_size=batch_size,
            )
        elif self.dynamic_memory_enabled:
            dynamic_memory_analysis_stats = {}
            dynamic_memory_analysis_stats.update(self._dynamic_memory_bank_size_stats())
            dynamic_memory_analysis_stats.update(self._dynamic_memory_bank_utility_stats())
            dynamic_memory_analysis_stats.update(
                self._finalize_dynamic_memory_retrieval_stats(dynamic_memory_retrieval_stats or {})
            )
            dynamic_memory_analysis_stats.update(
                self._finalize_dynamic_memory_efficiency_stats(dynamic_memory_efficiency_stats or {})
            )
            self.last_dynamic_memory_stats = dynamic_memory_analysis_stats
            self.dynamic_memory_eval_stats_history.append(dynamic_memory_analysis_stats)
            self._attach_dynamic_memory_scalar_success(
                success=success,
                stats=dynamic_memory_analysis_stats,
                batch_size=batch_size,
            )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def dynamic_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs):
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while (
            len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n
            and try_count < max_try_count
        ):
            if len(total_batch_list) > 0:
                print(
                    f"valid num={len(total_batch_list)} < target num="
                    f"{self.config.data.train_batch_size * self.config.env.rollout.n}. "
                    f"Keep generating... ({try_count}/{max_try_count})"
                )
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=True,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                config=self.config,
                last_try=(try_count == max_try_count),
            )

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        dynamic_memory_time_keys = [
            "dynamic_memory_retrieval_time_s",
            "dynamic_memory_task_retrieval_time_s",
            "dynamic_memory_state_retrieval_time_s",
            "dynamic_memory_skill_generation_time_s",
            "dynamic_memory_task_skill_generation_time_s",
            "dynamic_memory_state_skill_generation_time_s",
            "dynamic_memory_skill_write_time_s",
            "dynamic_memory_task_skill_write_time_s",
            "dynamic_memory_state_skill_write_time_s",
        ]
        dynamic_memory_time_sums = {
            key: sum(
                float(success[key][0])
                for success in total_success
                if key in success and len(success[key]) > 0
            )
            for key in dynamic_memory_time_keys
        }
        total_success = {
            key: np.concatenate([success[key] for success in total_success], axis=0)
            for key in total_success[0].keys()
        }
        for key, value in dynamic_memory_time_sums.items():
            if key in total_success:
                total_success[key] = np.array([float(value)] * len(total_batch_list), dtype=np.float32)
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return (
            total_batch_list,
            total_episode_rewards,
            total_episode_lengths,
            total_success,
            total_traj_uid,
            total_tool_callings,
        )

    def multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs, is_train: bool = True):
        if not is_train:
            if self.eval_inject_skill:
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = (
                    self.vanilla_multi_turn_loop(
                        gen_batch=gen_batch,
                        actor_rollout_wg=actor_rollout_wg,
                        envs=envs,
                        rollout_skill_mask=np.ones(len(gen_batch.batch), dtype=bool),
                        is_train=False,
                    )
                )
                per_step_episode_rewards, total_discounted_returns = self._assign_step_credit(
                    total_batch_list,
                    total_episode_rewards,
                )
                self._strip_cbsd_record_keys(total_batch_list)
                return self._collector.gather_rollout_data(
                    total_batch_list=total_batch_list,
                    episode_rewards=per_step_episode_rewards,
                    episode_lengths=total_episode_lengths,
                    success=total_success,
                    traj_uid=total_traj_uid,
                    tool_callings=total_tool_callings,
                    discounted_returns=total_discounted_returns,
                )
            return self._collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=False,
            )

        gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        if self.config.algorithm.filter_groups.enable:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = (
                self.dynamic_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            )
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = (
                self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            )

        total_success = self._attach_rollout_branch_success_rates(total_success)

        per_step_episode_rewards, total_discounted_returns = self._assign_step_credit(
            total_batch_list,
            total_episode_rewards,
        )

        cbsd_aux_batch = self._build_cbsd_auxiliary_batch(
            total_batch_list=total_batch_list,
            step_returns_by_traj=total_discounted_returns,
        )
        reflection_train_batch = self._build_dynamic_memory_reflection_train_batch(total_batch_list)
        self._strip_cbsd_record_keys(total_batch_list)

        rollout_batch = self._collector.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=per_step_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
            discounted_returns=total_discounted_returns,
        )
        cbsd_analysis_stats = getattr(self, "last_cbsd_analysis_stats", None)
        if isinstance(cbsd_analysis_stats, dict):
            rollout_batch.meta_info.update(cbsd_analysis_stats)
        if cbsd_aux_batch is not None and len(cbsd_aux_batch) > 0:
            rollout_batch = self._concat_aligned_batches(rollout_batch, cbsd_aux_batch)
        if reflection_train_batch is not None and len(reflection_train_batch) > 0:
            # Reflection self-training uses a longer response length than acting rollouts.
            # Keep it out of the rollout tensor batch to avoid padding every action update.
            rollout_batch.meta_info["dynamic_memory_reflection_train_batch"] = reflection_train_batch
        return rollout_batch


if __name__ == "__main__":
    main()
