import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.multi_turn_rollout.utils import filter_group_data, to_list_of_dict, torch_to_numpy


def _extract_env_kwargs(non_tensor_batch):
    env_kwargs = non_tensor_batch.get("env_kwargs")
    if env_kwargs is None:
        return None
    if isinstance(env_kwargs, np.ndarray):
        return env_kwargs
    return np.asarray(env_kwargs, dtype=object)


class Skill1TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.step_gamma = config.algorithm.get("step_gamma", 0.95)
        self.traj_gamma = config.algorithm.get("traj_gamma", 0.6)
        self.intrinsic_reward_coefficient = config.algorithm.get("intrinsic_reward_coefficient", -1)
        self.enable_credit_assignment = config.algorithm.get("credit_assignment", False)
        mem_cfg = config.env.get("skill_library", {})
        self.selection_trainable = mem_cfg.get("selection_trainable", True)
        self.lambda_rerank = mem_cfg.get("lambda_rerank", 1.0)
        self.lambda_distill = mem_cfg.get("lambda_distill", 1.0)
        self.use_ref_policy_for_distill = config.algorithm.get("distill_reference_policy", False)

    def preprocess_single_sample(self, item: int, gen_batch: DataProto, obs: Dict[str, Any]):
        raw_prompt = gen_batch.non_tensor_batch["raw_prompt"][item]
        data_source = gen_batch.non_tensor_batch.get("data_source", ["unknown"] * len(gen_batch))[item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        obs_texts = obs.get("text", None)
        obs_images = obs.get("image", None)
        obs_anchors = obs.get("anchor", None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None
        obs_content = obs_text if obs_text is not None else ""
        chat = np.array([{"content": obs_content, "role": "user"}])
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )

        row_dict = {}
        if is_multi_modal:
            from agent_system.multi_turn_rollout.utils import process_image

            raw_prompt = prompt_with_chat_template.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["multi_modal_data"] = {"image": [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict["multi_modal_data"]["image"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict["multi_modal_inputs"] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>" + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length) + "<|vision_end|>",
                        1,
                    )
                    index += 1
                prompt_with_chat_template = prompt_with_chat_template.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        if is_multi_modal:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}."
                )

        row_dict.update(
            {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "anchor_obs": torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor,
                "index": item,
                "data_source": data_source,
            }
        )
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()
        return row_dict

    def preprocess_batch(self, gen_batch: DataProto, obs: Dict[str, Any]) -> DataProto:
        processed = [self.preprocess_single_sample(i, gen_batch, obs) for i in range(len(gen_batch.batch["input_ids"]))]
        batch = collate_fn(processed)
        return DataProto.from_single_dict(data=batch, meta_info=gen_batch.meta_info)

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        discounted_returns: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        success_rate = {key: np.mean(value) for key, value in success.items()}
        effective_batch = []
        for bs in range(len(total_batch_list)):
            for step_idx, data in enumerate(total_batch_list[bs]):
                if data["active_masks"]:
                    data["episode_rewards"] = np.array(episode_rewards[bs][step_idx], dtype=np.float32)
                    data["step_returns"] = torch.tensor(float(discounted_returns[bs][step_idx]), dtype=torch.float32)
                    data["episode_lengths"] = episode_lengths[bs]
                    data["tool_callings"] = tool_callings[bs]
                    data["turn_step"] = step_idx
                    for key, value in success_rate.items():
                        data[key] = value
                    effective_batch.append(data)
        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def no_credit_assignment(self, total_batch_list, final_episode_rewards):
        num_trajectories = len(total_batch_list)
        total_step_rewards = np.empty(num_trajectories, dtype=object)
        total_discounted_returns = np.empty(num_trajectories, dtype=object)
        for traj_idx, steps in enumerate(total_batch_list):
            n = len(steps)
            episode_rewards = np.zeros(n, dtype=np.float64)
            discounted_returns = np.zeros(n, dtype=np.float64)
            final_reward = float(final_episode_rewards[traj_idx]) if len(final_episode_rewards) > traj_idx else 0.0
            for t, step in enumerate(steps):
                if step.get("active_masks", True):
                    episode_rewards[t] = final_reward
                    discounted_returns[t] = final_reward
            total_step_rewards[traj_idx] = episode_rewards
            total_discounted_returns[traj_idx] = discounted_returns
        return total_step_rewards, total_discounted_returns

    def credit_assignment(self, total_batch_list, step_gamma=0.95):
        num_trajectories = len(total_batch_list)
        total_episode_rewards = np.empty(num_trajectories, dtype=object)
        total_discounted_returns = np.empty(num_trajectories, dtype=object)
        for traj_idx, steps in enumerate(total_batch_list):
            n = len(steps)
            episode_rewards = np.zeros(n, dtype=np.float64)
            discounted_returns = np.zeros(n, dtype=np.float64)
            cumulative_reward = 0.0
            for step in steps:
                if step.get("active_masks", True):
                    cumulative_reward += float(step.get("rewards", 0.0))
            for t, step in enumerate(steps):
                if step.get("active_masks", True):
                    episode_rewards[t] = cumulative_reward
            running_return = 0.0
            for t in reversed(range(n)):
                step = steps[t]
                if not step.get("active_masks", True):
                    continue
                reward_val = float(step.get("rewards", 0.0))
                running_return = reward_val + step_gamma * running_return
                discounted_returns[t] = running_return
            total_episode_rewards[traj_idx] = episode_rewards
            total_discounted_returns[traj_idx] = discounted_returns
        return total_episode_rewards, total_discounted_returns

    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs: EnvironmentManagerBase):
        batch_size = len(gen_batch.batch)
        env_kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)
        if env_kwargs is None:
            env_kwargs = _extract_env_kwargs(gen_batch.non_tensor_batch)
        obs, infos = envs.reset(kwargs=env_kwargs)
        assert len(gen_batch.batch) == (len(obs["text"]) if obs["text"] is not None else len(obs["image"]))

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(os.urandom(8).hex())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(os.urandom(8).hex())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(os.urandom(8).hex()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
            batch_input.meta_info = gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)
            text_actions = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            next_obs, rewards, dones, infos = envs.step(text_actions)
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)
            batch.non_tensor_batch["is_action_valid"] = np.array([info.get("is_action_valid", True) for info in infos], dtype=bool)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1
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
        success = envs.success_evaluator(total_infos=total_infos, total_batch_list=total_batch_list, episode_rewards=episode_rewards, episode_lengths=episode_lengths)
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs: EnvironmentManagerBase, is_train: bool = True):
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        if self.config.algorithm.filter_groups.enable and is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = filter_group_data(
                batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=total_tool_callings,
                config=self.config,
                last_try=True,
            )
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
        if self.enable_credit_assignment:
            per_step_episode_rewards, total_discounted_returns = self.credit_assignment(total_batch_list, self.step_gamma)
        else:
            per_step_episode_rewards, total_discounted_returns = self.no_credit_assignment(total_batch_list, total_episode_rewards)
        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=per_step_episode_rewards,
            discounted_returns=total_discounted_returns,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
        )
