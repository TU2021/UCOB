# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy, _as_bool
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


def _require_thinking_from_config(config) -> bool:
    return _as_bool(config.env.get("require_thinking", False))


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    thinking_instruction=self.thinking_instruction("search"),
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                    thinking_instruction=self.thinking_instruction("search"),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def get_task_descriptions(self) -> List[str]:
        return list(self.tasks)

    def get_terminal_successes(self, infos: List[Dict]) -> List[bool]:
        return [bool(info.get("won", False)) for info in infos]

    @staticmethod
    def _truncate_search_reflection_text(text: str, max_chars: int = 2000) -> str:
        text = str(text or "").strip()
        if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
            return text
        keep_left = max_chars // 2
        keep_right = max_chars - keep_left
        return text[:keep_left].rstrip() + "\n...[truncated]...\n" + text[-keep_right:].lstrip()

    def get_reflection_trajectories(
        self,
        max_steps: int = None,
        max_to_show: int = None,
        episode_lengths: List[float] = None,
    ) -> List[str]:
        trajectories = []
        result_max_chars = 2200
        for env_idx in range(self.memory.batch_size):
            records = list(self.memory[env_idx])
            if episode_lengths is not None:
                records = records[: max(int(episode_lengths[env_idx]), 0)]
            effective_total_len = len(records)
            if max_steps is not None and max_steps > 0:
                records = records[-max_steps:]
                start_idx = max(0, effective_total_len - len(records))
            else:
                start_idx = 0

            lines = []
            for j, rec in enumerate(records):
                step_num = start_idx + j + 1
                action = self._truncate_search_reflection_text(rec.get("search", ""), max_chars=1200)
                information = self._truncate_search_reflection_text(
                    rec.get("information", ""),
                    max_chars=result_max_chars,
                )
                if information:
                    lines.append(
                        f"[Step {step_num}: Agent output: '{action}', Search result: '{information}']"
                    )
                else:
                    lines.append(f"[Step {step_num}: Agent output: '{action}', Search result: '']")
            trajectories.append("\n".join(lines) if lines else "No search steps were recorded.")
        return trajectories

    def build_reflection_observations(
        self,
        infos: List[Dict],
        dynamic_memory_provider,
        episode_lengths: List[float] = None,
    ) -> Dict[str, Any]:
        max_steps = getattr(dynamic_memory_provider, "max_trajectory_steps", self.config.env.max_steps)
        max_to_show = getattr(dynamic_memory_provider, "max_reflection_items", 6)
        trajectories = self.get_reflection_trajectories(
            max_steps=max_steps,
            max_to_show=max_to_show,
            episode_lengths=episode_lengths,
        )
        successes = self.get_terminal_successes(infos)
        postprocess_text_obs = []
        task_categories = []
        use_skill_reflection = getattr(dynamic_memory_provider, "skill_format", "") == "claude_style"

        for i, task in enumerate(self.tasks):
            is_won = successes[i]
            success_text = "successfully" if is_won else "unsuccessfully"
            task_category = dynamic_memory_provider.infer_task_category(task)
            task_categories.append(task_category)
            if is_won:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="failure")
                reference_trajectory = (
                    "Reference Failed Trajectory (for comparison):\n" + reference
                    if reference
                    else "No failed attempts available for comparison."
                )
            else:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="success")
                reference_trajectory = (
                    "Reference Successful Trajectory (for comparison):\n" + reference
                    if reference
                    else "No successful attempts available for reference."
                )

            reflect_template = SEARCH_SKILL_REFLECT_TEMPLATE if use_skill_reflection else SEARCH_REFLECT_TEMPLATE
            format_kwargs = dict(
                task_description=task,
                success=success_text,
                reference_trajectory=reference_trajectory,
                current_trajectory=trajectories[i],
            )
            if use_skill_reflection:
                format_kwargs["task_category"] = task_category
            obs = reflect_template.format(**format_kwargs)
            postprocess_text_obs.append(obs)

        return {
            "text": postprocess_text_obs,
            "image": None,
            "anchor": postprocess_text_obs.copy(),
            "tasks": list(self.tasks),
            "task_categories": task_categories,
            "trajectories": trajectories,
            "successes": successes,
        }


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def get_task_descriptions(self) -> List[str]:
        return list(self.tasks)

    def get_terminal_successes(self, infos: List[Dict]) -> List[bool]:
        return [bool(info.get("won", False)) for info in infos]

    def get_reflection_trajectories(
        self,
        max_steps: int = None,
        max_to_show: int = None,
        episode_lengths: List[float] = None,
    ) -> List[str]:
        trajectories = []
        for env_idx in range(self.memory.batch_size):
            records = list(self.memory[env_idx])
            if episode_lengths is not None:
                records = records[: max(int(episode_lengths[env_idx]), 0)]
            effective_total_len = len(records)
            if max_steps is not None and max_steps > 0:
                records = records[-max_steps:]
                start_idx = max(0, effective_total_len - len(records))
            else:
                start_idx = 0

            lines = []
            for j, rec in enumerate(records):
                step_num = start_idx + j + 1
                obs = rec.get("text_obs", "")
                act = rec.get("action", "")
                lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']")
            trajectories.append("\n".join(lines) if lines else "No actions were recorded.")
        return trajectories

    def build_reflection_observations(
        self,
        infos: List[Dict],
        dynamic_memory_provider,
        episode_lengths: List[float] = None,
    ) -> Dict[str, Any]:
        max_steps = getattr(dynamic_memory_provider, "max_trajectory_steps", self.config.env.max_steps)
        trajectories = self.get_reflection_trajectories(max_steps=max_steps, episode_lengths=episode_lengths)
        successes = self.get_terminal_successes(infos)
        postprocess_text_obs = []
        task_categories = []
        use_skill_reflection = getattr(dynamic_memory_provider, "skill_format", "") == "claude_style"

        for i, task in enumerate(self.tasks):
            is_won = successes[i]
            success_text = "successfully" if is_won else "unsuccessfully"
            task_category = dynamic_memory_provider.infer_task_category(task)
            task_categories.append(task_category)
            if is_won:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="failure")
                reference_trajectory = (
                    "Reference Failed Trajectory (for comparison):\n" + reference
                    if reference
                    else "No failed attempts available for comparison."
                )
            else:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="success")
                reference_trajectory = (
                    "Reference Successful Trajectory (for comparison):\n" + reference
                    if reference
                    else "No successful attempts available for reference."
                )

            reflect_template = ALFWORLD_SKILL_REFLECT_TEMPLATE if use_skill_reflection else ALFWORLD_REFLECT_TEMPLATE
            format_kwargs = dict(
                task_description=task,
                success=success_text,
                reference_trajectory=reference_trajectory,
                current_trajectory=trajectories[i],
            )
            if use_skill_reflection:
                format_kwargs["task_category"] = task_category
            obs = reflect_template.format(**format_kwargs)
            postprocess_text_obs.append(obs)

        return {
            "text": postprocess_text_obs,
            "image": None,
            "anchor": postprocess_text_obs.copy(),
            "tasks": list(self.tasks),
            "task_categories": task_categories,
            "trajectories": trajectories,
            "successes": successes,
        }
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    thinking_instruction=self.thinking_instruction("alfworld"),
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    thinking_instruction=self.thinking_instruction("alfworld"),
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            initial_task_obs = self.pre_text_obs
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            initial_task_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.tasks = [self._build_task_description(board) for board in initial_task_obs]
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = (
                    SOKOBAN_VISUAL_TEMPLATE.format(thinking_instruction=self.thinking_instruction("sokoban"))
                    if self.is_multi_modal
                    else SOKOBAN_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        thinking_instruction=self.thinking_instruction("sokoban"),
                    )
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE.format(thinking_instruction=self.thinking_instruction("sokoban"))
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        thinking_instruction=self.thinking_instruction("sokoban"),
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    @staticmethod
    def _format_sokoban_board(board) -> str:
        if isinstance(board, np.ndarray):
            return np.array2string(board, threshold=10000)
        if isinstance(board, (list, tuple)):
            return "\n".join(str(row) for row in board)
        return str(board)

    def _build_task_description(self, initial_board) -> str:
        dim_room = getattr(self.config.env.sokoban, "dim_room", None)
        if dim_room is None:
            dim_text = "unknown size"
        else:
            dim_values = list(dim_room)
            dim_text = "x".join(str(value) for value in dim_values)
        num_boxes = getattr(self.config.env.sokoban, "num_boxes", "unknown")
        board_text = self._format_sokoban_board(initial_board)
        return (
            f"Sokoban level ({dim_text}, {num_boxes} box(es)).\n"
            "Goal: push every box onto a target without creating deadlocks.\n"
            f"Initial board:\n{board_text}"
        )

    def get_task_descriptions(self) -> List[str]:
        return list(getattr(self, "tasks", []))

    def get_terminal_successes(self, infos: List[Dict]) -> List[bool]:
        return [bool(info.get("won", False)) for info in infos]

    def get_reflection_trajectories(
        self,
        max_steps: int = None,
        max_to_show: int = None,
        episode_lengths: List[float] = None,
    ) -> List[str]:
        trajectories = []
        for env_idx in range(self.memory.batch_size):
            records = list(self.memory[env_idx])
            if episode_lengths is not None:
                records = records[: max(int(episode_lengths[env_idx]), 0)]
            effective_total_len = len(records)
            if max_steps is not None and max_steps > 0:
                records = records[-max_steps:]
                start_idx = max(0, effective_total_len - len(records))
            else:
                start_idx = 0

            lines = []
            for j, rec in enumerate(records):
                step_num = start_idx + j + 1
                obs = self._format_sokoban_board(rec.get("text_obs", ""))
                act = rec.get("action", "")
                lines.append(f"[Observation {step_num}:\n{obs}\nAction {step_num}: {act}]")
            trajectories.append("\n".join(lines) if lines else "No actions were recorded.")
        return trajectories

    def build_reflection_observations(
        self,
        infos: List[Dict],
        dynamic_memory_provider,
        episode_lengths: List[float] = None,
    ) -> Dict[str, Any]:
        max_steps = getattr(dynamic_memory_provider, "max_trajectory_steps", self.config.env.max_steps)
        trajectories = self.get_reflection_trajectories(
            max_steps=max_steps,
            episode_lengths=episode_lengths,
        )
        successes = self.get_terminal_successes(infos)
        postprocess_text_obs = []
        task_categories = []
        use_skill_reflection = getattr(dynamic_memory_provider, "skill_format", "") == "claude_style"

        for i, task in enumerate(self.tasks):
            is_won = successes[i]
            success_text = "successfully" if is_won else "unsuccessfully"
            task_category = dynamic_memory_provider.infer_task_category(task)
            task_categories.append(task_category)
            if is_won:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="failure")
                reference_trajectory = (
                    "Reference Failed Trajectory (for comparison):\n" + reference
                    if reference
                    else "No failed attempts available for comparison."
                )
            else:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="success")
                reference_trajectory = (
                    "Reference Successful Trajectory (for comparison):\n" + reference
                    if reference
                    else "No successful attempts available for reference."
                )

            reflect_template = SOKOBAN_SKILL_REFLECT_TEMPLATE if use_skill_reflection else SOKOBAN_REFLECT_TEMPLATE
            format_kwargs = dict(
                task_description=task,
                success=success_text,
                reference_trajectory=reference_trajectory,
                current_trajectory=trajectories[i],
            )
            if use_skill_reflection:
                format_kwargs["task_category"] = task_category
            obs = reflect_template.format(**format_kwargs)
            postprocess_text_obs.append(obs)

        return {
            "text": postprocess_text_obs,
            "image": None,
            "anchor": postprocess_text_obs.copy(),
            "tasks": list(self.tasks),
            "task_categories": task_categories,
            "trajectories": trajectories,
            "successes": successes,
        }


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks

    def get_task_descriptions(self) -> List[str]:
        return list(self.tasks)

    def get_terminal_successes(self, infos: List[Dict]) -> List[bool]:
        return [bool(info.get("won", False)) for info in infos]

    def _truncate_reflection_obs(self, obs: str, max_to_show: int = None) -> str:
        if max_to_show is None or max_to_show <= 0:
            return obs
        parts = str(obs).split(" [SEP] ")
        if len(parts) < max_to_show:
            return obs
        return " [SEP] ".join(parts[:max_to_show]) + f" ... ({len(parts) - max_to_show} more)"

    def get_reflection_trajectories(
        self,
        max_steps: int = None,
        max_to_show: int = 6,
        episode_lengths: List[float] = None,
    ) -> List[str]:
        trajectories = []
        for env_idx in range(self.memory.batch_size):
            records = list(self.memory[env_idx])
            if episode_lengths is not None:
                records = records[: max(int(episode_lengths[env_idx]), 0)]
            effective_total_len = len(records)
            if max_steps is not None and max_steps > 0:
                records = records[-max_steps:]
                start_idx = max(0, effective_total_len - len(records))
            else:
                start_idx = 0

            lines = []
            for j, rec in enumerate(records):
                step_num = start_idx + j + 1
                obs = rec.get("text_obs", "")
                if j != len(records) - 1:
                    obs = self._truncate_reflection_obs(obs, max_to_show=max_to_show)
                act = rec.get("action", "")
                lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']")
            trajectories.append("\n".join(lines) if lines else "No actions were recorded.")
        return trajectories

    def build_reflection_observations(
        self,
        infos: List[Dict],
        dynamic_memory_provider,
        episode_lengths: List[float] = None,
    ) -> Dict[str, Any]:
        max_steps = getattr(dynamic_memory_provider, "max_trajectory_steps", self.config.env.max_steps)
        max_to_show = getattr(dynamic_memory_provider, "max_reflection_items", 6)
        trajectories = self.get_reflection_trajectories(
            max_steps=max_steps,
            max_to_show=max_to_show,
            episode_lengths=episode_lengths,
        )
        successes = self.get_terminal_successes(infos)
        postprocess_text_obs = []
        task_categories = []
        use_skill_reflection = getattr(dynamic_memory_provider, "skill_format", "") == "claude_style"

        for i, task in enumerate(self.tasks):
            is_won = successes[i]
            success_text = "successfully" if is_won else "unsuccessfully"
            task_category = dynamic_memory_provider.infer_task_category(task)
            task_categories.append(task_category)
            if is_won:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="failure")
                reference_trajectory = (
                    "Reference Failed Trajectory (for comparison):\n" + reference
                    if reference
                    else "No failed attempts available for comparison."
                )
            else:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="success")
                reference_trajectory = (
                    "Reference Successful Trajectory (for comparison):\n" + reference
                    if reference
                    else "No successful attempts available for reference."
                )

            reflect_template = WEBSHOP_SKILL_REFLECT_TEMPLATE if use_skill_reflection else WEBSHOP_REFLECT_TEMPLATE
            format_kwargs = dict(
                task_description=task,
                success=success_text,
                reference_trajectory=reference_trajectory,
                current_trajectory=trajectories[i],
            )
            if use_skill_reflection:
                format_kwargs["task_category"] = task_category
            obs = reflect_template.format(**format_kwargs)
            postprocess_text_obs.append(obs)

        return {
            "text": postprocess_text_obs,
            "image": None,
            "anchor": postprocess_text_obs.copy(),
            "tasks": list(self.tasks),
            "task_categories": task_categories,
            "trajectories": trajectories,
            "successes": successes,
        }
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                    thinking_instruction=self.thinking_instruction("webshop"),
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                    thinking_instruction=self.thinking_instruction("webshop"),
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions,
                        thinking_instruction=self.thinking_instruction("webshop"),
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def get_task_descriptions(self) -> List[str]:
        return list(getattr(self, "tasks", []))

    def get_terminal_successes(self, infos: List[Dict]) -> List[bool]:
        return [bool(info.get("won", False)) for info in infos]

    @staticmethod
    def _truncate_reflection_text(text: str, max_chars: int = 2500) -> str:
        text = str(text or "")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        keep_left = max_chars // 2
        keep_right = max_chars - keep_left
        return text[:keep_left] + "\n...[truncated]...\n" + text[-keep_right:]

    def get_reflection_trajectories(
        self,
        max_steps: int = None,
        max_to_show: int = None,
        episode_lengths: List[float] = None,
    ) -> List[str]:
        trajectories = []
        result_max_chars = 2500
        if max_to_show is not None and max_to_show > 0:
            result_max_chars = max(800, int(max_to_show) * 500)

        for env_idx in range(self.memory.batch_size):
            records = list(self.memory[env_idx])
            if episode_lengths is not None:
                records = records[: max(int(episode_lengths[env_idx]), 0)]
            effective_total_len = len(records)
            if max_steps is not None and max_steps > 0:
                records = records[-max_steps:]
                start_idx = max(0, effective_total_len - len(records))
            else:
                start_idx = 0

            lines = []
            for j, rec in enumerate(records):
                step_num = start_idx + j + 1
                action = self._truncate_reflection_text(rec.get("action", ""), max_chars=2200)
                result = self._truncate_reflection_text(rec.get("text_obs", ""), max_chars=result_max_chars)
                lines.append(
                    f"[Code {step_num}]\n{action}\n\n"
                    f"[Result {step_num}]\n{result}"
                )
            trajectories.append("\n\n".join(lines) if lines else "No code cells were recorded.")
        return trajectories

    def build_reflection_observations(
        self,
        infos: List[Dict],
        dynamic_memory_provider,
        episode_lengths: List[float] = None,
    ) -> Dict[str, Any]:
        max_steps = getattr(dynamic_memory_provider, "max_trajectory_steps", self.config.env.max_steps)
        max_to_show = getattr(dynamic_memory_provider, "max_reflection_items", 6)
        trajectories = self.get_reflection_trajectories(
            max_steps=max_steps,
            max_to_show=max_to_show,
            episode_lengths=episode_lengths,
        )
        successes = self.get_terminal_successes(infos)
        postprocess_text_obs = []
        task_categories = []
        use_skill_reflection = getattr(dynamic_memory_provider, "skill_format", "") == "claude_style"

        for i, task in enumerate(self.tasks):
            is_won = successes[i]
            success_text = "successfully" if is_won else "unsuccessfully"
            task_category = dynamic_memory_provider.infer_task_category(task)
            task_categories.append(task_category)
            if is_won:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="failure")
                reference_trajectory = (
                    "Reference Failed Trajectory (for comparison):\n" + reference
                    if reference
                    else "No failed attempts available for comparison."
                )
            else:
                reference = dynamic_memory_provider.retrieve_reference_trajectory(task, target_type="success")
                reference_trajectory = (
                    "Reference Successful Trajectory (for comparison):\n" + reference
                    if reference
                    else "No successful attempts available for reference."
                )

            reflect_template = APPWORLD_SKILL_REFLECT_TEMPLATE if use_skill_reflection else APPWORLD_REFLECT_TEMPLATE
            format_kwargs = dict(
                task_description=task,
                success=success_text,
                reference_trajectory=reference_trajectory,
                current_trajectory=trajectories[i],
            )
            if use_skill_reflection:
                format_kwargs["task_category"] = task_category
            obs = reflect_template.format(**format_kwargs)
            postprocess_text_obs.append(obs)

        return {
            "text": postprocess_text_obs,
            "image": None,
            "anchor": postprocess_text_obs.copy(),
            "tasks": list(self.tasks),
            "task_categories": task_categories,
            "trajectories": trajectories,
            "successes": successes,
        }
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        thinking_instruction=self.thinking_instruction("appworld"),
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        thinking_instruction=self.thinking_instruction("appworld_history"),
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create training and validation environments.
    """
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    require_thinking = _require_thinking_from_config(config)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection

        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)
        projection_f = partial(search_projection, require_thinking=require_thinking)
        return SearchEnvironmentManager(_envs, projection_f, config), SearchEnvironmentManager(_val_envs, projection_f, config)

    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection

        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        return GymCardEnvironmentManager(_envs, projection_f, config), GymCardEnvironmentManager(_val_envs, projection_f, config)

    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset,
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        projection_f = partial(alfworld_projection, require_thinking=require_thinking)
        return AlfWorldEnvironmentManager(_envs, projection_f, config), AlfWorldEnvironmentManager(_val_envs, projection_f, config)

    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection

        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth,
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        projection_f = partial(sokoban_projection, require_thinking=require_thinking)
        return SokobanEnvironmentManager(_envs, projection_f, config), SokobanEnvironmentManager(_val_envs, projection_f, config)

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
            'observation_mode': 'text',
            'num_products': None,
            'human_goals': config.env.webshop.human_goals,
            'file_path': file_path,
            'attr_path': attr_path,
        }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        projection_f = partial(webshop_projection, require_thinking=require_thinking)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)
        return envs, val_envs

    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection

        _envs = build_appworld_envs(dataset_name='train', max_interactions=config.env.max_steps, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', max_interactions=config.env.max_steps, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size * group_n, resources_per_worker=resources_per_worker)
        projection_f = partial(appworld_projection, require_thinking=require_thinking)
        return AppWorldEnvironmentManager(_envs, projection_f, config), AppWorldEnvironmentManager(_val_envs, projection_f, config)

    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")


def make_val_envs(config):
    """
    Create validation environments only.

    This is used by multi-environment evaluation where training continues in
    one environment while validation may run in several differently configured
    environments.
    """
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    require_thinking = _require_thinking_from_config(config)

    if "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset,
        }
        _val_envs = build_alfworld_envs(
            alf_config_path,
            config.env.seed + 1000,
            config.data.val_batch_size,
            1,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        projection_f = partial(alfworld_projection, require_thinking=require_thinking)
        return AlfWorldEnvironmentManager(_val_envs, projection_f, config)

    if "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
            'observation_mode': 'text',
            'num_products': None,
            'human_goals': config.env.webshop.human_goals,
            'file_path': file_path,
            'attr_path': attr_path,
        }
        _val_envs = build_webshop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        projection_f = partial(webshop_projection, require_thinking=require_thinking)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep(config.data.val_batch_size * 0.1)
        return val_envs

    if "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection

        start_server_id = config.data.train_batch_size * max(int(config.env.rollout.n), 1)
        _val_envs = build_appworld_envs(
            dataset_name='test_normal',
            max_interactions=config.env.max_steps,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            start_server_id=start_server_id,
            resources_per_worker=resources_per_worker,
        )
        projection_f = partial(appworld_projection, require_thinking=require_thinking)
        return AppWorldEnvironmentManager(_val_envs, projection_f, config)

    eval_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    eval_config.data.train_batch_size = 1
    eval_config.env.rollout.n = 1
    _, val_envs = make_envs(eval_config)
    return val_envs
