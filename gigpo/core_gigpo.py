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

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


"""
Core functions to implement the GiGPO algorithm (https://arxiv.org/abs/2505.10978).
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def compute_gagpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                    step_returns: torch.Tensor,
                                    response_mask: torch.Tensor,
                                    anchor_obs: np.array,
                                    index: np.array,
                                    traj_index: np.array,
                                    turn_step: Any = None,
                                    gamma: float = 0.95,
                                    gae_lambda: float = 0.8,
                                    epsilon: float = 1e-6,
                                    mode: str = "mean_std_norm",
                                    enable_similarity: bool = False,
                                    similarity_thresh: float = 0.95,
                                    ):
    """
    Compute GAGPO advantages.

    GAGPO keeps GiGPO's rollout/state grouping, but replaces the MC-style
    step-return comparison with a critic-free TD/GAE estimator:

        V_bar(s) = mean discounted return over the same anchor state
        delta_t = r_t + gamma * V_bar(s_{t+1}) - V_bar(s_t)
        A_t = delta_t + gamma * lambda * A_{t+1}

    The final step advantages are normalized within each rollout group and
    broadcast to all response tokens of the corresponding environment action.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    response_length = response_mask.shape[-1]
    device = response_mask.device
    dtype = token_level_rewards.dtype
    bsz = response_mask.shape[0]

    step_returns = step_returns.to(device=device, dtype=torch.float32)
    step_returns = torch.nan_to_num(step_returns, nan=0.0, posinf=1.0, neginf=-1.0)
    response_mask_f = response_mask.to(device=device, dtype=torch.float32)
    immediate_rewards = (token_level_rewards.to(device=device, dtype=torch.float32) * response_mask_f).sum(dim=-1)
    immediate_rewards = torch.nan_to_num(immediate_rewards, nan=0.0, posinf=1.0, neginf=-1.0)

    # Reuse GiGPO's exact/similarity anchor-state grouping inside each rollout uid.
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    group_returns = defaultdict(list)
    for row_idx, group_uid in enumerate(step_group_uids):
        group_returns[group_uid].append(step_returns[row_idx])
    value_by_group = {
        group_uid: torch.stack(values).mean()
        for group_uid, values in group_returns.items()
    }

    state_values = torch.stack([value_by_group[group_uid] for group_uid in step_group_uids]).to(device=device)
    next_state_values = torch.zeros_like(state_values)

    if turn_step is None:
        turn_step = np.arange(bsz, dtype=np.int64)
    else:
        turn_step = np.asarray(turn_step, dtype=object)

    def _turn_key(row_idx: int):
        value = turn_step[row_idx]
        if isinstance(value, np.ndarray):
            value = value.item()
        try:
            return int(value)
        except Exception:
            return row_idx

    traj_to_rows = defaultdict(list)
    for row_idx, traj_uid in enumerate(traj_index):
        traj_to_rows[str(traj_uid)].append(row_idx)

    sorted_traj_rows = {}
    for traj_uid, rows in traj_to_rows.items():
        sorted_rows = sorted(rows, key=_turn_key)
        sorted_traj_rows[traj_uid] = sorted_rows
        for pos, row_idx in enumerate(sorted_rows[:-1]):
            next_row_idx = sorted_rows[pos + 1]
            next_state_values[row_idx] = state_values[next_row_idx]

    deltas = immediate_rewards + float(gamma) * next_state_values - state_values
    deltas = torch.nan_to_num(deltas, nan=0.0, posinf=1.0, neginf=-1.0)
    raw_advantages = torch.zeros_like(deltas)
    for rows in sorted_traj_rows.values():
        running_advantage = torch.zeros((), device=device, dtype=torch.float32)
        for row_idx in reversed(rows):
            running_advantage = deltas[row_idx] + float(gamma) * float(gae_lambda) * running_advantage
            raw_advantages[row_idx] = running_advantage

    normalized_advantages = raw_advantages.clone()
    for group_uid in np.unique(index):
        locs = np.where(index == group_uid)[0]
        if len(locs) == 0:
            continue
        loc_tensor = torch.as_tensor(locs, device=device, dtype=torch.long)
        group_scores = raw_advantages[loc_tensor]
        group_mean = group_scores.mean()
        if remove_std:
            normalized_advantages[loc_tensor] = group_scores - group_mean
        else:
            group_std = group_scores.std(unbiased=False).clamp_min(float(epsilon))
            normalized_advantages[loc_tensor] = (group_scores - group_mean) / (group_std + float(epsilon))

    normalized_advantages = torch.nan_to_num(normalized_advantages, nan=0.0, posinf=10.0, neginf=-10.0)
    advantages = normalized_advantages.to(dtype=dtype).unsqueeze(-1).tile([1, response_length]) * response_mask
    return advantages, advantages


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages
