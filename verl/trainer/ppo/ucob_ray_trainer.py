"""
UCOB (Skill-based Self-Distillation) Trainer.

Extends RLSDRayTrainer. Unlike RLSD which replaces advantages with token-level
teacher-weighted advantages, UCOB keeps the original GRPO advantages and
adds an auxiliary SDL loss computed inside the actor's update_policy().
"""

from pprint import pprint

import json
import os

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    _timer,
    apply_credit_assignment_returns,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask

from agent_system.multi_turn_rollout import adjust_batch


DSDAR_BAD_TRAJ_REWARD_THRESHOLD = 0.0
DSDAR_SUCCESS_REWARD_THRESHOLD = 1.0


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class UCOBRayTrainer(RLSDRayTrainer):
    """
    UCOB trainer that extends RLSDRayTrainer.

    Keeps the original GRPO advantages unchanged and passes teacher_log_probs
    to the actor, where the SDL loss is computed inside update_policy().
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        skillsd_cfg = self.config.algorithm.get("skillsd", {})
        self.sdl_lambda = skillsd_cfg.get("sdl_lambda", 0.1)
        self.sdl_warmdown_steps = skillsd_cfg.get("warmdown_steps", -1)
        actor_cfg = self.config.actor_rollout_ref.actor
        self.teacher_loss_enable = _as_bool(actor_cfg.get("use_sdl_loss", False)) or _as_bool(
            actor_cfg.get("use_sdar_loss", False)
        )
        self._dynamic_memory_snapshot_steps = set()

    @staticmethod
    def _repeat_zero_like(tensor: torch.Tensor, rows: int) -> torch.Tensor:
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
                left_tensors[key] = UCOBRayTrainer._repeat_zero_like(right_tensors[key], len(left))
            if key not in right_tensors:
                right_tensors[key] = UCOBRayTrainer._repeat_zero_like(left_tensors[key], len(right))

        left_non_tensors = dict(left.non_tensor_batch)
        right_non_tensors = dict(right.non_tensor_batch)
        for key in sorted(set(left_non_tensors.keys()) | set(right_non_tensors.keys())):
            if key not in left_non_tensors:
                left_non_tensors[key] = UCOBRayTrainer._empty_non_tensor_like(
                    len(left),
                    np.asarray(right_non_tensors[key], dtype=object),
                )
            if key not in right_non_tensors:
                right_non_tensors[key] = UCOBRayTrainer._empty_non_tensor_like(
                    len(right),
                    np.asarray(left_non_tensors[key], dtype=object),
                )
            left_non_tensors[key] = UCOBRayTrainer._align_non_tensor_ndim(
                left_non_tensors[key],
                len(left),
                np.asarray(right_non_tensors[key], dtype=object),
            )
            right_non_tensors[key] = UCOBRayTrainer._align_non_tensor_ndim(
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
    def _split_ucob_cbsd_batch(batch: DataProto) -> tuple[DataProto, DataProto | None]:
        if batch.batch is None or "ucob_cbsd_loss_mask" not in batch.batch.keys():
            return batch, None
        loss_mask = batch.batch["ucob_cbsd_loss_mask"]
        flat_mask = loss_mask.reshape(loss_mask.shape[0], -1)
        is_aux = flat_mask.sum(dim=-1).detach().cpu().numpy() > 0
        if not np.any(is_aux):
            return batch, None
        original_indices = np.where(~is_aux)[0]
        aux_indices = np.where(is_aux)[0]
        if len(original_indices) == 0:
            return batch, None
        original = batch.select_idxs(original_indices)
        aux = batch.select_idxs(aux_indices)
        return original, aux

    @staticmethod
    def _split_dynamic_memory_reflection_batch(batch: DataProto) -> tuple[DataProto, DataProto | None]:
        if batch.batch is None or "dynamic_memory_reflection_train_mask" not in batch.batch.keys():
            return batch, None
        train_mask = batch.batch["dynamic_memory_reflection_train_mask"]
        flat_mask = train_mask.reshape(train_mask.shape[0], -1)
        is_aux = flat_mask.sum(dim=-1).detach().cpu().numpy() > 0
        if not np.any(is_aux):
            return batch, None
        original_indices = np.where(~is_aux)[0]
        aux_indices = np.where(is_aux)[0]
        if len(original_indices) == 0:
            return batch, None
        original = batch.select_idxs(original_indices)
        aux = batch.select_idxs(aux_indices)
        return original, aux

    @staticmethod
    def _cbsd_aux_analysis_metrics(aux: DataProto | None, meta_info: dict | None = None) -> dict[str, float]:
        metrics = {
            "analysis/mech/aux_rows": 0.0,
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
            "analysis/mech/teacher_to_none_rows": 0.0,
            "analysis/mech/teacher_to_skill_rows": 0.0,
            "analysis/mech/teacher_to_none_ratio": 0.0,
            "analysis/mech/teacher_to_skill_ratio": 0.0,
            "analysis/mech/return_gap_teacher_to_none_mean": 0.0,
            "analysis/mech/return_gap_teacher_to_skill_mean": 0.0,
            "analysis/mech/gate_pass_ratio": 0.0,
            "analysis/mech/loss_weight_mean": 0.0,
        }

        def _merge_meta_info(source: dict | None) -> None:
            if not source:
                return
            for key in list(metrics.keys()):
                if key not in source:
                    continue
                try:
                    value = float(source[key])
                except (TypeError, ValueError):
                    value = 0.0
                metrics[key] = value if np.isfinite(value) else 0.0

        _merge_meta_info(meta_info)
        if aux is None or len(aux) == 0:
            return metrics
        _merge_meta_info(getattr(aux, "meta_info", None))

        rows = len(aux)
        total = float(max(rows, 1))

        def _string_array(key: str) -> np.ndarray:
            values = aux.non_tensor_batch.get(key, None)
            if values is None:
                return np.array([""] * rows, dtype=object)
            values = np.asarray(values, dtype=object)
            if values.shape[0] != rows:
                return np.array([""] * rows, dtype=object)
            if values.ndim > 1:
                values = values.reshape(rows, -1)[:, 0]
            return np.asarray([str(value).strip().lower() for value in values], dtype=object)

        def _float_array(key: str) -> np.ndarray:
            values = aux.non_tensor_batch.get(key, None)
            if values is None:
                return np.zeros(rows, dtype=np.float32)
            try:
                values = np.asarray(values, dtype=np.float32)
            except Exception:
                return np.zeros(rows, dtype=np.float32)
            if values.shape[0] != rows:
                return np.zeros(rows, dtype=np.float32)
            if values.ndim > 1:
                values = values.reshape(rows, -1)[:, 0]
            return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            return float(np.mean(values[mask]))

        sources = _string_array("ucob_cbsd_source_branch")
        targets = _string_array("ucob_cbsd_target_branch")
        return_gaps = _float_array("ucob_cbsd_return_gap")

        source_skill = sources == "good"
        source_none = sources == "none"
        teacher_to_none = source_skill & (targets == "none")
        teacher_to_skill = source_none & (targets == "good")

        teacher_to_none_rows = float(np.sum(teacher_to_none))
        teacher_to_skill_rows = float(np.sum(teacher_to_skill))
        metrics["analysis/mech/aux_rows"] = float(rows)
        metrics["analysis/mech/teacher_to_none_rows"] = teacher_to_none_rows
        metrics["analysis/mech/teacher_to_skill_rows"] = teacher_to_skill_rows
        metrics["analysis/mech/teacher_to_none_ratio"] = teacher_to_none_rows / total
        metrics["analysis/mech/teacher_to_skill_ratio"] = teacher_to_skill_rows / total
        metrics["analysis/mech/return_gap_teacher_to_none_mean"] = _masked_mean(return_gaps, teacher_to_none)
        metrics["analysis/mech/return_gap_teacher_to_skill_mean"] = _masked_mean(return_gaps, teacher_to_skill)

        raw_pair_count = float(metrics.get("analysis/mech/raw_pair_count", 0.0) or 0.0)
        if raw_pair_count <= 0.0:
            raw_pair_count = total
            metrics["analysis/mech/raw_pair_count"] = raw_pair_count
        if metrics.get("analysis/mech/skill_to_noskill_pair_count", 0.0) == 0.0:
            metrics["analysis/mech/skill_to_noskill_pair_count"] = teacher_to_none_rows
        if metrics.get("analysis/mech/noskill_to_skill_pair_count", 0.0) == 0.0:
            metrics["analysis/mech/noskill_to_skill_pair_count"] = teacher_to_skill_rows
        if metrics.get("analysis/mech/activated_pair_count", 0.0) == 0.0:
            metrics["analysis/mech/activated_pair_count"] = (
                metrics["analysis/mech/skill_to_noskill_pair_count"]
                + metrics["analysis/mech/noskill_to_skill_pair_count"]
            )
        activated_pair_count = float(metrics["analysis/mech/activated_pair_count"])
        ignored_pair_count = max(
            raw_pair_count
            - float(metrics["analysis/mech/skill_to_noskill_pair_count"])
            - float(metrics["analysis/mech/noskill_to_skill_pair_count"]),
            0.0,
        )
        metrics["analysis/mech/ignored_pair_count"] = ignored_pair_count
        metrics["analysis/mech/ignored_pair_ratio"] = ignored_pair_count / max(raw_pair_count, 1.0)
        metrics["analysis/mech/activated_pair_ratio"] = activated_pair_count / max(raw_pair_count, 1.0)
        metrics["analysis/mech/skill_to_noskill_pair_ratio"] = (
            float(metrics["analysis/mech/skill_to_noskill_pair_count"]) / max(raw_pair_count, 1.0)
        )
        metrics["analysis/mech/noskill_to_skill_pair_ratio"] = (
            float(metrics["analysis/mech/noskill_to_skill_pair_count"]) / max(raw_pair_count, 1.0)
        )

        if "ucob_cbsd_loss_weight" in aux.batch.keys():
            loss_weight = aux.batch["ucob_cbsd_loss_weight"].detach().float().cpu()
            if loss_weight.dim() == 1:
                loss_weight = loss_weight.unsqueeze(-1)
            active_mask = torch.ones_like(loss_weight)
            if "ucob_cbsd_loss_mask" in aux.batch.keys():
                loss_mask = aux.batch["ucob_cbsd_loss_mask"].detach().float().cpu()
                if loss_mask.dim() == 1:
                    loss_mask = loss_mask.unsqueeze(-1)
                if loss_mask.shape[-1] != loss_weight.shape[-1]:
                    loss_mask = loss_mask[:, -loss_weight.shape[-1]:]
                if loss_mask.shape == loss_weight.shape:
                    active_mask = (loss_mask > 0).to(dtype=loss_weight.dtype)
            denom = active_mask.sum(dim=-1).clamp(min=1.0)
            row_weights = ((loss_weight * active_mask).sum(dim=-1) / denom).numpy()
            metrics["analysis/mech/loss_weight_mean"] = float(np.mean(row_weights)) if row_weights.size else 0.0
            metrics["analysis/mech/gate_pass_ratio"] = float(np.mean(row_weights > 0.0)) if row_weights.size else 0.0

        return metrics

    @staticmethod
    def _dynamic_memory_metric_key_map() -> dict[str, str]:
        return {
            "dynamic_memory_bank_size_total": "analysis/memory/bank_size_total",
            "dynamic_memory_bank_size_task": "analysis/memory/bank_size_task",
            "dynamic_memory_bank_size_state": "analysis/memory/bank_size_state",
            "dynamic_memory_bank_utility_mean": "analysis/memory/bank_utility_mean",
            "dynamic_memory_task_bank_utility_mean": "analysis/memory/task_bank_utility_mean",
            "dynamic_memory_state_bank_utility_mean": "analysis/memory/state_bank_utility_mean",
            "dynamic_memory_bank_utility_p25": "analysis/memory/bank_utility_p25",
            "dynamic_memory_bank_utility_p50": "analysis/memory/bank_utility_p50",
            "dynamic_memory_bank_utility_p75": "analysis/memory/bank_utility_p75",
            "dynamic_memory_task_bank_utility_p25": "analysis/memory/task_bank_utility_p25",
            "dynamic_memory_task_bank_utility_p50": "analysis/memory/task_bank_utility_p50",
            "dynamic_memory_task_bank_utility_p75": "analysis/memory/task_bank_utility_p75",
            "dynamic_memory_state_bank_utility_p25": "analysis/memory/state_bank_utility_p25",
            "dynamic_memory_state_bank_utility_p50": "analysis/memory/state_bank_utility_p50",
            "dynamic_memory_state_bank_utility_p75": "analysis/memory/state_bank_utility_p75",
            "dynamic_memory_useful_skill_count": "analysis/memory/useful_skill_count",
            "dynamic_memory_task_useful_skill_count": "analysis/memory/task_useful_skill_count",
            "dynamic_memory_state_useful_skill_count": "analysis/memory/state_useful_skill_count",
            "dynamic_memory_useful_skill_ratio": "analysis/memory/useful_skill_ratio",
            "dynamic_memory_task_useful_skill_ratio": "analysis/memory/task_useful_skill_ratio",
            "dynamic_memory_state_useful_skill_ratio": "analysis/memory/state_useful_skill_ratio",
            "dynamic_memory_retrieval_time_s": "analysis/eff/retrieval_time_s",
            "dynamic_memory_task_retrieval_time_s": "analysis/eff/task_retrieval_time_s",
            "dynamic_memory_state_retrieval_time_s": "analysis/eff/state_retrieval_time_s",
            "dynamic_memory_skill_generation_time_s": "analysis/eff/skill_generation_time_s",
            "dynamic_memory_task_skill_generation_time_s": "analysis/eff/task_skill_generation_time_s",
            "dynamic_memory_state_skill_generation_time_s": "analysis/eff/state_skill_generation_time_s",
            "dynamic_memory_skill_write_time_s": "analysis/eff/skill_write_time_s",
            "dynamic_memory_task_skill_write_time_s": "analysis/eff/task_skill_write_time_s",
            "dynamic_memory_state_skill_write_time_s": "analysis/eff/state_skill_write_time_s",
            "dynamic_memory_retrieved_count_mean": "analysis/memory/retrieved_count_mean",
            "dynamic_memory_task_retrieved_count_mean": "analysis/memory/task_retrieved_count_mean",
            "dynamic_memory_state_retrieved_count_mean": "analysis/memory/state_retrieved_count_mean",
            "dynamic_memory_retrieved_utility_mean": "analysis/memory/retrieved_utility_mean",
            "dynamic_memory_task_retrieved_utility_mean": "analysis/memory/task_retrieved_utility_mean",
            "dynamic_memory_state_retrieved_utility_mean": "analysis/memory/state_retrieved_utility_mean",
            "dynamic_memory_retrieved_relevance_mean": "analysis/memory/retrieved_relevance_mean",
            "dynamic_memory_task_retrieved_relevance_mean": "analysis/memory/task_retrieved_relevance_mean",
            "dynamic_memory_state_retrieved_relevance_mean": "analysis/memory/state_retrieved_relevance_mean",
            "dynamic_memory_retrieved_score_mean": "analysis/memory/retrieved_score_mean",
            "dynamic_memory_task_retrieved_score_mean": "analysis/memory/task_retrieved_score_mean",
            "dynamic_memory_state_retrieved_score_mean": "analysis/memory/state_retrieved_score_mean",
            "dynamic_memory_task_write_attempted": "analysis/memory/task_write_attempted",
            "dynamic_memory_task_write_added": "analysis/memory/task_write_added",
            "dynamic_memory_task_write_success_rate": "analysis/memory/task_write_success_rate",
            "dynamic_memory_state_write_attempted": "analysis/memory/state_write_attempted",
            "dynamic_memory_state_write_added": "analysis/memory/state_write_added",
            "dynamic_memory_state_write_success_rate": "analysis/memory/state_write_success_rate",
            "dynamic_memory_write_added_total": "analysis/memory/write_added_total",
            "dynamic_memory_write_success_rate_total": "analysis/memory/write_success_rate_total",
            "dynamic_memory_task_write_reason_parse_failed": "analysis/memory/task_write_reason_parse_failed",
            "dynamic_memory_task_write_reason_success_mismatch": "analysis/memory/task_write_reason_success_mismatch",
            "dynamic_memory_task_write_reason_empty_skill": "analysis/memory/task_write_reason_empty_skill",
            "dynamic_memory_task_write_reason_empty_lesson": "analysis/memory/task_write_reason_empty_lesson",
            "dynamic_memory_task_write_reason_add_failed": "analysis/memory/task_write_reason_add_failed",
            "dynamic_memory_task_write_reason_other": "analysis/memory/task_write_reason_other",
            "dynamic_memory_state_write_reason_parse_failed": "analysis/memory/state_write_reason_parse_failed",
            "dynamic_memory_state_write_reason_success_mismatch": "analysis/memory/state_write_reason_success_mismatch",
            "dynamic_memory_state_write_reason_empty_skill": "analysis/memory/state_write_reason_empty_skill",
            "dynamic_memory_state_write_reason_empty_lesson": "analysis/memory/state_write_reason_empty_lesson",
            "dynamic_memory_state_write_reason_add_failed": "analysis/memory/state_write_reason_add_failed",
            "dynamic_memory_state_write_reason_other": "analysis/memory/state_write_reason_other",
            "dynamic_memory_utility_update_groups": "analysis/memory/utility_update_groups",
            "dynamic_memory_utility_updates_total": "analysis/memory/utility_updates_total",
            "dynamic_memory_task_utility_updates": "analysis/memory/task_utility_updates",
            "dynamic_memory_state_utility_updates": "analysis/memory/state_utility_updates",
            "dynamic_memory_base_prompt_tokens_mean": "analysis/eff/base_prompt_tokens_mean",
            "dynamic_memory_skill_prompt_tokens_mean": "analysis/eff/skill_prompt_tokens_mean",
            "dynamic_memory_memory_context_tokens_mean": "analysis/eff/memory_context_tokens_mean",
            "dynamic_memory_prompt_token_overhead_ratio": "analysis/eff/prompt_token_overhead_ratio",
        }

    @staticmethod
    def _dynamic_memory_rollout_metrics(batch: DataProto | None) -> dict[str, float]:
        key_map = UCOBRayTrainer._dynamic_memory_metric_key_map()
        if batch is None:
            return {}
        non_tensors = getattr(batch, "non_tensor_batch", {}) or {}
        metrics = {metric_key: 0.0 for metric_key in key_map.values()}
        if not any(key in non_tensors for key in key_map):
            return metrics
        for source_key, metric_key in key_map.items():
            if source_key not in non_tensors:
                continue
            try:
                values = np.asarray(non_tensors[source_key], dtype=np.float32)
            except Exception:
                continue
            if values.size == 0:
                continue
            metrics[metric_key] = float(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).mean())
        return metrics

    @staticmethod
    def _dynamic_memory_stats_metrics(stats: dict | None, prefix: str = "") -> dict[str, float]:
        if not stats:
            return {}
        key_map = UCOBRayTrainer._dynamic_memory_metric_key_map()
        prefix = str(prefix or "").strip("/")
        metrics = {}
        for source_key, metric_key in key_map.items():
            if source_key not in stats:
                continue
            target_key = metric_key
            if prefix:
                target_key = f"{prefix}/{metric_key}"
            try:
                value = float(stats[source_key])
            except (TypeError, ValueError):
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            metrics[target_key] = value
        return metrics

    @staticmethod
    def _aggregate_dynamic_memory_stats(stats_list) -> dict:
        stats_list = [stats for stats in (stats_list or []) if isinstance(stats, dict) and stats]
        if not stats_list:
            return {}
        key_map = UCOBRayTrainer._dynamic_memory_metric_key_map()
        time_keys = {
            "dynamic_memory_retrieval_time_s",
            "dynamic_memory_task_retrieval_time_s",
            "dynamic_memory_state_retrieval_time_s",
            "dynamic_memory_skill_generation_time_s",
            "dynamic_memory_task_skill_generation_time_s",
            "dynamic_memory_state_skill_generation_time_s",
            "dynamic_memory_skill_write_time_s",
            "dynamic_memory_task_skill_write_time_s",
            "dynamic_memory_state_skill_write_time_s",
        }
        last_value_keys = {
            "dynamic_memory_bank_size_total",
            "dynamic_memory_bank_size_task",
            "dynamic_memory_bank_size_state",
            "dynamic_memory_bank_utility_mean",
            "dynamic_memory_task_bank_utility_mean",
            "dynamic_memory_state_bank_utility_mean",
            "dynamic_memory_bank_utility_p25",
            "dynamic_memory_bank_utility_p50",
            "dynamic_memory_bank_utility_p75",
            "dynamic_memory_task_bank_utility_p25",
            "dynamic_memory_task_bank_utility_p50",
            "dynamic_memory_task_bank_utility_p75",
            "dynamic_memory_state_bank_utility_p25",
            "dynamic_memory_state_bank_utility_p50",
            "dynamic_memory_state_bank_utility_p75",
            "dynamic_memory_useful_skill_count",
            "dynamic_memory_task_useful_skill_count",
            "dynamic_memory_state_useful_skill_count",
            "dynamic_memory_useful_skill_ratio",
            "dynamic_memory_task_useful_skill_ratio",
            "dynamic_memory_state_useful_skill_ratio",
        }
        result = {}
        for source_key in key_map:
            values = []
            for stats in stats_list:
                if source_key not in stats:
                    continue
                try:
                    value = float(stats[source_key])
                except (TypeError, ValueError):
                    value = 0.0
                if np.isfinite(value):
                    values.append(value)
            if not values:
                continue
            if source_key in time_keys:
                result[source_key] = float(np.sum(values))
            elif source_key in last_value_keys:
                result[source_key] = float(values[-1])
            else:
                result[source_key] = float(np.mean(values))
        return result

    @staticmethod
    def _cbsd_token_fraction(original: DataProto | None, aux: DataProto | None) -> float:
        if original is None or aux is None or len(original) == 0 or len(aux) == 0:
            return 0.0
        try:
            response_length = int(original.batch["responses"].shape[-1])
            original_response_mask = original.batch["attention_mask"][:, -response_length:].detach().float()
            original_tokens = float(original_response_mask.sum().item())
            if original_tokens <= 0.0:
                return 0.0
            if "ucob_cbsd_loss_mask" in aux.batch.keys():
                aux_mask = aux.batch["ucob_cbsd_loss_mask"].detach().float()
                aux_tokens = float(aux_mask.sum().item())
            else:
                aux_response_length = int(aux.batch["responses"].shape[-1])
                aux_tokens = float(aux.batch["attention_mask"][:, -aux_response_length:].detach().float().sum().item())
            return float(aux_tokens / max(original_tokens, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _efficiency_overhead_metrics(metrics: dict, timing_raw: dict) -> dict[str, float]:
        # Efficiency analysis aliases for paper reporting; no training tensors are changed.
        step_time = float(timing_raw.get("step", 0.0) or 0.0)
        gen_time = float(timing_raw.get("gen", 0.0) or 0.0)
        update_actor_time = float(timing_raw.get("update_actor", 0.0) or 0.0)
        reflection_time = float(timing_raw.get("update_reflection_actor", 0.0) or 0.0)
        reflection_ratio = float(reflection_time / step_time) if step_time > 0.0 else 0.0
        skill_generation_time = float(metrics.get("analysis/eff/skill_generation_time_s", 0.0) or 0.0)
        skill_write_time = float(metrics.get("analysis/eff/skill_write_time_s", 0.0) or 0.0)
        skill_generation_step_ratio = float(skill_generation_time / step_time) if step_time > 0.0 else 0.0
        skill_generation_gen_ratio = float(skill_generation_time / gen_time) if gen_time > 0.0 else 0.0
        skill_write_step_ratio = float(skill_write_time / step_time) if step_time > 0.0 else 0.0
        reflection_actor_update_actor_ratio = float(reflection_time / update_actor_time) if update_actor_time > 0.0 else 0.0
        return {
            "analysis/eff/cbsd_actor_row_ratio": float(metrics.get("analysis/eff/cbsd_actor_row_ratio", 0.0) or 0.0),
            "analysis/eff/reflection_actor_time_ratio": reflection_ratio,
            "analysis/eff/reflection_actor_vs_policy_update_ratio": reflection_actor_update_actor_ratio,
            "analysis/eff/cbsd_token_ratio": float(metrics.get("analysis/eff/cbsd_token_ratio", 0.0) or 0.0),
            "analysis/eff/skill_generation_step_ratio": skill_generation_step_ratio,
            "analysis/eff/skill_generation_gen_ratio": skill_generation_gen_ratio,
            "analysis/eff/skill_write_step_ratio": skill_write_step_ratio,
            "analysis/eff/train_extra_time_ratio": reflection_ratio + skill_generation_step_ratio + skill_write_step_ratio,
        }

    @staticmethod
    def _dynamic_memory_pool(item: dict) -> str:
        item_pool = str(item.get("memory_pool", "") or "").strip().lower()
        if item_pool in {"task", "state"}:
            return item_pool
        memory_type = str(item.get("memory_type", "") or "").strip().lower()
        return "state" if memory_type in {"state_skill", "step_skill"} else "task"

    @staticmethod
    def _safe_snapshot_field(value, max_chars: int = 600):
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return value
        return " ".join(str(value).strip().split())[:max_chars]

    @staticmethod
    def _safe_snapshot_number(value, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(number):
            return float(default)
        return number

    def _maybe_dump_dynamic_memory_snapshot(self, global_step: int) -> None:
        test_freq = int(self.config.trainer.get("test_freq", 0) or 0)
        save_freq = int(self.config.trainer.get("save_freq", 0) or 0)
        is_snapshot_step = (
            (test_freq > 0 and global_step % test_freq == 0)
            or (save_freq > 0 and global_step % save_freq == 0)
        )
        if not is_snapshot_step or global_step in self._dynamic_memory_snapshot_steps:
            return

        provider = getattr(self.traj_collector, "dynamic_memory_provider", None)
        memory = getattr(provider, "memory", None)
        data = list(getattr(memory, "data", []) or [])
        if not data:
            return

        self._dynamic_memory_snapshot_steps.add(global_step)
        task_items = [item for item in data if self._dynamic_memory_pool(item) != "state"]
        state_items = [item for item in data if self._dynamic_memory_pool(item) == "state"]

        def _top_items(items):
            ranked = sorted(
                items,
                key=lambda item: (
                    self._safe_snapshot_number(item.get("utility_score", 0.0), default=0.0),
                    self._safe_snapshot_number(item.get("count", 1), default=1.0),
                ),
                reverse=True,
            )
            safe_keys = [
                "memory_id",
                "memory_pool",
                "utility_score",
                "count",
                "title",
                "principle",
                "when_to_apply",
                "task_category",
                "created_at_progress",
                "source",
            ]
            snapshots = []
            for item in ranked[:5]:
                record = {key: self._safe_snapshot_field(item.get(key, "")) for key in safe_keys}
                if not record.get("memory_pool"):
                    record["memory_pool"] = self._dynamic_memory_pool(item)
                snapshots.append(record)
            return snapshots

        payload = {
            "global_step": int(global_step),
            "bank_size_total": len(data),
            "bank_size_task": len(task_items),
            "bank_size_state": len(state_items),
            "top_task_skills": _top_items(task_items),
            "top_state_skills": _top_items(state_items),
        }
        save_dir = os.environ.get("DYNAMIC_MEMORY_SNAPSHOT_DIR", "outputs/dynamic_memory_snapshots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{global_step}.json")
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[UCOB][skill_memory] saved memory snapshot to {save_path}")

    def _prepare_cbsd_aux_for_actor(self, aux: DataProto, reference: DataProto) -> DataProto:
        aux.meta_info = reference.meta_info
        ucob_cfg = self.config.algorithm.get("ucob", {}) or {}
        cbsd_cfg = ucob_cfg.get("cbsd", {}) or {}
        if not cbsd_cfg:
            cbsd_cfg = ucob_cfg
        actor_cfg = self.config.actor_rollout_ref.actor
        loss_type = str(
            cbsd_cfg.get("loss_type", actor_cfg.get("ucob_cbsd_loss_type", "nll"))
        ).strip().lower()
        topk_cbsd_loss_types = {"topk_opd", "topk-opd", "topk"}
        sampled_cbsd_loss_types = {"opd"}
        if loss_type not in topk_cbsd_loss_types | sampled_cbsd_loss_types:
            return aux
        if (
            "ucob_cbsd_ref_prompt_ids" not in aux.batch.keys()
            or "ucob_cbsd_ref_prompt_attention_mask" not in aux.batch.keys()
        ):
            return aux

        response_length = aux.batch["responses"].size(1)
        ref_prompts = aux.batch["ucob_cbsd_ref_prompt_ids"]
        ref_prompt_mask = aux.batch["ucob_cbsd_ref_prompt_attention_mask"]
        response_mask = aux.batch["attention_mask"][:, -response_length:].to(dtype=ref_prompt_mask.dtype)
        ref_input_ids = torch.cat([ref_prompts, aux.batch["responses"]], dim=-1)
        ref_attention_mask = torch.cat([ref_prompt_mask, response_mask], dim=-1)
        ref_position_ids = compute_position_id_with_mask(ref_attention_mask)

        ref_batch = DataProto.from_dict(
            tensors={
                "input_ids": ref_input_ids,
                "attention_mask": ref_attention_mask,
                "position_ids": ref_position_ids,
                "responses": aux.batch["responses"],
            }
        )
        ref_batch.meta_info = dict(reference.meta_info)
        ref_batch.meta_info["calculate_entropy"] = False
        if loss_type not in sampled_cbsd_loss_types:
            ref_batch.meta_info["kl_topk_k"] = int(
                cbsd_cfg.get("topk_k", actor_cfg.get("ucob_cbsd_topk_k", 32))
            )
        ref_batch_padded, pad_size = pad_dataproto_to_divisor(
            ref_batch,
            self.actor_rollout_wg.world_size,
        )
        if loss_type in sampled_cbsd_loss_types:
            ref_output_padded = self.actor_rollout_wg.compute_log_prob(ref_batch_padded)
            ref_output = unpad_dataproto(ref_output_padded, pad_size=pad_size)
            if "entropys" in ref_output.batch.keys():
                ref_output.batch.pop("entropys")
            aux.batch["ucob_cbsd_ref_log_probs"] = ref_output.batch["old_log_probs"]
            return aux

        ref_output_padded = self.actor_rollout_wg.compute_log_prob_with_logits(ref_batch_padded)
        ref_output = unpad_dataproto(ref_output_padded, pad_size=pad_size)

        aux.batch["ucob_cbsd_ref_log_probs"] = ref_output.batch["old_log_probs"]
        aux.batch["ucob_cbsd_ref_logits_k"] = ref_output.batch["actor_logits_k"]
        aux.batch["ucob_cbsd_ref_logsumexp"] = ref_output.batch["actor_logsumexp"]
        if "actor_topk_indices" in ref_output.batch.keys():
            aux.batch["ucob_cbsd_ref_topk_indices"] = ref_output.batch["actor_topk_indices"]
        return aux

    def _prepare_reflection_aux_for_actor(self, aux: DataProto, reference: DataProto) -> DataProto:
        aux.meta_info = dict(reference.meta_info)
        aux.meta_info["multi_turn"] = False
        aux.meta_info["calculate_entropy"] = False
        if "temperature" not in aux.meta_info and "temperature" in reference.meta_info:
            aux.meta_info["temperature"] = reference.meta_info["temperature"]

        for key in list(aux.batch.keys()):
            if (
                key.startswith("ucob_cbsd_")
                or key.startswith("dsdar_ref_")
                or key in {"sdar_gate_sign", "sdar_direction_id", "action_token_mask"}
            ):
                aux.batch.pop(key)

        aux.batch["response_mask"] = compute_response_mask(aux)
        response_length = aux.batch["responses"].size(1)
        prompt_mask = aux.batch["attention_mask"][:, :-response_length]
        aux.batch["loss_mask"] = torch.cat(
            [
                torch.zeros_like(prompt_mask),
                aux.batch["response_mask"].to(dtype=prompt_mask.dtype),
            ],
            dim=-1,
        )
        weight = float(self.config.algorithm.dynamic_memory.get("reflection_loss_weight", 1.0))
        if "advantages" in aux.batch.keys():
            aux.batch["advantages"] = aux.batch["advantages"] * weight
        if "returns" in aux.batch.keys():
            aux.batch["returns"] = aux.batch["returns"] * weight

        aux_padded, pad_size = pad_dataproto_to_divisor(aux, self.actor_rollout_wg.world_size)
        old_log_prob_padded = self.actor_rollout_wg.compute_log_prob(aux_padded)
        old_log_prob = unpad_dataproto(old_log_prob_padded, pad_size=pad_size)
        if "entropys" in old_log_prob.batch.keys():
            old_log_prob.batch.pop("entropys")
        aux = aux.union(old_log_prob)

        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_log_prob_padded = self.ref_policy_wg.compute_ref_log_prob(aux_padded)
            else:
                ref_log_prob_padded = self.actor_rollout_wg.compute_ref_log_prob(aux_padded)
            ref_log_prob = unpad_dataproto(ref_log_prob_padded, pad_size=pad_size)
            aux = aux.union(ref_log_prob)

        actor_cfg = self.config.actor_rollout_ref.actor
        if _as_bool(actor_cfg.get("use_sdar_loss", False)) or _as_bool(actor_cfg.get("use_sdl_loss", False)):
            aux.batch["teacher_log_probs"] = torch.zeros_like(aux.batch["old_log_probs"])
            aux.batch["sdar_loss_weight"] = torch.zeros_like(aux.batch["old_log_probs"])

        aux.meta_info["global_token_num"] = torch.sum(aux.batch["attention_mask"], dim=-1).tolist()
        return aux

    @staticmethod
    def _pad_batch_to_divisor(batch: DataProto, divisor: int) -> DataProto:
        divisor = int(divisor)
        if divisor <= 1 or len(batch) == 0:
            return batch
        remainder = len(batch) % divisor
        if remainder == 0:
            return batch
        to_add = divisor - remainder
        dup_indices = np.random.choice(len(batch), to_add, replace=to_add > len(batch))
        dup_proto = batch.select_idxs(dup_indices)
        return DataProto.concat([batch, dup_proto])

    @staticmethod
    def _iter_batch_chunks(batch: DataProto, chunk_size: int) -> list[DataProto]:
        chunk_size = int(chunk_size)
        if chunk_size <= 0 or len(batch) <= chunk_size:
            return [batch]
        return [
            batch.select_idxs(np.arange(start, min(start + chunk_size, len(batch))))
            for start in range(0, len(batch), chunk_size)
        ]

    @staticmethod
    def _extend_metric_lists(target: dict, source: dict) -> None:
        for key, values in source.items():
            if isinstance(values, (list, tuple)):
                target.setdefault(key, []).extend(values)
            else:
                target.setdefault(key, []).append(values)

    def _reflection_aux_metrics(self, aux: DataProto | None) -> dict:
        metrics = {
            "analysis/memory/reflection/candidate_rows": 0.0,
            "analysis/memory/reflection/task_rows": 0.0,
            "analysis/memory/reflection/state_rows": 0.0,
            "analysis/memory/reflection/task_utility_mean": 0.0,
            "analysis/memory/reflection/state_utility_mean": 0.0,
            "analysis/memory/reflection/task_advantage_abs_mean": 0.0,
            "analysis/memory/reflection/state_advantage_abs_mean": 0.0,
            "analysis/memory/reflection/utility_mean": 0.0,
            "analysis/memory/reflection/utility_abs_mean": 0.0,
            "analysis/memory/reflection/advantage_mean": 0.0,
            "analysis/memory/reflection/advantage_abs_mean": 0.0,
            "analysis/memory/reflection/step_count_mean": 0.0,
        }
        if aux is None or len(aux) == 0:
            return metrics

        metrics["analysis/memory/reflection/candidate_rows"] = float(len(aux))
        pools = None
        if "dynamic_memory_reflection_pool" in aux.non_tensor_batch:
            pools = np.asarray(aux.non_tensor_batch["dynamic_memory_reflection_pool"], dtype=object)
            if pools.ndim > 1:
                pools = pools.reshape(len(aux), -1)[:, 0]
            pools = np.asarray([str(pool).strip().lower() for pool in pools], dtype=object)
            metrics["analysis/memory/reflection/task_rows"] = float(np.sum(pools == "task"))
            metrics["analysis/memory/reflection/state_rows"] = float(np.sum(pools == "state"))
        if "dynamic_memory_reflection_utility" in aux.batch.keys():
            utilities = aux.batch["dynamic_memory_reflection_utility"].detach().float()
            metrics["analysis/memory/reflection/utility_mean"] = utilities.mean().item()
            metrics["analysis/memory/reflection/utility_abs_mean"] = utilities.abs().mean().item()
            if pools is not None and len(pools) == utilities.shape[0]:
                utilities_np = utilities.cpu().numpy()
                task_mask = pools == "task"
                state_mask = pools == "state"
                if np.any(task_mask):
                    metrics["analysis/memory/reflection/task_utility_mean"] = float(np.mean(utilities_np[task_mask]))
                if np.any(state_mask):
                    metrics["analysis/memory/reflection/state_utility_mean"] = float(np.mean(utilities_np[state_mask]))
        if "dynamic_memory_reflection_advantage" in aux.batch.keys():
            advantages = aux.batch["dynamic_memory_reflection_advantage"].detach().float()
            metrics["analysis/memory/reflection/advantage_mean"] = advantages.mean().item()
            metrics["analysis/memory/reflection/advantage_abs_mean"] = advantages.abs().mean().item()
            if pools is not None and len(pools) == advantages.shape[0]:
                advantages_abs_np = advantages.abs().cpu().numpy()
                task_mask = pools == "task"
                state_mask = pools == "state"
                if np.any(task_mask):
                    metrics["analysis/memory/reflection/task_advantage_abs_mean"] = float(
                        np.mean(advantages_abs_np[task_mask])
                    )
                if np.any(state_mask):
                    metrics["analysis/memory/reflection/state_advantage_abs_mean"] = float(
                        np.mean(advantages_abs_np[state_mask])
                    )
        if "dynamic_memory_reflection_step_count" in aux.batch.keys():
            step_counts = aux.batch["dynamic_memory_reflection_step_count"].detach().float()
            metrics["analysis/memory/reflection/step_count_mean"] = step_counts.mean().item()
        return metrics

    def _update_reflection_actor(self, aux: DataProto, reference: DataProto, timing_raw: dict) -> dict:
        if aux is None or len(aux) == 0:
            return {"analysis/memory/reflection/actor_rows": 0.0}

        aux = self._prepare_reflection_aux_for_actor(aux, reference)
        mini_batch_size = int(
            self.config.algorithm.dynamic_memory.get("reflection_train_mini_batch_size", len(aux))
        )
        if mini_batch_size <= 0:
            mini_batch_size = len(aux)

        raw_rows = len(aux)
        chunks = self._iter_batch_chunks(aux, mini_batch_size)
        actor_metric_lists = {}
        padded_rows = 0
        with _timer("update_reflection_actor", timing_raw):
            for chunk in chunks:
                chunk = self._pad_batch_to_divisor(chunk, self.actor_rollout_wg.world_size)
                chunk.meta_info["global_token_num"] = torch.sum(
                    chunk.batch["attention_mask"],
                    dim=-1,
                ).tolist()
                padded_rows += len(chunk)
                actor_output = self.actor_rollout_wg.update_actor(chunk)
                self._extend_metric_lists(actor_metric_lists, actor_output.meta_info["metrics"])

        metrics = {
            "analysis/memory/reflection/actor_rows": float(raw_rows),
            "analysis/memory/reflection/actor_padded_rows": float(padded_rows),
            "analysis/memory/reflection/update_chunks": float(len(chunks)),
        }
        actor_output_metrics = reduce_metrics(actor_metric_lists) if actor_metric_lists else {}
        metrics.update({f"analysis/memory/reflection/{key}": value for key, value in actor_output_metrics.items()})
        return metrics

    @staticmethod
    def _concat_interleaved_batches(original: DataProto, aux: DataProto) -> DataProto:
        if aux is None or len(aux) == 0:
            return original
        if original is None or len(original) == 0:
            return aux

        combined = UCOBRayTrainer._concat_aligned_batches(original, aux)
        original_len = len(original)
        aux_len = len(aux)
        total_len = original_len + aux_len
        indices = []
        original_idx = 0
        aux_idx = 0
        for pos in range(total_len):
            expected_aux = ((pos + 1) * aux_len) // total_len
            if aux_idx < expected_aux:
                indices.append(original_len + aux_idx)
                aux_idx += 1
            elif original_idx < original_len:
                indices.append(original_idx)
                original_idx += 1
            else:
                indices.append(original_len + aux_idx)
                aux_idx += 1
        return combined.select_idxs(indices)

    def _pad_actor_batch_to_divisor(self, batch: DataProto) -> DataProto:
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        if "multi_modal_inputs" in batch.non_tensor_batch:
            size_divisor = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        else:
            size_divisor = int(self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu) * int(world_size)
        if size_divisor <= 1:
            return batch
        remainder = len(batch) % size_divisor
        if remainder == 0:
            return batch
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(len(batch), to_add, replace=to_add > len(batch))
        dup_proto = batch.select_idxs(dup_indices)
        return DataProto.concat([batch, dup_proto])

    def _apply_adaptive_dsdar_weights(self, batch: DataProto, metrics: dict) -> None:
        dsdar_cfg = self.config.algorithm.get("dsdar", {})
        if not _as_bool(dsdar_cfg.get("adaptive_enable", False)):
            return
        if "dsdar_rollout_mode" not in batch.non_tensor_batch or "uid" not in batch.non_tensor_batch:
            return
        reward_key = "trajectory_episode_rewards" if "trajectory_episode_rewards" in batch.non_tensor_batch else "episode_rewards"
        if reward_key not in batch.non_tensor_batch or "traj_uid" not in batch.non_tensor_batch:
            return

        modes = np.asarray(batch.non_tensor_batch["dsdar_rollout_mode"], dtype=object)
        uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
        traj_uids = np.asarray(batch.non_tensor_batch["traj_uid"], dtype=object)
        memory_branches = np.asarray(
            batch.non_tensor_batch.get("dynamic_memory_branch", np.array([""] * len(modes), dtype=object)),
            dtype=object,
        )
        episode_rewards = np.asarray(batch.non_tensor_batch[reward_key], dtype=np.float32)
        bs = len(modes)

        reward_margin_eps = float(dsdar_cfg.get("reward_margin_eps", 0.0))
        bad_reward_threshold = DSDAR_BAD_TRAJ_REWARD_THRESHOLD
        success_reward_threshold = DSDAR_SUCCESS_REWARD_THRESHOLD
        rlrt_enable = _as_bool(dsdar_cfg.get("rlrt_enable", False), default=False)
        correct_coef = float(dsdar_cfg.get("correct_coef", 1.0))
        rlrt_coef = float(dsdar_cfg.get("rlrt_coef", 1.0))
        force_skill_from_no_skill = _as_bool(dsdar_cfg.get("force_skill_from_no_skill", False), default=False)
        bad_memory_unidirectional = _as_bool(dsdar_cfg.get("bad_memory_unidirectional", True), default=True)
        is_bad_memory_counterfactual = np.any(memory_branches == "bad") and np.any(memory_branches == "good")
        allow_bad_to_good = force_skill_from_no_skill or not (
            bad_memory_unidirectional and is_bad_memory_counterfactual
        )
        aux_target_branch = str(dsdar_cfg.get("aux_target_branch", "both")).strip().lower()
        aux_target_aliases = {
            "all": "both",
            "both": "both",
            "no_skill": "no_skill_only",
            "no-skill": "no_skill_only",
            "noskill": "no_skill_only",
            "no_skill_only": "no_skill_only",
            "skill": "skill_only",
            "skill_only": "skill_only",
        }
        aux_target_branch = aux_target_aliases.get(aux_target_branch, aux_target_branch)
        if aux_target_branch not in {"both", "no_skill_only", "skill_only"}:
            raise ValueError(
                f"Unsupported algorithm.dsdar.aux_target_branch={aux_target_branch!r}; "
                "choose from 'both', 'no_skill_only', or 'skill_only'."
            )

        def _branch_allowed(mode: str) -> bool:
            if aux_target_branch == "both":
                return True
            if aux_target_branch == "no_skill_only":
                return mode == "no_skill"
            return mode == "skill"

        row_weights = np.zeros(bs, dtype=np.float32)
        row_gate_signs = np.ones(bs, dtype=np.float32)
        directions = np.array(["none"] * bs, dtype=object)
        direction_ids = np.zeros(bs, dtype=np.int64)
        group_winners = {}

        traj_records = {}
        for i in range(bs):
            key = str(traj_uids[i])
            if key not in traj_records:
                traj_records[key] = {
                    "uid": str(uids[i]),
                    "mode": str(modes[i]),
                    "memory_branch": str(memory_branches[i]) if i < len(memory_branches) else "",
                    "reward": float(episode_rewards[i]),
                }

        all_no_skill_rewards = [rec["reward"] for rec in traj_records.values() if rec["mode"] == "no_skill"]
        all_skill_rewards = [rec["reward"] for rec in traj_records.values() if rec["mode"] == "skill"]
        all_bad_memory_rewards = [rec["reward"] for rec in traj_records.values() if rec.get("memory_branch") == "bad"]
        all_good_memory_rewards = [rec["reward"] for rec in traj_records.values() if rec.get("memory_branch") == "good"]

        for uid in sorted({rec["uid"] for rec in traj_records.values()}):
            group_records = [rec for rec in traj_records.values() if rec["uid"] == uid]
            no_skill_rewards = [rec["reward"] for rec in group_records if rec["mode"] == "no_skill"]
            skill_rewards = [rec["reward"] for rec in group_records if rec["mode"] == "skill"]
            if len(no_skill_rewards) == 0 or len(skill_rewards) == 0:
                group_winners[uid] = "tie"
                continue

            no_skill_mean = float(np.mean(no_skill_rewards))
            skill_mean = float(np.mean(skill_rewards))
            diff = skill_mean - no_skill_mean
            if diff > reward_margin_eps:
                group_winners[uid] = "skill"
            elif diff < -reward_margin_eps:
                group_winners[uid] = "no_skill"
            else:
                group_winners[uid] = "tie"

        for i in range(bs):
            winner = group_winners.get(str(uids[i]), "tie")
            mode = str(modes[i])
            reward = float(episode_rewards[i])
            is_bad_traj = reward <= bad_reward_threshold
            is_success_traj = reward >= success_reward_threshold
            if not _branch_allowed(mode):
                continue

            if force_skill_from_no_skill and mode == "skill":
                row_weights[i] = correct_coef
                directions[i] = "skill_from_no_skill"
                direction_ids[i] = 2
                row_gate_signs[i] = 1.0
            elif winner == "skill" and mode == "no_skill" and is_bad_traj:
                row_weights[i] = correct_coef
                directions[i] = "no_skill_from_skill"
                direction_ids[i] = 1
                row_gate_signs[i] = 1.0
            elif allow_bad_to_good and winner == "no_skill" and mode == "skill" and is_bad_traj:
                row_weights[i] = correct_coef
                directions[i] = "skill_from_no_skill"
                direction_ids[i] = 2
                row_gate_signs[i] = 1.0
            elif rlrt_enable and winner == "skill" and mode == "no_skill" and is_success_traj:
                row_weights[i] = rlrt_coef
                directions[i] = "no_skill_rlrt_from_skill"
                direction_ids[i] = 3
                row_gate_signs[i] = -1.0
            elif allow_bad_to_good and rlrt_enable and winner == "no_skill" and mode == "skill" and is_success_traj:
                row_weights[i] = rlrt_coef
                directions[i] = "skill_rlrt_from_no_skill"
                direction_ids[i] = 4
                row_gate_signs[i] = -1.0

        response_mask = batch.batch["response_mask"]
        weight_tensor = torch.from_numpy(row_weights).to(device=response_mask.device, dtype=response_mask.dtype)
        gate_sign_tensor = torch.from_numpy(row_gate_signs).to(device=response_mask.device, dtype=response_mask.dtype)
        direction_tensor = torch.from_numpy(direction_ids).to(device=response_mask.device)
        batch.batch["sdar_loss_weight"] = weight_tensor.unsqueeze(-1).expand_as(response_mask)
        batch.batch["sdar_direction_id"] = direction_tensor.unsqueeze(-1).expand_as(response_mask)
        batch.batch["sdar_gate_sign"] = gate_sign_tensor.unsqueeze(-1).expand_as(response_mask)
        batch.non_tensor_batch["dsdar_adaptive_direction"] = directions

        num_groups = max(len(group_winners), 1)
        skill_win = sum(1 for winner in group_winners.values() if winner == "skill")
        no_skill_win = sum(1 for winner in group_winners.values() if winner == "no_skill")
        tie = sum(1 for winner in group_winners.values() if winner == "tie")
        active = row_weights > 0
        correct_active = active & np.isin(direction_ids, [1, 2])
        rlrt_active = active & np.isin(direction_ids, [3, 4])
        no_skill_mask = modes == "no_skill"
        skill_mask = modes == "skill"

        metrics["dsdar/reward_no_skill_mean"] = float(np.mean(all_no_skill_rewards)) if all_no_skill_rewards else 0.0
        metrics["dsdar/reward_skill_mean"] = float(np.mean(all_skill_rewards)) if all_skill_rewards else 0.0
        metrics["dsdar/reward_gap_skill_minus_no_skill"] = (
            metrics["dsdar/reward_skill_mean"] - metrics["dsdar/reward_no_skill_mean"]
        )
        if all_bad_memory_rewards or all_good_memory_rewards:
            metrics["dsdar/reward_bad_memory_mean"] = (
                float(np.mean(all_bad_memory_rewards)) if all_bad_memory_rewards else 0.0
            )
            metrics["dsdar/reward_good_memory_mean"] = (
                float(np.mean(all_good_memory_rewards)) if all_good_memory_rewards else 0.0
            )
            metrics["dsdar/reward_gap_good_minus_bad_memory"] = (
                metrics["dsdar/reward_good_memory_mean"] - metrics["dsdar/reward_bad_memory_mean"]
            )
        metrics["dsdar/group_skill_win_ratio"] = skill_win / num_groups
        metrics["dsdar/group_no_skill_win_ratio"] = no_skill_win / num_groups
        metrics["dsdar/group_tie_ratio"] = tie / num_groups
        metrics["dsdar/correct_no_skill_from_skill_ratio"] = float(np.mean(directions == "no_skill_from_skill"))
        metrics["dsdar/correct_skill_from_no_skill_ratio"] = float(np.mean(directions == "skill_from_no_skill"))
        metrics["dsdar/rlrt_no_skill_from_skill_ratio"] = float(np.mean(directions == "no_skill_rlrt_from_skill"))
        metrics["dsdar/rlrt_skill_from_no_skill_ratio"] = float(np.mean(directions == "skill_rlrt_from_no_skill"))
        metrics["dsdar/bad_traj_no_skill_ratio"] = (
            float(np.mean(np.asarray(all_no_skill_rewards) <= bad_reward_threshold)) if all_no_skill_rewards else 0.0
        )
        metrics["dsdar/bad_traj_skill_ratio"] = (
            float(np.mean(np.asarray(all_skill_rewards) <= bad_reward_threshold)) if all_skill_rewards else 0.0
        )
        metrics["dsdar/success_traj_no_skill_ratio"] = (
            float(np.mean(np.asarray(all_no_skill_rewards) >= success_reward_threshold)) if all_no_skill_rewards else 0.0
        )
        metrics["dsdar/success_traj_skill_ratio"] = (
            float(np.mean(np.asarray(all_skill_rewards) >= success_reward_threshold)) if all_skill_rewards else 0.0
        )

        if "is_action_valid" in batch.non_tensor_batch:
            valid = np.asarray(batch.non_tensor_batch["is_action_valid"], dtype=bool)
            metrics["dsdar/invalid_no_skill_ratio"] = (
                float(np.mean(~valid[no_skill_mask])) if np.any(no_skill_mask) else 0.0
            )
            metrics["dsdar/invalid_skill_ratio"] = (
                float(np.mean(~valid[skill_mask])) if np.any(skill_mask) else 0.0
            )

    def _get_sdl_lambda(self, step: int) -> float:
        if self.sdl_warmdown_steps <= 0:
            return self.sdl_lambda
        if step >= self.sdl_warmdown_steps:
            return 0.0
        return self.sdl_lambda * (1.0 - step / self.sdl_warmdown_steps)

    def fit(self):
        """
        The training loop of UCOB. Identical to RLSD except:
        - Advantages are NOT replaced with token-level advantages
        - teacher_log_probs are passed through to the actor for SDL loss
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        if hasattr(self.traj_collector, "set_training_progress"):
                            self.traj_collector.set_training_progress(self.global_steps, self.total_training_steps)
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output
                    reflection_aux_batch = batch.meta_info.pop("dynamic_memory_reflection_train_batch", None)
                    batch, inline_reflection_aux_batch = self._split_dynamic_memory_reflection_batch(batch)
                    if reflection_aux_batch is None:
                        reflection_aux_batch = inline_reflection_aux_batch
                    elif inline_reflection_aux_batch is not None and len(inline_reflection_aux_batch) > 0:
                        reflection_aux_batch = self._concat_aligned_batches(
                            reflection_aux_batch,
                            inline_reflection_aux_batch,
                        )
                    metrics.update(self._reflection_aux_metrics(reflection_aux_batch))
                    batch, cbsd_aux_batch = self._split_ucob_cbsd_batch(batch)
                    metrics.update(self._cbsd_aux_analysis_metrics(cbsd_aux_batch, meta_info=batch.meta_info))
                    metrics["analysis/eff/cbsd_token_ratio"] = self._cbsd_token_fraction(batch, cbsd_aux_batch)
                    metrics.update(self._dynamic_memory_rollout_metrics(batch))
                    self._maybe_dump_dynamic_memory_snapshot(self.global_steps)

                    adv_estimator = self.config.algorithm.adv_estimator
                    adv_estimator = getattr(adv_estimator, "value", adv_estimator)
                    if str(adv_estimator).lower() in {"gigpo", "gagpo"}:
                        from gigpo import core_gigpo

                        batch.batch["step_rewards"] = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma,
                        )

                    batch = adjust_batch(self.config, batch)
                    if apply_credit_assignment_returns(batch, self.config):
                        metrics["env/credit_assignment_step_gamma"] = self.config.algorithm.get("step_gamma", 0.95)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if (
                        _as_bool(self.config.actor_rollout_ref.actor.get("action_loss_mask_enable", False))
                        and "loss_mask" in batch.batch.keys()
                    ):
                        response_mask = batch.batch["response_mask"]
                        response_length = response_mask.size(1)
                        action_mask = batch.batch["loss_mask"][:, -response_length:].to(
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        )
                        valid_tokens = response_mask.bool()
                        action_tokens = action_mask.bool() & valid_tokens
                        metrics["action_loss_mask/token_fraction"] = (
                            action_tokens.float().sum() / valid_tokens.float().sum().clamp_min(1.0)
                        ).item()
                        metrics["action_loss_mask/empty_row_fraction"] = (
                            ~action_tokens.any(dim=-1)
                        ).float().mean().item()
                        if "action_loss_mask_parse_success" in batch.non_tensor_batch:
                            metrics["action_loss_mask/parse_success_rate"] = np.asarray(
                                batch.non_tensor_batch["action_loss_mask_parse_success"],
                                dtype=np.float32,
                            ).mean().item()
                    if (
                        _as_bool(self.config.actor_rollout_ref.actor.get("sdar_topk_action_mask_enable", False))
                        and "action_token_mask" in batch.batch.keys()
                    ):
                        response_mask = batch.batch["response_mask"]
                        response_length = response_mask.size(1)
                        topk_action_mask = batch.batch["action_token_mask"][:, -response_length:].to(
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        )
                        valid_tokens = response_mask.bool()
                        action_tokens = topk_action_mask.bool() & valid_tokens
                        metrics["sdar/topk_action_mask_token_fraction"] = (
                            action_tokens.float().sum() / valid_tokens.float().sum().clamp_min(1.0)
                        ).item()
                        metrics["sdar/topk_action_mask_empty_row_fraction"] = (
                            ~action_tokens.any(dim=-1)
                        ).float().mean().item()
                        if "topk_action_mask_parse_success" in batch.non_tensor_batch:
                            metrics["sdar/topk_action_mask_parse_success_rate"] = np.asarray(
                                batch.non_tensor_batch["topk_action_mask_parse_success"],
                                dtype=np.float32,
                            ).mean().item()

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- UCOB/DSDAR: optional teacher forward pass ----
                    if self.teacher_loss_enable:
                        with _timer("teacher_forward", timing_raw):
                            teacher_log_probs = self._compute_teacher_log_probs(batch)
                            batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        self._apply_adaptive_dsdar_weights(batch, metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            gigpo_gae_lambda=self.config.algorithm.gigpo.get("gae_lambda", 0.8),
                        )

                        # ---- UCOB: Do NOT replace advantages ----
                        # Advantages stay as standard GRPO sequence-level advantages.
                        # The SDL loss is computed inside dp_actor.update_policy().

                        if "teacher_log_probs" in batch.batch.keys():
                            # Log SDAR gate-delta metrics. In dual-view DSDAR this
                            # may be reference-target or target-reference depending
                            # on sdar_gate_sign.
                            response_mask = batch.batch["response_mask"]
                            target_log_probs = batch.batch["old_log_probs"]
                            reference_log_probs = batch.batch["teacher_log_probs"]
                            gate_delta = reference_log_probs - target_log_probs
                            if "sdar_gate_sign" in batch.batch.keys():
                                gate_sign = batch.batch["sdar_gate_sign"]
                                if gate_sign.dim() == 1:
                                    gate_sign = gate_sign.unsqueeze(-1)
                                gate_delta = gate_delta * gate_sign.to(device=gate_delta.device, dtype=gate_delta.dtype)
                            gate_delta = gate_delta * response_mask
                            current_sdl_lambda = self._get_sdl_lambda(self.global_steps)
                            gate_delta_mean = masked_mean(gate_delta, response_mask).item()
                            gate_delta_std = masked_mean(gate_delta ** 2, response_mask).sqrt().item()
                            metrics["skillsd/gate_delta_mean"] = gate_delta_mean
                            metrics["skillsd/gate_delta_std"] = gate_delta_std
                            metrics["skillsd/teacher_student_gap_mean"] = gate_delta_mean
                            metrics["skillsd/teacher_student_gap_std"] = gate_delta_std
                            metrics["skillsd/sdl_lambda"] = current_sdl_lambda
                            if "dsdar_adaptive_direction" in batch.non_tensor_batch:
                                directions = np.asarray(batch.non_tensor_batch["dsdar_adaptive_direction"], dtype=object)
                                device = response_mask.device

                                def _direction_mean(direction_name: str) -> float:
                                    row_mask_np = directions == direction_name
                                    if not np.any(row_mask_np):
                                        return 0.0
                                    row_mask = torch.from_numpy(row_mask_np).to(device=device).unsqueeze(-1)
                                    token_mask = response_mask * row_mask.to(dtype=response_mask.dtype)
                                    if token_mask.sum().item() == 0:
                                        return 0.0
                                    return masked_mean(gate_delta, token_mask).item()

                                metrics["dsdar/gate_delta_no_skill_from_skill_mean"] = _direction_mean("no_skill_from_skill")
                                metrics["dsdar/gate_delta_skill_from_no_skill_mean"] = _direction_mean("skill_from_no_skill")
                                metrics["dsdar/gate_delta_no_skill_rlrt_from_skill_mean"] = _direction_mean("no_skill_rlrt_from_skill")
                                metrics["dsdar/gate_delta_skill_rlrt_from_no_skill_mean"] = _direction_mean("skill_rlrt_from_no_skill")
                                if "sdar_loss_weight" in batch.batch.keys():
                                    active_mask = response_mask * (batch.batch["sdar_loss_weight"] > 0).to(dtype=response_mask.dtype)
                                    if active_mask.sum().item() > 0:
                                        metrics["dsdar/gate_delta_active_mean"] = masked_mean(gate_delta, active_mask).item()
                                    else:
                                        metrics["dsdar/gate_delta_active_mean"] = 0.0

                            # Save per-token gap data if SAVE_SDAR_DEBUG=1, at test_freq interval
                            if os.environ.get("SAVE_SDAR_DEBUG", "0") == "1" and \
                                    self.config.trainer.test_freq > 0 and \
                                    self.global_steps % self.config.trainer.test_freq == 0:
                                save_dir = os.environ.get(
                                    "SAVE_SDAR_DEBUG_DIR",
                                    "outputs/sdar_debug"
                                )
                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")

                                bs = response_mask.shape[0]
                                turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
                                traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([""] * bs, dtype=object))
                                episode_rewards_arr = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
                                episode_lengths_arr = batch.non_tensor_batch.get("episode_lengths", np.zeros(bs, dtype=object))
                                response_ids = batch.batch["responses"]

                                with open(save_path, "w") as f:
                                    for i in range(bs):
                                        mask_i = response_mask[i].bool()
                                        valid_count = mask_i.sum().item()
                                        if valid_count == 0:
                                            continue
                                        token_ids_i = response_ids[i][mask_i].cpu().tolist()
                                        tokens_i = [self.tokenizer.decode([tid]) for tid in token_ids_i]
                                        gaps_i = gate_delta[i][mask_i].cpu().tolist()
                                        teacher_lps_i = reference_log_probs[i][mask_i].cpu().tolist()
                                        student_lps_i = target_log_probs[i][mask_i].cpu().tolist()

                                        record = {
                                            "global_step": self.global_steps,
                                            "sample_idx": i,
                                            "turn_step": int(turn_steps[i]),
                                            "traj_uid": str(traj_uids[i]),
                                            "episode_reward": float(episode_rewards_arr[i]),
                                            "episode_length": float(episode_lengths_arr[i]),
                                            "tokens": tokens_i,
                                            "token_ids": token_ids_i,
                                            "gaps": gaps_i,
                                            "teacher_log_probs": teacher_lps_i,
                                            "student_log_probs": student_lps_i,
                                        }
                                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                print(f"[SDAR Debug] Saved per-token gap data to {save_path} ({bs} samples)")

                    rollout_metrics_batch = batch
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            if cbsd_aux_batch is not None and len(cbsd_aux_batch) > 0:
                                cbsd_aux_batch = self._prepare_cbsd_aux_for_actor(cbsd_aux_batch, batch)
                                batch = self._concat_interleaved_batches(batch, cbsd_aux_batch)
                                batch = self._pad_actor_batch_to_divisor(batch)
                                batch.meta_info["global_token_num"] = torch.sum(
                                    batch.batch["attention_mask"],
                                    dim=-1,
                                ).tolist()
                                metrics["analysis/eff/cbsd_actor_row_ratio"] = float(len(cbsd_aux_batch) / max(len(batch), 1))
                            else:
                                metrics["analysis/eff/cbsd_actor_row_ratio"] = 0.0
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if reflection_aux_batch is not None and len(reflection_aux_batch) > 0:
                            reflection_metrics = self._update_reflection_actor(
                                reflection_aux_batch,
                                rollout_metrics_batch,
                                timing_raw,
                            )
                            metrics.update(reflection_metrics)
                        else:
                            metrics["analysis/memory/reflection/actor_rows"] = 0.0

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(rollout_metrics_batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(rollout_metrics_batch.batch["responses"], skip_special_tokens=True)
                            scores = rollout_metrics_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=rollout_metrics_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=rollout_metrics_batch, timing_raw=timing_raw))
                metrics.update(self._efficiency_overhead_metrics(metrics, timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=rollout_metrics_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
