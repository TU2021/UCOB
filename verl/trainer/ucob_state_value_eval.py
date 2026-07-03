"""Evaluation-only turn-level value-gap collection for UCOB motivation analysis."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import defaultdict
from typing import Any

import numpy as np


def stringify_anchor(anchor: Any) -> str:
    if anchor is None:
        return ""
    if isinstance(anchor, bytes):
        return anchor.decode("utf-8", errors="ignore")
    if isinstance(anchor, (list, tuple)):
        return " ".join(str(item) for item in anchor)
    return str(anchor)


def anchor_obs_hash(anchor_obs: str) -> str:
    if not anchor_obs:
        return ""
    return hashlib.sha256(anchor_obs.encode("utf-8")).hexdigest()[:16]


def _reward_value(raw: Any) -> float:
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def extract_turn_rows_from_rollout(
    total_batch_list,
    step_returns_by_traj,
    episode_rewards,
    terminal_success,
    *,
    eval_task_keys,
    sample_indices,
    rollout_mode: str,
    run_id: str,
    traj_uids=None,
) -> list[dict]:
    """Extract per-turn rollout records for one skill or no-skill pass."""
    records: list[dict] = []
    rollout_mode = str(rollout_mode).strip().lower()
    if rollout_mode not in {"skill", "no_skill"}:
        raise ValueError(f"Unsupported rollout_mode={rollout_mode!r}; expected 'skill' or 'no_skill'.")

    for traj_idx, steps in enumerate(total_batch_list):
        if traj_idx >= len(step_returns_by_traj):
            continue
        returns = step_returns_by_traj[traj_idx]
        trajectory_return = _reward_value(episode_rewards[traj_idx])
        success = _reward_value(terminal_success[traj_idx]) if terminal_success is not None else float("nan")
        uid = str(eval_task_keys[traj_idx])
        sample_idx = int(sample_indices[traj_idx]) if sample_indices is not None else 0
        traj_uid = (
            str(traj_uids[traj_idx])
            if traj_uids is not None and traj_idx < len(traj_uids)
            else ""
        )

        for step_idx, step in enumerate(steps):
            if step_idx >= len(returns):
                continue
            if not bool(step.get("active_masks", True)):
                continue

            turn_step = step.get("turn_step", step_idx)
            try:
                turn_step = int(turn_step)
            except (TypeError, ValueError):
                turn_step = int(step_idx)

            anchor_obs = stringify_anchor(step.get("anchor_obs", ""))
            immediate_reward = _reward_value(step.get("rewards", float("nan")))

            records.append(
                {
                    "run_id": run_id,
                    "uid": uid,
                    "traj_uid": traj_uid,
                    "sample_idx": sample_idx,
                    "turn_step": turn_step,
                    "rollout_mode": rollout_mode,
                    "anchor_obs": anchor_obs,
                    "anchor_obs_hash": anchor_obs_hash(anchor_obs),
                    "step_return": float(returns[step_idx]),
                    "trajectory_return": trajectory_return,
                    "immediate_reward": immediate_reward,
                    "success": success,
                }
            )
    return records


def aggregate_turn_summary(
    rows: list[dict],
    *,
    run_id: str,
    margin: float = 0.05,
) -> list[dict]:
    """Aggregate uid+turn pairs, then summarize by turn_step."""
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"skill": [], "no_skill": []}
    )
    for row in rows:
        grouped[(row["uid"], int(row["turn_step"]))][row["rollout_mode"]].append(float(row["step_return"]))

    uid_turn_pairs: list[dict] = []
    turn_pair_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "valid": 0})
    for (uid, turn_step), bucket in grouped.items():
        turn_pair_counts[int(turn_step)]["total"] += 1
        skill_values = bucket["skill"]
        no_skill_values = bucket["no_skill"]
        if not skill_values or not no_skill_values:
            continue
        turn_pair_counts[int(turn_step)]["valid"] += 1
        skill_mean = float(np.mean(skill_values))
        no_skill_mean = float(np.mean(no_skill_values))
        gap = skill_mean - no_skill_mean
        uid_turn_pairs.append(
            {
                "uid": uid,
                "turn_step": int(turn_step),
                "skill_mean": skill_mean,
                "no_skill_mean": no_skill_mean,
                "gap": gap,
            }
        )

    by_turn: dict[int, list[dict]] = defaultdict(list)
    for pair in uid_turn_pairs:
        by_turn[int(pair["turn_step"])].append(pair)

    summary_rows: list[dict] = []
    for turn_step in sorted(turn_pair_counts.keys()):
        pairs = by_turn.get(turn_step, [])
        counts = turn_pair_counts[turn_step]
        if pairs:
            gaps = np.asarray([pair["gap"] for pair in pairs], dtype=np.float64)
            skill_means = np.asarray([pair["skill_mean"] for pair in pairs], dtype=np.float64)
            no_skill_means = np.asarray([pair["no_skill_mean"] for pair in pairs], dtype=np.float64)
            abs_gaps = np.abs(gaps)
            summary_rows.append(
                {
                    "run_id": run_id,
                    "turn_step": int(turn_step),
                    "num_state_pairs": int(counts["total"]),
                    "num_valid_pairs": int(counts["valid"]),
                    "mean_value_skill": float(skill_means.mean()),
                    "mean_value_no_skill": float(no_skill_means.mean()),
                    "mean_gap_skill_minus_no_skill": float(gaps.mean()),
                    "mean_abs_gap_skill_minus_no_skill": float(abs_gaps.mean()),
                    "skill_better_ratio": float(np.mean(gaps > margin)),
                    "no_skill_better_ratio": float(np.mean(gaps < -margin)),
                    "tie_ratio": float(np.mean(abs_gaps <= margin)),
                }
            )
        else:
            summary_rows.append(
                {
                    "run_id": run_id,
                    "turn_step": int(turn_step),
                    "num_state_pairs": int(counts["total"]),
                    "num_valid_pairs": 0,
                    "mean_value_skill": float("nan"),
                    "mean_value_no_skill": float("nan"),
                    "mean_gap_skill_minus_no_skill": float("nan"),
                    "mean_abs_gap_skill_minus_no_skill": float("nan"),
                    "skill_better_ratio": float("nan"),
                    "no_skill_better_ratio": float("nan"),
                    "tie_ratio": float("nan"),
                }
            )
    return summary_rows


def aggregate_traj_summary(rows: list[dict], *, run_id: str, margin: float = 0.05) -> list[dict]:
    """Pair skill/no-skill trajectories by uid + sample_idx."""
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
        lambda: {"skill": [], "no_skill": []}
    )
    for row in rows:
        key = (row["uid"], int(row["sample_idx"]))
        grouped[key][row["rollout_mode"]].append(row)

    summary_rows: list[dict] = []
    for (uid, sample_idx), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        skill_rows = bucket["skill"]
        no_skill_rows = bucket["no_skill"]
        if not skill_rows or not no_skill_rows:
            continue

        skill_row = skill_rows[0]
        no_skill_row = no_skill_rows[0]
        traj_return_skill = float(skill_row["trajectory_return"])
        traj_return_no_skill = float(no_skill_row["trajectory_return"])
        gap = traj_return_skill - traj_return_no_skill
        if gap > margin:
            teacher = "skill"
        elif gap < -margin:
            teacher = "no_skill"
        else:
            teacher = "tie"

        summary_rows.append(
            {
                "run_id": run_id,
                "uid": uid,
                "sample_idx": int(sample_idx),
                "traj_uid_skill": str(skill_row.get("traj_uid", "")),
                "traj_uid_no_skill": str(no_skill_row.get("traj_uid", "")),
                "trajectory_return_skill": traj_return_skill,
                "trajectory_return_no_skill": traj_return_no_skill,
                "trajectory_gap_skill_minus_no_skill": gap,
                "trajectory_teacher": teacher,
                "success_skill": float(skill_row.get("success", float("nan"))),
                "success_no_skill": float(no_skill_row.get("success", float("nan"))),
            }
        )
    return summary_rows


TURN_ROWS_FIELDNAMES = [
    "run_id",
    "uid",
    "traj_uid",
    "sample_idx",
    "turn_step",
    "rollout_mode",
    "anchor_obs",
    "anchor_obs_hash",
    "step_return",
    "trajectory_return",
    "immediate_reward",
    "success",
]

TURN_SUMMARY_FIELDNAMES = [
    "run_id",
    "turn_step",
    "num_state_pairs",
    "num_valid_pairs",
    "mean_value_skill",
    "mean_value_no_skill",
    "mean_gap_skill_minus_no_skill",
    "mean_abs_gap_skill_minus_no_skill",
    "skill_better_ratio",
    "no_skill_better_ratio",
    "tie_ratio",
]

TRAJ_SUMMARY_FIELDNAMES = [
    "run_id",
    "uid",
    "sample_idx",
    "traj_uid_skill",
    "traj_uid_no_skill",
    "trajectory_return_skill",
    "trajectory_return_no_skill",
    "trajectory_gap_skill_minus_no_skill",
    "trajectory_teacher",
    "success_skill",
    "success_no_skill",
]


def write_csv(rows: list[dict], output_csv: str, fieldnames: list[str]) -> str:
    output_csv = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_csv


def write_motivation_csvs(
    turn_rows: list[dict],
    *,
    run_id: str,
    output_dir: str,
    gap_margin: float = 0.05,
) -> dict[str, str | int]:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    turn_rows_csv = write_csv(
        turn_rows,
        os.path.join(output_dir, "motivation_turn_rows.csv"),
        TURN_ROWS_FIELDNAMES,
    )
    turn_summary_rows = aggregate_turn_summary(turn_rows, run_id=run_id, margin=gap_margin)
    turn_summary_csv = write_csv(
        turn_summary_rows,
        os.path.join(output_dir, "motivation_turn_summary.csv"),
        TURN_SUMMARY_FIELDNAMES,
    )
    traj_summary_rows = aggregate_traj_summary(turn_rows, run_id=run_id, margin=gap_margin)
    traj_summary_csv = write_csv(
        traj_summary_rows,
        os.path.join(output_dir, "motivation_traj_summary.csv"),
        TRAJ_SUMMARY_FIELDNAMES,
    )

    readme_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "motivation_eval", "README.md")
    )
    readme_dst = os.path.join(output_dir, "README.md")
    if os.path.isfile(readme_src):
        shutil.copy2(readme_src, readme_dst)

    return {
        "turn_rows_csv": turn_rows_csv,
        "turn_summary_csv": turn_summary_csv,
        "traj_summary_csv": traj_summary_csv,
        "num_turn_summary_rows": len(turn_summary_rows),
        "num_traj_summary_rows": len(traj_summary_rows),
        "num_valid_turn_summary_rows": sum(
            1 for row in turn_summary_rows if int(row["num_valid_pairs"]) > 0
        ),
    }
