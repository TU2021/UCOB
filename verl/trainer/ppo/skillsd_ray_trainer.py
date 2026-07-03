"""Backward-compatible wrapper for the renamed UCOB trainer."""

from verl.trainer.ppo.ucob_ray_trainer import *  # noqa: F401,F403
from verl.trainer.ppo.ucob_ray_trainer import UCOBRayTrainer

SkillSDRayTrainer = UCOBRayTrainer
