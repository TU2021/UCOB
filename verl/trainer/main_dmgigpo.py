"""Backward-compatible legacy entry point.

New runs should use ``python -m verl.trainer.main_ucob``.
"""

import hydra

from verl.trainer.main_ucob import run_ucob


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ucob(config)


if __name__ == "__main__":
    main()
