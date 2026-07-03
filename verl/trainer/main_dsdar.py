"""Backward-compatible entry point for legacy DSDAR imports.

New UCOB runs should use ``verl.trainer.main_ucob``.
"""

from verl.trainer.main_ucob import UCOBTaskRunner, UCOBTrajectoryCollector, main, run_ucob

run_dsdar = run_ucob
DSDARTaskRunner = UCOBTaskRunner
DSDARTrajectoryCollector = UCOBTrajectoryCollector


if __name__ == "__main__":
    main()
