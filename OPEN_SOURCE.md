# Open-Source Upload Plan

This repository should be published as a focused UCOB code release rather than
as a snapshot of the full local experiment workspace.

## Upload

- `README.md`, `LICENSE`, `Notice.txt`
- `pyproject.toml`, `setup.py`, `requirements*.txt`, `environment.yml`
- `.gitignore`, `.pre-commit-config.yaml`, `.readthedocs.yaml`
- `env.example.sh`
- `verl/`
- `agent_system/`
  - UCOB keeps the ALFWorld, WebShop, and Search environment packages.
  - Non-main-table environment packages such as AppWorld, Sokoban, and Gym
    Cards are excluded from the public upload.
- `gigpo/`
- `skills/`
- `examples/ucob_trainer/`
- `examples/search/`
- `examples/data_preprocess/`
- `figs/`
- `tests/`
- `docker/`
- `scripts/model_merger.py`
- `scripts/diagnose.py`
- `scripts/converter_hf_to_mcore.py`

## Do Not Upload

- `env.sh` or any file containing local API keys
- `checkpoints/`, `outputs/`, `swanlog/`, `wandb/`, `experiments/`
- `models/`, `data/`, `dataset/`, `references/`, `ICLR_2027/`
- old/deprecated trainer launchers outside `examples/ucob_trainer/`
- local tmux/cluster helper scripts under `scripts/*.sh`
- original veRL GitHub workflows under `.github/workflows/`
- platform-specific binaries such as WebShop `chromedriver`

## Notes

- Datasets, model weights, Search-R1 indexes, and generated skill memories should
  be downloaded or produced by users locally.
- `env.example.sh` is the sanitized template for local paths and optional API
  credentials. Users should copy it to `env.sh`.
- `algorithm.gigpo` and `algorithm.dsdar` names still appear in some internal
  compatibility paths. The public launchers and documentation expose the method
  as UCOB.
