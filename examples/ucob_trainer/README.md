# UCOB Full Scripts

This directory keeps the nine full UCOB launchers used for the main table. Each
script is intentionally explicit: model path, batch sizes, UCOB/CBSD knobs,
skill-memory settings, and trainer overrides are written in the script itself.

Deprecated legacy, ablation, and debug scripts are archived under
`examples/deprecated/`.

## Main Table Scripts

```bash
bash examples/ucob_trainer/run_alfworld_qwen3_1.7b_full.sh
bash examples/ucob_trainer/run_webshop_qwen3_1.7b_full.sh
bash examples/ucob_trainer/run_search_qwen3_1.7b_full.sh

bash examples/ucob_trainer/run_alfworld_qwen25_3b_full.sh
bash examples/ucob_trainer/run_webshop_qwen25_3b_full.sh
bash examples/ucob_trainer/run_search_qwen25_3b_full.sh

bash examples/ucob_trainer/run_alfworld_qwen25_7b_full.sh
bash examples/ucob_trainer/run_webshop_qwen25_7b_full.sh
bash examples/ucob_trainer/run_search_qwen25_7b_full.sh
```

The base defaults live in `verl/trainer/config/ucob_trainer.yaml`.
Search-specific defaults live in `verl/trainer/config/ucob_search_trainer.yaml`,
but the launch scripts still spell out the main training overrides for easy
paper-run auditing.

## Editing Style

For a normal run, edit the variable block near the top of the target script:

- `MODEL_PATH`
- `train_data_size`, `val_data_size`, `group_size`
- `cbsd_*`
- `skillmem_*`
- `reflection_*`
- `experiment_name`
- `skill_memory_file`

Extra Hydra overrides can still be appended after the script command because
each launcher ends with `"$@"`.

## Case Study Debug

The paper-only qualitative logger is off by default. Enable it with:

```bash
UCOB_CASE_STUDY_DEBUG=1 \
UCOB_CASE_STUDY_DEBUG_DIR=outputs/case_study_debug/alfworld \
bash examples/ucob_trainer/run_alfworld_qwen3_1.7b_full.sh
```

Equivalent config fields are under `algorithm.analysis.case_study_debug_*`.
