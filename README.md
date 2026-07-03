<h1 align="center">
UCOB: Learning to Utilize and Evolve Agentic Skills via Credit-Aware On-Policy Bidirectional Self-Distillation
</h1>

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/Method-UCOB-blue" alt="Method"/>
    <img src="https://img.shields.io/badge/Backbones-Qwen3%20%7C%20Qwen2.5-green" alt="Backbones"/>
    <img src="https://img.shields.io/badge/Tasks-ALFWorld%20%7C%20WebShop%20%7C%20Search--QA-orange" alt="Tasks"/>
  </p>
</div>

## Overview

**UCOB** learns to utilize and evolve agentic skills through **Credit-Aware On-Policy Bidirectional Self-Distillation**.
Retrieved skills can be useful in one state and misleading in another, so UCOB does not treat the skill-conditioned prompt as a fixed privileged teacher.
Instead, it builds two on-policy context views, compares skill and no-skill branches at the same task and anchor state, and lets the higher-return branch teach the other direction.

<div align="center">
  <img src="figs/image%201.png" alt="UCOB overview and results" style="width:95%;">
</div>

UCOB combines four pieces:

- **Dual-granularity skill memory**: task-level and state-level skills are retrieved together.
- **Mixed skill/no-skill rollouts**: the same online model explores with and without retrieved skills.
- **CBSD**: same-anchor return gaps select the local teacher direction, skill to no-skill or no-skill to skill.
- **Skill evolution**: rollout reflections write new skills, update utilities, and self-train the skill writer.

<div align="center">
  <img src="figs/image%204.png" alt="UCOB method" style="width:95%;">
</div>

On ALFWorld, WebShop, and Search-QA, UCOB improves over skill-free RL, skill-augmented agents, and self-distillation baselines across Qwen3-1.7B, Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct.
The gains are especially large on the compact Qwen3-1.7B backbone.

<div align="center">
  <img src="figs/image%205.png" alt="UCOB main results" style="width:95%;">
</div>

## News

- `2026-06-28`: Initial UCOB paper/code release draft.

## Installation

### Python Environment

```bash
conda create -n ucob python==3.12 -y
conda activate ucob

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

The training scripts use SwanLab by default:

```bash
export SWANLAB_API_KEY=your_key_here
```

The repository also uses `env.sh` to set local paths such as `VERL_AGENT_DATA_DIR`, `ALFWORLD_DATA`, `RAY_TMPDIR`, and Search-R1 index locations.
Create it from the sanitized template and adjust paths for your machine before launching long runs:

```bash
cp env.example.sh env.sh
```

### Supported Environments

#### ALFWorld

```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
alfworld-download -f
```

#### WebShop

WebShop is easiest to install in a Python 3.10 environment:

```bash
conda create -n ucob-webshop python==3.10 -y
conda activate ucob-webshop

cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

Dependency warnings from `spacy`/`weasel` about `typer` can be ignored for the WebShop runs used here.

#### Search-QA

```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2

cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```

Search-QA uses a local retrieval server. Build the retriever environment:

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

Download the Search-R1 index:

```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

The UCOB Search scripts check whether the retriever is alive and start `examples/search/retriever/retrieval_launch.sh` automatically when needed.
You can also start it manually:

```bash
conda activate retriever
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 2>&1
```

## Training

Full UCOB launchers are under `examples/ucob_trainer/`.
Each script is intentionally explicit: model path, batch sizes, CBSD knobs, skill-memory settings, reflection self-training, and trainer overrides are visible in the script itself.

Before running, edit the `MODEL_PATH` near the top of the script to point to your local model checkpoint.
For the Qwen3-1.7B runs:

```bash
bash examples/ucob_trainer/run_alfworld_qwen3_1.7b_full.sh
bash examples/ucob_trainer/run_webshop_qwen3_1.7b_full.sh
bash examples/ucob_trainer/run_search_qwen3_1.7b_full.sh
```

The main-table launchers are:

```bash
# Qwen3-1.7B
bash examples/ucob_trainer/run_alfworld_qwen3_1.7b_full.sh
bash examples/ucob_trainer/run_webshop_qwen3_1.7b_full.sh
bash examples/ucob_trainer/run_search_qwen3_1.7b_full.sh

# Qwen2.5-3B-Instruct
bash examples/ucob_trainer/run_alfworld_qwen25_3b_full.sh
bash examples/ucob_trainer/run_webshop_qwen25_3b_full.sh
bash examples/ucob_trainer/run_search_qwen25_3b_full.sh

# Qwen2.5-7B-Instruct
bash examples/ucob_trainer/run_alfworld_qwen25_7b_full.sh
bash examples/ucob_trainer/run_webshop_qwen25_7b_full.sh
bash examples/ucob_trainer/run_search_qwen25_7b_full.sh
```

Core config files:

- `verl/trainer/config/ucob_trainer.yaml`
- `verl/trainer/config/ucob_search_trainer.yaml`

Main training entry point:

```bash
python3 -m verl.trainer.main_ucob
```

Typical outputs:

- Checkpoints: `checkpoints/UCOB_<env>/<experiment_name>/global_step_150`
- Skill memory: `outputs/dynamic_memory/<experiment_name>/<env>_reflections.json`
- Search retriever logs: `outputs/logs/search_retriever/`

## Citation

If you find this project useful, please cite UCOB.

```bibtex
@article{tu2026ucob,
  title={UCOB: Learning to Utilize and Evolve Agentic Skills via Credit-Aware On-Policy Bidirectional Self-Distillation},
  author={Tu, Songjun and Xu, Chengdong and Zhang, Qichao and Ma, Yiwen and Zhang, Yaocheng and Li, Linjing and Li, Dong and Lan, Xiangyuan and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2606.29502},
  year={2026}
}
```

## Acknowledgement

This project builds on [verl-agent](https://github.com/langfengQ/verl-agent), [veRL](https://github.com/volcengine/verl), [ALFWorld](https://github.com/alfworld/alfworld), [WebShop](https://github.com/princeton-nlp/WebShop), [SkillRL](https://github.com/aiming-lab/SkillRL), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), and [SDAR](https://github.com/ZJU-REAL/SDAR). We thank the authors of those projects.
