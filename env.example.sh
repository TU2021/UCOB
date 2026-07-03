#!/bin/bash
# Sanitized runtime environment template for UCOB.
# Copy this file to env.sh and fill in local paths/API keys before running scripts.

export REAL_HOME="${REAL_HOME:-${HOME}/ucob_data}"
export CONDA_ROOT="${CONDA_ROOT:-${REAL_HOME}/miniconda3}"
export UCOB_ROOT="${UCOB_ROOT:-$(pwd)}"
export SDAR_ROOT="${SDAR_ROOT:-${UCOB_ROOT}}"

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-swanlog}"
export SWANLAB_DISABLE_PROMPT=1
export SWANLAB_NON_INTERACTIVE=1

export RAY_worker_register_timeout_seconds=600

# External OpenAI-compatible API settings for UCOB skill-memory reflection.
# The backend switch itself is set in individual training scripts.
export DYNAMIC_MEMORY_REFLECTION_API_MODEL="${DYNAMIC_MEMORY_REFLECTION_API_MODEL:-}"
export DYNAMIC_MEMORY_REFLECTION_API_KEY="${DYNAMIC_MEMORY_REFLECTION_API_KEY:-}"
export DYNAMIC_MEMORY_REFLECTION_API_URL="${DYNAMIC_MEMORY_REFLECTION_API_URL:-}"
export DYNAMIC_MEMORY_REFLECTION_API_BASE_URL="${DYNAMIC_MEMORY_REFLECTION_API_BASE_URL:-}"
export DYNAMIC_MEMORY_REFLECTION_API_TIMEOUT="${DYNAMIC_MEMORY_REFLECTION_API_TIMEOUT:-300}"
export DYNAMIC_MEMORY_REFLECTION_API_MAX_WORKERS="${DYNAMIC_MEMORY_REFLECTION_API_MAX_WORKERS:-8}"
export DYNAMIC_MEMORY_REFLECTION_API_RETRIES="${DYNAMIC_MEMORY_REFLECTION_API_RETRIES:-2}"

export TMPDIR="${TMPDIR:-${REAL_HOME}/tmp}"
mkdir -p "$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-${REAL_HOME}/ray}"
mkdir -p "$RAY_TMPDIR"

# ALFWorld data root should contain json_2.1.1, logic/, detectors/.
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"

# verl-agent placeholder parquet directory used by ALFWorld/WebShop scripts.
export VERL_AGENT_DATA_DIR="${VERL_AGENT_DATA_DIR:-${HOME}/data/verl-agent}"

# Search-R1 data/index paths, only needed for search experiments.
export SEARCH_DATA_DIR="${SEARCH_DATA_DIR:-${HOME}/data/searchR1_processed_direct}"
export SEARCHR1_INDEX_DIR="${SEARCHR1_INDEX_DIR:-${HOME}/data/searchR1}"
