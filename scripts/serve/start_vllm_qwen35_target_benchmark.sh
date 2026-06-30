#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi

source "$SCRIPT_DIR/../../.venv/bin/activate"
hash -r

VENV_NVIDIA_LIBS="$(python - <<'PY'
from pathlib import Path
import site

for site_dir in site.getsitepackages():
    nvidia_dir = Path(site_dir) / "nvidia"
    if not nvidia_dir.exists():
        continue
    for lib_dir in sorted(nvidia_dir.glob("cu*/lib")):
        if lib_dir.exists():
            print(lib_dir)
            raise SystemExit
PY
)"
if [[ -n "${VENV_NVIDIA_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-${PIPELINE_PARALLEL_SIZE:-1}}"
DP_SIZE="${DP_SIZE:-${DATA_PARALLEL_SIZE:-1}}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

cmd=("vllm" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --data-parallel-size "$DP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --attention-backend flashinfer \
  --enable-auto-tool-choice)

if [[ -n "$MAX_BATCHED_TOKENS" ]]; then
  cmd+=(--max-num-batched-tokens "$MAX_BATCHED_TOKENS")
fi

if [[ -n "$MAX_NUM_SEQS" ]]; then
  cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "$TOOL_CALL_PARSER" && "$TOOL_CALL_PARSER" != "none" ]]; then
  cmd+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" && "$REASONING_PARSER" != "none" ]]; then
  cmd+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ -n "$TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS" || -n "$TARGET_YARN_FACTOR" || -n "$TARGET_YARN_MAX_POSITION_EMBEDDINGS" ]]; then
  export TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS
  export TARGET_YARN_MAX_POSITION_EMBEDDINGS
  export TARGET_YARN_FACTOR
  if [[ -z "$TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS" ]]; then
    echo "TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS is required when enabling target YaRN" >&2
    exit 1
  fi
  if [[ -z "$TARGET_YARN_FACTOR" ]]; then
    if [[ -z "$TARGET_YARN_MAX_POSITION_EMBEDDINGS" ]]; then
      echo "TARGET_YARN_FACTOR or TARGET_YARN_MAX_POSITION_EMBEDDINGS is required when enabling target YaRN" >&2
      exit 1
    fi
    TARGET_YARN_FACTOR=$(python - <<'PY'
import os
print(float(os.environ["TARGET_YARN_MAX_POSITION_EMBEDDINGS"]) / float(os.environ["TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"]))
PY
)
    export TARGET_YARN_FACTOR
  fi
  if [[ -z "$TARGET_YARN_MAX_POSITION_EMBEDDINGS" ]]; then
    TARGET_YARN_MAX_POSITION_EMBEDDINGS=$(python - <<'PY'
import math
import os
print(int(math.ceil(float(os.environ["TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"]) * float(os.environ["TARGET_YARN_FACTOR"]))))
PY
)
    export TARGET_YARN_MAX_POSITION_EMBEDDINGS
  fi
  HF_OVERRIDES=$(python - <<'PY'
import json
import os
print(json.dumps({
    "max_position_embeddings": int(os.environ["TARGET_YARN_MAX_POSITION_EMBEDDINGS"]),
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": float(os.environ["TARGET_YARN_FACTOR"]),
        "original_max_position_embeddings": int(os.environ["TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"]),
    },
}, separators=(",", ":")))
PY
)
  cmd+=(--hf-overrides "$HF_OVERRIDES")
fi

if [[ -n "$ENABLE_CHUNKED_PREFILL" && "$ENABLE_CHUNKED_PREFILL" != "default" ]]; then
  if [[ "$ENABLE_CHUNKED_PREFILL" != "0" ]]; then
    cmd+=(--enable-chunked-prefill)
  else
    cmd+=(--no-enable-chunked-prefill)
  fi
fi
if [[ -n "$ENABLE_PREFIX_CACHING" && "$ENABLE_PREFIX_CACHING" != "default" ]]; then
  if [[ "$ENABLE_PREFIX_CACHING" != "0" ]]; then
    cmd+=(--enable-prefix-caching)
  else
    cmd+=(--no-enable-prefix-caching)
  fi
fi
if [[ -n "$ENFORCE_EAGER" && "$ENFORCE_EAGER" != "0" ]]; then
  cmd+=(--enforce-eager)
fi
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $EXTRA_VLLM_ARGS )
  cmd+=("${extra_args[@]}")
fi

exec "${cmd[@]}"
