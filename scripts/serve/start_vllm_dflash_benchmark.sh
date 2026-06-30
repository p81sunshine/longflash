#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-z-lab/Qwen3-8B-DFlash-b16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-${PIPELINE_PARALLEL_SIZE:-1}}"
DP_SIZE="${DP_SIZE:-${DATA_PARALLEL_SIZE:-1}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
DFLASH_WINDOW_SIZE="${DFLASH_WINDOW_SIZE:-full}"
DFLASH_WINDOW_MODE="${DFLASH_WINDOW_MODE:-sink_recent_suffix}"
DFLASH_RECENT_TOKENS="${DFLASH_RECENT_TOKENS:-}"
DFLASH_SUFFIX_MATCH_TOKENS="${DFLASH_SUFFIX_MATCH_TOKENS:-}"
DFLASH_SUFFIX_KEEP_TOKENS="${DFLASH_SUFFIX_KEEP_TOKENS:-}"
DFLASH_SUFFIX_MIDDLE_BUDGET="${DFLASH_SUFFIX_MIDDLE_BUDGET:-}"
DFLASH_SUFFIX_DECODING="${DFLASH_SUFFIX_DECODING:-0}"
DFLASH_SUFFIX_MAX_QUERY_LEN="${DFLASH_SUFFIX_MAX_QUERY_LEN:-16}"
DFLASH_SUFFIX_MIN_QUERY_LEN="${DFLASH_SUFFIX_MIN_QUERY_LEN:-2}"
DFLASH_SUFFIX_MAX_PREDICT_LEN="${DFLASH_SUFFIX_MAX_PREDICT_LEN:-}"
DFLASH_SUFFIX_ALPHA="${DFLASH_SUFFIX_ALPHA:-1}"
DFLASH_SUFFIX_MAX_SPEC_OFFSET="${DFLASH_SUFFIX_MAX_SPEC_OFFSET:-0}"
DFLASH_SUFFIX_MIN_TOKEN_PROB="${DFLASH_SUFFIX_MIN_TOKEN_PROB:-0}"
DFLASH_SUFFIX_THRESHOLD="${DFLASH_SUFFIX_THRESHOLD:-0}"
DFLASH_SUFFIX_MAX_MATCHES="${DFLASH_SUFFIX_MAX_MATCHES:-0}"
DFLASH_POSITION_MODE="${DFLASH_POSITION_MODE:-compact}"
DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
DRAFT_STATIC_YARN_MAX_POSITION_EMBEDDINGS="${DRAFT_STATIC_YARN_MAX_POSITION_EMBEDDINGS:-}"
DRAFT_STATIC_YARN_FACTOR="${DRAFT_STATIC_YARN_FACTOR:-}"
ORIGINAL_MAX_POSITION_EMBEDDING="${ORIGINAL_MAX_POSITION_EMBEDDING:-}"
DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
DRAFT_DYNAMIC_YARN_MAX_FACTOR="${DRAFT_DYNAMIC_YARN_MAX_FACTOR:-}"
DRAFT_DYNAMIC_YARN_MODE="${DRAFT_DYNAMIC_YARN_MODE:-}"
DRAFT_DYNAMIC_YARN_LENGTH_RATIO="${DRAFT_DYNAMIC_YARN_LENGTH_RATIO:-}"
TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# sink_recent_suffix

if [[ -n "$ORIGINAL_MAX_POSITION_EMBEDDING" ]]; then
  if [[ -n "$DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS" || -n "$DRAFT_STATIC_YARN_MAX_POSITION_EMBEDDINGS" || -n "$DRAFT_STATIC_YARN_FACTOR" ]]; then
    echo "ORIGINAL_MAX_POSITION_EMBEDDING cannot be combined with draft static YaRN" >&2
    exit 1
  fi
  export DRAFT_MODEL_PATH
  export ORIGINAL_MAX_POSITION_EMBEDDING
  export ROOT_DIR
  DRAFT_MODEL_PATH=$(
    python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

src = Path(os.environ["DRAFT_MODEL_PATH"]).resolve()
if not src.is_dir():
    raise SystemExit(f"DRAFT_MODEL_PATH is not a directory: {src}")
max_position = int(os.environ["ORIGINAL_MAX_POSITION_EMBEDDING"])
if max_position <= 0:
    raise SystemExit("ORIGINAL_MAX_POSITION_EMBEDDING must be positive")
cache_root = Path(
    os.environ.get("ORIGINAL_MAX_POSITION_CACHE_DIR")
    or os.environ.get("VLLM_CACHE_ROOT")
    or (Path(os.environ["ROOT_DIR"]) / ".cache" / "vllm_original_max_position")
)
digest = hashlib.sha1(f"{src}:{max_position}".encode()).hexdigest()[:12]
dst = cache_root / f"{src.name}-maxpos{max_position}-{digest}"
dst.mkdir(parents=True, exist_ok=True)
for child in src.iterdir():
    target = dst / child.name
    if child.name == "config.json" or target.exists() or target.is_symlink():
        continue
    target.symlink_to(child, target_is_directory=child.is_dir())
cfg_path = src / "config.json"
cfg = json.loads(cfg_path.read_text())
cfg["max_position_embeddings"] = max_position
cfg.pop("rope_scaling", None)
cfg.pop("rope_parameters", None)
(dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print(dst)
PY
  )
fi

export DRAFT_MODEL_PATH
export NUM_SPEC_TOKENS
export SPECULATIVE_METHOD
export DFLASH_WINDOW_SIZE
export DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS
export DRAFT_STATIC_YARN_MAX_POSITION_EMBEDDINGS
export DRAFT_STATIC_YARN_FACTOR
export DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS
export DRAFT_DYNAMIC_YARN_MAX_FACTOR
export DRAFT_DYNAMIC_YARN_MODE
export DRAFT_DYNAMIC_YARN_LENGTH_RATIO

SPEC_CONFIG=$(
  python - <<'PY'
import json
import os

cfg = {
    "method": os.environ.get("SPECULATIVE_METHOD", "dflash"),
    "model": os.environ["DRAFT_MODEL_PATH"],
    "num_speculative_tokens": int(os.environ["NUM_SPEC_TOKENS"]),
}
window = os.environ.get("DFLASH_WINDOW_SIZE", "full")
if cfg["method"] == "dflash" and window != "full":
    cfg["dflash_window_size"] = int(window)
optional = {
    "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS": (
        "dflash_dynamic_yarn_original_max_position_embeddings",
        int,
    ),
    "DRAFT_DYNAMIC_YARN_MAX_FACTOR": ("dflash_dynamic_yarn_max_factor", float),
    "DRAFT_DYNAMIC_YARN_MODE": ("dflash_dynamic_yarn_mode", str),
    "DRAFT_DYNAMIC_YARN_LENGTH_RATIO": ("dflash_dynamic_yarn_length_ratio", float),
}
for env_name, (key, caster) in optional.items():
    value = os.environ.get(env_name)
    if cfg["method"] == "dflash" and value:
        cfg[key] = caster(value)
print(json.dumps(cfg, separators=(",", ":")))
PY
)

if [[ "$DFLASH_WINDOW_SIZE" != "full" ]]; then
  export DFLASH_WINDOW_MODE
  [[ -n "$DFLASH_RECENT_TOKENS" ]] && export DFLASH_RECENT_TOKENS
  [[ -n "$DFLASH_SUFFIX_MATCH_TOKENS" ]] && export DFLASH_SUFFIX_MATCH_TOKENS
  [[ -n "$DFLASH_SUFFIX_KEEP_TOKENS" ]] && export DFLASH_SUFFIX_KEEP_TOKENS
  [[ -n "$DFLASH_SUFFIX_MIDDLE_BUDGET" ]] && export DFLASH_SUFFIX_MIDDLE_BUDGET
  export DFLASH_POSITION_MODE
fi

if [[ "$DFLASH_SUFFIX_DECODING" == "1" ]]; then
  export DFLASH_SUFFIX_DECODING
  export DFLASH_SUFFIX_MAX_QUERY_LEN
  export DFLASH_SUFFIX_MIN_QUERY_LEN
  [[ -n "$DFLASH_SUFFIX_MAX_PREDICT_LEN" ]] && export DFLASH_SUFFIX_MAX_PREDICT_LEN
  export DFLASH_SUFFIX_ALPHA
  export DFLASH_SUFFIX_MAX_SPEC_OFFSET
  export DFLASH_SUFFIX_MIN_TOKEN_PROB
  export DFLASH_SUFFIX_THRESHOLD
  export DFLASH_SUFFIX_MAX_MATCHES
  if [[ "$EXTRA_VLLM_ARGS" != *"--async-scheduling"* ]]; then
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }--no-async-scheduling"
  fi
fi

cmd=("vllm" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --data-parallel-size "$DP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --attention-backend flashinfer \
  --enable-auto-tool-choice \
  --max-model-len "$MAX_MODEL_LEN" \
  --speculative-config "$SPEC_CONFIG")

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
