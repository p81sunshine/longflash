#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

SUFFIX_MAX_QUERY_LEN="${SUFFIX_MAX_QUERY_LEN:-16}"
SUFFIX_MIN_QUERY_LEN="${SUFFIX_MIN_QUERY_LEN:-10}"
SUFFIX_MAX_PREDICT_LEN="${SUFFIX_MAX_PREDICT_LEN:-15}"
SUFFIX_ALPHA="${SUFFIX_ALPHA:-2.0}"
SUFFIX_MAX_SPEC_OFFSET="${SUFFIX_MAX_SPEC_OFFSET:-0.0}"
SUFFIX_MIN_TOKEN_PROB="${SUFFIX_MIN_TOKEN_PROB:-0.0}"
SUFFIX_THRESHOLD="${SUFFIX_THRESHOLD:-4.0}"
SUFFIX_MAX_MATCHES="${SUFFIX_MAX_MATCHES:-0}"
SUFFIX_VERIFIER="${SUFFIX_VERIFIER:-linear}"
RUN_BASELINES="${RUN_BASELINES:-1}"
EXPERIMENT_VARIANTS="${EXPERIMENT_VARIANTS:-}"
CONCURRENCY="${CONCURRENCY:-1}"
CONCURRENCY_SCHEDULER="${CONCURRENCY_SCHEDULER:-${concurrency_scheduler:-sliding}}"
NO_MANAGE_SERVER="${NO_MANAGE_SERVER:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/reproduced/vllm_manual}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30001/v1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
TP_SIZE="${TP_SIZE:-2}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-z-lab/Qwen3-8B-DFlash-b16}"
SAMPLES="${SAMPLES:-${ROOT_DIR}/benchmarks/terminal/bucket_0_32768.jsonl}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DRAFT_YARN_ORIGINAL="${DRAFT_YARN_ORIGINAL:-3072}"
DRAFT_YARN_FACTOR="${DRAFT_YARN_FACTOR:-12}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MIN_TOKENS="${MIN_TOKENS:-}"
IGNORE_EOS="${IGNORE_EOS:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-}"
ORIGINAL_MAX_POSITION_EMBEDDING="${ORIGINAL_MAX_POSITION_EMBEDDING:-}"
START_SCRIPT="${START_SCRIPT:-${ROOT_DIR}/scripts/serve/start_vllm_dflash_benchmark.sh}"
TARGET_START_SCRIPT="${TARGET_START_SCRIPT:-${ROOT_DIR}/scripts/serve/start_vllm_qwen35_target_benchmark.sh}"

BENCH_CONCURRENCY_ARGS=(--concurrency "${CONCURRENCY}" --concurrency-scheduler "${CONCURRENCY_SCHEDULER}")

case "${CONCURRENCY_SCHEDULER}" in
  batch|sliding)
    ;;
  *)
    echo "CONCURRENCY_SCHEDULER must be 'batch' or 'sliding', got: ${CONCURRENCY_SCHEDULER}" >&2
    exit 1
    ;;
esac

BENCH_THINKING_ARGS=()
case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES|on|ON)
    BENCH_THINKING_ARGS+=(--disable-thinking)
    ;;
esac

BENCH_EAGER_ARGS=()
case "${ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES|on|ON)
    BENCH_EAGER_ARGS+=(--enforce-eager)
    ;;
esac

BENCH_PREFIX_CACHING_ARGS=()
case "${ENABLE_PREFIX_CACHING}" in
  1|true|TRUE|yes|YES|on|ON)
    BENCH_PREFIX_CACHING_ARGS+=(--enable-prefix-caching)
    ;;
esac

BENCH_SERVER_ARGS=()
case "${NO_MANAGE_SERVER}" in
  1|true|TRUE|yes|YES|on|ON)
    BENCH_SERVER_ARGS+=(--no-manage-server)
    ;;
esac

BENCH_SAMPLING_ARGS=()
if [[ -n "${MIN_TOKENS}" ]]; then
  BENCH_SAMPLING_ARGS+=(--min-tokens "${MIN_TOKENS}")
fi
case "${IGNORE_EOS}" in
  1|true|TRUE|yes|YES|on|ON)
    BENCH_SAMPLING_ARGS+=(--ignore-eos)
    ;;
esac

TARGET_YARN_ARGS=()
if [[ -n "${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" ]]; then
  TARGET_YARN_ARGS+=(--target-yarn-original-max-position-embeddings "${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${TARGET_YARN_MAX_POSITION_EMBEDDINGS}" ]]; then
  TARGET_YARN_ARGS+=(--target-yarn-max-position-embeddings "${TARGET_YARN_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${TARGET_YARN_FACTOR}" ]]; then
  TARGET_YARN_ARGS+=(--target-yarn-factor "${TARGET_YARN_FACTOR}")
fi

is_qwen3_model() {
  case "$(basename "${MODEL_PATH}")" in
    Qwen3-4B|Qwen3-8B)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if is_qwen3_model && [[ -z "${ORIGINAL_MAX_POSITION_EMBEDDING}" && -n "${TARGET_YARN_FACTOR}" ]]; then
  if [[ -n "${TARGET_YARN_MAX_POSITION_EMBEDDINGS}" ]]; then
    ORIGINAL_MAX_POSITION_EMBEDDING="${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
  elif [[ -n "${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" ]]; then
    ORIGINAL_MAX_POSITION_EMBEDDING="$("${PYTHON}" - "${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" "${TARGET_YARN_FACTOR}" <<'PY'
import math
import sys

print(int(math.ceil(float(sys.argv[1]) * float(sys.argv[2]))))
PY
)"
  fi
fi

SUFFIX_EXPERIMENT="name=dflash-static-yarn-suffix,window=full,window_mode=full,sink_tokens=0,draft_yarn_original=${DRAFT_YARN_ORIGINAL},draft_yarn_factor=${DRAFT_YARN_FACTOR}"
SUFFIX_EXPERIMENT+=",suffix_decoding=true"
SUFFIX_EXPERIMENT+=",suffix_max_query_len=${SUFFIX_MAX_QUERY_LEN}"
SUFFIX_EXPERIMENT+=",suffix_min_query_len=${SUFFIX_MIN_QUERY_LEN}"
SUFFIX_EXPERIMENT+=",suffix_max_predict_len=${SUFFIX_MAX_PREDICT_LEN}"
SUFFIX_EXPERIMENT+=",suffix_alpha=${SUFFIX_ALPHA}"
SUFFIX_EXPERIMENT+=",suffix_max_spec_offset=${SUFFIX_MAX_SPEC_OFFSET}"
SUFFIX_EXPERIMENT+=",suffix_min_token_prob=${SUFFIX_MIN_TOKEN_PROB}"
SUFFIX_EXPERIMENT+=",suffix_threshold=${SUFFIX_THRESHOLD}"
SUFFIX_EXPERIMENT+=",suffix_max_matches=${SUFFIX_MAX_MATCHES}"
SUFFIX_EXPERIMENT+=",suffix_verifier=${SUFFIX_VERIFIER}"

ORIGINAL_EXPERIMENT="name=original,window=full,window_mode=full,sink_tokens=0"
if [[ -n "${ORIGINAL_MAX_POSITION_EMBEDDING}" ]]; then
  ORIGINAL_EXPERIMENT+=",original_max_position_embedding=${ORIGINAL_MAX_POSITION_EMBEDDING}"
fi

DEFAULT_EXPERIMENT_VARIANTS="dflash-static-yarn-suffix"
if [[ "${RUN_BASELINES}" != "0" ]]; then
  DEFAULT_EXPERIMENT_VARIANTS+=" dflash-static-yarn target-only original"
fi
EXPERIMENT_VARIANTS="${EXPERIMENT_VARIANTS:-${DEFAULT_EXPERIMENT_VARIANTS}}"

EXPERIMENT_ARGS=()
for variant in ${EXPERIMENT_VARIANTS}; do
  case "${variant}" in
    dflash-static-yarn-suffix|yarn-suffix|suffix)
      EXPERIMENT_ARGS+=(--experiment "${SUFFIX_EXPERIMENT}")
      ;;
    dflash-static-yarn|yarn)
      EXPERIMENT_ARGS+=(--experiment "name=dflash-static-yarn,window=full,window_mode=full,sink_tokens=0,draft_yarn_original=${DRAFT_YARN_ORIGINAL},draft_yarn_factor=${DRAFT_YARN_FACTOR}")
      ;;
    eagle3-linear|eagle3_linear)
      EXPERIMENT_ARGS+=(--experiment "name=eagle3-linear,window=full,window_mode=full,sink_tokens=0")
      ;;
    target-only|target_only)
      EXPERIMENT_ARGS+=(--experiment "name=target-only,window=full,window_mode=target_only,target_only=true")
      ;;
    original)
      EXPERIMENT_ARGS+=(--experiment "${ORIGINAL_EXPERIMENT}")
      ;;
    *)
      echo "Unknown EXPERIMENT_VARIANTS entry: ${variant}" >&2
      exit 1
      ;;
  esac
done

echo "Benchmark concurrency=${CONCURRENCY} scheduler=${CONCURRENCY_SCHEDULER}"

TP_SIZE="${TP_SIZE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" "${ROOT_DIR}/scripts/benchmark/run_agentic_memory_benchmark.py" \
--model "${MODEL_PATH}" \
--draft-model "${DRAFT_MODEL_PATH}" \
--base-url "${BASE_URL}" \
--start-script "${START_SCRIPT}" \
--target-start-script "${TARGET_START_SCRIPT}" \
"${EXPERIMENT_ARGS[@]}" \
"${BENCH_THINKING_ARGS[@]}" \
--max-samples "${MAX_SAMPLES}" \
--max-tokens "${MAX_TOKENS}" \
--num-spec-tokens "${NUM_SPEC_TOKENS}" \
"${BENCH_CONCURRENCY_ARGS[@]}" \
--temperature "${TEMPERATURE}" \
--top-p "${TOP_P}" \
--top-k "${TOP_K}" \
"${BENCH_SAMPLING_ARGS[@]}" \
--max-model-len "${MAX_MODEL_LEN}" \
--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
--tool-call-parser "${TOOL_CALL_PARSER}" \
--reasoning-parser "${REASONING_PARSER}" \
"${TARGET_YARN_ARGS[@]}" \
"${BENCH_EAGER_ARGS[@]}" \
"${BENCH_SERVER_ARGS[@]}" \
"${BENCH_PREFIX_CACHING_ARGS[@]}" \
--output-dir "${OUTPUT_DIR}" \
--enable-chunked-prefill \
--samples "${SAMPLES}"
