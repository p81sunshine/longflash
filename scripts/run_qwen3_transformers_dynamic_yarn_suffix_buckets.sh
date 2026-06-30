#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUCKET_RUNNER="${BUCKET_RUNNER:-${SCRIPT_DIR}/run_transformers_buckets.sh}"

TERMINAL_BUCKET_DIR="${TERMINAL_BUCKET_DIR:-${ROOT_DIR}/benchmarks/terminal}"
SWEBENCH_BUCKET_DIR="${SWEBENCH_BUCKET_DIR:-${ROOT_DIR}/benchmarks/swebench}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/reproduced/transformers_main}"

MODELS="${MODELS:-qwen3-8b qwen3-4b}"
TEMPERATURES="${TEMPERATURES:-0 1}"
VARIANTS="${VARIANTS:-dynamic_yarn original_dflash dynamic_yarn_suffix}"
DATASET_GROUPS="${DATASET_GROUPS:-terminal swebench}"

NUM_PROCS="${NUM_PROCS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29900}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NO_SAVE_RESPONSES="${NO_SAVE_RESPONSES:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
DRY_RUN="${DRY_RUN:-0}"

DRAFT_DYNAMIC_YARN_ORIGINAL="${DRAFT_DYNAMIC_YARN_ORIGINAL:-3072}"
DRAFT_DYNAMIC_YARN_MODE="${DRAFT_DYNAMIC_YARN_MODE:-continuous}"
TARGET_ORIGINAL="${TARGET_ORIGINAL:-32768}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-4}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-$((TARGET_ORIGINAL * TARGET_YARN_FACTOR))}"
TARGET_YARN_ENABLED="${TARGET_YARN_ENABLED:-1}"
SUFFIX_RUN_MIN_QUERY_LEN="${SUFFIX_RUN_MIN_QUERY_LEN:-10}"
SUFFIX_RUN_PAPER_THRESHOLD="${SUFFIX_RUN_PAPER_THRESHOLD:-4}"
SUFFIX_RUN_PAPER_ALPHA="${SUFFIX_RUN_PAPER_ALPHA:-2}"
SUFFIX_RUN_MIN_QUERY_LEN_LABEL="${SUFFIX_RUN_MIN_QUERY_LEN//./p}"
SUFFIX_RUN_PAPER_THRESHOLD_LABEL="${SUFFIX_RUN_PAPER_THRESHOLD//./p}"
SUFFIX_RUN_PAPER_ALPHA_LABEL="${SUFFIX_RUN_PAPER_ALPHA//./p}"
DRAFT_SLIDING_WINDOW_SIZE="${DRAFT_SLIDING_WINDOW_SIZE:-3072}"
DRAFT_SLIDING_WINDOW_SIZE_LABEL="${DRAFT_SLIDING_WINDOW_SIZE//./p}"

mkdir -p "${OUTPUT_DIR}"

model_paths() {
  local model_key="$1"
  case "${model_key}" in
    qwen3-8b)
      MODEL_PATH="Qwen/Qwen3-8B"
      DRAFT_MODEL_PATH="z-lab/Qwen3-8B-DFlash-b16"
      MODEL_LABEL="qwen3_8b"
      ;;
    qwen3-4b)
      MODEL_PATH="Qwen/Qwen3-4B"
      DRAFT_MODEL_PATH="z-lab/Qwen3-4B-DFlash-b16"
      MODEL_LABEL="qwen3_4b"
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac
}

bucket_list() {
  local bucket_dir="$1"
  find "${bucket_dir}" -maxdepth 1 -name 'bucket_*.jsonl' -printf '%f\n' \
    | sed 's/^bucket_//; s/\.jsonl$//' \
    | sort -V \
    | tr '\n' ' '
}

variant_env() {
  local variant="$1"
  VARIANT_TARGET_YARN_ENABLED="${TARGET_YARN_ENABLED}"
  case "${variant}" in
    original_dflash)
      VARIANT_LABEL="original_dflash"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE="
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE="
        "SUFFIX_DECODING=0"
      )
      ;;
    dynamic_yarn)
      VARIANT_LABEL="dynamic_yarn"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_DYNAMIC_YARN_ORIGINAL}"
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE=${DRAFT_DYNAMIC_YARN_MODE}"
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE="
        "SUFFIX_DECODING=0"
      )
      ;;
    dynamic_yarn_suffix)
      VARIANT_LABEL="dynamic_yarn_suffix_minq${SUFFIX_RUN_MIN_QUERY_LEN_LABEL}_tau${SUFFIX_RUN_PAPER_THRESHOLD_LABEL}_alpha${SUFFIX_RUN_PAPER_ALPHA_LABEL}"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_DYNAMIC_YARN_ORIGINAL}"
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE=${DRAFT_DYNAMIC_YARN_MODE}"
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE="
        "SUFFIX_DECODING=1"
        "SUFFIX_STRATEGY=paper"
        "SUFFIX_MAX_QUERY_LEN=16"
        "SUFFIX_MIN_QUERY_LEN=${SUFFIX_RUN_MIN_QUERY_LEN}"
        "SUFFIX_TOP_K=4"
        "SUFFIX_MIN_SUPPORT=3"
        "SUFFIX_MIN_PREDICT_LEN=8"
        "SUFFIX_MAX_PREDICT_LEN="
        "SUFFIX_PAPER_ALPHA=${SUFFIX_RUN_PAPER_ALPHA}"
        "SUFFIX_PAPER_MAX_SPEC_OFFSET=0"
        "SUFFIX_PAPER_MIN_TOKEN_PROB=0"
        "SUFFIX_PAPER_THRESHOLD=${SUFFIX_RUN_PAPER_THRESHOLD}"
        "SUFFIX_PAPER_MAX_MATCHES=0"
        "SUFFIX_PAPER_VERIFIER=linear"
        "SUFFIX_PAPER_TREE_ATTN_IMPL=sdpa"
        "SUFFIX_FALLBACK=dflash"
        "SAVE_SUFFIX_TRACE=0"
      )
      ;;
    dynamic_yarn_suffix_swa3072)
      VARIANT_LABEL="dynamic_yarn_suffix_minq${SUFFIX_RUN_MIN_QUERY_LEN_LABEL}_tau${SUFFIX_RUN_PAPER_THRESHOLD_LABEL}_alpha${SUFFIX_RUN_PAPER_ALPHA_LABEL}_swa${DRAFT_SLIDING_WINDOW_SIZE_LABEL}"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_DYNAMIC_YARN_ORIGINAL}"
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE=${DRAFT_DYNAMIC_YARN_MODE}"
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE=${DRAFT_SLIDING_WINDOW_SIZE}"
        "SUFFIX_DECODING=1"
        "SUFFIX_STRATEGY=paper"
        "SUFFIX_MAX_QUERY_LEN=16"
        "SUFFIX_MIN_QUERY_LEN=${SUFFIX_RUN_MIN_QUERY_LEN}"
        "SUFFIX_TOP_K=4"
        "SUFFIX_MIN_SUPPORT=3"
        "SUFFIX_MIN_PREDICT_LEN=8"
        "SUFFIX_MAX_PREDICT_LEN="
        "SUFFIX_PAPER_ALPHA=${SUFFIX_RUN_PAPER_ALPHA}"
        "SUFFIX_PAPER_MAX_SPEC_OFFSET=0"
        "SUFFIX_PAPER_MIN_TOKEN_PROB=0"
        "SUFFIX_PAPER_THRESHOLD=${SUFFIX_RUN_PAPER_THRESHOLD}"
        "SUFFIX_PAPER_MAX_MATCHES=0"
        "SUFFIX_PAPER_VERIFIER=linear"
        "SUFFIX_PAPER_TREE_ATTN_IMPL=sdpa"
        "SUFFIX_FALLBACK=dflash"
        "SAVE_SUFFIX_TRACE=0"
      )
      ;;
    suffix_dflash)
      VARIANT_LABEL="suffix_dflash_minq${SUFFIX_RUN_MIN_QUERY_LEN_LABEL}_tau${SUFFIX_RUN_PAPER_THRESHOLD_LABEL}_alpha${SUFFIX_RUN_PAPER_ALPHA_LABEL}"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE="
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE="
        "SUFFIX_DECODING=1"
        "SUFFIX_STRATEGY=paper"
        "SUFFIX_MAX_QUERY_LEN=16"
        "SUFFIX_MIN_QUERY_LEN=${SUFFIX_RUN_MIN_QUERY_LEN}"
        "SUFFIX_TOP_K=4"
        "SUFFIX_MIN_SUPPORT=3"
        "SUFFIX_MIN_PREDICT_LEN=8"
        "SUFFIX_MAX_PREDICT_LEN="
        "SUFFIX_PAPER_ALPHA=${SUFFIX_RUN_PAPER_ALPHA}"
        "SUFFIX_PAPER_MAX_SPEC_OFFSET=0"
        "SUFFIX_PAPER_MIN_TOKEN_PROB=0"
        "SUFFIX_PAPER_THRESHOLD=${SUFFIX_RUN_PAPER_THRESHOLD}"
        "SUFFIX_PAPER_MAX_MATCHES=0"
        "SUFFIX_PAPER_VERIFIER=linear"
        "SUFFIX_PAPER_TREE_ATTN_IMPL=sdpa"
        "SUFFIX_FALLBACK=dflash"
        "SAVE_SUFFIX_TRACE=0"
      )
      ;;
    suffix_only)
      VARIANT_LABEL="suffix_only_minq${SUFFIX_RUN_MIN_QUERY_LEN_LABEL}_tau${SUFFIX_RUN_PAPER_THRESHOLD_LABEL}_alpha${SUFFIX_RUN_PAPER_ALPHA_LABEL}"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE="
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE="
        "SUFFIX_DECODING=1"
        "SUFFIX_STRATEGY=paper"
        "SUFFIX_MAX_QUERY_LEN=16"
        "SUFFIX_MIN_QUERY_LEN=${SUFFIX_RUN_MIN_QUERY_LEN}"
        "SUFFIX_TOP_K=4"
        "SUFFIX_MIN_SUPPORT=3"
        "SUFFIX_MIN_PREDICT_LEN=8"
        "SUFFIX_MAX_PREDICT_LEN="
        "SUFFIX_PAPER_ALPHA=${SUFFIX_RUN_PAPER_ALPHA}"
        "SUFFIX_PAPER_MAX_SPEC_OFFSET=0"
        "SUFFIX_PAPER_MIN_TOKEN_PROB=0"
        "SUFFIX_PAPER_THRESHOLD=${SUFFIX_RUN_PAPER_THRESHOLD}"
        "SUFFIX_PAPER_MAX_MATCHES=0"
        "SUFFIX_PAPER_VERIFIER=linear"
        "SUFFIX_PAPER_TREE_ATTN_IMPL=sdpa"
        "SUFFIX_FALLBACK=target"
        "SAVE_SUFFIX_TRACE=0"
      )
      ;;
    swa3072)
      VARIANT_LABEL="swa${DRAFT_SLIDING_WINDOW_SIZE_LABEL}"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE="
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE=${DRAFT_SLIDING_WINDOW_SIZE}"
        "SUFFIX_DECODING=0"
      )
      ;;
    suffix_swa3072)
      VARIANT_LABEL="suffix_minq${SUFFIX_RUN_MIN_QUERY_LEN_LABEL}_tau${SUFFIX_RUN_PAPER_THRESHOLD_LABEL}_alpha${SUFFIX_RUN_PAPER_ALPHA_LABEL}_swa${DRAFT_SLIDING_WINDOW_SIZE_LABEL}"
      VARIANT_ENV=(
        "CTX_SINK_TOKENS=0"
        "CTX_RECENT_WINDOW=0"
        "CTX_STRIDE=0"
        "CTX_SUFFIX_MATCH_TOKENS=0"
        "CTX_SUFFIX_KEEP_TOKENS=0"
        "CTX_MIDDLE_BUDGET=0"
        "CTX_TOTAL_BUDGET="
        "CTX_DYNAMIC_BUDGET_RATIO="
        "CTX_BUDGET_ORDER=default"
        "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
        "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
        "DRAFT_DYNAMIC_YARN_MODE="
        "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
        "DRAFT_SLIDING_WINDOW_SIZE=${DRAFT_SLIDING_WINDOW_SIZE}"
        "SUFFIX_DECODING=1"
        "SUFFIX_STRATEGY=paper"
        "SUFFIX_MAX_QUERY_LEN=16"
        "SUFFIX_MIN_QUERY_LEN=${SUFFIX_RUN_MIN_QUERY_LEN}"
        "SUFFIX_TOP_K=4"
        "SUFFIX_MIN_SUPPORT=3"
        "SUFFIX_MIN_PREDICT_LEN=8"
        "SUFFIX_MAX_PREDICT_LEN="
        "SUFFIX_PAPER_ALPHA=${SUFFIX_RUN_PAPER_ALPHA}"
        "SUFFIX_PAPER_MAX_SPEC_OFFSET=0"
        "SUFFIX_PAPER_MIN_TOKEN_PROB=0"
        "SUFFIX_PAPER_THRESHOLD=${SUFFIX_RUN_PAPER_THRESHOLD}"
        "SUFFIX_PAPER_MAX_MATCHES=0"
        "SUFFIX_PAPER_VERIFIER=linear"
        "SUFFIX_PAPER_TREE_ATTN_IMPL=sdpa"
        "SUFFIX_FALLBACK=dflash"
        "SAVE_SUFFIX_TRACE=0"
      )
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

common_env() {
  COMMON_ENV=(
    "MODEL=${MODEL_PATH}"
    "DRAFT_MODEL=${DRAFT_MODEL_PATH}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "NUM_PROCS=${NUM_PROCS}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "MAX_SAMPLES=${MAX_SAMPLES}"
    "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
    "TEMPERATURE=${TEMPERATURE}"
    "SAMPLE_SEED=${SAMPLE_SEED}"
    "BLOCK_SIZE=${BLOCK_SIZE}"
    "NO_SAVE_RESPONSES=${NO_SAVE_RESPONSES}"
    "DRAFT_DENOISE_STEPS=1"
    "SAVE_VERIFY_TRACE=0"
    "VERIFY_TRACE_MAX_ROUNDS=0"
    "VERIFY_CONFIDENCE_THRESHOLD=0"
    "VERIFY_MIN_DRAFT_TOKENS=1"
    "PROFILER=0"
  )
}

run_bucket_group() {
  local group_name="$1"
  local bucket_dir="$2"
  local port_base="$3"
  local buckets
  buckets="$(bucket_list "${bucket_dir}")"
  if [[ -z "${buckets// }" ]]; then
    echo "No bucket files found in ${bucket_dir}" >&2
    exit 1
  fi

  local env_args=(
    "${COMMON_ENV[@]}"
    "${VARIANT_ENV[@]}"
    "BUCKET_DIR=${bucket_dir}"
    "DATASET_LABEL=${group_name}"
    "BUCKETS=${buckets}"
    "RUN_PREFIX=original_${MODEL_LABEL}_temp${TEMP_LABEL}_${VARIANT_LABEL}"
    "MASTER_PORT_BASE=${port_base}"
    "TARGET_YARN_ENABLED=${VARIANT_TARGET_YARN_ENABLED}"
    "SKIP_COMPLETED=${SKIP_COMPLETED}"
    "CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE}"
  )
  if [[ "${VARIANT_TARGET_YARN_ENABLED}" == "1" ]]; then
    env_args+=(
      "TARGET_ORIGINAL=${TARGET_ORIGINAL}"
      "TARGET_YARN_FACTOR=${TARGET_YARN_FACTOR}"
      "TARGET_YARN_MAX_POSITION_EMBEDDINGS=${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
    )
  else
    env_args+=(
      "TARGET_ORIGINAL="
      "TARGET_YARN_FACTOR="
      "TARGET_YARN_MAX_POSITION_EMBEDDINGS="
    )
  fi

  echo "===== ${MODEL_LABEL} temp=${TEMPERATURE} ${group_name} ${VARIANT_LABEL} ====="
  echo "bucket_dir=${bucket_dir}"
  echo "buckets=${buckets}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' bash %q\n\n' "${BUCKET_RUNNER}"
  else
    env "${env_args[@]}" bash "${BUCKET_RUNNER}"
    echo
  fi
}

echo "output_dir=${OUTPUT_DIR}"
echo "models=${MODELS}"
echo "temperatures=${TEMPERATURES}"
echo "variants=${VARIANTS}"
echo "dataset_groups=${DATASET_GROUPS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "num_procs=${NUM_PROCS}"
echo "max_samples=${MAX_SAMPLES}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "default_target_yarn_enabled=${TARGET_YARN_ENABLED}"
echo "target_original=${TARGET_ORIGINAL}"
echo "target_yarn_max_position_embeddings=${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
echo "draft_sliding_window_size=${DRAFT_SLIDING_WINDOW_SIZE}"
echo

run_idx=0
for model_key in ${MODELS}; do
  model_paths "${model_key}"
  for TEMPERATURE in ${TEMPERATURES}; do
    TEMP_LABEL="${TEMPERATURE//./p}"
    for variant in ${VARIANTS}; do
      variant_env "${variant}"
      common_env
      run_idx=$((run_idx + 1))
      for group in ${DATASET_GROUPS}; do
        case "${group}" in
          terminal)
            run_bucket_group "terminal" "${TERMINAL_BUCKET_DIR}" "$((MASTER_PORT_BASE + run_idx * 1000))"
            ;;
          swebench)
            run_bucket_group "swebench" "${SWEBENCH_BUCKET_DIR}" "$((MASTER_PORT_BASE + run_idx * 1000 + 300))"
            ;;
          *)
            echo "Unknown dataset group: ${group}" >&2
            exit 1
            ;;
        esac
      done
    done
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_dataset_runs.py" \
    --run-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/summary_table.csv"
fi

echo "Done. output_dir=${OUTPUT_DIR}"
echo "summary_csv=${OUTPUT_DIR}/summary_table.csv"
