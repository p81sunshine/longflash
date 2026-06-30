#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_transformers_single.sh}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/reproduced/yarn_scan_${STAMP}}"
DATASET="${DATASET:-${OUTPUT_DIR}/sampled_terminal_swebench_buckets.jsonl}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/sample_manifest.json}"
TERMINAL_BUCKET_DIR="${TERMINAL_BUCKET_DIR:-${ROOT_DIR}/benchmarks/terminal}"
SWEBENCH_BUCKET_DIR="${SWEBENCH_BUCKET_DIR:-${ROOT_DIR}/benchmarks/swebench}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NUM_PROCS="${NUM_PROCS:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31670}"
SAMPLES_PER_BUCKET="${SAMPLES_PER_BUCKET:-2}"
SAMPLE_SEED="${SAMPLE_SEED:-20260610}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NO_SAVE_RESPONSES="${NO_SAVE_RESPONSES:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-4}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-$((TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS * TARGET_YARN_FACTOR))}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -s "${DATASET}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/build_static_yarn_scan_dataset.py" \
    --terminal-bucket-dir "${TERMINAL_BUCKET_DIR}" \
    --swebench-bucket-dir "${SWEBENCH_BUCKET_DIR}" \
    --output-jsonl "${DATASET}" \
    --manifest-json "${MANIFEST}" \
    --samples-per-bucket "${SAMPLES_PER_BUCKET}" \
    --seed "${SAMPLE_SEED}"
fi

if [[ -n "${CONFIG_ITEMS:-}" ]]; then
  read -r -a CONFIGS <<<"${CONFIG_ITEMS}"
else
  CONFIGS=(
    "3072:42:orig3072_factor42_max129024"
    "4096:32:orig4096_factor32_max131072"
    "6144:22:orig6144_factor22_max135168"
    "8192:16:orig8192_factor16_max131072"
    "12288:11:orig12288_factor11_max135168"
    "16384:8:orig16384_factor8_max131072"
    "24576:5.333333333333333:orig24576_factor5p333_max131072"
    "32768:4:orig32768_factor4_max131072"
  )
fi

echo "output_dir=${OUTPUT_DIR}"
echo "dataset=${DATASET}"
echo "manifest=${MANIFEST}"
echo "terminal_bucket_dir=${TERMINAL_BUCKET_DIR}"
echo "swebench_bucket_dir=${SWEBENCH_BUCKET_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "num_procs=${NUM_PROCS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "target_yarn=${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}x${TARGET_YARN_FACTOR}"
echo "configs=${CONFIGS[*]}"
echo

idx=0
for item in "${CONFIGS[@]}"; do
  IFS=: read -r draft_original draft_factor label <<<"${item}"
  idx=$((idx + 1))
  run_name="qwen3_8b_static_yarn_suffix_${label}_mixed_terminal_swebench_spb${SAMPLES_PER_BUCKET}_mn${MAX_NEW_TOKENS}"
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    existing_summary="$(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -name summary.json -path "*_${run_name}/summary.json" -print -quit)"
    if [[ -n "${existing_summary}" ]]; then
      echo "===== ${run_name} ====="
      echo "skip_completed=1: found ${existing_summary}"
      echo
      continue
    fi
  fi

  echo "===== ${run_name} ====="
  if [[ "${TRACE_RUNNER:-0}" == "1" ]]; then
    runner_cmd=(bash -x "${RUNNER}")
  else
    runner_cmd=(bash "${RUNNER}")
  fi
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_PROCS="${NUM_PROCS}" \
    MASTER_PORT="$((MASTER_PORT_BASE + idx))" \
    MODEL="${MODEL}" \
    DRAFT_MODEL="${DRAFT_MODEL}" \
    DATA_SET="${DATASET}" \
    MAX_SAMPLES=1000000000 \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    TEMPERATURE="${TEMPERATURE}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    RUN_NAME="${run_name}" \
    SAMPLE_SEED=0 \
    BLOCK_SIZE="${BLOCK_SIZE}" \
    NO_SAVE_RESPONSES="${NO_SAVE_RESPONSES}" \
    CTX_SINK_TOKENS=0 \
    CTX_RECENT_WINDOW=0 \
    CTX_STRIDE=0 \
    CTX_SUFFIX_MATCH_TOKENS=0 \
    CTX_SUFFIX_KEEP_TOKENS=0 \
    CTX_MIDDLE_BUDGET=0 \
    CTX_BUDGET_ORDER=default \
    DRAFT_DENOISE_STEPS=1 \
    SAVE_VERIFY_TRACE=0 \
    VERIFY_TRACE_MAX_ROUNDS=0 \
    VERIFY_CONFIDENCE_THRESHOLD=0 \
    VERIFY_MIN_DRAFT_TOKENS=1 \
    SUFFIX_DECODING=1 \
    SUFFIX_STRATEGY=paper \
    SUFFIX_MAX_QUERY_LEN=16 \
    SUFFIX_MIN_QUERY_LEN=10 \
    SUFFIX_TOP_K=4 \
    SUFFIX_MIN_SUPPORT=3 \
    SUFFIX_MIN_PREDICT_LEN=8 \
    SUFFIX_PAPER_ALPHA=2 \
    SUFFIX_PAPER_MAX_SPEC_OFFSET=0 \
    SUFFIX_PAPER_MIN_TOKEN_PROB=0 \
    SUFFIX_PAPER_THRESHOLD=4 \
    SUFFIX_PAPER_MAX_MATCHES=0 \
    SUFFIX_PAPER_VERIFIER=linear \
    SUFFIX_PAPER_TREE_ATTN_IMPL=sdpa \
    SAVE_SUFFIX_TRACE=0 \
    PROFILER=0 \
    DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${draft_original}" \
    DRAFT_YARN_MAX_POSITION_EMBEDDINGS="$("${PYTHON}" - "${draft_original}" "${draft_factor}" <<'PY'
import math
import sys
print(int(math.ceil(float(sys.argv[1]) * float(sys.argv[2]))))
PY
)" \
    DRAFT_YARN_FACTOR="${draft_factor}" \
    DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS= \
    DRAFT_DYNAMIC_YARN_MAX_FACTOR= \
    DRAFT_DYNAMIC_YARN_MODE= \
    DRAFT_DYNAMIC_YARN_LENGTH_RATIO= \
    TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" \
    TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS}" \
    TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR}" \
    "${runner_cmd[@]}"
  echo
done

"${PYTHON}" "${SCRIPT_DIR}/summarize_static_yarn_scan.py" --run-dir "${OUTPUT_DIR}"
