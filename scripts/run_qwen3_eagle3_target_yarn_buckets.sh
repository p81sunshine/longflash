#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SINGLE_RUNNER="${SINGLE_RUNNER:-${SCRIPT_DIR}/run_eagle3_transformers.py}"
EAGLE_ROOT="${EAGLE_ROOT:-${ROOT_DIR}/third_party/EAGLE}"
EAGLE_PYTHON="${EAGLE_PYTHON:-${EAGLE_ROOT}/.venv/bin/python}"
EAGLE_TORCHRUN="${EAGLE_TORCHRUN:-${EAGLE_ROOT}/.venv/bin/torchrun}"
if [[ ! -x "${EAGLE_TORCHRUN}" ]]; then
  EAGLE_TORCHRUN="$(command -v torchrun || true)"
fi
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "${EAGLE_TORCHRUN}" && "${DRY_RUN}" == "1" ]]; then
  EAGLE_TORCHRUN="torchrun"
fi

TERMINAL_BUCKET_DIR="${TERMINAL_BUCKET_DIR:-${ROOT_DIR}/benchmarks/terminal}"
SWEBENCH_BUCKET_DIR="${SWEBENCH_BUCKET_DIR:-${ROOT_DIR}/benchmarks/swebench}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/reproduced/eagle3_transformers}"

MODELS="${MODELS:-qwen3-8b qwen3-4b}"
TEMPERATURES="${TEMPERATURES:-0 1}"
VARIANTS="${VARIANTS:-eagle3_spec16 eagle3_tree60}"
DATASET_GROUPS="${DATASET_GROUPS:-terminal swebench}"

NUM_PROCS="${NUM_PROCS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30900}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
NO_SAVE_RESPONSES="${NO_SAVE_RESPONSES:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
DEVICE_MAP="${DEVICE_MAP:-}"
ONLY_RUN_NAMES="${ONLY_RUN_NAMES:-}"
TARGET_PREFILL_ATTN_IMPLEMENTATION="${TARGET_PREFILL_ATTN_IMPLEMENTATION:-flash_attention_2}"
TARGET_VERIFY_ATTN_IMPLEMENTATION="${TARGET_VERIFY_ATTN_IMPLEMENTATION:-sdpa}"
DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-flash_attention_2}"
TARGET_ORIGINAL="${TARGET_ORIGINAL:-32768}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-4}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-$((TARGET_ORIGINAL * TARGET_YARN_FACTOR))}"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${EAGLE_ROOT}:${ROOT_DIR}:${PYTHONPATH:-}"

model_paths() {
  local model_key="$1"
  case "${model_key}" in
    qwen3-8b)
      MODEL_PATH="Qwen/Qwen3-8B"
      EAGLE_MODEL_PATH="${EAGLE3_QWEN3_8B_MODEL:?set EAGLE3_QWEN3_8B_MODEL to the Qwen3-8B EAGLE-3 draft model path}"
      MODEL_LABEL="qwen3_8b"
      ;;
    qwen3-4b)
      MODEL_PATH="Qwen/Qwen3-4B"
      EAGLE_MODEL_PATH="${EAGLE3_QWEN3_4B_MODEL:?set EAGLE3_QWEN3_4B_MODEL to the Qwen3-4B EAGLE-3 draft model path}"
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
  case "${variant}" in
    eagle3_spec16)
      VARIANT_LABEL="eagle3_spec16"
      TOTAL_TOKEN=16
      EAGLE_DEPTH=8
      EAGLE_TOP_K=4
      ;;
    eagle3_tree60)
      VARIANT_LABEL="eagle3_tree60"
      TOTAL_TOKEN=60
      EAGLE_DEPTH=7
      EAGLE_TOP_K=10
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

run_name_selected() {
  local run_name="$1"
  if [[ -z "${ONLY_RUN_NAMES// }" ]]; then
    return 0
  fi
  local selected
  for selected in ${ONLY_RUN_NAMES}; do
    if [[ "${selected}" == "${run_name}" ]]; then
      return 0
    fi
  done
  return 1
}

run_single() {
  local dataset="$1"
  local run_name="$2"
  local port="$3"
  local env_args=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "PYTHONPATH=${PYTHONPATH}"
  )
  local args=(
    --model "${MODEL_PATH}"
    --eagle-model "${EAGLE_MODEL_PATH}"
    --dataset "${dataset}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${run_name}"
    --max-samples "${MAX_SAMPLES}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --sample-seed "${SAMPLE_SEED}"
    --total-token "${TOTAL_TOKEN}"
    --depth "${EAGLE_DEPTH}"
    --top-k "${EAGLE_TOP_K}"
    --target-yarn-original-max-position-embeddings "${TARGET_ORIGINAL}"
    --target-yarn-max-position-embeddings "${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
    --target-yarn-factor "${TARGET_YARN_FACTOR}"
    --target-prefill-attn-implementation "${TARGET_PREFILL_ATTN_IMPLEMENTATION}"
    --target-verify-attn-implementation "${TARGET_VERIFY_ATTN_IMPLEMENTATION}"
    --draft-attn-implementation "${DRAFT_ATTN_IMPLEMENTATION}"
  )
  if [[ "${NO_SAVE_RESPONSES}" == "1" ]]; then
    args+=(--no-save-responses)
  fi
  if [[ "${ENABLE_THINKING}" == "1" ]]; then
    args+=(--enable-thinking)
  fi
  if [[ -n "${DEVICE_MAP}" ]]; then
    args+=(--device-map "${DEVICE_MAP}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'env'
    printf ' %q' "${env_args[@]}"
    if [[ "${NUM_PROCS}" -gt 1 ]]; then
      printf ' %q' "${EAGLE_TORCHRUN}" --nproc_per_node="${NUM_PROCS}" --master_port "${port}"
    else
      printf ' %q' "${EAGLE_PYTHON}"
    fi
    printf ' %q' "${SINGLE_RUNNER}"
    printf ' %q' "${args[@]}"
    printf '\n'
    return 0
  fi

  set +e
  if [[ "${NUM_PROCS}" -gt 1 ]]; then
    env "${env_args[@]}" "${EAGLE_TORCHRUN}" \
      --nproc_per_node="${NUM_PROCS}" \
      --master_port "${port}" \
      "${SINGLE_RUNNER}" \
      "${args[@]}"
  else
    env "${env_args[@]}" "${EAGLE_PYTHON}" "${SINGLE_RUNNER}" "${args[@]}"
  fi
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "runner_exit_status=${status}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${status}"
    fi
  fi
}

run_dataset() {
  local dataset_label="$1"
  local dataset="$2"
  local run_name="$3"
  local port="$4"
  if ! run_name_selected "${run_name}"; then
    return
  fi
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    local existing_summary
    existing_summary="$(
      find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -name summary.json -path "*_${run_name}/summary.json" -print \
        | while IFS= read -r summary_path; do
            if "${ROOT_DIR}/.venv/bin/python" - "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text("utf-8"))
samples_total = int(summary.get("samples_total") or 0)
samples_ok = int(summary.get("samples_ok") or 0)
samples_oom = int(summary.get("samples_oom") or 0)
samples_error = int(summary.get("samples_error") or 0)
raise SystemExit(0 if samples_total > 0 and samples_ok == samples_total and samples_oom == 0 and samples_error == 0 else 1)
PY
            then
              echo "${summary_path}"
              break
            fi
          done
    )"
    if [[ -n "${existing_summary}" ]]; then
      echo "===== ${MODEL_LABEL} temp=${TEMPERATURE} ${dataset_label} ${VARIANT_LABEL} ====="
      echo "skip_completed=1: found ${existing_summary}"
      echo
      return
    fi
  fi

  echo "===== ${MODEL_LABEL} temp=${TEMPERATURE} ${dataset_label} ${VARIANT_LABEL} ====="
  echo "dataset=${dataset}"
  echo "run_name=${run_name}"
  echo "target_yarn_original=${TARGET_ORIGINAL}"
  echo "target_yarn_max_position_embeddings=${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
  echo "target_yarn_factor=${TARGET_YARN_FACTOR}"
  echo "target_prefill_attn_implementation=${TARGET_PREFILL_ATTN_IMPLEMENTATION}"
  echo "target_verify_attn_implementation=${TARGET_VERIFY_ATTN_IMPLEMENTATION}"
  echo "draft_attn_implementation=${DRAFT_ATTN_IMPLEMENTATION}"
  echo "eagle_total_token=${TOTAL_TOKEN}"
  echo "eagle_depth=${EAGLE_DEPTH}"
  echo "eagle_top_k=${EAGLE_TOP_K}"
  run_single "${dataset}" "${run_name}" "${port}"
  echo
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

  echo "bucket_dir=${bucket_dir}"
  echo "buckets=${buckets}"
  local idx=0
  for bucket in ${buckets}; do
    idx=$((idx + 1))
    local dataset="${bucket_dir}/bucket_${bucket}.jsonl"
    local run_name="original_${MODEL_LABEL}_temp${TEMP_LABEL}_${VARIANT_LABEL}_${group_name}_${bucket}_ms${MAX_SAMPLES}_mn${MAX_NEW_TOKENS}"
    run_dataset "${group_name}/${bucket}" "${dataset}" "${run_name}" "$((port_base + idx))"
  done
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
echo "target_original=${TARGET_ORIGINAL}"
echo "target_yarn_max_position_embeddings=${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
echo "target_prefill_attn_implementation=${TARGET_PREFILL_ATTN_IMPLEMENTATION}"
echo "target_verify_attn_implementation=${TARGET_VERIFY_ATTN_IMPLEMENTATION}"
echo "draft_attn_implementation=${DRAFT_ATTN_IMPLEMENTATION}"
echo "device_map=${DEVICE_MAP}"
echo

run_idx=0
for model_key in ${MODELS}; do
  model_paths "${model_key}"
  for TEMPERATURE in ${TEMPERATURES}; do
    TEMP_LABEL="${TEMPERATURE//./p}"
    for variant in ${VARIANTS}; do
      variant_env "${variant}"
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
