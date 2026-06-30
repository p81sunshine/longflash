#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_transformers_single.sh}"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

PROTECT_SUPERVISOR_SIGNALS="${PROTECT_SUPERVISOR_SIGNALS:-1}"
if [[ "${PROTECT_SUPERVISOR_SIGNALS}" == "1" ]]; then
  _supervisor_signal() {
    echo "supervisor_received_signal=$1; keeping bucket runner alive" >&2
  }
  trap '_supervisor_signal TERM' TERM
  trap '_supervisor_signal HUP' HUP
fi

# Enable target YaRN for buckets whose lower bound reaches TARGET_ORIGINAL.
# Lowercase target_original is accepted for convenient one-off shell usage.
TARGET_YARN_ENABLED="${TARGET_YARN_ENABLED:-1}"
if [[ "${TARGET_YARN_ENABLED}" == "1" ]]; then
  TARGET_ORIGINAL="${TARGET_ORIGINAL:-${target_original:-32768}}"
  TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-4}"
  TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}"
else
  TARGET_ORIGINAL="${TARGET_ORIGINAL:-${target_original:-}}"
  TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-}"
  TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}"
fi
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
BUCKET_DIR="${BUCKET_DIR:-${ROOT_DIR}/benchmarks/swebench}"
RUN_PREFIX="${RUN_PREFIX:-current_dflash}"
DATASET_LABEL="${DATASET_LABEL:-}"

NUM_PROCS="${NUM_PROCS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29580}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NO_SAVE_RESPONSES="${NO_SAVE_RESPONSES:-0}"

# Space-separated bucket suffixes. Override with, for example:
#   BUCKETS="0_32768 32768_65536" bash scripts/run_transformers_buckets.sh
BUCKETS="${BUCKETS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/reproduced/transformers_buckets}"
CTX_SINK_TOKENS="${CTX_SINK_TOKENS:-0}"
CTX_RECENT_WINDOW="${CTX_RECENT_WINDOW:-0}"
# CTX_SINK_TOKENS="${CTX_SINK_TOKENS:-0}"
# CTX_RECENT_WINDOW="${CTX_RECENT_WINDOW:-0}"
CTX_STRIDE="${CTX_STRIDE:-0}"
# CTX_SUFFIX_MATCH_TOKENS="${CTX_SUFFIX_MATCH_TOKENS:-8}"
# CTX_SUFFIX_KEEP_TOKENS="${CTX_SUFFIX_KEEP_TOKENS:-128}"
CTX_SUFFIX_MATCH_TOKENS="${CTX_SUFFIX_MATCH_TOKENS:-0}"
CTX_SUFFIX_KEEP_TOKENS="${CTX_SUFFIX_KEEP_TOKENS:-0}"
# CTX_MIDDLE_BUDGET="${CTX_MIDDLE_BUDGET:-1000}"
CTX_MIDDLE_BUDGET="${CTX_MIDDLE_BUDGET:-0}"
CTX_TOTAL_BUDGET="${CTX_TOTAL_BUDGET:-}"
CTX_DYNAMIC_BUDGET_RATIO="${CTX_DYNAMIC_BUDGET_RATIO:-}"
CTX_BUDGET_ORDER="${CTX_BUDGET_ORDER:-default}"
CTX_INDEXER_ENABLE="${CTX_INDEXER_ENABLE:-0}"
CTX_INDEXER_BLOCK_SIZE="${CTX_INDEXER_BLOCK_SIZE:-4}"
CTX_INDEXER_TOP_K_BLOCKS="${CTX_INDEXER_TOP_K_BLOCKS:-512}"
CTX_INDEXER_QUERY_TOKENS="${CTX_INDEXER_QUERY_TOKENS:-512}"
CTX_INDEXER_SCORE_REDUCE="${CTX_INDEXER_SCORE_REDUCE:-max}"
DRAFT_DENOISE_STEPS="${DRAFT_DENOISE_STEPS:-1}"
DRAFT_SLIDING_WINDOW_SIZE="${DRAFT_SLIDING_WINDOW_SIZE:-}"
PROFILER="${PROFILER:-0}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/summary_table.csv}"
SUMMARY_MD="${SUMMARY_MD:-${OUTPUT_DIR}/summary_table.md}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
ISOLATE_RUNNER_PROCESS_GROUP="${ISOLATE_RUNNER_PROCESS_GROUP:-1}"


SAVE_VERIFY_TRACE="${SAVE_VERIFY_TRACE:-0}"
VERIFY_TRACE_MAX_ROUNDS="${VERIFY_TRACE_MAX_ROUNDS:-256}"
VERIFY_CONFIDENCE_THRESHOLD="${VERIFY_CONFIDENCE_THRESHOLD:-0}"
VERIFY_MIN_DRAFT_TOKENS="${VERIFY_MIN_DRAFT_TOKENS:-2}"
SUFFIX_DECODING="${SUFFIX_DECODING:-0}"
SUFFIX_STRATEGY="${SUFFIX_STRATEGY:-consensus}"
SUFFIX_MAX_QUERY_LEN="${SUFFIX_MAX_QUERY_LEN:-16}"
SUFFIX_MIN_QUERY_LEN="${SUFFIX_MIN_QUERY_LEN:-2}"
SUFFIX_TOP_K="${SUFFIX_TOP_K:-4}"
SUFFIX_MIN_SUPPORT="${SUFFIX_MIN_SUPPORT:-3}"
SUFFIX_MIN_PREDICT_LEN="${SUFFIX_MIN_PREDICT_LEN:-8}"
SUFFIX_MAX_PREDICT_LEN="${SUFFIX_MAX_PREDICT_LEN:-}"
SUFFIX_PAPER_ALPHA="${SUFFIX_PAPER_ALPHA:-1}"
SUFFIX_PAPER_MAX_SPEC_OFFSET="${SUFFIX_PAPER_MAX_SPEC_OFFSET:-0}"
SUFFIX_PAPER_MIN_TOKEN_PROB="${SUFFIX_PAPER_MIN_TOKEN_PROB:-0}"
SUFFIX_PAPER_THRESHOLD="${SUFFIX_PAPER_THRESHOLD:-0}"
SUFFIX_PAPER_MAX_MATCHES="${SUFFIX_PAPER_MAX_MATCHES:-0}"
SUFFIX_PAPER_VERIFIER="${SUFFIX_PAPER_VERIFIER:-linear}"
SUFFIX_PAPER_TREE_ATTN_IMPL="${SUFFIX_PAPER_TREE_ATTN_IMPL:-sdpa}"
SUFFIX_FALLBACK="${SUFFIX_FALLBACK:-dflash}"
SAVE_SUFFIX_TRACE="${SAVE_SUFFIX_TRACE:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
# BUCKETS="4096_8192"

if [[ "${TARGET_YARN_ENABLED}" == "1" && -n "${TARGET_ORIGINAL}" && ! "${TARGET_ORIGINAL}" =~ ^[0-9]+$ ]]; then
  echo "TARGET_ORIGINAL/target_original must be a positive integer, got: ${TARGET_ORIGINAL}" >&2
  exit 1
fi
if [[ "${TARGET_YARN_ENABLED}" == "1" && -n "${TARGET_ORIGINAL}" && "${TARGET_ORIGINAL}" -le 0 ]]; then
  echo "TARGET_ORIGINAL/target_original must be a positive integer, got: ${TARGET_ORIGINAL}" >&2
  exit 1
fi
if [[ "${TARGET_YARN_ENABLED}" == "1" && -n "${TARGET_ORIGINAL}" && -z "${TARGET_YARN_MAX_POSITION_EMBEDDINGS}" ]]; then
  TARGET_YARN_MAX_POSITION_EMBEDDINGS=$((TARGET_ORIGINAL * TARGET_YARN_FACTOR))
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${BUCKETS// }" ]]; then
  BUCKETS="$(
    find "${BUCKET_DIR}" -maxdepth 1 -name 'bucket_*.jsonl' -printf '%f\n' \
      | sed 's/^bucket_//; s/\.jsonl$//' \
      | sort -V \
      | tr '\n' ' '
  )"
fi

if [[ -z "${DATASET_LABEL}" ]]; then
  case "${BUCKET_DIR}" in
    *terminla_bench*|*terminal_bench*)
      DATASET_LABEL="terminal"
      ;;
    *swebench*)
      DATASET_LABEL="swebench"
      ;;
    *)
      DATASET_LABEL="$(basename "${BUCKET_DIR}" | tr -c '[:alnum:]_' '_')"
      DATASET_LABEL="${DATASET_LABEL%_}"
      ;;
  esac
fi

summary_matches_dataset() {
  local summary_path="$1"
  local expected_dataset="$2"
  local config_path
  config_path="$(dirname "${summary_path}")/run_config.json"
  [[ -f "${config_path}" ]] || return 1
  "${VENV_PYTHON}" - "${config_path}" "${expected_dataset}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text("utf-8"))
sys.exit(0 if config.get("dataset") == sys.argv[2] else 1)
PY
}

echo "runner=${RUNNER}"
echo "model=${MODEL}"
echo "draft_model=${DRAFT_MODEL}"
echo "bucket_dir=${BUCKET_DIR}"
echo "dataset_label=${DATASET_LABEL}"
echo "output_dir=${OUTPUT_DIR}"
echo "run_prefix=${RUN_PREFIX}"
echo "num_procs=${NUM_PROCS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "master_port_base=${MASTER_PORT_BASE}"
echo "max_samples=${MAX_SAMPLES}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "temperature=${TEMPERATURE}"
echo "sample_seed=${SAMPLE_SEED}"
echo "block_size=${BLOCK_SIZE}"
echo "ctx_sink_tokens=${CTX_SINK_TOKENS}"
echo "ctx_recent_window=${CTX_RECENT_WINDOW}"
echo "ctx_stride=${CTX_STRIDE}"
echo "ctx_suffix_match_tokens=${CTX_SUFFIX_MATCH_TOKENS}"
echo "ctx_suffix_keep_tokens=${CTX_SUFFIX_KEEP_TOKENS}"
echo "ctx_middle_budget=${CTX_MIDDLE_BUDGET}"
echo "ctx_total_budget=${CTX_TOTAL_BUDGET}"
echo "ctx_dynamic_budget_ratio=${CTX_DYNAMIC_BUDGET_RATIO}"
echo "ctx_budget_order=${CTX_BUDGET_ORDER}"
echo "ctx_indexer_enable=${CTX_INDEXER_ENABLE}"
echo "ctx_indexer_block_size=${CTX_INDEXER_BLOCK_SIZE}"
echo "ctx_indexer_top_k_blocks=${CTX_INDEXER_TOP_K_BLOCKS}"
echo "ctx_indexer_query_tokens=${CTX_INDEXER_QUERY_TOKENS}"
echo "ctx_indexer_score_reduce=${CTX_INDEXER_SCORE_REDUCE}"
echo "draft_denoise_steps=${DRAFT_DENOISE_STEPS}"
echo "profiler=${PROFILER}"
echo "draft_dynamic_yarn_original_max_position_embeddings=${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
echo "draft_dynamic_yarn_max_factor=${DRAFT_DYNAMIC_YARN_MAX_FACTOR:-}"
echo "draft_dynamic_yarn_mode=${DRAFT_DYNAMIC_YARN_MODE:-}"
echo "draft_dynamic_yarn_length_ratio=${DRAFT_DYNAMIC_YARN_LENGTH_RATIO:-}"
echo "draft_sliding_window_size=${DRAFT_SLIDING_WINDOW_SIZE}"
echo "target_yarn_enabled=${TARGET_YARN_ENABLED}"
echo "target_original=${TARGET_ORIGINAL}"
echo "target_yarn_max_position_embeddings=${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
echo "target_yarn_factor=${TARGET_YARN_FACTOR}"
echo "summary_csv=${SUMMARY_CSV}"
echo "summary_md=${SUMMARY_MD}"
echo "save_verify_trace=${SAVE_VERIFY_TRACE}"
echo "verify_trace_max_rounds=${VERIFY_TRACE_MAX_ROUNDS}"
echo "verify_confidence_threshold=${VERIFY_CONFIDENCE_THRESHOLD}"
echo "verify_min_draft_tokens=${VERIFY_MIN_DRAFT_TOKENS}"
echo "suffix_decoding=${SUFFIX_DECODING}"
echo "suffix_strategy=${SUFFIX_STRATEGY}"
echo "suffix_max_query_len=${SUFFIX_MAX_QUERY_LEN}"
echo "suffix_min_query_len=${SUFFIX_MIN_QUERY_LEN}"
echo "suffix_top_k=${SUFFIX_TOP_K}"
echo "suffix_min_support=${SUFFIX_MIN_SUPPORT}"
echo "suffix_min_predict_len=${SUFFIX_MIN_PREDICT_LEN}"
echo "suffix_max_predict_len=${SUFFIX_MAX_PREDICT_LEN}"
echo "suffix_paper_alpha=${SUFFIX_PAPER_ALPHA}"
echo "suffix_paper_max_spec_offset=${SUFFIX_PAPER_MAX_SPEC_OFFSET}"
echo "suffix_paper_min_token_prob=${SUFFIX_PAPER_MIN_TOKEN_PROB}"
echo "suffix_paper_threshold=${SUFFIX_PAPER_THRESHOLD}"
echo "suffix_paper_max_matches=${SUFFIX_PAPER_MAX_MATCHES}"
echo "suffix_fallback=${SUFFIX_FALLBACK}"
echo "suffix_paper_verifier=${SUFFIX_PAPER_VERIFIER}"
echo "suffix_paper_tree_attn_impl=${SUFFIX_PAPER_TREE_ATTN_IMPL}"
echo "save_suffix_trace=${SAVE_SUFFIX_TRACE}"
echo "buckets=${BUCKETS}"
echo "dry_run=${DRY_RUN}"
echo "skip_completed=${SKIP_COMPLETED}"
echo "continue_on_failure=${CONTINUE_ON_FAILURE}"
echo "isolate_runner_process_group=${ISOLATE_RUNNER_PROCESS_GROUP}"
echo "protect_supervisor_signals=${PROTECT_SUPERVISOR_SIGNALS}"
echo

idx=0
for bucket in ${BUCKETS}; do
  data_set="${BUCKET_DIR}/bucket_${bucket}.jsonl"
  if [[ ! -f "${data_set}" ]]; then
    echo "Missing bucket file: ${data_set}" >&2
    exit 1
  fi

  idx=$((idx + 1))
  run_name="${RUN_PREFIX}_${DATASET_LABEL}_${bucket}_ms${MAX_SAMPLES}_mn${MAX_NEW_TOKENS}"
  master_port=$((MASTER_PORT_BASE + idx))
  bucket_start="${bucket%%_*}"
  use_target_yarn=0
  if [[ "${TARGET_YARN_ENABLED}" == "1" && -n "${TARGET_ORIGINAL}" && "${bucket_start}" -ge "${TARGET_ORIGINAL}" ]]; then
    use_target_yarn=1
  fi

  echo "===== ${run_name} ====="
  echo "target_yarn=${use_target_yarn}"
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    existing_summary=""
    while IFS= read -r candidate_summary; do
      if summary_matches_dataset "${candidate_summary}" "${data_set}"; then
        existing_summary="${candidate_summary}"
        break
      fi
    done < <(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -name summary.json -path "*_${run_name}/summary.json" -print)
    if [[ -n "${existing_summary}" ]]; then
      echo "skip_completed=1: found ${existing_summary}"
      echo
      continue
    fi
  fi
  RUN_ENV=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "NUM_PROCS=${NUM_PROCS}"
    "MASTER_PORT=${master_port}"
    "MODEL=${MODEL}"
    "DRAFT_MODEL=${DRAFT_MODEL}"
    "DATA_SET=${data_set}"
    "MAX_SAMPLES=${MAX_SAMPLES}"
    "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
    "TEMPERATURE=${TEMPERATURE}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "RUN_NAME=${run_name}"
    "SAMPLE_SEED=${SAMPLE_SEED}"
    "BLOCK_SIZE=${BLOCK_SIZE}"
    "NO_SAVE_RESPONSES=${NO_SAVE_RESPONSES}"
    "CTX_SINK_TOKENS=${CTX_SINK_TOKENS}"
    "CTX_RECENT_WINDOW=${CTX_RECENT_WINDOW}"
    "CTX_STRIDE=${CTX_STRIDE}"
    "CTX_SUFFIX_MATCH_TOKENS=${CTX_SUFFIX_MATCH_TOKENS}"
    "CTX_SUFFIX_KEEP_TOKENS=${CTX_SUFFIX_KEEP_TOKENS}"
    "CTX_MIDDLE_BUDGET=${CTX_MIDDLE_BUDGET}"
    "CTX_BUDGET_ORDER=${CTX_BUDGET_ORDER}"
    "CTX_INDEXER_ENABLE=${CTX_INDEXER_ENABLE}"
    "CTX_INDEXER_BLOCK_SIZE=${CTX_INDEXER_BLOCK_SIZE}"
    "CTX_INDEXER_TOP_K_BLOCKS=${CTX_INDEXER_TOP_K_BLOCKS}"
    "CTX_INDEXER_QUERY_TOKENS=${CTX_INDEXER_QUERY_TOKENS}"
    "CTX_INDEXER_SCORE_REDUCE=${CTX_INDEXER_SCORE_REDUCE}"
    "DRAFT_DENOISE_STEPS=${DRAFT_DENOISE_STEPS}"
    "SAVE_VERIFY_TRACE=${SAVE_VERIFY_TRACE}"
    "VERIFY_TRACE_MAX_ROUNDS=${VERIFY_TRACE_MAX_ROUNDS}"
    "VERIFY_CONFIDENCE_THRESHOLD=${VERIFY_CONFIDENCE_THRESHOLD}"
    "VERIFY_MIN_DRAFT_TOKENS=${VERIFY_MIN_DRAFT_TOKENS}"
    "SUFFIX_DECODING=${SUFFIX_DECODING}"
    "SUFFIX_STRATEGY=${SUFFIX_STRATEGY}"
    "SUFFIX_MAX_QUERY_LEN=${SUFFIX_MAX_QUERY_LEN}"
    "SUFFIX_MIN_QUERY_LEN=${SUFFIX_MIN_QUERY_LEN}"
    "SUFFIX_TOP_K=${SUFFIX_TOP_K}"
    "SUFFIX_MIN_SUPPORT=${SUFFIX_MIN_SUPPORT}"
    "SUFFIX_MIN_PREDICT_LEN=${SUFFIX_MIN_PREDICT_LEN}"
    "SUFFIX_MAX_PREDICT_LEN=${SUFFIX_MAX_PREDICT_LEN}"
    "SUFFIX_PAPER_ALPHA=${SUFFIX_PAPER_ALPHA}"
    "SUFFIX_PAPER_MAX_SPEC_OFFSET=${SUFFIX_PAPER_MAX_SPEC_OFFSET}"
    "SUFFIX_PAPER_MIN_TOKEN_PROB=${SUFFIX_PAPER_MIN_TOKEN_PROB}"
    "SUFFIX_PAPER_THRESHOLD=${SUFFIX_PAPER_THRESHOLD}"
    "SUFFIX_PAPER_MAX_MATCHES=${SUFFIX_PAPER_MAX_MATCHES}"
    "SUFFIX_PAPER_VERIFIER=${SUFFIX_PAPER_VERIFIER}"
    "SUFFIX_PAPER_TREE_ATTN_IMPL=${SUFFIX_PAPER_TREE_ATTN_IMPL}"
    "SUFFIX_FALLBACK=${SUFFIX_FALLBACK}"
    "SAVE_SUFFIX_TRACE=${SAVE_SUFFIX_TRACE}"
    "PROFILER=${PROFILER}"
    "DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
    "DRAFT_YARN_MAX_POSITION_EMBEDDINGS="
    "DRAFT_YARN_FACTOR="
    "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
    "DRAFT_DYNAMIC_YARN_MAX_FACTOR="
    "DRAFT_DYNAMIC_YARN_MODE="
    "DRAFT_DYNAMIC_YARN_LENGTH_RATIO="
    "DRAFT_SLIDING_WINDOW_SIZE="
    "TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="
    "TARGET_YARN_MAX_POSITION_EMBEDDINGS="
    "TARGET_YARN_FACTOR="
  )
  if [[ -n "${CTX_TOTAL_BUDGET}" ]]; then
    RUN_ENV+=("CTX_TOTAL_BUDGET=${CTX_TOTAL_BUDGET}")
  fi
  if [[ -n "${CTX_DYNAMIC_BUDGET_RATIO}" ]]; then
    RUN_ENV+=("CTX_DYNAMIC_BUDGET_RATIO=${CTX_DYNAMIC_BUDGET_RATIO}")
  fi
  if [[ -n "${DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}" ]]; then
    RUN_ENV+=("DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}")
  fi
  if [[ -n "${DRAFT_YARN_MAX_POSITION_EMBEDDINGS:-}" ]]; then
    RUN_ENV+=("DRAFT_YARN_MAX_POSITION_EMBEDDINGS=${DRAFT_YARN_MAX_POSITION_EMBEDDINGS}")
  fi
  if [[ -n "${DRAFT_YARN_FACTOR:-}" ]]; then
    RUN_ENV+=("DRAFT_YARN_FACTOR=${DRAFT_YARN_FACTOR}")
  fi
  if [[ -n "${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}" ]]; then
    RUN_ENV+=("DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}")
  fi
  if [[ -n "${DRAFT_DYNAMIC_YARN_MAX_FACTOR:-}" ]]; then
    RUN_ENV+=("DRAFT_DYNAMIC_YARN_MAX_FACTOR=${DRAFT_DYNAMIC_YARN_MAX_FACTOR}")
  fi
  if [[ -n "${DRAFT_DYNAMIC_YARN_MODE:-}" ]]; then
    RUN_ENV+=("DRAFT_DYNAMIC_YARN_MODE=${DRAFT_DYNAMIC_YARN_MODE}")
  fi
  if [[ -n "${DRAFT_DYNAMIC_YARN_LENGTH_RATIO:-}" ]]; then
    RUN_ENV+=("DRAFT_DYNAMIC_YARN_LENGTH_RATIO=${DRAFT_DYNAMIC_YARN_LENGTH_RATIO}")
  fi
  if [[ -n "${DRAFT_SLIDING_WINDOW_SIZE}" ]]; then
    RUN_ENV+=("DRAFT_SLIDING_WINDOW_SIZE=${DRAFT_SLIDING_WINDOW_SIZE}")
  fi
  if [[ "${use_target_yarn}" == "1" ]]; then
    RUN_ENV+=(
      "TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${TARGET_ORIGINAL}"
      "TARGET_YARN_MAX_POSITION_EMBEDDINGS=${TARGET_YARN_MAX_POSITION_EMBEDDINGS}"
      "TARGET_YARN_FACTOR=${TARGET_YARN_FACTOR}"
    )
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'env'
    printf ' %q' "${RUN_ENV[@]}"
    printf ' bash %q\n' "${RUNNER}"
  else
    set +e
    if [[ "${ISOLATE_RUNNER_PROCESS_GROUP}" == "1" ]]; then
      setsid env "${RUN_ENV[@]}" bash "${RUNNER}"
    else
      env "${RUN_ENV[@]}" bash "${RUNNER}"
    fi
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      echo "runner_exit_status=${status}"
      if [[ "${CONTINUE_ON_FAILURE}" == "1" ]]; then
        latest_run_dir="$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" | sort | tail -n1)"
        if [[ -n "${latest_run_dir}" ]]; then
          "${VENV_PYTHON}" - "${latest_run_dir}" "${run_name}" "${status}" "${MAX_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
run_name = sys.argv[2]
exit_status = int(sys.argv[3])
max_samples = int(sys.argv[4])

summary = {
    "samples_total": max_samples,
    "samples_ok": 0,
    "samples_skipped": 0,
    "samples_oom": 0,
    "samples_error": max_samples,
    "mean_prompt_tokens": None,
    "mean_baseline_tpot": None,
    "mean_dflash_tpot": None,
    "mean_speedup": None,
    "mean_acceptance_length": None,
    "mean_acceptance_rate": None,
    "mean_draft_forward_passes": None,
    "mean_draft_dynamic_yarn_factor": None,
    "min_draft_dynamic_yarn_factor": None,
    "max_draft_dynamic_yarn_factor": None,
    "mean_verify_draft_tokens": None,
    "mean_ctx_suffix_match_count": None,
    "max_ctx_suffix_match_count": None,
    "total_ctx_suffix_match_count": 0,
    "mean_ctx_suffix_match_kept_tokens": None,
    "mean_ctx_middle_tokens_before_budget": None,
    "mean_ctx_middle_tokens_after_budget": None,
    "mean_ctx_middle_budget_dropped_tokens": None,
    "mean_ctx_total_budget": None,
    "mean_ctx_recent_tokens_after_budget": None,
    "mean_ctx_hidden_tokens_after": None,
    "mean_ctx_indexer_selected_blocks": None,
    "mean_ctx_indexer_forced_blocks": None,
    "throughput_baseline_tok_s": None,
    "throughput_dflash_tok_s": None,
    "run_name": run_name,
    "runner_exit_status": exit_status,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
PY
        fi
      fi
      echo
      continue
    fi
  fi
  echo
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_swebench_matrix_results.py" \
    --run-dir "${OUTPUT_DIR}" \
    --output-csv "${SUMMARY_CSV}" \
    --output-md "${SUMMARY_MD}"
fi

echo "Done. output_dir=${OUTPUT_DIR}"
echo "summary_csv=${SUMMARY_CSV}"
echo "summary_md=${SUMMARY_MD}"
