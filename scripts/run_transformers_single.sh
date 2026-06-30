#!/usr/bin/env bash
set -euo pipefail

NUM_PROCS="${NUM_PROCS:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
TORCHRUN="${TORCHRUN:-${SCRIPT_DIR}/../.venv/bin/torchrun}"
if [[ ! -x "${TORCHRUN}" ]]; then
  TORCHRUN="$(command -v torchrun)"
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
# DATA_SET="${DATA_SET:-cache/swebench-buckets/kimi-k2-5-high/bucket_4096_8192.jsonl}"
DATA_SET="${DATA_SET:-${SCRIPT_DIR}/../benchmarks/swebench/bucket_65536_122880.jsonl}"
TARGET_DEVICE_MAP="${TARGET_DEVICE_MAP:-}"
DRAFT_DEVICE_MAP="${DRAFT_DEVICE_MAP:-}"
TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
TARGET_YARN_FACTOR="${TARGET_YARN_FACTOR:-}"
TARGET_YARN_MAX_POSITION_EMBEDDINGS="${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../results/reproduced/smoke_transformers}"
RUN_NAME="${RUN_NAME:-transformers_profile}"
DRAFT_DENOISE_STEPS="${DRAFT_DENOISE_STEPS:-1}"

MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
CTX_SINK_TOKENS="${CTX_SINK_TOKENS:-1000}"
CTX_RECENT_WINDOW="${CTX_RECENT_WINDOW:-2072}"
CTX_STRIDE="${CTX_STRIDE:-0}"
CTX_SUFFIX_MATCH_TOKENS="${CTX_SUFFIX_MATCH_TOKENS:-0}"
CTX_SUFFIX_KEEP_TOKENS="${CTX_SUFFIX_KEEP_TOKENS:-0}"
CTX_MIDDLE_BUDGET="${CTX_MIDDLE_BUDGET:-0}"
CTX_TOTAL_BUDGET="${CTX_TOTAL_BUDGET:-}"
CTX_DYNAMIC_BUDGET_RATIO="${CTX_DYNAMIC_BUDGET_RATIO:-}"
CTX_BUDGET_ORDER="${CTX_BUDGET_ORDER:-default}"
CTX_INDEXER_ENABLE="${CTX_INDEXER_ENABLE:-0}"
CTX_INDEXER_BLOCK_SIZE="${CTX_INDEXER_BLOCK_SIZE:-4}"
CTX_INDEXER_TOP_K_BLOCKS="${CTX_INDEXER_TOP_K_BLOCKS:-512}"
CTX_INDEXER_QUERY_TOKENS="${CTX_INDEXER_QUERY_TOKENS:-512}"
CTX_INDEXER_SCORE_REDUCE="${CTX_INDEXER_SCORE_REDUCE:-max}"
VERIFY_TRACE_MAX_ROUNDS="${VERIFY_TRACE_MAX_ROUNDS:-0}"
VERIFY_CONFIDENCE_THRESHOLD="${VERIFY_CONFIDENCE_THRESHOLD:-0}"
VERIFY_MIN_DRAFT_TOKENS="${VERIFY_MIN_DRAFT_TOKENS:-1}"
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
export TRANSFORMERS_ATTN_IMPL="${TRANSFORMERS_ATTN_IMPL:-flash_attention_2}"
SUFFIX_PAPER_TREE_ATTN_IMPL="${SUFFIX_PAPER_TREE_ATTN_IMPL:-${TRANSFORMERS_ATTN_IMPL:-flash_attention_2}}"
SUFFIX_FALLBACK="${SUFFIX_FALLBACK:-dflash}"
SAVE_SUFFIX_TRACE="${SAVE_SUFFIX_TRACE:-0}"
PROFILER="${PROFILER:-1}"


EXTRA_ARGS=()
if [[ -n "${BLOCK_SIZE}" ]]; then
  EXTRA_ARGS+=(--block-size "${BLOCK_SIZE}")
fi
if [[ -n "${DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}" ]]; then
  EXTRA_ARGS+=(--draft-yarn-original-max-position-embeddings "${DRAFT_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${DRAFT_YARN_MAX_POSITION_EMBEDDINGS:-}" ]]; then
  EXTRA_ARGS+=(--draft-yarn-max-position-embeddings "${DRAFT_YARN_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${DRAFT_YARN_FACTOR:-}" ]]; then
  EXTRA_ARGS+=(--draft-yarn-factor "${DRAFT_YARN_FACTOR}")
fi
if [[ -n "${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}" ]]; then
  EXTRA_ARGS+=(--draft-dynamic-yarn-original-max-position-embeddings "${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${DRAFT_DYNAMIC_YARN_MAX_FACTOR:-}" ]]; then
  EXTRA_ARGS+=(--draft-dynamic-yarn-max-factor "${DRAFT_DYNAMIC_YARN_MAX_FACTOR}")
fi
if [[ -n "${DRAFT_DYNAMIC_YARN_MODE:-}" ]]; then
  EXTRA_ARGS+=(--draft-dynamic-yarn-mode "${DRAFT_DYNAMIC_YARN_MODE}")
fi
if [[ -n "${DRAFT_DYNAMIC_YARN_LENGTH_RATIO:-}" ]]; then
  EXTRA_ARGS+=(--draft-dynamic-yarn-length-ratio "${DRAFT_DYNAMIC_YARN_LENGTH_RATIO}")
fi
DRAFT_SLIDING_WINDOW_SIZE="${DRAFT_SLIDING_WINDOW_SIZE:-}"
if [[ -n "${DRAFT_SLIDING_WINDOW_SIZE}" ]]; then
  EXTRA_ARGS+=(--draft-sliding-window-size "${DRAFT_SLIDING_WINDOW_SIZE}")
fi
if [[ -n "${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}" ]]; then
  EXTRA_ARGS+=(--target-yarn-original-max-position-embeddings "${TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${TARGET_YARN_MAX_POSITION_EMBEDDINGS:-}" ]]; then
  EXTRA_ARGS+=(--target-yarn-max-position-embeddings "${TARGET_YARN_MAX_POSITION_EMBEDDINGS}")
fi
if [[ -n "${TARGET_YARN_FACTOR:-}" ]]; then
  EXTRA_ARGS+=(--target-yarn-factor "${TARGET_YARN_FACTOR}")
fi
if [[ -n "${TARGET_DEVICE_MAP}" ]]; then
  EXTRA_ARGS+=(--target-device-map "${TARGET_DEVICE_MAP}")
fi
if [[ -n "${DRAFT_DEVICE_MAP}" ]]; then
  EXTRA_ARGS+=(--draft-device-map "${DRAFT_DEVICE_MAP}")
fi
if [[ "${NO_SAVE_RESPONSES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-save-responses)
fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-thinking)
fi
if [[ "${SAVE_VERIFY_TRACE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-verify-trace)
fi
if [[ "${SUFFIX_DECODING}" == "1" ]]; then
  EXTRA_ARGS+=(--suffix-decoding)
fi
if [[ -n "${SUFFIX_STRATEGY}" ]]; then
  EXTRA_ARGS+=(--suffix-strategy "${SUFFIX_STRATEGY}")
fi
if [[ -n "${SUFFIX_MAX_QUERY_LEN}" ]]; then
  EXTRA_ARGS+=(--suffix-max-query-len "${SUFFIX_MAX_QUERY_LEN}")
fi
if [[ -n "${SUFFIX_MIN_QUERY_LEN}" ]]; then
  EXTRA_ARGS+=(--suffix-min-query-len "${SUFFIX_MIN_QUERY_LEN}")
fi
if [[ -n "${SUFFIX_TOP_K}" ]]; then
  EXTRA_ARGS+=(--suffix-top-k "${SUFFIX_TOP_K}")
fi
if [[ -n "${SUFFIX_MIN_SUPPORT}" ]]; then
  EXTRA_ARGS+=(--suffix-min-support "${SUFFIX_MIN_SUPPORT}")
fi
if [[ -n "${SUFFIX_MIN_PREDICT_LEN}" ]]; then
  EXTRA_ARGS+=(--suffix-min-predict-len "${SUFFIX_MIN_PREDICT_LEN}")
fi
if [[ -n "${SUFFIX_MAX_PREDICT_LEN}" ]]; then
  EXTRA_ARGS+=(--suffix-max-predict-len "${SUFFIX_MAX_PREDICT_LEN}")
fi
if [[ -n "${SUFFIX_PAPER_ALPHA}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-alpha "${SUFFIX_PAPER_ALPHA}")
fi
if [[ -n "${SUFFIX_PAPER_MAX_SPEC_OFFSET}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-max-spec-offset "${SUFFIX_PAPER_MAX_SPEC_OFFSET}")
fi
if [[ -n "${SUFFIX_PAPER_MIN_TOKEN_PROB}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-min-token-prob "${SUFFIX_PAPER_MIN_TOKEN_PROB}")
fi
if [[ -n "${SUFFIX_PAPER_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-threshold "${SUFFIX_PAPER_THRESHOLD}")
fi
if [[ -n "${SUFFIX_PAPER_MAX_MATCHES}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-max-matches "${SUFFIX_PAPER_MAX_MATCHES}")
fi
if [[ -n "${SUFFIX_PAPER_VERIFIER}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-verifier "${SUFFIX_PAPER_VERIFIER}")
fi
if [[ -n "${SUFFIX_PAPER_TREE_ATTN_IMPL}" ]]; then
  EXTRA_ARGS+=(--suffix-paper-tree-attn-impl "${SUFFIX_PAPER_TREE_ATTN_IMPL}")
fi
if [[ -n "${SUFFIX_FALLBACK}" ]]; then
  EXTRA_ARGS+=(--suffix-fallback "${SUFFIX_FALLBACK}")
fi
if [[ "${SAVE_SUFFIX_TRACE}" == "1" ]]; then
  EXTRA_ARGS+=(--save-suffix-trace)
fi
if [[ "${PROFILER}" == "1" ]]; then
  EXTRA_ARGS+=(--profiler)
fi
if [[ "${CTX_INDEXER_ENABLE}" == "1" ]]; then
  EXTRA_ARGS+=(--ctx-indexer-enable)
  EXTRA_ARGS+=(--ctx-indexer-block-size "${CTX_INDEXER_BLOCK_SIZE}")
  EXTRA_ARGS+=(--ctx-indexer-top-k-blocks "${CTX_INDEXER_TOP_K_BLOCKS}")
  EXTRA_ARGS+=(--ctx-indexer-query-tokens "${CTX_INDEXER_QUERY_TOKENS}")
  EXTRA_ARGS+=(--ctx-indexer-score-reduce "${CTX_INDEXER_SCORE_REDUCE}")
fi

echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "num_procs=${NUM_PROCS}"
echo "master_port=${MASTER_PORT}"
echo "model=${MODEL}"
echo "draft_model=${DRAFT_MODEL}"
echo "dataset=${DATA_SET}"
echo "transformers_attn_impl=${TRANSFORMERS_ATTN_IMPL}"
echo "target_device_map=${TARGET_DEVICE_MAP}"
echo "draft_device_map=${DRAFT_DEVICE_MAP}"
echo "max_samples=${MAX_SAMPLES}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "temperature=${TEMPERATURE}"
echo "output_dir=${OUTPUT_DIR}"
echo "run_name=${RUN_NAME}"
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
echo "draft_dynamic_yarn_original_max_position_embeddings=${DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}"
echo "draft_dynamic_yarn_max_factor=${DRAFT_DYNAMIC_YARN_MAX_FACTOR:-}"
echo "draft_dynamic_yarn_mode=${DRAFT_DYNAMIC_YARN_MODE:-}"
echo "draft_dynamic_yarn_length_ratio=${DRAFT_DYNAMIC_YARN_LENGTH_RATIO:-}"
echo "draft_sliding_window_size=${DRAFT_SLIDING_WINDOW_SIZE}"
echo "save_verify_trace=${SAVE_VERIFY_TRACE:-0}"
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
echo "suffix_paper_verifier=${SUFFIX_PAPER_VERIFIER}"
echo "suffix_paper_tree_attn_impl=${SUFFIX_PAPER_TREE_ATTN_IMPL}"
echo "suffix_fallback=${SUFFIX_FALLBACK}"
echo "save_suffix_trace=${SAVE_SUFFIX_TRACE}"
echo "profiler=${PROFILER}"
echo "extra_args=${EXTRA_ARGS[*]}"

COMMON_ARGS=(
  --model "${MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --dataset "${DATA_SET}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --sample-seed "${SAMPLE_SEED}"
  --ctx-sink-tokens "${CTX_SINK_TOKENS}"
  --ctx-recent-window "${CTX_RECENT_WINDOW}"
  --ctx-stride "${CTX_STRIDE}"
  --ctx-suffix-match-tokens "${CTX_SUFFIX_MATCH_TOKENS}"
  --ctx-suffix-keep-tokens "${CTX_SUFFIX_KEEP_TOKENS}"
  --ctx-middle-budget "${CTX_MIDDLE_BUDGET}"
  --ctx-budget-order "${CTX_BUDGET_ORDER}"
  --draft-denoise-steps "${DRAFT_DENOISE_STEPS}"
  --verify-trace-max-rounds "${VERIFY_TRACE_MAX_ROUNDS}"
  --verify-confidence-threshold "${VERIFY_CONFIDENCE_THRESHOLD}"
  --verify-min-draft-tokens "${VERIFY_MIN_DRAFT_TOKENS}"
)
if [[ -n "${CTX_TOTAL_BUDGET}" ]]; then
  COMMON_ARGS+=(--ctx-total-budget "${CTX_TOTAL_BUDGET}")
fi
if [[ -n "${CTX_DYNAMIC_BUDGET_RATIO}" ]]; then
  COMMON_ARGS+=(--ctx-dynamic-budget-ratio "${CTX_DYNAMIC_BUDGET_RATIO}")
fi
COMMON_ARGS+=("${EXTRA_ARGS[@]}")

if [[ "${NUM_PROCS}" -gt 1 ]]; then
  "${TORCHRUN}" \
    --nproc_per_node="${NUM_PROCS}" \
    --master_port "${MASTER_PORT}" \
    -m dflash.benchmark \
    "${COMMON_ARGS[@]}"
else
  "${VENV_PYTHON}" -m dflash.benchmark "${COMMON_ARGS[@]}"
fi
