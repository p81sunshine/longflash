#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/run_benchmark_vllm_manual.sh}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

TERMINAL_BUCKET_DIR="${TERMINAL_BUCKET_DIR:-${ROOT_DIR}/benchmarks/terminal}"
SWEBENCH_BUCKET_DIR="${SWEBENCH_BUCKET_DIR:-${ROOT_DIR}/benchmarks/swebench}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/reproduced/vllm_by_variant_static_yarn_suffix}"

MODELS="${MODELS:-qwen3-8b}"
DATASET_GROUPS="${DATASET_GROUPS:-terminal swebench}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 2 4 8 16 32}"
VARIANT_ASSIGNMENTS="${VARIANT_ASSIGNMENTS:-target-only:0,1:40000 original:2,3:41000 dflash-static-yarn:4,5:42000 dflash-static-yarn-suffix:6,7:43000}"
CONCURRENCY_SCHEDULER="${CONCURRENCY_SCHEDULER:-${concurrency_scheduler:-sliding}}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MODEL_LEN_PAD="${MODEL_LEN_PAD:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_BATCHED_TOKENS_CAP="${MAX_NUM_BATCHED_TOKENS_CAP:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
REUSE_SERVER_FOR_CONCURRENCY_VALUES="${REUSE_SERVER_FOR_CONCURRENCY_VALUES:-1}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
MAX_INITIAL_GPU_USED_MB="${MAX_INITIAL_GPU_USED_MB:-1024}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
START_SCRIPT="${START_SCRIPT:-${ROOT_DIR}/scripts/serve/start_vllm_dflash_benchmark.sh}"
TARGET_START_SCRIPT="${TARGET_START_SCRIPT:-${ROOT_DIR}/scripts/serve/start_vllm_qwen35_target_benchmark.sh}"

SUFFIX_MIN_QUERY_LEN="${SUFFIX_MIN_QUERY_LEN:-8}"
SUFFIX_THRESHOLD="${SUFFIX_THRESHOLD:-8}"
SUFFIX_MAX_QUERY_LEN="${SUFFIX_MAX_QUERY_LEN:-16}"
SUFFIX_MAX_PREDICT_LEN="${SUFFIX_MAX_PREDICT_LEN:-15}"
SUFFIX_ALPHA="${SUFFIX_ALPHA:-2}"
SUFFIX_MAX_SPEC_OFFSET="${SUFFIX_MAX_SPEC_OFFSET:-0}"
SUFFIX_MIN_TOKEN_PROB="${SUFFIX_MIN_TOKEN_PROB:-0}"
SUFFIX_MAX_MATCHES="${SUFFIX_MAX_MATCHES:-0}"
SUFFIX_VERIFIER="${SUFFIX_VERIFIER:-linear}"

mkdir -p "${OUTPUT_DIR}"

case "${CONCURRENCY_SCHEDULER}" in
  batch|sliding)
    ;;
  *)
    echo "CONCURRENCY_SCHEDULER must be 'batch' or 'sliding', got: ${CONCURRENCY_SCHEDULER}" >&2
    exit 1
    ;;
esac

device_count() {
  local devices="$1"
  awk -F',' '{print NF}' <<< "${devices}"
}

devices_are_free() {
  local devices="$1"
  local gpu used
  IFS=',' read -r -a gpu_ids <<< "${devices}"
  for gpu in "${gpu_ids[@]}"; do
    used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')"
    if (( used > MAX_INITIAL_GPU_USED_MB )); then
      return 1
    fi
  done
}

wait_for_devices() {
  local devices="$1"
  if [[ "${WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return
  fi
  while ! devices_are_free "${devices}"; do
    echo "waiting_for_free_gpus=${devices} threshold_mb=${MAX_INITIAL_GPU_USED_MB}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep "${GPU_WAIT_SECONDS}"
  done
}

bucket_list() {
  local bucket_dir="$1"
  find "${bucket_dir}" -maxdepth 1 -name 'bucket_*.jsonl' -printf '%f\n' \
    | sed 's/^bucket_//; s/\.jsonl$//' \
    | sort -V \
    | tr '\n' ' '
}

bucket_end() {
  local bucket="$1"
  echo "${bucket##*_}"
}

ceil_factor() {
  local numerator="$1"
  local denominator="$2"
  "${PYTHON}" - "$numerator" "$denominator" <<'PY'
import math
import sys

num = float(sys.argv[1])
den = float(sys.argv[2])
print(max(1, math.ceil(num / den)))
PY
}

variant_label() {
  local variant="$1"
  case "${variant}" in
    target-only|target_only)
      echo "target_only"
      ;;
    original)
      echo "original"
      ;;
    eagle3-linear|eagle3_linear)
      echo "eagle3_linear"
      ;;
    dflash-static-yarn|yarn)
      echo "dflash_static_yarn"
      ;;
    dflash-static-yarn-suffix|yarn-suffix|suffix)
      echo "dflash_static_yarn_suffix"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

experiment_name() {
  local variant="$1"
  case "${variant}" in
    target-only|target_only)
      echo "target-only"
      ;;
    original)
      echo "original"
      ;;
    eagle3-linear|eagle3_linear)
      echo "eagle3-linear"
      ;;
    dflash-static-yarn|yarn)
      echo "dflash-static-yarn"
      ;;
    dflash-static-yarn-suffix|yarn-suffix|suffix)
      echo "dflash-static-yarn-suffix"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

max_model_len_for_samples() {
  local samples_path="$1"
  local prompt_lower_bound="$2"
  "${PYTHON}" - "${MODEL_PATH}" "${samples_path}" "${prompt_lower_bound}" "${MAX_TOKENS}" "${MODEL_LEN_PAD}" "${DISABLE_THINKING}" <<'PY'
import copy
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

model_path = sys.argv[1]
samples_path = Path(sys.argv[2])
prompt_lower_bound = int(sys.argv[3])
max_tokens = int(sys.argv[4])
pad = int(sys.argv[5])
disable_thinking = sys.argv[6].lower() in {"1", "true", "yes", "on"}

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def flatten_ids(ids):
    if not isinstance(ids, list) and hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids

def int_field(row, *names):
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None

def normalize_tool_call_arguments(messages):
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                function["arguments"] = parsed
    return normalized

def prompt_len(row):
    messages = normalize_tool_call_arguments(row.get("messages") or [])
    tools = row.get("tools") if isinstance(row.get("tools"), list) and row.get("tools") else None
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    if tools:
        kwargs["tools"] = tools
    try:
        return len(flatten_ids(tokenizer.apply_chat_template(messages, **kwargs)))
    except TypeError:
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("enable_thinking", None)
        try:
            return len(flatten_ids(tokenizer.apply_chat_template(messages, **retry_kwargs)))
        except Exception:
            pass
    except Exception:
        pass
    precomputed = int_field(row, "prompt_token_count", "prompt_tokens", "input_tokens")
    if precomputed is not None:
        return precomputed
    text = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    return len(tokenizer.encode(text, add_special_tokens=False))

max_prompt = prompt_lower_bound
for line in samples_path.read_text("utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    max_prompt = max(max_prompt, prompt_len(row))

print(max_prompt + max_tokens + pad)
PY
}

checked_max_model_len_for_samples() {
  local samples_path="$1"
  local prompt_lower_bound="$2"
  local value
  if [[ "${DRY_RUN}" == "1" ]]; then
    value="$((prompt_lower_bound + MAX_TOKENS + MODEL_LEN_PAD))"
    if (( value <= 0 )); then
      echo "invalid dry-run max_model_len='${value}' for samples=${samples_path}" >&2
      exit 1
    fi
    echo "${value}"
    return
  fi
  if ! value="$(max_model_len_for_samples "${samples_path}" "${prompt_lower_bound}")"; then
    echo "failed to compute max_model_len for samples=${samples_path}" >&2
    exit 1
  fi
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -le 0 ]]; then
    echo "invalid max_model_len='${value}' for samples=${samples_path}" >&2
    exit 1
  fi
  echo "${value}"
}

model_config() {
  local model_key="$1"
  DEFAULT_EAGLE3_DRAFT_MODEL_PATH=
  case "${model_key}" in
    qwen3-8b)
      MODEL_PATH="Qwen/Qwen3-8B"
      DRAFT_MODEL_PATH="z-lab/Qwen3-8B-DFlash-b16"
      DEFAULT_EAGLE3_DRAFT_MODEL_PATH="${EAGLE3_QWEN3_8B_MODEL:-<set EAGLE3_QWEN3_8B_MODEL>}"
      MODEL_LABEL="qwen3_8b"
      DRAFT_ORIGINAL=3072
      TARGET_ORIGINAL=32768
      TARGET_YARN_FACTOR_VALUE=4
      ;;
    qwen3-4b)
      MODEL_PATH="Qwen/Qwen3-4B"
      DRAFT_MODEL_PATH="z-lab/Qwen3-4B-DFlash-b16"
      DEFAULT_EAGLE3_DRAFT_MODEL_PATH="${EAGLE3_QWEN3_4B_MODEL:-<set EAGLE3_QWEN3_4B_MODEL>}"
      MODEL_LABEL="qwen3_4b"
      DRAFT_ORIGINAL=3072
      TARGET_ORIGINAL=32768
      TARGET_YARN_FACTOR_VALUE=4
      ;;
    qwen35-27b)
      MODEL_PATH="Qwen/Qwen3.5-27B"
      DRAFT_MODEL_PATH="z-lab/Qwen3.5-27B-DFlash"
      MODEL_LABEL="qwen35_27b"
      DRAFT_ORIGINAL=4096
      TARGET_ORIGINAL=262144
      TARGET_YARN_FACTOR_VALUE=
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac
}

target_yarn_env() {
  local max_model_len="$1"
  TARGET_YARN_ENV=()
  if [[ -n "${TARGET_YARN_FACTOR_VALUE}" && "${max_model_len}" -gt "${TARGET_ORIGINAL}" ]]; then
    TARGET_YARN_ENV=(
      "TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${TARGET_ORIGINAL}"
      "TARGET_YARN_FACTOR=${TARGET_YARN_FACTOR_VALUE}"
      "TARGET_YARN_MAX_POSITION_EMBEDDINGS=$((TARGET_ORIGINAL * TARGET_YARN_FACTOR_VALUE))"
    )
  fi
}

original_max_position_env() {
  local max_model_len="$1"
  ORIGINAL_MAX_POSITION_ENV=()
  ORIGINAL_MAX_POSITION_EXPECTED=
  case "${MODEL_LABEL}" in
    qwen3_4b|qwen3_8b)
      if [[ -n "${TARGET_YARN_FACTOR_VALUE}" && "${max_model_len}" -gt "${TARGET_ORIGINAL}" ]]; then
        ORIGINAL_MAX_POSITION_EXPECTED="$((TARGET_ORIGINAL * TARGET_YARN_FACTOR_VALUE))"
        ORIGINAL_MAX_POSITION_ENV=(
          "ORIGINAL_MAX_POSITION_EMBEDDING=${ORIGINAL_MAX_POSITION_EXPECTED}"
        )
      fi
      ;;
  esac
}

run_finished() {
  local run_root="$1"
  local expected_experiment="$2"
  local expected_original_max_position="${3:-}"
  "${PYTHON}" - "${run_root}" "${expected_experiment}" "${expected_original_max_position}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
expected_experiment = sys.argv[2]
expected_original_max_position = sys.argv[3]

def same_number(actual, expected_value):
    if expected_value == "":
        return True
    if actual is None:
        return False
    try:
        return float(actual) == float(expected_value)
    except (TypeError, ValueError):
        return str(actual) == expected_value

def original_max_position_matches(summary_path):
    if expected_experiment != "original":
        return True
    if expected_original_max_position == "":
        return True
    config_path = summary_path.parent / "run_config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for experiment in config.get("experiments", []):
        if experiment.get("name") != "original":
            continue
        has_old_yarn = any(
            experiment.get(name) not in (None, "")
            for name in (
                "draft_yarn_original",
                "draft_yarn_factor",
                "draft_yarn_max_position_embeddings",
            )
        )
        return (
            not has_old_yarn
            and same_number(
                experiment.get("original_max_position_embedding"),
                expected_original_max_position,
            )
        )
    return False

for summary_path in sorted(run_root.glob("*/summary.csv"), key=lambda path: path.stat().st_mtime, reverse=True):
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            names = {row.get("experiment", "") for row in csv.DictReader(handle)}
    except OSError:
        continue
    if expected_experiment in names and original_max_position_matches(summary_path):
        print(summary_path)
        break
PY
}

stop_vllm_server() {
  local port="$1"
  "${PYTHON}" - "${ROOT_DIR}" "${MODEL_PATH}" "${port}" <<'PY'
import sys

root, model, port_text = sys.argv[1:]
sys.path.insert(0, f"{root}/scripts/benchmark")
import server_utils as base

base.stop_existing_server(model, port=int(port_text))
PY
}

wait_vllm_server() {
  local port="$1"
  "${PYTHON}" - "${ROOT_DIR}" "${port}" <<'PY'
import sys

root, port_text = sys.argv[1:]
sys.path.insert(0, f"{root}/scripts/benchmark")
import server_utils as base

base.wait_for_server(f"http://127.0.0.1:{int(port_text)}", 900.0, 2.0)
PY
}

port_in_use() {
  local port="$1"
  "${PYTHON}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

select_port() {
  local base_port="$1"
  local run_index="$2"
  local offset=0
  local port
  while true; do
    port="$((base_port + run_index + offset))"
    if ! port_in_use "${port}"; then
      echo "${port}"
      return
    fi
    offset=$((offset + 1))
    if (( offset > 2000 )); then
      echo "could not find a free port near ${base_port} for run_index=${run_index}" >&2
      return 1
    fi
  done
}

start_variant_server() {
  local variant="$1"
  local devices="$2"
  local port="$3"
  local config_root="$4"
  local max_model_len="$5"
  local max_batched_tokens="$6"
  local draft_yarn_factor="$7"
  local label="$8"
  local expected_experiment="$9"

  local effective_tp="${TP_SIZE:-$(device_count "${devices}")}"
  local start_script="${START_SCRIPT}"
  local server_env=(
    "MODEL_PATH=${MODEL_PATH}"
    "CUDA_VISIBLE_DEVICES=${devices}"
    "TP_SIZE=${effective_tp}"
    "PORT=${port}"
    "MAX_MODEL_LEN=${max_model_len}"
    "MAX_BATCHED_TOKENS=${max_batched_tokens}"
    "MAX_NUM_SEQS="
    "VLLM_CACHE_ROOT=${config_root}/vllm_cache"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    "TOOL_CALL_PARSER=qwen3_coder"
    "REASONING_PARSER=qwen3"
    "ENABLE_CHUNKED_PREFILL=1"
    "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
    "ENFORCE_EAGER=${ENFORCE_EAGER}"
    "${TARGET_YARN_ENV[@]}"
  )

  case "${variant}" in
    target-only|target_only)
      start_script="${TARGET_START_SCRIPT}"
      ;;
    original)
      server_env+=(
        "DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH}"
        "DFLASH_WINDOW_SIZE=full"
      )
      if [[ -n "${ORIGINAL_MAX_POSITION_EXPECTED}" ]]; then
        server_env+=(
          "ORIGINAL_MAX_POSITION_EMBEDDING=${ORIGINAL_MAX_POSITION_EXPECTED}"
        )
      fi
      ;;
    eagle3-linear|eagle3_linear)
      server_env+=(
        "DRAFT_MODEL_PATH=${EAGLE3_DRAFT_MODEL_PATH:-${DEFAULT_EAGLE3_DRAFT_MODEL_PATH:-${DRAFT_MODEL_PATH}}}"
        "SPECULATIVE_METHOD=eagle3"
      )
      ;;
    dflash-static-yarn|yarn)
      server_env+=(
        "DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH}"
        "DFLASH_WINDOW_SIZE=full"
        "DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_ORIGINAL}"
        "DRAFT_STATIC_YARN_FACTOR=${draft_yarn_factor}"
      )
      ;;
    dflash-static-yarn-suffix|yarn-suffix|suffix)
      server_env+=(
        "DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH}"
        "DFLASH_WINDOW_SIZE=full"
        "DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${DRAFT_ORIGINAL}"
        "DRAFT_STATIC_YARN_FACTOR=${draft_yarn_factor}"
        "DFLASH_SUFFIX_DECODING=1"
        "DFLASH_SUFFIX_MAX_QUERY_LEN=${SUFFIX_MAX_QUERY_LEN}"
        "DFLASH_SUFFIX_MIN_QUERY_LEN=${SUFFIX_MIN_QUERY_LEN}"
        "DFLASH_SUFFIX_MAX_PREDICT_LEN=${SUFFIX_MAX_PREDICT_LEN}"
        "DFLASH_SUFFIX_ALPHA=${SUFFIX_ALPHA}"
        "DFLASH_SUFFIX_MAX_SPEC_OFFSET=${SUFFIX_MAX_SPEC_OFFSET}"
        "DFLASH_SUFFIX_MIN_TOKEN_PROB=${SUFFIX_MIN_TOKEN_PROB}"
        "DFLASH_SUFFIX_THRESHOLD=${SUFFIX_THRESHOLD}"
        "DFLASH_SUFFIX_MAX_MATCHES=${SUFFIX_MAX_MATCHES}"
        "DFLASH_SUFFIX_VERIFIER=${SUFFIX_VERIFIER}"
      )
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  echo "starting_server ${MODEL_LABEL}/${label} experiment=${expected_experiment} port=${port}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'env'
    printf ' %q' "${server_env[@]}"
    printf ' bash %q\n' "${start_script}"
    return
  fi

  stop_vllm_server "${port}"
  mkdir -p "${config_root}"
  env "${server_env[@]}" bash "${start_script}" > "${config_root}/server.log" 2>&1 &
  local server_pid="$!"
  echo "${server_pid}" > "${config_root}/server.pid"
  if ! wait_vllm_server "${port}"; then
    echo "server_start_failed port=${port} log=${config_root}/server.log" >&2
    tail -n 80 "${config_root}/server.log" >&2 || true
    stop_vllm_server "${port}"
    return 1
  fi
}

run_one_variant_concurrency() {
  local variant="$1"
  local devices="$2"
  local port="$3"
  local dataset_label="$4"
  local samples_path="$5"
  local length_label="$6"
  local max_model_len="$7"
  local concurrency="$8"
  local run_root="$9"
  local no_manage_server="${10}"
  local label="${11}"
  local expected_experiment="${12}"
  local draft_yarn_factor="${13}"
  local max_batched_tokens="${14}"

  local effective_tp="${TP_SIZE:-$(device_count "${devices}")}"
  local effective_draft_model_path="${DRAFT_MODEL_PATH}"
  case "${variant}" in
    eagle3-linear|eagle3_linear)
      effective_draft_model_path="${EAGLE3_DRAFT_MODEL_PATH:-${DEFAULT_EAGLE3_DRAFT_MODEL_PATH:-${DRAFT_MODEL_PATH}}}"
      ;;
  esac
  if [[ "${SKIP_COMPLETED}" == "1" && -n "$(run_finished "${run_root}" "${expected_experiment}" "${ORIGINAL_MAX_POSITION_EXPECTED}")" ]]; then
    echo "skip_completed=1: ${run_root}"
    return 0
  fi

  mkdir -p "${run_root}"

  local env_args=(
    "MODEL_PATH=${MODEL_PATH}"
    "DRAFT_MODEL_PATH=${effective_draft_model_path}"
    "SAMPLES=${samples_path}"
    "OUTPUT_DIR=${run_root}"
    "BASE_URL=http://127.0.0.1:${port}/v1"
    "CUDA_VISIBLE_DEVICES=${devices}"
    "TP_SIZE=${effective_tp}"
    "CONCURRENCY=${concurrency}"
    "CONCURRENCY_SCHEDULER=${CONCURRENCY_SCHEDULER}"
    "MAX_NUM_SEQS="
    "NO_MANAGE_SERVER=${no_manage_server}"
    "MAX_SAMPLES=${MAX_SAMPLES}"
    "MAX_TOKENS=${MAX_TOKENS}"
    "NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-15}"
    "TEMPERATURE=${TEMPERATURE}"
    "MAX_MODEL_LEN=${max_model_len}"
    "MAX_NUM_BATCHED_TOKENS=${max_batched_tokens}"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    "ENFORCE_EAGER=${ENFORCE_EAGER}"
    "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
    "DRAFT_YARN_ORIGINAL=${DRAFT_ORIGINAL}"
    "DRAFT_YARN_FACTOR=${draft_yarn_factor}"
    "DISABLE_THINKING=${DISABLE_THINKING}"
    "TOOL_CALL_PARSER=qwen3_coder"
    "REASONING_PARSER=qwen3"
    "START_SCRIPT=${START_SCRIPT}"
    "TARGET_START_SCRIPT=${TARGET_START_SCRIPT}"
    "SUFFIX_MAX_QUERY_LEN=${SUFFIX_MAX_QUERY_LEN}"
    "SUFFIX_MIN_QUERY_LEN=${SUFFIX_MIN_QUERY_LEN}"
    "SUFFIX_MAX_PREDICT_LEN=${SUFFIX_MAX_PREDICT_LEN}"
    "SUFFIX_ALPHA=${SUFFIX_ALPHA}"
    "SUFFIX_MAX_SPEC_OFFSET=${SUFFIX_MAX_SPEC_OFFSET}"
    "SUFFIX_MIN_TOKEN_PROB=${SUFFIX_MIN_TOKEN_PROB}"
    "SUFFIX_THRESHOLD=${SUFFIX_THRESHOLD}"
    "SUFFIX_MAX_MATCHES=${SUFFIX_MAX_MATCHES}"
    "SUFFIX_VERIFIER=${SUFFIX_VERIFIER}"
    "EXPERIMENT_VARIANTS=${variant}"
    "${TARGET_YARN_ENV[@]}"
    "${ORIGINAL_MAX_POSITION_ENV[@]}"
  )

  echo "===== ${MODEL_LABEL} ${label} ${dataset_label}/${length_label} bs=${concurrency} ====="
  echo "samples=${samples_path}"
  echo "max_model_len=${max_model_len}"
  echo "draft_yarn_original=${DRAFT_ORIGINAL}"
  echo "draft_yarn_factor=${draft_yarn_factor}"
  echo "target_yarn=${TARGET_YARN_ENV[*]:-none}"
  echo "original_max_position=${ORIGINAL_MAX_POSITION_ENV[*]:-none}"
  echo "cuda_visible_devices=${devices}"
  echo "tp_size=${effective_tp}"
  echo "port=${port}"
  echo "no_manage_server=${no_manage_server}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' bash %q\n\n' "${BASE_SCRIPT}"
    return
  fi

  set +e
  env "${env_args[@]}" bash "${BASE_SCRIPT}" 2>&1 | tee "${run_root}/launcher.log"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "runner_exit_status=${status}"
  fi
  return "${status}"
}

run_one_variant_samples_file() {
  local variant="$1"
  local devices="$2"
  local base_port="$3"
  local dataset_label="$4"
  local samples_path="$5"
  local length_label="$6"
  local max_model_len="$7"
  local run_index="$8"

  local label
  label="$(variant_label "${variant}")"
  local expected_experiment
  expected_experiment="$(experiment_name "${variant}")"
  local draft_yarn_factor
  draft_yarn_factor="$(ceil_factor "${max_model_len}" "${DRAFT_ORIGINAL}")"
  local max_batched_tokens="${max_model_len}"
  if (( max_batched_tokens > MAX_NUM_BATCHED_TOKENS_CAP )); then
    max_batched_tokens="${MAX_NUM_BATCHED_TOKENS_CAP}"
  fi
  target_yarn_env "${max_model_len}"
  original_max_position_env "${max_model_len}"

  local port
  port="$(select_port "${base_port}" "${run_index}")"
  local config_root="${OUTPUT_DIR}/${MODEL_LABEL}/${label}/${dataset_label}/${length_label}"
  local pending_concurrency=()
  local concurrency run_root
  for concurrency in ${CONCURRENCY_VALUES}; do
    run_root="${config_root}/bs${concurrency}"
    if [[ "${SKIP_COMPLETED}" == "1" && -n "$(run_finished "${run_root}" "${expected_experiment}" "${ORIGINAL_MAX_POSITION_EXPECTED}")" ]]; then
      echo "skip_completed=1: ${run_root}"
    else
      pending_concurrency+=("${concurrency}")
    fi
  done

  if (( ${#pending_concurrency[@]} == 0 )); then
    return 0
  fi

  wait_for_devices "${devices}"

  if [[ "${REUSE_SERVER_FOR_CONCURRENCY_VALUES}" != "1" ]]; then
    local status=0
    for concurrency in "${pending_concurrency[@]}"; do
      run_root="${config_root}/bs${concurrency}"
      run_one_variant_concurrency \
        "${variant}" "${devices}" "${port}" \
        "${dataset_label}" "${samples_path}" "${length_label}" "${max_model_len}" \
        "${concurrency}" "${run_root}" "0" "${label}" "${expected_experiment}" \
        "${draft_yarn_factor}" "${max_batched_tokens}" || {
          status=1
          if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
            break
          fi
        }
    done
    if [[ "${status}" -ne 0 && "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${status}"
    fi
    if [[ "${CONTINUE_ON_FAILURE}" == "1" ]]; then
      return 0
    fi
    return "${status}"
  fi

  echo "===== ${MODEL_LABEL} ${label} ${dataset_label}/${length_label} concurrency_sweep ====="
  echo "samples=${samples_path}"
  echo "max_model_len=${max_model_len}"
  echo "draft_yarn_original=${DRAFT_ORIGINAL}"
  echo "draft_yarn_factor=${draft_yarn_factor}"
  echo "target_yarn=${TARGET_YARN_ENV[*]:-none}"
  echo "original_max_position=${ORIGINAL_MAX_POSITION_ENV[*]:-none}"
  echo "cuda_visible_devices=${devices}"
  echo "tp_size=${TP_SIZE:-$(device_count "${devices}")}"
  echo "port=${port}"
  echo "concurrency_values=${pending_concurrency[*]}"
  echo "max_num_seqs=default"

  local status=0
  if ! start_variant_server \
    "${variant}" "${devices}" "${port}" "${config_root}" "${max_model_len}" \
    "${max_batched_tokens}" "${draft_yarn_factor}" "${label}" "${expected_experiment}"; then
    status=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    return "${status}"
  fi

  for concurrency in "${pending_concurrency[@]}"; do
    run_root="${config_root}/bs${concurrency}"
    run_one_variant_concurrency \
      "${variant}" "${devices}" "${port}" "${dataset_label}" "${samples_path}" \
      "${length_label}" "${max_model_len}" "${concurrency}" "${run_root}" "1" \
      "${label}" "${expected_experiment}" "${draft_yarn_factor}" \
      "${max_batched_tokens}" || {
        status=1
        if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
          break
        fi
      }
  done

  if [[ "${DRY_RUN}" != "1" ]]; then
    stop_vllm_server "${port}"
  fi
  if [[ "${status}" -ne 0 && "${CONTINUE_ON_FAILURE}" != "1" ]]; then
    exit "${status}"
  fi
  if [[ "${CONTINUE_ON_FAILURE}" == "1" ]]; then
    return 0
  fi
  return "${status}"
}

run_variant_worker() {
  local variant="$1"
  local devices="$2"
  local base_port="$3"
  local run_idx=0

  wait_for_devices "${devices}"

  for group in ${DATASET_GROUPS}; do
    case "${group}" in
      terminal)
        for bucket in $(bucket_list "${TERMINAL_BUCKET_DIR}"); do
          samples="${TERMINAL_BUCKET_DIR}/bucket_${bucket}.jsonl"
          if ! max_model_len="$(checked_max_model_len_for_samples "${samples}" "$(bucket_end "${bucket}")")"; then
            exit 1
          fi
          run_idx=$((run_idx + 1))
          run_one_variant_samples_file "${variant}" "${devices}" "${base_port}" "terminal" "${samples}" "bucket_${bucket}" "${max_model_len}" "${run_idx}"
        done
        ;;
      swebench)
        for bucket in $(bucket_list "${SWEBENCH_BUCKET_DIR}"); do
          samples="${SWEBENCH_BUCKET_DIR}/bucket_${bucket}.jsonl"
          if ! max_model_len="$(checked_max_model_len_for_samples "${samples}" "$(bucket_end "${bucket}")")"; then
            exit 1
          fi
          run_idx=$((run_idx + 1))
          run_one_variant_samples_file "${variant}" "${devices}" "${base_port}" "swebench" "${samples}" "bucket_${bucket}" "${max_model_len}" "${run_idx}"
        done
        ;;
      *)
        echo "Unknown dataset group: ${group}" >&2
        exit 1
        ;;
    esac
  done
}

aggregate_results() {
  "${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
headers = []
seen = set()
latest_by_run = {}
for summary_path in root.glob("*/*/*/*/bs*/*/summary.csv"):
    parts = summary_path.relative_to(root).parts
    key = parts[:5]
    current = latest_by_run.get(key)
    if current is None or summary_path.stat().st_mtime > current.stat().st_mtime:
        latest_by_run[key] = summary_path

for key in sorted(latest_by_run):
    summary_path = latest_by_run[key]
    run_dir = summary_path.parent
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text("utf-8")) if config_path.exists() else {}
    parts = summary_path.relative_to(root).parts
    model, variant, dataset, length_label, bs = parts[:5]
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            merged = {
                "model_label": model,
                "assigned_variant": variant,
                "dataset_label": dataset,
                "length_label": length_label,
                "batch_size": bs.removeprefix("bs"),
                "run_dir": str(run_dir),
                "model": config.get("model", ""),
                "draft_model": config.get("draft_model", ""),
                "max_model_len": config.get("max_model_len", ""),
                "max_tokens": config.get("max_tokens", ""),
                "temperature": config.get("temperature", ""),
                "concurrency": config.get("concurrency", ""),
                "concurrency_scheduler": config.get("concurrency_scheduler", ""),
                **row,
            }
            rows.append(merged)
            for key in merged:
                if key not in seen:
                    seen.add(key)
                    headers.append(key)

out = root / "aggregate_summary.csv"
if rows:
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
else:
    out.write_text("", "utf-8")
print(f"rows={len(rows)}")
print(f"csv={out}")
PY
}

run_model() {
  local model_key="$1"
  model_config "${model_key}"

  echo "===== model=${MODEL_LABEL} ====="
  echo "output_dir=${OUTPUT_DIR}"
  echo "dataset_groups=${DATASET_GROUPS}"
  echo "concurrency_values=${CONCURRENCY_VALUES}"
  echo "variant_assignments=${VARIANT_ASSIGNMENTS}"
  echo "concurrency_scheduler=${CONCURRENCY_SCHEDULER}"
  echo "reuse_server_for_concurrency_values=${REUSE_SERVER_FOR_CONCURRENCY_VALUES}"
  echo "max_samples=${MAX_SAMPLES}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "temperature=${TEMPERATURE}"
  echo "disable_thinking=${DISABLE_THINKING}"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo

  local pids=()
  local assignment variant devices base_port label log_path
  mkdir -p "${OUTPUT_DIR}/${MODEL_LABEL}"
  for assignment in ${VARIANT_ASSIGNMENTS}; do
    IFS=: read -r variant devices base_port <<< "${assignment}"
    label="$(variant_label "${variant}")"
    log_path="${OUTPUT_DIR}/${MODEL_LABEL}/worker_${label}.log"
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_variant_worker "${variant}" "${devices}" "${base_port}" 2>&1 | tee "${log_path}"
    else
      (
        run_variant_worker "${variant}" "${devices}" "${base_port}"
      ) > "${log_path}" 2>&1 &
      local pid="$!"
      pids+=("${pid}")
      echo "started ${MODEL_LABEL}/${label} pid=${pid} devices=${devices} log=${log_path}"
    fi
  done

  local status=0
  if [[ "${DRY_RUN}" != "1" ]]; then
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
  fi
  aggregate_results
  return "${status}"
}

overall_status=0
for model_key in ${MODELS}; do
  if ! run_model "${model_key}"; then
    overall_status=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
  fi
done

echo "Done. output_dir=${OUTPUT_DIR}"
echo "aggregate_csv=${OUTPUT_DIR}/aggregate_summary.csv"
exit "${overall_status}"
