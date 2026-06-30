#!/usr/bin/env python3
"""Agentic DFlash memory benchmark focused on acceptance and speed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import server_utils as base
except ModuleNotFoundError:  # pragma: no cover - fallback for direct module loading
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import server_utils as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_SAMPLES = str(ROOT / "data/current/qwen35_tau2_targettrace_samples.jsonl")
DEFAULT_OUTPUT_DIR = "results/reproduced/qwen35_tau2_benchmark"


@dataclass(frozen=True)
class Experiment:
    name: str
    window: str
    window_mode: str = "sink_recent_suffix"
    sink_tokens: int | str = 768
    recent_tokens: int | None = None
    suffix_match_tokens: int = 0
    suffix_keep_tokens: int = 0
    suffix_middle_budget: int = 0
    select_ranges: str = ""
    position_mode: str = "compact"
    target_only: bool = False
    dynamic_budget_ratio: float | None = None
    draft_yarn_original: int | None = None
    draft_yarn_factor: float | None = None
    draft_yarn_max_position_embeddings: int | None = None
    original_max_position_embedding: int | None = None
    dynamic_yarn_original: int | None = None
    dynamic_yarn_max_factor: float | None = None
    dynamic_yarn_mode: str | None = None
    dynamic_yarn_length_ratio: float | None = None
    target_yarn_original: int | None = None
    target_yarn_max_position_embeddings: int | None = None
    target_yarn_factor: float | None = None
    suffix_decoding: bool = False
    suffix_max_query_len: int | None = None
    suffix_min_query_len: int | None = None
    suffix_max_predict_len: int | None = None
    suffix_alpha: float | None = None
    suffix_max_spec_offset: float | None = None
    suffix_min_token_prob: float | None = None
    suffix_threshold: float | None = None
    suffix_max_matches: int | None = None
    suffix_verifier: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic DFlash acceptance/speed benchmark")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--start-script", default=str(ROOT / "scripts/serve/start_vllm_dflash_benchmark.sh"))
    parser.add_argument("--target-start-script", default=str(ROOT / "scripts/serve/start_vllm_qwen35_target_benchmark.sh"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiment",
        dest="experiment_configs",
        action="append",
        default=[],
        help=(
            "Dynamic experiment config. Repeatable. Accepts either a JSON object "
            "or comma/space separated key=value pairs, e.g. "
            "--experiment name=dflash-static-yarn-suffix,window=full,"
            "draft_yarn_original=3072,draft_yarn_factor=42,suffix_decoding=true"
        ),
    )
    parser.add_argument(
        "--experiment-config-file",
        type=Path,
        default=None,
        help=(
            "JSON/JSONL experiment config file. JSON may be a list, one object, "
            "or an object with an 'experiments' list."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help=(
            "Number of unrecorded requests to send after each managed server starts. "
            "Uses the first sample for normal per-experiment servers and the current "
            "sample for per-sample state servers."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of requests to issue concurrently per experiment. Values >1 "
            "exercise vLLM server-side batching."
        ),
    )
    parser.add_argument(
        "--concurrency-scheduler",
        choices=["batch", "sliding"],
        default="batch",
        help=(
            "batch sends one concurrency-sized group and waits for all requests before "
            "submitting the next group. sliding keeps up to --concurrency requests "
            "in flight and reports speculative metrics as one experiment-level delta."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--tool-call-parser", default=None)
    parser.add_argument("--reasoning-parser", default=None)
    parser.add_argument("--allow-long-max-model-len", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--target-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--target-yarn-max-position-embeddings", type=int, default=None)
    parser.add_argument("--target-yarn-factor", type=float, default=None)
    parser.add_argument("--dflash-recent-tokens", type=int, default=None)
    parser.add_argument("--dflash-suffix-match-tokens", type=int, default=0)
    parser.add_argument("--dflash-suffix-keep-tokens", type=int, default=0)
    parser.add_argument("--dflash-suffix-middle-budget", type=int, default=0)
    parser.add_argument("--dynamic-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--dynamic-yarn-max-factor", type=float, default=None)
    parser.add_argument("--dynamic-yarn-mode", choices=["continuous", "bucket"], default="continuous")
    parser.add_argument("--dynamic-yarn-length-ratio", type=float, default=None)
    parser.add_argument(
        "--vllm-cache-root",
        default=None,
        help=(
            "Cache root for managed vLLM servers. Defaults to a vllm_cache "
            "directory under the current benchmark output run."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Optional vLLM sampling min_tokens passed through extra_body.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Pass ignore_eos=true through extra_body to force generation to max_tokens.",
    )
    parser.add_argument(
        "--request-mode",
        choices=["chat", "completion"],
        default="chat",
        help=(
            "Use OpenAI chat completions, or render the chat template locally "
            "and send the exact prompt through /v1/completions."
        ),
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--num-spec-tokens", type=int, default=15)
    parser.add_argument("--request-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=900.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--expected-tool", default="issue_store_credit")
    parser.add_argument("--expected-customer-id", default="cus_71b9f3")
    parser.add_argument("--expected-amount-cents", type=int, default=2000)
    parser.add_argument("--json-response-format", action="store_true", default=False)
    parser.add_argument("--no-json-response-format", dest="json_response_format", action="store_false")
    parser.add_argument(
        "--debug-action-validation",
        action="store_true",
        help="Optionally validate final JSON against the trajectory tool call. Not a core benchmark metric.",
    )
    parser.add_argument("--request-id-prefix", default="dflash")
    parser.add_argument("--no-manage-server", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log experiment/sample failures and continue with later experiments. Partial records still get summarized.",
    )
    parser.add_argument(
        "--state-per-sample-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restart managed server per sample for state4096 so DFLASH_SELECT_RANGES is sample-specific.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text("utf-8").splitlines()):
        if line.strip():
            row = json.loads(line)
            if row.get("sample_id") is None:
                row["sample_id"] = row.get("instance_id") or f"sample_{idx}"
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No samples found in {path}")
    return rows


def messages_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    sample_id = sample.get("sample_id", "<unknown>")
    raise ValueError(f"Sample {sample_id} is missing non-empty `messages`; current benchmark does not accept prompt-only samples")


def tools_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = sample.get("tools")
    return tools if isinstance(tools, list) and tools else None


def tool_choice_for_sample(sample: dict[str, Any]) -> Any:
    tools = tools_for_sample(sample)
    if not tools:
        return None
    return sample.get("tool_choice") or "auto"


def render_sample_text(
    tokenizer: Any,
    sample: dict[str, Any],
    add_generation_prompt: bool,
    *,
    enable_thinking: bool | None = None,
) -> str:
    messages = messages_for_sample(sample)
    tools = tools_for_sample(sample)
    try:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if tools:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return json.dumps(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice_for_sample(sample),
            },
            ensure_ascii=False,
        )


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    try:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = tools
        ids = tokenizer.apply_chat_template(
            messages,
            **kwargs,
        )
        if not isinstance(ids, list) and hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return len(tokenizer.encode(text, add_special_tokens=False))


def generation_prompt_tail_token_ids(
    tokenizer: Any,
    sample: dict[str, Any],
    *,
    enable_thinking: bool,
) -> list[int] | None:
    content_text = render_sample_text(
        tokenizer,
        sample,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    prompt_text = render_sample_text(
        tokenizer,
        sample,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    content_ids = tokenizer.encode(content_text)
    prompt_ids = tokenizer.encode(prompt_text)
    if prompt_ids[: len(content_ids)] != content_ids:
        return None
    return [int(token_id) for token_id in prompt_ids[len(content_ids) :]]


def unique_generation_prompt_tail_token_ids(
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    enable_thinking: bool,
) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    tails: list[list[int]] = []
    for sample in samples:
        tail = generation_prompt_tail_token_ids(
            tokenizer,
            sample,
            enable_thinking=enable_thinking,
        )
        if not tail:
            continue
        key = tuple(tail)
        if key in seen:
            continue
        seen.add(key)
        tails.append(tail)
    tails.sort(key=len, reverse=True)
    return tails


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((s, e) for s, e in ranges if e > s):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def char_span_to_token_range(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    token_ids = [idx for idx, (tok_start, tok_end) in enumerate(offsets) if tok_end > start and tok_start < end]
    if not token_ids:
        return None
    return min(token_ids), max(token_ids) + 1


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def reference_tool_call_for_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tool_call = None
    reference_visible = sample.get("reference_visible_message")
    if not tool_call and isinstance(reference_visible, dict):
        tool_calls = reference_visible.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tool_call = tool_calls[0]
    if isinstance(tool_call, dict) and tool_call.get("tool"):
        arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        return {"tool": tool_call["tool"], "arguments": arguments}
    return {
        "tool": args.expected_tool,
        "arguments": {
            "customer_id": args.expected_customer_id,
            "amount_cents": args.expected_amount_cents,
        },
    }


def state_needles_for_sample(sample: dict[str, Any], args: argparse.Namespace) -> list[str]:
    tool_call = reference_tool_call_for_sample(sample, args)
    reference_args = tool_call.get("arguments", {})
    needles = [str(tool_call.get("tool", ""))]
    needles.extend(str(value) for value in reference_args.values())
    for key in ("domain", "sub_domain", "turn_type"):
        if sample.get(key):
            needles.append(str(sample[key]))
    if isinstance(sample.get("state_needles"), list):
        needles.extend(str(value) for value in sample["state_needles"])
    return unique_strings(needles)


def compute_state_ranges(
    tokenizer: Any,
    rendered: str,
    state_needles: list[str] | None = None,
    max_mandatory_tokens: int = 1300,
) -> str:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    ranges: list[tuple[int, int]] = [(0, 768)]
    needles = unique_strings(state_needles or [])
    archive_start = rendered.find("ARCHIVED WORKSPACE EVENT")
    live_text_end = archive_start if archive_start >= 0 else len(rendered)

    for needle in needles:
        search_texts = [(0, live_text_end)]
        for area_start, area_end in search_texts:
            search_start = area_start
            found_for_needle = 0
            while search_start < area_end:
                idx = rendered.find(needle, search_start, area_end)
                if idx < 0:
                    break
                window_start = max(0, idx - 90)
                window_end = min(len(rendered), idx + len(needle) + 160)
                token_range = char_span_to_token_range(offsets, window_start, window_end)
                if token_range is not None:
                    start_tok, end_tok = token_range
                    if end_tok > 768:
                        ranges.append((max(start_tok, 768), end_tok))
                        found_for_needle += 1
                search_start = idx + len(needle)
                if found_for_needle >= 3:
                    break
        if len(ranges) >= 32:
            break

    capped: list[tuple[int, int]] = []
    total = 0
    for start, end in merge_ranges(ranges):
        length = end - start
        if total + length > max_mandatory_tokens:
            keep = max(0, max_mandatory_tokens - total)
            if keep > 0:
                capped.append((start, start + keep))
            break
        capped.append((start, end))
        total += length
    return ",".join(f"{start}:{end}" for start, end in capped)


EXPERIMENT_ALIASES = {
    "sink": "sink_tokens",
    "recent": "recent_tokens",
    "mode": "window_mode",
    "window_size": "window",
    "target": "target_only",
    "suffix_match": "suffix_match_tokens",
    "suffix_keep": "suffix_keep_tokens",
    "suffix_middle": "suffix_middle_budget",
    "dflash_recent_tokens": "recent_tokens",
    "dflash_suffix_match_tokens": "suffix_match_tokens",
    "dflash_suffix_keep_tokens": "suffix_keep_tokens",
    "dflash_suffix_middle_budget": "suffix_middle_budget",
    "dflash_dynamic_budget_ratio": "dynamic_budget_ratio",
    "draft_static_yarn_original_max_position_embeddings": "draft_yarn_original",
    "draft_static_yarn_factor": "draft_yarn_factor",
    "draft_static_yarn_max_position_embeddings": "draft_yarn_max_position_embeddings",
    "original_max_position_embedding": "original_max_position_embedding",
    "draft_dynamic_yarn_original_max_position_embeddings": "dynamic_yarn_original",
    "draft_dynamic_yarn_max_factor": "dynamic_yarn_max_factor",
    "draft_dynamic_yarn_mode": "dynamic_yarn_mode",
    "draft_dynamic_yarn_length_ratio": "dynamic_yarn_length_ratio",
    "target_yarn_original_max_position_embeddings": "target_yarn_original",
}


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"none", "null", ""}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _parse_key_value_config(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected key=value in experiment config part: {part!r}")
        key, value = part.split("=", 1)
        config[key.strip()] = _parse_scalar(value)
    return config


def _apply_compound_experiment_fields(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    suffix = normalized.pop("suffix", None)
    if suffix is not None:
        if isinstance(suffix, str):
            parts = [part for part in re.split(r"[:/]", suffix) if part != ""]
        elif isinstance(suffix, (list, tuple)):
            parts = list(suffix)
        else:
            raise ValueError("suffix must be 'match:keep[:budget]' or a 2/3-item list")
        if len(parts) not in {2, 3}:
            raise ValueError("suffix must be match:keep or match:keep:budget")
        normalized["suffix_match_tokens"] = _parse_scalar(parts[0])
        normalized["suffix_keep_tokens"] = _parse_scalar(parts[1])
        if len(parts) == 3:
            normalized["suffix_middle_budget"] = _parse_scalar(parts[2])
    return normalized


def _load_experiment_config_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty experiment config")
    if stripped.startswith("{"):
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError("--experiment JSON must be an object")
        return loaded
    return _parse_key_value_config(stripped)


def _load_experiment_config_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError(f"JSONL experiment row must be an object: {path}")
                rows.append(loaded)
        return rows
    loaded = json.loads(path.read_text("utf-8"))
    if isinstance(loaded, dict) and isinstance(loaded.get("experiments"), list):
        loaded = loaded["experiments"]
    if isinstance(loaded, dict):
        return [loaded]
    if isinstance(loaded, list) and all(isinstance(item, dict) for item in loaded):
        return loaded
    raise ValueError(f"Unsupported experiment config file shape: {path}")


def experiment_from_config(config: dict[str, Any]) -> Experiment:
    field_names = {field.name for field in fields(Experiment)}
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in _apply_compound_experiment_fields(config).items():
        key = raw_key.strip().replace("-", "_")
        key = EXPERIMENT_ALIASES.get(key, key)
        if key not in field_names:
            raise ValueError(f"Unknown experiment config key {raw_key!r}")
        normalized[key] = _parse_scalar(raw_value)
    if "name" not in normalized:
        normalized["name"] = str(normalized.get("window", "experiment"))
    if "window" not in normalized:
        raise ValueError(f"Experiment config is missing required key 'window': {config}")
    if "window_mode" not in normalized:
        normalized["window_mode"] = "sink_recent_suffix"
    for key in ("name", "window", "window_mode", "select_ranges", "position_mode"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = str(normalized[key])
    allowed_modes = {
        "full",
        "target_only",
        "attention",
        "range_recent_once",
        "sink_recent_suffix",
    }
    if normalized.get("window_mode") not in allowed_modes:
        raise ValueError(
            f"Unsupported window_mode {normalized.get('window_mode')!r}; "
            f"use one of {sorted(allowed_modes)}"
        )
    return Experiment(**normalized)


def load_experiments(args: argparse.Namespace) -> list[Experiment]:
    configs: list[dict[str, Any]] = []
    if args.experiment_config_file is not None:
        configs.extend(_load_experiment_config_file(args.experiment_config_file))
    configs.extend(_load_experiment_config_text(text) for text in args.experiment_configs)
    if not configs:
        raise ValueError("Provide at least one --experiment or --experiment-config-file")
    return [experiment_from_config(config) for config in configs]


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_tool_call(obj: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    tool = obj.get("tool", obj.get("name"))
    raw_arguments = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
    return tool, arguments


def values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, int) and isinstance(actual, str):
        try:
            return int(actual.strip()) == expected
        except ValueError:
            return False
    if isinstance(expected, str) and not isinstance(actual, str):
        return str(actual) == expected
    return False


def validate_action(text: str, sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    obj = extract_json_object(text)
    reference = reference_tool_call_for_sample(sample, args)
    reference_args = reference.get("arguments", {})
    result: dict[str, Any] = {
        "is_json": obj is not None,
        "reference_tool": reference.get("tool"),
        "reference_arguments": json.dumps(reference_args, sort_keys=True),
        "reference_tool_match": False,
        "reference_args_match": False,
        "matched_reference_args": 0,
        "reference_arg_count": len(reference_args),
        "customer_id_match": None,
        "order_id_match": None,
        "tracking_id_match": None,
        "case_id_match": None,
        "amount_match": None,
        "template_match": None,
        "policy_id_match": None,
        "queue_match": None,
        "valid_reference_tool_call": False,
    }
    if obj is None:
        return result
    tool, arguments = normalize_tool_call(obj)
    result["tool"] = tool
    result["arguments"] = arguments
    result["reference_tool_match"] = tool == reference.get("tool")
    matched = 0
    for key, reference_value in reference_args.items():
        key_match = values_equal(arguments.get(key), reference_value)
        matched += int(key_match)
        if key == "amount_cents":
            result["amount_match"] = key_match
        elif key in {
            "customer_id",
            "order_id",
            "tracking_id",
            "case_id",
            "template",
            "policy_id",
            "queue",
        }:
            result[f"{key}_match"] = key_match
    result["matched_reference_args"] = matched
    result["reference_args_match"] = matched == len(reference_args)
    result["valid_reference_tool_call"] = bool(
        result["reference_tool_match"] and result["reference_args_match"]
    )
    return result


def output_text_for_validation(result: dict[str, Any]) -> str:
    content = result.get("output_content") or ""
    if content:
        return str(content)
    tool_calls = result.get("output_tool_calls") or []
    if tool_calls:
        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
    return ""


def make_request_id(prefix: str, sample_id: Any, experiment: str, repeat_idx: int) -> str:
    raw = f"{prefix}-{sample_id}-{experiment}-r{repeat_idx}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    if len(safe) > 160:
        safe = safe[:160].rstrip("_")
    return f"{safe}-{digest}" if safe else f"{prefix}-{digest}"


def mean_optional(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def position_metric_fields(delta: dict[str, Any] | None) -> dict[str, Any]:
    if not delta:
        return {}
    fields: dict[str, Any] = {}
    accepted_per_pos = delta.get("accepted_tokens_per_pos") or []
    rate_per_pos = delta.get("acceptance_rate_per_pos") or []
    for idx, accepted in enumerate(accepted_per_pos, start=1):
        fields[f"accepted_tokens_pos_{idx}"] = accepted
    for idx, rate in enumerate(rate_per_pos, start=1):
        fields[f"acceptance_rate_pos_{idx}"] = rate
    buckets = {
        "acceptance_rate_pos_1_4": rate_per_pos[:4],
        "acceptance_rate_pos_5_8": rate_per_pos[4:8],
        "acceptance_rate_pos_9_12": rate_per_pos[8:12],
        "acceptance_rate_pos_13_15": rate_per_pos[12:15],
    }
    fields.update({name: mean_optional(values) for name, values in buckets.items()})
    return fields


def make_server_args(
    args: argparse.Namespace,
    experiment: Experiment,
    vllm_cache_root: Path | None = None,
    select_ranges: str | None = None,
) -> SimpleNamespace:
    port = base.server_port_from_base_url(args.base_url)
    recent_tokens = (
        experiment.recent_tokens
        if experiment.recent_tokens is not None
        else args.dflash_recent_tokens
    )
    suffix_match_tokens = (
        experiment.suffix_match_tokens
        if experiment.suffix_match_tokens
        else args.dflash_suffix_match_tokens
    )
    suffix_keep_tokens = (
        experiment.suffix_keep_tokens
        if experiment.suffix_keep_tokens
        else args.dflash_suffix_keep_tokens
    )
    suffix_middle_budget = (
        experiment.suffix_middle_budget
        if experiment.suffix_middle_budget
        else args.dflash_suffix_middle_budget
    )
    dynamic_yarn_original = (
        experiment.dynamic_yarn_original
        if experiment.dynamic_yarn_original is not None
        else args.dynamic_yarn_original_max_position_embeddings
    )
    dynamic_yarn_max_factor = (
        experiment.dynamic_yarn_max_factor
        if experiment.dynamic_yarn_max_factor is not None
        else args.dynamic_yarn_max_factor
    )
    dynamic_yarn_length_ratio = (
        experiment.dynamic_yarn_length_ratio
        if experiment.dynamic_yarn_length_ratio is not None
        else args.dynamic_yarn_length_ratio
    )
    dynamic_yarn_mode = experiment.dynamic_yarn_mode
    if dynamic_yarn_mode is None and (
        dynamic_yarn_original is not None
        or dynamic_yarn_max_factor is not None
        or dynamic_yarn_length_ratio is not None
    ):
        dynamic_yarn_mode = args.dynamic_yarn_mode
    return SimpleNamespace(
        model=args.model,
        draft_model=args.draft_model,
        start_script=args.target_start_script if experiment.target_only else args.start_script,
        num_spec_tokens=args.num_spec_tokens,
        request_max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        allow_long_max_model_len=args.allow_long_max_model_len,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        vllm_cache_root=str(vllm_cache_root) if vllm_cache_root is not None else None,
        port=port,
        window_mode=experiment.window_mode,
        sink_tokens=int(experiment.sink_tokens) if experiment.sink_tokens else 0,
        recent_tokens=recent_tokens,
        suffix_match_tokens=suffix_match_tokens,
        suffix_keep_tokens=suffix_keep_tokens,
        suffix_middle_budget=suffix_middle_budget,
        select_ranges=experiment.select_ranges if select_ranges is None else select_ranges,
        position_mode=experiment.position_mode,
        dynamic_budget_ratio=experiment.dynamic_budget_ratio,
        draft_yarn_original=experiment.draft_yarn_original,
        draft_yarn_factor=experiment.draft_yarn_factor,
        draft_yarn_max_position_embeddings=experiment.draft_yarn_max_position_embeddings,
        original_max_position_embedding=experiment.original_max_position_embedding,
        dynamic_yarn_original=dynamic_yarn_original,
        dynamic_yarn_max_factor=dynamic_yarn_max_factor,
        dynamic_yarn_mode=dynamic_yarn_mode,
        dynamic_yarn_length_ratio=dynamic_yarn_length_ratio,
        suffix_source_tail_token_ids=getattr(args, "suffix_source_tail_token_ids", None),
        target_yarn_original=(
            experiment.target_yarn_original
            if experiment.target_yarn_original is not None
            else args.target_yarn_original_max_position_embeddings
        ),
        target_yarn_max_position_embeddings=(
            experiment.target_yarn_max_position_embeddings
            if experiment.target_yarn_max_position_embeddings is not None
            else args.target_yarn_max_position_embeddings
        ),
        target_yarn_factor=(
            experiment.target_yarn_factor
            if experiment.target_yarn_factor is not None
            else args.target_yarn_factor
        ),
        suffix_decoding=experiment.suffix_decoding,
        suffix_max_query_len=experiment.suffix_max_query_len,
        suffix_min_query_len=experiment.suffix_min_query_len,
        suffix_max_predict_len=experiment.suffix_max_predict_len,
        suffix_alpha=experiment.suffix_alpha,
        suffix_max_spec_offset=experiment.suffix_max_spec_offset,
        suffix_min_token_prob=experiment.suffix_min_token_prob,
        suffix_threshold=experiment.suffix_threshold,
        suffix_max_matches=experiment.suffix_max_matches,
        suffix_verifier=experiment.suffix_verifier,
    )


def run_one_request(
    client: OpenAI,
    sample: dict[str, Any],
    args: argparse.Namespace,
    request_id: str | None = None,
) -> dict[str, Any]:
    def response_spec_delta(response: Any) -> dict[str, Any] | None:
        metadata = getattr(response, "metadata", None)
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
        if not isinstance(metadata, dict):
            return None

        accept_length = metadata.get("spec_accept_length")
        verify_ct = metadata.get("spec_verify_ct")
        proposed = metadata.get("spec_num_proposed_drafts")
        correct = metadata.get("spec_num_correct_drafts")
        accept_rate = metadata.get("spec_accept_rate")
        if accept_length is None and verify_ct is None and proposed is None:
            return None

        def as_float(value: Any) -> float | None:
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        def as_int(value: Any) -> int | None:
            try:
                return None if value is None else int(value)
            except (TypeError, ValueError):
                return None

        accept_length_f = as_float(accept_length)
        proposed_i = as_int(proposed)
        correct_i = as_int(correct)
        verify_ct_i = as_int(verify_ct)
        accept_rate_f = as_float(accept_rate)
        if accept_rate_f is None and proposed_i and correct_i is not None:
            accept_rate_f = correct_i / proposed_i

        return {
            "num_drafts": verify_ct_i,
            "draft_tokens": proposed_i,
            "accepted_tokens": correct_i,
            "acceptance_rate": accept_rate_f * 100.0
            if accept_rate_f is not None
            else None,
            "acceptance_length": accept_length_f,
            "accepted_tokens_per_pos": [],
            "acceptance_rate_per_pos": [],
        }

    start = time.perf_counter()
    if args.request_mode == "completion":
        tokenizer = getattr(args, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("completion request mode requires args._tokenizer")
        prompt = render_sample_text(
            tokenizer,
            sample,
            add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
        )
        request_kwargs = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "timeout": args.request_timeout_sec,
            "extra_body": {},
        }
        if args.min_tokens is not None:
            request_kwargs["extra_body"]["min_tokens"] = args.min_tokens
        if args.ignore_eos:
            request_kwargs["extra_body"]["ignore_eos"] = True
        if args.top_k != 1:
            request_kwargs["extra_body"]["top_k"] = args.top_k
        if not request_kwargs["extra_body"]:
            del request_kwargs["extra_body"]
        if request_id:
            request_kwargs["extra_headers"] = {"X-Request-Id": request_id}
        response = client.completions.create(**request_kwargs)
        elapsed = time.perf_counter() - start
        usage = response.usage.model_dump() if response.usage else {}
        choice = response.choices[0]
        content = choice.text or ""
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return {
            "elapsed_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
            "completion_tps": completion_tokens / elapsed if elapsed > 0 else None,
            "request_id": request_id,
            "finish_reason": choice.finish_reason,
            "output_content": content,
            "output_tool_calls": [],
            "_spec_delta": response_spec_delta(response),
        }

    messages = messages_for_sample(sample)
    tools = tools_for_sample(sample)
    tool_choice = tool_choice_for_sample(sample)
    request_kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.request_timeout_sec,
        "extra_body": {},
    }
    if args.min_tokens is not None:
        request_kwargs["extra_body"]["min_tokens"] = args.min_tokens
    if args.ignore_eos:
        request_kwargs["extra_body"]["ignore_eos"] = True
    if args.top_k != 1:
        request_kwargs["extra_body"]["top_k"] = args.top_k
    request_kwargs["extra_body"]["chat_template_kwargs"] = {
        "enable_thinking": not args.disable_thinking
    }
    if not request_kwargs["extra_body"]:
        del request_kwargs["extra_body"]
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = tool_choice
    if args.json_response_format and not tools:
        request_kwargs["response_format"] = {"type": "json_object"}
    if request_id:
        request_kwargs["extra_headers"] = {"X-Request-Id": request_id}
    response = client.chat.completions.create(**request_kwargs)
    elapsed = time.perf_counter() - start
    usage = response.usage.model_dump() if response.usage else {}
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    tool_calls = [
        tool_call.model_dump(mode="json") if hasattr(tool_call, "model_dump") else tool_call
        for tool_call in (message.tool_calls or [])
    ]
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
        "completion_tps": completion_tokens / elapsed if elapsed > 0 else None,
        "request_id": request_id,
        "finish_reason": choice.finish_reason,
        "output_content": content,
        "output_tool_calls": tool_calls,
        "_spec_delta": response_spec_delta(response),
    }


def make_record_row(
    experiment: Experiment,
    sample: dict[str, Any],
    args: argparse.Namespace,
    repeat_idx: int,
    result: dict[str, Any],
    delta: dict[str, Any] | None,
    *,
    metric_scope: str,
    concurrency_group: int | None = None,
    concurrency_group_size: int = 1,
) -> dict[str, Any]:
    action = (
        validate_action(output_text_for_validation(result), sample, args)
        if args.debug_action_validation
        else {}
    )
    public_result = {k: v for k, v in result.items() if not k.startswith("_")}
    position_fields = position_metric_fields(delta)
    return {
        "experiment": experiment.name,
        "window": experiment.window,
        "window_mode": "target_only"
        if experiment.target_only
        else (experiment.window_mode if experiment.window != "full" else "full"),
        "sink_tokens": int(experiment.sink_tokens) if experiment.sink_tokens else 0,
        "suffix_decoding": experiment.suffix_decoding,
        "suffix_threshold": experiment.suffix_threshold,
        "suffix_alpha": experiment.suffix_alpha,
        "suffix_max_predict_len": experiment.suffix_max_predict_len,
        "suffix_verifier": experiment.suffix_verifier,
        "repeat_idx": repeat_idx,
        "sample_id": sample.get("sample_id"),
        "bucket": sample.get("bucket"),
        "prompt_tokens_est": sample.get("prompt_tokens_est"),
        "turn_type": sample.get("turn_type"),
        "prompt_length_bin": sample.get("prompt_length_bin"),
        "metric_scope": metric_scope,
        "concurrency": args.concurrency,
        "concurrency_scheduler": args.concurrency_scheduler,
        "concurrency_group": concurrency_group,
        "concurrency_group_size": concurrency_group_size,
        "acceptance_rate": delta["acceptance_rate"] if delta else None,
        "acceptance_length": delta["acceptance_length"] if delta else None,
        **public_result,
        "num_drafts": delta["num_drafts"] if delta else None,
        "draft_tokens": delta["draft_tokens"] if delta else None,
        "accepted_tokens": delta["accepted_tokens"] if delta else None,
        **position_fields,
        **action,
    }


def run_sample_repeats(
    handle: Any,
    client: OpenAI,
    server_root: str,
    experiment: Experiment,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    for repeat_idx in range(args.repeats):
        request_id = make_request_id(
            args.request_id_prefix,
            sample.get("sample_id", "sample"),
            experiment.name,
            repeat_idx,
        )
        before = base.fetch_spec_decode_metrics(server_root)
        try:
            result = run_one_request(client, sample, args, request_id=request_id)
        except Exception as e:
            error_msg = str(e)
            if "maximum context length" in error_msg or "400" in error_msg:
                print(
                    f"[SKIP] {experiment.name} sample={sample.get('sample_id')} "
                    f"repeat={repeat_idx}: {error_msg[:120]}"
                )
                break
            raise
        after = base.fetch_spec_decode_metrics(server_root)
        delta = result.get("_spec_delta") or base.compute_delta(before, after)
        row = make_record_row(
            experiment,
            sample,
            args,
            repeat_idx,
            result,
            delta,
            metric_scope="request",
        )
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()
        print(
            f"[{experiment.name}] sample={row['sample_id']} repeat={repeat_idx} "
            f"elapsed={row['elapsed_sec']:.3f}s tps={row['completion_tps']:.2f} "
            f"accept_rate={row['acceptance_rate']} accept_len={row['acceptance_length']}"
        )


def _request_tasks(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], int]]:
    return [
        (sample, repeat_idx)
        for sample in samples
        for repeat_idx in range(args.repeats)
    ]


def run_concurrent_sample_repeats(
    handle: Any,
    client: OpenAI,
    server_root: str,
    experiment: Experiment,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    del client  # Each worker creates its own OpenAI client.
    tasks = _request_tasks(samples, args)
    concurrency = max(1, int(args.concurrency))
    for group_idx, start_idx in enumerate(range(0, len(tasks), concurrency)):
        group = tasks[start_idx : start_idx + concurrency]
        before = base.fetch_spec_decode_metrics(server_root)
        results: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
        skipped = False
        group_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(group)) as executor:
            futures = {}
            for sample, repeat_idx in group:
                request_id = make_request_id(
                    args.request_id_prefix,
                    sample.get("sample_id", "sample"),
                    experiment.name,
                    repeat_idx,
                )
                worker_client = OpenAI(
                    base_url=args.base_url,
                    api_key="EMPTY",
                    timeout=args.request_timeout_sec,
                )
                future = executor.submit(
                    run_one_request,
                    worker_client,
                    sample,
                    args,
                    request_id,
                )
                futures[future] = (sample, repeat_idx)
            for future in as_completed(futures):
                sample, repeat_idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    error_msg = str(e)
                    if "maximum context length" in error_msg or "400" in error_msg:
                        print(
                            f"[SKIP] {experiment.name} sample={sample.get('sample_id')} "
                            f"repeat={repeat_idx}: {error_msg[:120]}"
                        )
                        skipped = True
                        continue
                    raise
                results.append((sample, repeat_idx, result))
        group_elapsed = time.perf_counter() - group_start
        after = base.fetch_spec_decode_metrics(server_root)
        group_delta = base.compute_delta(before, after)
        group_completion_tokens = sum(
            int(result.get("completion_tokens") or 0)
            for _, _, result in results
        )
        group_prompt_tokens = sum(
            int(result.get("prompt_tokens") or 0)
            for _, _, result in results
        )
        group_total_tokens = sum(
            int(result.get("total_tokens") or 0)
            for _, _, result in results
        )
        group_completion_tps = (
            group_completion_tokens / group_elapsed
            if group_elapsed > 0
            else None
        )
        for sample, repeat_idx, result in sorted(
            results,
            key=lambda item: (
                str(item[0].get("sample_id")),
                int(item[1]),
            ),
        ):
            delta = result.get("_spec_delta") or group_delta
            row = make_record_row(
                experiment,
                sample,
                args,
                repeat_idx,
                result,
                delta,
                metric_scope="concurrency_group",
                concurrency_group=group_idx,
                concurrency_group_size=len(results),
            )
            row.update(
                {
                    "group_elapsed_sec": group_elapsed,
                    "group_prompt_tokens": group_prompt_tokens,
                    "group_completion_tokens": group_completion_tokens,
                    "group_total_tokens": group_total_tokens,
                    "group_completion_tps": group_completion_tps,
                }
            )
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(
                f"[{experiment.name}] group={group_idx} sample={row['sample_id']} "
                f"repeat={repeat_idx} elapsed={row['elapsed_sec']:.3f}s "
                f"req_tps={row['completion_tps']:.2f} "
                f"group_tps={group_completion_tps:.2f} "
                f"group_accept_rate={row['acceptance_rate']} "
                f"group_accept_len={row['acceptance_length']}"
            )
        handle.flush()
        if skipped and not results:
            continue


def _write_global_metric_row(
    handle: Any,
    experiment: Experiment,
    args: argparse.Namespace,
    delta: dict[str, Any] | None,
    *,
    wall_time_sec: float | None = None,
) -> None:
    if delta is None and wall_time_sec is None:
        return
    row: dict[str, Any] = {
        "experiment": experiment.name,
        "window": experiment.window,
        "window_mode": "target_only"
        if experiment.target_only
        else (experiment.window_mode if experiment.window != "full" else "full"),
        "sink_tokens": int(experiment.sink_tokens) if experiment.sink_tokens else 0,
        "suffix_decoding": experiment.suffix_decoding,
        "suffix_threshold": experiment.suffix_threshold,
        "suffix_alpha": experiment.suffix_alpha,
        "suffix_max_predict_len": experiment.suffix_max_predict_len,
        "suffix_verifier": experiment.suffix_verifier,
        "metric_scope": "experiment",
        "concurrency": args.concurrency,
        "concurrency_scheduler": args.concurrency_scheduler,
        "acceptance_rate": delta["acceptance_rate"] if delta else None,
        "acceptance_length": delta["acceptance_length"] if delta else None,
        "num_drafts": delta["num_drafts"] if delta else None,
        "draft_tokens": delta["draft_tokens"] if delta else None,
        "accepted_tokens": delta["accepted_tokens"] if delta else None,
        "experiment_wall_time_sec": wall_time_sec,
    }
    row.update(position_metric_fields(delta))
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    handle.flush()
    print(
        f"[{experiment.name}] experiment_accept_rate={row['acceptance_rate']} "
        f"experiment_accept_len={row['acceptance_length']} "
        f"num_drafts={row['num_drafts']}"
    )


def run_sliding_sample_repeats(
    handle: Any,
    client: OpenAI,
    server_root: str,
    experiment: Experiment,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    del client  # Each worker creates its own OpenAI client.
    tasks = _request_tasks(samples, args)
    if not tasks:
        return
    concurrency = max(1, int(args.concurrency))
    before = base.fetch_spec_decode_metrics(server_root)
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for sample, repeat_idx in tasks:
            request_id = make_request_id(
                args.request_id_prefix,
                sample.get("sample_id", "sample"),
                experiment.name,
                repeat_idx,
            )
            worker_client = OpenAI(
                base_url=args.base_url,
                api_key="EMPTY",
                timeout=args.request_timeout_sec,
            )
            future = executor.submit(
                run_one_request,
                worker_client,
                sample,
                args,
                request_id,
            )
            futures[future] = (sample, repeat_idx)
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{experiment.name} sliding"):
            sample, repeat_idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                error_msg = str(e)
                if "maximum context length" in error_msg or "400" in error_msg:
                    print(
                        f"[SKIP] {experiment.name} sample={sample.get('sample_id')} "
                        f"repeat={repeat_idx}: {error_msg[:120]}"
                    )
                    continue
                raise
            row = make_record_row(
                experiment,
                sample,
                args,
                repeat_idx,
                result,
                None,
                metric_scope="request",
            )
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(
                f"[{experiment.name}] sample={row['sample_id']} repeat={repeat_idx} "
                f"elapsed={row['elapsed_sec']:.3f}s tps={row['completion_tps']:.2f}"
            )
    wall_time_sec = time.perf_counter() - wall_start
    after = base.fetch_spec_decode_metrics(server_root)
    _write_global_metric_row(
        handle,
        experiment,
        args,
        base.compute_delta(before, after),
        wall_time_sec=wall_time_sec,
    )


def run_warmup_requests(
    client: OpenAI,
    server_root: str,
    experiment: Experiment,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if args.warmup_requests <= 0 or not samples:
        return
    for warmup_idx in range(args.warmup_requests):
        sample = samples[warmup_idx % len(samples)]
        request_id = make_request_id(
            args.request_id_prefix,
            sample.get("sample_id", "sample"),
            experiment.name,
            f"warmup{warmup_idx}",
        )
        before = base.fetch_spec_decode_metrics(server_root)
        try:
            result = run_one_request(client, sample, args, request_id=request_id)
        except Exception as e:
            error_msg = str(e)
            if "maximum context length" in error_msg or "400" in error_msg:
                print(
                    f"[WARMUP SKIP] {experiment.name} sample={sample.get('sample_id')} "
                    f"warmup={warmup_idx}: {error_msg[:120]}"
                )
                break
            raise
        after = base.fetch_spec_decode_metrics(server_root)
        delta = result.get("_spec_delta") or base.compute_delta(before, after)
        print(
            f"[WARMUP {experiment.name}] sample={sample.get('sample_id')} warmup={warmup_idx} "
            f"elapsed={result['elapsed_sec']:.3f}s tps={result['completion_tps']:.2f} "
            f"accept_rate={delta['acceptance_rate'] if delta else None} "
            f"accept_len={delta['acceptance_length'] if delta else None}"
        )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_summary(
    records_path: Path,
    per_sample_path: Path,
    csv_path: Path,
    json_path: Path,
) -> None:
    rows = [json.loads(line) for line in records_path.read_text("utf-8").splitlines() if line.strip()]
    if not rows:
        return
    _write_csv(per_sample_path, rows)
    df = pd.DataFrame(rows)
    numeric_cols = [
        "elapsed_sec",
        "completion_tps",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "group_elapsed_sec",
        "group_prompt_tokens",
        "group_completion_tokens",
        "group_total_tokens",
        "group_completion_tps",
        "acceptance_rate",
        "acceptance_length",
        "num_drafts",
        "draft_tokens",
        "accepted_tokens",
    ]
    numeric_cols.extend(
        col
        for col in df.columns
        if col.startswith("acceptance_rate_pos_") or col.startswith("accepted_tokens_pos_")
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    group_keys = ["experiment", "window_mode", "window"]
    if "metric_scope" in df.columns:
        row_metric_df = df[df["metric_scope"] != "experiment"]
    else:
        row_metric_df = df
    group_agg: dict[str, tuple[str, str]] = {
        "samples": ("sample_id", "count"),
        "mean_elapsed": ("elapsed_sec", "mean"),
        "mean_tps": ("completion_tps", "mean"),
        "mean_prompt_tokens": ("prompt_tokens", "mean"),
        "mean_completion_tokens": ("completion_tokens", "mean"),
        "mean_accept_rate": ("acceptance_rate", "mean"),
        "mean_accept_len": ("acceptance_length", "mean"),
    }
    for col in (
        "acceptance_rate_pos_1_4",
        "acceptance_rate_pos_5_8",
        "acceptance_rate_pos_9_12",
        "acceptance_rate_pos_13_15",
    ):
        if col in df.columns:
            group_agg[f"mean_{col}"] = (col, "mean")
    grouped = row_metric_df.groupby(group_keys, dropna=False).agg(
        **group_agg
    ).reset_index()

    experiment_metrics = df[df.get("metric_scope") == "experiment"]
    if not experiment_metrics.empty:
        experiment_cols = [
            col
            for col in [
                "acceptance_rate",
                "acceptance_length",
                "num_drafts",
                "draft_tokens",
                "accepted_tokens",
                "experiment_wall_time_sec",
            ]
            if col in experiment_metrics.columns
        ]
        experiment_cols.extend(
            col
            for col in experiment_metrics.columns
            if col.startswith("acceptance_rate_pos_")
            or col.startswith("accepted_tokens_pos_")
        )
        rename_cols = {
            col: f"global_{col}"
            for col in experiment_cols
            if col != "experiment_wall_time_sec"
        }
        rename_cols["experiment_wall_time_sec"] = "global_wall_time_sec"
        global_metrics = (
            experiment_metrics[group_keys + experiment_cols]
            .drop_duplicates(subset=group_keys, keep="last")
            .rename(columns=rename_cols)
        )
        grouped = grouped.merge(global_metrics, on=group_keys, how="left")

    total_rows = []
    if "group_completion_tps" in df.columns:
        concurrent_df = row_metric_df[
            (row_metric_df.get("metric_scope") == "concurrency_group")
            & row_metric_df["group_completion_tokens"].notna()
            & row_metric_df["group_elapsed_sec"].notna()
        ]
    else:
        concurrent_df = pd.DataFrame()
    if not concurrent_df.empty:
        unique_groups = concurrent_df.drop_duplicates(
            subset=[*group_keys, "concurrency_group"]
        )
        for key, sub_df in unique_groups.groupby(group_keys, dropna=False):
            total_completion_tokens = float(sub_df["group_completion_tokens"].sum())
            total_elapsed = float(sub_df["group_elapsed_sec"].sum())
            total_rows.append(
                {
                    **dict(zip(group_keys, key if isinstance(key, tuple) else (key,))),
                    "total_completion_tokens": total_completion_tokens,
                    "total_elapsed_sec": total_elapsed,
                    "total_completion_tps": (
                        total_completion_tokens / total_elapsed
                        if total_elapsed > 0
                        else None
                    ),
                }
            )
    serial_df = row_metric_df[
        (row_metric_df.get("metric_scope") != "concurrency_group")
        & row_metric_df["completion_tokens"].notna()
        & row_metric_df["elapsed_sec"].notna()
    ]
    if not serial_df.empty:
        for key, sub_df in serial_df.groupby(group_keys, dropna=False):
            total_completion_tokens = float(sub_df["completion_tokens"].sum())
            total_elapsed = float(sub_df["elapsed_sec"].sum())
            total_rows.append(
                {
                    **dict(zip(group_keys, key if isinstance(key, tuple) else (key,))),
                    "total_completion_tokens": total_completion_tokens,
                    "total_elapsed_sec": total_elapsed,
                    "total_completion_tps": (
                        total_completion_tokens / total_elapsed
                        if total_elapsed > 0
                        else None
                    ),
                }
            )
    if total_rows:
        grouped = grouped.merge(
            pd.DataFrame(total_rows),
            on=group_keys,
            how="left",
        )
        if "global_wall_time_sec" in grouped.columns:
            use_global_wall = grouped["global_wall_time_sec"].notna()
            grouped.loc[use_global_wall, "total_elapsed_sec"] = grouped.loc[
                use_global_wall, "global_wall_time_sec"
            ]
            grouped.loc[use_global_wall, "total_completion_tps"] = (
                grouped.loc[use_global_wall, "total_completion_tokens"]
                / grouped.loc[use_global_wall, "global_wall_time_sec"]
            )

    target_rows = grouped[
        (grouped["window_mode"] == "target_only")
        | (grouped["experiment"] == "target-only")
    ]
    if not target_rows.empty:
        target_elapsed = float(target_rows["mean_elapsed"].iloc[0])
        target_tps = float(target_rows["mean_tps"].iloc[0])
        target_total_tps = (
            float(target_rows["total_completion_tps"].iloc[0])
            if "total_completion_tps" in target_rows
            and pd.notna(target_rows["total_completion_tps"].iloc[0])
            else target_tps
        )
        grouped["latency_speedup_vs_target_only"] = grouped["mean_elapsed"].apply(
            lambda value: target_elapsed / value if value and value > 0 else None
        )
        grouped["tps_speedup_vs_target_only"] = grouped["mean_tps"].apply(
            lambda value: value / target_tps if target_tps and target_tps > 0 else None
        )
        if "total_completion_tps" in grouped:
            grouped["total_tps_speedup_vs_target_only"] = grouped[
                "total_completion_tps"
            ].apply(
                lambda value: value / target_total_tps
                if target_total_tps and target_total_tps > 0
                else None
            )
    original_rows = grouped[grouped["experiment"] == "original"]
    if not original_rows.empty:
        original_elapsed = float(original_rows["mean_elapsed"].iloc[0])
        original_tps = float(original_rows["mean_tps"].iloc[0])
        original_total_tps = (
            float(original_rows["total_completion_tps"].iloc[0])
            if "total_completion_tps" in original_rows
            and pd.notna(original_rows["total_completion_tps"].iloc[0])
            else original_tps
        )
        grouped["latency_speedup_vs_original"] = grouped["mean_elapsed"].apply(
            lambda value: original_elapsed / value if value and value > 0 else None
        )
        grouped["tps_speedup_vs_original"] = grouped["mean_tps"].apply(
            lambda value: value / original_tps if original_tps and original_tps > 0 else None
        )
        if "total_completion_tps" in grouped:
            grouped["total_tps_speedup_vs_original"] = grouped[
                "total_completion_tps"
            ].apply(
                lambda value: value / original_total_tps
                if original_total_tps and original_total_tps > 0
                else None
            )
    grouped.to_csv(csv_path, index=False)
    summary_records = json.loads(grouped.to_json(orient="records"))
    json_path.write_text(
        json.dumps({"by_experiment": summary_records}, indent=2),
        "utf-8",
    )


def main() -> int:
    args = parse_args()
    args.concurrency = max(1, int(args.concurrency))
    samples = load_samples(Path(args.samples))
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    for sample in samples:
        messages_for_sample(sample)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    args._tokenizer = tokenizer
    suffix_source_tail_token_ids = unique_generation_prompt_tail_token_ids(
        tokenizer,
        samples,
        enable_thinking=not args.disable_thinking,
    )
    args.suffix_source_tail_token_ids = suffix_source_tail_token_ids
    experiments = load_experiments(args)
    needs_state_ranges = any(experiment.window_mode == "range_recent_once" for experiment in experiments)
    state_ranges_by_sample = {
        str(sample.get("sample_id")): compute_state_ranges(
            tokenizer,
            render_sample_text(
                tokenizer,
                sample,
                add_generation_prompt=True,
            ),
            state_needles_for_sample(sample, args),
        )
        for sample in samples
    } if needs_state_ranges else {}

    server_root = base.server_root_from_base_url(args.base_url)
    server_port = base.server_port_from_base_url(args.base_url)
    manage_server = not args.no_manage_server
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    vllm_cache_root = Path(args.vllm_cache_root) if args.vllm_cache_root else out_dir / "vllm_cache"
    if manage_server:
        vllm_cache_root.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    per_sample_path = out_dir / "per_sample.csv"
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    config = {
        "created_utc": timestamp,
        "samples": args.samples,
        "sample_count": len(samples),
        "sample_ids": [sample.get("sample_id") for sample in samples],
        "experiments": [experiment.__dict__ for experiment in experiments],
        "start_script": args.start_script,
        "target_start_script": args.target_start_script,
        "state_ranges_by_sample": state_ranges_by_sample,
        "repeats": args.repeats,
        "warmup_requests": args.warmup_requests,
        "concurrency": args.concurrency,
        "concurrency_scheduler": args.concurrency_scheduler,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tool_call_parser": args.tool_call_parser,
        "reasoning_parser": args.reasoning_parser,
        "allow_long_max_model_len": args.allow_long_max_model_len,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_tokens": args.min_tokens,
        "ignore_eos": args.ignore_eos,
        "request_mode": args.request_mode,
        "num_spec_tokens": args.num_spec_tokens,
        "model": args.model,
        "message_style": (
            "local_rendered_prompt"
            if args.request_mode == "completion"
            else "request_replay_messages"
        ),
        "json_response_format": args.json_response_format,
        "debug_action_validation": args.debug_action_validation,
        "continue_on_error": args.continue_on_error,
        "state_per_sample_server": args.state_per_sample_server,
        "vllm_cache_root": str(vllm_cache_root) if manage_server else None,
        "dflash_recent_tokens": args.dflash_recent_tokens,
        "dflash_suffix_match_tokens": args.dflash_suffix_match_tokens,
        "dflash_suffix_keep_tokens": args.dflash_suffix_keep_tokens,
        "dflash_suffix_middle_budget": args.dflash_suffix_middle_budget,
        "dflash_suffix_source_tail_token_ids": suffix_source_tail_token_ids,
        "dynamic_yarn_original_max_position_embeddings": args.dynamic_yarn_original_max_position_embeddings,
        "dynamic_yarn_max_factor": args.dynamic_yarn_max_factor,
        "dynamic_yarn_mode": args.dynamic_yarn_mode,
        "dynamic_yarn_length_ratio": args.dynamic_yarn_length_ratio,
        "target_yarn_original_max_position_embeddings": args.target_yarn_original_max_position_embeddings,
        "target_yarn_max_position_embeddings": args.target_yarn_max_position_embeddings,
        "target_yarn_factor": args.target_yarn_factor,
        "original_max_position_embedding": [
            experiment.original_max_position_embedding for experiment in experiments
        ],
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), "utf-8")
    print(json.dumps(config, indent=2))

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.request_timeout_sec)
    run_failed = False
    with records_path.open("w", encoding="utf-8") as handle:
        try:
            for experiment in experiments:
                per_sample_state_server = (
                    manage_server
                    and args.state_per_sample_server
                    and experiment.window_mode == "range_recent_once"
                )
                if per_sample_state_server:
                    for sample in samples:
                        sample_id = str(sample.get("sample_id"))
                        try:
                            base.stop_existing_server(args.model, port=server_port)
                            server_args = make_server_args(
                                args,
                                experiment,
                                vllm_cache_root=vllm_cache_root,
                                select_ranges=state_ranges_by_sample.get(sample_id, ""),
                            )
                            proc = base.start_server(
                                server_args,
                                experiment.window,
                                out_dir / f"server_{experiment.name}_{sample_id}.log",
                            )
                            base.wait_for_server(server_root, args.startup_timeout_sec, args.poll_interval_sec, proc)
                            run_warmup_requests(client, server_root, experiment, [sample], args)
                            run_sample_repeats(handle, client, server_root, experiment, sample, args)
                        except Exception as e:
                            print(
                                f"[ERROR] {experiment.name} sample={sample_id}: "
                                f"{type(e).__name__}: {str(e)[:240]}"
                            )
                            run_failed = True
                            if not args.continue_on_error:
                                raise
                        finally:
                            base.stop_existing_server(args.model, port=server_port)
                    continue

                try:
                    if manage_server:
                        base.stop_existing_server(args.model, port=server_port)
                        select_ranges = ""
                        if experiment.window_mode == "range_recent_once" and samples:
                            select_ranges = state_ranges_by_sample.get(str(samples[0].get("sample_id")), "")
                        server_args = make_server_args(
                            args,
                            experiment,
                            vllm_cache_root=vllm_cache_root,
                            select_ranges=select_ranges,
                        )
                        proc = base.start_server(
                            server_args,
                            experiment.window,
                            out_dir / f"server_{experiment.name}.log",
                        )
                        base.wait_for_server(server_root, args.startup_timeout_sec, args.poll_interval_sec, proc)
                        run_warmup_requests(client, server_root, experiment, samples, args)
                    if args.concurrency > 1 and args.concurrency_scheduler == "batch":
                        run_concurrent_sample_repeats(
                            handle,
                            client,
                            server_root,
                            experiment,
                            samples,
                            args,
                        )
                    elif args.concurrency > 1:
                        run_sliding_sample_repeats(
                            handle,
                            client,
                            server_root,
                            experiment,
                            samples,
                            args,
                        )
                    else:
                        for sample in samples:
                            run_sample_repeats(
                                handle,
                                client,
                                server_root,
                                experiment,
                                sample,
                                args,
                            )
                except Exception as e:
                    print(
                        f"[ERROR] {experiment.name}: {type(e).__name__}: "
                        f"{str(e)[:240]}"
                    )
                    run_failed = True
                    if not args.continue_on_error:
                        raise
                finally:
                    if manage_server:
                        base.stop_existing_server(args.model, port=server_port)
        finally:
            handle.flush()
            write_summary(records_path, per_sample_path, csv_path, json_path)
    print(f"Records: {records_path}")
    print(f"Per-sample CSV: {per_sample_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    return 1 if run_failed and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
