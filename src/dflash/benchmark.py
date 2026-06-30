from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from .benchmark_common import (
    _apply_chat_template,
    _check_transformers_model,
    _dist_gather,
    _dist_init,
    _dist_is_main,
    _dist_local_rank,
    _dist_rank,
    _dist_size,
    _get_transformers_attn_impl,
    _maybe_configure_draft_yarn_rope,
    _maybe_configure_target_yarn_rope,
    load_and_process_dataset,
)


def limit_dataset(dataset: list[dict[str, Any]], max_samples: int | None, *, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or len(dataset) <= max_samples:
        return list(dataset)
    selected = list(dataset)
    random.Random(seed).shuffle(selected)
    return selected[:max_samples]


def shard_dataset(dataset: list[dict[str, Any]], *, rank: int, world_size: int) -> list[tuple[int, dict[str, Any]]]:
    return [(idx, dataset[idx]) for idx in range(rank, len(dataset), world_size)]


def compute_acceptance_metrics(acceptance_lengths: list[int], block_size: int) -> dict[str, Any]:
    histogram = Counter(str(int(length)) for length in acceptance_lengths)
    accepted_tokens = int(sum(acceptance_lengths))
    decode_rounds = len(acceptance_lengths)
    mean_acceptance_length = accepted_tokens / decode_rounds if decode_rounds else 0.0
    acceptance_rate = (
        accepted_tokens / (decode_rounds * block_size)
        if decode_rounds and block_size > 0
        else 0.0
    )
    return {
        "decode_rounds": decode_rounds,
        "accepted_tokens": accepted_tokens,
        "mean_acceptance_length": mean_acceptance_length,
        "acceptance_rate": acceptance_rate,
        "acceptance_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _flatten_numeric_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if not isinstance(raw, list):
            continue
        values.extend(float(value) for value in raw if value is not None)
    return values


def _configure_draft_sliding_window(config: Any, window_size: int | None) -> bool:
    if window_size is None:
        return False
    if window_size <= 0:
        raise ValueError(f"draft sliding window size must be positive, got {window_size}")
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError("draft config must define a positive num_hidden_layers for sliding window attention")

    config.sliding_window = int(window_size)
    config.layer_types = ["sliding_attention"] * num_hidden_layers
    return True


def _summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in records if row.get("status") == "ok"]
    skipped_rows = [row for row in records if str(row.get("status", "")).startswith("skipped")]
    oom_rows = [row for row in records if row.get("status") == "oom"]
    error_rows = [row for row in records if row.get("status") == "error"]

    baseline_tpot = _mean([float(row["baseline_tpot"]) for row in ok_rows if row.get("baseline_tpot") is not None])
    dflash_tpot = _mean([float(row["dflash_tpot"]) for row in ok_rows if row.get("dflash_tpot") is not None])
    summary = {
        "samples_total": len(records),
        "samples_ok": len(ok_rows),
        "samples_skipped": len(skipped_rows),
        "samples_oom": len(oom_rows),
        "samples_error": len(error_rows),
        "mean_prompt_tokens": _mean([float(row["prompt_tokens"]) for row in ok_rows if row.get("prompt_tokens") is not None]),
        "mean_baseline_tpot": baseline_tpot,
        "mean_dflash_tpot": dflash_tpot,
        "mean_speedup": _mean([float(row["speedup"]) for row in ok_rows if row.get("speedup") is not None]),
        "mean_acceptance_length": _mean(
            [float(row["mean_acceptance_length"]) for row in ok_rows if row.get("mean_acceptance_length") is not None]
        ),
        "mean_acceptance_rate": _mean([float(row["acceptance_rate"]) for row in ok_rows if row.get("acceptance_rate") is not None]),
        "mean_draft_forward_passes": _mean(
            [float(row["draft_forward_passes"]) for row in ok_rows if row.get("draft_forward_passes") is not None]
        ),
        "mean_draft_dynamic_yarn_factor": _mean(
            [
                float(row["draft_dynamic_yarn_factor"])
                for row in ok_rows
                if row.get("draft_dynamic_yarn_factor") is not None
            ]
        ),
        "min_draft_dynamic_yarn_factor": min(
            [
                float(row["draft_dynamic_yarn_factor"])
                for row in ok_rows
                if row.get("draft_dynamic_yarn_factor") is not None
            ],
            default=None,
        ),
        "max_draft_dynamic_yarn_factor": max(
            [
                float(row["draft_dynamic_yarn_factor"])
                for row in ok_rows
                if row.get("draft_dynamic_yarn_factor") is not None
            ],
            default=None,
        ),
        "mean_verify_draft_tokens": _mean(
            [float(row["mean_verify_draft_tokens"]) for row in ok_rows if row.get("mean_verify_draft_tokens") is not None]
        ),
        "mean_suffix_match_rounds": _mean(
            [float(row["suffix_match_rounds"]) for row in ok_rows if row.get("suffix_match_rounds") is not None]
        ),
        "total_suffix_match_rounds": sum(
            [int(row["suffix_match_rounds"]) for row in ok_rows if row.get("suffix_match_rounds") is not None]
        ),
        "mean_suffix_verify_rounds": _mean(
            [float(row["suffix_verify_rounds"]) for row in ok_rows if row.get("suffix_verify_rounds") is not None]
        ),
        "total_suffix_verify_rounds": sum(
            [int(row["suffix_verify_rounds"]) for row in ok_rows if row.get("suffix_verify_rounds") is not None]
        ),
        "mean_suffix_recovery_rounds": _mean(
            [float(row["suffix_recovery_rounds"]) for row in ok_rows if row.get("suffix_recovery_rounds") is not None]
        ),
        "total_suffix_recovery_rounds": sum(
            [int(row["suffix_recovery_rounds"]) for row in ok_rows if row.get("suffix_recovery_rounds") is not None]
        ),
        "mean_suffix_zero_accept_rounds": _mean(
            [float(row["suffix_zero_accept_rounds"]) for row in ok_rows if row.get("suffix_zero_accept_rounds") is not None]
        ),
        "total_suffix_zero_accept_rounds": sum(
            [int(row["suffix_zero_accept_rounds"]) for row in ok_rows if row.get("suffix_zero_accept_rounds") is not None]
        ),
        "mean_suffix_exhausted_rounds": _mean(
            [float(row["suffix_exhausted_rounds"]) for row in ok_rows if row.get("suffix_exhausted_rounds") is not None]
        ),
        "total_suffix_exhausted_rounds": sum(
            [int(row["suffix_exhausted_rounds"]) for row in ok_rows if row.get("suffix_exhausted_rounds") is not None]
        ),
        "mean_suffix_paper_score": _mean(_flatten_numeric_list(ok_rows, "suffix_paper_scores")),
        "mean_suffix_paper_token_score": _mean(_flatten_numeric_list(ok_rows, "suffix_paper_token_scores")),
        "mean_suffix_paper_tree_size": _mean(_flatten_numeric_list(ok_rows, "suffix_paper_tree_sizes")),
        "mean_suffix_paper_best_path_score": _mean(_flatten_numeric_list(ok_rows, "suffix_paper_best_path_scores")),
        "mean_suffix_paper_max_spec": _mean(_flatten_numeric_list(ok_rows, "suffix_paper_max_specs")),
        "mean_ctx_suffix_match_count": _mean(
            [float(row["ctx_suffix_match_count"]) for row in ok_rows if row.get("ctx_suffix_match_count") is not None]
        ),
        "max_ctx_suffix_match_count": max(
            [int(row["ctx_suffix_match_count"]) for row in ok_rows if row.get("ctx_suffix_match_count") is not None],
            default=None,
        ),
        "total_ctx_suffix_match_count": sum(
            [int(row["ctx_suffix_match_count"]) for row in ok_rows if row.get("ctx_suffix_match_count") is not None]
        ),
        "mean_ctx_suffix_match_kept_tokens": _mean(
            [float(row["ctx_suffix_match_kept_tokens"]) for row in ok_rows if row.get("ctx_suffix_match_kept_tokens") is not None]
        ),
        "mean_ctx_middle_tokens_before_budget": _mean(
            [
                float(row["ctx_middle_tokens_before_budget"])
                for row in ok_rows
                if row.get("ctx_middle_tokens_before_budget") is not None
            ]
        ),
        "mean_ctx_middle_tokens_after_budget": _mean(
            [
                float(row["ctx_middle_tokens_after_budget"])
                for row in ok_rows
                if row.get("ctx_middle_tokens_after_budget") is not None
            ]
        ),
        "mean_ctx_middle_budget_dropped_tokens": _mean(
            [
                float(row["ctx_middle_budget_dropped_tokens"])
                for row in ok_rows
                if row.get("ctx_middle_budget_dropped_tokens") is not None
            ]
        ),
        "mean_ctx_total_budget": _mean(
            [float(row["ctx_total_budget"]) for row in ok_rows if row.get("ctx_total_budget") is not None]
        ),
        "mean_ctx_recent_tokens_after_budget": _mean(
            [
                float(row["ctx_recent_tokens_after_budget"])
                for row in ok_rows
                if row.get("ctx_recent_tokens_after_budget") is not None
            ]
        ),
        "mean_ctx_hidden_tokens_after": _mean(
            [float(row["ctx_hidden_tokens_after"]) for row in ok_rows if row.get("ctx_hidden_tokens_after") is not None]
        ),
        "mean_ctx_indexer_selected_blocks": _mean(
            [
                float(row["ctx_indexer_selected_blocks"])
                for row in ok_rows
                if row.get("ctx_indexer_selected_blocks") is not None
            ]
        ),
        "mean_ctx_indexer_forced_blocks": _mean(
            [
                float(row["ctx_indexer_forced_blocks"])
                for row in ok_rows
                if row.get("ctx_indexer_forced_blocks") is not None
            ]
        ),
        "throughput_baseline_tok_s": 1.0 / baseline_tpot if baseline_tpot and baseline_tpot > 0 else None,
        "throughput_dflash_tok_s": 1.0 / dflash_tpot if dflash_tpot and dflash_tpot > 0 else None,
    }
    profile_rows = [row for row in ok_rows if row.get("profiler_target_verify_time_s") is not None]
    if profile_rows:
        def profiler_per_call(time_key: str, calls_key: str) -> float | None:
            total_time = sum(float(row.get(time_key) or 0.0) for row in profile_rows)
            total_calls = sum(int(row.get(calls_key) or 0) for row in profile_rows)
            return total_time / total_calls if total_calls else None

        summary.update(
            profiler_first_prefill_time_per_call_s=profiler_per_call(
                "profiler_first_prefill_time_s",
                "profiler_first_prefill_calls",
            ),
            profiler_draft_prefill_time_per_call_s=profiler_per_call(
                "profiler_draft_prefill_time_s",
                "profiler_draft_prefill_calls",
            ),
            profiler_draft_generate_time_per_call_s=profiler_per_call(
                "profiler_draft_generate_time_s",
                "profiler_draft_generate_calls",
            ),
            profiler_target_verify_time_per_call_s=profiler_per_call(
                "profiler_target_verify_time_s",
                "profiler_target_verify_calls",
            ),
        )
    return summary


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


def write_benchmark_outputs(out_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_records = [_profiler_output_record(record) for record in records]
    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summary_from_records(records)
    _write_csv(out_dir / "per_sample.csv", output_records)
    summary_rows = [summary]
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    return summary


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _per_call(row: dict[str, Any], time_key: str, calls_key: str) -> float | None:
    calls = int(row.get(calls_key) or 0)
    return float(row.get(time_key) or 0.0) / calls if calls else None


def _profiler_output_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in record.items()
        if not (key.startswith("profiler_") and (key.endswith("_time_s") or key.endswith("_calls")))
    }
    if record.get("profiler_target_verify_time_s") is not None:
        output.update(
            profiler_first_prefill_time_per_call_s=_per_call(
                record,
                "profiler_first_prefill_time_s",
                "profiler_first_prefill_calls",
            ),
            profiler_draft_prefill_time_per_call_s=_per_call(
                record,
                "profiler_draft_prefill_time_s",
                "profiler_draft_prefill_calls",
            ),
            profiler_draft_generate_time_per_call_s=_per_call(
                record,
                "profiler_draft_generate_time_s",
                "profiler_draft_generate_calls",
            ),
            profiler_target_verify_time_per_call_s=_per_call(
                record,
                "profiler_target_verify_time_s",
                "profiler_target_verify_calls",
            ),
        )
    return output


def _write_profiler_markdown(out_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    profile_rows = [row for row in records if row.get("status") == "ok" and row.get("profiler_target_verify_time_s") is not None]
    path = out_dir / "profiler.md"
    lines = [
        "# DFlash Profiler",
        "",
        "## Definitions",
        "",
        "- `first_prefill_time`: initial prompt prefill through first sampled target token and context hidden preparation.",
        "- `draft_prefill_time`: first draft generation stage with an empty draft KV cache.",
        "- `draft_generate_time`: subsequent draft stages that generate draft tokens.",
        "- `target_verify_time`: target model verify forward calls during DFlash decoding.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| first_prefill_time_per_call_s | {_format_seconds(summary.get('profiler_first_prefill_time_per_call_s'))} |",
        f"| draft_prefill_time_per_call_s | {_format_seconds(summary.get('profiler_draft_prefill_time_per_call_s'))} |",
        f"| draft_generate_time_per_call_s | {_format_seconds(summary.get('profiler_draft_generate_time_per_call_s'))} |",
        f"| target_verify_time_per_call_s | {_format_seconds(summary.get('profiler_target_verify_time_per_call_s'))} |",
        "",
        "## Per Sample",
        "",
        "| sample_index | sample_id | first_prefill_per_call_s | draft_prefill_per_call_s | draft_generate_per_call_s | target_verify_per_call_s |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in profile_rows:
        output_row = _profiler_output_record(row)
        lines.append(
            "| "
            f"{row.get('sample_index')} | "
            f"{_markdown_cell(row.get('sample_id'))} | "
            f"{_format_seconds(output_row.get('profiler_first_prefill_time_per_call_s'))} | "
            f"{_format_seconds(output_row.get('profiler_draft_prefill_time_per_call_s'))} | "
            f"{_format_seconds(output_row.get('profiler_draft_generate_time_per_call_s'))} | "
            f"{_format_seconds(output_row.get('profiler_target_verify_time_per_call_s'))} |"
        )
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


def _sample_id(sample: dict[str, Any], sample_index: int) -> str:
    return str(sample.get("sample_id") or sample.get("instance_id") or sample.get("id") or f"sample_{sample_index}")


def _render_prompt(
    tokenizer: Any,
    sample: dict[str, Any],
    enable_thinking: bool,
    *,
    add_generation_prompt: bool = True,
) -> str:
    if "messages" in sample:
        return _apply_chat_template(
            tokenizer,
            sample["messages"],
            enable_thinking,
            tools=sample.get("tools"),
            add_generation_prompt=add_generation_prompt,
        )
    messages = [{"role": "user", "content": turn} for turn in sample["turns"]]
    return _apply_chat_template(tokenizer, messages, enable_thinking, add_generation_prompt=add_generation_prompt)


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return safe[:120] or "sample"


def _write_response_texts(
    responses_dir: Path,
    sample_index: int,
    sample_id: str,
    tokenizer: Any,
    baseline: Any,
    dflash: Any,
) -> tuple[str, str]:
    responses_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{sample_index:05d}_{_safe_filename(sample_id)}"
    baseline_path = responses_dir / f"{prefix}_baseline.txt"
    dflash_path = responses_dir / f"{prefix}_dflash.txt"
    baseline_ids = baseline.output_ids[0, baseline.num_input_tokens:]
    dflash_ids = dflash.output_ids[0, dflash.num_input_tokens:]
    baseline_path.write_text(tokenizer.decode(baseline_ids, skip_special_tokens=True), "utf-8")
    dflash_path.write_text(tokenizer.decode(dflash_ids, skip_special_tokens=True), "utf-8")
    return str(baseline_path), str(dflash_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed local Transformers DFlash benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target-device-map", type=str, default=None)
    parser.add_argument("--draft-device-map", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/dflash_benchmark")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--ctx-sink-tokens", type=int, default=0)
    parser.add_argument("--ctx-recent-window", type=int, default=0)
    parser.add_argument("--ctx-stride", type=int, default=0)
    parser.add_argument("--ctx-suffix-match-tokens", type=int, default=0)
    parser.add_argument("--ctx-suffix-keep-tokens", type=int, default=0)
    parser.add_argument("--ctx-middle-budget", type=int, default=0)
    parser.add_argument("--ctx-total-budget", type=int, default=None)
    parser.add_argument("--ctx-dynamic-budget-ratio", type=float, default=None)
    parser.add_argument("--ctx-budget-order", choices=["default", "suffix_then_recent"], default="default")
    parser.add_argument("--ctx-indexer-enable", action="store_true")
    parser.add_argument("--ctx-indexer-block-size", type=int, default=4)
    parser.add_argument("--ctx-indexer-top-k-blocks", type=int, default=512)
    parser.add_argument("--ctx-indexer-query-tokens", type=int, default=512)
    parser.add_argument("--ctx-indexer-score-reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--draft-denoise-steps", type=int, default=1)
    parser.add_argument("--save-verify-trace", action="store_true")
    parser.add_argument("--verify-trace-max-rounds", type=int, default=0)
    parser.add_argument("--verify-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--verify-min-draft-tokens", type=int, default=1)
    parser.add_argument("--suffix-decoding", action="store_true")
    parser.add_argument("--suffix-strategy", choices=["consensus", "paper"], default="consensus")
    parser.add_argument("--suffix-max-query-len", type=int, default=16)
    parser.add_argument("--suffix-min-query-len", type=int, default=2)
    parser.add_argument("--suffix-top-k", type=int, default=4)
    parser.add_argument("--suffix-min-support", type=int, default=3)
    parser.add_argument("--suffix-min-predict-len", type=int, default=8)
    parser.add_argument("--suffix-max-predict-len", type=int, default=None)
    parser.add_argument("--suffix-paper-alpha", type=float, default=1.0)
    parser.add_argument("--suffix-paper-max-spec-offset", type=float, default=0.0)
    parser.add_argument("--suffix-paper-min-token-prob", type=float, default=0.0)
    parser.add_argument("--suffix-paper-threshold", type=float, default=0.0)
    parser.add_argument("--suffix-paper-max-matches", type=int, default=0)
    parser.add_argument("--suffix-paper-verifier", choices=["linear", "tree"], default="linear")
    parser.add_argument("--suffix-paper-tree-attn-impl", default="flash_attention_2")
    parser.add_argument("--suffix-fallback", choices=["dflash", "target", "none"], default="dflash")
    parser.add_argument("--save-suffix-trace", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--draft-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--draft-yarn-max-position-embeddings", type=int, default=None)
    parser.add_argument("--draft-yarn-factor", type=float, default=None)
    parser.add_argument("--draft-dynamic-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--draft-dynamic-yarn-max-factor", type=float, default=None)
    parser.add_argument("--draft-dynamic-yarn-mode", choices=["continuous", "bucket"], default="continuous")
    parser.add_argument("--draft-dynamic-yarn-length-ratio", type=float, default=None)
    parser.add_argument(
        "--draft-sliding-window-size",
        "--draft-swa-window-size",
        dest="draft_sliding_window_size",
        type=int,
        default=None,
        help="Force all draft layers to sliding-window attention with this window size.",
    )
    parser.add_argument("--target-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--target-yarn-max-position-embeddings", type=int, default=None)
    parser.add_argument("--target-yarn-factor", type=float, default=None)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--no-save-responses", action="store_true")
    return parser.parse_args()


def _make_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{_safe_filename(args.run_name)}" if args.run_name else ""
    return Path(args.output_dir).expanduser() / f"{timestamp}{suffix}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import distributed as torch_dist
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .model import DFlashDraftModel, dflash_generate

    _check_transformers_model(args.model)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    _dist_init(torch_dist)
    if args.draft_dynamic_yarn_original_max_position_embeddings is not None and (
        args.draft_yarn_original_max_position_embeddings is not None
        or args.draft_yarn_max_position_embeddings is not None
        or args.draft_yarn_factor is not None
    ):
        raise ValueError("draft dynamic YaRN cannot be combined with static draft YaRN arguments")
    if torch.cuda.is_available():
        torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA is not available; running on CPU will be very slow.")
    attn_impl = _get_transformers_attn_impl()
    if _dist_is_main():
        logger.info("Using Transformers attention implementation: {}", attn_impl)

    out_dir = _make_output_dir(args)
    if torch_dist.is_initialized():
        out_dir_objs = [str(out_dir)] if _dist_is_main() else [None]
        torch_dist.broadcast_object_list(out_dir_objs, src=0)
        out_dir = Path(str(out_dir_objs[0]))
    if _dist_is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = out_dir / "responses"

    target_config = AutoConfig.from_pretrained(args.model)
    if _maybe_configure_target_yarn_rope(target_config, args) and _dist_is_main():
        logger.info(
            "Enabled target YaRN RoPE: max_position_embeddings={}, original_max_position_embeddings={}, factor={}",
            target_config.max_position_embeddings,
            target_config.rope_parameters["original_max_position_embeddings"],
            target_config.rope_parameters["factor"],
        )
    target_kwargs = {
        "config": target_config,
        "attn_implementation": attn_impl,
        "dtype": torch.bfloat16,
    }
    if args.target_device_map:
        target_kwargs["device_map"] = args.target_device_map
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        **target_kwargs,
    )
    if not args.target_device_map:
        target = target.to(device)
    target = target.eval()

    draft_config = AutoConfig.from_pretrained(args.draft_model)
    if _maybe_configure_draft_yarn_rope(draft_config, args) and _dist_is_main():
        logger.info(
            "Enabled draft YaRN RoPE: max_position_embeddings={}, original_max_position_embeddings={}, factor={}",
            draft_config.max_position_embeddings,
            draft_config.rope_parameters["original_max_position_embeddings"],
            draft_config.rope_parameters["factor"],
        )
    if _configure_draft_sliding_window(draft_config, args.draft_sliding_window_size) and _dist_is_main():
        logger.info(
            "Enabled draft sliding-window attention: window_size={}, layers={}",
            draft_config.sliding_window,
            len(draft_config.layer_types),
        )
    draft_kwargs = {
        "config": draft_config,
        "attn_implementation": attn_impl,
        "dtype": torch.bfloat16,
    }
    if args.draft_device_map:
        draft_kwargs["device_map"] = args.draft_device_map
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        **draft_kwargs,
    )
    if not args.draft_device_map:
        draft = draft.to(device)
    draft = draft.eval()

    device = target.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    block_size = args.block_size if args.block_size is not None else draft.block_size
    dataset = limit_dataset(load_and_process_dataset(args.dataset), args.max_samples, seed=args.sample_seed)
    rank_samples = shard_dataset(dataset, rank=_dist_rank(), world_size=_dist_size())
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "draft_model": args.draft_model,
        "dataset": args.dataset,
        "transformers_attn_impl": attn_impl,
        "target_device_map": args.target_device_map,
        "draft_device_map": args.draft_device_map,
        "sample_count": len(dataset),
        "world_size": _dist_size(),
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "block_size": block_size,
        "ctx_sink_tokens": args.ctx_sink_tokens,
        "ctx_recent_window": args.ctx_recent_window,
        "ctx_stride": args.ctx_stride,
        "ctx_suffix_match_tokens": args.ctx_suffix_match_tokens,
        "ctx_suffix_keep_tokens": args.ctx_suffix_keep_tokens,
        "ctx_suffix_source": "content",
        "ctx_middle_budget": args.ctx_middle_budget,
        "ctx_total_budget": args.ctx_total_budget,
        "ctx_dynamic_budget_ratio": args.ctx_dynamic_budget_ratio,
        "ctx_budget_order": args.ctx_budget_order,
        "ctx_indexer_enable": args.ctx_indexer_enable,
        "ctx_indexer_block_size": args.ctx_indexer_block_size,
        "ctx_indexer_top_k_blocks": args.ctx_indexer_top_k_blocks,
        "ctx_indexer_query_tokens": args.ctx_indexer_query_tokens,
        "ctx_indexer_score_reduce": args.ctx_indexer_score_reduce,
        "draft_denoise_steps": args.draft_denoise_steps,
        "save_verify_trace": args.save_verify_trace,
        "verify_trace_max_rounds": args.verify_trace_max_rounds,
        "verify_confidence_threshold": args.verify_confidence_threshold,
        "verify_min_draft_tokens": args.verify_min_draft_tokens,
        "suffix_decoding": args.suffix_decoding,
        "suffix_strategy": args.suffix_strategy,
        "suffix_fallback": args.suffix_fallback,
        "suffix_max_query_len": args.suffix_max_query_len,
        "suffix_min_query_len": args.suffix_min_query_len,
        "suffix_top_k": args.suffix_top_k,
        "suffix_min_support": args.suffix_min_support,
        "suffix_min_predict_len": args.suffix_min_predict_len,
        "suffix_max_predict_len": args.suffix_max_predict_len,
        "suffix_paper_alpha": args.suffix_paper_alpha,
        "suffix_paper_max_spec_offset": args.suffix_paper_max_spec_offset,
        "suffix_paper_min_token_prob": args.suffix_paper_min_token_prob,
        "suffix_paper_threshold": args.suffix_paper_threshold,
        "suffix_paper_max_matches": args.suffix_paper_max_matches,
        "suffix_paper_verifier": args.suffix_paper_verifier,
        "suffix_paper_tree_attn_impl": args.suffix_paper_tree_attn_impl,
        "save_suffix_trace": args.save_suffix_trace,
        "draft_dynamic_yarn_original_max_position_embeddings": args.draft_dynamic_yarn_original_max_position_embeddings,
        "draft_dynamic_yarn_max_factor": args.draft_dynamic_yarn_max_factor,
        "draft_dynamic_yarn_mode": args.draft_dynamic_yarn_mode,
        "draft_dynamic_yarn_length_ratio": args.draft_dynamic_yarn_length_ratio,
        "draft_sliding_window_size": args.draft_sliding_window_size,
        "draft_sliding_window": getattr(draft.config, "sliding_window", None),
        "draft_layer_types": list(getattr(draft.config, "layer_types", []) or []),
        "profiler": args.profiler,
        "target_rope_parameters": getattr(target.config, "rope_parameters", None),
        "draft_rope_parameters": getattr(draft.config, "rope_parameters", None),
    }
    if _dist_is_main():
        (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), "utf-8")

    records: list[dict[str, Any]] = []
    verify_trace_records: list[dict[str, Any]] = []
    for sample_index, sample in tqdm(rank_samples, desc=f"Benchmarking rank {_dist_rank()}", disable=not _dist_is_main()):
        sample_id = _sample_id(sample, sample_index)
        record: dict[str, Any] = {"sample_index": sample_index, "sample_id": sample_id, "rank": _dist_rank()}
        input_ids = None
        try:
            prompt = _render_prompt(tokenizer, sample, args.enable_thinking, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            prompt_tokens = int(input_ids.shape[1])
            content_prompt = _render_prompt(tokenizer, sample, args.enable_thinking, add_generation_prompt=False)
            ctx_suffix_source_end = len(tokenizer.encode(content_prompt))
            record["prompt_tokens"] = prompt_tokens
            record["ctx_suffix_source_end"] = ctx_suffix_source_end
            if prompt_tokens > tokenizer.model_max_length:
                record.update(
                    status="skipped_length",
                    error=f"{prompt_tokens} tokens > tokenizer.model_max_length {tokenizer.model_max_length}",
                )
                records.append(record)
                continue

            baseline = dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=1,
                return_stats=True,
            )
            dflash = dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=block_size,
                return_stats=True,
                ctx_sink_tokens=args.ctx_sink_tokens,
                ctx_recent_window=args.ctx_recent_window,
                ctx_stride=args.ctx_stride,
                ctx_suffix_match_tokens=args.ctx_suffix_match_tokens,
                ctx_suffix_keep_tokens=args.ctx_suffix_keep_tokens,
                ctx_suffix_source_end=ctx_suffix_source_end,
                ctx_middle_budget=args.ctx_middle_budget,
                ctx_total_budget=args.ctx_total_budget,
                ctx_dynamic_budget_ratio=args.ctx_dynamic_budget_ratio,
                ctx_budget_order=args.ctx_budget_order,
                ctx_indexer_enable=args.ctx_indexer_enable,
                ctx_indexer_block_size=args.ctx_indexer_block_size,
                ctx_indexer_top_k_blocks=args.ctx_indexer_top_k_blocks,
                ctx_indexer_query_tokens=args.ctx_indexer_query_tokens,
                ctx_indexer_score_reduce=args.ctx_indexer_score_reduce,
                draft_denoise_steps=args.draft_denoise_steps,
                return_verify_trace=args.save_verify_trace,
                verify_trace_max_rounds=args.verify_trace_max_rounds,
                verify_confidence_threshold=args.verify_confidence_threshold,
                verify_min_draft_tokens=args.verify_min_draft_tokens,
                draft_dynamic_yarn_original_max_position_embeddings=(
                    args.draft_dynamic_yarn_original_max_position_embeddings
                ),
                draft_dynamic_yarn_max_factor=args.draft_dynamic_yarn_max_factor,
                draft_dynamic_yarn_mode=args.draft_dynamic_yarn_mode,
                draft_dynamic_yarn_length_ratio=args.draft_dynamic_yarn_length_ratio,
                profiler=args.profiler,
                suffix_decoding=args.suffix_decoding,
                suffix_strategy=args.suffix_strategy,
                suffix_max_query_len=args.suffix_max_query_len,
                suffix_min_query_len=args.suffix_min_query_len,
                suffix_top_k=args.suffix_top_k,
                suffix_min_support=args.suffix_min_support,
                suffix_min_predict_len=args.suffix_min_predict_len,
                suffix_max_predict_len=args.suffix_max_predict_len,
                suffix_paper_alpha=args.suffix_paper_alpha,
                suffix_paper_max_spec_offset=args.suffix_paper_max_spec_offset,
                suffix_paper_min_token_prob=args.suffix_paper_min_token_prob,
                suffix_paper_threshold=args.suffix_paper_threshold,
                suffix_paper_max_matches=args.suffix_paper_max_matches,
                suffix_paper_verifier=args.suffix_paper_verifier,
                suffix_paper_tree_attn_impl=args.suffix_paper_tree_attn_impl,
                suffix_fallback=args.suffix_fallback,
                return_suffix_trace=args.save_suffix_trace,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

            acceptance = compute_acceptance_metrics([int(x) for x in dflash.acceptance_lengths], block_size)
            baseline_tpot = float(baseline.time_per_output_token)
            dflash_tpot = float(dflash.time_per_output_token)
            record.update(
                status="ok",
                baseline_output_tokens=int(baseline.num_output_tokens),
                dflash_output_tokens=int(dflash.num_output_tokens),
                baseline_tpot=baseline_tpot,
                dflash_tpot=dflash_tpot,
                speedup=baseline_tpot / dflash_tpot if dflash_tpot > 0 else None,
                time_to_first_token=float(dflash.time_to_first_token),
                acceptance_lengths=[int(x) for x in dflash.acceptance_lengths],
                verify_draft_lengths=[int(x) for x in dflash.verify_draft_lengths],
                mean_verify_draft_tokens=(
                    sum(int(x) for x in dflash.verify_draft_lengths) / len(dflash.verify_draft_lengths)
                    if dflash.verify_draft_lengths
                    else None
                ),
                draft_denoise_steps=int(dflash.draft_denoise_steps),
                draft_forward_passes=int(dflash.draft_forward_passes),
                draft_dynamic_yarn_factor=dflash.draft_yarn_factor,
                suffix_decoding_enabled=bool(dflash.suffix_decoding_enabled),
                suffix_strategy=dflash.suffix_strategy,
                suffix_fallback=dflash.suffix_fallback,
                suffix_match_rounds=int(dflash.suffix_match_rounds),
                suffix_verify_rounds=int(dflash.suffix_verify_rounds),
                suffix_recovery_rounds=int(dflash.suffix_recovery_rounds),
                suffix_zero_accept_rounds=int(
                    getattr(dflash, "suffix_zero_accept_rounds", dflash.suffix_recovery_rounds)
                ),
                suffix_exhausted_rounds=int(getattr(dflash, "suffix_exhausted_rounds", 0)),
                suffix_pred_lengths=[int(x) for x in dflash.suffix_pred_lengths],
                suffix_supports=[int(x) for x in dflash.suffix_supports],
                suffix_query_lengths=[int(x) for x in dflash.suffix_query_lengths],
                suffix_acceptance_lengths=[int(x) for x in dflash.suffix_acceptance_lengths],
                suffix_paper_alpha=float(dflash.suffix_paper_alpha),
                suffix_paper_max_spec_offset=float(dflash.suffix_paper_max_spec_offset),
                suffix_paper_min_token_prob=float(dflash.suffix_paper_min_token_prob),
                suffix_paper_threshold=float(dflash.suffix_paper_threshold),
                suffix_paper_max_matches=int(dflash.suffix_paper_max_matches),
                suffix_paper_verifier=dflash.suffix_paper_verifier,
                suffix_paper_tree_attn_impl=dflash.suffix_paper_tree_attn_impl,
                suffix_paper_scores=[float(x) for x in dflash.suffix_paper_scores],
                suffix_paper_token_scores=[float(x) for x in dflash.suffix_paper_token_scores],
                suffix_paper_tree_sizes=[int(x) for x in dflash.suffix_paper_tree_sizes],
                suffix_paper_best_path_scores=[float(x) for x in dflash.suffix_paper_best_path_scores],
                suffix_paper_max_specs=[int(x) for x in dflash.suffix_paper_max_specs],
                **acceptance,
            )
            if args.save_suffix_trace and dflash.suffix_trace is not None:
                record["suffix_trace"] = dflash.suffix_trace
            if dflash.profiler_stats is not None:
                record.update(dflash.profiler_stats)
            if dflash.ctx_prune_stats is not None:
                record.update(dflash.ctx_prune_stats)
            if args.save_verify_trace and dflash.verify_trace is not None:
                for trace_round in dflash.verify_trace:
                    verify_trace_records.append(
                        {
                            "sample_index": sample_index,
                            "sample_id": sample_id,
                            "rank": _dist_rank(),
                            **trace_round,
                        }
                    )
            if not args.no_save_responses:
                baseline_path, dflash_path = _write_response_texts(
                    responses_dir,
                    sample_index,
                    sample_id,
                    tokenizer,
                    baseline,
                    dflash,
                )
                record["baseline_response_path"] = baseline_path
                record["dflash_response_path"] = dflash_path
            records.append(record)
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            record.update(status="oom", error=str(exc))
            if input_ids is not None:
                record.setdefault("prompt_tokens", int(input_ids.shape[1]))
            records.append(record)
        except Exception as exc:  # Keep the benchmark moving and preserve per-sample failures.
            record.update(status="error", error=repr(exc))
            if input_ids is not None:
                record.setdefault("prompt_tokens", int(input_ids.shape[1]))
            records.append(record)

    gathered = _dist_gather(torch_dist, records)
    gathered_verify_trace = _dist_gather(torch_dist, verify_trace_records)
    if not _dist_is_main():
        return {"output_dir": str(out_dir), "summary": None}

    merged_records = [record for rank_records in gathered for record in rank_records]
    merged_records.sort(key=lambda row: int(row["sample_index"]))
    summary = write_benchmark_outputs(out_dir, merged_records)
    if args.profiler:
        profiler_path = _write_profiler_markdown(out_dir, merged_records, summary)
        logger.info("Profiler report: {}", profiler_path)
    if args.save_verify_trace:
        merged_verify_trace = [record for rank_records in gathered_verify_trace for record in rank_records]
        merged_verify_trace.sort(key=lambda row: (int(row["sample_index"]), int(row["round_idx"])))
        verify_trace_path = out_dir / "verify_trace.jsonl"
        with verify_trace_path.open("w", encoding="utf-8") as handle:
            for trace_record in merged_verify_trace:
                handle.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
        logger.info("Verify trace: {}", verify_trace_path)
    logger.info("Records: {}", out_dir / "records.jsonl")
    logger.info("Per-sample CSV: {}", out_dir / "per_sample.csv")
    logger.info("Summary CSV: {}", out_dir / "summary.csv")
    logger.info("Summary JSON: {}", out_dir / "summary.json")
    if summary.get("total_ctx_suffix_match_count") is not None:
        logger.info(
            "Suffix matches: total={}, mean={}, max={}",
            summary.get("total_ctx_suffix_match_count"),
            summary.get("mean_ctx_suffix_match_count"),
            summary.get("max_ctx_suffix_match_count"),
        )
    if summary.get("mean_ctx_middle_budget_dropped_tokens") is not None:
        logger.info(
            "Middle budget: mean_before={}, mean_after={}, mean_dropped={}",
            summary.get("mean_ctx_middle_tokens_before_budget"),
            summary.get("mean_ctx_middle_tokens_after_budget"),
            summary.get("mean_ctx_middle_budget_dropped_tokens"),
        )
    return {"output_dir": str(out_dir), "summary": summary}


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
