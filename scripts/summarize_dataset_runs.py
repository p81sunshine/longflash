#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


BUCKET_RUN_RE = re.compile(
    r"^(?P<draft>original|trained)_(?P<variant>.+)_(?P<dataset>terminal|swebench)_"
    r"(?P<start>\d+)_(?P<end>\d+)_ms(?P<max_samples>\d+)_mn(?P<max_new_tokens>\d+)$"
)


def _variant_label(run_variant: str) -> str:
    variant_key = re.sub(r"^(terminal|swebench)_", "", run_variant)
    return variant_key


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))


def _round(value: Any, digits: int = 6) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _run_name_from_dir(run_dir: Path) -> str:
    parts = run_dir.name.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]
    return run_dir.name


def _parse_run_name(run_name: str) -> dict[str, Any]:
    match = BUCKET_RUN_RE.match(run_name)
    if match:
        groups = match.groupdict()
        start = int(groups["start"])
        end = int(groups["end"])
        return {
            "draft": groups["draft"],
            "run_variant": groups["variant"],
            "dataset_label": groups["dataset"],
            "bucket": f"{start}_{end}",
            "bucket_start": start,
            "bucket_end": end,
            "max_samples_from_name": int(groups["max_samples"]),
            "max_new_tokens_from_name": int(groups["max_new_tokens"]),
        }

    match = re.match(r"^(?P<draft>original|trained)_(?P<variant>.+)_(terminal|swebench)_", run_name)
    if match:
        groups = match.groupdict()
        return {
            "draft": groups["draft"],
            "run_variant": groups["variant"],
            "dataset_label": "",
            "bucket": "",
            "bucket_start": "",
            "bucket_end": "",
            "max_samples_from_name": "",
            "max_new_tokens_from_name": "",
        }

    return {
        "draft": "",
        "run_variant": "",
        "dataset_label": "",
        "bucket": "",
        "bucket_start": "",
        "bucket_end": "",
        "max_samples_from_name": "",
        "max_new_tokens_from_name": "",
    }


def _dataset_group(dataset: str) -> str:
    if "benchmarks/terminal/" in dataset or "terminal_bench" in dataset:
        return "terminal"
    if "benchmarks/swebench/" in dataset:
        return "swebench"
    return ""


def _fill_bucket_from_dataset(parsed: dict[str, Any], dataset: str) -> None:
    if parsed.get("bucket"):
        return
    match = re.search(r"bucket_(\d+)_(\d+)\.jsonl$", dataset)
    if not match:
        return
    start = int(match.group(1))
    end = int(match.group(2))
    parsed["bucket"] = f"{start}_{end}"
    parsed["bucket_start"] = start
    parsed["bucket_end"] = end


def _infer_run_variant(config: dict[str, Any], run_name: str) -> str:
    suffix_decoding = bool(config.get("suffix_decoding"))
    dynamic_original = config.get("draft_dynamic_yarn_original_max_position_embeddings")

    if dynamic_original:
        if suffix_decoding:
            return "dynamic_yarn_suffix"
        return "dynamic_yarn_continuous_uncapped"

    if suffix_decoding:
        return "suffix"
    return "original_dflash" if "original_dflash" in run_name else ""


def _rope_factor(config: dict[str, Any], key: str) -> Any:
    rope = config.get(key) or {}
    if rope.get("rope_type") != "yarn":
        return ""
    return _round(rope.get("factor"))


def _rope_original(config: dict[str, Any], key: str) -> Any:
    rope = config.get(key) or {}
    if rope.get("rope_type") != "yarn":
        return ""
    return rope.get("original_max_position_embeddings", "")


def _row_from_run_dir(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    config = _load_json(run_dir / "run_config.json")
    run_name = _run_name_from_dir(run_dir)
    parsed = _parse_run_name(run_name)
    dataset = str(config.get("dataset", ""))
    _fill_bucket_from_dataset(parsed, dataset)

    run_variant = parsed["run_variant"]
    if not run_variant:
        run_variant = _infer_run_variant(config, run_name)
    draft = parsed["draft"] or ("original" if run_name.startswith("original_") else "")
    return {
        "dataset_group": _dataset_group(dataset),
        "dataset_label": parsed["dataset_label"],
        "draft": draft,
        "variant": _variant_label(run_variant),
        "run_variant": run_variant,
        "bucket": parsed["bucket"],
        "bucket_start": parsed["bucket_start"],
        "bucket_end": parsed["bucket_end"],
        "run_dir": run_dir.name,
        "samples_total": summary.get("samples_total", ""),
        "samples_ok": summary.get("samples_ok", ""),
        "samples_skipped": summary.get("samples_skipped", ""),
        "samples_oom": summary.get("samples_oom", ""),
        "samples_error": summary.get("samples_error", ""),
        "mean_prompt_tokens": _round(summary.get("mean_prompt_tokens"), 2),
        "mean_speedup": _round(summary.get("mean_speedup")),
        "mean_acceptance_length": _round(summary.get("mean_acceptance_length")),
        "mean_acceptance_rate": _round(summary.get("mean_acceptance_rate")),
        "throughput_baseline_tok_s": _round(summary.get("throughput_baseline_tok_s")),
        "throughput_dflash_tok_s": _round(summary.get("throughput_dflash_tok_s")),
        "mean_draft_forward_passes": _round(summary.get("mean_draft_forward_passes")),
        "mean_draft_dynamic_yarn_factor": _round(summary.get("mean_draft_dynamic_yarn_factor")),
        "min_draft_dynamic_yarn_factor": _round(summary.get("min_draft_dynamic_yarn_factor")),
        "max_draft_dynamic_yarn_factor": _round(summary.get("max_draft_dynamic_yarn_factor")),
        "mean_verify_draft_tokens": _round(summary.get("mean_verify_draft_tokens")),
        "mean_suffix_match_rounds": _round(summary.get("mean_suffix_match_rounds")),
        "total_suffix_match_rounds": summary.get("total_suffix_match_rounds", ""),
        "mean_suffix_verify_rounds": _round(summary.get("mean_suffix_verify_rounds")),
        "total_suffix_verify_rounds": summary.get("total_suffix_verify_rounds", ""),
        "mean_suffix_recovery_rounds": _round(summary.get("mean_suffix_recovery_rounds")),
        "total_suffix_recovery_rounds": summary.get("total_suffix_recovery_rounds", ""),
        "mean_suffix_exhausted_rounds": _round(summary.get("mean_suffix_exhausted_rounds")),
        "total_suffix_exhausted_rounds": summary.get("total_suffix_exhausted_rounds", ""),
        "mean_suffix_paper_score": _round(summary.get("mean_suffix_paper_score")),
        "mean_suffix_paper_token_score": _round(summary.get("mean_suffix_paper_token_score")),
        "mean_suffix_paper_tree_size": _round(summary.get("mean_suffix_paper_tree_size")),
        "mean_suffix_paper_best_path_score": _round(summary.get("mean_suffix_paper_best_path_score")),
        "mean_suffix_paper_max_spec": _round(summary.get("mean_suffix_paper_max_spec")),
        "mean_ctx_suffix_match_count": _round(summary.get("mean_ctx_suffix_match_count")),
        "max_ctx_suffix_match_count": summary.get("max_ctx_suffix_match_count", ""),
        "total_ctx_suffix_match_count": summary.get("total_ctx_suffix_match_count", ""),
        "mean_ctx_suffix_match_kept_tokens": _round(summary.get("mean_ctx_suffix_match_kept_tokens"), 2),
        "mean_ctx_middle_tokens_before_budget": _round(summary.get("mean_ctx_middle_tokens_before_budget"), 2),
        "mean_ctx_middle_tokens_after_budget": _round(summary.get("mean_ctx_middle_tokens_after_budget"), 2),
        "mean_ctx_middle_budget_dropped_tokens": _round(summary.get("mean_ctx_middle_budget_dropped_tokens"), 2),
        "mean_ctx_total_budget": _round(summary.get("mean_ctx_total_budget"), 2),
        "mean_ctx_recent_tokens_after_budget": _round(summary.get("mean_ctx_recent_tokens_after_budget"), 2),
        "mean_ctx_hidden_tokens_after": _round(summary.get("mean_ctx_hidden_tokens_after"), 2),
        "model": config.get("model", ""),
        "draft_model": config.get("draft_model", ""),
        "dataset": dataset,
        "sample_count": config.get("sample_count", ""),
        "world_size": config.get("world_size", ""),
        "max_samples": config.get("max_samples", parsed["max_samples_from_name"]),
        "max_new_tokens": config.get("max_new_tokens", parsed["max_new_tokens_from_name"]),
        "temperature": config.get("temperature", ""),
        "sample_seed": config.get("sample_seed", ""),
        "block_size": config.get("block_size", ""),
        "ctx_sink_tokens": config.get("ctx_sink_tokens", ""),
        "ctx_recent_window": config.get("ctx_recent_window", ""),
        "ctx_stride": config.get("ctx_stride", ""),
        "ctx_suffix_match_tokens": config.get("ctx_suffix_match_tokens", ""),
        "ctx_suffix_keep_tokens": config.get("ctx_suffix_keep_tokens", ""),
        "ctx_middle_budget": config.get("ctx_middle_budget", ""),
        "ctx_total_budget": config.get("ctx_total_budget", ""),
        "ctx_dynamic_budget_ratio": config.get("ctx_dynamic_budget_ratio", ""),
        "ctx_budget_order": config.get("ctx_budget_order", ""),
        "draft_denoise_steps": config.get("draft_denoise_steps", ""),
        "save_verify_trace": config.get("save_verify_trace", ""),
        "verify_trace_max_rounds": config.get("verify_trace_max_rounds", ""),
        "verify_confidence_threshold": config.get("verify_confidence_threshold", ""),
        "verify_min_draft_tokens": config.get("verify_min_draft_tokens", ""),
        "suffix_decoding": config.get("suffix_decoding", ""),
        "suffix_strategy": config.get("suffix_strategy", ""),
        "suffix_fallback": config.get("suffix_fallback", ""),
        "suffix_max_query_len": config.get("suffix_max_query_len", ""),
        "suffix_min_query_len": config.get("suffix_min_query_len", ""),
        "suffix_top_k": config.get("suffix_top_k", ""),
        "suffix_min_support": config.get("suffix_min_support", ""),
        "suffix_min_predict_len": config.get("suffix_min_predict_len", ""),
        "suffix_max_predict_len": config.get("suffix_max_predict_len", ""),
        "suffix_paper_alpha": config.get("suffix_paper_alpha", ""),
        "suffix_paper_max_spec_offset": config.get("suffix_paper_max_spec_offset", ""),
        "suffix_paper_min_token_prob": config.get("suffix_paper_min_token_prob", ""),
        "suffix_paper_threshold": config.get("suffix_paper_threshold", ""),
        "suffix_paper_max_matches": config.get("suffix_paper_max_matches", ""),
        "suffix_paper_verifier": config.get("suffix_paper_verifier", ""),
        "suffix_paper_tree_attn_impl": config.get("suffix_paper_tree_attn_impl", ""),
        "draft_dynamic_yarn_original": config.get("draft_dynamic_yarn_original_max_position_embeddings", ""),
        "draft_dynamic_yarn_max_factor": config.get("draft_dynamic_yarn_max_factor", ""),
        "draft_dynamic_yarn_mode": config.get("draft_dynamic_yarn_mode", ""),
        "draft_dynamic_yarn_length_ratio": config.get("draft_dynamic_yarn_length_ratio", ""),
        "draft_sliding_window_size": config.get("draft_sliding_window_size", ""),
        "draft_sliding_window": config.get("draft_sliding_window", ""),
        "target_yarn_factor": _rope_factor(config, "target_rope_parameters"),
        "target_yarn_original": _rope_original(config, "target_rope_parameters"),
        "draft_yarn_factor": _rope_factor(config, "draft_rope_parameters"),
        "draft_yarn_original": _rope_original(config, "draft_rope_parameters"),
    }


def collect_rows(run_dir: Path, *, latest_only: bool = False) -> list[dict[str, Any]]:
    rows = []
    latest_by_run_name: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        row = _row_from_run_dir(summary_path.parent)
        if row is not None:
            if latest_only:
                latest_by_run_name[_run_name_from_dir(summary_path.parent)] = row
            else:
                rows.append(row)
    if latest_only:
        rows = list(latest_by_run_name.values())
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset_group"]),
            str(row["variant"]),
            int(row["bucket_start"] or -1),
            str(row["run_dir"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", "utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize bucketed terminal/swebench benchmark runs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--latest-only", action="store_true", help="Keep only the newest directory for each run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.run_dir.expanduser().resolve(), latest_only=args.latest_only)
    write_csv(args.output_csv.expanduser().resolve(), rows)
    print(f"rows={len(rows)}")
    print(f"csv={args.output_csv.expanduser().resolve()}")


if __name__ == "__main__":
    main()
