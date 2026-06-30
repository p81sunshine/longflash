#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def _resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    artifact_path = ARTIFACT_ROOT / path
    if artifact_path.exists():
        return artifact_path
    return path


def _rope_value(config: dict[str, Any], key: str) -> Any:
    rope = config.get("draft_rope_parameters") or {}
    return rope.get(key, "")


def _draft_yarn_max(config: dict[str, Any]) -> Any:
    rope = config.get("draft_rope_parameters") or {}
    original = rope.get("original_max_position_embeddings")
    factor = rope.get("factor")
    if original not in (None, "") and factor not in (None, ""):
        return int(float(original) * float(factor))
    return ""


def _row_quality_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    samples_ok = int(row.get("samples_ok") or 0)
    samples_oom = int(row.get("samples_oom") or 0)
    samples_error = int(row.get("samples_error") or 0)
    return (samples_ok, -samples_oom, -samples_error, str(row.get("run_dir") or ""))


def _select_best_config_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["draft_yarn_original"] or 0), float(row["draft_yarn_factor"] or 0))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for candidates in grouped.values():
        best = max(candidates, key=_row_quality_key)
        best = dict(best)
        best["duplicate_candidates"] = len(candidates)
        best["superseded_run_dirs"] = ";".join(
            sorted(str(row["run_dir"]) for row in candidates if row["run_dir"] != best["run_dir"])
        )
        selected.append(best)
    return selected


def collect_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        config_path = summary_path.parent / "run_config.json"
        if not config_path.exists():
            continue
        summary = _load_json(summary_path)
        config = _load_json(config_path)
        rows.append(
            {
                "run_dir": summary_path.parent.name,
                "draft_yarn_original": _rope_value(config, "original_max_position_embeddings"),
                "draft_yarn_factor": _rope_value(config, "factor"),
                "draft_yarn_max_position_embeddings": _draft_yarn_max(config),
                "target_yarn_original": (config.get("target_rope_parameters") or {}).get(
                    "original_max_position_embeddings",
                    "",
                ),
                "target_yarn_factor": (config.get("target_rope_parameters") or {}).get("factor", ""),
                "samples_total": summary.get("samples_total", ""),
                "samples_ok": summary.get("samples_ok", ""),
                "samples_oom": summary.get("samples_oom", ""),
                "samples_error": summary.get("samples_error", ""),
                "mean_prompt_tokens": summary.get("mean_prompt_tokens", ""),
                "mean_acceptance_length": summary.get("mean_acceptance_length", ""),
                "mean_acceptance_rate": summary.get("mean_acceptance_rate", ""),
                "mean_speedup": summary.get("mean_speedup", ""),
                "throughput_baseline_tok_s": summary.get("throughput_baseline_tok_s", ""),
                "throughput_dflash_tok_s": summary.get("throughput_dflash_tok_s", ""),
            }
        )
    return sorted(
        _select_best_config_rows(rows),
        key=lambda row: (
            float(row["draft_yarn_original"] or 0),
            float(row["draft_yarn_factor"] or 0),
        ),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def collect_bucket_rows(run_dir: Path, selected_run_dirs: set[str] | None = None) -> list[dict[str, Any]]:
    bucket_rows: list[dict[str, Any]] = []
    for records_path in sorted(run_dir.glob("*/records.jsonl")):
        if selected_run_dirs is not None and records_path.parent.name not in selected_run_dirs:
            continue
        config_path = records_path.parent / "run_config.json"
        if not config_path.exists():
            continue
        config = _load_json(config_path)
        dataset_path = _resolve_path(Path(config["dataset"]))
        dataset_rows = _load_jsonl(dataset_path)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in _load_jsonl(records_path):
            if record.get("status") != "ok":
                continue
            sample_index = int(record["sample_index"])
            source = dataset_rows[sample_index].get("_static_yarn_scan_source", {})
            key = (str(source.get("dataset", "")), str(source.get("bucket", "")))
            grouped.setdefault(key, []).append(record)
        for (dataset, bucket), records in grouped.items():
            bucket_rows.append(
                {
                    "run_dir": records_path.parent.name,
                    "draft_yarn_original": _rope_value(config, "original_max_position_embeddings"),
                    "draft_yarn_factor": _rope_value(config, "factor"),
                    "draft_yarn_max_position_embeddings": _draft_yarn_max(config),
                    "dataset": dataset,
                    "bucket": bucket,
                    "samples_ok": len(records),
                    "mean_prompt_tokens": _mean([float(row["prompt_tokens"]) for row in records]),
                    "mean_acceptance_length": _mean([float(row["mean_acceptance_length"]) for row in records]),
                    "mean_acceptance_rate": _mean([float(row["acceptance_rate"]) for row in records]),
                    "mean_speedup": _mean([float(row["speedup"]) for row in records]),
                }
            )
    return sorted(
        bucket_rows,
        key=lambda row: (
            float(row["draft_yarn_original"] or 0),
            float(row["draft_yarn_factor"] or 0),
            str(row["dataset"]),
            str(row["bucket"]),
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


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", "utf-8")
        return
    cols = [
        "draft_yarn_original",
        "draft_yarn_factor",
        "samples_ok",
        "mean_acceptance_length",
        "mean_acceptance_rate",
        "mean_speedup",
    ]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        rendered = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                rendered.append(f"{value:.6g}")
            else:
                rendered.append(str(value))
        lines.append("|" + "|".join(rendered) + "|")
    path.write_text("\n".join(lines) + "\n", "utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen3-8B static YaRN acceptance scan runs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-bucket-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or args.run_dir / "static_yarn_scan_summary.csv"
    output_md = args.output_md or args.run_dir / "static_yarn_scan_summary.md"
    output_bucket_csv = args.output_bucket_csv or args.run_dir / "static_yarn_scan_bucket_summary.csv"
    rows = collect_rows(args.run_dir)
    selected_run_dirs = {str(row["run_dir"]) for row in rows}
    bucket_rows = collect_bucket_rows(args.run_dir, selected_run_dirs)
    write_csv(output_csv, rows)
    write_csv(output_bucket_csv, bucket_rows)
    write_markdown(output_md, rows)
    print(f"summary_csv={output_csv}")
    print(f"summary_md={output_md}")
    print(f"bucket_summary_csv={output_bucket_csv}")
    for row in rows:
        print(
            "result="
            f"original={row['draft_yarn_original']} "
            f"factor={row['draft_yarn_factor']} "
            f"ok={row['samples_ok']} "
            f"accept_len={row['mean_acceptance_length']} "
            f"speedup={row['mean_speedup']}"
        )


if __name__ == "__main__":
    main()
