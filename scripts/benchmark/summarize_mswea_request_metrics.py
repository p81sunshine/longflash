#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize mini-swe-agent per-request metrics.")
    parser.add_argument("input", type=Path, help="Path to llm_requests.jsonl")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _merged_busy_time(rows: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (
            float(row["request_start"]),
            float(row["request_end"]),
        )
        for row in rows
        if row.get("request_start") is not None and row.get("request_end") is not None
    )
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["latency_sec"]) for row in rows if row.get("latency_sec") is not None)
    total_prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows if row.get("prompt_tokens") is not None)
    total_completion_tokens = sum(
        int(row["completion_tokens"]) for row in rows if row.get("completion_tokens") is not None
    )
    total_tokens = sum(int(row["total_tokens"]) for row in rows if row.get("total_tokens") is not None)
    total_reasoning_tokens = sum(
        int(row["reasoning_tokens"]) for row in rows if row.get("reasoning_tokens") is not None
    )
    total_request_latency = sum(latencies)
    busy_time_sec = _merged_busy_time(rows)
    wall_time_sec = 0.0
    if rows:
        starts = [float(row["request_start"]) for row in rows if row.get("request_start") is not None]
        ends = [float(row["request_end"]) for row in rows if row.get("request_end") is not None]
        if starts and ends:
            wall_time_sec = max(ends) - min(starts)

    by_instance = {row.get("instance_id") for row in rows if row.get("instance_id")}
    summary = {
        "request_count": len(rows),
        "instance_count": len(by_instance),
        "sum_prompt_tokens": total_prompt_tokens,
        "sum_completion_tokens": total_completion_tokens,
        "sum_total_tokens": total_tokens,
        "sum_reasoning_tokens": total_reasoning_tokens,
        "sum_request_latency_sec": total_request_latency,
        "busy_time_sec": busy_time_sec,
        "wall_time_sec": wall_time_sec,
        "avg_latency_sec": statistics.fmean(latencies) if latencies else None,
        "p50_latency_sec": _quantile(latencies, 0.50),
        "p95_latency_sec": _quantile(latencies, 0.95),
        "request_avg_output_tps": (
            total_completion_tokens / total_request_latency if total_request_latency > 0 else None
        ),
        "request_avg_total_tps": total_tokens / total_request_latency if total_request_latency > 0 else None,
        "server_busy_output_tps": total_completion_tokens / busy_time_sec if busy_time_sec > 0 else None,
        "server_busy_total_tps": total_tokens / busy_time_sec if busy_time_sec > 0 else None,
        "wall_output_tps": total_completion_tokens / wall_time_sec if wall_time_sec > 0 else None,
        "wall_total_tps": total_tokens / wall_time_sec if wall_time_sec > 0 else None,
    }
    return summary


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.input)
    summary = summarize(rows)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
