import json
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark.run_agentic_memory_benchmark import (
    Experiment,
    _write_global_metric_row,
    write_summary,
)


def test_target_only_sliding_uses_global_wall_time(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    per_sample_path = tmp_path / "per_sample.csv"
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"
    rows = [
        {
            "experiment": "target-only",
            "window": "full",
            "window_mode": "target_only",
            "metric_scope": "request",
            "sample_id": "a",
            "elapsed_sec": 10.0,
            "completion_tps": 10.0,
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 200,
            "acceptance_rate": None,
            "acceptance_length": None,
        },
        {
            "experiment": "target-only",
            "window": "full",
            "window_mode": "target_only",
            "metric_scope": "request",
            "sample_id": "b",
            "elapsed_sec": 10.0,
            "completion_tps": 10.0,
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 200,
            "acceptance_rate": None,
            "acceptance_length": None,
        },
    ]
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        _write_global_metric_row(
            handle,
            Experiment(name="target-only", window="full", target_only=True),
            SimpleNamespace(concurrency=2, concurrency_scheduler="sliding"),
            None,
            wall_time_sec=11.0,
        )

    write_summary(records_path, per_sample_path, csv_path, json_path)

    [summary] = json.loads(json_path.read_text("utf-8"))["by_experiment"]
    assert summary["global_wall_time_sec"] == 11.0
    assert summary["total_elapsed_sec"] == 11.0
    assert abs(summary["total_completion_tps"] - (200.0 / 11.0)) < 1e-9
