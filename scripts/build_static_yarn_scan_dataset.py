#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bucket_files(bucket_dir: Path) -> list[Path]:
    files = sorted(bucket_dir.glob("bucket_*.jsonl"), key=lambda path: path.name)
    if not files:
        raise SystemExit(f"no bucket_*.jsonl files found in {bucket_dir}")
    return files


def _parse_exclude_sources(values: list[str]) -> set[tuple[str, str, int]]:
    excludes: set[tuple[str, str, int]] = set()
    for value in values:
        try:
            dataset, bucket, source_index = value.split(":", 2)
            excludes.add((dataset, bucket, int(source_index)))
        except ValueError as exc:
            raise SystemExit(
                "--exclude-source entries must be formatted as dataset:bucket:source_index"
            ) from exc
    return excludes


def _parse_replace_sources(values: list[str]) -> list[tuple[str, str, int, int]]:
    replacements: list[tuple[str, str, int, int]] = []
    for value in values:
        try:
            dataset, bucket, old_index, new_index = value.split(":", 3)
            replacements.append((dataset, bucket, int(old_index), int(new_index)))
        except ValueError as exc:
            raise SystemExit(
                "--replace-source entries must be formatted as dataset:bucket:old_index:new_index"
            ) from exc
    return replacements


def _source_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for dataset, bucket_dir in [
        ("terminal", args.terminal_bucket_dir),
        ("swebench", args.swebench_bucket_dir),
    ]:
        for path in _bucket_files(bucket_dir):
            specs.append(
                {
                    "dataset": dataset,
                    "bucket": path.stem.removeprefix("bucket_"),
                    "source_file": str(path),
                }
            )
    return specs


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    exclude_sources = _parse_exclude_sources(args.exclude_source)
    replace_sources = _parse_replace_sources(args.replace_source)

    output_rows: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    for source in _source_specs(args):
        source_path = Path(source["source_file"])
        rows = _load_jsonl(source_path)
        indexed_rows = list(enumerate(rows))
        rng.shuffle(indexed_rows)

        source_excludes = {
            source_index
            for dataset, bucket, source_index in exclude_sources
            if dataset == source["dataset"] and bucket == source["bucket"]
        }
        indexed_rows = [
            (source_index, row)
            for source_index, row in indexed_rows
            if source_index not in source_excludes
        ]
        take = min(args.samples_per_bucket, len(indexed_rows))
        picked_by_index = dict(sorted(indexed_rows[:take], key=lambda item: item[0]))

        source_replacements: list[dict[str, int]] = []
        for dataset, bucket, old_index, new_index in replace_sources:
            if dataset != source["dataset"] or bucket != source["bucket"]:
                continue
            if old_index not in picked_by_index:
                raise SystemExit(
                    f"Cannot replace unselected source {dataset}:{bucket}:{old_index}"
                )
            if new_index in picked_by_index:
                raise SystemExit(
                    f"Cannot replace {dataset}:{bucket}:{old_index} with already selected {new_index}"
                )
            if new_index < 0 or new_index >= len(rows):
                raise SystemExit(
                    f"Replacement source index out of range: {dataset}:{bucket}:{new_index}"
                )
            del picked_by_index[old_index]
            picked_by_index[new_index] = rows[new_index]
            source_replacements.append({"old_index": old_index, "new_index": new_index})

        picked = sorted(picked_by_index.items(), key=lambda item: item[0])
        manifest_sources.append(
            {
                **source,
                "available": len(rows),
                "excluded_indices": sorted(source_excludes),
                "replaced_indices": source_replacements,
                "selected": len(picked),
                "selected_indices": [idx for idx, _ in picked],
            }
        )
        for source_index, row in picked:
            annotated = dict(row)
            annotated["_static_yarn_scan_source"] = {
                "dataset": source["dataset"],
                "bucket": source["bucket"],
                "source_index": source_index,
                "source_file": str(source_path),
            }
            output_rows.append(annotated)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "terminal_bucket_dir": str(args.terminal_bucket_dir),
        "swebench_bucket_dir": str(args.swebench_bucket_dir),
        "seed": args.seed,
        "samples_per_bucket": args.samples_per_bucket,
        "output_jsonl": str(args.output_jsonl),
        "total_selected": len(output_rows),
        "sources": manifest_sources,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified terminal/swebench sample set for static YaRN scans."
    )
    parser.add_argument(
        "--terminal-bucket-dir",
        type=Path,
        default=ROOT_DIR / "benchmarks" / "terminal",
    )
    parser.add_argument(
        "--swebench-bucket-dir",
        type=Path,
        default=ROOT_DIR / "benchmarks" / "swebench",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Exclude a source row formatted as dataset:bucket:source_index. May be repeated.",
    )
    parser.add_argument(
        "--replace-source",
        action="append",
        default=[],
        help="Replace a selected source row formatted as dataset:bucket:old_index:new_index. May be repeated.",
    )
    args = parser.parse_args()
    if args.samples_per_bucket <= 0:
        raise SystemExit("--samples-per-bucket must be positive")
    return args


def main() -> None:
    manifest = build_dataset(parse_args())
    print(f"wrote_dataset={manifest['output_jsonl']}")
    print(f"total_selected={manifest['total_selected']}")
    for source in manifest["sources"]:
        print(
            "source="
            f"{source['dataset']}:{source['bucket']} "
            f"selected={source['selected']}/{source['available']} "
            f"indices={source['selected_indices']}"
        )


if __name__ == "__main__":
    main()
