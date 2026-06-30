#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
EAGLE_DIR = Path(os.environ.get("EAGLE_ROOT", ROOT_DIR / "third_party" / "EAGLE")).expanduser()
sys.path.insert(0, str(EAGLE_DIR))
sys.path.insert(0, str(ROOT_DIR))

from eagle.model.ea_model import EaModel  # noqa: E402
from eagle.model.kv_cache import initialize_past_key_values  # noqa: E402


def _dist_init() -> None:
    if "RANK" in os.environ and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")


def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(obj: Any) -> list[Any] | None:
    if not torch.distributed.is_initialized():
        return [obj]
    if _dist_is_main():
        gathered = [None for _ in range(_dist_size())]
        torch.distributed.gather_object(obj, gathered, dst=0)
        return gathered
    torch.distributed.gather_object(obj, dst=0)
    return None


def load_dataset(path: str) -> list[dict[str, Any]]:
    dataset_path = Path(path).expanduser()
    with dataset_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def limit_dataset(dataset: list[dict[str, Any]], max_samples: int | None, *, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or len(dataset) <= max_samples:
        return list(dataset)
    selected = list(dataset)
    random.Random(seed).shuffle(selected)
    return selected[:max_samples]


def shard_dataset(dataset: list[dict[str, Any]], *, rank: int, world_size: int) -> list[tuple[int, dict[str, Any]]]:
    return [(idx, dataset[idx]) for idx in range(rank, len(dataset), world_size)]


def load_sample_indices(path: str | None) -> set[int] | None:
    if not path:
        return None
    indices: set[int] = set()
    indices_path = Path(path).expanduser()
    with indices_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                if "sample_index" not in parsed:
                    raise ValueError(f"{indices_path}:{line_number}: JSON object is missing sample_index")
                indices.add(int(parsed["sample_index"]))
                continue
            if isinstance(parsed, int):
                indices.add(parsed)
                continue
            for part in line.replace(",", " ").split():
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    start = int(start_s)
                    end = int(end_s)
                    if end < start:
                        raise ValueError(f"{indices_path}:{line_number}: invalid descending range {part!r}")
                    indices.update(range(start, end + 1))
                else:
                    indices.add(int(part))
    return indices


def filter_samples(
    samples: list[tuple[int, dict[str, Any]]],
    sample_indices: set[int] | None,
) -> list[tuple[int, dict[str, Any]]]:
    if sample_indices is None:
        return samples
    return [(idx, sample) for idx, sample in samples if idx in sample_indices]


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_prompt(
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


def sample_id(sample: dict[str, Any], sample_index: int) -> str:
    return str(sample.get("sample_id") or sample.get("instance_id") or sample.get("id") or f"sample_{sample_index}")


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return safe[:120] or "sample"


def make_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{safe_filename(args.run_name)}" if args.run_name else ""
    return Path(args.output_dir).expanduser() / f"{timestamp}{suffix}"


def write_response_texts(
    responses_dir: Path,
    sample_index: int,
    sid: str,
    tokenizer: Any,
    input_len: int,
    baseline_ids: torch.Tensor,
    eagle_ids: torch.Tensor,
) -> tuple[str, str]:
    responses_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{sample_index:05d}_{safe_filename(sid)}"
    baseline_path = responses_dir / f"{prefix}_baseline.txt"
    eagle_path = responses_dir / f"{prefix}_eagle3.txt"
    baseline_path.write_text(tokenizer.decode(baseline_ids[0, input_len:], skip_special_tokens=True), "utf-8")
    eagle_path.write_text(tokenizer.decode(eagle_ids[0, input_len:], skip_special_tokens=True), "utf-8")
    return str(baseline_path), str(eagle_path)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_eagle_acceptance_metrics(new_tokens: int, decode_rounds: int, total_token: int) -> dict[str, Any]:
    mean_acceptance_length = new_tokens / decode_rounds if decode_rounds else 0.0
    acceptance_rate = new_tokens / (decode_rounds * total_token) if decode_rounds and total_token > 0 else 0.0
    return {
        "decode_rounds": decode_rounds,
        "accepted_tokens": int(new_tokens),
        "mean_acceptance_length": mean_acceptance_length,
        "acceptance_rate": acceptance_rate,
        "acceptance_histogram": {},
    }


def summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in records if row.get("status") == "ok"]
    skipped_rows = [row for row in records if str(row.get("status", "")).startswith("skipped")]
    oom_rows = [row for row in records if row.get("status") == "oom"]
    error_rows = [row for row in records if row.get("status") == "error"]
    baseline_tpot = _mean([float(row["baseline_tpot"]) for row in ok_rows if row.get("baseline_tpot") is not None])
    eagle_tpot = _mean([float(row["dflash_tpot"]) for row in ok_rows if row.get("dflash_tpot") is not None])
    return {
        "samples_total": len(records),
        "samples_ok": len(ok_rows),
        "samples_skipped": len(skipped_rows),
        "samples_oom": len(oom_rows),
        "samples_error": len(error_rows),
        "mean_prompt_tokens": _mean([float(row["prompt_tokens"]) for row in ok_rows if row.get("prompt_tokens") is not None]),
        "mean_baseline_tpot": baseline_tpot,
        "mean_dflash_tpot": eagle_tpot,
        "mean_speedup": _mean([float(row["speedup"]) for row in ok_rows if row.get("speedup") is not None]),
        "mean_acceptance_length": _mean(
            [float(row["mean_acceptance_length"]) for row in ok_rows if row.get("mean_acceptance_length") is not None]
        ),
        "mean_acceptance_rate": _mean([float(row["acceptance_rate"]) for row in ok_rows if row.get("acceptance_rate") is not None]),
        "mean_draft_forward_passes": _mean(
            [float(row["draft_forward_passes"]) for row in ok_rows if row.get("draft_forward_passes") is not None]
        ),
        "mean_draft_dynamic_yarn_factor": None,
        "min_draft_dynamic_yarn_factor": None,
        "max_draft_dynamic_yarn_factor": None,
        "mean_verify_draft_tokens": _mean(
            [float(row["mean_verify_draft_tokens"]) for row in ok_rows if row.get("mean_verify_draft_tokens") is not None]
        ),
        "mean_ctx_suffix_match_count": None,
        "max_ctx_suffix_match_count": None,
        "total_ctx_suffix_match_count": 0,
        "throughput_baseline_tok_s": 1.0 / baseline_tpot if baseline_tpot and baseline_tpot > 0 else None,
        "throughput_dflash_tok_s": 1.0 / eagle_tpot if eagle_tpot and eagle_tpot > 0 else None,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def write_outputs(out_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(out_dir / "per_sample.csv", records)
    summary = summary_from_records(records)
    write_csv(out_dir / "summary.csv", [summary])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    return summary


def write_rank_records(out_dir: Path, rank: int, records: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"rank_{rank}.records.jsonl.tmp"
    final_path = out_dir / f"rank_{rank}.records.jsonl"
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(final_path)


def write_rank_done(out_dir: Path, rank: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"rank_{rank}.done.tmp"
    final_path = out_dir / f"rank_{rank}.done"
    tmp_path.write_text(json.dumps({"rank": rank, "done_at": time.time()}) + "\n", "utf-8")
    tmp_path.replace(final_path)


def read_rank_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_records_path(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    records_path = Path(path).expanduser()
    if records_path.is_dir():
        records_path = records_path / "records.jsonl"
    return read_rank_records(records_path)


def merge_with_existing_records(records: list[dict[str, Any]], merge_records_from: str | None) -> list[dict[str, Any]]:
    if merge_records_from:
        full_records_by_index = {
            int(record["sample_index"]): record
            for record in read_records_path(merge_records_from)
        }
        for record in records:
            full_records_by_index[int(record["sample_index"])] = record
        final_records = [full_records_by_index[idx] for idx in sorted(full_records_by_index)]
    else:
        final_records = list(records)
    final_records.sort(key=lambda row: int(row["sample_index"]))
    return final_records


def wait_for_rank_records(out_dir: Path, world_size: int, timeout_s: float = 3600.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    paths = [out_dir / f"rank_{rank}.records.jsonl" for rank in range(world_size)]
    done_paths = [out_dir / f"rank_{rank}.done" for rank in range(world_size)]
    while True:
        if all(path.exists() for path in done_paths) and all(path.exists() for path in paths):
            merged: list[dict[str, Any]] = []
            for path in paths:
                merged.extend(read_rank_records(path))
            return merged
        if time.monotonic() > deadline:
            missing_records = [str(path) for path in paths if not path.exists()]
            missing_done = [str(path) for path in done_paths if not path.exists()]
            raise TimeoutError(
                f"timed out waiting for rank completion; missing_records={missing_records}, missing_done={missing_done}"
            )
        time.sleep(2.0)


def configure_target_yarn(config: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.target_yarn_original_max_position_embeddings is None:
        return None
    max_positions = args.target_yarn_max_position_embeddings or config.max_position_embeddings
    factor = args.target_yarn_factor
    if factor is None:
        factor = max_positions / args.target_yarn_original_max_position_embeddings
    config.max_position_embeddings = int(max_positions)
    rope_scaling = {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": int(args.target_yarn_original_max_position_embeddings),
    }
    config.rope_scaling = rope_scaling
    return rope_scaling


def timed_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    method: str,
    temperature: float,
    max_new_tokens: int,
    max_length: int,
) -> tuple[torch.Tensor, int, int | None, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    if method == "baseline":
        output_ids, new_tokens, rounds = model.naivegenerate(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
        )
    elif method == "eagle":
        output_ids, new_tokens, rounds = model.eagenerate(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
        )
    else:
        raise ValueError(f"unknown generation method: {method}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output_ids, int(new_tokens), int(rounds) if rounds is not None else None, elapsed


def empty_generation_result(input_ids: torch.Tensor) -> tuple[torch.Tensor, int, int | None, float]:
    return input_ids, 0, None, 0.0


def cache_capacity(model: Any) -> int:
    data = getattr(model, "past_key_values_data", None)
    if not data:
        return 0
    first = data[0] if isinstance(data, list) else data
    return int(first.shape[-2])


def ensure_target_kv_cache(model: Any, max_length: int) -> None:
    if cache_capacity(model) < max_length:
        (
            model.past_key_values,
            model.past_key_values_data,
            model.current_length_data,
        ) = initialize_past_key_values(model.base_model, max_length=max_length)
    else:
        model.current_length_data.zero_()


def ensure_eagle_buffers_on_device(model: Any) -> None:
    device = model.ea_layer.lm_head.weight.device
    model.ea_layer.init_tree()
    for name in ("d2t", "t2d"):
        if hasattr(model.ea_layer, name):
            value = getattr(model.ea_layer, name)
            if torch.is_tensor(value) and value.device != device:
                setattr(model.ea_layer, name, value.to(device))


def clear_runtime_cache(model: Any) -> None:
    for name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, name):
            delattr(model, name)
    if hasattr(model, "ea_layer"):
        try:
            model.ea_layer.reset_kv()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "device-side assert" not in str(exc):
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed local Transformers EAGLE3 benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eagle-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "results" / "reproduced" / "eagle3_benchmark"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--target-yarn-max-position-embeddings", type=int, default=None)
    parser.add_argument("--target-yarn-factor", type=float, default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--target-prefill-attn-implementation", default="flash_attention_2")
    parser.add_argument("--target-verify-attn-implementation", default="sdpa")
    parser.add_argument("--draft-attn-implementation", default="flash_attention_2")
    parser.add_argument("--sample-indices-file", default=None)
    parser.add_argument("--merge-records-from", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-save-responses", action="store_true")
    parser.add_argument("--clear-cache-between-samples", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-eagle", action="store_true")
    parser.add_argument("--max-length-pad", type=int, default=128)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    _dist_init()
    if torch.cuda.is_available() and args.device_map is None:
        torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}" if torch.cuda.is_available() else "cpu")

    out_dir = make_output_dir(args)
    if torch.distributed.is_initialized():
        out_dir_objs = [str(out_dir)] if _dist_is_main() else [None]
        torch.distributed.broadcast_object_list(out_dir_objs, src=0)
        out_dir = Path(str(out_dir_objs[0]))
    if _dist_is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = out_dir / "responses"

    target_config = AutoConfig.from_pretrained(args.model)
    target_rope = configure_target_yarn(target_config, args)
    if args.attn_implementation:
        target_config._attn_implementation = args.attn_implementation
    model_kwargs: dict[str, Any] = {
        "config": target_config,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = EaModel.from_pretrained(
        base_model_path=args.model,
        ea_model_path=args.eagle_model,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        use_eagle3=True,
        target_prefill_attn_implementation=args.target_prefill_attn_implementation,
        target_verify_attn_implementation=args.target_verify_attn_implementation,
        draft_attn_implementation=args.draft_attn_implementation,
        **model_kwargs,
    )
    if args.device_map is None:
        model = model.to(device)
    ensure_eagle_buffers_on_device(model)
    model.eval()
    tokenizer = model.get_tokenizer()

    selected_indices = load_sample_indices(args.sample_indices_file)
    dataset = limit_dataset(load_dataset(args.dataset), args.max_samples, seed=args.sample_seed)
    rank_samples = filter_samples(
        shard_dataset(dataset, rank=_dist_rank(), world_size=_dist_size()),
        selected_indices,
    )
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "eagle3_transformers",
        "model": args.model,
        "draft_model": args.eagle_model,
        "eagle_model": args.eagle_model,
        "dataset": args.dataset,
        "sample_count": len(dataset),
        "selected_sample_count": len(selected_indices) if selected_indices is not None else len(dataset),
        "sample_indices_file": args.sample_indices_file,
        "merge_records_from": args.merge_records_from,
        "clear_cache_between_samples": args.clear_cache_between_samples,
        "skip_baseline": args.skip_baseline,
        "skip_eagle": args.skip_eagle,
        "world_size": _dist_size(),
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "block_size": args.total_token,
        "eagle_total_token": args.total_token,
        "eagle_depth": args.depth,
        "eagle_top_k": args.top_k,
        "attn_implementation": args.attn_implementation,
        "target_prefill_attn_implementation": args.target_prefill_attn_implementation,
        "target_verify_attn_implementation": args.target_verify_attn_implementation,
        "draft_attn_implementation": args.draft_attn_implementation,
        "target_config_attn_implementation": getattr(target_config, "_attn_implementation", None),
        "target_rope_parameters": target_rope,
        "draft_rope_parameters": None,
        "draft_yarn_enabled": False,
        "target_config_max_position_embeddings": target_config.max_position_embeddings,
        "target_config_output_hidden_states": target_config.output_hidden_states,
    }
    if _dist_is_main():
        (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), "utf-8")

    records: list[dict[str, Any]] = []
    total_rank_samples = len(rank_samples)
    for sample_index, sample in tqdm(rank_samples, desc=f"Benchmarking rank {_dist_rank()}", disable=not _dist_is_main()):
        sid = sample_id(sample, sample_index)
        record: dict[str, Any] = {"sample_index": sample_index, "sample_id": sid, "rank": _dist_rank()}
        input_ids = None
        try:
            prompt = render_prompt(tokenizer, sample, args.enable_thinking, add_generation_prompt=True)
            input_ids = torch.as_tensor(tokenizer([prompt]).input_ids, device=device)
            prompt_tokens = int(input_ids.shape[1])
            record["prompt_tokens"] = prompt_tokens
            max_length = prompt_tokens + args.max_new_tokens + args.total_token + args.max_length_pad

            ensure_target_kv_cache(model, max_length)
            if args.skip_baseline:
                baseline_ids, baseline_new_tokens, baseline_rounds, baseline_elapsed = empty_generation_result(input_ids)
            else:
                record["active_phase"] = "baseline"
                baseline_ids, baseline_new_tokens, baseline_rounds, baseline_elapsed = timed_generate(
                    model,
                    input_ids,
                    method="baseline",
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    max_length=max_length,
                )
            if args.clear_cache_between_samples:
                if hasattr(model, "ea_layer"):
                    model.ea_layer.reset_kv()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                model.ea_layer.reset_kv()
            ensure_target_kv_cache(model, max_length)
            if args.skip_eagle:
                eagle_ids, eagle_new_tokens, eagle_rounds, eagle_elapsed = empty_generation_result(input_ids)
            else:
                record["active_phase"] = "eagle"
                eagle_ids, eagle_new_tokens, eagle_rounds, eagle_elapsed = timed_generate(
                    model,
                    input_ids,
                    method="eagle",
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    max_length=max_length,
                )
            baseline_tpot = baseline_elapsed / baseline_new_tokens if baseline_new_tokens > 0 else None
            eagle_tpot = eagle_elapsed / eagle_new_tokens if eagle_new_tokens > 0 else None
            baseline_round_count = (baseline_rounds + 1) if baseline_rounds is not None else None
            eagle_round_count = (eagle_rounds + 1) if eagle_rounds is not None else None
            acceptance = compute_eagle_acceptance_metrics(eagle_new_tokens, eagle_round_count or 0, args.total_token)
            record.update(
                status="ok",
                active_phase=None,
                baseline_output_tokens=baseline_new_tokens,
                dflash_output_tokens=eagle_new_tokens,
                eagle_output_tokens=eagle_new_tokens,
                baseline_tpot=baseline_tpot,
                dflash_tpot=eagle_tpot,
                eagle_tpot=eagle_tpot,
                speedup=(baseline_tpot / eagle_tpot) if baseline_tpot and eagle_tpot and eagle_tpot > 0 else None,
                time_to_first_token=None,
                acceptance_lengths=[],
                verify_draft_lengths=[args.total_token for _ in range(max(0, eagle_round_count or 0))],
                mean_verify_draft_tokens=args.total_token if eagle_round_count else None,
                draft_forward_passes=eagle_round_count,
                draft_dynamic_yarn_factor=None,
                baseline_elapsed_s=baseline_elapsed,
                eagle_elapsed_s=eagle_elapsed,
                baseline_rounds=baseline_round_count,
                eagle_rounds=eagle_round_count,
                max_length=max_length,
                **acceptance,
            )
            if not args.no_save_responses:
                baseline_path, eagle_path = write_response_texts(
                    responses_dir, sample_index, sid, tokenizer, prompt_tokens, baseline_ids, eagle_ids
                )
                record["baseline_response_path"] = baseline_path
                record["dflash_response_path"] = eagle_path
                record["eagle_response_path"] = eagle_path
            records.append(record)
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError as cache_exc:
                    if "device-side assert" not in str(cache_exc):
                        raise
            record.update(status="oom", error=str(exc))
            if input_ids is not None:
                record.setdefault("prompt_tokens", int(input_ids.shape[1]))
            records.append(record)
        except Exception as exc:
            record.update(status="error", error=repr(exc), traceback=traceback.format_exc())
            if input_ids is not None:
                record.setdefault("prompt_tokens", int(input_ids.shape[1]))
            records.append(record)
        if args.clear_cache_between_samples:
            clear_runtime_cache(model)
        write_rank_records(out_dir, _dist_rank(), records)
        if _dist_is_main() and _dist_size() == 1:
            partial_records = merge_with_existing_records(records, args.merge_records_from)
            partial_summary = write_outputs(out_dir, partial_records)
            print(
                json.dumps(
                    {
                        "output_dir": str(out_dir),
                        "progress": f"{len(records)}/{total_rank_samples}",
                        "last_sample_index": sample_index,
                        "last_status": record.get("status"),
                        "summary": partial_summary,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_rank_records(out_dir, _dist_rank(), records)
    write_rank_done(out_dir, _dist_rank())
    if not _dist_is_main():
        return {"output_dir": str(out_dir), "summary": None}
    merged_records = wait_for_rank_records(out_dir, _dist_size())
    final_records = merge_with_existing_records(merged_records, args.merge_records_from)
    summary = write_outputs(out_dir, final_records)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2, ensure_ascii=False))
    return {"output_dir": str(out_dir), "summary": summary}


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
