#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import server_utils as base
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import server_utils as base


ROOT = Path(__file__).resolve().parents[2]
MINI_ROOT = Path(os.environ.get("MINI_SWE_AGENT_ROOT", ROOT / "third_party" / "mini-swe-agent"))
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "reproduced" / "end_to_end_swebench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--start-script", default=str(ROOT / "scripts/serve/start_vllm_dflash_benchmark.sh"))
    parser.add_argument(
        "--target-start-script",
        default=str(ROOT / "scripts/serve/start_vllm_qwen35_target_benchmark.sh"),
    )
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--slice-start", type=int, default=0)
    parser.add_argument(
        "--instance-ids",
        default="",
        help="Comma/space separated SWE-bench instance IDs to run. Overrides --sample-count/--slice-start slicing.",
    )
    parser.add_argument(
        "--instance-ids-file",
        type=Path,
        default=None,
        help="File containing SWE-bench instance IDs to run. Accepts one ID per line or comma/space separated IDs. Lines may use # comments.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="4,5,6,7")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--num-spec-tokens", type=int, default=15)
    parser.add_argument("--request-timeout-sec", type=float, default=28800.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument("--reasoning-parser", default="qwen3")
    parser.add_argument("--target-yarn-original-max-position-embeddings", type=int, default=32768)
    parser.add_argument("--target-yarn-factor", type=float, default=16.0)
    parser.add_argument(
        "--disable-target-yarn",
        action="store_true",
        help="Do not pass target YaRN hf-overrides for yarn variants; draft YaRN flags still apply.",
    )
    parser.add_argument(
        "--target-yarn-baselines",
        action="store_true",
        help="Apply target YaRN to target-only/original variants too. Useful when a target-only 128k baseline exceeds the target model's native context.",
    )
    parser.add_argument(
        "--original-max-position-embedding",
        type=int,
        default=None,
        help=(
            "For the original DFlash variant, use a temporary draft model config "
            "with max_position_embeddings set to this value and no draft YaRN."
        ),
    )
    parser.add_argument("--draft-yarn-original-max-position-embeddings", type=int, default=None)
    parser.add_argument("--draft-yarn-factor", type=float, default=None)
    parser.add_argument("--suffix-max-query-len", type=int, default=16)
    parser.add_argument("--suffix-min-query-len", type=int, default=10)
    parser.add_argument("--suffix-max-predict-len", type=int, default=15)
    parser.add_argument("--suffix-alpha", type=float, default=2.0)
    parser.add_argument("--suffix-max-spec-offset", type=int, default=0)
    parser.add_argument("--suffix-min-token-prob", type=float, default=0.0)
    parser.add_argument("--suffix-threshold", type=float, default=4.0)
    parser.add_argument("--suffix-max-matches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--variants",
        default="target-only,original,yarn",
        help=(
            "Comma/space separated variants to run: target-only, original, yarn, "
            "yarn-suffix."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip variants with run_summary.json.")
    parser.add_argument("--skip-server-stop", action="store_true")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--environment-class", default="kubernetes")
    parser.add_argument("--kubernetes-namespace", default="debug-gym-swe")
    parser.add_argument(
        "--environment-timeout",
        type=int,
        default=None,
        help="Override mini-swe-agent environment.timeout for each shell command.",
    )
    parser.add_argument(
        "--agent-step-limit",
        type=int,
        default=None,
        help="Override mini-swe-agent agent.step_limit, e.g. 100.",
    )
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=None,
        help="Set mini-swe-agent model.model_kwargs.max_tokens for each LLM request.",
    )
    parser.add_argument(
        "--kubernetes-run-arg",
        action="append",
        default=[],
        help="Additional argument to pass through mini-swe-agent KubernetesEnvironment.run_args. Repeat for multiple args.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false through vLLM chat_template_kwargs for Qwen reasoning models.",
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable vLLM prefix caching. Prefix caching is enabled by default.",
    )
    args = parser.parse_args()
    if (
        args.draft_yarn_original_max_position_embeddings is None
    ) != (args.draft_yarn_factor is None):
        parser.error(
            "--draft-yarn-original-max-position-embeddings and --draft-yarn-factor "
            "must be set together."
        )
    return args


def _parse_csv(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def _read_instance_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.instance_ids:
        ids.extend(_parse_csv(args.instance_ids))
    if args.instance_ids_file is not None:
        text = args.instance_ids_file.read_text(encoding="utf-8")
        cleaned_lines = []
        for line in text.splitlines():
            cleaned_lines.append(line.split("#", 1)[0])
        ids.extend(_parse_csv("\n".join(cleaned_lines)))

    deduped = list(dict.fromkeys(ids))
    if len(deduped) != len(ids):
        print(
            f"Deduplicated instance IDs: {len(ids)} -> {len(deduped)}",
            file=sys.stderr,
        )
    return deduped


def _instance_filter_regex(instance_ids: list[str]) -> str:
    return "^(?:" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"


def _normalize_variant_name(name: str) -> str:
    aliases = {
        "yarn16": "yarn",
        "yarn16-suffix": "yarn-suffix",
    }
    return aliases.get(name, name)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _model_registry_path(run_dir: Path, model_name: str) -> Path:
    registry = {
        model_name: {
            "max_tokens": 131072,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "hosted_vllm",
            "mode": "chat",
        }
    }
    path = run_dir / "litellm_model_registry.json"
    _write_json(path, registry)
    return path


def _agent_override_path(
    run_dir: Path,
    model_name: str,
    api_base: str,
    *,
    environment_class: str,
    kubernetes_namespace: str,
    kubernetes_run_args: list[str],
    environment_timeout: int | None,
    agent_step_limit: int | None,
    agent_max_tokens: int | None,
    disable_thinking: bool,
) -> Path:
    text = (
        "model:\n"
        f'  model_name: "{model_name}"\n'
        "  model_kwargs:\n"
        f'    api_base: "{api_base}"\n'
    )
    if agent_max_tokens is not None:
        text += f"    max_tokens: {agent_max_tokens}\n"
    if disable_thinking:
        text += (
            "    extra_body:\n"
            "      chat_template_kwargs:\n"
            "        enable_thinking: false\n"
        )
    text += (
        "environment:\n"
        f'  environment_class: "{environment_class}"\n'
    )
    if environment_timeout is not None:
        text += f"  timeout: {environment_timeout}\n"
    if environment_class in {"kubernetes", "k8s"}:
        text += f'  namespace: "{kubernetes_namespace}"\n'
        if kubernetes_run_args:
            text += "  run_args:\n"
            for arg in kubernetes_run_args:
                text += f"    - {json.dumps(arg)}\n"
    if agent_step_limit is not None:
        text += (
            "agent:\n"
            f"  step_limit: {agent_step_limit}\n"
        )
    path = run_dir / "local_vllm_override.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _variant_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    yarn_original = (
        None
        if args.disable_target_yarn
        else args.target_yarn_original_max_position_embeddings
    )
    yarn_factor = None if args.disable_target_yarn else args.target_yarn_factor
    yarn_max_position = (
        None
        if args.disable_target_yarn
        else int(args.target_yarn_original_max_position_embeddings * args.target_yarn_factor)
    )
    draft_yarn_max_position = (
        int(args.draft_yarn_original_max_position_embeddings * args.draft_yarn_factor)
        if args.draft_yarn_original_max_position_embeddings is not None
        and args.draft_yarn_factor is not None
        else None
    )
    variants = [
        {
            "name": "target-only",
            "window": "full",
            "window_mode": "target_only",
            "target_only": True,
            "start_script": args.target_start_script,
            "target_yarn_original": yarn_original if args.target_yarn_baselines else None,
            "target_yarn_factor": yarn_factor if args.target_yarn_baselines else None,
            "target_yarn_max_position_embeddings": (
                yarn_max_position if args.target_yarn_baselines else None
            ),
        },
        {
            "name": "original",
            "window": "full",
            "window_mode": "full",
            "target_only": False,
            "start_script": args.start_script,
            "target_yarn_original": yarn_original if args.target_yarn_baselines else None,
            "target_yarn_factor": yarn_factor if args.target_yarn_baselines else None,
            "target_yarn_max_position_embeddings": (
                yarn_max_position if args.target_yarn_baselines else None
            ),
            "original_max_position_embedding": (
                args.original_max_position_embedding
            ),
        },
        {
            "name": "yarn",
            "window": "full",
            "window_mode": "full",
            "target_only": False,
            "start_script": args.start_script,
            "target_yarn_original": yarn_original,
            "target_yarn_factor": yarn_factor,
            "target_yarn_max_position_embeddings": yarn_max_position,
            "draft_yarn_original": args.draft_yarn_original_max_position_embeddings,
            "draft_yarn_factor": args.draft_yarn_factor,
            "draft_yarn_max_position_embeddings": draft_yarn_max_position,
        },
        {
            "name": "yarn-suffix",
            "window": "full",
            "window_mode": "full",
            "target_only": False,
            "start_script": args.start_script,
            "target_yarn_original": yarn_original,
            "target_yarn_factor": yarn_factor,
            "target_yarn_max_position_embeddings": yarn_max_position,
            "draft_yarn_original": args.draft_yarn_original_max_position_embeddings,
            "draft_yarn_factor": args.draft_yarn_factor,
            "draft_yarn_max_position_embeddings": draft_yarn_max_position,
            "suffix_decoding": True,
            "suffix_max_query_len": args.suffix_max_query_len,
            "suffix_min_query_len": args.suffix_min_query_len,
            "suffix_max_predict_len": args.suffix_max_predict_len,
            "suffix_alpha": args.suffix_alpha,
            "suffix_max_spec_offset": args.suffix_max_spec_offset,
            "suffix_min_token_prob": args.suffix_min_token_prob,
            "suffix_threshold": args.suffix_threshold,
            "suffix_max_matches": args.suffix_max_matches,
        },
    ]
    variant_by_name = {variant["name"]: variant for variant in variants}
    selected = [_normalize_variant_name(name) for name in _parse_csv(args.variants)]
    unknown = set(selected) - set(variant_by_name)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    return [variant_by_name[name] for name in selected]


def _request_metrics_summary(log_path: Path, summary_path: Path) -> None:
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/benchmark/summarize_mswea_request_metrics.py"),
        str(log_path),
        "--output-json",
        str(summary_path),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def _fetch_server_counters(server_root_url: str) -> dict[str, float]:
    import httpx

    resp = httpx.get(f"{server_root_url}/metrics", timeout=30.0)
    resp.raise_for_status()
    metrics = {
        "generation_tokens": 0.0,
        "prompt_tokens": 0.0,
        "request_success": 0.0,
    }
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            parsed = float(value)
        except ValueError:
            continue
        bare_name = name.split("{", 1)[0]
        if bare_name in {"vllm:generation_tokens", "vllm:generation_tokens_total"}:
            metrics["generation_tokens"] += parsed
        elif bare_name in {"vllm:prompt_tokens", "vllm:prompt_tokens_total"}:
            metrics["prompt_tokens"] += parsed
        elif bare_name in {"vllm:request_success", "vllm:request_success_total"}:
            metrics["request_success"] += parsed
    return metrics


def _server_counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in before}


def run_variant(args: argparse.Namespace, run_root: Path, variant: dict[str, Any]) -> dict[str, Any]:
    run_dir = (run_root / variant["name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_summary = run_dir / "run_summary.json"
    if args.resume and existing_summary.exists():
        return json.loads(existing_summary.read_text("utf-8"))
    request_log_path = run_dir / "llm_requests.jsonl"
    registry_path = _model_registry_path(run_dir, f"hosted_vllm/{args.model}")
    agent_override_path = _agent_override_path(
        run_dir,
        f"hosted_vllm/{args.model}",
        args.base_url,
        environment_class=args.environment_class,
        kubernetes_namespace=args.kubernetes_namespace,
        kubernetes_run_args=args.kubernetes_run_arg,
        environment_timeout=args.environment_timeout,
        agent_step_limit=args.agent_step_limit,
        agent_max_tokens=args.agent_max_tokens,
        disable_thinking=args.disable_thinking,
    )
    server_root_url = base.server_root_from_base_url(args.base_url)

    server_args = SimpleNamespace(
        model=args.model,
        draft_model=None if variant["target_only"] else args.draft_model,
        port=args.port,
        num_spec_tokens=args.num_spec_tokens,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        start_script=variant["start_script"],
        window_mode=variant["window_mode"],
        sink_tokens=0,
        position_mode="compact",
        select_ranges="",
        recent_tokens=None,
        suffix_match_tokens=0,
        suffix_keep_tokens=0,
        suffix_middle_budget=0,
        suffix_source_tail_token_ids=None,
        suffix_decoding=variant.get("suffix_decoding", False),
        suffix_max_query_len=variant.get("suffix_max_query_len"),
        suffix_min_query_len=variant.get("suffix_min_query_len"),
        suffix_max_predict_len=variant.get("suffix_max_predict_len"),
        suffix_alpha=variant.get("suffix_alpha"),
        suffix_max_spec_offset=variant.get("suffix_max_spec_offset"),
        suffix_min_token_prob=variant.get("suffix_min_token_prob"),
        suffix_threshold=variant.get("suffix_threshold"),
        suffix_max_matches=variant.get("suffix_max_matches"),
        dynamic_budget_ratio=None,
        draft_yarn_original=variant.get("draft_yarn_original"),
        draft_yarn_factor=variant.get("draft_yarn_factor"),
        draft_yarn_max_position_embeddings=variant.get("draft_yarn_max_position_embeddings"),
        original_max_position_embedding=variant.get(
            "original_max_position_embedding"
        ),
        dynamic_yarn_original=None,
        dynamic_yarn_max_factor=None,
        dynamic_yarn_mode=None,
        dynamic_yarn_length_ratio=None,
        request_max_tokens=None,
        target_yarn_original=variant["target_yarn_original"],
        target_yarn_max_position_embeddings=variant["target_yarn_max_position_embeddings"],
        target_yarn_factor=variant["target_yarn_factor"],
        extra_vllm_args="",
        allow_long_max_model_len=True,
        enable_chunked_prefill=None,
        enable_prefix_caching=not args.disable_prefix_cache,
        enforce_eager=False,
        vllm_cache_root=str(run_dir / "vllm_cache"),
    )

    if not args.skip_server_stop:
        base.stop_existing_server(args.model, port=args.port)
    server_log_path = run_dir / "server.log"
    if args.dry_run:
        return {
            "variant": variant["name"],
            "run_dir": str(run_dir),
            "request_log_path": str(request_log_path),
            "server_log_path": str(server_log_path),
        }

    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_tp = os.environ.get("TP_SIZE")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["TP_SIZE"] = str(args.tp_size)
    server_proc = base.start_server(server_args, variant["window"], server_log_path)
    if old_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
    if old_tp is None:
        os.environ.pop("TP_SIZE", None)
    else:
        os.environ["TP_SIZE"] = old_tp
    try:
        base.wait_for_server(server_root_url, args.startup_timeout_sec, args.poll_interval_sec, proc=server_proc)
        before_metrics = _fetch_server_counters(server_root_url)

        cmd = [
            str(MINI_ROOT / ".venv/bin/mini-extra"),
            "swebench",
            "--subset",
            args.subset,
            "--split",
            args.split,
            "--workers",
            str(args.workers),
            "--output",
            str(run_dir / "swebench_output"),
            "--model",
            f"hosted_vllm/{args.model}",
            "-c",
            "swebench.yaml",
            "-c",
            str(agent_override_path),
        ]
        if args.selected_instance_ids:
            cmd.extend(["--filter", _instance_filter_regex(args.selected_instance_ids)])
        else:
            cmd.extend(["--slice", f"{args.slice_start}:{args.slice_start + args.sample_count}"])
        mini_env = os.environ.copy()
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            mini_env.pop(key, None)
        mini_env["NO_PROXY"] = "127.0.0.1,localhost"
        mini_env["no_proxy"] = "127.0.0.1,localhost"
        mini_env["LITELLM_MODEL_REGISTRY_PATH"] = str(registry_path)
        mini_env["MSWEA_REQUEST_LOG_PATH"] = str(request_log_path)
        mini_env["MSWEA_REQUEST_RUN_LABEL"] = args.run_label or f"{variant['name']}-w{args.workers}"
        mini_env["MSWEA_COST_TRACKING"] = "ignore_errors"
        mini_env.setdefault("OPENAI_API_KEY", "EMPTY")
        start = time.time()
        subprocess.run(
            cmd,
            check=True,
            cwd=MINI_ROOT,
            env=mini_env,
            timeout=args.request_timeout_sec,
        )
        end = time.time()

        after_metrics = _fetch_server_counters(server_root_url)
        server_delta = _server_counter_delta(before_metrics, after_metrics)
        request_summary_path = run_dir / "llm_request_summary.json"
        _request_metrics_summary(request_log_path, request_summary_path)
        request_summary = json.loads(request_summary_path.read_text("utf-8"))
        if args.workers == 1:
            request_summary["server_busy_output_tps_from_vllm"] = (
                server_delta["generation_tokens"] / request_summary["wall_time_sec"]
                if request_summary.get("wall_time_sec")
                else None
            )
            request_summary["server_busy_total_tps_from_vllm"] = (
                (server_delta["generation_tokens"] + server_delta["prompt_tokens"])
                / request_summary["wall_time_sec"]
                if request_summary.get("wall_time_sec")
                else None
            )
        summary = {
            "variant": variant["name"],
            "workers": args.workers,
            "sample_count": len(args.selected_instance_ids) if args.selected_instance_ids else args.sample_count,
            "selected_instance_ids": args.selected_instance_ids,
            "wall_time_sec": end - start,
            "server_counter_delta": server_delta,
            "request_summary": request_summary,
            "request_log_path": str(request_log_path),
            "server_log_path": str(server_log_path),
        }
        _write_json(run_dir / "run_summary.json", summary)
        return summary
    finally:
        if server_proc.poll() is None:
            try:
                base.stop_existing_server(args.model, port=args.port)
            except Exception:
                server_proc.terminate()
                server_proc.wait(timeout=30)


def main() -> None:
    args = parse_args()
    args.selected_instance_ids = _read_instance_ids(args)
    if args.run_root is not None:
        run_root = args.run_root.resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sample_label = (
            f"ids{len(args.selected_instance_ids)}"
            if args.selected_instance_ids
            else f"s{args.slice_start}_n{args.sample_count}"
        )
        run_root = (
            args.output_root
            / f"swebench_verified_{timestamp}_{sample_label}_w{args.workers}"
        ).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    if run_config.get("instance_ids_file") is not None:
        run_config["instance_ids_file"] = str(run_config["instance_ids_file"])
    _write_json(run_root / "run_config.json", run_config)

    results = []
    for variant in _variant_specs(args):
        results.append(run_variant(args, run_root, variant))
    _write_json(run_root / "aggregate_summary.json", results)
    print(run_root)


if __name__ == "__main__":
    main()
