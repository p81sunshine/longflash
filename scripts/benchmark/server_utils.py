"""Shared vLLM server and speculative decoding metric helpers."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = ROOT


def server_root_from_base_url(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url.rstrip("/")


def server_port_from_base_url(base_url: str) -> int | None:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    return parsed.port


def fetch_spec_decode_metrics(server_root_url: str):
    resp = httpx.get(f"{server_root_url}/metrics", timeout=30.0)
    if resp.status_code != 200:
        return None
    num_drafts = num_draft_tokens = num_accepted_tokens = 0
    accepted_tokens_per_pos: dict[int, int] = {}
    found = False
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        if "_created" in line:
            continue
        found = True
        parts = line.split()
        try:
            value = int(float(parts[-1]))
        except ValueError:
            continue
        if "num_accepted_tokens_per_pos" in line:
            pos_label = 'position="'
            if pos_label in line:
                try:
                    start = line.index(pos_label) + len(pos_label)
                    end = line.index('"', start)
                    pos = int(line[start:end])
                except (ValueError, IndexError):
                    continue
                accepted_tokens_per_pos[pos] = (
                    accepted_tokens_per_pos.get(pos, 0) + value
                )
        elif "num_draft_tokens" in line:
            num_draft_tokens += value
        elif "num_accepted_tokens" in line:
            num_accepted_tokens += value
        elif "num_drafts" in line:
            num_drafts += value
    if not found:
        return None
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "accepted_tokens_per_pos": accepted_tokens_per_pos,
    }


def compute_delta(before, after):
    if before is None or after is None:
        return None
    num_drafts = after["num_drafts"] - before["num_drafts"]
    draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    accepted_tokens = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    before_per_pos = before.get("accepted_tokens_per_pos", {})
    after_per_pos = after.get("accepted_tokens_per_pos", {})
    max_pos = max([*before_per_pos.keys(), *after_per_pos.keys()], default=-1)
    accepted_tokens_per_pos = [
        after_per_pos.get(pos, 0) - before_per_pos.get(pos, 0)
        for pos in range(max_pos + 1)
    ]
    acceptance_rate_per_pos = [
        (accepted / num_drafts) * 100.0 if num_drafts > 0 else None
        for accepted in accepted_tokens_per_pos
    ]
    if draft_tokens <= 0:
        return None
    return {
        "num_drafts": num_drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": (accepted_tokens / draft_tokens) * 100.0,
        "acceptance_length": 1.0 + (accepted_tokens / num_drafts)
        if num_drafts > 0
        else None,
        "accepted_tokens_per_pos": accepted_tokens_per_pos,
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
    }


def get_matching_pids(model_path: str, port: int | None = None) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", f"vllm serve {model_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "pgrep failed")
    pids = [int(pid) for pid in result.stdout.split() if pid.strip()]
    if port is not None:
        pids = [pid for pid in pids if process_uses_port(pid, port)]
    return pids


def is_own_dflash_process(pid: int) -> bool:
    try:
        proc_path = Path(f"/proc/{pid}")
        if proc_path.stat().st_uid != os.getuid():
            return False
        return proc_path.joinpath("cwd").resolve() == DEFAULT_WORKDIR
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def process_uses_port(pid: int, port: int | None) -> bool:
    if port is None:
        return True
    cmdline = process_cmdline(pid)
    return bool(re.search(rf"(?:^|\s)--port\s+{port}(?:\s|$)", cmdline))


def get_vllm_worker_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "VLLM::EngineCore|VLLM::Worker_TP|VLLM::APIServer"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "pgrep vLLM workers failed")
    return [
        int(pid)
        for pid in result.stdout.split()
        if pid.strip() and is_own_dflash_process(int(pid))
    ]


def get_live_pids(pids: list[int]) -> list[int]:
    live = []
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("State:"):
                    if line.split()[1] != "Z":
                        live.append(pid)
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
    return live


def get_descendant_pids(pids: list[int]) -> list[int]:
    descendants: list[int] = []
    queue = list(pids)
    while queue:
        parent = queue.pop(0)
        result = subprocess.run(
            ["pgrep", "-P", str(parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.strip() or "pgrep children failed")
        children = [int(pid) for pid in result.stdout.split() if pid.strip()]
        descendants.extend(children)
        queue.extend(children)
    return descendants


def stop_existing_server(model_path: str, port: int | None = None) -> None:
    pids = get_matching_pids(model_path, port=port)
    all_pids = set(pids + get_descendant_pids(pids))
    if port is None:
        all_pids.update(get_vllm_worker_pids())
    pids = sorted(all_pids, reverse=True)
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20.0
    remaining = set(get_live_pids(pids))
    while remaining and time.monotonic() < deadline:
        time.sleep(0.5)
        remaining = set(get_live_pids(list(remaining)))
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while remaining and time.monotonic() < deadline:
        time.sleep(0.5)
        remaining = set(get_live_pids(list(remaining)))
    if remaining:
        raise TimeoutError(f"PIDs did not exit: {remaining}")


def _set_env_if_present(env: dict[str, str], args, attr: str, name: str) -> None:
    value = getattr(args, attr, None)
    if value is not None and value != "":
        env[name] = str(value)


def start_server(args, window: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["DFLASH_WINDOW_SIZE"] = window
    env["NUM_SPEC_TOKENS"] = str(args.num_spec_tokens)
    model_path = getattr(args, "model", None)
    if model_path:
        env["MODEL_PATH"] = model_path
    draft_model = getattr(args, "draft_model", None)
    if draft_model:
        env["DRAFT_MODEL_PATH"] = draft_model
    port = getattr(args, "port", None)
    if port is not None:
        env["PORT"] = str(port)
    max_model_len = getattr(args, "max_model_len", None)
    if max_model_len is not None:
        env["MAX_MODEL_LEN"] = str(max_model_len)
    max_num_seqs = getattr(args, "max_num_seqs", None)
    if max_num_seqs is not None:
        env["MAX_NUM_SEQS"] = str(max_num_seqs)
    vllm_cache_root = getattr(args, "vllm_cache_root", None)
    if vllm_cache_root:
        env["VLLM_CACHE_ROOT"] = str(vllm_cache_root)
    if window != "full":
        env["DFLASH_WINDOW_MODE"] = args.window_mode
        env["DFLASH_SINK_TOKENS"] = str(args.sink_tokens)
        position_mode = getattr(args, "position_mode", "")
        if position_mode:
            env["DFLASH_POSITION_MODE"] = position_mode
        select_ranges = getattr(args, "select_ranges", "")
        if select_ranges:
            env["DFLASH_SELECT_RANGES"] = select_ranges

    _set_env_if_present(env, args, "recent_tokens", "DFLASH_RECENT_TOKENS")
    _set_env_if_present(env, args, "suffix_match_tokens", "DFLASH_SUFFIX_MATCH_TOKENS")
    _set_env_if_present(env, args, "suffix_keep_tokens", "DFLASH_SUFFIX_KEEP_TOKENS")
    _set_env_if_present(env, args, "suffix_middle_budget", "DFLASH_SUFFIX_MIDDLE_BUDGET")
    suffix_tail_ids = getattr(args, "suffix_source_tail_token_ids", None)
    if suffix_tail_ids:
        env["DFLASH_SUFFIX_SOURCE_TAIL_TOKEN_IDS"] = json.dumps(suffix_tail_ids)
    if getattr(args, "suffix_decoding", False):
        env["DFLASH_SUFFIX_DECODING"] = "1"
    _set_env_if_present(env, args, "suffix_max_query_len", "DFLASH_SUFFIX_MAX_QUERY_LEN")
    _set_env_if_present(env, args, "suffix_min_query_len", "DFLASH_SUFFIX_MIN_QUERY_LEN")
    _set_env_if_present(env, args, "suffix_max_predict_len", "DFLASH_SUFFIX_MAX_PREDICT_LEN")
    _set_env_if_present(env, args, "suffix_alpha", "DFLASH_SUFFIX_ALPHA")
    _set_env_if_present(env, args, "suffix_max_spec_offset", "DFLASH_SUFFIX_MAX_SPEC_OFFSET")
    _set_env_if_present(env, args, "suffix_min_token_prob", "DFLASH_SUFFIX_MIN_TOKEN_PROB")
    _set_env_if_present(env, args, "suffix_threshold", "DFLASH_SUFFIX_THRESHOLD")
    _set_env_if_present(env, args, "suffix_max_matches", "DFLASH_SUFFIX_MAX_MATCHES")
    _set_env_if_present(env, args, "dynamic_budget_ratio", "DFLASH_DYNAMIC_BUDGET_RATIO")
    _set_env_if_present(env, args, "draft_yarn_original", "DRAFT_STATIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS")
    _set_env_if_present(env, args, "draft_yarn_factor", "DRAFT_STATIC_YARN_FACTOR")
    _set_env_if_present(env, args, "draft_yarn_max_position_embeddings", "DRAFT_STATIC_YARN_MAX_POSITION_EMBEDDINGS")
    _set_env_if_present(env, args, "original_max_position_embedding", "ORIGINAL_MAX_POSITION_EMBEDDING")
    _set_env_if_present(env, args, "dynamic_yarn_original", "DRAFT_DYNAMIC_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS")
    _set_env_if_present(env, args, "dynamic_yarn_max_factor", "DRAFT_DYNAMIC_YARN_MAX_FACTOR")
    _set_env_if_present(env, args, "dynamic_yarn_mode", "DRAFT_DYNAMIC_YARN_MODE")
    _set_env_if_present(env, args, "dynamic_yarn_length_ratio", "DRAFT_DYNAMIC_YARN_LENGTH_RATIO")
    _set_env_if_present(env, args, "request_max_tokens", "DFLASH_REQUEST_MAX_TOKENS")
    _set_env_if_present(env, args, "target_yarn_original", "TARGET_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS")
    _set_env_if_present(env, args, "target_yarn_max_position_embeddings", "TARGET_YARN_MAX_POSITION_EMBEDDINGS")
    _set_env_if_present(env, args, "target_yarn_factor", "TARGET_YARN_FACTOR")
    _set_env_if_present(env, args, "max_num_batched_tokens", "MAX_BATCHED_TOKENS")
    _set_env_if_present(env, args, "gpu_memory_utilization", "GPU_MEMORY_UTILIZATION")
    _set_env_if_present(env, args, "tool_call_parser", "TOOL_CALL_PARSER")
    _set_env_if_present(env, args, "reasoning_parser", "REASONING_PARSER")
    _set_env_if_present(env, args, "extra_vllm_args", "EXTRA_VLLM_ARGS")
    if getattr(args, "allow_long_max_model_len", False):
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    if hasattr(args, "enable_chunked_prefill"):
        value = getattr(args, "enable_chunked_prefill", None)
        if value is None:
            env["ENABLE_CHUNKED_PREFILL"] = "default"
        else:
            env["ENABLE_CHUNKED_PREFILL"] = "1" if value else "0"
    if hasattr(args, "enable_prefix_caching"):
        value = getattr(args, "enable_prefix_caching", None)
        if value is None:
            env["ENABLE_PREFIX_CACHING"] = "default"
        else:
            env["ENABLE_PREFIX_CACHING"] = "1" if value else "0"
    if getattr(args, "enforce_eager", False):
        env["ENFORCE_EAGER"] = "1"

    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["bash", args.start_script],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def wait_for_server(server_root_url: str, timeout: float, poll: float, proc=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}")
        try:
            health = httpx.get(f"{server_root_url}/health", timeout=10.0)
            models = httpx.get(f"{server_root_url}/v1/models", timeout=10.0)
            if health.status_code == 200 and models.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(poll)
    raise TimeoutError(f"Server not ready at {server_root_url}")
