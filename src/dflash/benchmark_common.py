from __future__ import annotations

import argparse
import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any


CACHE_DIR = Path(__file__).parent.parent / "cache"

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / f"{name}.jsonl"
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")

    print(f"[download] {name} ...")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in dataset:
            if cfg.get("multi_turn"):
                turns = cfg["format"](row)
            else:
                turns = [cfg["format"](row)]
            handle.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)

    with open(out_path, encoding="utf-8") as handle:
        num_samples = sum(1 for _ in handle)
    print(f"[cached] {out_path}  ({num_samples} samples)")
    return out_path


def load_and_process_dataset(data_name: str) -> list[dict]:
    dataset_path = Path(data_name).expanduser()
    if dataset_path.is_file():
        with open(dataset_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    path = CACHE_DIR / f"{data_name}.jsonl"
    if not path.exists():
        _prepare_dataset(data_name)

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _apply_chat_template(
    tokenizer,
    messages: list[dict],
    enable_thinking: bool,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def configure_yarn_rope(
    config: Any,
    *,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    factor: float | None = None,
    error_prefix: str,
) -> None:
    if original_max_position_embeddings <= 0:
        raise ValueError(f"--{error_prefix}-yarn-original-max-position-embeddings must be positive")
    if max_position_embeddings <= 0:
        raise ValueError(f"--{error_prefix}-yarn-max-position-embeddings must be positive")

    yarn_factor = factor
    if yarn_factor is None:
        yarn_factor = max_position_embeddings / original_max_position_embeddings
    if yarn_factor <= 1.0:
        raise ValueError("YaRN factor must be greater than 1.0")

    rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
    config.rope_parameters = {
        **rope_parameters,
        "rope_type": "yarn",
        "factor": float(yarn_factor),
        "original_max_position_embeddings": int(original_max_position_embeddings),
    }
    config.max_position_embeddings = int(max_position_embeddings)


def _maybe_configure_yarn_rope(config: Any, args: argparse.Namespace, *, prefix: str) -> bool:
    original = getattr(args, f"{prefix}_yarn_original_max_position_embeddings", None)
    max_positions = getattr(args, f"{prefix}_yarn_max_position_embeddings", None)
    factor = getattr(args, f"{prefix}_yarn_factor", None)
    if original is None and max_positions is None and factor is None:
        return False
    if original is None:
        raise ValueError(f"--{prefix}-yarn-original-max-position-embeddings is required when enabling {prefix} YaRN")
    if max_positions is None:
        max_positions = getattr(config, "max_position_embeddings")
    configure_yarn_rope(
        config,
        max_position_embeddings=max_positions,
        original_max_position_embeddings=original,
        factor=factor,
        error_prefix=prefix,
    )
    return True


def _maybe_configure_draft_yarn_rope(config: Any, args: argparse.Namespace) -> bool:
    return _maybe_configure_yarn_rope(config, args, prefix="draft")


def _maybe_configure_target_yarn_rope(config: Any, args: argparse.Namespace) -> bool:
    return _maybe_configure_yarn_rope(config, args, prefix="target")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _dist_init(torch_dist) -> None:
    if "RANK" not in os.environ:
        warnings.warn("RANK not set. Skipping distributed initialization.")
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _dist_rank() -> int:
    return _env_int("RANK", 0)


def _dist_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(torch_dist, obj: Any, dst: int = 0):
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


_TRANSFORMERS_SUPPORTED_PATTERN = re.compile(r"qwen3(?!\.5)[\w-]*|llama.*3\.1.*8b.*instruct", re.IGNORECASE)


def _check_transformers_model(model_name: str) -> None:
    if not _TRANSFORMERS_SUPPORTED_PATTERN.search(model_name):
        raise ValueError(
            f"Transformers backend does not support '{model_name}'. "
            f"Only Qwen3 series and LLaMA-3.1-8B-Instruct are supported."
        )


def _get_transformers_attn_impl() -> str:
    env_impl = os.environ.get("TRANSFORMERS_ATTN_IMPL")
    if env_impl:
        if env_impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "TRANSFORMERS_ATTN_IMPL=flash_attention_2 requires flash_attn to be installed."
                ) from exc
        return env_impl
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Transformers backend requires flash_attention_2 by default, but flash_attn is not installed. "
            "Install flash-attn or explicitly set TRANSFORMERS_ATTN_IMPL to another implementation."
        ) from exc
    return "flash_attention_2"
