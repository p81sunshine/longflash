import math
import time
import torch
from types import SimpleNamespace
from typing import Callable, Optional
from typing_extensions import Unpack
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from .suffix_decoding import PaperSuffixNode, SuffixMatcher


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _module_dtype(module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration):
        return torch.float32


def _set_attn_implementation(module, attn_impl: Optional[str]) -> Optional[str]:
    config = getattr(module, "config", None)
    if config is None or not hasattr(config, "_attn_implementation"):
        return None
    previous = config._attn_implementation
    if attn_impl:
        config._attn_implementation = attn_impl
    return previous


def _make_tree_attention_mask(
    nodes: list[PaperSuffixNode],
    *,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    query_length = len(nodes)
    key_length = past_length + query_length
    mask = torch.full((1, 1, query_length, key_length), torch.finfo(dtype).min, dtype=dtype, device=device)
    if past_length > 0:
        mask[:, :, :, :past_length] = 0
    for node_idx, node in enumerate(nodes):
        cursor = node_idx
        while cursor >= 0:
            mask[:, :, node_idx, past_length + cursor] = 0
            cursor = nodes[cursor].parent
    return {"full_attention": mask, "sliding_attention": mask}


def _select_cache_sequence(past_key_values, keep_indices: torch.LongTensor) -> None:
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(int(keep_indices.numel()))
        return
    for layer in layers:
        if not getattr(layer, "is_initialized", False):
            continue
        if getattr(layer, "keys", None) is None or layer.keys.numel() == 0:
            continue
        layer.keys = layer.keys.index_select(-2, keep_indices)
        layer.values = layer.values.index_select(-2, keep_indices)


def _follow_suffix_tree(
    nodes: list[PaperSuffixNode],
    posterior: torch.LongTensor,
) -> tuple[list[int], int]:
    children: dict[int, list[int]] = {}
    for idx, node in enumerate(nodes[1:], start=1):
        children.setdefault(node.parent, []).append(idx)

    path = [0]
    cursor = 0
    while True:
        next_token = int(posterior[0, cursor].item())
        next_node = None
        for child_idx in children.get(cursor, ()):
            if nodes[child_idx].token == next_token:
                next_node = child_idx
                break
        if next_node is None:
            return path, next_token
        path.append(next_node)
        cursor = next_node


def _validate_draft_denoise_steps(draft_denoise_steps: int) -> int:
    if draft_denoise_steps < 1:
        raise ValueError(f"draft_denoise_steps must be >= 1, got {draft_denoise_steps}")
    return int(draft_denoise_steps)


def _num_tokens_to_unmask(num_masked: int, remaining_steps: int) -> int:
    if remaining_steps < 1:
        raise ValueError(f"remaining_steps must be >= 1, got {remaining_steps}")
    if num_masked <= 0:
        return 0
    return (num_masked + remaining_steps - 1) // remaining_steps


def _sample_without_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    logits = logits.clone()
    logits[..., token_id] = -torch.inf
    return sample(logits)


def _unmask_most_confident_tokens(
    block_output_ids: torch.LongTensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.LongTensor,
    *,
    mask_token_id: int,
    num_tokens: int,
) -> torch.LongTensor:
    if num_tokens <= 0:
        return block_output_ids

    tail_ids = block_output_ids[:, 1:]
    mask_positions = tail_ids == mask_token_id
    if not mask_positions.any():
        return block_output_ids

    confidence = torch.log_softmax(draft_logits.float(), dim=-1)
    confidence = confidence.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    confidence = confidence.masked_fill(~mask_positions, -torch.inf)
    k = min(num_tokens, tail_ids.shape[1])
    selected = torch.zeros_like(mask_positions)
    selected.scatter_(1, confidence.topk(k, dim=1).indices, True)
    selected &= mask_positions

    updated = block_output_ids.clone()
    updated[:, 1:] = torch.where(selected, draft_tokens, tail_ids)
    return updated


def _make_verify_trace_round(
    *,
    round_idx: int,
    start_pos: int,
    prompt_tokens: int,
    block_size: int,
    draft_logits: torch.Tensor,
    block_output_ids: torch.LongTensor,
    posterior: torch.LongTensor,
    acceptance_length: int,
    verify_draft_tokens: int,
) -> dict[str, object]:
    draft_tokens = block_output_ids[:, 1:]
    target_tokens = posterior[:, :-1]
    log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    selected_logprob = log_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    selected_prob = selected_logprob.exp()
    top_logprob, top_token = log_probs.topk(k=2, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    positions = []
    for idx in range(draft_tokens.shape[1]):
        verified = idx < verify_draft_tokens
        target_token_id = int(target_tokens[0, idx].item()) if verified else None
        draft_token_id = int(draft_tokens[0, idx].item())
        positions.append(
            {
                "position": int(idx + 1),
                "absolute_position": int(start_pos + idx + 1),
                "is_verified": bool(verified),
                "draft_token_id": draft_token_id,
                "target_token_id": target_token_id,
                "is_match_target": (draft_token_id == target_token_id) if verified else None,
                "draft_token_logprob": float(selected_logprob[0, idx].item()),
                "draft_token_prob": float(selected_prob[0, idx].item()),
                "top1_token_id": int(top_token[0, idx, 0].item()),
                "top1_logprob": float(top_logprob[0, idx, 0].item()),
                "top1_prob": float(top_logprob[0, idx, 0].exp().item()),
                "top2_token_id": int(top_token[0, idx, 1].item()),
                "top2_logprob": float(top_logprob[0, idx, 1].item()),
                "top2_prob": float(top_logprob[0, idx, 1].exp().item()),
                "top1_top2_logprob_margin": float((top_logprob[0, idx, 0] - top_logprob[0, idx, 1]).item()),
                "entropy": float(entropy[0, idx].item()),
            }
        )

    return {
        "round_idx": int(round_idx),
        "start_pos": int(start_pos),
        "prompt_tokens": int(prompt_tokens),
        "block_size": int(block_size),
        "verify_draft_tokens": int(verify_draft_tokens),
        "unverified_draft_tokens": int(max(0, draft_tokens.shape[1] - verify_draft_tokens)),
        "accepted_draft_tokens": int(acceptance_length),
        "emitted_tokens": int(acceptance_length + 1),
        "positions": positions,
    }


def _select_verify_draft_tokens_by_confidence(
    draft_logits: torch.Tensor,
    draft_tokens: torch.LongTensor,
    *,
    threshold: float,
    min_draft_tokens: int,
) -> int:
    max_draft_tokens = draft_tokens.shape[1]
    if threshold <= 0:
        return max_draft_tokens

    log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    selected_probs = log_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).exp()
    prefix_len = 0
    for prob in selected_probs[0].tolist():
        if prob < threshold:
            break
        prefix_len += 1
    return max(1, min(max_draft_tokens, max(min_draft_tokens, prefix_len)))


def choose_dynamic_yarn_factor(
    needed_length: int,
    original_max_position_embeddings: int,
    *,
    max_factor: Optional[float] = None,
    mode: str = "continuous",
) -> float:
    if original_max_position_embeddings <= 0:
        raise ValueError(
            "original_max_position_embeddings must be positive, "
            f"got {original_max_position_embeddings}"
        )
    if needed_length <= original_max_position_embeddings:
        return 1.0

    factor = needed_length / original_max_position_embeddings
    if mode == "bucket":
        factor = 2 ** math.ceil(math.log2(factor))
    elif mode != "continuous":
        raise ValueError(f"dynamic_yarn_mode must be 'continuous' or 'bucket', got {mode!r}")

    if max_factor is not None:
        if max_factor < 1.0:
            raise ValueError(f"dynamic_yarn_max_factor must be >= 1.0, got {max_factor}")
        factor = min(factor, max_factor)
    return max(1.0, float(factor))


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _new_profiler_stats() -> dict[str, float | int]:
    return {
        "profiler_first_prefill_time_s": 0.0,
        "profiler_draft_prefill_time_s": 0.0,
        "profiler_draft_generate_time_s": 0.0,
        "profiler_target_verify_time_s": 0.0,
        "profiler_first_prefill_calls": 0,
        "profiler_draft_prefill_calls": 0,
        "profiler_draft_generate_calls": 0,
        "profiler_target_verify_calls": 0,
    }


def prune_target_hidden(
    target_hidden: torch.Tensor,
    *,
    input_ids: Optional[torch.LongTensor] = None,
    sink_tokens: int = 0,
    recent_window: int = 0,
    stride: int = 0,
    suffix_match_tokens: int = 0,
    suffix_keep_tokens: int = 0,
    suffix_source_end: Optional[int] = None,
    middle_budget: int = 0,
    total_budget: Optional[int] = None,
    budget_order: str = "default",
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    seq_length = target_hidden.shape[1]
    sink_tokens = max(0, sink_tokens)
    recent_window = max(0, recent_window)
    stride = max(0, stride)
    suffix_match_tokens = max(0, suffix_match_tokens)
    suffix_keep_tokens = max(0, suffix_keep_tokens)
    middle_budget = max(0, middle_budget)
    suffix_enabled = suffix_match_tokens > 0 and input_ids is not None
    suffix_source_end = seq_length if suffix_source_end is None else max(0, min(suffix_source_end, seq_length))
    if total_budget is not None:
        total_budget = max(0, int(total_budget))
    if budget_order not in {"default", "suffix_then_recent"}:
        raise ValueError(f"budget_order must be 'default' or 'suffix_then_recent', got {budget_order!r}")
    budget_enabled = middle_budget > 0
    sink_end = min(sink_tokens, seq_length)
    recent_start = max(seq_length - recent_window, 0)
    stats: dict[str, object] = {
        "ctx_hidden_tokens_before": int(seq_length),
        "ctx_hidden_tokens_after": int(seq_length),
        "ctx_hidden_pruned_tokens": 0,
        "ctx_suffix_match_count": 0,
        "ctx_suffix_match_positions": [],
        "ctx_suffix_match_kept_tokens": 0,
        "ctx_suffix_source_end": int(suffix_source_end),
        "ctx_middle_budget": int(middle_budget),
        "ctx_total_budget": int(total_budget) if total_budget is not None else 0,
        "ctx_budget_order": budget_order,
        "ctx_recent_tokens_after_budget": 0,
        "ctx_middle_tokens_before_budget": 0,
        "ctx_middle_tokens_after_budget": 0,
        "ctx_middle_budget_dropped_tokens": 0,
    }

    def finish(
        result: torch.Tensor,
        indices: set[int] | None = None,
        middle_indices: set[int] | None = None,
    ):
        kept_tokens = result.shape[1]
        stats["ctx_hidden_tokens_after"] = int(kept_tokens)
        stats["ctx_hidden_pruned_tokens"] = int(seq_length - kept_tokens)
        if middle_indices is not None:
            stats["ctx_middle_tokens_after_budget"] = int(len(middle_indices))
        if indices is not None:
            suffix_positions = stats["ctx_suffix_match_positions"]
            if isinstance(suffix_positions, list) and suffix_positions:
                stats["ctx_suffix_match_kept_tokens"] = int(
                    len((indices & suffix_indices) - base_indices - stride_indices)
                )
        if return_stats:
            return result, stats
        return result

    def collect_suffix_indices(search_end: int) -> set[int]:
        if not suffix_enabled:
            return set()
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("suffix-match pruning requires input_ids with batch size 1")
        if input_ids.shape[1] != seq_length:
            raise ValueError(
                "suffix-match pruning requires input_ids length to match target_hidden sequence length"
            )

        match_stop = min(
            search_end - suffix_match_tokens + 1,
            suffix_source_end - suffix_match_tokens,
        )
        collected: set[int] = set()
        if suffix_match_tokens <= suffix_source_end and sink_end < match_stop:
            token_ids = input_ids[0].tolist()
            suffix = token_ids[suffix_source_end - suffix_match_tokens : suffix_source_end]
            match_positions: list[int] = []
            for pos in range(sink_end, match_stop):
                if token_ids[pos : pos + suffix_match_tokens] == suffix:
                    match_positions.append(pos)
                    keep_end = min(pos + suffix_match_tokens + suffix_keep_tokens, seq_length)
                    collected.update(range(pos, keep_end))
            stats["ctx_suffix_match_count"] = len(match_positions)
            stats["ctx_suffix_match_positions"] = match_positions
        return collected

    if total_budget is not None and budget_order == "suffix_then_recent":
        base_indices = set(range(sink_end))
        stride_indices: set[int] = set()
        remaining_budget = max(0, min(total_budget, seq_length) - len(base_indices))
        suffix_indices = {idx for idx in collect_suffix_indices(seq_length) if idx >= sink_end}
        stats["ctx_middle_tokens_before_budget"] = int(len(suffix_indices))
        if len(suffix_indices) > remaining_budget:
            suffix_indices = set(sorted(suffix_indices)[:remaining_budget])
        stats["ctx_middle_tokens_after_budget"] = int(len(suffix_indices))
        stats["ctx_middle_budget_dropped_tokens"] = int(
            stats["ctx_middle_tokens_before_budget"] - len(suffix_indices)
        )
        remaining_budget -= len(suffix_indices)

        recent_indices: set[int] = set()
        if remaining_budget > 0:
            for idx in range(seq_length - 1, sink_end - 1, -1):
                if idx in suffix_indices:
                    continue
                recent_indices.add(idx)
                if len(recent_indices) >= remaining_budget:
                    break
        stats["ctx_recent_tokens_after_budget"] = int(len(recent_indices))

        indices = base_indices | suffix_indices | recent_indices
        if len(indices) >= seq_length:
            return finish(target_hidden, indices, suffix_indices)
        if not indices:
            return finish(target_hidden, indices, suffix_indices)
        idx = torch.tensor(sorted(indices), dtype=torch.long, device=target_hidden.device)
        return finish(target_hidden.index_select(1, idx), indices, suffix_indices)

    if not suffix_enabled and not budget_enabled and (
        sink_tokens + recent_window <= 0 or sink_tokens + recent_window >= seq_length
    ):
        return finish(target_hidden)

    if not suffix_enabled and not budget_enabled and recent_start <= sink_end:
        return finish(target_hidden)

    base_indices = set(range(sink_end))
    base_indices.update(range(recent_start, seq_length))
    stride_indices: set[int] = set()
    suffix_indices: set[int] = set()
    if stride >= 1 and recent_start > sink_end:
        stride_indices.update(range(sink_end, recent_start, stride))

    if suffix_enabled:
        suffix_indices = collect_suffix_indices(recent_start)

    middle_indices = stride_indices | suffix_indices
    middle_indices = {idx for idx in middle_indices if sink_end <= idx < recent_start}
    stats["ctx_middle_tokens_before_budget"] = int(len(middle_indices))
    if budget_enabled and len(middle_indices) > middle_budget:
        middle_indices = set(sorted(middle_indices)[:middle_budget])
    stats["ctx_middle_budget_dropped_tokens"] = int(
        stats["ctx_middle_tokens_before_budget"] - len(middle_indices)
    )
    indices = base_indices | middle_indices

    if len(indices) >= seq_length:
        return finish(target_hidden, indices, middle_indices)

    if not indices:
        return finish(target_hidden, indices, middle_indices)

    idx = torch.tensor(sorted(indices), dtype=torch.long, device=target_hidden.device)
    return finish(target_hidden.index_select(1, idx), indices, middle_indices)


def select_indexed_target_hidden(
    target_hidden: torch.Tensor,
    *,
    block_size: int = 4,
    top_k_blocks: int = 512,
    query_tokens: int = 512,
    sink_tokens: int = 0,
    recent_window: int = 0,
    score_reduce: str = "max",
    return_stats: bool = False,
) -> tuple[torch.Tensor, torch.LongTensor] | tuple[torch.Tensor, torch.LongTensor, dict[str, object]]:
    if target_hidden.dim() != 3:
        raise ValueError(f"target_hidden must have shape [B, T, D], got {tuple(target_hidden.shape)}")
    if score_reduce not in {"max", "mean"}:
        raise ValueError(f"score_reduce must be 'max' or 'mean', got {score_reduce!r}")

    bsz, seq_len, hidden_size = target_hidden.shape
    device = target_hidden.device
    block_size = max(1, int(block_size))
    top_k_blocks = max(0, int(top_k_blocks))
    query_tokens = max(1, int(query_tokens))
    sink_tokens = max(0, int(sink_tokens))
    recent_window = max(0, int(recent_window))
    positions = torch.arange(seq_len, device=device).view(1, -1).expand(bsz, -1)
    stats: dict[str, object] = {
        "ctx_hidden_tokens_before": int(seq_len),
        "ctx_hidden_tokens_after": int(seq_len),
        "ctx_hidden_pruned_tokens": 0,
        "ctx_indexer_enabled": True,
        "ctx_indexer_block_size": int(block_size),
        "ctx_indexer_top_k_blocks": int(top_k_blocks),
        "ctx_indexer_query_tokens": int(query_tokens),
        "ctx_indexer_score_reduce": score_reduce,
        "ctx_indexer_num_blocks": 0,
        "ctx_indexer_selected_blocks": 0,
        "ctx_indexer_forced_blocks": 0,
        "ctx_indexer_sink_tokens": int(sink_tokens),
        "ctx_indexer_recent_window": int(recent_window),
    }
    if seq_len == 0:
        if return_stats:
            return target_hidden, positions, stats
        return target_hidden, positions

    num_blocks = (seq_len + block_size - 1) // block_size
    pad_tokens = num_blocks * block_size - seq_len
    padded_hidden = torch.nn.functional.pad(target_hidden, (0, 0, 0, pad_tokens)) if pad_tokens else target_hidden
    blocks = padded_hidden.view(bsz, num_blocks, block_size, hidden_size)
    block_token_mask = torch.ones((bsz, num_blocks, block_size), dtype=torch.bool, device=device)
    if pad_tokens:
        block_token_mask[:, -1, -pad_tokens:] = False
    weights = block_token_mask.unsqueeze(-1).to(dtype=blocks.dtype)
    compressed = (blocks * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

    query_start = max(seq_len - query_tokens, 0)
    query_hidden = target_hidden[:, query_start:, :]
    q = torch.nn.functional.normalize(query_hidden.float(), dim=-1)
    k = torch.nn.functional.normalize(compressed.float(), dim=-1)
    scores = torch.einsum("bqd,bnd->bqn", q, k)
    if score_reduce == "max":
        block_scores = scores.max(dim=1).values
    else:
        block_scores = scores.mean(dim=1)

    selected_blocks = torch.zeros((bsz, num_blocks), dtype=torch.bool, device=device)
    if top_k_blocks > 0:
        k_blocks = min(top_k_blocks, num_blocks)
        selected_blocks.scatter_(1, block_scores.topk(k_blocks, dim=1).indices, True)

    forced_blocks = torch.zeros_like(selected_blocks)
    if sink_tokens > 0:
        forced_blocks[:, : min(num_blocks, (min(sink_tokens, seq_len) + block_size - 1) // block_size)] = True
    if recent_window > 0:
        recent_start = max(seq_len - recent_window, 0)
        forced_blocks[:, recent_start // block_size :] = True
    selected_blocks |= forced_blocks

    token_mask = selected_blocks.repeat_interleave(block_size, dim=1)[:, :seq_len]
    if not token_mask.any(dim=1).all():
        empty_rows = (~token_mask.any(dim=1)).nonzero(as_tuple=True)[0]
        token_mask[empty_rows, seq_len - 1] = True

    counts = token_mask.sum(dim=1)
    max_count = int(counts.max().item())
    selected_positions = torch.zeros((bsz, max_count), dtype=torch.long, device=device)
    selected_hidden = target_hidden.new_zeros((bsz, max_count, hidden_size))
    for batch_idx in range(bsz):
        idx = token_mask[batch_idx].nonzero(as_tuple=True)[0]
        selected_positions[batch_idx, : idx.numel()] = idx
        selected_hidden[batch_idx, : idx.numel()] = target_hidden[batch_idx].index_select(0, idx)

    stats["ctx_hidden_tokens_after"] = int(max_count)
    stats["ctx_hidden_pruned_tokens"] = int(seq_len - max_count)
    stats["ctx_indexer_num_blocks"] = int(num_blocks)
    stats["ctx_indexer_selected_blocks"] = int(selected_blocks.sum(dim=1).max().item())
    stats["ctx_indexer_forced_blocks"] = int(forced_blocks.sum(dim=1).max().item())
    if return_stats:
        return selected_hidden, selected_positions, stats
    return selected_hidden, selected_positions


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
    ctx_sink_tokens: int = 0,
    ctx_recent_window: int = 0,
    ctx_stride: int = 0,
    ctx_suffix_match_tokens: int = 0,
    ctx_suffix_keep_tokens: int = 0,
    ctx_suffix_source_end: Optional[int] = None,
    ctx_middle_budget: int = 0,
    ctx_total_budget: Optional[int] = None,
    ctx_dynamic_budget_ratio: Optional[float] = None,
    ctx_budget_order: str = "default",
    ctx_indexer_enable: bool = False,
    ctx_indexer_block_size: int = 4,
    ctx_indexer_top_k_blocks: int = 512,
    ctx_indexer_query_tokens: int = 512,
    ctx_indexer_score_reduce: str = "max",
    draft_denoise_steps: int = 1,
    return_verify_trace: bool = False,
    verify_trace_max_rounds: int = 0,
    verify_confidence_threshold: float = 0.0,
    verify_min_draft_tokens: int = 1,
    draft_dynamic_yarn_original_max_position_embeddings: Optional[int] = None,
    draft_dynamic_yarn_max_factor: Optional[float] = None,
    draft_dynamic_yarn_mode: str = "continuous",
    draft_dynamic_yarn_length_ratio: Optional[float] = None,
    profiler: bool = False,
    suffix_decoding: bool = False,
    suffix_strategy: str = "consensus",
    suffix_max_query_len: int = 16,
    suffix_min_query_len: int = 2,
    suffix_top_k: int = 4,
    suffix_min_support: int = 3,
    suffix_min_predict_len: int = 8,
    suffix_max_predict_len: Optional[int] = None,
    suffix_paper_alpha: float = 1.0,
    suffix_paper_max_spec_offset: float = 0.0,
    suffix_paper_min_token_prob: float = 0.0,
    suffix_paper_threshold: float = 0.0,
    suffix_paper_max_matches: int = 0,
    suffix_paper_verifier: str = "linear",
    suffix_paper_tree_attn_impl: str = "sdpa",
    suffix_fallback: str = "dflash",
    return_suffix_trace: bool = False,
):
    draft_denoise_steps = _validate_draft_denoise_steps(draft_denoise_steps)
    verify_confidence_threshold = max(0.0, verify_confidence_threshold)
    verify_min_draft_tokens = max(1, verify_min_draft_tokens)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id
    suffix_decoding_enabled = bool(suffix_decoding and block_size > 1)
    if suffix_strategy not in {"consensus", "paper"}:
        raise ValueError(f"suffix_strategy must be 'consensus' or 'paper', got {suffix_strategy!r}")
    if suffix_paper_verifier not in {"linear", "tree"}:
        raise ValueError(
            f"suffix_paper_verifier must be 'linear' or 'tree', got {suffix_paper_verifier!r}"
        )
    if suffix_fallback not in {"dflash", "target", "none"}:
        raise ValueError(
            f"suffix_fallback must be 'dflash', 'target', or 'none', got {suffix_fallback!r}"
        )
    if suffix_max_predict_len is None:
        suffix_max_predict_len = max(0, block_size - 1)
    else:
        suffix_max_predict_len = max(0, int(suffix_max_predict_len))
    suffix_max_query_len = max(1, int(suffix_max_query_len))
    suffix_min_query_len = max(1, int(suffix_min_query_len))
    suffix_top_k = max(1, int(suffix_top_k))
    suffix_min_support = max(1, int(suffix_min_support))
    suffix_min_predict_len = max(1, int(suffix_min_predict_len))
    suffix_paper_alpha = max(0.0, float(suffix_paper_alpha))
    suffix_paper_max_spec_offset = float(suffix_paper_max_spec_offset)
    suffix_paper_min_token_prob = max(0.0, float(suffix_paper_min_token_prob))
    suffix_paper_threshold = float(suffix_paper_threshold)
    suffix_paper_max_matches = max(0, int(suffix_paper_max_matches))
    draft_yarn_factor = None
    dynamic_request_length = num_input_tokens + max_new_tokens
    if block_size > 1 and draft_dynamic_yarn_original_max_position_embeddings is not None:
        if draft_dynamic_yarn_length_ratio is None:
            dynamic_needed_length = dynamic_request_length + block_size
        else:
            if draft_dynamic_yarn_length_ratio <= 0:
                raise ValueError(
                    "draft_dynamic_yarn_length_ratio must be positive, "
                    f"got {draft_dynamic_yarn_length_ratio}"
                )
            dynamic_needed_length = max(1, int(math.ceil(dynamic_request_length * draft_dynamic_yarn_length_ratio)))
        draft_yarn_factor = choose_dynamic_yarn_factor(
            dynamic_needed_length,
            draft_dynamic_yarn_original_max_position_embeddings,
            max_factor=draft_dynamic_yarn_max_factor,
            mode=draft_dynamic_yarn_mode,
        )
        model.set_yarn_factor(
            draft_yarn_factor,
            original_max_position_embeddings=draft_dynamic_yarn_original_max_position_embeddings,
        )

    if ctx_total_budget is not None and ctx_dynamic_budget_ratio is not None:
        raise ValueError("ctx_total_budget and ctx_dynamic_budget_ratio are mutually exclusive")
    if ctx_total_budget is not None:
        ctx_total_budget = max(0, int(ctx_total_budget))
    elif ctx_dynamic_budget_ratio is not None:
        if ctx_dynamic_budget_ratio <= 0:
            raise ValueError(f"ctx_dynamic_budget_ratio must be positive, got {ctx_dynamic_budget_ratio}")
        ctx_total_budget = max(0, int(math.floor(dynamic_request_length * ctx_dynamic_budget_ratio)))

    output_extra = max(block_size, suffix_max_predict_len + 1 if suffix_decoding_enabled else block_size)
    output_ids = torch.full(
        (1, max_length + output_extra), mask_token_id, dtype=torch.long, device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    profiler_enabled = bool(profiler and return_stats)
    profiler_stats = _new_profiler_stats() if profiler_enabled else None

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    suffix_matcher = None
    if suffix_decoding_enabled:
        suffix_matcher = SuffixMatcher()
        suffix_matcher.extend(input_ids[0].tolist())
        suffix_matcher.extend([int(output_ids[0, num_input_tokens].item())])
    ctx_prune_stats = None
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        if ctx_indexer_enable:
            target_hidden, _selected_positions, indexer_stats = select_indexed_target_hidden(
                target_hidden,
                block_size=ctx_indexer_block_size,
                top_k_blocks=ctx_indexer_top_k_blocks,
                query_tokens=ctx_indexer_query_tokens,
                sink_tokens=ctx_sink_tokens,
                recent_window=ctx_recent_window,
                score_reduce=ctx_indexer_score_reduce,
                return_stats=True,
            )
            ctx_prune_stats = indexer_stats if return_stats else None
        elif return_stats:
            target_hidden, ctx_prune_stats = prune_target_hidden(
                target_hidden,
                input_ids=input_ids,
                sink_tokens=ctx_sink_tokens,
                recent_window=ctx_recent_window,
                stride=ctx_stride,
                suffix_match_tokens=ctx_suffix_match_tokens,
                suffix_keep_tokens=ctx_suffix_keep_tokens,
                suffix_source_end=ctx_suffix_source_end,
                middle_budget=ctx_middle_budget,
                total_budget=ctx_total_budget,
                budget_order=ctx_budget_order,
                return_stats=True,
            )
        else:
            target_hidden = prune_target_hidden(
                target_hidden,
                input_ids=input_ids,
                sink_tokens=ctx_sink_tokens,
                recent_window=ctx_recent_window,
                stride=ctx_stride,
                suffix_match_tokens=ctx_suffix_match_tokens,
                suffix_keep_tokens=ctx_suffix_keep_tokens,
                suffix_source_end=ctx_suffix_source_end,
                middle_budget=ctx_middle_budget,
                total_budget=ctx_total_budget,
                budget_order=ctx_budget_order,
            )
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None
    if profiler_enabled:
        profiler_stats["profiler_first_prefill_time_s"] = time_to_first_token
        profiler_stats["profiler_first_prefill_calls"] = 1

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    verify_draft_lengths = []
    verify_trace = []
    suffix_match_rounds = 0
    suffix_verify_rounds = 0
    suffix_recovery_rounds = 0
    suffix_zero_accept_rounds = 0
    suffix_exhausted_rounds = 0
    suffix_pred_lengths = []
    suffix_supports = []
    suffix_query_lengths = []
    suffix_acceptance_lengths = []
    suffix_paper_scores = []
    suffix_paper_token_scores = []
    suffix_paper_tree_sizes = []
    suffix_paper_best_path_scores = []
    suffix_paper_max_specs = []
    suffix_trace = []
    start = num_input_tokens
    draft_seq_pos = target_hidden.shape[1] if block_size > 1 else 0
    draft_prefill = True
    draft_forward_passes = 0
    decode_start_reset_allowed = True

    def _append_pending_hidden(
        pending_hidden: torch.Tensor,
        new_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if pending_hidden.shape[1] == 0:
            return new_hidden
        if new_hidden.shape[1] == 0:
            return pending_hidden
        return torch.cat([pending_hidden, new_hidden], dim=1)

    def _commit_verified_round(
        *,
        verify_output_ids: torch.LongTensor,
        posterior: torch.LongTensor,
        verifier_output,
        acceptance_length: int,
        append_hidden: bool,
    ) -> tuple[int, int]:
        nonlocal start, target_hidden, draft_seq_pos

        emitted_length = acceptance_length + 1
        output_ids[:, start : start + emitted_length] = verify_output_ids[:, :emitted_length]
        output_ids[:, start + emitted_length] = posterior[:, acceptance_length]

        if suffix_matcher is not None:
            committed_tokens = verify_output_ids[0, 1:emitted_length].tolist()
            committed_tokens.append(int(posterior[0, acceptance_length].item()))
            suffix_matcher.extend(committed_tokens)

        start += emitted_length
        past_key_values_target.crop(start)
        acceptance_lengths.append(emitted_length)

        if block_size > 1:
            draft_seq_pos += emitted_length
            new_hidden = extract_context_feature(verifier_output.hidden_states, model.target_layer_ids)[
                :, :emitted_length, :
            ]
            if append_hidden:
                target_hidden = _append_pending_hidden(target_hidden, new_hidden)
            else:
                target_hidden = new_hidden

        return emitted_length, int(posterior[0, acceptance_length].item())

    def _commit_verified_tree_round(
        *,
        verify_output_ids: torch.LongTensor,
        posterior: torch.LongTensor,
        verifier_output,
        path_indices: list[int],
        append_hidden: bool,
    ) -> tuple[int, int]:
        nonlocal start, target_hidden, draft_seq_pos

        path_index_tensor = torch.tensor(path_indices, dtype=torch.long, device=target.device)
        emitted_length = len(path_indices)
        path_output_ids = verify_output_ids.index_select(1, path_index_tensor)
        bonus_token = int(posterior[0, path_indices[-1]].item())

        output_ids[:, start : start + emitted_length] = path_output_ids
        output_ids[:, start + emitted_length] = bonus_token

        if suffix_matcher is not None:
            committed_tokens = path_output_ids[0, 1:].tolist()
            committed_tokens.append(bonus_token)
            suffix_matcher.extend(committed_tokens)

        old_start = start
        start += emitted_length
        keep_indices = torch.cat(
            [
                torch.arange(old_start, device=target.device, dtype=torch.long),
                old_start + path_index_tensor,
            ]
        )
        _select_cache_sequence(past_key_values_target, keep_indices)
        acceptance_lengths.append(emitted_length)

        if block_size > 1:
            draft_seq_pos += emitted_length
            selected_hidden = extract_context_feature(verifier_output.hidden_states, model.target_layer_ids)
            new_hidden = selected_hidden.index_select(1, path_index_tensor)
            if append_hidden:
                target_hidden = _append_pending_hidden(target_hidden, new_hidden)
            else:
                target_hidden = new_hidden

        return emitted_length, bonus_token

    def _run_dflash_round() -> tuple[int, int]:
        nonlocal decode_start, decode_start_reset_allowed, draft_forward_passes, draft_prefill

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        trace_draft_logits = None
        remaining_output_tokens = max(1, max_length - start)
        max_verify_draft_tokens = max(0, min(block_size - 1, remaining_output_tokens - 1))
        verify_draft_tokens = max_verify_draft_tokens
        if block_size > 1:
            draft_stage_start = _cuda_time() if profiler_enabled else None
            is_draft_prefill_stage = draft_prefill
            ctx_len = target_hidden.shape[1]
            draft_pos_ids = torch.arange(
                draft_seq_pos - ctx_len,
                draft_seq_pos + block_size,
                device=target.device,
            ).unsqueeze(0)

            if draft_denoise_steps == 1:
                noise_embedding = target.model.embed_tokens(block_output_ids)
                draft_logits = target.lm_head(model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=draft_pos_ids,
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, 1 - block_size :, :])
                past_key_values_draft.crop(draft_seq_pos)
                draft_forward_passes += 1
                block_output_ids[:, 1:] = sample(draft_logits)
                trace_draft_logits = draft_logits
                if draft_prefill and return_stats:
                    draft_prefill = False
                    if decode_start_reset_allowed:
                        decode_start = _cuda_time()
            else:
                block_output_ids[:, 1:] = mask_token_id
                for denoise_step in range(draft_denoise_steps):
                    num_masked = int((block_output_ids[:, 1:] == mask_token_id).sum(dim=1).max().item())
                    if num_masked == 0:
                        break

                    noise_embedding = target.model.embed_tokens(block_output_ids)
                    draft_logits = target.lm_head(model(
                        target_hidden=target_hidden,
                        noise_embedding=noise_embedding,
                        position_ids=draft_pos_ids,
                        past_key_values=past_key_values_draft,
                        use_cache=True,
                        is_causal=False,
                    )[:, 1 - block_size :, :])
                    past_key_values_draft.crop(draft_seq_pos)
                    draft_forward_passes += 1
                    trace_draft_logits = draft_logits
                    if draft_prefill and return_stats:
                        draft_prefill = False
                        if decode_start_reset_allowed:
                            decode_start = _cuda_time()

                    remaining_steps = draft_denoise_steps - denoise_step
                    num_transfer = _num_tokens_to_unmask(num_masked, remaining_steps)
                    draft_tokens = _sample_without_token(draft_logits, mask_token_id)
                    block_output_ids = _unmask_most_confident_tokens(
                        block_output_ids,
                        draft_logits,
                        draft_tokens,
                        mask_token_id=mask_token_id,
                        num_tokens=num_transfer,
                    )

            verify_draft_tokens = _select_verify_draft_tokens_by_confidence(
                trace_draft_logits,
                block_output_ids[:, 1:],
                threshold=verify_confidence_threshold,
                min_draft_tokens=verify_min_draft_tokens,
            )
            verify_draft_tokens = min(verify_draft_tokens, max_verify_draft_tokens)
            if profiler_enabled:
                draft_stage_time = _cuda_time() - draft_stage_start
                if is_draft_prefill_stage:
                    profiler_stats["profiler_draft_prefill_time_s"] += draft_stage_time
                    profiler_stats["profiler_draft_prefill_calls"] += 1
                else:
                    profiler_stats["profiler_draft_generate_time_s"] += draft_stage_time
                    profiler_stats["profiler_draft_generate_calls"] += 1

        verify_length = verify_draft_tokens + 1 if block_size > 1 else block_size
        verify_output_ids = block_output_ids[:, :verify_length]
        verify_position_ids = block_position_ids[:, :verify_length]

        verify_stage_start = _cuda_time() if profiler_enabled else None
        verifier_output = target(
            verify_output_ids,
            position_ids=verify_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        if profiler_enabled:
            profiler_stats["profiler_target_verify_time_s"] += _cuda_time() - verify_stage_start
            profiler_stats["profiler_target_verify_calls"] += 1

        posterior = sample(verifier_output.logits, temperature)
        acceptance_length = (verify_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        if (
            return_verify_trace
            and block_size > 1
            and trace_draft_logits is not None
            and (verify_trace_max_rounds <= 0 or len(verify_trace) < verify_trace_max_rounds)
        ):
            verify_trace.append(
                _make_verify_trace_round(
                    round_idx=len(acceptance_lengths),
                    start_pos=start,
                    prompt_tokens=num_input_tokens,
                    block_size=block_size,
                    draft_logits=trace_draft_logits,
                    block_output_ids=block_output_ids,
                    posterior=posterior,
                    acceptance_length=acceptance_length,
                    verify_draft_tokens=verify_draft_tokens,
                )
            )

        emitted_length, bonus_token = _commit_verified_round(
            verify_output_ids=verify_output_ids,
            posterior=posterior,
            verifier_output=verifier_output,
            acceptance_length=int(acceptance_length),
            append_hidden=False,
        )
        if block_size > 1:
            verify_draft_lengths.append(verify_draft_tokens)
        return emitted_length, bonus_token

    def _run_target_round() -> tuple[int, int]:
        nonlocal decode_start_reset_allowed

        decode_start_reset_allowed = False
        verify_output_ids = output_ids[:, start : start + 1]
        verify_position_ids = position_ids[:, start : start + 1]

        verify_stage_start = _cuda_time() if profiler_enabled else None
        verifier_output = target(
            verify_output_ids,
            position_ids=verify_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        if profiler_enabled:
            profiler_stats["profiler_target_verify_time_s"] += _cuda_time() - verify_stage_start
            profiler_stats["profiler_target_verify_calls"] += 1

        posterior = sample(verifier_output.logits, temperature)
        return _commit_verified_round(
            verify_output_ids=verify_output_ids,
            posterior=posterior,
            verifier_output=verifier_output,
            acceptance_length=0,
            append_hidden=True,
        )

    while start < max_length:
        trace_record = None
        use_suffix_path = False
        if suffix_matcher is not None:
            if suffix_strategy == "paper":
                prediction = suffix_matcher.predict_paper(
                    max_predict_len=suffix_max_predict_len,
                    max_query_len=suffix_max_query_len,
                    min_query_len=suffix_min_query_len,
                    alpha=suffix_paper_alpha,
                    max_spec_offset=suffix_paper_max_spec_offset,
                    min_token_prob=suffix_paper_min_token_prob,
                    max_match_count=suffix_paper_max_matches,
                )
                suffix_paper_scores.append(float(prediction.score))
                suffix_paper_token_scores.append(float(prediction.token_score))
                suffix_paper_tree_sizes.append(int(prediction.tree_size))
                suffix_paper_best_path_scores.append(float(prediction.best_path_score))
                suffix_paper_max_specs.append(int(prediction.max_spec))
                use_suffix_path = prediction.is_high_confidence(threshold=suffix_paper_threshold)
            else:
                prediction = suffix_matcher.predict(
                    max_predict_len=suffix_max_predict_len,
                    max_query_len=suffix_max_query_len,
                    min_query_len=suffix_min_query_len,
                    top_k=suffix_top_k,
                )
                use_suffix_path = prediction.is_high_confidence(
                    min_support=suffix_min_support,
                    min_predict_len=suffix_min_predict_len,
                )
            suffix_pred_lengths.append(len(prediction.tokens))
            suffix_supports.append(prediction.support)
            suffix_query_lengths.append(prediction.query_len)
            if prediction.support > 0:
                suffix_match_rounds += 1
            if return_suffix_trace:
                trace_record = {
                    "round_idx": len(acceptance_lengths),
                    "start_pos": int(start),
                    "prediction": [int(token_id) for token_id in prediction.tokens],
                    "prediction_length": len(prediction.tokens),
                    "support": int(prediction.support),
                    "query_len": int(prediction.query_len),
                    "match_positions": [int(pos) for pos in prediction.match_positions],
                    "strategy": suffix_strategy,
                    "route": "suffix" if use_suffix_path else "dflash",
                }
                if suffix_strategy == "paper":
                    trace_record.update(
                        paper_score=float(prediction.score),
                        paper_token_score=float(prediction.token_score),
                        paper_tree_size=int(prediction.tree_size),
                        paper_best_path_score=float(prediction.best_path_score),
                        paper_max_spec=int(prediction.max_spec),
                        paper_verifier=suffix_paper_verifier,
                    )
        else:
            prediction = None

        if use_suffix_path and prediction is not None:
            suffix_verify_rounds += 1
            decode_start_reset_allowed = False
            suffix_room = max(0, output_ids.shape[1] - start - 1)
            root_token_id = int(output_ids[0, start].item())

            if (
                suffix_strategy == "paper"
                and suffix_paper_verifier == "tree"
                and getattr(prediction, "nodes", None)
            ):
                tree_spec_len = min(suffix_max_predict_len, suffix_room, max(0, len(prediction.nodes) - 1))
                tree_nodes = list(prediction.nodes[: tree_spec_len + 1])
                verify_output_ids = torch.tensor(
                    [[root_token_id] + [int(node.token) for node in tree_nodes[1:]]],
                    dtype=torch.long,
                    device=target.device,
                )
                verify_position_ids = torch.tensor(
                    [[start + int(node.depth) for node in tree_nodes]],
                    dtype=torch.long,
                    device=target.device,
                )
                tree_attention_mask = _make_tree_attention_mask(
                    tree_nodes,
                    past_length=start,
                    dtype=_module_dtype(target),
                    device=target.device,
                )

                verify_stage_start = _cuda_time() if profiler_enabled else None
                previous_attn_impl = _set_attn_implementation(target, suffix_paper_tree_attn_impl)
                try:
                    verifier_output = target(
                        verify_output_ids,
                        attention_mask=tree_attention_mask,
                        position_ids=verify_position_ids,
                        past_key_values=past_key_values_target,
                        use_cache=True,
                        output_hidden_states=block_size > 1,
                    )
                finally:
                    if previous_attn_impl is not None:
                        _set_attn_implementation(target, previous_attn_impl)
                if profiler_enabled:
                    profiler_stats["profiler_target_verify_time_s"] += _cuda_time() - verify_stage_start
                    profiler_stats["profiler_target_verify_calls"] += 1

                posterior = sample(verifier_output.logits, temperature)
                path_indices, _tree_bonus = _follow_suffix_tree(tree_nodes, posterior)
                suffix_acceptance_length = len(path_indices) - 1
            else:
                suffix_tokens = prediction.tokens[: min(suffix_max_predict_len, suffix_room)]
                verify_output_ids = torch.tensor(
                    [[root_token_id] + suffix_tokens],
                    dtype=torch.long,
                    device=target.device,
                )
                verify_position_ids = position_ids[:, start : start + verify_output_ids.shape[1]]

                verify_stage_start = _cuda_time() if profiler_enabled else None
                verifier_output = target(
                    verify_output_ids,
                    position_ids=verify_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=block_size > 1,
                )
                if profiler_enabled:
                    profiler_stats["profiler_target_verify_time_s"] += _cuda_time() - verify_stage_start
                    profiler_stats["profiler_target_verify_calls"] += 1

                posterior = sample(verifier_output.logits, temperature)
                suffix_acceptance_length = int(
                    (verify_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
                )
                path_indices = []
            suffix_acceptance_lengths.append(suffix_acceptance_length)

            if suffix_acceptance_length > 0:
                if path_indices:
                    _commit_verified_tree_round(
                        verify_output_ids=verify_output_ids,
                        posterior=posterior,
                        verifier_output=verifier_output,
                        path_indices=path_indices,
                        append_hidden=True,
                    )
                else:
                    _commit_verified_round(
                        verify_output_ids=verify_output_ids,
                        posterior=posterior,
                        verifier_output=verifier_output,
                        acceptance_length=suffix_acceptance_length,
                        append_hidden=True,
                    )
                if trace_record is not None:
                    trace_record["route"] = "suffix"
                    trace_record["accepted_suffix_tokens"] = suffix_acceptance_length
            else:
                suffix_zero_accept_rounds += 1
                if path_indices:
                    _commit_verified_tree_round(
                        verify_output_ids=verify_output_ids,
                        posterior=posterior,
                        verifier_output=verifier_output,
                        path_indices=path_indices,
                        append_hidden=True,
                    )
                else:
                    _commit_verified_round(
                        verify_output_ids=verify_output_ids,
                        posterior=posterior,
                        verifier_output=verifier_output,
                        acceptance_length=0,
                        append_hidden=True,
                    )
                if trace_record is not None:
                    trace_record["route"] = "suffix_no_recovery"
                    trace_record["accepted_suffix_tokens"] = 0
        else:
            if suffix_decoding_enabled and suffix_fallback == "none":
                suffix_exhausted_rounds += 1
                if trace_record is not None:
                    trace_record["route"] = "suffix_exhausted"
                    trace_record["accepted_suffix_tokens"] = 0
                    trace_record["committed_length"] = 0
                    suffix_trace.append(trace_record)
                break
            elif suffix_decoding_enabled and suffix_fallback == "target":
                suffix_exhausted_rounds += 1
                if trace_record is not None:
                    trace_record["route"] = "target"
                    trace_record["accepted_suffix_tokens"] = 0
                _run_target_round()
            else:
                _run_dflash_round()

        if trace_record is not None:
            trace_record["committed_length"] = int(acceptance_lengths[-1])
            suffix_trace.append(trace_record)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        ctx_prune_stats=ctx_prune_stats,
        draft_denoise_steps=draft_denoise_steps,
        draft_forward_passes=draft_forward_passes,
        verify_confidence_threshold=verify_confidence_threshold,
        verify_min_draft_tokens=verify_min_draft_tokens,
        verify_draft_lengths=verify_draft_lengths,
        verify_trace=verify_trace if return_verify_trace else None,
        draft_yarn_factor=draft_yarn_factor,
        profiler_stats=profiler_stats,
        suffix_decoding_enabled=suffix_decoding_enabled,
        suffix_match_rounds=suffix_match_rounds,
        suffix_verify_rounds=suffix_verify_rounds,
        suffix_recovery_rounds=suffix_recovery_rounds,
        suffix_zero_accept_rounds=suffix_zero_accept_rounds,
        suffix_exhausted_rounds=suffix_exhausted_rounds,
        suffix_pred_lengths=suffix_pred_lengths,
        suffix_supports=suffix_supports,
        suffix_query_lengths=suffix_query_lengths,
        suffix_acceptance_lengths=suffix_acceptance_lengths,
        suffix_strategy=suffix_strategy if suffix_decoding_enabled else None,
        suffix_fallback=suffix_fallback if suffix_decoding_enabled else None,
        suffix_paper_alpha=suffix_paper_alpha,
        suffix_paper_max_spec_offset=suffix_paper_max_spec_offset,
        suffix_paper_min_token_prob=suffix_paper_min_token_prob,
        suffix_paper_threshold=suffix_paper_threshold,
        suffix_paper_max_matches=suffix_paper_max_matches,
        suffix_paper_verifier=suffix_paper_verifier,
        suffix_paper_tree_attn_impl=suffix_paper_tree_attn_impl,
        suffix_paper_scores=suffix_paper_scores,
        suffix_paper_token_scores=suffix_paper_token_scores,
        suffix_paper_tree_sizes=suffix_paper_tree_sizes,
        suffix_paper_best_path_scores=suffix_paper_best_path_scores,
        suffix_paper_max_specs=suffix_paper_max_specs,
        suffix_trace=suffix_trace if return_suffix_trace else None,
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def set_yarn_factor(
        self,
        factor: float,
        *,
        original_max_position_embeddings: int,
    ) -> None:
        """Update draft RoPE frequencies before starting a generation."""
        if factor < 1.0:
            raise ValueError(f"YaRN factor must be >= 1.0, got {factor}")
        if original_max_position_embeddings <= 0:
            raise ValueError(
                "original_max_position_embeddings must be positive, "
                f"got {original_max_position_embeddings}"
            )

        rotary = self.rotary_emb
        device = rotary.inv_freq.device
        if factor <= 1.0 + 1e-8:
            rotary.rope_type = "default"
            rotary.attention_scaling = 1.0
            default_inv_freq = rotary.original_inv_freq.to(device)
            rotary.register_buffer("inv_freq", default_inv_freq, persistent=False)
            self.config.max_position_embeddings = int(original_max_position_embeddings)
            self.config.rope_parameters = {
                **dict(getattr(self.config, "rope_parameters", None) or {}),
                "rope_type": "default",
            }
            return

        rope_parameters = dict(getattr(self.config, "rope_parameters", None) or {})
        self.config.rope_parameters = {
            **rope_parameters,
            "rope_type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(original_max_position_embeddings),
        }
        self.config.max_position_embeddings = max(
            int(math.ceil(original_max_position_embeddings * factor)),
            int(original_max_position_embeddings),
        )
        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["yarn"](self.config, device=device)
        rotary.rope_type = "yarn"
        rotary.attention_scaling = attention_scaling
        rotary.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        ctx_sink_tokens: int = 0,
        ctx_recent_window: int = 0,
        ctx_stride: int = 0,
        ctx_suffix_match_tokens: int = 0,
        ctx_suffix_keep_tokens: int = 0,
        ctx_suffix_source_end: Optional[int] = None,
        ctx_middle_budget: int = 0,
        ctx_total_budget: Optional[int] = None,
        ctx_indexer_enable: bool = False,
        ctx_indexer_block_size: int = 4,
        ctx_indexer_top_k_blocks: int = 512,
        ctx_indexer_query_tokens: int = 512,
        ctx_indexer_score_reduce: str = "max",
        draft_denoise_steps: int = 1,
        return_verify_trace: bool = False,
        verify_trace_max_rounds: int = 0,
        verify_confidence_threshold: float = 0.0,
        verify_min_draft_tokens: int = 1,
        draft_dynamic_yarn_original_max_position_embeddings: Optional[int] = None,
        draft_dynamic_yarn_max_factor: Optional[float] = None,
        draft_dynamic_yarn_mode: str = "continuous",
        draft_dynamic_yarn_length_ratio: Optional[float] = None,
        ctx_dynamic_budget_ratio: Optional[float] = None,
        ctx_budget_order: str = "default",
        suffix_decoding: bool = False,
        suffix_strategy: str = "consensus",
        suffix_max_query_len: int = 16,
        suffix_min_query_len: int = 2,
        suffix_top_k: int = 4,
        suffix_min_support: int = 3,
        suffix_min_predict_len: int = 8,
        suffix_max_predict_len: Optional[int] = None,
        suffix_paper_alpha: float = 1.0,
        suffix_paper_max_spec_offset: float = 0.0,
        suffix_paper_min_token_prob: float = 0.0,
        suffix_paper_threshold: float = 0.0,
        suffix_paper_max_matches: int = 0,
        suffix_paper_verifier: str = "linear",
        suffix_paper_tree_attn_impl: str = "sdpa",
        suffix_fallback: str = "dflash",
        return_suffix_trace: bool = False,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            ctx_sink_tokens=ctx_sink_tokens,
            ctx_recent_window=ctx_recent_window,
            ctx_stride=ctx_stride,
            ctx_suffix_match_tokens=ctx_suffix_match_tokens,
            ctx_suffix_keep_tokens=ctx_suffix_keep_tokens,
            ctx_suffix_source_end=ctx_suffix_source_end,
            ctx_middle_budget=ctx_middle_budget,
            ctx_total_budget=ctx_total_budget,
            ctx_indexer_enable=ctx_indexer_enable,
            ctx_indexer_block_size=ctx_indexer_block_size,
            ctx_indexer_top_k_blocks=ctx_indexer_top_k_blocks,
            ctx_indexer_query_tokens=ctx_indexer_query_tokens,
            ctx_indexer_score_reduce=ctx_indexer_score_reduce,
            draft_denoise_steps=draft_denoise_steps,
            return_verify_trace=return_verify_trace,
            verify_trace_max_rounds=verify_trace_max_rounds,
            verify_confidence_threshold=verify_confidence_threshold,
            verify_min_draft_tokens=verify_min_draft_tokens,
            draft_dynamic_yarn_original_max_position_embeddings=(
                draft_dynamic_yarn_original_max_position_embeddings
            ),
            draft_dynamic_yarn_max_factor=draft_dynamic_yarn_max_factor,
            draft_dynamic_yarn_mode=draft_dynamic_yarn_mode,
            draft_dynamic_yarn_length_ratio=draft_dynamic_yarn_length_ratio,
            ctx_dynamic_budget_ratio=ctx_dynamic_budget_ratio,
            ctx_budget_order=ctx_budget_order,
            suffix_decoding=suffix_decoding,
            suffix_strategy=suffix_strategy,
            suffix_max_query_len=suffix_max_query_len,
            suffix_min_query_len=suffix_min_query_len,
            suffix_top_k=suffix_top_k,
            suffix_min_support=suffix_min_support,
            suffix_min_predict_len=suffix_min_predict_len,
            suffix_max_predict_len=suffix_max_predict_len,
            suffix_paper_alpha=suffix_paper_alpha,
            suffix_paper_max_spec_offset=suffix_paper_max_spec_offset,
            suffix_paper_min_token_prob=suffix_paper_min_token_prob,
            suffix_paper_threshold=suffix_paper_threshold,
            suffix_paper_max_matches=suffix_paper_max_matches,
            suffix_paper_verifier=suffix_paper_verifier,
            suffix_paper_tree_attn_impl=suffix_paper_tree_attn_impl,
            suffix_fallback=suffix_fallback,
            return_suffix_trace=return_suffix_trace,
        )
