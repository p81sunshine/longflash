from __future__ import annotations

from types import SimpleNamespace

import torch

from dflash.suffix_decoding import PaperSuffixNode, PaperSuffixPrediction, SuffixMatcher, SuffixPrediction


def test_suffix_matcher_uses_longest_suffix():
    matcher = SuffixMatcher()
    matcher.extend([9, 1, 2, 3, 7, 8, 1, 2, 3, 7, 1, 2, 3])

    prediction = matcher.predict(max_predict_len=1, max_query_len=3, min_query_len=2, top_k=4)

    assert prediction.query_len == 3
    assert prediction.support == 2
    assert prediction.tokens == [7]


def test_suffix_matcher_uses_recent_top_k_matches():
    matcher = SuffixMatcher()
    matcher.extend([1, 2, 8, 1, 2, 9, 1, 2, 9, 1, 2, 9, 1, 2, 9, 1, 2])

    prediction = matcher.predict(max_predict_len=1, max_query_len=2, min_query_len=2, top_k=4)

    assert prediction.match_positions == [13, 10, 7, 4]
    assert prediction.tokens == [9]


def test_suffix_matcher_requires_strict_majority_and_stops_on_divergence():
    matcher = SuffixMatcher()
    matcher.extend([1, 2, 9, 2, 1, 2, 9, 1, 1, 2])

    prediction = matcher.predict(max_predict_len=2, max_query_len=2, min_query_len=2, top_k=2)

    assert prediction.support == 2
    assert prediction.tokens == [9]

    matcher = SuffixMatcher()
    matcher.extend([1, 2, 8, 1, 2, 9, 1, 2, 8, 1, 2, 9, 1, 2])

    prediction = matcher.predict(max_predict_len=1, max_query_len=2, min_query_len=2, top_k=4)

    assert prediction.support == 4
    assert prediction.tokens == []


def test_suffix_prediction_confidence_gate_and_incremental_extend():
    matcher = SuffixMatcher()
    matcher.extend([4, 5, 6])
    matcher.extend([4, 5, 6])
    matcher.extend([4, 5])

    prediction = matcher.predict(max_predict_len=1, max_query_len=2, min_query_len=2, top_k=4)

    assert len(matcher) == 8
    assert prediction.tokens == [6]
    assert prediction.is_high_confidence(min_support=2, min_predict_len=1)
    assert not prediction.is_high_confidence(min_support=3, min_predict_len=1)
    assert not prediction.is_high_confidence(min_support=2, min_predict_len=2)


def test_paper_suffix_matcher_expands_tree_and_scores():
    matcher = SuffixMatcher()
    matcher.extend([1, 2, 3, 5, 1, 2, 3, 6, 1, 2, 4, 7, 1, 2])

    prediction = matcher.predict_paper(
        max_predict_len=4,
        max_query_len=2,
        min_query_len=2,
        alpha=2.0,
    )

    assert prediction.query_len == 2
    assert prediction.support == 3
    assert prediction.max_spec == 4
    assert prediction.tree_size == 4
    assert prediction.tokens == [3, 5, 1]
    assert prediction.is_high_confidence(threshold=1.0)
    assert not prediction.is_high_confidence(threshold=1.5)
    assert round(prediction.score, 6) == round(1.0 + 2 / 3 + 1 / 3 + 1 / 3 + 1 / 3, 6)
    assert round(prediction.best_path_score, 6) == round(2 / 3 + 1 / 3 + 1 / 3, 6)


def test_paper_suffix_matcher_alpha_controls_tree_size():
    matcher = SuffixMatcher()
    matcher.extend([1, 2, 3, 5, 1, 2, 3, 6, 1, 2, 4, 7, 1, 2])

    short_prediction = matcher.predict_paper(
        max_predict_len=8,
        max_query_len=2,
        min_query_len=2,
        alpha=1.0,
    )
    long_prediction = matcher.predict_paper(
        max_predict_len=4,
        max_query_len=2,
        min_query_len=2,
        alpha=2.0,
    )

    assert short_prediction.max_spec == 2
    assert short_prediction.tree_size == 2
    assert long_prediction.max_spec == 4
    assert long_prediction.tree_size == 4


def test_paper_suffix_matcher_uses_parent_count_and_min_token_prob():
    matcher = SuffixMatcher()
    matcher.extend([1, 2, 3, 4, 1, 2, 3])

    prediction = matcher._expand_paper_tree(
        matches=[1, 5],
        query_len=2,
        max_spec=2,
        min_token_prob=0.0,
        include_root_score=True,
        match_position_limit=16,
    )

    assert prediction.tree_size == 2
    assert prediction.nodes[1].token == 3
    assert prediction.nodes[1].d_score == 1.0
    assert prediction.nodes[2].token == 4
    assert prediction.nodes[2].d_score == 0.5

    filtered = matcher._expand_paper_tree(
        matches=[1, 5],
        query_len=2,
        max_spec=2,
        min_token_prob=0.75,
        include_root_score=True,
        match_position_limit=16,
    )

    assert filtered.tree_size == 1
    assert [node.token for node in filtered.nodes] == [None, 3]


class _FakeCache:
    def __init__(self):
        self.crop_lengths: list[int] = []

    def crop(self, length: int) -> None:
        self.crop_lengths.append(int(length))

    def get_seq_length(self) -> int:
        return self.crop_lengths[-1] if self.crop_lengths else 0


class _FakeInnerModel:
    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1)


class _FakeTarget:
    def __init__(self, sequence: list[int], *, vocab_size: int = 64):
        self.sequence = sequence
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.model = _FakeInnerModel()
        self.calls: list[dict[str, object]] = []

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        token_ids = hidden[..., 0].round().long().clamp(0, self.vocab_size - 1)
        logits = torch.full((*token_ids.shape, self.vocab_size), -1000.0)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 1000.0)
        return logits

    def __call__(
        self,
        input_ids: torch.LongTensor,
        *,
        position_ids: torch.LongTensor,
        past_key_values,
        use_cache: bool,
        output_hidden_states: bool = False,
        logits_to_keep: int | None = None,
        **kwargs,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "has_attention_mask": "attention_mask" in kwargs,
            }
        )
        next_tokens = []
        for pos in position_ids[0].tolist():
            next_tokens.append(self.sequence[int(pos) + 1])
        token_ids = torch.tensor([next_tokens], dtype=torch.long)
        logits = self.lm_head(token_ids.float().unsqueeze(-1))
        if logits_to_keep is not None:
            logits = logits[:, -int(logits_to_keep) :, :]

        hidden = position_ids.float().unsqueeze(-1)
        hidden_states = [hidden, hidden] if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class _FakeDraft:
    def __init__(self, draft_tokens: list[list[int]], *, block_size: int):
        self.block_size = block_size
        self.mask_token_id = 0
        self.target_layer_ids = [0]
        self.device = torch.device("cpu")
        self.draft_tokens = list(draft_tokens)
        self.target_hidden_lengths: list[int] = []

    def __call__(
        self,
        *,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values,
        use_cache: bool,
        is_causal: bool,
    ) -> torch.Tensor:
        self.target_hidden_lengths.append(int(target_hidden.shape[1]))
        tokens = self.draft_tokens.pop(0) if self.draft_tokens else [1] * (self.block_size - 1)
        values = [0] + tokens[: self.block_size - 1]
        return torch.tensor([[values]], dtype=torch.float32).transpose(1, 2)


def test_suffix_decoding_disabled_uses_dflash_path(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)
    target = _FakeTarget([1, 2, 3, 4, 5, 6])
    draft = _FakeDraft([[3]], block_size=2)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=1,
        stop_token_ids=None,
        temperature=0.0,
        block_size=2,
        return_stats=True,
        suffix_decoding=False,
    )

    assert draft.target_hidden_lengths == [2]
    assert result.suffix_decoding_enabled is False
    assert result.suffix_verify_rounds == 0


def test_high_confidence_suffix_path_skips_draft(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)
    prompt = [1, 2, 5, 1, 2, 5, 1, 2, 5, 1]
    sequence = prompt + [2, 5, 1, 2, 9]
    target = _FakeTarget(sequence)
    draft = _FakeDraft([[99, 99, 99]], block_size=4)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([prompt]),
        max_new_tokens=4,
        stop_token_ids=None,
        temperature=0.0,
        block_size=4,
        return_stats=True,
        suffix_decoding=True,
        suffix_max_query_len=2,
        suffix_min_predict_len=2,
    )

    assert draft.target_hidden_lengths == []
    assert result.suffix_verify_rounds == 1
    assert result.suffix_recovery_rounds == 0
    assert result.acceptance_lengths == [4]


def test_suffix_first_token_miss_commits_target_sample_without_recovery(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)
    prompt = [1, 2, 5, 1, 2, 5, 1, 2, 5, 1]
    sequence = prompt + [2, 6, 7, 8]
    target = _FakeTarget(sequence)
    draft = _FakeDraft([[6, 7]], block_size=3)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([prompt]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        return_stats=True,
        suffix_decoding=True,
        suffix_max_query_len=2,
        suffix_min_predict_len=1,
    )

    assert draft.target_hidden_lengths == [11]
    assert result.suffix_verify_rounds == 1
    assert result.suffix_recovery_rounds == 0
    assert result.suffix_acceptance_lengths == [0]
    assert result.acceptance_lengths == [1, 1]


def test_consecutive_suffix_rounds_accumulate_pending_hidden_for_next_dflash(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def __init__(self):
            self.calls = 0

        def extend(self, new_tokens):
            return None

        def predict(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SuffixPrediction(tokens=[11], support=3, query_len=2, match_positions=[0, 1, 2])
            if self.calls == 2:
                return SuffixPrediction(tokens=[13], support=3, query_len=2, match_positions=[0, 1, 2])
            return SuffixPrediction(tokens=[], support=0, query_len=0, match_positions=[])

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 11, 12, 13, 14, 15, 16])
    draft = _FakeDraft([[15]], block_size=2)

    dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=5,
        stop_token_ids=None,
        temperature=0.0,
        block_size=2,
        return_stats=True,
        suffix_decoding=True,
        suffix_min_predict_len=1,
    )

    assert draft.target_hidden_lengths == [6]


def test_explicit_suffix_max_predict_len_can_exceed_block_size(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def extend(self, new_tokens):
            return None

        def predict(self, **kwargs):
            return SuffixPrediction(tokens=[11, 12, 13, 14, 15], support=3, query_len=2, match_positions=[0, 1, 2])

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 11, 12, 13, 14, 15, 16])
    draft = _FakeDraft([[99]], block_size=2)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=5,
        stop_token_ids=None,
        temperature=0.0,
        block_size=2,
        return_stats=True,
        suffix_decoding=True,
        suffix_min_predict_len=1,
        suffix_max_predict_len=5,
    )

    assert target.calls[-1]["input_ids"].shape[1] == 6
    assert result.suffix_verify_rounds == 1
    assert result.suffix_acceptance_lengths == [5]


def test_paper_suffix_strategy_uses_path_score_threshold_and_skips_draft(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def extend(self, new_tokens):
            return None

        def predict_paper(self, **kwargs):
            return PaperSuffixPrediction(
                tokens=[11, 12],
                support=2,
                query_len=2,
                match_positions=[0, 1],
                score=3.0,
                token_score=2.0,
                max_spec=2,
                tree_size=2,
                best_path_score=2.0,
                nodes=[
                    PaperSuffixNode(token=None, parent=-1, depth=0, count=2, d_score=1.0),
                    PaperSuffixNode(token=11, parent=0, depth=1, count=2, d_score=1.0),
                    PaperSuffixNode(token=12, parent=1, depth=2, count=2, d_score=1.0),
                ],
            )

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 11, 12, 13])
    draft = _FakeDraft([[99, 99]], block_size=3)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=3,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        return_stats=True,
        suffix_decoding=True,
        suffix_strategy="paper",
        suffix_paper_threshold=1.5,
    )

    assert draft.target_hidden_lengths == []
    assert result.suffix_strategy == "paper"
    assert result.suffix_verify_rounds == 1
    assert result.suffix_paper_scores == [3.0]
    assert result.suffix_paper_tree_sizes == [2]
    assert result.acceptance_lengths == [3]


def test_paper_tree_verifier_accepts_tree_path_and_skips_draft(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def extend(self, new_tokens):
            return None

        def predict_paper(self, **kwargs):
            return PaperSuffixPrediction(
                tokens=[11, 12],
                support=3,
                query_len=2,
                match_positions=[0, 1, 2],
                score=4.0,
                token_score=3.0,
                max_spec=3,
                tree_size=3,
                best_path_score=2.0,
                nodes=[
                    PaperSuffixNode(token=None, parent=-1, depth=0, count=3, d_score=1.0),
                    PaperSuffixNode(token=11, parent=0, depth=1, count=2, d_score=2 / 3),
                    PaperSuffixNode(token=20, parent=0, depth=1, count=1, d_score=1 / 3),
                    PaperSuffixNode(token=12, parent=1, depth=2, count=2, d_score=2 / 3),
                ],
            )

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 11, 12, 13, 14])
    draft = _FakeDraft([[99, 99]], block_size=3)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=3,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        return_stats=True,
        suffix_decoding=True,
        suffix_strategy="paper",
        suffix_paper_threshold=1.0,
        suffix_paper_verifier="tree",
        suffix_max_predict_len=3,
    )

    assert draft.target_hidden_lengths == []
    assert target.calls[-1]["input_ids"].tolist() == [[10, 11, 20, 12]]
    assert target.calls[-1]["position_ids"].tolist() == [[2, 3, 3, 4]]
    assert target.calls[-1]["has_attention_mask"] is True
    assert result.suffix_verify_rounds == 1
    assert result.suffix_acceptance_lengths == [2]
    assert result.acceptance_lengths == [3]
    assert result.suffix_paper_verifier == "tree"


def test_paper_suffix_strategy_below_threshold_uses_dflash(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def extend(self, new_tokens):
            return None

        def predict_paper(self, **kwargs):
            return PaperSuffixPrediction(
                tokens=[11, 12],
                support=2,
                query_len=2,
                match_positions=[0, 1],
                score=3.0,
                token_score=1.0,
                max_spec=2,
                tree_size=2,
                best_path_score=1.0,
                nodes=[],
            )

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 20, 21, 22])
    draft = _FakeDraft([[20, 21]], block_size=3)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        return_stats=True,
        suffix_decoding=True,
        suffix_strategy="paper",
        suffix_paper_threshold=2.0,
    )

    assert draft.target_hidden_lengths == [2]
    assert result.suffix_verify_rounds == 0
    assert result.suffix_paper_scores == [3.0]


def test_paper_suffix_strategy_below_threshold_with_none_fallback_stops(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def extend(self, new_tokens):
            return None

        def predict_paper(self, **kwargs):
            return PaperSuffixPrediction(
                tokens=[11, 12],
                support=2,
                query_len=2,
                match_positions=[0, 1],
                score=3.0,
                token_score=1.0,
                max_spec=2,
                tree_size=2,
                best_path_score=1.0,
                nodes=[],
            )

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 20, 21, 22])
    draft = _FakeDraft([[20, 21]], block_size=3)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        return_stats=True,
        suffix_decoding=True,
        suffix_strategy="paper",
        suffix_paper_threshold=2.0,
        suffix_fallback="none",
    )

    assert draft.target_hidden_lengths == []
    assert result.suffix_verify_rounds == 0
    assert result.suffix_exhausted_rounds == 1
    assert result.draft_forward_passes == 0
    assert result.acceptance_lengths == []


def test_paper_suffix_strategy_below_threshold_with_target_fallback_generates(monkeypatch):
    from dflash import model as dm

    monkeypatch.setattr(dm, "DynamicCache", _FakeCache)
    monkeypatch.setattr(dm, "_cuda_time", lambda: 0.0)

    class FakeMatcher:
        def extend(self, new_tokens):
            return None

        def predict_paper(self, **kwargs):
            return PaperSuffixPrediction(
                tokens=[11, 12],
                support=2,
                query_len=2,
                match_positions=[0, 1],
                score=1.0,
                token_score=1.0,
                max_spec=2,
                tree_size=2,
                best_path_score=1.0,
                nodes=[],
            )

    monkeypatch.setattr(dm, "SuffixMatcher", FakeMatcher)
    target = _FakeTarget([1, 2, 10, 20, 21])
    draft = _FakeDraft([[99, 99]], block_size=3)

    result = dm.dflash_generate(
        draft,
        target,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        return_stats=True,
        suffix_decoding=True,
        suffix_strategy="paper",
        suffix_paper_threshold=2.0,
        suffix_fallback="target",
    )

    assert result.output_ids.tolist() == [[1, 2, 10, 20]]
    assert draft.target_hidden_lengths == []
    assert result.suffix_verify_rounds == 0
    assert result.suffix_exhausted_rounds == 2
    assert result.draft_forward_passes == 0
    assert result.acceptance_lengths == [1, 1]
    assert [call["input_ids"].tolist() for call in target.calls[1:]] == [[[10]], [[20]]]
