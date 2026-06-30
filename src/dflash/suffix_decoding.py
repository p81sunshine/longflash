from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class SuffixPrediction:
    tokens: list[int]
    support: int
    query_len: int
    match_positions: list[int]

    def is_high_confidence(self, *, min_support: int, min_predict_len: int) -> bool:
        return self.support >= min_support and len(self.tokens) >= min_predict_len


@dataclass(frozen=True)
class PaperSuffixNode:
    token: int | None
    parent: int
    depth: int
    count: int
    d_score: float


@dataclass(frozen=True)
class PaperSuffixPrediction:
    tokens: list[int]
    support: int
    query_len: int
    match_positions: list[int]
    score: float
    token_score: float
    max_spec: int
    tree_size: int
    best_path_score: float
    nodes: list[PaperSuffixNode]

    def is_high_confidence(self, *, threshold: float) -> bool:
        return bool(self.tokens) and self.best_path_score > threshold


@dataclass
class SuffixMatcher:
    """Incremental per-request suffix matcher for suffix speculative decoding."""

    tokens: list[int] = field(default_factory=list)
    _positions: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, new_tokens: Iterable[int]) -> None:
        base = len(self.tokens)
        for offset, token_id in enumerate(new_tokens):
            token_id = int(token_id)
            self.tokens.append(token_id)
            self._positions[token_id].append(base + offset)

    def predict(
        self,
        *,
        max_predict_len: int = 16,
        max_query_len: int = 16,
        min_query_len: int = 2,
        top_k: int = 4,
    ) -> SuffixPrediction:
        if max_predict_len <= 0 or top_k <= 0:
            return SuffixPrediction(tokens=[], support=0, query_len=0, match_positions=[])

        tokens = self.tokens
        n = len(tokens)
        min_query_len = max(1, int(min_query_len))
        if n < min_query_len + 1:
            return SuffixPrediction(tokens=[], support=0, query_len=0, match_positions=[])

        upper_query_len = min(int(max_query_len), n - 1)
        for query_len in range(upper_query_len, min_query_len - 1, -1):
            query = tokens[n - query_len : n]
            if not query:
                continue

            matches: list[int] = []
            for pos in reversed(self._positions.get(query[-1], ())):
                if pos == n - 1 or pos < query_len - 1:
                    continue
                if tokens[pos - query_len + 1 : pos + 1] != query:
                    continue
                matches.append(pos)
                if len(matches) >= top_k:
                    break

            if not matches:
                continue

            continuations = [
                tokens[pos + 1 : min(pos + 1 + max_predict_len, n)]
                for pos in matches
            ]
            prediction = self._majority_prefix(continuations, max_predict_len=max_predict_len)
            return SuffixPrediction(
                tokens=prediction,
                support=len(matches),
                query_len=query_len,
                match_positions=matches,
            )

        return SuffixPrediction(tokens=[], support=0, query_len=0, match_positions=[])

    def predict_paper(
        self,
        *,
        max_predict_len: int = 16,
        max_query_len: int = 16,
        min_query_len: int = 1,
        alpha: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.0,
        max_match_count: int = 0,
        include_root_score: bool = True,
        match_position_limit: int = 16,
    ) -> PaperSuffixPrediction:
        """Generate a paper-style suffix speculation tree plus best path.

        This implements the SuffixDecoding paper's local tree construction:
        for each pattern length p, find prior occurrences, greedily add the
        child node with the largest D(N), score the resulting tree by sum D(N),
        and choose the best-scoring p. ``nodes`` preserves the speculation
        tree; ``tokens`` is the best single root-to-node path for linear
        verifier fallback and trace readability.
        """
        empty = PaperSuffixPrediction(
            tokens=[],
            support=0,
            query_len=0,
            match_positions=[],
            score=0.0,
            token_score=0.0,
            max_spec=0,
            tree_size=0,
            best_path_score=0.0,
            nodes=[],
        )
        tokens = self.tokens
        n = len(tokens)
        min_query_len = max(1, int(min_query_len))
        alpha = max(0.0, float(alpha))
        max_spec_offset = float(max_spec_offset)
        min_token_prob = max(0.0, float(min_token_prob))
        if max_predict_len <= 0 or (alpha <= 0 and max_spec_offset <= 0):
            return empty
        if n < min_query_len + 1:
            return empty

        best: PaperSuffixPrediction | None = None
        upper_query_len = min(int(max_query_len), n - 1)
        for query_len in range(min_query_len, upper_query_len + 1):
            matches = self._find_matches(
                query_len=query_len,
                max_match_count=max(0, int(max_match_count)),
            )
            if not matches:
                continue
            max_spec = min(
                max_predict_len,
                max(0, int(alpha * query_len + max_spec_offset + 1e-6)),
            )
            if max_spec <= 0:
                continue
            candidate = self._expand_paper_tree(
                matches=matches,
                query_len=query_len,
                max_spec=max_spec,
                min_token_prob=min_token_prob,
                include_root_score=include_root_score,
                match_position_limit=max(0, int(match_position_limit)),
            )
            if candidate.tree_size <= 0:
                continue
            if best is None or (
                candidate.best_path_score,
                candidate.query_len,
                len(candidate.tokens),
                candidate.score,
            ) > (
                best.best_path_score,
                best.query_len,
                len(best.tokens),
                best.score,
            ):
                best = candidate

        return best if best is not None else empty

    def _find_matches(self, *, query_len: int, max_match_count: int = 0) -> list[int]:
        tokens = self.tokens
        n = len(tokens)
        if query_len <= 0 or n < query_len + 1:
            return []
        query = tokens[n - query_len : n]
        if not query:
            return []

        matches: list[int] = []
        for pos in reversed(self._positions.get(query[-1], ())):
            if pos == n - 1 or pos < query_len - 1:
                continue
            if tokens[pos - query_len + 1 : pos + 1] != query:
                continue
            matches.append(pos)
            if max_match_count > 0 and len(matches) >= max_match_count:
                break
        return matches

    def _expand_paper_tree(
        self,
        *,
        matches: list[int],
        query_len: int,
        max_spec: int,
        min_token_prob: float,
        include_root_score: bool,
        match_position_limit: int,
    ) -> PaperSuffixPrediction:
        nodes = [PaperSuffixNode(token=None, parent=-1, depth=0, count=len(matches), d_score=1.0)]
        paths: list[tuple[int, ...]] = [()]
        path_to_index: dict[tuple[int, ...], int] = {(): 0}
        child_count_cache: dict[tuple[int, ...], Counter[int]] = {}
        prefix_count_cache: dict[tuple[int, ...], int] = {(): len(matches)}

        while len(nodes) - 1 < max_spec:
            best_parent = -1
            best_token = -1
            best_count = 0
            best_d_score = -1.0

            for parent_idx, prefix in enumerate(paths):
                child_counts = child_count_cache.get(prefix)
                if child_counts is None:
                    child_counts = self._paper_child_counts(matches=matches, prefix=prefix)
                    child_count_cache[prefix] = child_counts
                if not child_counts:
                    continue

                denom = prefix_count_cache.get(prefix)
                if denom is None:
                    denom = self._paper_prefix_count(matches=matches, prefix=prefix)
                    prefix_count_cache[prefix] = denom
                if denom <= 0:
                    continue
                parent_d = nodes[parent_idx].d_score
                for token_id, count in child_counts.items():
                    child_path = prefix + (token_id,)
                    if child_path in path_to_index:
                        continue
                    d_score = parent_d * (count / denom)
                    if d_score < min_token_prob:
                        continue
                    if (
                        d_score > best_d_score
                        or (d_score == best_d_score and count > best_count)
                        or (d_score == best_d_score and count == best_count and token_id < best_token)
                    ):
                        best_parent = parent_idx
                        best_token = token_id
                        best_count = count
                        best_d_score = d_score

            if best_parent < 0:
                break

            best_path = paths[best_parent] + (best_token,)
            path_to_index[best_path] = len(nodes)
            prefix_count_cache[best_path] = best_count
            nodes.append(
                PaperSuffixNode(
                    token=int(best_token),
                    parent=best_parent,
                    depth=len(best_path),
                    count=int(best_count),
                    d_score=float(best_d_score),
                )
            )
            paths.append(best_path)

        token_score = sum(node.d_score for node in nodes[1:])
        score = token_score + (nodes[0].d_score if include_root_score and len(nodes) > 1 else 0.0)
        best_path, best_path_score = self._best_paper_path(nodes, paths)
        if match_position_limit > 0:
            match_positions = matches[:match_position_limit]
        else:
            match_positions = []
        return PaperSuffixPrediction(
            tokens=list(best_path),
            support=len(matches),
            query_len=query_len,
            match_positions=match_positions,
            score=float(score),
            token_score=float(token_score),
            max_spec=max_spec,
            tree_size=max(0, len(nodes) - 1),
            best_path_score=float(best_path_score),
            nodes=nodes,
        )

    def _paper_prefix_count(self, *, matches: list[int], prefix: tuple[int, ...]) -> int:
        if not prefix:
            return len(matches)

        tokens = self.tokens
        prefix_len = len(prefix)
        count = 0
        for pos in matches:
            start = pos + 1
            end = start + prefix_len
            if end > len(tokens):
                continue
            if tuple(tokens[start:end]) == prefix:
                count += 1
        return count

    def _paper_child_counts(self, *, matches: list[int], prefix: tuple[int, ...]) -> Counter[int]:
        tokens = self.tokens
        prefix_len = len(prefix)
        counts: Counter[int] = Counter()
        for pos in matches:
            start = pos + 1
            next_pos = start + prefix_len
            if next_pos >= len(tokens):
                continue
            if prefix_len and tuple(tokens[start:next_pos]) != prefix:
                continue
            counts[int(tokens[next_pos])] += 1
        return counts

    @staticmethod
    def _best_paper_path(
        nodes: list[PaperSuffixNode],
        paths: list[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], float]:
        best_path: tuple[int, ...] = ()
        best_score = 0.0
        for idx in range(1, len(nodes)):
            score = 0.0
            cursor = idx
            while cursor > 0:
                score += nodes[cursor].d_score
                cursor = nodes[cursor].parent
            path = paths[idx]
            if score > best_score or (score == best_score and len(path) > len(best_path)):
                best_score = score
                best_path = path
        return best_path, best_score

    @staticmethod
    def _majority_prefix(continuations: list[list[int]], *, max_predict_len: int) -> list[int]:
        prediction: list[int] = []
        for offset in range(max_predict_len):
            counts: dict[int, int] = {}
            total = 0
            for continuation in continuations:
                if offset >= len(continuation):
                    continue
                token_id = continuation[offset]
                counts[token_id] = counts.get(token_id, 0) + 1
                total += 1

            if total == 0:
                break

            best_token, best_count = max(counts.items(), key=lambda item: item[1])
            if best_count * 2 <= total:
                break
            prediction.append(best_token)

        return prediction
