import os
import site
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="patched vLLM is not installed",
)


def _prepend_nvidia_wheel_libs():
    site_dirs = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(Path(user_site))

    lib_dirs = []
    for site_dir in site_dirs:
        nvidia_root = site_dir / "nvidia"
        if not nvidia_root.exists():
            continue
        lib_dirs.extend(path for path in nvidia_root.glob("cu*/lib") if path.is_dir())
        lib_dirs.extend(path for path in nvidia_root.glob("*/lib") if path.is_dir())

    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    deduped = []
    seen = set()
    for path in [*(str(path) for path in lib_dirs), *existing.split(os.pathsep)]:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(deduped)


def _load_dflash_proposer():
    _prepend_nvidia_wheel_libs()
    from vllm.v1.spec_decode.dflash import DFlashProposer

    return DFlashProposer


def _fake_proposer():
    DFlashProposer = _load_dflash_proposer()
    return object.__new__(DFlashProposer)


class FakeInputBatch:
    def __init__(self, rows, *, prompt_lens=None, computed_lens=None):
        self.req_ids = [f"req-{idx}" for idx in range(len(rows))]
        self.req_id_to_index = {req_id: idx for idx, req_id in enumerate(self.req_ids)}
        max_len = max(len(row) for row in rows)
        self.token_ids_cpu = torch.zeros(
            (len(rows), max_len),
            dtype=torch.int32,
        ).numpy()
        for idx, row in enumerate(rows):
            self.token_ids_cpu[idx, : len(row)] = row
        self.num_tokens_no_spec = torch.tensor(
            [len(row) for row in rows],
            dtype=torch.int32,
        ).numpy()
        self.num_prompt_tokens = torch.tensor(
            prompt_lens if prompt_lens is not None else [1] * len(rows),
            dtype=torch.int32,
        ).numpy()
        self.num_computed_tokens_cpu = torch.tensor(
            computed_lens if computed_lens is not None else [len(row) for row in rows],
            dtype=torch.int32,
        ).numpy()


def _fake_suffix_proposer(**overrides):
    proposer = _fake_proposer()
    proposer.dflash_suffix_decoding = True
    proposer.num_speculative_tokens = 4
    proposer.max_model_len = 128
    proposer.dflash_suffix_max_query_len = 2
    proposer.dflash_suffix_min_query_len = 2
    proposer.dflash_suffix_max_predict_len = 4
    proposer.dflash_suffix_alpha = 2.0
    proposer.dflash_suffix_max_spec_offset = 0.0
    proposer.dflash_suffix_min_token_prob = 0.0
    proposer.dflash_suffix_threshold = 0.0
    proposer.dflash_suffix_max_matches = 0
    proposer._dflash_suffix_matchers = {}
    for key, value in overrides.items():
        setattr(proposer, key, value)
    return proposer


def test_dflash_suffix_router_returns_paper_prediction():
    proposer = _fake_suffix_proposer()
    batch = FakeInputBatch([[1, 2, 3, 5, 1, 2, 3, 6, 1]])

    draft = proposer.propose_suffix_draft(batch, [[2]])

    assert draft == [[3, 5, 1, 2]]


def test_dflash_suffix_router_threshold_falls_back_to_dflash():
    proposer = _fake_suffix_proposer(dflash_suffix_threshold=99.0)
    batch = FakeInputBatch([[1, 2, 3, 5, 1, 2, 3, 6, 1]])

    draft = proposer.propose_suffix_draft(batch, [[2]])

    assert draft is None


def test_dflash_suffix_router_pads_single_request_hits_to_num_spec():
    proposer = _fake_suffix_proposer(dflash_suffix_max_predict_len=1)
    batch = FakeInputBatch([[1, 2, 3, 5, 1, 2, 3, 6, 1]])

    draft = proposer.propose_suffix_draft(batch, [[2]])

    assert draft == [[3, 3, 3, 3]]


def test_dflash_suffix_router_does_not_duplicate_bookkept_sample():
    proposer = _fake_suffix_proposer(dflash_suffix_max_predict_len=1)
    batch = FakeInputBatch([[1, 2, 3, 5, 1, 2, 3, 6, 1, 2]])

    draft = proposer.propose_suffix_draft(batch, [[2]])

    assert draft == [[3, 3, 3, 3]]
    assert len(proposer._dflash_suffix_matchers["req-0"]) == 10


def test_dflash_suffix_router_pads_batch_size_gt_one_hits():
    proposer = _fake_suffix_proposer()
    batch = FakeInputBatch(
        [
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
        ]
    )

    draft = proposer.propose_suffix_draft(batch, [[2], [2]])

    assert draft == [[3, 5, 1, 2], [3, 5, 1, 2]]


def test_dflash_suffix_router_pads_short_batch_hits_to_num_spec():
    proposer = _fake_suffix_proposer(dflash_suffix_max_predict_len=1)
    batch = FakeInputBatch(
        [
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
        ]
    )

    draft = proposer.propose_suffix_draft(batch, [[2], [2]])

    assert draft == [[3, 3, 3, 3], [3, 3, 3, 3]]


def test_dflash_suffix_router_batch_miss_falls_back_to_dflash():
    proposer = _fake_suffix_proposer(dflash_suffix_threshold=99.0)
    batch = FakeInputBatch(
        [
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
        ]
    )

    draft = proposer.propose_suffix_draft(batch, [[2], [2]])

    assert draft is None


def test_dflash_suffix_router_exposes_partial_batch_hits():
    proposer = _fake_suffix_proposer()
    batch = FakeInputBatch(
        [
            [1, 2, 3, 5, 1, 2, 3, 6, 1],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        ]
    )

    rows = proposer.propose_suffix_draft_rows(
        batch,
        [[2], [19]],
        pad_hits=True,
    )

    assert rows == [[3, 5, 1, 2], None]
