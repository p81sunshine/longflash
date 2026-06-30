# LongFlash vLLM Patch

This directory contains the suffix-routing vLLM modifications used for the
reported vLLM and end-to-end SWE-Bench experiments. The artifact does not
require the original development repository.

## Base Version

The patch was generated against upstream vLLM commit:

```text
8decbfa02c9bbc1699b2136ecc72b1ef30c438a0
```

The experiment environment reported this installed package version:

```text
vllm==0.20.2rc1.dev13+g8decbfa02
```

## Files

- `longflash-vllm-8decbfa02.patch`: unified patch to apply to a clean vLLM
  checkout at the base commit above.

The patch modifies:

```text
vllm/v1/spec_decode/dflash.py
vllm/v1/worker/gpu_model_runner.py
```

## Install

Run these commands from the artifact root directory:

```bash
git clone https://github.com/vllm-project/vllm.git third_party/vllm
cd third_party/vllm
git checkout 8decbfa02c9bbc1699b2136ecc72b1ef30c438a0
git apply ../../patches/vllm/longflash-vllm-8decbfa02.patch
python -m pip install -e .
```

If the patch does not apply because you are using a different vLLM revision,
start from the base commit above.

## What This Adds

The patch adds the suffix-routing part of LongFlash on top of vLLM's existing
DFlash support:

- `DFlashSuffixMatcher`, which keeps per-request token history and selects a
  high-confidence copied continuation from previous occurrences of the current
  suffix.
- `DFlashProposer.propose_suffix_draft_rows`, which exposes suffix proposals to
  the vLLM scheduler while falling back to normal DFlash when there is no hit.
- A `gpu_model_runner.py` scheduling hook that waits for CPU bookkeeping before
  suffix matching, because suffix routing needs the latest sampled token in the
  request history.

This patch intentionally contains only the suffix-routing integration. The
suffix route is enabled by
`DFLASH_SUFFIX_DECODING=1` and controlled by the `DFLASH_SUFFIX_*` environment
variables used by the artifact scripts.
