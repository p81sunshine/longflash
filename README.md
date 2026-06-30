# LongFlash Artifact

This directory is a standalone experiment artifact for reproducing the
LongFlash experiment code paths. It does not depend on the original development
repository.

## Contents

- `src/dflash/`: LongFlash/DFlash implementation used by the experiments.
- `scripts/`: experiment launchers and summarizers.
- `benchmarks/`: JSONL next-turn generation benchmarks.
- `patches/vllm/`: vLLM patch for suffix-routing integration.
- `tests/`: focused unit tests for suffix decoding, benchmark summary logic, and vLLM metric parsing.

Benchmark bucket manifests are in `benchmarks/terminal/manifest.json` and
`benchmarks/swebench/manifest.json`. Experiment outputs are not bundled; scripts
write new outputs under `results/reproduced/...` when run.

## Environment

Python 3.10+ is required. The original runs used NVIDIA A100 GPUs. Full
reproduction of GPU experiments requires access to Qwen target models and
DFlash draft models.

Create the local environment:

```bash
cd <artifact-root>
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For Transformers experiments, install FlashAttention if available:

```bash
python -m pip install flash-attn --no-build-isolation
```

For vLLM experiments, apply the included LongFlash vLLM patch:

```bash
git clone https://github.com/vllm-project/vllm.git third_party/vllm
cd third_party/vllm
git checkout 8decbfa02c9bbc1699b2136ecc72b1ef30c438a0
git apply ../../patches/vllm/longflash-vllm-8decbfa02.patch
python -m pip install -e .
cd ../..
```

For end-to-end SWE-Bench Verified, install mini-SWE-agent separately inside the
artifact tree:

```bash
git clone https://github.com/SWE-agent/mini-swe-agent.git third_party/mini-swe-agent
cd third_party/mini-swe-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
cd ../..
```

## Quick Checks

These checks do not require GPUs:

```bash
source .venv/bin/activate
python -m pytest \
  tests/test_suffix_decoding.py \
  tests/test_benchmark_stats.py \
  tests/test_agentic_memory_benchmark_summary.py
```

After installing the patched vLLM tree, also run:

```bash
python -m pytest tests/test_vllm_dflash_batch_packing.py
```

## Reproducing Experiments

All commands below are run from the artifact root directory. Set
`CUDA_VISIBLE_DEVICES`, `NUM_PROCS`, and model paths according to your machine.
Defaults use Hugging Face model IDs where possible:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-4B`
- `z-lab/Qwen3-8B-DFlash-b16`
- `z-lab/Qwen3-4B-DFlash-b16`

### Main Transformers Table

Smoke run:

```bash
MODELS="qwen3-8b" \
TEMPERATURES="0" \
VARIANTS="original_dflash dynamic_yarn dynamic_yarn_suffix swa3072" \
DATASET_GROUPS="terminal swebench" \
MAX_SAMPLES=2 \
MAX_NEW_TOKENS=256 \
NUM_PROCS=1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_qwen3_transformers_dynamic_yarn_suffix_buckets.sh
```

Full configuration:

```bash
MODELS="qwen3-8b qwen3-4b" \
TEMPERATURES="0 1" \
VARIANTS="original_dflash dynamic_yarn dynamic_yarn_suffix swa3072" \
DATASET_GROUPS="terminal swebench" \
MAX_SAMPLES=1000000000 \
MAX_NEW_TOKENS=4096 \
NUM_PROCS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_qwen3_transformers_dynamic_yarn_suffix_buckets.sh
```

### EAGLE-3 Baselines

EAGLE-3 requires an external EAGLE checkout and draft checkpoints. Put the
checkout at `third_party/EAGLE` or set `EAGLE_ROOT`, then set:

```bash
export EAGLE3_QWEN3_8B_MODEL=/path/to/qwen3-8b-eagle3-draft
export EAGLE3_QWEN3_4B_MODEL=/path/to/qwen3-4b-eagle3-draft
```

Run:

```bash
MODELS="qwen3-8b qwen3-4b" \
TEMPERATURES="0 1" \
VARIANTS="eagle3_spec16 eagle3_tree60" \
DATASET_GROUPS="terminal swebench" \
NUM_PROCS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_qwen3_eagle3_target_yarn_buckets.sh
```

### YaRN Sensitivity

This script builds its sampled evaluation set from `benchmarks/terminal` and
`benchmarks/swebench`.

Smoke run:

```bash
SAMPLES_PER_BUCKET=2 \
MAX_NEW_TOKENS=256 \
NUM_PROCS=1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_qwen3_8b_static_yarn_scan.sh
```

Full configuration:

```bash
SAMPLES_PER_BUCKET=20 \
MAX_NEW_TOKENS=4096 \
NUM_PROCS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_qwen3_8b_static_yarn_scan.sh
```

### vLLM Serving Throughput

Before running this section, install the suffix-routing vLLM patch described in
`patches/vllm/README.md`.

Smoke run over one bucket:

```bash
MAX_SAMPLES=4 \
MAX_TOKENS=256 \
CONCURRENCY=1 \
TP_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/benchmark/run_benchmark_vllm_manual.sh
```

Full matrix:

```bash
MODELS="qwen3-8b qwen3-4b" \
DATASET_GROUPS="terminal swebench" \
CONCURRENCY_VALUES="1 4 8 16 32" \
MAX_SAMPLES=50 \
TP_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/benchmark/run_qwen_vllm_by_variant_bucket_matrix.sh
```

The Qwen3.5-27B serving table uses tensor parallelism 2:

```bash
MODELS="qwen35-27b" \
DATASET_GROUPS="terminal swebench" \
CONCURRENCY_VALUES="1 4 8 16 32" \
MAX_SAMPLES=50 \
TP_SIZE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/benchmark/run_qwen_vllm_by_variant_bucket_matrix.sh
```

### End-to-End SWE-Bench

The end-to-end SWE-Bench Verified runs use mini-SWE-agent plus a vLLM server.
Re-running requires mini-SWE-agent, SWE-Bench Verified task environments, the
patched vLLM server scripts in `scripts/serve/`, and sufficient GPUs for 16
parallel agents.

Smoke run:

```bash
MINI_SWE_AGENT_ROOT=third_party/mini-swe-agent \
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/benchmark/run_qwen_swebench_verified.py \
  --run-root results/reproduced/end_to_end_swebench/smoke_qwen3_8b \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --base-url http://127.0.0.1:30000/v1 \
  --port 30000 \
  --sample-count 2 \
  --workers 1 \
  --tp-size 2 \
  --cuda-visible-devices 0,1 \
  --max-model-len 128000 \
  --max-num-batched-tokens 131072 \
  --target-yarn-original-max-position-embeddings 32768 \
  --target-yarn-factor 4 \
  --target-yarn-baselines \
  --original-max-position-embedding 128000 \
  --draft-yarn-original-max-position-embeddings 3072 \
  --draft-yarn-factor 42 \
  --variants target-only,original,yarn-suffix \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --disable-thinking \
  --environment-timeout 10 \
  --agent-step-limit 100 \
  --agent-max-tokens 2048
```

Full configuration:

```bash
MINI_SWE_AGENT_ROOT=third_party/mini-swe-agent \
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/benchmark/run_qwen_swebench_verified.py \
  --run-root results/reproduced/end_to_end_swebench/qwen3_8b_verified_500 \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --base-url http://127.0.0.1:30000/v1 \
  --port 30000 \
  --sample-count 500 \
  --workers 16 \
  --tp-size 2 \
  --cuda-visible-devices 0,1 \
  --max-model-len 128000 \
  --max-num-batched-tokens 131072 \
  --target-yarn-original-max-position-embeddings 32768 \
  --target-yarn-factor 4 \
  --target-yarn-baselines \
  --original-max-position-embedding 128000 \
  --draft-yarn-original-max-position-embeddings 3072 \
  --draft-yarn-factor 42 \
  --variants target-only,original,yarn-suffix \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --disable-thinking \
  --environment-timeout 10 \
  --agent-step-limit 100 \
  --agent-max-tokens 2048
```

## Notes

- Large model weights are not included.
- vLLM source is not vendored; the patch needed by the serving experiments is
  included in `patches/vllm/`.
- `results/` is intentionally not bundled. It is created by the scripts when
  experiments are run.
