# Financial workflow LLM (ToolACE + BFCL)

End-to-end take-home: fine-tune an 8B function-calling model on Hugging Face **Team-ACE/ToolACE**, evaluate the **Python** subset of BFCL, and serve with vLLM as a Nebius Token Factory analogue.

**Where things run:** this laptop holds git/scripts. Training, downloads, and GPU inference run on the Nebius **1×H100** VM (`nebius-assignment@89.169.99.143`).

Results and design notes: [REPORT.md](REPORT.md). Reviewer clone notes: [SUBMISSION.md](SUBMISSION.md).

## Official sources

| Source | What |
| --- | --- |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Training data |
| [Team-ACE/ToolACE-8B](https://huggingface.co/Team-ACE/ToolACE-8B) | Backbone used (Llama 3.1 8B Instruct already ToolACE-tuned; Meta gated base pending) |
| [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) `berkeley-function-call-leaderboard` | BFCL Python eval |

## VM setup

```bash
ssh nebius-assignment@89.169.99.143
cd ~/ml-ops
bash scripts/00_setup_vm.sh
bash scripts/01_download_assets.sh
bash scripts/02_prepare_data.sh
```

`HF_TOKEN` is needed for gated Meta Llama downloads. ToolACE / ToolACE-8B are public.

## Train (accuracy first)

Data prep writes BFCL Llama-3.1-FC JSON assistant targets (`{"name","parameters"}`). Prefer the accuracy-fix pipeline, or train LoRA / QLoRA / full SFT separately:

```bash
source .venv/bin/activate

# recommended: re-prep + LoRA + merge + serve + BFCL
bash scripts/06_fix_accuracy_pipeline.sh

# method matrix
python src/train/sft.py --config configs/train_lora.yaml    # → checkpoints/lora_bfcl
python src/train/sft.py --config configs/train_qlora.yaml   # → checkpoints/qlora
python src/train/sft.py --config configs/train_full.yaml    # needs Meta base weights
```

Merge + serve the LoRA winner:

```bash
python src/train/merge_lora.py \
  --base-model models/toolace-8b \
  --adapter checkpoints/lora_bfcl/final \
  --output checkpoints/merged-lora-bfcl

bash scripts/04_serve_merged.sh
```

## BFCL Python subset + latency

```bash
bash scripts/05_run_bfcl.sh
python src/bench/latency.py --config configs/bench.yaml
```

Concurrency targets: **1, 8, 16, 32**. Metrics: TTFT p50/p95/p99 and E2E latency.

Headline BFCL (fixed LoRA): non-live **~71%**, live **~71%**, Python simple **~67%** (was ~19% before format alignment). See [REPORT.md](REPORT.md).

## Token Factory mapping

vLLM on this H100 stands in for a Token Factory **Dedicated Endpoint**: merged weights, 1 GPU / replica, OpenAI-compatible `/v1/chat/completions`, native tool calling (`llama3_json`). Production would upload the merged checkpoint to Custom Weights Hub and point a Dedicated Endpoint at it.

## Why 8B / ToolACE-8B

Same backbone as published ToolACE-8B, Token Factory–friendly size, and 1×H100 can run LoRA, QLoRA, and full SFT. Accuracy over quantization; FP8/INT4 only if 16–32 concurrent TTFT is not acceptable in BF16.
