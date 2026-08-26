# Submission notes for reviewers

Repository: training, evaluation, serving, and benchmarking scripts for the Nebius ML take-home (ToolACE + BFCL Python + vLLM).

Private repo: https://github.com/AbdulWahabRaza123/nebius-toolace-llm-assignment

## Clone and reproduce on the H100 VM

```bash
git clone https://github.com/AbdulWahabRaza123/nebius-toolace-llm-assignment.git
cd nebius-toolace-llm-assignment
cp .env.example .env   # set HF_TOKEN if downloading gated Meta Llama
bash scripts/00_setup_vm.sh
bash scripts/01_download_assets.sh   # ToolACE-8B + ToolACE dataset
bash scripts/02_prepare_data.sh      # BFCL Llama-3.1-FC JSON targets
bash scripts/03_train.sh lora        # or: qlora / full
# or one-shot: bash scripts/06_fix_accuracy_pipeline.sh
```

See [README.md](README.md) for BFCL eval, vLLM serve, and latency bench. Full numbers and rationale: [REPORT.md](REPORT.md).

## Headline results (production candidate)

- Backbone: `Team-ACE/ToolACE-8B` + BFCL-aligned LoRA → `checkpoints/merged-lora-bfcl`
- BFCL Python: non-live **~71%**, live **~71%**, simple AST **~67%** (was ~19% before format fix)
- Latency @ concurrency 32: TTFT p95 ~111 ms, E2E p95 ~262 ms (BF16, no quant)

## What is not in git (by design)

- Model weights (`models/`, `checkpoints/`)
- Processed dataset shards (`data/processed/`)
- Hugging Face token (`.env`)
- Raw benchmark JSON (`results/`)

These live on the Nebius VM disk after running the scripts above.

## Official sources

- Training data: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- Backbone used: [Team-ACE/ToolACE-8B](https://huggingface.co/Team-ACE/ToolACE-8B)
- Intended Meta base (gated): [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Eval harness: [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) (`berkeley-function-call-leaderboard`)

## VM used for experiments

- Nebius 1× H100 80GB
- SSH: `nebius-assignment@89.169.99.143`
- Project path on VM: `~/ml-ops`
