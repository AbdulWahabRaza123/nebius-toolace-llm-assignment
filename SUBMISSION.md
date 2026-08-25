# Submission notes for reviewers

Repository: training, evaluation, serving, and benchmarking scripts for the Nebius ML take-home (ToolACE + BFCL Python + vLLM).

## Clone and reproduce on the H100 VM

```bash
git clone https://github.com/AbdulWahabRaza123/nebius-toolace-llm-assignment.git
cd nebius-toolace-llm-assignment
cp .env.example .env   # set HF_TOKEN=hf_... (Llama 3.1 gated access required)
bash scripts/00_setup_vm.sh
bash scripts/01_download_assets.sh
bash scripts/02_prepare_data.sh
bash scripts/03_train.sh lora    # or qlora / full
```

See [README.md](README.md) for BFCL eval, vLLM serve, and latency bench commands. Full numbers and rationale: [REPORT.md](REPORT.md).

## What is not in git (by design)

- Model weights (`models/`, `checkpoints/`)
- Processed dataset shards (`data/processed/`)
- Hugging Face token (`.env`)
- Raw benchmark JSON (`results/`)

These live on the Nebius VM disk after running the scripts above.

## Official sources

- Training data: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- Base model: [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Eval harness: [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) (`berkeley-function-call-leaderboard`)

## VM used for experiments

- Nebius 1× H100 80GB
- SSH: `nebius-assignment@89.169.99.143`
- Project path on VM: `~/ml-ops`
