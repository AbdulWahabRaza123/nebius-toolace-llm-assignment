# Financial workflow LLM (ToolACE + BFCL)

End-to-end take-home: fine-tune Llama 3.1 8B Instruct on the official Hugging Face **Team-ACE/ToolACE** dataset, evaluate the **Python** subset of BFCL from GitHub, and serve with vLLM as a Nebius Token Factory analogue.

**Where things run:** this laptop only holds git/scripts. Training, Hugging Face downloads, and GPU inference happen on the Nebius **1xH100** VM (`nebius-assignment@89.169.99.143`).

## Official sources

| Source | What |
| --- | --- |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Training data (`datasets.load_dataset`) |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Base model |
| [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) `berkeley-function-call-leaderboard` | BFCL Python eval |

## VM setup

```bash
ssh nebius-assignment@89.169.99.143
cd ~/ml-ops
bash scripts/00_setup_vm.sh
bash scripts/01_download_assets.sh
bash scripts/02_prepare_data.sh
```

`HF_TOKEN` must be able to access Llama 3.1 (accept the license on Hugging Face). ToolACE is public.

## Train (accuracy first)

```bash
source .venv/bin/activate
python src/train/sft.py --config configs/train_qlora.yaml
python src/train/sft.py --config configs/train_lora.yaml
python src/train/sft.py --config configs/train_full.yaml
```

Merge the LoRA winner, then serve:

```bash
python src/train/merge_lora.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter checkpoints/lora/final \
  --output checkpoints/merged-lora

bash src/serve/run_vllm.sh checkpoints/merged-lora llama-3.1-8b-toolace
```

## BFCL Python subset + latency

```bash
# other terminal, vLLM already up
bash src/eval/run_bfcl.sh llama-3.1-8b-toolace-FC python
python src/bench/latency.py --config configs/bench.yaml
```

Concurrency targets: **1, 8, 16, 32**. Metrics: TTFT p50/p95/p99 and E2E latency.

## Token Factory mapping

vLLM on this H100 is the local stand-in for a Token Factory **Dedicated Endpoint**: merged weights, 1 GPU / replica, OpenAI-compatible `/v1/chat/completions`, native tool calling (`llama3_json`). Production would upload the merged checkpoint to Custom Weights Hub and point a Dedicated Endpoint at it.

## Why Llama 3.1 8B

Same backbone as published ToolACE-8B, first-class on Token Factory, and 1xH100 can run LoRA, QLoRA, and full SFT. Accuracy is prioritized over quantization; FP8/INT4 only if 16–32 concurrent TTFT is not acceptable in BF16.
