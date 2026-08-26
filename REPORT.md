# Results and design notes

Assignment: FinTech function-calling agent on ToolACE, evaluated on BFCL Python, served as if for Nebius Token Factory.

**Presentation narrative (full journey):** [PRESENTATION.md](PRESENTATION.md)

## Environment

| Item | Value |
| --- | --- |
| VM | Nebius 1xH100 80GB HBM3 |
| Host | computeinstance-e00nbnkgv0ynn1zkpr |
| Public IP | 89.169.99.143 |
| User | nebius-assignment |
| Backbone used | `Team-ACE/ToolACE-8B` (Llama 3.1 8B Instruct fine-tuned with ToolACE; Meta base gated pending approval) |
| Train data | Hugging Face `Team-ACE/ToolACE` |
| Eval | GitHub `ShishirPatil/gorilla` BFCL (`--test-category python`) |
| Serve | vLLM OpenAI API, merged BFCL-aligned LoRA BF16, `max_model_len=32768` |

## Model choice

Llama 3.1 8B Instruct is the intended backbone (matches published ToolACE-8B, Token Factory support, fits LoRA/QLoRA/full SFT + 16–32 concurrent on 1xH100). Meta HF gated access was not yet granted for `meta-llama/Meta-Llama-3.1-8B-Instruct`, so production experiments used **ToolACE-8B** (same architecture) + LoRA. Quantization was **not** applied: BF16 already meets 16–32 concurrent latency targets (accuracy > latency > cost).

## Critical accuracy fix

First LoRA run scored ~15–19% on BFCL because ToolACE-8B’s stripped chat template dropped `tool_calls`, so ~71% of simple_python outputs were empty EOS. Fix: re-prep ToolACE into BFCL Llama-3.1-FC JSON targets (`{"name","parameters"}`), render with BFCL’s prompt layout, retrain LoRA → `checkpoints/lora_bfcl` → `checkpoints/merged-lora-bfcl`.

## Training status

| Run | Method | Status | Train loss | Eval loss | Notes |
| --- | --- | --- | --- | --- | --- |
| A | Base (no FT) | pending | — | — | blocked on Meta gated base |
| B | QLoRA r=16 α=32 (Qwen2.5-7B pipeline test) | **done** | 0.5168 | 0.4392 | ungated stack validation |
| C | LoRA BF16 (ToolACE-8B, broken targets) | **done** | 0.4679 | 0.4111 | invalid for BFCL (empty assistants) |
| D | LoRA BF16 BFCL-aligned (ToolACE-8B) | **done** | — | **0.3302** | production candidate; token acc ~91% |
| E | QLoRA BFCL-aligned (ToolACE-8B) | **running** | — | — | method matrix vs LoRA → `checkpoints/qlora_bfcl` |
| F | Full SFT | pending | — | — | config ready (`train_full.yaml`); optional |

Data prep (fixed): 11,300 → 11,294 converted → 10,730 train / 564 val; **9,174** rows with JSON assistant targets.

## BFCL (Python group, FC mode) — fixed model

- Harness commit: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- Model id: `meta-llama/Llama-3.1-8B-Instruct-FC` → served `checkpoints/merged-lora-bfcl`
- Context: `max_model_len=32768`

| Slice | Before (broken LoRA) | After (BFCL-aligned LoRA) |
| --- | --- | --- |
| Overall (BFCL aggregate) | 15.52% | **22.94%** |
| Non-live overall | 18.75% | **71.06%** |
| Python Simple AST | 19.50% | **66.75%** |
| Multiple AST | 38.50% | **87.00%** |
| Parallel AST | 19.00% | **91.00%** |
| Parallel Multiple AST | 11.00% | **84.00%** |
| Irrelevance (non-live) | 97.50% | **97.92%** |
| Live overall | 40.86% | **70.98%** |
| Live Simple | 41.47% | **61.24%** |
| Live Multiple | 41.79% | **73.88%** |
| Live Parallel / Parallel Multiple | 18.75% / 8.33% | **68.75% / 50.00%** |
| Relevance / Irrelevance (live) | 50.00% / 93.67% | **87.50% / 76.70%** |

**Interpretation:** Format alignment was the dominant quality lever. Non-live AST jumped from ~19% simple / ~19–38% multi/parallel to **67–91%**. Aggregate “Overall Acc” stays lower because BFCL V4 overall also weights categories outside the Python-focused slices (e.g. multi-turn 0%). For the assignment’s Python subset, the meaningful headline numbers are **non-live 71.06%** and **live 70.98%**.

## Latency (vLLM BF16, tool-calling prompts)

Endpoint: `http://127.0.0.1:8000/v1`, model `llama-3.1-8b-toolace` (`checkpoints/merged-lora-bfcl`), 32 requests per concurrency level, 0 errors.

| Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | RPS | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 57.2 ms | 57.3 ms | 88.9 ms | 193 ms | 193 ms | 5.1 | 0 |
| 8 | 70.4 ms | 74.6 ms | 75.1 ms | 211 ms | 215 ms | 36.9 | 0 |
| 16 | 82.7 ms | 85.0 ms | 88.9 ms | 227 ms | 229 ms | 67.6 | 0 |
| 32 | 104 ms | 111 ms | 112 ms | 259 ms | 262 ms | 114 | 0 |

**Verdict:** At the client’s **16–32 concurrent** target, TTFT stays ~85–111 ms p95 and E2E under ~0.26 s p95 on 1×H100 BF16. No quantization needed for this SLA.

## Token Factory mapping

| This PoC | Token Factory production |
| --- | --- |
| Merged HF checkpoint on disk | Custom Weights Hub upload |
| `vllm serve` 1×H100 | Dedicated Endpoint, 1 GPU/replica |
| OpenAI `/v1/chat/completions` + tools | Same client contract |
| Prefix caching + BF16 | Keep BF16 first; FP8 if cost becomes priority |

## Commands used

```bash
# accuracy fix pipeline (re-prep + LoRA + merge + serve + BFCL)
bash scripts/06_fix_accuracy_pipeline.sh

# or stepwise
python src/data/prepare_toolace.py --config configs/data.yaml
python src/train/sft.py --config configs/train_lora.yaml
python src/train/merge_lora.py --base-model models/toolace-8b --adapter checkpoints/lora_bfcl/final --output checkpoints/merged-lora-bfcl
bash scripts/04_serve_merged.sh
bash scripts/05_run_bfcl.sh
python src/bench/latency.py --config configs/bench.yaml
```

## Demo outline (15–20 min)

1. Client problem: precise internal API calls; accuracy then latency then cost.
2. Why 8B / ToolACE-8B on 1xH100 / Token Factory.
3. Failure mode found: empty FC targets from broken chat template → ~15% BFCL.
4. Fix: BFCL Llama-3.1-FC JSON alignment + retrain.
5. Results: non-live **71%**, live **71%**, simple Python **67%** (was ~19%).
6. vLLM ≈ Dedicated Endpoint; BF16 meets 16–32 concurrent (no quant needed).
7. Latency table (TTFT/E2E @ 1/8/16/32).
8. Next: Meta base when gated access lands; optional QLoRA/full SFT / FP8 for cost.
