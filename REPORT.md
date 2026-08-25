# Results and design notes

Assignment: FinTech function-calling agent on ToolACE, evaluated on BFCL Python, served as if for Nebius Token Factory.

## Environment

| Item | Value |
| --- | --- |
| VM | Nebius 1xH100 80GB HBM3 |
| Host | computeinstance-e00nbnkgv0ynn1zkpr |
| Public IP | 89.169.99.143 |
| User | nebius-assignment |
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` (pending Meta HF approval); interim: `Team-ACE/ToolACE-8B` |
| Train data | Hugging Face `Team-ACE/ToolACE` |
| Eval | GitHub `ShishirPatil/gorilla` BFCL, `--test-category python` |

## Model choice

Llama 3.1 8B Instruct is the primary backbone because it matches published ToolACE-8B, is supported on Token Factory (LoRA and full FT), and fits three fine-tuning methods plus 16–32 concurrent vLLM requests on one H100. Quantization is deferred unless BF16 misses latency targets (accuracy > latency > cost).

## Training status

| Run | Method | Status | Train loss | Eval loss | BFCL Python |
| --- | --- | --- | --- | --- | --- |
| A | Base (no FT) | pending | — | — | pending |
| B | QLoRA r=16 α=32 (Qwen2.5-7B pipeline test) | **done** | 0.5168 | 0.4392 | pending |
| C | LoRA BF16 r=16 α=32 (ToolACE-8B backbone) | **in progress** | — | — | pending |
| D | Full SFT | pending | — | — | pending |

Data prep: 11,300 ToolACE rows → 10,735 train / 565 val (`data/processed/toolace_llama31` on VM).

## BFCL

- Harness: clone of `https://github.com/ShishirPatil/gorilla` (commit recorded in `results/bfcl/gorilla_commit.txt`).
- Category: `python`.
- Decoding: temperature 0.

## Latency (vLLM, tool-calling prompts)

Fill after `python src/bench/latency.py`.

| Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | RPS | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | | | | | | | |
| 8 | | | | | | | |
| 16 | | | | | | | |
| 32 | | | | | | | |

## Commands used

See `README.md` and `scripts/`. Terminal transcripts will be appended as runs complete.

## Demo outline (15–20 min)

1. Client problem: precise internal API calls; accuracy then latency then cost.
2. Why 8B Instruct on 1xH100 / Token Factory.
3. ToolACE → native Llama tool calls (same contract as production).
4. LoRA vs QLoRA vs full SFT table.
5. BFCL Python scores and error modes.
6. vLLM ≈ Dedicated Endpoint; merge vs live LoRA.
7. Concurrency 16–32 TTFT; why we did or did not quantize.
8. Next: 14B if more GPUs, live BFCL, guardrails.
