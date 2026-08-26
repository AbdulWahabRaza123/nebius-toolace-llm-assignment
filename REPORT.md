# Results and design notes

Assignment: FinTech function-calling agent on ToolACE, evaluated on BFCL Python, served as if for Nebius Token Factory.

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
| Serve | vLLM OpenAI API, merged LoRA BF16, tool parser `llama3_json` |

## Model choice

Llama 3.1 8B Instruct is the intended backbone (matches published ToolACE-8B, Token Factory support, fits LoRA/QLoRA/full SFT + 16–32 concurrent on 1xH100). Meta HF gated access was not yet granted for `meta-llama/Meta-Llama-3.1-8B-Instruct`, so production experiments used **ToolACE-8B** (same architecture) + additional LoRA on ToolACE. Quantization was **not** applied: BF16 already meets 16–32 concurrent latency targets (accuracy > latency > cost).

## Training status

| Run | Method | Status | Train loss | Eval loss | Notes |
| --- | --- | --- | --- | --- | --- |
| A | Base (no FT) | pending | — | — | blocked on Meta gated base |
| B | QLoRA r=16 α=32 (Qwen2.5-7B pipeline test) | **done** | 0.5168 | 0.4392 | ungated stack validation |
| C | LoRA BF16 r=16 α=32 (ToolACE-8B) | **done** | 0.4679 | 0.4111 | production candidate |
| D | Full SFT | pending | — | — | optional if GPU time remains |

Data prep: 11,300 ToolACE rows → 10,735 train / 565 val.

## BFCL (Python group, FC mode)

- Harness commit: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- Model id: `meta-llama/Llama-3.1-8B-Instruct-FC` → served merged LoRA weights
- Partial eval (1 long live_irrelevance case exceeded 8k context during generate)

| Slice | Accuracy |
| --- | --- |
| Overall (BFCL aggregate for run) | **15.49%** |
| Non-live overall | **18.45%** |
| Python Simple AST | **18.75%** |
| Multiple AST | **38.50%** |
| Parallel AST | **18.00%** |
| Parallel Multiple AST | **11.06%** |
| Irrelevance (non-live) | **97.50%** |
| Live overall | **40.71%** |
| Live Simple | **41.09%** |
| Live Multiple | **41.79%** |
| Relevance / Irrelevance (live) | 50.00% / 93.88% |

**Interpretation:** Strong refusal/irrelevance behavior; AST tool-call match under FC handler is weaker than expected for ToolACE-8B lineage. Likely causes: (1) FC native tool schema vs ToolACE chat/tool formatting used in SFT, (2) 8k context truncation on a few long cases. Next accuracy step: re-eval in prompt mode and/or align training to BFCL FC JSON schema, then re-run with `max_model_len>=32k`.

## Latency (vLLM BF16, tool-calling prompts)

Endpoint: `http://127.0.0.1:8000/v1`, model `llama-3.1-8b-toolace`, 32 requests per concurrency level, 0 errors.

| Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | RPS | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.6 ms | 16.1 ms | 48.0 ms | 243 ms | 245 ms | 4.1 | 0 |
| 8 | 29.7 ms | 31.0 ms | 31.5 ms | 271 ms | 359 ms | 26.5 | 0 |
| 16 | 42.3 ms | 44.4 ms | 45.5 ms | 284 ms | 289 ms | 49.8 | 0 |
| 32 | 65.4 ms | 68.6 ms | 68.8 ms | 405 ms | 409 ms | 75.0 | 0 |

**Verdict:** At the client’s **16–32 concurrent** target, TTFT stays under ~70 ms p95 and E2E under ~0.41 s p95 on 1×H100 BF16. No quantization needed for this SLA.

## Token Factory mapping

| This PoC | Token Factory production |
| --- | --- |
| Merged HF checkpoint on disk | Custom Weights Hub upload |
| `vllm serve` 1×H100 | Dedicated Endpoint, 1 GPU/replica |
| OpenAI `/v1/chat/completions` + tools | Same client contract |
| Prefix caching + BF16 | Keep BF16 first; FP8 if cost becomes priority |

## Commands used

```bash
# merge + serve
python src/train/merge_lora.py --base-model models/toolace-8b --adapter checkpoints/lora/final --output checkpoints/merged-lora
bash scripts/04_serve_merged.sh

# BFCL python
bash scripts/05_run_bfcl.sh
bfcl evaluate --model meta-llama/Llama-3.1-8B-Instruct-FC --test-category python --partial-eval

# latency
python src/bench/latency.py --config configs/bench.yaml
```

## Demo outline (15–20 min)

1. Client problem: precise internal API calls; accuracy then latency then cost.
2. Why 8B / ToolACE-8B on 1xH100 / Token Factory.
3. ToolACE → chat+tools formatting; LoRA recipe.
4. Training table (Qwen QLoRA + ToolACE-8B LoRA).
5. BFCL Python scores + format/context caveats.
6. vLLM ≈ Dedicated Endpoint; merge vs live LoRA.
7. Concurrency 16–32 TTFT/E2E; why no quantization.
8. Next: Meta base access, FC-format alignment, optional full SFT / FP8.
