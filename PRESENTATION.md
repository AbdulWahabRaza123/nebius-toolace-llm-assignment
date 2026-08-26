# Nebius Take-Home — Presentation Journey Document

**Role:** Senior ML Solutions Architect (Customer Solution Architect)  
**Exercise:** FinTech function-calling agent — ToolACE train → BFCL Python eval → production-style serve on 1×H100  
**Repo:** https://github.com/AbdulWahabRaza123/nebius-toolace-llm-assignment  
**Companion metrics sheet:** [REPORT.md](REPORT.md)

This document is the **final narrative for the 15–20 minute demo**: what we built, why we chose it, what broke, what we fixed, and what we would ship.

---

## 1. One-minute executive summary

A FinTech client needs an agent that turns natural language into **precise internal API / function calls**. Priorities are **accuracy → latency → cost**.

We delivered an end-to-end PoC on Nebius **1×H100**:

| Layer | Choice |
| --- | --- |
| Data | Official HF `Team-ACE/ToolACE` |
| Backbone | `Team-ACE/ToolACE-8B` (Llama 3.1 8B Instruct lineage; Meta gated base pending) |
| Winning FT | BFCL-aligned **LoRA BF16** → merged checkpoint |
| Eval | Official BFCL GitHub harness, **Python** subset, FC mode |
| Serve | **vLLM** OpenAI API = Token Factory Dedicated Endpoint analogue |
| Precision | **BF16** (no quant) — already meets 16–32 concurrent SLA |

**Headline quality (production candidate):**

- BFCL **non-live overall: 71.06%**
- BFCL **live overall: 70.98%**
- Python **simple AST: 66.75%** (was ~19.5% before the format fix)
- Multiple / Parallel / Parallel-multiple AST: **87% / 91% / 84%**

**Headline latency (merged LoRA, concurrency 32):**

- TTFT p95 ≈ **111 ms**
- E2E p95 ≈ **262 ms**
- Throughput ≈ **114 RPS**, **0 errors**

**Critical lesson:** The first LoRA looked “trained” but scored ~15–19% on BFCL because the chat template dropped `tool_calls`, so the model learned to emit empty assistants. Fixing **train ↔ eval format alignment** was the dominant quality lever—not more epochs or quantization.

---

## 2. Client problem (how we frame the demo)

**Background:** FinTech startup automating workflows (fraud, transactions). The agent must call internal APIs correctly from natural language.

**Requirements mapped to workstreams:**

1. Fine-tune on proprietary-style data (**ToolACE**)
2. Deploy as if for **Nebius Token Factory**
3. Optimize in order: **accuracy → latency → cost**
4. Evaluate on BFCL **Python** subset
5. Prove **16–32 concurrent** requests with TTFT / latency metrics
6. Hand over reproducible scripts + reported results

**What “good” looks like for this interview:** not a research paper—a **Customer Solution Architect** story: right model size for the GPU, honest tradeoffs, production mapping, and a debugging journey that improved quality.

---

## 3. Constraints & environment

| Item | Value |
| --- | --- |
| GPU | Nebius 1×H100 80GB HBM3 |
| Host | `computeinstance-e00nbnkgv0ynn1zkpr` |
| Public IP | `89.169.99.143` |
| User | `nebius-assignment` |
| Project path | `~/ml-ops` |
| Local laptop | Git, scripts, docs only (no model training) |

**Official sources used:**

- Train: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- Backbone: [Team-ACE/ToolACE-8B](https://huggingface.co/Team-ACE/ToolACE-8B)
- Intended Meta base (gated): `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Eval: [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) BFCL (`berkeley-function-call-leaderboard`), commit `6ea57973…`

---

## 4. Model choice — why 8B / ToolACE-8B

| Criterion | Decision |
| --- | --- |
| Function-calling fit | Same lineage as published ToolACE-8B |
| Token Factory | 8B is a practical Dedicated Endpoint size |
| 1×H100 budget | Fits LoRA, QLoRA, and full SFT experiments |
| Gating | Meta Llama 3.1 Instruct was not downloadable yet → use **ToolACE-8B** (already Llama-3.1-8B + ToolACE) |
| Quantization | Deferred: BF16 already satisfied concurrency SLA |

**Accuracy > latency > cost** ⇒ we did **not** start with INT4/FP8. Quantization is a cost lever after quality and latency are proven.

---

## 5. Journey timeline (what we actually did)

### Phase A — Scaffold & VM

- Built reproducible repo: configs, `src/train`, `src/data`, `src/eval`, `src/bench`, `src/serve`, scripts
- Provisioned / connected Nebius H100 VM
- Installed PyTorch, TRL/PEFT, vLLM, HF stack, cloned BFCL

### Phase B — Pipeline validation (ungated)

- Meta Llama gated → downloaded **Qwen2.5-7B-Instruct** and ran **QLoRA smoke** to prove train stack works
- Qwen QLoRA: train loss **0.5168**, eval loss **0.4392**

### Phase C — ToolACE data + first ToolACE-8B LoRA

- Loaded official ToolACE (11,300 rows)
- First prep/train used ToolACE-8B chat template
- First LoRA completed (train/eval looked healthy: ~0.47 / ~0.41)
- Merged + served with vLLM; ran BFCL Python

**Result:** BFCL looked bad (~15–19% simple). Latency was already fine.

### Phase D — Diagnosis (the turning point)

Root causes found:

1. ToolACE-8B **stripped chat template ignored `tool_calls`** → SFT targets were often **empty assistants** → model learned EOS
2. ~**71%** of `simple_python` generations were empty
3. Tool extraction / train format ≠ BFCL **LlamaHandler_3_1** JSON (`{"name","parameters"}`; decode prefers `True`/`False`/`None`)

This is the story beat for the demo: **metrics without format alignment lie**.

### Phase E — Accuracy fix + retrain

- Re-wrote `prepare_toolace.py`: bracket-match tools, BFCL JSON assistant targets, Arrow-safe serialization
- Hand-built BFCL Llama 3.1 FC prompt render in `sft.py` (`render_bfcl_llama31`)
- Patched chat template; empty-assistant guards
- One-shot pipeline: `scripts/06_fix_accuracy_pipeline.sh`
- Retrained LoRA → `checkpoints/lora_bfcl` → merged `checkpoints/merged-lora-bfcl`
- Re-served with `max_model_len=32768`, tool parser `llama3_json`
- Re-ran BFCL + latency

### Phase F — Method matrix (in progress / planned)

| Run | Method | Status | Role in story |
| --- | --- | --- | --- |
| A | Meta base (no FT) | Blocked (gated) | Ideal A/B when access lands |
| B | Qwen QLoRA | Done | Stack validation |
| C | ToolACE LoRA (broken targets) | Done | Negative control / failure case |
| D | ToolACE LoRA BFCL-aligned | **Done — production pick** | Winning system |
| E | ToolACE QLoRA BFCL-aligned | **Running** | Method comparison (memory/cost vs LoRA) |
| F | Full SFT | Config ready; optional | Completeness if GPU time remains |

---

## 6. Data preparation (fixed)

| Stat | Value |
| --- | --- |
| Raw ToolACE rows | 11,300 |
| Converted | 11,294 |
| Dropped | 6 |
| Train / val | 10,730 / 564 |
| Rows with tools | 10,612 |
| Rows with JSON assistant targets | **9,174** |
| Format | `bfcl_llama31_fc_json` |

Assistant targets serialized as BFCL-style JSON arrays of `{"name","parameters"}` so train and BFCL FC decode agree.

---

## 7. Training results

### Production LoRA (BFCL-aligned)

| Item | Value |
| --- | --- |
| Config | `configs/train_lora.yaml` |
| Base | `models/toolace-8b` |
| Method | LoRA r=16, α=32, BF16 |
| Epochs | 2 |
| Seq length | 8192 |
| Output | `checkpoints/lora_bfcl/final` |
| Merged | `checkpoints/merged-lora-bfcl` |
| Eval loss | **≈ 0.330** |
| Mean token accuracy | **≈ 91.1%** |

### Earlier / supporting runs

| Run | Eval loss | Notes |
| --- | --- | --- |
| Qwen QLoRA | 0.4392 | Pipeline proof |
| Broken ToolACE LoRA | 0.4111 | Invalid for BFCL (empty assistants) |

### QLoRA (BFCL-aligned) — status at doc freeze

- Config: `configs/train_qlora.yaml` → `checkpoints/qlora_bfcl`
- Purpose: same data/format as winning LoRA, lower memory / cost profile
- **Status:** training in progress on VM; final loss + optional BFCL compare to be filled when complete
- *Update slot:* train loss ___ / eval loss ___ / BFCL non-live ___ (fill after run)

---

## 8. BFCL accuracy — what to say on stage

**Harness:** BFCL FC mode, model id aliased as `meta-llama/Llama-3.1-8B-Instruct-FC`, weights = `merged-lora-bfcl`.

### Before vs after format fix

| Slice | Broken LoRA | Fixed LoRA |
| --- | --- | --- |
| Overall aggregate (CSV) | 15.52% | 22.94% |
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

### How to interpret “Overall Acc” 22.94%

Do **not** lead with aggregate Overall Acc. BFCL V4 overall also folds categories outside the assignment’s Python focus (e.g. multi-turn scored 0%). For this take-home, the fair headlines are:

- **Non-live 71.06%**
- **Live 70.98%**
- **Python simple 66.75%** + strong multi/parallel AST

---

## 9. Optimization, deployment & Token Factory mapping

### Serving stack

- Engine: **vLLM** OpenAI-compatible `/v1/chat/completions`
- Precision: **BF16**
- Context: `max_model_len=32768`
- Tool calling: `--enable-auto-tool-choice --tool-call-parser llama3_json`
- Prefix caching enabled; `max-num-seqs 64`

### Token Factory analogue

| This PoC | Production Token Factory |
| --- | --- |
| Merged HF checkpoint on disk | Custom Weights Hub upload |
| `vllm serve` on 1×H100 | Dedicated Endpoint, 1 GPU / replica |
| OpenAI chat + tools | Same client contract |
| BF16 + prefix cache | Keep BF16 first; FP8 later for cost |

### Latency benchmark (tool-calling prompts)

Endpoint `http://127.0.0.1:8000/v1`, model `llama-3.1-8b-toolace`, 32 requests per concurrency level.

| Concurrency | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | RPS | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 57.2 ms | 57.3 ms | 193 ms | 193 ms | 5.1 | 0 |
| 8 | 70.4 ms | 74.6 ms | 211 ms | 215 ms | 36.9 | 0 |
| 16 | 82.7 ms | 85.0 ms | 227 ms | 229 ms | 67.6 | 0 |
| 32 | 104 ms | **111 ms** | 259 ms | **262 ms** | **114** | 0 |

**Optimization verdict:** At the client’s **16–32 concurrent** target, BF16 already clears a practical SLA. Quantization was **intentionally skipped** (would trade accuracy risk for cost savings we do not need yet).

---

## 10. Decisions & tradeoffs (demo Q&A ammo)

| Question | Answer |
| --- | --- |
| Why not Meta Llama base? | Gated HF access not granted in time; ToolACE-8B is same architecture and stronger FC starting point |
| Why LoRA over QLoRA? | LoRA BF16 is production pick today; QLoRA is the cost/memory comparison run |
| Why no FP8/INT4? | Accuracy first; latency already OK at 16–32 concurrency |
| Why is Overall Acc low? | Aggregate includes out-of-scope categories; report Python non-live/live AST |
| Biggest quality unlock? | Align ToolACE SFT targets to BFCL Llama-3.1-FC JSON + prompt layout |
| What would you do next with the client? | Meta base A/B, full SFT if quality plateau, then FP8 for $ / GPU density |

---

## 11. Reproducible commands

```bash
# VM
ssh nebius-assignment@89.169.99.143
cd ~/ml-ops
source .venv/bin/activate

# Full accuracy-fix path
bash scripts/06_fix_accuracy_pipeline.sh

# Or stepwise
bash scripts/02_prepare_data.sh
python src/train/sft.py --config configs/train_lora.yaml
python src/train/merge_lora.py \
  --base-model models/toolace-8b \
  --adapter checkpoints/lora_bfcl/final \
  --output checkpoints/merged-lora-bfcl
bash scripts/04_serve_merged.sh
bash scripts/05_run_bfcl.sh
python src/bench/latency.py --config configs/bench.yaml

# Method matrix
python src/train/sft.py --config configs/train_qlora.yaml
python src/train/sft.py --config configs/train_full.yaml   # optional
```

Artifacts **not** in git (by design): weights, processed data, `.env`, raw `results/` JSON. Live on the VM after scripts run. See [SUBMISSION.md](SUBMISSION.md).

---

## 12. Suggested 15–20 minute demo arc

| Min | Beat |
| --- | --- |
| 0–2 | Client: precise API calls; accuracy → latency → cost; 1×H100 |
| 2–4 | Why ToolACE-8B / 8B / Token Factory Dedicated Endpoint |
| 4–7 | Journey: pipeline → first LoRA → “good loss, bad BFCL” |
| 7–10 | Root cause: empty `tool_calls` / format mismatch; show before/after table |
| 10–13 | Fixed results: ~71% non-live/live; multi/parallel AST |
| 13–16 | Serve map to Token Factory; latency @ 16–32; why no quant |
| 16–18 | Method matrix status (LoRA win; QLoRA comparison) |
| 18–20 | Next steps + Q&A |

---

## 13. Deliverables checklist vs assignment

| Deliverable | Status |
| --- | --- |
| Fine-tune suitable model + explain choice | Done |
| Experiment with multiple FT methods | LoRA done; Qwen QLoRA done; ToolACE QLoRA running; full SFT optional |
| Best practices for quality | Done (format alignment, FC render, longer context serve) |
| Reproducible train/eval/serve scripts | Done in private GitHub |
| Report training + BFCL + latency | Done in REPORT + this document |
| Optimize inference | Done (vLLM, prefix cache, BF16, 32k context) |
| Deploy production-style + BFCL Python | Done |
| Measure TTFT/latency @ 16–32 concurrent | Done |
| Quantization scripts | Not required given SLA; justified skip |

---

## 14. Open items (honest close)

1. **QLoRA BFCL-aligned** — finish training; add metrics / optional BFCL compare to this doc  
2. **Full SFT** — optional completeness on ToolACE-8B  
3. **Meta Llama 3.1 Instruct base** — when gated access lands, true from-scratch vs ToolACE-8B A/B  
4. **Cost optimization** — only after product accepts current quality (FP8 Dedicated Endpoint)

---

## 15. Bottom line for reviewers

We did not chase leaderboard noise. We built a **shippable function-calling stack** on the assignment GPU, found a **silent training bug** that destroyed BFCL scores, fixed **train–eval contract alignment**, and showed that **BF16 LoRA + vLLM** already serves **16–32 concurrent** users with strong Python FC accuracy (~**71%** non-live / live). That is the Customer Solution Architect outcome: diagnose, fix, measure, map to Token Factory, and know what to do next.
