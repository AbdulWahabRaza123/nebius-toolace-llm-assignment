# Nebius Take-Home — Presentation Journey Document

**Role:** Senior ML Solutions Architect (Customer Solution Architect)  
**Exercise:** FinTech function-calling agent — ToolACE train → BFCL Python eval → production-style serve on 1×H100  
**Repo:** https://github.com/AbdulWahabRaza123/nebius-toolace-llm-assignment  
**Companion metrics sheet:** [REPORT.md](REPORT.md)

This document is the **final narrative for the 15–20 minute demo**: what we built, why we chose it, what broke, what we fixed, and what we would ship.

**Also includes full Assignment Q&A (Section 10)** answering every take-home task item (1a–1d, 2a–2c, deliverables), plus curl smoke tests (Section 12).

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
- **Status:** training in progress (~step 200/1342); mid-run `eval_loss ≈ 0.441`, mean token acc ≈ **88.8%**
- *Final update slot:* train loss ___ / eval loss ___ / BFCL non-live ___ (fill when run completes)

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

## 10. Assignment Q&A (answer every brief item)

Use this section verbatim in the demo if asked. Each item maps to the take-home text.

### Task 1 — Fine-tuning & evaluation

**Q1a. Which model did you fine-tune, and why is it most suitable for the client?**

**A:** We fine-tuned **`Team-ACE/ToolACE-8B`** (Llama 3.1 8B Instruct lineage already specialized for ToolACE-style function calling).

Why it fits a FinTech FC agent on 1×H100 / Token Factory:

- Same backbone family as published ToolACE work → strong function-calling prior
- **8B** is a practical Token Factory Dedicated Endpoint size (quality vs cost)
- Fits LoRA / QLoRA / full SFT on **1×H100 80GB**
- Intended Meta `Meta-Llama-3.1-8B-Instruct` was **HF-gated** and not downloadable in time; ToolACE-8B is the honest production substitute with the same architecture

**Q1b. What fine-tuning methods did you try, and which is most suitable?**

**A:** Method matrix:

| Method | Backbone | Outcome | Suitable for client? |
| --- | --- | --- | --- |
| QLoRA (smoke) | Qwen2.5-7B | Pipeline validation; eval loss 0.439 | No — wrong backbone for final ship |
| LoRA BF16 (broken targets) | ToolACE-8B | Trained OK, BFCL ~15–19% simple | No — silent format bug |
| **LoRA BF16 (BFCL-aligned)** | ToolACE-8B | **Production pick**; BFCL ~71% non-live/live | **Yes** |
| QLoRA BFCL-aligned | ToolACE-8B | Running for cost/memory comparison | Contender if GPU memory / $ tight |
| Full SFT | ToolACE-8B | Config ready; optional | Only if LoRA plateaus |

**Most suitable today: BFCL-aligned LoRA BF16** — best measured quality, simple merge+serve path, meets latency without quant.

**Q1c. What best practices did you apply for quality (without endless HPO)?**

**A:**

1. Official ToolACE + official BFCL Python harness (no toy data)
2. **Train ↔ eval contract alignment** (BFCL Llama-3.1-FC JSON `{"name","parameters"}` + matching prompt render)
3. Diagnosed empty generations (~71% empty on simple_python) instead of blind retraining
4. Patched broken ToolACE-8B chat template / `tool_calls` handling
5. Assistant-target validation; longer serve context (`max_model_len=32768`)
6. Seeded configs, reproducible scripts, merge LoRA before production serve
7. Prioritized accuracy over premature quantization

**Q1d. Where are reproducible scripts and metrics for all models?**

**A:**

- Code: private GitHub `nebius-toolace-llm-assignment` (`src/`, `scripts/`, `configs/`)
- Metrics: [REPORT.md](REPORT.md) + this document
- One-shot path: `bash scripts/06_fix_accuracy_pipeline.sh`
- Train: `python src/train/sft.py --config configs/train_{lora,qlora,full}.yaml`
- Eval: `bash scripts/05_run_bfcl.sh`
- Bench: `python src/bench/latency.py --config configs/bench.yaml`

### Task 2 — Optimization, deployment & benchmarking

**Q2a. How did you optimize throughput / latency?**

**A:**

- Serve with **vLLM** (continuous batching, PagedAttention)
- **BF16** (no accuracy hit from low-bit quant)
- **Prefix caching** enabled
- `max_num_seqs=64`, `gpu_memory_utilization=0.92`
- Merged LoRA into a single HF checkpoint (no runtime adapter overhead)
- Native tool parser `llama3_json`

We did **not** quantize: accuracy is first priority and BF16 already meets the concurrency SLA.

**Q2b. How did you deploy for production (Token Factory), and what are BFCL Python results?**

**A — Deployment (Token Factory analogue):**

| PoC | Token Factory production |
| --- | --- |
| Merged weights on disk | Upload to Custom Weights Hub |
| `vllm serve` on 1×H100 | Dedicated Endpoint, 1 GPU/replica |
| OpenAI `/v1/chat/completions` + tools | Same client contract |

Serve script: `bash scripts/04_serve_merged.sh` → model id `llama-3.1-8b-toolace` on port **8000**.

**A — BFCL Python (fixed production model):**

- Non-live overall **71.06%** | Live overall **70.98%**
- Simple AST **66.75%** | Multiple **87%** | Parallel **91%** | Parallel Multiple **84%**
- Do not lead with aggregate Overall Acc **22.94%** (diluted by out-of-scope categories e.g. multi-turn)

**Q2c. 16–32 concurrent — TTFT and latency?**

**A:** Measured on the deployed merged LoRA (tool-calling prompts, 32 reqs/level):

| Concurrency | TTFT p95 | E2E p95 | RPS |
| --- | --- | --- | --- |
| 16 | **85 ms** | **229 ms** | 67.6 |
| 32 | **111 ms** | **262 ms** | 114 |

Errors: **0**. Client concurrency target is met in BF16.

### Deliverables & process questions

**Q. What do you ship as deliverables?**

**A:** Training / eval / deploy code; reported train + BFCL + latency metrics; model choice explanation — all in GitHub + `PRESENTATION.md` / `REPORT.md`.

**Q. Did you quantize?**

**A:** No. Scripts/path exist conceptually for later FP8; skipped because BF16 already hits latency/throughput and accuracy ranks first. We would introduce FP8 only after product accepts current quality and wants lower $/token.

**Q. Why was the first BFCL score so low if training loss looked fine?**

**A:** ToolACE-8B chat template dropped `tool_calls` → empty SFT targets → model learned to stop. Loss can look healthy while eval format is wrong. Fixing alignment lifted simple AST from ~19.5% → **66.75%** and non-live/live to ~**71%**.

**Q. What would you tell the FinTech client as next steps?**

**A:** Ship LoRA merged endpoint now; A/B Meta base when gated access lands; finish QLoRA comparison for cost; add FP8 if cost becomes priority; monitor live tool schema drift.

### Extra demo Q&A (likely interviewer follow-ups)

| Question | Answer |
| --- | --- |
| Why Llama/ToolACE-8B not a 70B? | 1×H100 budget; Token Factory cost; 8B enough when format-aligned |
| LoRA vs full SFT? | LoRA wins on iterate/serve simplicity; full SFT only if quality plateaus |
| How do you know BFCL is the right proxy? | Assignment-mandated Python subset; mirrors tool-call correctness |
| Live vs non-live gap? | Live is harder / noisier; we still ~71% overall live |
| Security for FinTech APIs? | AuthZ outside the LLM; allowlisted tools; schema validation; never trust raw generations |
| Observability? | Log tool names/args, TTFT, error rates; canary new adapters |
| How to promote to Token Factory? | Upload `merged-lora-bfcl` → Dedicated Endpoint → same OpenAI client |

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

## 12. Curl smoke tests (after vLLM is up)

Start serve (only when GPU is free — after QLoRA finishes):

```bash
bash scripts/04_serve_merged.sh
```

**List models**
```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
```

**Plain chat**
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-toolace",
    "temperature": 0,
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "What is 2+2? Reply briefly."}
    ]
  }' | jq .
```

**Tool calling (deployment test)**
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-toolace",
    "temperature": 0,
    "max_tokens": 256,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "messages": [
      {"role": "user", "content": "What is the weather in Berlin right now?"}
    ]
  }' | jq .
```

From laptop: `ssh -L 8000:127.0.0.1:8000 nebius-assignment@89.169.99.143` then same curls to `http://127.0.0.1:8000`.

---

## 13. Suggested 15–20 minute demo arc

| Min | Beat |
| --- | --- |
| 0–2 | Client: precise API calls; accuracy → latency → cost; 1×H100 |
| 2–4 | Why ToolACE-8B / 8B / Token Factory Dedicated Endpoint |
| 4–7 | Journey: pipeline → first LoRA → “good loss, bad BFCL” |
| 7–10 | Root cause: empty `tool_calls` / format mismatch; show before/after table |
| 10–13 | Fixed results: ~71% non-live/live; multi/parallel AST |
| 13–16 | Serve map to Token Factory; latency @ 16–32; why no quant |
| 16–18 | Method matrix status (LoRA win; QLoRA comparison) |
| 18–20 | Assignment Q&A (Section 10) |

---

## 14. Deliverables checklist vs assignment

| Deliverable | Status |
| --- | --- |
| Fine-tune suitable model + explain choice | Done (Q1a) |
| Experiment with multiple FT methods | LoRA done; Qwen QLoRA done; ToolACE QLoRA running; full SFT optional (Q1b) |
| Best practices for quality | Done (Q1c) |
| Reproducible train/eval/serve scripts | Done (Q1d) |
| Report training + BFCL + latency | Done in REPORT + this document |
| Optimize inference | Done (Q2a) |
| Deploy production-style + BFCL Python | Done (Q2b) |
| Measure TTFT/latency @ 16–32 concurrent | Done (Q2c) |
| Quantization scripts | Justified skip — BF16 meets SLA |

---

## 15. Open items (honest close)

1. **QLoRA BFCL-aligned** — in progress (~15% at step 200; mid-run eval_loss ≈ **0.441**, token acc ≈ **88.8%**); fill final numbers when complete  
2. **Full SFT** — optional completeness on ToolACE-8B  
3. **Meta Llama 3.1 Instruct base** — when gated access lands, true from-scratch vs ToolACE-8B A/B  
4. **Cost optimization** — only after product accepts current quality (FP8 Dedicated Endpoint)

---

## 16. Bottom line for reviewers

We did not chase leaderboard noise. We built a **shippable function-calling stack** on the assignment GPU, found a **silent training bug** that destroyed BFCL scores, fixed **train–eval contract alignment**, and showed that **BF16 LoRA + vLLM** already serves **16–32 concurrent** users with strong Python FC accuracy (~**71%** non-live / live). That is the Customer Solution Architect outcome: diagnose, fix, measure, map to Token Factory, and know what to do next.

**For the presentation:** speak from Sections **1 → 5 → 8 → 9 → 10**; keep curls (Section **12**) for a live smoke demo after vLLM is restarted.
