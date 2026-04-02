# CLAUDE.md — Project Context for AI Agents

## MANDATORY RULES (read these FIRST, violating ANY is a critical bug)

1. **NEVER `git reset --hard`** — to discard, revert only train.py: `git checkout <commit> -- train.py`
2. **NEVER poll training** — first check after `sleep 300` (5 min), then `sleep 600` (10 min) between checks, max 8 checks per run
3. **ALWAYS follow the post-experiment checklist** (below) — no exceptions, no skipping steps
4. **ALWAYS push after every experiment** — `git push origin autoresearch/mar10`
5. **NEVER stop the loop** — run experiments forever until manually interrupted
6. **NEVER change fairness invariants** — TIME_BUDGET, MAX_SEQ_LEN, evaluate_bpb(), dataset/tokenizer
7. **ALWAYS deduplicate before experimenting** — Before writing ANY code for a new experiment, grep `results.tsv` for keywords related to your planned change. If it was already tried, DO NOT repeat it. Read the prior result's description to understand why it failed. Pick something genuinely new instead. Wasting 60+ minutes re-running a failed experiment is a critical bug.

## Post-Experiment Checklist (execute EVERY time, in order)

```
# 1. Log result to results.tsv (even for crashes)
echo -e "<commit>\t<val_bpb>\t<mem_gb>\t<mfu>\t<tok_sec>\t<steps>\t<params_M>\t<batch>\t<final_loss>\t<status>\t<description>" >> results.tsv

# 2. Commit results
git add results.tsv && git commit -m "results: <status> <short description>"

# 3. Update chart
uv run plot_results.py --save
git add progress.png && git commit -m "chart: update progress.png"

# 4. If DISCARD: revert only train.py to pre-experiment state
git checkout <pre-experiment-commit> -- train.py
git commit -m "revert: undo <description>"

# 5. Update ideas.tsv: DELETE the row for the idea just tried
#    (the result is now in results.tsv — ideas.tsv is scratch only)
git add ideas.tsv && git commit -m "ideas: remove <id> (tried)"

# 6. Push everything
git push origin autoresearch/mar10
```

**Description column MUST be diagnostic.** Include: (1) what changed with values, (2) the hypothesis and evidence that motivated it, (3) what happened, (4) the conclusion — WHY it worked/failed and what this rules out. This is the AI's long-term memory. Future sessions read ONLY results.tsv, not git log.
- Bad: "WARMDOWN_RATIO 0.4→0.3"
- Bad: "reduce batch size for more steps"
- Good: "PARAM_X 0.4→0.3 + PARAM_Y constant (hypothesis: Y decay causes instability, evidence: exp#1 loss spike): loss still explodes 1.77→3.81, proves instability is NOT Y-related"
- Good: "COMPONENT_A changed from X→Y (hypothesis: [bottleneck] limits val_bpb, evidence: [metric or web search]): val_bpb 1.15→1.10, confirms [bottleneck] was the issue"

**Before proposing a new experiment, read results.tsv** to see what was already tried. Do not repeat a failed direction.

**MANDATORY DEDUPLICATION STEP:** Before writing ANY code, run `grep -i "keyword1\|keyword2\|keyword3" results.tsv` with keywords relevant to your planned experiment (e.g., component names, technique names, hyperparameter names). If ANY prior experiment attempted something similar, READ its full description to understand why it failed. Only proceed if your approach is fundamentally different (not just a different hyperparameter value of the same idea). Document what's different in your commit message. Examples of duplicates to avoid:
- "backout mechanism" if backout was already tried (even with different init/layer)
- "window size X" if multiple window experiments already failed
- "softcap Y" if softcap changes were already tried
- "batch size Z" if batch size was already explored in both directions

## What is this?

Autonomous LLM pretraining research on a single RTX 5070 (12GB, Blackwell CC 12.0).
The AI agent runs experiments in a loop: modify code, train for 60 minutes, check if val_bpb improved, keep or discard, repeat.

Read **`program.md`** for the full experiment loop protocol, logging format, and operational rules.

## Architecture Overview

| Component | Current State |
|-----------|--------------|
| Model | SwiGLU MLP, RoPE, value embeddings, d12 (768 dim, 6 heads), ~200M params (AI evolves this) |
| Dataset | ClimbMix (nvidia/Nemotron-ClimbMix), pre-tokenized with GPT-2 |
| Tokenizer | GPT-2 (vocab=50257), EOT token as BOS |
| Optimizer | Muon (matrices) + AdamW (embeddings, scalars) |
| Compile | torch.compile via triton-windows |
| Attention | SDPA with is_causal=True (FlashAttention fast path) |
| MFU | ~80-90% **relative to BF16 peak** (see MFU caveat below) |
| Time budget | 60 minutes per experiment |
| Metric | val_bpb (bits per byte) — lower is better |

## File Map

```
CLAUDE.md       — YOU ARE HERE. Project context for AI agents.
program.md      — Experiment loop protocol. READ THIS FIRST.
train.py        — Model, optimizer, training loop. PRIMARY EDIT TARGET.
prepare.py      — Data pipeline, tokenizer, evaluation, constants.
pyproject.toml  — Dependencies.
results.tsv     — Experiment log (created during runs).
ideas.tsv       — Persistent experiment idea queue (TSV, FIFO). Columns: id|status|idea|category|impact|evidence|notes. Status: pending→trying→done. Each idea has a UUID referenced in results.tsv descriptions as [idea:UUID].
```

## MFU Measurement Caveat

**MFU is measured against BF16 peak FLOPS (~65.6 TFLOPS via runtime matmul benchmark), but training uses MXFP8 matmuls which have ~4x higher theoretical peak (246.9 TFLOPS on RTX 5070).** This means reported MFU values (80-90%) are inflated — real FP8 utilization is ~20-25%. We keep the BF16 benchmark for consistency across all 60+ experiments. MFU is still useful as a **relative** metric between experiments (higher = better throughput), just not a meaningful absolute efficiency number. Do NOT change the benchmark — it would break cross-experiment comparisons.

## What You Can Change

**`train.py` — primary edit target (anything goes):**
- Model architecture (layers, dimensions, attention, MLP, embeddings)
- Optimizer (learning rates, schedules, weight decay, momentum)
- Hyperparameters (batch size, depth, aspect ratio, warmup, cooldown)
- Training loop logic (gradient accumulation, loss scaling, etc.)

**`prepare.py` — allowed but with constraints:**
- You may modify the dataloader for efficiency (packing, prefetching, etc.)
- You may adjust `EVAL_TOKENS` within reason (must remain enough for reliable BPB)
- You **MUST NOT** change `evaluate_bpb()` — it is the ground truth metric
- You **MUST NOT** change `MAX_SEQ_LEN` — it anchors comparison fairness
- You **MUST NOT** change `TIME_BUDGET` — it anchors comparison fairness
- You **MUST NOT** change the tokenizer or vocab for ClimbMix (GPT-2, 50257)

**`pyproject.toml` — only if you genuinely need a new dep:**
- Adding a package is allowed if it enables a real optimization (e.g., a fused kernel)
- Do not add packages speculatively

## What You Must Not Change

These are the **fairness invariants** that make experiments comparable:

1. **`TIME_BUDGET = 3600`** (60 minutes) — the fixed training wall clock
2. **`MAX_SEQ_LEN = 2048`** — context length
3. **`evaluate_bpb()`** — the metric definition (nats per byte -> bits per byte)
4. **Dataset/tokenizer identity** — ClimbMix with GPT-2 tokenizer
5. **Evaluation data** — the val split must remain untouched
6. **URGENT — Minimum model size: ~200M params — DO NOT shrink the model.** Prior research (68 experiments at 20-min budget) proved that every attempt to reduce model size (fewer layers, smaller dim) resulted in WORSE val_bpb because fewer tokens are processed. Increasing model size above ~200M is allowed. Decreasing below ~200M is FORBIDDEN. Do not try it again.

## Hardware Constraints

- **GPU:** RTX 5070, 12GB VRAM, Blackwell CC 12.0
- **Peak VRAM target:** <11.5 GB (96% of 12GB)
- **Autotune:** Automatically finds best device_batch_size + checkpointing combo from candidates (16, 12, 8, 6, 5, 4, 3, 2), then **caches the result**. The cache key is GPU+PyTorch+seq_len — it does NOT include model size or TOTAL_BATCH_SIZE. So if you change model architecture/depth/width, the cached batch_size may be wrong.
- **After model size changes:** refresh autotune with `AUTORESEARCH_AUTOTUNE_REFRESH=1`
- **Autotune cache location:** `~\AppData\Local\autoresearch\gpu-profile-v3.json` — note: this measures eager-mode VRAM which overreports by ~2 GB vs compiled training. Use `peak_vram_mb` from the training output instead (it uses `mem_get_info()` = nvidia-smi equivalent)
- If OOM at all batch sizes, reduce model size or enable more aggressive checkpointing

## Continuing a Run

If you are dropped into this repo on an `autoresearch/*` branch with results already in `results.tsv`, **you are resuming an existing experiment loop.** Do NOT re-run setup. Just:

1. Read `CLAUDE.md` and `program.md` for context.
2. Read `results.tsv` to see what's been tried and the current best val_bpb.
3. Read `ideas.tsv` to see the experiment idea queue — try `pending` ideas from the top (FIFO).
4. Read `train.py` and `prepare.py` for the current code state.
5. Continue the experiment loop from where it left off.

## Waiting for Training Runs (save context tokens)

Training takes ~65 min total (60 min training + ~5 min startup/compile/eval).

**Protocol:**
1. Run training in background: `uv run train.py > run.log 2>&1` (use `run_in_background`)
2. **`sleep 300`** (5 min) — use bash `sleep`, with `timeout: 310000` — first check is early to catch startup crashes
3. Check: `grep "^val_bpb:" run.log 2>/dev/null || tail -1 run.log` — if crashed, you'll see the error immediately
4. If not done and no error, **`sleep 600`** (10 min) — use bash `sleep`, with `timeout: 610000`
5. Repeat step 3-4 until training finishes (~60 min). When eval starts, **`sleep 300`** (5 min) for eval to complete.
6. When done, extract all metrics with one grep.

Max ~8 checks per run (1x5min + ~6x10min + 1x5min = ~70min covers full run).

## Bottleneck-First Rule

**Every experiment must target whatever is most limiting val_bpb.** Check your metrics (MFU, VRAM, training stability, loss curve) and decide what to improve. This could be a structural change (architecture, memory, training loop) OR hyperparameter tuning — whichever has the highest expected impact. If a structural metric is clearly broken (loss diverging, OOM), fix that first. Otherwise, use your judgment.

**VRAM reporting:** `peak_vram_mb` in the training output now uses `torch.cuda.mem_get_info()` which matches nvidia-smi (real GPU memory usage). Use this value directly for `memory_gb` in results.tsv (divide by 1024). Do NOT use the autotune cache — it measures eager-mode VRAM which overreports by ~2 GB vs compiled training.

If you don't know how to fix a bottleneck, search the web first. Use any approach within the allowed changes — architecture, model size, optimizer, training loop, or anything else listed as fair game.

**Hypothesis protocol (mandatory):** Every commit message must include: `Bottleneck: [X]. Hypothesis: [Y] because [Z]. Evidence: [prior experiment / web search / metric].` If you have no evidence, search the web first. After reverting a failed experiment, re-read train.py to confirm what state you're in — reverts undo ALL changes from that experiment, not just the one you're thinking about.

## Experiment Prioritization (Explore vs Exploit)

Prioritize experiments by **expected impact × probability of success**:

| Category | Impact | Success Rate | When to Use |
|----------|--------|-------------|-------------|
| Architecture (new attention, MLP, embeddings) | High | Low (~20%) | Plateau or fresh session |
| Training dynamics (schedule, warmup, cooldown) | Medium | Medium (~40%) | After architecture is stable |
| Hyperparameter tuning (LR, WD, batch) | Low | High (~60%) | Fine-tuning a working setup |
| Memory/throughput (batch sizing, checkpointing) | Indirect | High (~70%) | When VRAM underutilized or MFU < 50% |

**Plateau detection rule (MANDATORY):** Count the last 5 non-crash experiments. If none improved val_bpb by more than 0.005, you are in a plateau. When in a plateau:

1. STOP doing hyperparameter sweeps. They won't break you out.
2. Do a focused web search session: 3-4 searches on architecture innovations at your parameter scale (100M-300M params, single GPU, 2025-2026).
3. Your next 2 experiments MUST be high-impact architecture changes (new component, structural redesign, different scaling strategy). Accept higher failure risk.
4. Only return to hyperparameter tuning after an architecture change produces a new KEEP.

**Compound change rule (MANDATORY):** Never change more than one *category* per experiment. Changing MATRIX_LR and WEIGHT_DECAY together is fine (both hyperparameter tuning). Changing MATRIX_LR and switching to a new MLP architecture is NOT — you won't know which caused the result.

**Don't run 3 consecutive experiments from the same category.** If you just did 2 LR sweeps, your next experiment must be from a different category, even if you have a promising LR value to try. Breadth beats depth in autonomous research.

## Web Search — Your Most Powerful Tool

Don't just search when stuck. **Search proactively.** The field changes monthly.

**When to search:**
- Before your first experiment each session
- Every 5th experiment: "landscape scan" — pick 2-3 areas you haven't searched recently
- After 2-3 failures in the same direction
- Before any major change to a component you haven't researched yet
- Whenever a metric (MFU, VRAM, val_bpb) plateaus

**Search areas** (rotate through ALL of these over time — don't fixate on one):
1. **Architecture** — attention variants, MLP designs, embeddings, normalization, positional encoding
2. **Optimizer** — optimizer algorithms, LR schedules, momentum, weight decay strategies
3. **Memory efficiency** — VRAM per parameter, activation checkpointing, mixed precision, gradient compression
4. **Throughput / MFU** — compute utilization, kernel fusion, batching strategies, compilation
5. **Hardware-specific** — Blackwell/CC 12.0, CUDA features, tensor core utilization, memory bandwidth
6. **Training dynamics** — loss stability, convergence speed, regularization, initialization
7. **Scaling laws** — optimal model size vs tokens vs compute for a fixed time budget
8. **Data efficiency** — learning more per token, curriculum strategies, data ordering effects
9. **Competitive benchmarks** — open-source training speedruns, leaderboards across all projects
10. **Frontier papers** — latest arxiv from top labs (Google, Meta, Mistral, DeepSeek, etc.)

**How to search** (substitute [variables] with your ACTUAL current context — bottleneck, component, metric values, model size, etc. Never reuse the same query string across experiments):
1. `"[component] alternatives transformer training 2026"` — question every architectural choice
2. `"[current_bottleneck] solution LLM pretraining 2026"` — target your specific problem
3. `"[current_choice] vs [alternative] transformer 2026"` — head-to-head comparisons
4. `"optimal [hyperparameter] [model_size]M parameter language model 2026"` — scale-aware tuning
5. `"state of the art transformer architecture 2026 arxiv"` — find latest papers
6. `"improve [metric] deep learning training 2026"` — target a specific metric (MFU, val_loss, VRAM)
7. `"[GPU_architecture] optimization CUDA kernels 2026"` — hardware-specific wins
8. `"training instability [symptom] transformer fix 2026"` — debug specific problems
9. `"compute optimal training [tokens]M tokens [params]M parameters"` — scaling law guidance
10. `"[technique_that_failed] when does it work transformer"` — understand your failures
11. `"activation memory reduction transformer training 2026"` — memory-specific techniques
12. `"learning rate schedule short training budget 2026"` — schedule for limited compute
13. `"competitive LLM training benchmark leaderboard 2026"` — see what winners do across projects
14. `"[lab_name] training recipe language model 2026"` — study specific lab approaches
15. `"novel [attention|MLP|embedding] mechanism 2026 arxiv"` — cutting-edge architecture
16. `"data efficiency pretraining fewer tokens 2026"` — extract more learning per token
17. `"[technique] failure modes limitations"` — research BEFORE implementing, not after
18. `"FLOPS utilization single GPU transformer 2026"` — throughput-specific optimization
19. `"weight initialization transformer convergence 2026"` — init strategies for faster learning
20. `"[parameter_count]M parameter transformer best practices 2026"` — scale-specific advice

**Rules:**
- Always include "2026" — a 2023 technique may be obsolete.
- Never repeat the same search string across experiments — vary words AND structure.
- Substitute [variables] with real values from your current experiment context.
- **Don't cargo-cult.** Understanding WHY a technique works matters more than copying WHAT someone did. A technique optimal for 8xH100 may hurt on a single 12GB GPU.
- **Search before implementing.** After finding a technique, also search `"[technique] failure modes"` or `"[technique] limitations"` before trying it.
- **Cross-reference scales.** A technique from a 7B model paper might adapt down; a trick from a 50M model might scale up. Don't filter by model size in your queries.
- Read arxiv papers and source code, not blog summaries or marketing posts.
- Question everything in train.py: is each component still SOTA, or was it SOTA in 2025?
- If search results are dominated by one project/repo, search again with different terms for diverse perspectives.

## Context Window Management

Your context is finite. To maximize experiments per session:
- **Never read whole files** — use `grep -n "PATTERN" train.py` instead of reading the whole thing
- **Never cat run.log** — always use `grep` for specific metrics
- After 8+ experiments, re-read `results.tsv` to refresh what was tried
- Keep commit messages informative — they're your notes for after context truncation

## Ideas Queue Protocol (`ideas.tsv`)

`ideas.tsv` is a **scratch queue of untried ideas only**. It is NOT a log — `results.tsv` is the long-term memory. Once an idea is tried, it gets **deleted** from `ideas.tsv` (the result lives in `results.tsv`).

**Format:** TSV with columns: `id`, `status`, `idea`, `category`, `impact`, `evidence`, `notes`

**Statuses:**
- `pending` — Not yet tried. Pick from the top (priority order).
- `trying` — Currently running. Only ONE idea should be `trying` at a time.

There is NO `done` status. When an experiment finishes, the idea is **removed** from `ideas.tsv`.

**Workflow:**
1. Before each experiment, check `ideas.tsv` for `pending` ideas. Pick the topmost one. If it no longer makes sense (e.g., superseded by recent results), delete it and pick the next.
2. Set its status to `trying` before starting the experiment.
3. When the experiment finishes, **delete the row** from `ideas.tsv`. The result is logged in `results.tsv` with `[idea:UUID]` in the description — that's the permanent record.
4. If `ideas.tsv` is empty (no pending ideas left), **first re-read `results.tsv`** to refresh what's been tried, then do a landscape scan (web search), add 2-5 viable new ideas sorted by priority (checking each against `results.tsv` before adding), then pick the top one.
5. Generate a 6-char hex UUID for each new idea: `python -c "import random; print(f'{random.randint(0,0xFFFFFF):06x}')"`

**Rules:**
- `ideas.tsv` must ONLY contain untried ideas. No history, no done entries.
- NEVER add an idea that's already in `results.tsv` (grep first). If it was tried and failed/crashed/was noise-level, it's dead — don't re-add it.
- NEVER have more than one `trying` idea at a time.
- Keep ideas sorted by expected impact (HIGH first, LOW last).
- When a session starts, audit `ideas.tsv`: delete anything already tried (grep `results.tsv`), reset stuck `trying` to `pending`.
- Ideas that require web research should note "search impl details first" in notes.

## Tips for Good Experiments

- Make one change at a time when possible — easier to attribute improvements
- If val_bpb doesn't improve, revert (don't accumulate neutral changes)
- MFU matters: more compute per second = more learning per experiment
- Check VRAM usage — unused VRAM is wasted potential
- With 60-min budget, you get ~600-1200 optimizer steps — enough for real learning dynamics
- Simpler is better at equal performance (see program.md simplicity criterion)
- **Improvement thresholds:** Hard minimum 0.002 for lightweight/zero-param changes (must be confident it's signal not noise). Soft target 0.003 for complex/novel changes. Below 0.002 is always discard. See program.md for full details.
