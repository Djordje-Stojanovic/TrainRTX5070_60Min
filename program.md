# autoresearch

This is an experiment to have the LLM do its own research.

> **Start here:** Read `CLAUDE.md` for project context, architecture overview, file map, and fairness rules.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `CLAUDE.md` — project context, what you can/can't change, fairness rules.
   - `train.py` — model architecture (SwiGLU d12, ~162M params), optimizer (Muon+AdamW), training loop.
   - `prepare.py` — data pipeline (ClimbMix + GPT-2 tokenizer), dataloader, evaluation.
4. **Verify data exists**: Check that the cache directory contains ClimbMix shards and a tokenizer. If not, tell the human to run `uv run prepare.py --dataset climbmix`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single RTX 5070 (12GB). The training script runs for a **fixed time budget of 60 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN change** (see `CLAUDE.md` for details):
- `train.py` — primary edit target. Architecture, optimizer, hyperparameters, training loop, batch size, model size. Everything is fair game.
- `prepare.py` — dataloader efficiency improvements are allowed. But do not touch the fairness invariants (see below).
- `pyproject.toml` — add a dependency only if it enables a real optimization.

**Fairness invariants** (DO NOT change these — they make experiments comparable):
- `TIME_BUDGET = 3600` (60 minutes)
- `MAX_SEQ_LEN = 2048`
- `evaluate_bpb()` function definition
- Dataset identity (ClimbMix) and tokenizer (GPT-2, vocab 50257)
- Validation split data
- **URGENT — Minimum model size: ~200M params.** DO NOT shrink the model below ~200M params. Prior research (68 experiments at 20-min budget) definitively proved smaller models are worse. Increasing above ~200M is allowed. Decreasing is FORBIDDEN.

**The goal is simple: get the lowest val_bpb on ClimbMix.** Since the time budget is fixed, you don't need to worry about training time — it's always 60 minutes. Everything else is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a hard constraint at 12GB. The autotune system handles batch sizing automatically. If your architecture change OOMs at all batch sizes, scale back.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**Minimum improvement threshold**: val_bpb improvements smaller than **0.003** are not reliably distinguishable from run-to-run variance (measured σ ≈ 0.0014–0.0020 across 4 identical runs). Don't keep a change that improves val_bpb by less than 0.003 unless it also simplifies the code.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 3600.1
total_seconds:    1325.9
peak_vram_mb:     7000.5
eval_vram_mb:     4500.3
mfu_percent:      30.50
total_tokens_M:   84.0
num_steps:        320
num_params_M:     162.0
depth:            12
dataset:          climbmix
```

`peak_vram_mb` is **real GPU VRAM** during training (measured via `torch.cuda.mem_get_info()`, matches nvidia-smi). Use this value divided by 1024 for the `memory_gb` column in results.tsv.

**MFU caveat:** `mfu_percent` is measured against **BF16 peak FLOPS** (~65.6 TFLOPS), but training uses MXFP8 which has ~4x higher theoretical peak. Reported MFU values (80-90%) are relative to BF16 only — useful for comparing experiments, not as absolute efficiency. Do not change the benchmark mid-run.

Note that the script is configured to always stop after 60 minutes. You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 11 columns:

```
commit	val_bpb	memory_gb	mfu	tok_per_sec	num_steps	num_params_M	batch_size	final_loss	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak training VRAM in GB, round to .1f (e.g. 7.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. mfu percent (e.g. 24.3) — GPU compute efficiency — use 0.0 for crashes
5. tok_per_sec — throughput (e.g. 37000) — use 0 for crashes
6. num_steps — optimizer steps completed in 60 min — use 0 for crashes
7. num_params_M — model parameter count in millions (e.g. 162.1) — use 0.0 for crashes
8. batch_size — device batch size selected by autotune — use 0 for crashes
9. final_loss — training loss at last step (e.g. 1.850) — use 0.000 for crashes
10. status: `keep`, `discard`, or `crash`
11. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	mfu	tok_per_sec	num_steps	num_params_M	batch_size	final_loss	status	description
a1b2c3d	1.150000	7.0	24.3	37000	157	162.1	4	1.850	keep	baseline (SwiGLU d12 ClimbMix)
b2c3d4e	1.142000	7.2	25.1	38500	163	162.1	4	1.820	keep	increase LR to 0.05
c3d4e5f	1.160000	7.0	24.0	37000	157	162.1	4	1.900	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	0.0	0	0	0.0	0	0.000	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

0. **Plan:** Read `results.tsv` (what's been tried) AND `ideas.md` (experiment idea queue). Pick the top idea from the queue if one exists, otherwise identify what is most limiting val_bpb (see Bottleneck-First Rule in `CLAUDE.md`). Your next experiment must target the highest expected impact change. Write your commit message with: `Bottleneck: [what's limiting val_bpb]. Hypothesis: [change] will improve because [reason]. Evidence: [prior experiment / web search / metric].` If you have no evidence, search the web first. **Every 5th experiment**, do a landscape scan across ALL search areas listed in `CLAUDE.md` — architecture, training, optimizer, hardware, memory, throughput, frontier techniques. Don't get tunnel-visioned on one area. After a landscape scan, add any promising NEW ideas to the bottom of `ideas.md` (check `results.tsv` first to avoid duplicates).

   **MANDATORY DEDUPLICATION CHECK (before EVERY experiment):**
   Before writing any code, grep `results.tsv` for keywords related to your planned change (e.g., `grep -i "backout\|window\|batch.*size\|softcap"` etc.). If a similar experiment was already tried, DO NOT repeat it — the result will be the same. Read the description of the prior attempt to understand WHY it failed, then either (a) pick a genuinely different experiment, or (b) if you believe the prior attempt was flawed in a specific way you can fix, document exactly what's different this time in your commit message. "Same idea but different hyperparameter" counts as a repeat if the prior experiment's conclusion rules it out.
1. `git pull origin autoresearch/mar10` — pick up any doc updates pushed between experiments.
2. Make your experimental change (primarily `train.py`, but other files if needed per the rules in `CLAUDE.md`).
3. git commit (with the hypothesis from step 0 in the message)
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~60 minutes total (+ a few minutes for startup/compilation and eval overhead). If a run exceeds 75 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~65 minutes then you can run approx 1/hour, for a total of about 8 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
