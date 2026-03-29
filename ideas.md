# Ideas Tasklist — Experiment Queue

> **How to use:** Try ideas from the TOP of the list first (FIFO). Add new ideas at the BOTTOM.
> Before adding, ALWAYS check `results.tsv` to ensure the idea hasn't been tried already.
> After trying an idea, move it to the "Completed" section with the result.
> This file persists across sessions so ideas aren't lost.

## Queue (try from top)

1. **MUD optimizer** (replace Muon) — March 2026 paper (arxiv 2603.17970) reports 10-50% wall-clock speedup over Muon with 1.3-2.6x peak tok/s. Momentum Decorrelation uses cheaper orthogonalization. HIGH IMPACT but invasive. Search for implementation details before coding.

2. **Blockwise LR multipliers** — ICML 2025 paper (arxiv 2502.19002) found sharpness disparity across transformer layers. Per-layer LR scaling (e.g., linear ramp from 0.5x to 1.5x across depth) gives lower terminal loss and ~2x speedup. Compatible with existing Muon+AdamW split. MEDIUM IMPACT, low risk.

3. **GEGLU activation** (GeLU replacing SiLU in SwiGLU gate) — One-line change: `F.gelu(gate) * up` instead of `F.silu(gate) * up`. Never tested. Zero throughput cost. LOW IMPACT but zero risk.

4. **Nesterov momentum for Muon** — Change Muon's momentum computation to Nesterov-style look-ahead. Different from changing momentum VALUE (already tried 0.95→0.975, failed). Nesterov is a different ALGORITHM that should converge faster. MEDIUM IMPACT.

5. **Cyclical LR / warm restarts** — Split training into 2-3 cycles with mini warmup→decay phases. Never tried any multi-cycle schedule. Could help escape local minima. MEDIUM IMPACT but risky (less warmdown time per cycle).

6. **Sequence packing in dataloader** — Check if prepare.py wastes tokens via padding. If sequences are shorter than 2048, packing multiple sequences per slot increases effective tokens seen. MEDIUM IMPACT if padding is significant.

7. **Per-head learned temperature** (init=1.0, not 0.2) — Exp #77 tried attn_scale init=0.2 (way too low, made attention uniform). A per-head scale initialized at 1.0 with small LR could allow heads to sharpen/soften individually. Different experiment entirely.

8. **Cosine attention** (normalize Q,K to unit vectors, use cosine similarity) — Different from current RMSNorm QK-norm. Fully normalizes to unit sphere removing magnitude information entirely. Could improve or hurt.

## Completed

_(Move ideas here after trying, with exp# and result)_

- **Dual-stream EMA residual** (Hyper-Connections inspired) — Exp #118 (running), added secondary EMA stream with learnable mixing. Pending result.
- **QK-Norm** — Already in model (exp #32 era), `q, k = norm(q), norm(k)` after RoPE
- **Logit Softcapping** — Already in model, softcap=15
- **WSD Schedule** — Already essentially WSD (5% warmup, 25% stable, 70% cosine decay)
- **Peri-LN / HybridNorm** — Already in model (exp #22, +0.0055)
- **Vocab padding 50257→50272** — Exp #115, CRASHED (MXFP8 contiguity error)
- **Differential attention** — Exp #14 and #76, catastrophically worse both times
