# 03 — Experiment plan

Same discipline as the predecessor repo: **cheapest falsification first**, explicit
gates, binding kill criteria.

## Hypotheses

- **H1 (offline acceptance).** On real agentic-coding traces at ≥8K context, a
  composite lookup+MTP drafter reaches **≥2.0× tokens per target-model pass**, measured
  against the target model's *own* greedy output.
- **H2 (end-to-end).** That converts to **≥1.5× measured wall-clock decode speedup**
  over dense greedy at matched context, with output token-identical to dense greedy.
- **H3 (composition).** Lookup and MTP are complementary rather than redundant: the
  composite beats the better of the two alone by **≥15%** in tokens per pass.

## Target metric

Largest **lossless speedup over the dense baseline at matched context**. No absolute
tok/s gate — context length moves the baseline too much (17.6 tok/s at 256, 12.5 at
16K) for an absolute number to mean anything. Every result is reported as a ratio at
256 / 8K / 16K / 32K.

## Phase A0 — offline acceptance (no decode loop) → `analysis/`

Can falsify the thesis in half a day.

1. **Rebuild the trace corpus at full agentic context.** ≥24K prefix budget, tool
   results untruncated. The predecessor's 2048-token cap understated lookup by 28%
   (1.57× → 2.01×), so this is not a detail.
2. **Generate the target model's own greedy continuation** for each trace prefix
   (~512 tokens, dense, no speculation), once, and freeze it. *This is the
   methodological fix that A0 exists for*: acceptance must be measured against what
   Qwen3.8 actually emits, not against the human transcript used as a stand-in.
3. **Replay drafters offline** against those frozen streams: lookup (sweep `n`,
   `kmax`, policy), MTP, and the composite. Report tokens per pass, hit rate, accepted
   length distribution, and the **overlap** between lookup-wins and MTP-wins.
4. Split results by context bucket and by language.

**Gate:** H1 — composite ≥2.0× at ≥8K context → proceed to A1.
**Kill:** <1.5× → speculation via these two sources is not the lever; stop and write up.
**Within noise:** stop and request review.

## Phase A1 — end-to-end decode loop → `src/`, `bench/`

1. Wire the winning A0 configuration into a real decode loop on
   `mlx_vlm.speculative`, using the **disjoint-fallback** merge only (`docs/02` C2.1).
2. Assert token-identical output vs. dense greedy on every trace — this is the
   correctness gate, and it is not optional.
3. Measure at 256 / 8K / 16K / 32K context: tok/s, ms/token, E[accepted], hit rate,
   and the **draft-vs-verify wall-clock split** (this is what tests the
   "Performance or Illusion?" critique on our hardware rather than assuming it away).

**Gate:** H2 — ≥1.5× over dense at matched context.
**Kill:** <1.25× after the obvious overhead fixes → verification overhead ate the win;
publish the negative result with the draft/verify split as evidence.

## Phase A2 — tuning → `src/`, `bench/`

1. Budgeted merge into `build_ddtree` with calibrated synthetic log-probs (C2.2).
2. Dynamic draft length conditioned on recent hit state.
3. Tree shape sweep.

**Gate:** ≥10% over A1's best. Below that, stop — the simple version is the product.

## Phase B — lossy levers (only if A plateaus)

Not started unless A1/A2 land short. Requires building the perplexity + EvalPlus
harness that Phase A deliberately avoids. Candidates in priority order: mixed-precision
quantization (~1.2–1.3×), training-free layer-skip drafting (SWIFT/CLaSp, 1.3–1.7×
reported). Each needs its own quality gate at ≤~1% degradation.

## Controls & hygiene

- **Lossless assertion on every run.** Speculative output must match dense greedy token
  for token. A run that fails this is void, not "slightly degraded".
- Fixed seeds; greedy decoding throughout (speculation is exact, so greedy is the only
  regime where "identical output" is even definable).
- Identical model artifacts across all configurations.
- 3 runs per timing cell, report median; ≥2-min cooldown; plugged in, High Power mode.
- Every run writes a `results/` entry: config, environment (macOS build, engine
  versions, `iogpu.wired_limit_mb`, power state), raw numbers, median, gate decision.
- Deterministic non-timing statistics may be run once if proven byte-identical across
  two runs; record the proof.
- Never commit weights, traces, generated streams, or raw dumps.

## Kill criteria (explicit and binding)

- **A0:** composite < 1.5× tokens per pass at ≥8K context → **project dead**.
- **A1:** < 1.25× measured end-to-end after overhead fixes → **project dead**;
  publish the draft/verify split as the negative result.
- **A2:** < 10% over A1 → stop tuning, ship A1's configuration.
- **Any phase:** speculative output ever diverges from dense greedy → **stop
  immediately**; this is a correctness bug, not a tuning parameter.
