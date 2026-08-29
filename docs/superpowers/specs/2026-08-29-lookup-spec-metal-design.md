# Design — multi-source speculative decoding for agentic coding on Apple Silicon

Date: 2026-08-29 · Status: approved, A0 in progress

## Problem

Dense 4-bit decode of Qwen3.8-27B on an M3 Max 36 GB runs at 17.6 tok/s, reading
14.42 GB of weights per token at **254 GB/s effective against a 251 GB/s measured
streaming roof**. The machine is saturated. Two prior levers are closed: activation
sparsity (falsified in the predecessor repo — union sparsity 10.8% at k=4 where ≥30%
was needed) and kernel optimization (no headroom exists).

Under a lossless constraint the only remaining lever is **tokens emitted per weight
pass**.

## Approach

Speculative decoding with a **composite drafter**:

- **Lookup** — n-gram match against `prefix + generated`, proposing the continuation
  that followed the most recent earlier occurrence. Free (no weights read).
  Workload-specific: measured 2.01× on agentic traces vs 1.24× on prose.
- **MTP** — the model's own 1-layer draft head, 0.239 GB, 6.6% of a verify pass
  including `lm_head`. Workload-agnostic. Already implemented in `mlx-vlm`.

Verified by `Qwen3_5ExactSpeculativeVerifier`, whose singleton-equivalent numerics make
k-token verification bit-equivalent to k single-token decodes. **Correctness is
therefore a token-identity assertion against dense greedy**, not an eval suite.

## Components

| # | Component | Location | Notes |
|---|---|---|---|
| 1 | `LookupDrafter` | `src/lookup.py` | n-gram match; params `n`, `kmax`, policy. Pure Python, no model. |
| 2 | Candidate merge | `src/compose.py` | A1: disjoint fallback (lookup if hit, else MTP). A2: budgeted merge into `build_ddtree` with calibrated synthetic log-probs. |
| 3 | Offline simulator | `analysis/replay.py` | Replays any drafter config against a frozen generation. Makes the A0 sweep free. |
| 4 | Trace corpus | `analysis/build_traces.py` | Ported from the predecessor repo; prefix budget raised to ≥24K, tool results untruncated. |
| 5 | Generation freezer | `analysis/generate.py` | One dense greedy continuation per trace, frozen to disk (gitignored). |
| 6 | Bench harness | `bench/run.py` | Matched-context timing; env capture and `results/` discipline ported from the predecessor. |

## Data flow

```
session logs ──► build_traces.py ──► traces/*.json          (gitignored: private code)
                                          │
                                          ├─► generate.py ──► generated/*.json  (frozen greedy stream)
                                          │                        │
                                          │                        ▼
                                          │                   replay.py ──► analysis/summary/*.json
                                          │                   (sweep n, kmax, policy, MTP, composite)
                                          │                        │
                                          │                        ▼  winning config
                                          └────────────► compose.py + lookup.py
                                                                   │
                                                                   ▼
                                                       mlx_vlm.speculative verifier
                                                                   │
                                                                   ▼
                                                          bench/run.py ──► results/*.md
```

## Interfaces

`LookupDrafter.propose(context_ids, generated_ids) -> list[int]` — returns up to
`kmax` draft tokens, empty on miss. No model access, no state beyond its parameters.

`Composer.propose(context_ids, generated_ids, mtp_fn) -> list[int]` — calls the lookup
drafter first; on miss, delegates to `mtp_fn`. A1 keeps this deliberately trivial so
that the *measurement*, not the merge policy, is what A1 tests.

`replay(stream, drafter) -> AcceptanceStats` — pure function of a frozen token stream
and a drafter. No model. This is what makes the A0 sweep cost minutes rather than days.

## Testing

- `LookupDrafter` and `replay` are pure functions over token lists — unit tested
  directly, including the boundary cases (no match, match at position 0, match
  overlapping the generation boundary, `kmax` exceeding the remaining stream).
- The acceptance simulator is validated against a hand-computed fixture.
- **The end-to-end correctness test is the token-identity assertion**: speculative
  output must equal dense greedy output exactly, on every trace, in every timing run.
- Every aggregate in a report regenerates deterministically from committed summary
  statistics, with no model and no traces present. `--check` exits 0.

## Error handling

- A drafter that returns garbage costs throughput, never correctness — verification is
  exact. This is stated because it defines what must be defensive and what need not be.
- Any divergence from dense greedy is a hard stop, not a tuning signal.
- Missing MTP sidecar → lookup-only mode with a warning, so A0 can proceed without it.

## Phases and gates

See `docs/03-experiment-plan.md`. A0 gate ≥2.0× tokens/pass at ≥8K (kill <1.5×);
A1 gate ≥1.5× measured (kill <1.25×); A2 gate ≥10% over A1.

## Known risks

1. **The 2.01× is a proxy** — measured against human transcripts, not the model's own
   output. A0 step 2 corrects it; the number may fall.
2. **Quantization mismatch** between the `group_size 32` MTP sidecar and the
   `group_size 64` trunk. Lowers acceptance at worst; cannot affect correctness.
3. **Verification overhead** may swamp the win (arXiv 2601.11580). A1 measures the
   draft/verify split rather than assuming the batch-1 case inverts the critique.
4. **Redundancy** — lookup and MTP may win on the same tokens. A0 measures the overlap
   directly, and H3 gates on complementarity.
5. **Long context** degrades throughput 29% from 256 to 16K independently of any of
   this. Reported, not solved.
