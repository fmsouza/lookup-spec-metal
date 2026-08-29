# PLAN — lookup-spec-metal

Maps `docs/03-experiment-plan.md` to concrete tasks. Estimates are wall-clock on the
target machine (M3 Max 36 GB), one operator, sequential.

**Legend:** ☐ todo · ◐ in progress · ☑ done · ✗ killed by gate

## Inherited, already measured (no need to redo)

| Fact | Value | Source |
|---|---|---|
| Dense 4-bit decode, ctx 256 | 17.6 tok/s | measured 2026-08-29 |
| … ctx 8K / 16K | 16.9 / 12.5 tok/s | measured |
| Weights read per decode token | 14.42 GB | measured |
| Effective bandwidth | 254 GB/s (roof: 251 measured, ~300 spec) | measured |
| Peak MLX memory, 4-bit | 15.6 GB | measured |
| Activation sparsity + MTP verification | falsified | `sparse-spec-metal` |
| Component-aware self-spec on sequential hybrids | ruled out | arXiv 2605.01106 |

## Phase A0 — offline acceptance · est. 4–7 h

| # | Task | Artifact | Est. |
|---|---|---|---|
| A0.1 ☑ | Repo, docs, spec, env | this repo | 1.0 h |
| A0.2 ☐ | Trace corpus at ≥24K prefix, tool results untruncated | `analysis/traces/` (gitignored) | 0.5 h |
| A0.3 ☐ | `LookupDrafter` + unit tests | `src/lookup.py` | 1.0 h |
| A0.4 ☐ | Offline replay + sweep, `--check` self-tests | `analysis/replay.py` | 1.0 h |
| A0.5 ☐ | Freeze one dense greedy generation per trace | `analysis/generate.py`, `generated/` | 1.0 h |
| A0.6 ☐ | Pull the 0.239 GB MTP sidecar; wire the `qwen3_5_mtp` drafter | `src/mtp.py` | 1.0 h |
| A0.7 ☐ | Sweep lookup × MTP × composite; measure win-overlap | `analysis/summary/a0.json` | 0.5 h |
| A0.8 ☐ | Report + H1 gate | `analysis/report.md`, `results/a0-*.md` | 1.0 h |

**Gate:** composite ≥2.0× tokens/pass at ≥8K → A1. **Kill:** <1.5×. **Noise:** stop, ask.

## Phase A1 — end-to-end · est. 12–20 h

| # | Task | Artifact | Est. |
|---|---|---|---|
| A1.1 ☐ | Disjoint-fallback composer | `src/compose.py` | 2 h |
| A1.2 ☐ | Decode loop on `mlx_vlm.speculative` | `src/decode.py` | 6 h |
| A1.3 ☐ | Lossless assertion vs dense greedy, every trace | `src/decode.py` tests | 2 h |
| A1.4 ☐ | Timing harness, 256/8K/16K/32K, draft-vs-verify split | `bench/run.py` | 4 h |
| A1.5 ☐ | Report + H2 gate | `results/a1-*.md` | 2 h |

**Gate:** ≥1.5× measured at matched context. **Kill:** <1.25% → publish the negative
result with the draft/verify split as evidence.

## Phase A2 — tuning · est. 12–20 h

| # | Task | Est. |
|---|---|---|
| A2.1 ☐ | Budgeted merge into `build_ddtree` with calibrated synthetic log-probs | 6 h |
| A2.2 ☐ | Dynamic draft length by recent hit state | 4 h |
| A2.3 ☐ | Tree-shape sweep | 4 h |
| A2.4 ☐ | Report + gate (≥10% over A1) | 2 h |

## Phase B — lossy levers (only if A plateaus)

Not started unless A1/A2 land short. Requires building the perplexity + EvalPlus
harness Phase A deliberately avoids. Priority: mixed-precision quantization
(~1.2–1.3×), then training-free layer-skip drafting (SWIFT/CLaSp).

## Download budget

| Item | Size | Running total |
|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` | 16.08 GB | 16.08 GB (already local) |
| `mtp.safetensors` sidecar only | 0.24 GB | 16.32 GB |
| WikiText-2-raw | 0.02 GB | 16.34 GB |

Deliberately **not** pulling MTPLX's 19.5 GB trunk — only its sidecar. Anything beyond
this → ask first.

## Hygiene (binding)

- Lossless assertion on every timing run; a failing run is void, not degraded.
- 3 runs per timing cell, median; ≥2-min cooldown; plugged in, High Power.
- Deterministic non-timing stats may run once if proven byte-identical twice; record it.
- Every run gets a `results/` entry per `results/README.md`.
- Never commit weights, traces, or generated streams.
- Commit per milestone: `"a0: <what>"`.
- Ask before: sudo, downloads beyond the budget above, paid APIs, force-pushes.
