# lookup-spec-metal

**Multi-source speculative decoding for agentic coding on Apple Silicon.**

Status: **Phase A0 in progress** — offline acceptance measurement, no decode loop yet.

## Why

Dense 4-bit `Qwen3.8-27B` on an M3 Max 36 GB decodes at **17.6 tok/s**, reading
**14.42 GB** of weights per token at **254 GB/s effective** against a **251 GB/s
measured streaming roof**. The machine is already saturated — there is no kernel
headroom to reclaim.

That closes every lever except one. Reading fewer bytes is lossy. So under a lossless
constraint the only thing left is **emitting more tokens per weight pass**.

## Thesis

For agentic coding, the largest untapped source of draft tokens is **the context
itself** — the model spends much of an edit turn re-emitting code it has already read.

Two drafters, one verifier:

| Drafter | Cost | Character | Status |
|---|---|---|---|
| **Lookup** — n-gram match against `prefix + generated` | free (no weights read) | workload-specific: **2.01×** on agentic traces vs **1.24×** on prose | **this repo builds it** |
| **MTP** — the model's own 1-layer draft head | 0.239 GB = 6.6% of a verify pass | workload-agnostic | already in `mlx-vlm` |

`mlx-vlm` already ships the MTP drafter, the draft tree (`build_ddtree`), and
`Qwen3_5ExactSpeculativeVerifier` with hand-written Metal QMV kernels whose
*singleton-equivalent numerics* make k-token verification bit-equivalent to k
single-token decodes. **The gap is a lookup drafter and a way to merge its candidates
into that tree.** That gap is this repo.

Because verification is exact, correctness is a **token-identity assertion against
dense greedy** — not a perplexity/EvalPlus suite. A bad draft costs throughput and
nothing else.

## The measurement this rests on

Prompt-lookup acceptance, replayed offline over 20 real agentic-coding traces
(5 projects, 10 languages) and 4 WikiText controls:

| prefix budget | corpus | best config | tokens/pass | hit rate |
|---|---|---|---|---|
| 2 048 | agentic | n=2, kmax=16 | 1.57× | 11.4% |
| ~6 400 (median) | agentic | n=2, kmax=16 | **2.01×** | 18.1% |
| ~6 400 | WikiText | n=2, kmax=16 | 1.24× | 9.3% |

The gain is workload-specific, and it **rises with context** — the opposite of how the
rest of the system behaves. It is also, currently, **a proxy**: acceptance was measured
against human transcripts rather than the model's own output. Correcting that is the
first task of Phase A0, and the number may fall.

## Target

Largest **lossless speedup over the dense baseline at matched context**. No absolute
tok/s goal: the baseline moves too much with context (17.6 tok/s at 256, 12.5 at 16K)
for one to be meaningful. Everything is reported as a ratio at 256 / 8K / 16K / 32K.

## Phases

| | | Gate | Kill |
|---|---|---|---|
| **A0** | offline acceptance, no decode loop | composite ≥2.0× tokens/pass at ≥8K | <1.5× |
| **A1** | end-to-end decode loop | ≥1.5× measured at matched context | <1.25× |
| **A2** | tuning: budgeted merge, dynamic draft length | ≥10% over A1 | — |
| **B** | lossy levers + eval harness | only if A plateaus | — |

Kill criteria are binding. Any divergence from dense greedy output is a hard stop.

## Repo map

- `docs/01-context.md` — measured hardware/model facts and the bandwidth argument
- `docs/02-design.md` — components C1–C3, correctness model, non-goals
- `docs/03-experiment-plan.md` — hypotheses, phases, gates, kill criteria
- `docs/04-references.md` — papers, including the ones that rule directions *out*
- `docs/superpowers/specs/` — the approved design spec
- `analysis/` — trace building, generation freezing, offline acceptance replay
- `src/` — `LookupDrafter` and the composer
- `bench/` — matched-context timing harness
- `results/` — one entry per run

## What came before

[`sparse-spec-metal`](https://github.com/fmsouza/sparse-spec-metal) tested whether
TEAL-style activation sparsity composes with multi-token MTP verification. **It does
not** — union sparsity at k=4 is 10.8% where ≥30% was needed, adjacent-token mask
Jaccard is 0.380, and the masks are only 1.59× better than statistically independent.
That repo also produced the baseline measurements and the trace tooling this one
inherits.

## Ground rules

- Cheapest falsification first. A0 needs no decode loop and can kill the thesis in
  half a day.
- Lossless in Phase A. Lossy levers wait for Phase B and bring their own quality gates.
- Never commit weights, traces, or generated streams — traces contain private source
  code.
- 3 runs per timing cell, median, ≥2-min cooldown, plugged in, High Power mode.
