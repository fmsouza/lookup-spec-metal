# Phase A0 — offline acceptance

## Gate

**A0 FAILS its gate on every path.** Best is **1.49×** (cheap verification) or
**~1.2×** (exact verification), against a **≥2.0× gate**. See "Finding 5" for the
divergence experiment that decides which of those is available, and
"Decision" for what is actually on the table.

**A0 FAILS its gate, and sits at the kill line.** Best measured configuration is
**1.49× median speedup** (`gated(n=2,k=64,agree>=4)`) against a **≥2.0× gate** and a
**<1.5× kill** criterion — and that figure uses the *cheap, non-exact* verification
path. On the exact verifier the project's correctness model requires, the same
configuration is far worse.

- best median speedup **1.493×** (pooled 1.494×), WikiText control 1.309×
- best tokens/pass 1.82 median / 1.96 pooled — the acceptance is real
- what kills it is **verification cost**, not acceptance
- **recommendation: stop, or re-scope to a non-exact verification path.** Awaiting review.

## What was measured

28 frozen greedy generations from `mlx-community/Qwen3.8-27B-4bit` (24 agentic-coding
traces across 5 projects and 9 languages, 4 WikiText controls), median prefix 6,081
tokens, max 17,426. Every drafter configuration was replayed offline against those
frozen streams, and each round was charged its **measured** verification cost.

## Finding 1 — acceptance is high, and higher than the proxy suggested

Against the model's own greedy output — not the human transcript used in the earlier
proxy — prompt-lookup drafts are accepted far more often than the 2.01× estimate
implied. Individual traces reach 17–23× tokens per pass on `[Edit]` turns that copy a
locale JSON verbatim, and on bash commands carrying UUIDs and long paths. Those were
inspected specifically to rule out degenerate decode loops: they are legitimate
copy-from-context, which is exactly the thesis.

So the *premise* of this repo — that agentic edit turns re-emit large verbatim spans
from context — is **confirmed**.

## Finding 2 — verification is not cheap on this architecture, and that is decisive

Speculative decoding assumes verifying k tokens costs about the same as verifying one,
because the weights are read once. On a dense transformer that holds. **On this hybrid
it does not.**

Cost of a k-token pass, in units of a single-token decode (mlx-lm path):

| ctx | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| 2 048 | 1.23× | 2.11× | 3.26× | 4.23× | 4.50× |
| 8 192 | 1.62× | 2.35× | 3.67× | 4.42× | 5.17× |
| 16 384 | 1.75× | 3.08× | 4.11× | 4.68× | 5.48× |

48 of 64 layers are Gated DeltaNet. A single-token step is a cheap recurrent update; a
k-token step is a chunked scan costing far more. The marginal cost per drafted token
does fall with k (0.62 at k=2 down to 0.14 at k=32), so *long* drafts amortise well —
but only if they are accepted.

## Finding 3 — charging real cost inverts the ranking

| drafter | median speedup | tokens/pass | hit rate | waste |
|---|---|---|---|---|
| `gated(n=2,k=64,agree>=4)` | **1.493×** | 1.82 | 7.9% | 79.3% |
| `gated(n=2,k=32,agree>=4)` | 1.470× | 1.81 | 8.8% | 65.7% |
| `lookup(n=4,k=32,most_recent)` | 1.392× | 2.03 | 16.4% | 76.2% |
| `lookup(n=2,k=32,most_recent)` | 0.973× | 2.72 | 48.7% | 88.2% |
| `lookup(n=1,k=16,most_recent)` | **0.693×** | 2.67 | 89.3% | 88.2% |

The configurations with the **highest** tokens-per-pass are the ones that make decoding
**slower than dense**. A drafter that fires on 89% of rounds and is wrong 88% of the
time pays full verification cost for nothing. What wins is precision: the best drafter
fires on **7.9%** of rounds. This is the opposite of the usual prompt-lookup tuning
advice, and it follows directly from the cost curve above.

Confidence gating — requiring the n-gram hit to be corroborated by ≥4 further agreeing
tokens of history — lifted the best result from 1.39× to 1.49×.

## Finding 4 — exactness and cheap verification are in conflict here

`mlx-vlm` ships `Qwen3_5ExactSpeculativeVerifier` with hand-written Metal QMV kernels,
advertising *singleton-equivalent numerics*: verifying k tokens is bit-equivalent to k
separate single-token decodes. That guarantee is what would have made the correctness
model a cheap token-identity assertion. Measured against the plain path:

| ctx | path | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| 2 048 | plain | 1.61× | 3.06× | 3.90× | 4.14× |
| 2 048 | **exact verify** | 1.73× | 3.73× | **12.47×** | **24.68×** |
| 8 192 | plain | 2.69× | 4.81× | 5.19× | 5.27× |
| 8 192 | **exact verify** | 1.09× | 1.98× | **7.10×** | **15.21×** |

The exact verifier is competitive to k≈8 and then diverges badly. That is mechanistically
expected rather than a bug: bit-equivalence to *sequential* single-token decoding forces
the DeltaNet recurrent state to be advanced one position at a time, which is precisely
the work a chunked scan exists to avoid. **On a recurrent architecture you can have
exact verification or cheap parallel verification, not both.**

Since the best drafters want k=32–64, and the exact path costs 15–25× there, the
lossless configuration is confined to k≤8 — where speedup is roughly 1.2×.

## Finding 5 — the cheap path changes only decisions the model never made

A0's first pass concluded that losslessness forces the expensive verifier. That
depended on an untested assumption: that the plain multi-token path meaningfully
changes outputs. It was tested directly.

The frozen generations are, by construction, the sequential greedy argmaxes. Feeding
the same streams back through the model in chunks of k and comparing the argmax at
every position gives, over **10,187 positions × 5 chunk sizes**:

| k | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| positions diverging | 0.442% | 0.442% | 0.432% | 0.442% | 0.402% |

**The rate is flat in k.** That is the whole story. If chunked arithmetic were
perturbing results, divergence would grow with chunk size — more positions batched,
more accumulated cache drift. It does not move at all between k=2 and k=32.

Because the cause is not chunking. It is **ties**. Measuring the top1−top2 logit
margin at each diverging position, over four traces (1,574 positions, 19 divergences):

| | value |
|---|---|
| margins at **diverging** positions | 0.0 (×9), 0.125 (×9), 0.25 (×1) — **all ≤ 0.25** |
| median margin over all positions | **6.125** |
| exact ties (margin = 0) overall | 2.22% of positions |

Not one divergence occurred at a position decided by more than a quarter of a logit,
against a typical margin ~50× larger. Every divergence is a coin-flip the model itself
had no opinion about: at 4-bit weights with bf16 logits over a 248,320-token vocabulary,
**2.2% of greedy decisions are exact ties**, and roughly a quarter of those resolve
differently under a different summation order.

The consequence cuts both ways.

**Against the cheap path:** 0.44% per position compounds. A 500-token generation is
token-identical with probability `0.9956^500 ≈ 11%` — so ~89% of generations would
differ somewhere. Token identity, the correctness gate `docs/02` specified, is **not
available** on the plain path.

**Against the premise of that gate:** "lossless greedy decoding" was never a property
of this model. At 2.2% of positions the greedy argmax is a tie, and which token wins is
decided by floating-point summation order, not by the model. The exact verifier
preserves *one particular* arbitrary tie-break — it does not recover a uniquely correct
answer, because there isn't one. The guarantee is thinner than it sounds.

So the honest claim available on the cheap path is not "identical output" but:
**no decision with a margin above 0.25 logits is ever changed; divergences occur only
at top-2 ties, affecting 0.4% of positions.** That is a defensible engineering claim.
It is not the claim `docs/02` was written around.

## Decision

| path | speedup | correctness claim |
|---|---|---|
| exact verifier, k≤8 | **~1.2×** | token-identical to one arbitrary tie-break order |
| plain multi-token, k=32–64 | **1.49×** offline, ~1.3–1.45× expected end-to-end | no confident decision altered; 0.4% of positions differ, all at ties |

Both fail the ≥2.0× gate. The realistic ceiling for this direction is **~1.3–1.5×**,
not the 2–3× the plan was built on, and the gap is architectural: Gated DeltaNet makes
multi-token verification expensive, which is the one cost speculative decoding assumes
away.

Whether ~1.4× justifies Phase A1's 12–20 h is a scoping call, not a gate outcome. The
gate outcome is: **A0 fails.**

## What this means

The thesis was right about the workload and wrong about the machine. Agentic edit turns
really do copy heavily from context, and a cheap n-gram drafter really does predict
them. But this model's hybrid architecture makes the *verification* half of speculative
decoding expensive enough to consume most of the gain, and makes the exact-verification
guarantee expensive enough to consume all of it.

`docs/03`'s A0 kill criterion is **<1.5×**. The measured best is 1.493× on the cheap
path and ~1.2× on the exact path. This is a fail on the gate (≥2.0×) and at or below
the kill line.

## Not yet tested

- **MTP as a second source.** The 0.239 GB sidecar is downloaded but not wired
  (A0.6). It could add on the ~92% of rounds where the gated drafter stays silent, but
  every MTP draft token still pays the same verification curve, so the ceiling is
  bounded by Finding 2 regardless.
- **k≤4 tuning on the exact path.** If the project continues under a lossless
  constraint, this is the only region worth exploring, and the ceiling there is ~1.2×.
- ~~Relaxing exactness~~ — **done, see Finding 5.**
- **End-to-end validation.** Every number here is offline. A1 would also pay DeltaNet
  state snapshot/rollback for rejected drafts (48 layers × ~3.1 MB ≈ 151 MB per
  speculative step, ~1% of a step at measured bandwidth) and n-gram search cost, so the
  1.49× offline ceiling should be read as ~1.3–1.45× in practice.
- **MTP as a second source** (A0.6). Sidecar downloaded, not wired. Bounded by
  Finding 2 regardless of how well it drafts.

## Reproducing

```bash
uv run python analysis/replay.py --check     # 15 self-tests, no model needed
uv run python bench/verify_cost.py           # -> results/raw/verify-cost.json
uv run python analysis/replay.py --sweep     # -> analysis/summary/a0-sweep.json
```
