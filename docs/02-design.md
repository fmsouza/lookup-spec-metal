# 02 — Design

## Thesis

Decode on this machine is pinned at the memory-bandwidth roof (254 GB/s effective vs
251 GB/s measured), so the only lossless lever is **tokens emitted per weight pass**.
For agentic coding, the largest untapped source of draft tokens is **the context
itself**: the model spends much of an edit turn re-emitting code it has already read.

MTP drafting is workload-agnostic and already implemented in `mlx-vlm`. Lookup
drafting is workload-specific, unimplemented, and measured at 2.01× standalone on real
traces. **Composing the two under one draft-tree budget is the contribution.**

## C1 — `LookupDrafter`

Draft by finding the most recent (or longest) earlier occurrence of the last `n`
tokens in `prefix + generated`, and proposing the `kmax` tokens that followed it.

Parameters: `n` (match length), `kmax` (draft length), match policy
(`most_recent` | `longest` | `multi`), and a minimum-match-quality filter.

Costs nothing but a substring search — no weights are read, so a lookup draft is free
against the bandwidth budget that governs everything else.

**Why it should work here, with numbers.** Measured offline on 20 real agentic-coding
traces (see `docs/03`, task A0):

| prefix budget | best config | tokens per pass | hit rate |
|---|---|---|---|
| 2 048 tokens | n=2, kmax=16 | 1.57× | 11.4% |
| ~6 400 tokens (median) | n=2, kmax=16 | **2.01×** | 18.1% |
| WikiText control, same config | n=2, kmax=16 | 1.24× | 9.3% |

Two things matter in that table. The gain is **workload-specific** — prose gets 1.24×,
code editing gets 2.01×. And it is **rising with context**, which is the opposite of
how the rest of the system behaves.

## C2 — Candidate merge into the draft tree

`build_ddtree` builds a best-first tree from per-depth log-probs. MTP supplies those
naturally; lookup supplies **hard token sequences with no distribution**. Merging them
needs a defined convention, and this is the one real design decision in the repo.

Approach, in order of increasing ambition:

1. **Disjoint fallback (A1 baseline).** If lookup hits, use its draft; otherwise use
   MTP's. Trivially correct, no tree changes, and directly tests whether the two
   sources are complementary.
2. **Budgeted merge (A2).** Give lookup candidates a synthetic log-prob derived from
   observed per-position acceptance (calibrated in A0, committed as a table), then let
   `build_ddtree` allocate its node budget across both sources on equal footing.
3. **Verified-prefix reuse (A2, if warranted).** A rejected lookup draft still tells us
   where the match diverged; reuse that to reseed the next draft rather than
   re-searching from scratch.

Only (1) is in scope for A1. (2) and (3) are conditional on A1 clearing its gate.

## C3 — Offline acceptance simulator

Acceptance depends only on the drafter and the token stream being produced. Given one
**fixed greedy generation** per trace, any drafter configuration can be replayed
against it without touching the model. This makes the A0 parameter sweep essentially
free and is what lets A0 kill the thesis in half a day instead of a week.

## Correctness model

Verification is exact (`Qwen3_5ExactSpeculativeVerifier`, singleton-equivalent
numerics), so the speculative decoder must emit **token-identical output to dense
greedy decode**. This is asserted on every trace in every run, and it replaces the
entire perplexity/EvalPlus apparatus a lossy plan would need.

A corollary worth stating because it shapes the risk profile: **draft quality can
degrade arbitrarily without affecting correctness.** A mismatched quantization, a bad
`n`, a stale lookup — all cost throughput, none cost correctness. That is why pairing
a `group_size 32` MTP head with a `group_size 64` trunk is an acceptable risk.

## Non-goals

- **No kernels.** The bandwidth roof is already reached; there is nothing to win.
- **No training, fine-tuning, or distillation.**
- **No quantization changes** in Phase A (Phase B only, and only if A plateaus).
- **No lossy techniques** in Phase A — layer skipping, pruning, sparsity all wait.
- **No batching, no beam search.** Batch-1 single-user serving only.
- **No new activation-sparsity work.** That question is settled; see the predecessor
  repo.
