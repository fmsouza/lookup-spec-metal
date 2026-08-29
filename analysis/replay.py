#!/usr/bin/env python3
"""Phase A0.4 — offline acceptance replay.

Acceptance depends only on the drafter and on the token stream being produced. Given
one *frozen* greedy generation per trace, any drafter configuration can be replayed
without touching the model -- which is what makes the A0 sweep cost minutes instead of
days, and what lets A0 kill the thesis before a decode loop exists.

The simulated loop matches how a real speculative decoder behaves:

    at each round the target model emits one token for certain, and additionally
    confirms the longest correct prefix of the draft. So a round advances
    1 + accepted tokens, and `tokens per pass` = mean(1 + accepted).

    uv run python analysis/replay.py --check     # self-tests, no model needed
    uv run python analysis/replay.py --sweep     # parameter sweep over frozen streams
"""

import argparse
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.lookup import LookupDrafter, MultiLookupDrafter  # noqa: E402


@dataclass
class AcceptanceStats:
    rounds: int = 0
    accepted: int = 0          # Σ draft tokens confirmed
    hits: int = 0              # rounds where the drafter proposed anything
    hit_accepted: int = 0      # Σ accepted on rounds where it proposed
    drafted: int = 0           # Σ draft tokens proposed (for waste accounting)
    tokens: int = 0            # Σ tokens emitted (= rounds + accepted)
    draft_lens: list = None    # per-round proposed length, for verification cost

    def __post_init__(self):
        if self.draft_lens is None:
            self.draft_lens = []

    @property
    def tokens_per_pass(self) -> float:
        return self.tokens / self.rounds if self.rounds else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.rounds if self.rounds else 0.0

    @property
    def accepted_given_hit(self) -> float:
        return self.hit_accepted / self.hits if self.hits else 0.0

    @property
    def draft_waste(self) -> float:
        """Fraction of proposed draft tokens that were rejected."""
        return 1 - self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self):
        return {"rounds": self.rounds, "accepted": self.accepted,
                "tokens": self.tokens, "drafted": self.drafted,
                "tokens_per_pass": self.tokens_per_pass,
                "hit_rate": self.hit_rate,
                "accepted_given_hit": self.accepted_given_hit,
                "draft_waste": self.draft_waste}


def load_verify_cost(path="results/raw/verify-cost.json", ctx_hint=4096):
    """Measured cost of a k-token verification pass, in units of a 1-token decode.

    Without this file, tokens-per-pass would be reported as if verification were free,
    which it is not. Returns None when unavailable so callers can say so explicitly
    rather than silently assuming 1.0.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return None
    rows = json.loads(p.read_text())["rows"]
    ctxs = sorted({r["ctx"] for r in rows})
    ctx = min(ctxs, key=lambda c: abs(c - ctx_hint))
    tbl = {r["k"]: r["cost_vs_k1"] for r in rows if r["ctx"] == ctx}

    def cost(k):
        k = max(1, int(k))
        if k in tbl:
            return tbl[k]
        ks = sorted(tbl)
        if k > ks[-1]:                       # linear extrapolation past the grid
            (k0, k1) = ks[-2], ks[-1]
            slope = (tbl[k1] - tbl[k0]) / (k1 - k0)
            return tbl[k1] + slope * (k - k1)
        lo = max(x for x in ks if x <= k)
        hi = min(x for x in ks if x >= k)
        if lo == hi:
            return tbl[lo]
        f = (k - lo) / (hi - lo)
        return tbl[lo] * (1 - f) + tbl[hi] * f

    cost.ctx = ctx                                                    # type: ignore
    return cost


def speedup(st, cost) -> Optional[float]:
    """Wall-clock speedup over dense greedy, charging each round its real verify cost."""
    if cost is None or not st.rounds:
        return None
    total_cost = sum(cost(d + 1) for d in st.draft_lens)
    return st.tokens / total_cost


def replay(context: Sequence[int], stream: Sequence[int], propose: Callable,
           per_round: Optional[list] = None) -> AcceptanceStats:
    """Replay `propose` against a frozen `stream` that follows `context`."""
    st = AcceptanceStats()
    gen: List[int] = []
    i = 0
    while i < len(stream):
        draft = list(propose(context, gen))
        acc = 0
        for d in draft:
            if i + acc < len(stream) and stream[i + acc] == d:
                acc += 1
            else:
                break
        st.rounds += 1
        st.accepted += acc
        st.drafted += len(draft)
        st.draft_lens.append(len(draft))
        if draft:
            st.hits += 1
            st.hit_accepted += acc
        if per_round is not None:
            per_round.append(acc)
        take = min(acc + 1, len(stream) - i)   # the certain token, plus confirmations
        gen.extend(stream[i:i + take])
        st.tokens += take
        i += take
    return st


def replay_composite(context, stream, primary, secondary,
                     per_round=None) -> AcceptanceStats:
    """Disjoint fallback (docs/02 C2.1): use `primary` if it proposes, else `secondary`.

    Also records how often each source was the one that fired, which is what H3's
    complementarity claim is judged on.
    """
    st = AcceptanceStats()
    st.by_source = {"primary": 0, "secondary": 0, "none": 0}          # type: ignore
    st.acc_by_source = {"primary": 0, "secondary": 0}                 # type: ignore
    gen: List[int] = []
    i = 0
    while i < len(stream):
        draft = list(primary(context, gen))
        src = "primary"
        if not draft:
            draft = list(secondary(context, gen))
            src = "secondary" if draft else "none"
        acc = 0
        for d in draft:
            if i + acc < len(stream) and stream[i + acc] == d:
                acc += 1
            else:
                break
        st.rounds += 1
        st.accepted += acc
        st.drafted += len(draft)
        st.draft_lens.append(len(draft))
        st.by_source[src] += 1                                        # type: ignore
        if src != "none":
            st.hits += 1
            st.hit_accepted += acc
            st.acc_by_source[src] += acc                              # type: ignore
        if per_round is not None:
            per_round.append(acc)
        take = min(acc + 1, len(stream) - i)
        gen.extend(stream[i:i + take])
        st.tokens += take
        i += take
    return st


# --------------------------------------------------------------------------
def _check():
    ok = True

    def t(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")

    # a drafter that never proposes degenerates to plain autoregressive decode
    st = replay([1, 2, 3], [4, 5, 6], lambda c, g: [])
    t("no drafter -> 1.0 tokens/pass", st.rounds == 3 and abs(st.tokens_per_pass - 1) < 1e-12)
    t("no drafter -> zero hit rate", st.hit_rate == 0 and st.drafted == 0)

    # a perfect oracle drafting kmax tokens advances kmax+1 per round
    stream = list(range(20))
    st = replay([], stream, lambda c, g: stream[len(g):len(g) + 4])
    t("oracle k=4 -> 5.0 tokens/pass", abs(st.tokens_per_pass - 5.0) < 1e-9)
    t("oracle wastes nothing", abs(st.draft_waste) < 1e-12)

    # totals must always reconcile
    t("tokens == rounds + accepted", st.tokens == st.rounds + st.accepted)
    t("stream fully consumed", st.tokens == len(stream))

    # a drafter that is always wrong costs nothing but waste
    st = replay([], stream, lambda c, g: [999, 999])
    t("always-wrong -> 1.0 tokens/pass", abs(st.tokens_per_pass - 1.0) < 1e-12)
    t("always-wrong -> 100% waste", abs(st.draft_waste - 1.0) < 1e-12)
    t("always-wrong still counts as a hit", st.hit_rate == 1.0)

    # exact copy: stream repeats the context verbatim, so lookup should ride it
    ctx = [5, 6, 7, 8, 9, 10, 11, 12]
    st = replay(ctx, ctx, LookupDrafter(n=2, kmax=8).propose)
    t("verbatim repeat is mostly drafted", st.tokens_per_pass > 2.0)

    # composite: primary never fires -> identical to secondary alone
    a = replay(ctx, ctx, LookupDrafter(n=2, kmax=8).propose)
    b = replay_composite(ctx, ctx, lambda c, g: [], LookupDrafter(n=2, kmax=8).propose)
    t("composite falls through to secondary",
      abs(a.tokens_per_pass - b.tokens_per_pass) < 1e-12)
    t("composite source accounting sums to rounds",
      sum(b.by_source.values()) == b.rounds)

    # composite is never worse than its primary alone when secondary is a no-op
    c = replay_composite(ctx, ctx, LookupDrafter(n=2, kmax=8).propose, lambda c_, g: [])
    t("composite >= primary alone", c.tokens_per_pass >= a.tokens_per_pass - 1e-12)

    # single-token stream edge case
    st = replay([1], [7], lambda c, g: [7, 7, 7])
    t("single-token stream terminates", st.rounds == 1 and st.tokens == 1)

    # replay is deterministic
    d1 = replay(ctx, ctx, LookupDrafter(n=2, kmax=8).propose).as_dict()
    d2 = replay(ctx, ctx, LookupDrafter(n=2, kmax=8).propose).as_dict()
    t("replay deterministic", d1 == d2)

    from src.lookup import self_test
    self_test()
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def load_generations(path="generated"):
    out = []
    for p in sorted(pathlib.Path(path).glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("generated_ids"):
            out.append(d)
    return out


def ctx_bucket(n):
    for lo in (16384, 8192, 4096, 0):
        if n >= lo:
            return {16384: "16K+", 8192: "8K-16K", 4096: "4K-8K", 0: "<4K"}[lo]


def drafter_grid():
    """The configurations A0 sweeps. Kept small and explicit rather than generated."""
    grid = []
    for n in (1, 2, 3, 4):
        for k in (4, 8, 16, 32):
            grid.append(LookupDrafter(n=n, kmax=k, policy="most_recent"))
    for k in (8, 16, 32):
        grid.append(LookupDrafter(n=2, kmax=k, policy="longest"))
        grid.append(MultiLookupDrafter(ns=(4, 3, 2), kmax=k))
        grid.append(MultiLookupDrafter(ns=(8, 5, 3), kmax=k))
    return grid


def sweep(args):
    gens = load_generations(args.generated)
    if not gens:
        sys.exit(f"no frozen generations in {args.generated!r} -- run analysis/generate.py")
    cost = load_verify_cost(args.verify_cost)
    if cost is None:
        print(f"! {args.verify_cost} missing -- reporting tokens/pass only.\n"
              f"! Run bench/verify_cost.py; tokens/pass is NOT a speedup.\n")

    rows = []
    for d in drafter_grid():
        per_kind = {}
        for kind in ("agentic", "control"):
            gs = [g for g in gens if g["kind"] == kind]
            if not gs:
                continue
            tot = AcceptanceStats()
            per_trace, per_trace_sp, per_bucket = [], [], {}
            for g in gs:
                st = replay(g["context_ids"], g["generated_ids"], d.propose)
                for f in ("rounds", "accepted", "hits", "hit_accepted",
                          "drafted", "tokens"):
                    setattr(tot, f, getattr(tot, f) + getattr(st, f))
                tot.draft_lens.extend(st.draft_lens)
                per_trace.append(st.tokens_per_pass)
                sp = speedup(st, cost)
                if sp is not None:
                    per_trace_sp.append(sp)
                b = ctx_bucket(len(g["context_ids"]))
                per_bucket.setdefault(b, []).append(st.tokens_per_pass)
            r = tot.as_dict()
            r.pop("draft_lens", None)
            r["n_traces"] = len(gs)
            # the pooled mean is throughput-correct but heavy-tailed: a single long
            # copy-heavy turn can carry it. The per-trace median is what a typical
            # turn sees, so both are reported and the gate reads the median.
            r["median_tokens_per_pass"] = statistics.median(per_trace)
            r["sd_tokens_per_pass"] = (statistics.stdev(per_trace)
                                       if len(per_trace) > 1 else 0.0)
            r["pooled_speedup"] = speedup(tot, cost)
            r["median_speedup"] = (statistics.median(per_trace_sp)
                                   if per_trace_sp else None)
            r["by_ctx"] = {b: sum(v) / len(v) for b, v in sorted(per_bucket.items())}
            per_kind[kind] = r
        rows.append({"drafter": d.name, "by_kind": per_kind})

    key = (lambda r: r["by_kind"]["agentic"].get("median_speedup") or 0) if cost else \
          (lambda r: r["by_kind"]["agentic"]["median_tokens_per_pass"])
    best = max(rows, key=key)

    out = {
        "kind": "a0-sweep",
        "model": gens[0]["model"],
        "env": gens[0].get("env", {}),
        "verify_cost_file": args.verify_cost if cost else None,
        "verify_cost_ctx": getattr(cost, "ctx", None),
        "generations": [{"id": g["id"], "kind": g["kind"],
                         "ctx": len(g["context_ids"]),
                         "gen": len(g["generated_ids"]), "langs": g["langs"]}
                        for g in gens],
        "rows": rows,
        "best_agentic": best["drafter"],
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))

    w = max(len(r["drafter"]) for r in rows)
    print(f"{'drafter'.ljust(w)}  {'med spd':>8} {'pool spd':>9} {'med t/p':>8} "
          f"{'pool t/p':>9} {'hit':>7} {'waste':>7} {'ctl spd':>8}")
    for r in sorted(rows, key=key, reverse=True):
        a = r["by_kind"]["agentic"]
        c = r["by_kind"].get("control", {})
        f = lambda v: f"{v:8.3f}" if isinstance(v, float) else f"{'--':>8}"
        print(f"{r['drafter'].ljust(w)}  {f(a['median_speedup'])} "
              f"{f(a['pooled_speedup'])[:9]:>9} {a['median_tokens_per_pass']:>8.3f} "
              f"{a['tokens_per_pass']:>9.3f} {100*a['hit_rate']:>6.1f}% "
              f"{100*a['draft_waste']:>6.1f}% {f(c.get('median_speedup'))}")
    ba = best["by_kind"]["agentic"]
    print(f"\nbest agentic: {best['drafter']}")
    print(f"  median speedup {ba['median_speedup']}, pooled {ba['pooled_speedup']}")
    print(f"  tokens/pass median {ba['median_tokens_per_pass']:.3f}, "
          f"pooled {ba['tokens_per_pass']:.3f} (sd {ba['sd_tokens_per_pass']:.3f})")
    print("  by context:", {k: round(v, 3) for k, v in ba["by_ctx"].items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--generated", default="generated")
    ap.add_argument("--out", default="analysis/summary/a0-sweep.json")
    ap.add_argument("--verify-cost", default="results/raw/verify-cost.json")
    a = ap.parse_args()
    if a.check:
        sys.exit(_check())
    if a.sweep:
        sweep(a)
    else:
        ap.error("pass --check or --sweep")
