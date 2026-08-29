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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(_check())
    ap.error("nothing to do yet; --sweep lands with A0.7")
