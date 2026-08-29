"""Phase A0.3 — prompt-lookup drafting.

Draft by finding an earlier occurrence of the last `n` tokens in
`context + generated`, and proposing the tokens that followed it.

No weights are read, so a lookup draft is free against the memory-bandwidth budget
that governs everything else on this machine (see docs/01). A draft that is wrong
costs throughput and nothing else -- verification is exact -- so these are tuned for
hit rate, not for safety.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

POLICIES = ("most_recent", "longest", "first")


@dataclass(frozen=True)
class LookupDrafter:
    """n-gram lookup drafter.

    n         length of the suffix that must match
    kmax      maximum number of tokens to propose
    policy    which earlier occurrence to copy from:
                most_recent -- the latest match (cheapest, usually best: locality)
                longest     -- the match with the longest agreeing suffix beyond n
                first       -- the earliest match (baseline for ablation)
    min_ctx   do not draft until at least this many tokens exist to search
    """

    n: int = 2
    kmax: int = 16
    policy: str = "most_recent"
    min_ctx: int = 0

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.kmax < 1:
            raise ValueError("kmax must be >= 1")
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")

    # -- internals ---------------------------------------------------------
    def _match_positions(self, seq: Sequence[int]) -> List[int]:
        """Indices just past every earlier occurrence of the trailing n-gram.

        The trailing occurrence itself is excluded: copying from it would propose the
        tokens that follow the current position, which do not exist yet.
        """
        n = self.n
        if len(seq) < n + 1:
            return []
        pat = tuple(seq[-n:])
        out = []
        # stop at len(seq)-n-1 so the window never includes the trailing occurrence
        for j in range(len(seq) - n - 1, -1, -1):
            if tuple(seq[j:j + n]) == pat:
                out.append(j + n)
        return out          # newest first

    def _agree_len(self, seq: Sequence[int], end: int) -> int:
        """How far back the context agrees with the suffix, beyond the n matched."""
        i, k = end - self.n - 1, len(seq) - self.n - 1
        a = 0
        while i >= 0 and k >= 0 and seq[i] == seq[k]:
            i -= 1; k -= 1; a += 1
        return a

    # -- API ---------------------------------------------------------------
    def propose(self, context: Sequence[int],
                generated: Sequence[int] = ()) -> List[int]:
        """Return up to kmax draft tokens; empty list on a miss."""
        seq = list(context) + list(generated)
        if len(seq) < max(self.min_ctx, self.n + 1):
            return []
        hits = self._match_positions(seq)
        if not hits:
            return []
        if self.policy == "most_recent":
            end = hits[0]
        elif self.policy == "first":
            end = hits[-1]
        else:                                    # longest
            end = max(hits, key=lambda e: (self._agree_len(seq, e), e))
        return list(seq[end:end + self.kmax])

    @property
    def name(self) -> str:
        return f"lookup(n={self.n},k={self.kmax},{self.policy})"


class MultiLookupDrafter:
    """Several n values tried longest-first; the first hit wins.

    Longer n means a more specific match and a better draft when it hits, but a lower
    hit rate. Cascading recovers specificity without losing coverage.
    """

    def __init__(self, ns=(4, 3, 2), kmax: int = 16, policy: str = "most_recent"):
        self.drafters = [LookupDrafter(n=n, kmax=kmax, policy=policy)
                         for n in sorted(ns, reverse=True)]
        self.ns, self.kmax, self.policy = tuple(sorted(ns, reverse=True)), kmax, policy

    def propose(self, context, generated=()) -> List[int]:
        for d in self.drafters:
            got = d.propose(context, generated)
            if got:
                return got
        return []

    @property
    def name(self) -> str:
        return f"multilookup(ns={list(self.ns)},k={self.kmax},{self.policy})"


class GatedLookupDrafter:
    """Draft only when the match is trustworthy.

    Charging real verification cost inverts the naive ranking: on this model a
    k-token verify costs ~1.2x (k=2) to ~4.5x (k=32) a single-token decode, so a
    drafter that fires constantly and is usually wrong is *slower than not drafting
    at all*. What pays is precision, not recall.

    This gates on how far back the context agrees with the current suffix: an n-gram
    hit backed by `min_agree` further matching tokens is a much stronger signal that
    the model is mid-copy. Below the gate, propose nothing and let the round run at
    k=1 cost.
    """

    def __init__(self, n: int = 2, kmax: int = 32, min_agree: int = 8,
                 policy: str = "longest"):
        self.base = LookupDrafter(n=n, kmax=kmax, policy=policy)
        self.n, self.kmax, self.min_agree, self.policy = n, kmax, min_agree, policy

    def propose(self, context, generated=()) -> List[int]:
        seq = list(context) + list(generated)
        hits = self.base._match_positions(seq)
        if not hits:
            return []
        best, best_agree = None, -1
        for e in hits:
            a = self.base._agree_len(seq, e)
            if a > best_agree:
                best, best_agree = e, a
            if best_agree >= self.min_agree and self.policy == "most_recent":
                break
        if best_agree < self.min_agree:
            return []
        return list(seq[best:best + self.kmax])

    @property
    def name(self) -> str:
        return f"gated(n={self.n},k={self.kmax},agree>={self.min_agree})"


def self_test() -> None:
    """Boundary cases the sweep would otherwise silently mis-measure."""
    d = LookupDrafter(n=2, kmax=3)

    # basic copy: trailing [1,2] also occurs at index 0, followed by 3,4,5
    assert d.propose([1, 2, 3, 4, 5, 9, 1, 2]) == [3, 4, 5]
    # no earlier occurrence -> miss
    assert d.propose([7, 8, 9]) == []
    # too short to hold pattern + one earlier token
    assert d.propose([1, 2]) == []
    assert d.propose([]) == []
    # kmax truncates
    assert LookupDrafter(n=2, kmax=2).propose([1, 2, 3, 4, 5, 9, 1, 2]) == [3, 4]
    # match running off the end returns what exists, not padding
    assert LookupDrafter(n=1, kmax=5).propose([4, 1, 2, 4]) == [1, 2, 4]
    # the split between context and generated must not matter
    assert d.propose([1, 2, 3, 4, 5], [9, 1, 2]) == d.propose([1, 2, 3, 4, 5, 9, 1, 2])
    # most_recent vs first pick different occurrences
    seq = [1, 2, 7, 0, 0, 1, 2, 8, 0, 1, 2]
    assert LookupDrafter(n=2, kmax=1, policy="most_recent").propose(seq) == [8]
    assert LookupDrafter(n=2, kmax=1, policy="first").propose(seq) == [7]
    # longest prefers the occurrence with more agreeing history
    seq2 = [5, 5, 1, 2, 7, 0, 9, 1, 2, 8, 0, 5, 5, 1, 2]
    assert LookupDrafter(n=2, kmax=1, policy="longest").propose(seq2) == [7]
    # the trailing occurrence itself is never used as a source
    assert LookupDrafter(n=3, kmax=2).propose([1, 2, 3]) == []
    # cascade falls back from n=4 to n=2
    m = MultiLookupDrafter(ns=(4, 2), kmax=2)
    assert m.propose([1, 2, 9, 9, 9, 9, 1, 2]) == [9, 9]
    # min_ctx suppresses drafting
    assert LookupDrafter(n=2, kmax=3, min_ctx=100).propose([1, 2, 3, 1, 2]) == []
    # gated: a bare n-gram hit with no agreeing history is refused
    g = GatedLookupDrafter(n=2, kmax=4, min_agree=3)
    assert g.propose([9, 9, 1, 2, 7, 7, 7, 7, 0, 1, 2]) == []
    # ... but a long agreeing run passes the gate
    seq = [5, 6, 7, 1, 2, 8, 8, 8, 8, 0, 5, 6, 7, 1, 2]
    assert GatedLookupDrafter(n=2, kmax=3, min_agree=3).propose(seq) == [8, 8, 8]
    # min_agree=0 degenerates to the ungated longest policy
    a = GatedLookupDrafter(n=2, kmax=3, min_agree=0).propose(seq)
    b = LookupDrafter(n=2, kmax=3, policy="longest").propose(seq)
    assert a == b, (a, b)
    print("lookup self-test OK")


if __name__ == "__main__":
    self_test()
