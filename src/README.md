# src — drafters and composition

| | |
|---|---|
| `lookup.py` | `LookupDrafter` — n-gram match over `prefix + generated`; params `n`, `kmax`, policy. Pure Python, no model access. |
| `compose.py` | `Composer` — A1: disjoint fallback (lookup if hit, else MTP). A2: budgeted merge into `mlx_vlm.speculative.ddtree`. |

A drafter that returns nonsense costs throughput, never correctness: verification is
exact. Optimise these for hit rate, not for safety.
