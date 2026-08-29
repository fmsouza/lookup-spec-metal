#!/usr/bin/env python3
"""Phase A0 — measure the cost of verifying k tokens in one pass.

Tokens-per-pass is not speedup. A speculative round costs one *verification* pass over
k+1 positions, and that pass is not free relative to a single-token decode: the weights
are read once either way (this machine is bandwidth-bound, see docs/01), but attention,
the DeltaNet recurrent path, and the lm_head all scale with the number of positions.

    speedup(k) = tokens_per_pass(k) / (verify_cost(k) + draft_cost)

where verify_cost(k) is in units of a single-token dense decode. This script measures
verify_cost(k) directly, so A0's gate is expressed in wall-clock terms rather than in
accepted tokens.

Output: results/raw/verify-cost.json + a table on stdout.
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.generate import env_info  # noqa: E402


def main(args):
    from mlx_lm.utils import load as mlx_load

    model, tok = mlx_load(args.model)
    trunk = model.language_model.model
    lm_head = model.language_model.lm_head

    filler = tok.encode("def f(x):\n    return x + 1\n\n") * 64
    ks = [int(x) for x in args.ks.split(",")]
    rows = []

    for ctx in [int(c) for c in args.contexts.split(",")]:
        cache = model.make_cache()
        pre = (filler * (ctx // len(filler) + 2))[:ctx]
        for s in range(0, len(pre), 2048):
            trunk(mx.array([pre[s:s + 2048]]), cache=cache)
            mx.eval([c.state for c in cache])
        base = None
        for k in ks:
            # a fresh cache branch per k would be ideal; instead measure without
            # committing, by restoring cache offsets after each timed call
            offs = [c.offset for c in cache]
            samples = []
            for r in range(args.reps + args.warmup):
                toks = mx.array([filler[:k]])
                t0 = time.time()
                h = trunk(toks, cache=cache)
                mx.eval(lm_head(h[:, -1:, :]))
                dt = time.time() - t0
                for c, o in zip(cache, offs):
                    c.offset = o                      # roll back the speculative step
                if r >= args.warmup:
                    samples.append(dt)
            med = statistics.median(samples)
            if k == 1:
                base = med
            rows.append({"ctx": ctx, "k": k, "ms": 1000 * med,
                         "cost_vs_k1": med / base})
            print(f"  ctx {ctx:6d}  k {k:3d}  {1000*med:7.2f} ms  "
                  f"cost {med/base:5.3f}x", flush=True)

    out = {"kind": "verify-cost", "model": args.model, "env": env_info(),
           "reps": args.reps, "rows": rows}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--contexts", default="256,4096,16384")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results/raw/verify-cost.json")
    main(ap.parse_args())
