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

    def prefill(ctx):
        """Fresh cache at `ctx` tokens.

        The DeltaNet layers keep a recurrent state in an ArraysCache that is mutated
        in place and cannot be rolled back the way a KV offset can, so a speculative
        step cannot be un-done -- every measurement gets its own prefill.
        """
        cache = model.make_cache()
        pre = (filler * (ctx // len(filler) + 2))[:ctx]
        for s_ in range(0, len(pre), 2048):
            trunk(mx.array([pre[s_:s_ + 2048]]), cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    for ctx in [int(c) for c in args.contexts.split(",")]:
        base = None
        for k in ks:
            # One prefill per k, then single-token warmup steps to reach steady
            # state: the first step after a prefill is much slower (cold kernels,
            # graph build) and timing it would report ~190ms for k=1, three times
            # the steady-state cost. The cache then drifts by k per rep, which is
            # why reps is small and the smallest context measured is 2048.
            cache = prefill(ctx)
            for _ in range(args.warmup):
                trunk(mx.array([[filler[0]]]), cache=cache)
                mx.eval([c.state for c in cache])
            samples = []
            for r in range(args.reps):
                toks = mx.array([filler[:k]])
                t0 = time.time()
                h = trunk(toks, cache=cache)
                mx.eval(lm_head(h[:, -1:, :]))
                dt = time.time() - t0
                samples.append(dt)
            med = statistics.median(samples)
            if k == 1:
                base = med
            rows.append({"ctx": ctx, "k": k, "ms": 1000 * med,
                         "cost_vs_k1": med / base,
                         "drift_tokens": args.reps * k})
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
    ap.add_argument("--contexts", default="2048,8192,16384")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--out", default="results/raw/verify-cost.json")
    main(ap.parse_args())
