#!/usr/bin/env python3
"""Does the cheap multi-token path actually change what the model says?

A0 found that exact verification (`Qwen3_5ExactSpeculativeVerifier`) costs 12-25x a
single-token decode at k=16-32, because bit-equivalence to sequential decoding forces
the DeltaNet recurrent state to advance one position at a time. The plain multi-token
path is 3-5x instead -- and it is what mainstream speculative decoding implementations
use. Its outputs differ from sequential decoding *numerically*. The question this
script answers is whether they differ *semantically*: how often does the argmax move?

Method. The frozen generations in `generated/` are, by construction, the sequential
greedy argmaxes: generate.py fed one token at a time. So the reference already exists
and does not need recomputing. Here the same token sequence is teacher-forced through
the model again, but in chunks of k, and the argmax at every position is compared.

This captures both ways divergence can enter:
  * within a pass -- batching k positions perturbs the arithmetic;
  * across passes -- the cache state after a chunked step differs from the state
    after k single steps, and that drift accumulates over the whole stream.

Reported per k: fraction of positions whose argmax moved, how many traces are affected
at all, and where the first divergence lands. A rate near zero means the 1.49x on the
cheap path is available under a defensible correctness claim; a rate that is not near
zero means the lossless framing requires the expensive verifier.

Output: results/raw/divergence.json
"""

import argparse
import json
import pathlib
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

    gens = [json.loads(p.read_text())
            for p in sorted(pathlib.Path(args.generated).glob("*.json"))]
    gens = [g for g in gens if len(g.get("generated_ids", [])) >= args.min_tokens]
    if args.limit:
        gens = gens[:args.limit]
    ks = [int(x) for x in args.ks.split(",")]

    results, t0 = [], time.time()
    for k in ks:
        tot_pos = tot_div = 0
        traces_div = 0
        per_trace = []
        for g in gens:
            ctx, stream = g["context_ids"], g["generated_ids"]
            cache = model.make_cache()
            for s in range(0, len(ctx), args.chunk):        # identical dense prefill
                h = trunk(mx.array([ctx[s:s + args.chunk]]), cache=cache)
                mx.eval([c.state for c in cache])

            # prediction that follows the prefix must match stream[0]
            pred0 = int(mx.argmax(lm_head(h[:, -1:, :]), axis=-1).item())
            n_div = 0 if pred0 == stream[0] else 1
            first = None if pred0 == stream[0] else 0
            n_pos = 1

            # feed the stream in chunks of k; position j predicts stream[j+1]
            margins = []
            for s in range(0, len(stream) - 1, k):
                blk = stream[s:s + k]
                h = trunk(mx.array([blk]), cache=cache)
                lg = lm_head(h)[0].astype(mx.float32)
                preds = mx.argmax(lg, axis=-1).tolist()
                # top1 - top2 gap: how much slack the argmax had. A near-zero rate of
                # divergence is only meaningful alongside evidence that the decision
                # was not trivially safe everywhere.
                top2 = mx.topk(lg, 2, axis=-1)
                gap = (top2[:, -1] - top2[:, -2]).tolist()
                mx.eval([c.state for c in cache])
                for j, pr in enumerate(preds):
                    idx = s + j + 1
                    if idx >= len(stream):
                        break
                    n_pos += 1
                    margins.append(float(gap[j]))
                    if pr != stream[idx]:
                        n_div += 1
                        if first is None:
                            first = idx
            tot_pos += n_pos
            tot_div += n_div
            traces_div += 1 if n_div else 0
            margins.sort()
            per_trace.append({"id": g["id"], "kind": g["kind"], "positions": n_pos,
                              "divergences": n_div, "first_divergence": first,
                              "margin_min": margins[0] if margins else None,
                              "margin_p01": margins[max(0, len(margins) // 100)]
                              if margins else None,
                              "margin_median": margins[len(margins) // 2]
                              if margins else None})
        rate = tot_div / tot_pos if tot_pos else 0.0
        mins = [t["margin_min"] for t in per_trace if t["margin_min"] is not None]
        meds = [t["margin_median"] for t in per_trace if t["margin_median"] is not None]
        results.append({"k": k, "positions": tot_pos, "divergences": tot_div,
                        "rate": rate, "traces": len(gens),
                        "traces_with_divergence": traces_div,
                        "margin_min_overall": min(mins) if mins else None,
                        "margin_median_of_medians": sorted(meds)[len(meds) // 2]
                        if meds else None,
                        "per_trace": per_trace})
        print(f"  k={k:<3} {tot_div:6d}/{tot_pos:6d} positions diverge "
              f"({100*rate:6.3f}%)  traces affected {traces_div}/{len(gens)}  "
              f"margin min {min(mins):.3f} med {sorted(meds)[len(meds)//2]:.2f}  "
              f"({time.time()-t0:6.1f}s)", flush=True)

    out = {"kind": "divergence", "model": args.model, "env": env_info(),
           "note": "reference = frozen sequential greedy argmaxes from generate.py",
           "rows": results}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--generated", default="generated")
    ap.add_argument("--ks", default="2,4,8,16")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/raw/divergence.json")
    main(ap.parse_args())
