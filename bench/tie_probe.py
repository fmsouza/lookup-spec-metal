"""Are the diverging positions exactly the tied ones?

Trace-level correlation (every diverging trace also contains a zero-margin position)
is weaker than it looks. This records the top1-top2 margin at each *diverging*
position and compares it against the margin distribution over all positions.
"""
import json, pathlib, sys
import mlx.core as mx
from mlx_lm.utils import load as mlx_load

model, tok = mlx_load("mlx-community/Qwen3.8-27B-4bit")
trunk = model.language_model.model
lm_head = model.language_model.lm_head

ids = ["agentic-06", "agentic-20", "agentic-16", "agentic-09"]
K = 4
div_margins, all_margins = [], []
for tid in ids:
    g = json.loads(pathlib.Path(f"generated/{tid}.json").read_text())
    ctx, stream = g["context_ids"], g["generated_ids"]
    cache = model.make_cache()
    for s in range(0, len(ctx), 2048):
        h = trunk(mx.array([ctx[s:s+2048]]), cache=cache); mx.eval([c.state for c in cache])
    for s in range(0, len(stream) - 1, K):
        blk = stream[s:s+K]
        h = trunk(mx.array([blk]), cache=cache)
        lg = lm_head(h)[0].astype(mx.float32)
        preds = mx.argmax(lg, axis=-1).tolist()
        t2 = mx.topk(lg, 2, axis=-1)
        gaps = (t2[:, -1] - t2[:, -2]).tolist()
        mx.eval([c.state for c in cache])
        for j, p in enumerate(preds):
            idx = s + j + 1
            if idx >= len(stream): break
            all_margins.append(float(gaps[j]))
            if p != stream[idx]:
                div_margins.append(float(gaps[j]))

import statistics
all_margins.sort()
n = len(all_margins)
zero = sum(1 for m in all_margins if m == 0.0)
print(f"positions {n}, diverging {len(div_margins)}")
print(f"margin at DIVERGING positions: {sorted(div_margins)}")
print(f"all-position margin: min {all_margins[0]:.4f} p01 {all_margins[n//100]:.4f} "
      f"median {all_margins[n//2]:.3f}")
print(f"exact ties (margin == 0) overall: {zero}/{n} = {100*zero/n:.3f}%")
print(f"diverging positions with margin == 0: "
      f"{sum(1 for m in div_margins if m == 0.0)}/{len(div_margins)}")
