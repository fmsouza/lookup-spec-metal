#!/usr/bin/env python3
"""Phase A0.5 — freeze one dense greedy continuation per trace.

This is the methodological fix A0 exists for. Draft acceptance must be measured
against what *this model* emits, not against the human transcript the trace was built
from. The predecessor's 2.01x figure compared drafts to a Claude Code transcript; a
drafter that predicts Qwen3.8 well and Claude badly (or vice versa) would be
mis-scored. So: generate once, greedily, freeze to disk, and replay every drafter
configuration against that frozen stream.

Greedy is the right regime because speculative decoding is exact only relative to a
fixed sampling rule, and greedy is the one where "identical output" is definable.

Output: generated/<trace-id>.json  (gitignored -- derived from private traces)
"""

import argparse
import json
import pathlib
import platform
import subprocess
import time

import mlx.core as mx


def env_info():
    def sh(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "?"

    def powermode_ac(text):
        in_ac = False
        for line in text.splitlines():
            if line.strip().startswith("AC Power"):
                in_ac = True
            elif line.strip().startswith("Battery Power"):
                in_ac = False
            if in_ac and "powermode" in line:
                return line.split()[-1]
        return "?"

    import mlx_lm
    return {
        "macos": f"{sh('sw_vers','-productVersion')} ({sh('sw_vers','-buildVersion')})",
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "ram_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0),
        "iogpu_wired_limit_mb": sh("sysctl", "-n", "iogpu.wired_limit_mb"),
        "power": "AC" if "AC Power" in sh("pmset", "-g", "ps") else "battery",
        "powermode_ac": powermode_ac(sh("pmset", "-g", "custom")),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
    }


def main(args):
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(args.model)
    trunk = model.language_model.model          # skip lm_head where logits aren't needed
    lm_head = model.language_model.lm_head

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    traces = sorted(p for p in pathlib.Path(args.traces).glob("*.json")
                    if p.name != "MANIFEST.json")
    if args.limit:
        traces = traces[:args.limit]

    env = env_info()
    t0 = time.time()
    for i, p in enumerate(traces):
        rec = json.loads(p.read_text())
        dest = outdir / f"{rec['id']}.json"
        if dest.exists() and not args.force:
            print(f"  [{i+1}/{len(traces)}] {rec['id']:>14} cached", flush=True)
            continue

        ids = tokenizer.encode(rec["prefix"])
        cache = model.make_cache()
        for s in range(0, len(ids), args.chunk):        # dense prefill
            h = trunk(mx.array([ids[s:s + args.chunk]]), cache=cache)
            mx.eval([c.state for c in cache])
        tok_id = int(mx.argmax(lm_head(h[:, -1:, :]), axis=-1).item())

        out = [tok_id]
        for _ in range(args.max_tokens - 1):
            h = trunk(mx.array([[tok_id]]), cache=cache)
            tok_id = int(mx.argmax(lm_head(h[:, -1:, :]), axis=-1).item())
            if tok_id in tokenizer.eos_token_ids:
                break
            out.append(tok_id)

        dest.write_text(json.dumps({
            "id": rec["id"], "kind": rec["kind"], "project": rec["project"],
            "langs": rec["langs"], "model": args.model,
            "prefix_tokens": len(ids), "context_ids": ids,
            "generated_ids": out, "n_generated": len(out),
            "sampling": "greedy/argmax", "env": env,
        }))
        print(f"  [{i+1}/{len(traces)}] {rec['id']:>14} ctx {len(ids):6d} "
              f"gen {len(out):4d}  ({time.time()-t0:6.1f}s)", flush=True)
    print(f"wrote {len(traces)} generations to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--traces", default="analysis/traces")
    ap.add_argument("--out", default="generated")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
