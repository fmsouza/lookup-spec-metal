# 04 — References

Verification status noted per entry. Anything load-bearing is re-measured before it is
relied on; nothing in `docs/01`'s measured tables comes from this list.

## Core — the direction this repo pursues

- **Prompt Lookup Decoding** — Apoorv Saxena.
  github.com/apoorvumang/prompt-lookup-decoding. Draft by matching the last `n`
  generated tokens against the prompt and copying the continuation. Zero draft-model
  cost. Integrated in vLLM and HF transformers as n-gram speculation.
  *Basis for C1.*
- **REST: Retrieval-Based Speculative Decoding** — He et al., NAACL 2024,
  arXiv:2311.08252. Drafts retrieved from a corpus rather than the prompt, organised
  into a draft tree. *Relevant if A2 wants a corpus beyond the live context.*
- **RASD: Retrieval-Augmented Speculative Decoding** — arXiv:2503.03434. Combines
  retrieval drafting with a model drafter. *Closest prior art to this repo's C2 merge;
  read before implementing the budgeted merge.*
- **An Empirical Study of Speculative Decoding on Software Engineering Tasks** —
  arXiv:2604.26469. Model-based drafting wins on code generation; **model-free wins on
  repository-level repair and editing**. Directly motivates preferring lookup for the
  agentic-edit workload. *Abstract verified 2026-08-29; full numbers not yet extracted.*

## Counter-evidence — read before believing our own results

- **Speculative Decoding: Performance or Illusion?** — arXiv:2601.11580. On
  production vLLM, verification dominates and reported speedups fail to materialise;
  acceptance varies sharply by output position. Setting is high-batch and
  compute-bound; ours is batch-1 and bandwidth-bound, where the argument should invert.
  **A1 measures the draft/verify split explicitly rather than assuming it does.**
- **Component-Aware Self-Speculative Decoding in Hybrid Language Models** —
  arXiv:2605.01106. Using the linear-attention subgraph as a zero-cost draft works on
  parallel hybrids (Falcon-H1: acceptance 0.68 at k=2) and **fails on sequential
  hybrids: Qwen3.5 scores 0.038**, 18× worse; LayerSkip beat it 12× there. Qwen3.8-27B
  is a sequential hybrid. **This rules the direction out for us at zero cost.**

## Speculation machinery

- **EAGLE / EAGLE-2 / EAGLE-3** — feature-level draft models with draft trees;
  3–4.3× reported, integrated in vLLM and SGLang. *Reference for A2 tree design.*
- **TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees** —
  arXiv:2601.07353. *Reference for A2 dynamic draft length.*
- **Medusa / Hydra / Lookahead (Jacobi) decoding** — alternative multi-head and
  parallel-decoding families. Out of scope while an MTP head already ships.

## Phase B candidates (lossy — not in Phase A)

- **LayerSkip** — Elhoushi et al., ACL 2024, arXiv:2404.16710. Training-based
  early-exit self-speculation, up to 2.16×.
- **SWIFT** — arXiv:2410.06916. On-the-fly, training-free layer-skip self-speculation,
  1.3–1.6×.
- **CLaSp** — training-free dynamic-programming layer skipping, 1.3–1.7×.
- **KnapSpec** — arXiv:2602.20217. Layer selection as a knapsack problem.
- Mixed-precision quantization (WINDQuant arXiv:2605.26660, CoQuant arXiv:2604.26378,
  MosaicQuant arXiv:2606.15652). Community reports put this model's quality cliff below
  Q4; INT3 costs ~6 points vs ~1.6 at 4-bit. *(lit., unverified on this model.)*

## Prior negative result

- **`sparse-spec-metal`** — github.com/fmsouza/sparse-spec-metal. Union-mask activation
  sparsity does not compose with multi-token MTP verification on Qwen3.8-27B: 10.8%
  union sparsity at k=4 / 50% per-token against a ≥30% requirement; adjacent-token mask
  Jaccard 0.380; only 1.59× better than statistically independent masks. Also the
  source of this repo's baseline measurements and trace-construction tooling.
- **TEAL** — arXiv:2408.14690, and **SpQt** — arXiv:2511.04477. The methods that repo
  tested. Retained here only as context for why sparsity is not on this roadmap.

## Tooling artifacts

- `mlx-community/Qwen3.8-27B-4bit` — 16.08 GB, revision `3e6447f`. No MTP head.
- `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` — ships `mtp.safetensors` (0.239 GB)
  as a standalone sidecar. Its trunk is `group_size 32` and ~18% slower; **we take only
  the sidecar.**
- `mlx-vlm` ≥ 0.6.17 — `mlx_vlm.speculative` (framework, `ddtree`, `qwen3_5_mtp`
  drafter, `split_qwen3_5_mtp`) and `Qwen3_5ExactSpeculativeVerifier` (Metal QMV
  kernels, singleton-equivalent numerics). *Verified by reading the installed package
  on 2026-08-29.*
- `mlx` 0.32.2, `mlx-lm` 0.31.3.
