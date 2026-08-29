# 01 — Context: what is measured, and what it forces

Every number here was measured on the target machine on 2026-08-29 unless marked
otherwise. Items marked *(lit.)* come from published work and are re-verified before
being leaned on.

## Hardware

- **Apple M3 Max**, 14-core (10P/4E), 30-core GPU, **36 GB unified memory**.
- macOS 26.6.2 (25G83). `iogpu.wired_limit_mb` = 0 (system default ≈75% of RAM;
  raising it needs `sudo`, so the default is recorded and used).
- Power: AC, `powermode 2` (High Power).
- **Measured streaming read bandwidth: 251 GB/s** (512 MB fp16 reduction, MLX 0.32.2),
  against a ~300 GB/s spec figure.

## Model

`mlx-community/Qwen3.8-27B-4bit`, revision `3e6447f`, 16.08 GB on disk.

- 64 layers: **48 Gated DeltaNet + 16 full attention** (`full_attention_interval` 4).
- hidden 5120, FFN 17408, vocab 248320, head_dim 256, 24 q-heads / 4 kv-heads.
- 262K native context. KV only in the 16 full-attention layers → ~64 KB/token.
- `mtp_num_hidden_layers: 1` — an MTP draft head exists in the released weights.
- The MLX 4-bit conversion **drops the MTP head**; `mlx_lm.sanitize()` filters
  `mtp.*`. The head is available separately (see below).

### Bytes actually read per decode token

| | GB |
|---|---|
| trunk (all quantized linears, 4-bit / group 64) | 13.702 |
| `lm_head` | 0.715 |
| `embed_tokens` (row lookup, not a full read) | — |
| **total read per decode token** | **14.42** |

## The measurement that determines the whole plan

| context | decode | effective bandwidth from weights alone |
|---|---|---|
| 256 | 17.6 tok/s | **254 GB/s** |
| 8 192 | 16.9 tok/s | 243 GB/s |
| 16 384 | 12.5 tok/s | 180 GB/s |

At short context, decode moves **254 GB/s** against a **251 GB/s** measured streaming
roof and ~300 GB/s spec. **Decode is already at the memory-bandwidth roof.** There is
no kernel headroom — not 20%, not 5%. Two consequences:

1. **Optimizing kernels is not a lever.** Any plan whose speedup comes from a faster
   matmul is dead on arrival on this machine.
2. The only two levers are **read fewer bytes per pass** and **emit more tokens per
   pass**. Under a lossless constraint, only the second is available.

Long context is a separate, unaddressed problem: throughput falls 29% from 256 to 16K.
Speedups are therefore always reported at **matched context**.

## Why this repo exists: the prior negative result

The predecessor repo [`sparse-spec-metal`](https://github.com/fmsouza/sparse-spec-metal)
tested whether TEAL-style activation sparsity composes with multi-token MTP
verification. It does not. Measured over 20 real agentic-coding traces (11,171
teacher-forced decode tokens):

| | |
|---|---|
| union sparsity, k=4, per-token 50% | **10.8%** (needed ≥30%) |
| union sparsity, k=4, per-token 40% | **5.1%** (kill criterion: <20%) |
| adjacent-token mask Jaccard | 0.380 |
| vs. statistically independent masks | only **1.59×** better |

Adjacent tokens simply do not activate similar neurons on this model, so a k-token
union mask is barely sparser than chance. That killed the sparsity lever, and the
bandwidth measurement above killed the kernel lever. What remains is speculation.

## The speculation landscape for this model

**Already implemented in `mlx-vlm` 0.6.17** — verified by reading the package:

- `mlx_vlm.speculative` — a full speculative-decoding framework.
- `drafters/qwen3_5_mtp/` — an MTP drafter for exactly this model family.
- `speculative_verifier.py` → `Qwen3_5ExactSpeculativeVerifier`, with hand-written
  Metal QMV kernels and **"singleton-equivalent numerics"**: verifying k tokens is
  bit-equivalent to k separate single-token decodes. Lossless by construction.
- `ddtree.py` → `build_ddtree`, best-first draft-tree construction from log-probs.
- `drafters/qwen3_5_mtp/split.py` → `split_qwen3_5_mtp`, extracts an MTP head from a
  full checkpoint.

**Not implemented:** any lookup / n-gram drafter, and any way to merge non-probabilistic
candidates into the draft tree. That gap is this repo's subject.

### The MTP head

`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` ships the head as a standalone
`mtp.safetensors` sidecar, **0.239 GB**, 31 tensors — DeepSeek-style MTP:
`pre_fc_norm_embedding` + `pre_fc_norm_hidden` → `fc` (10240→5120) → one **full
attention** block (q 12288 / k 1024 / v 1024 / o 5120, q_norm & k_norm at 256) →
MLP (gate/up 17408, down 5120) → `norm`, sharing the main `lm_head`.

A draft step therefore costs 0.239 GB + 0.715 GB (`lm_head`) = **0.954 GB, 6.6% of a
14.42 GB verify pass**.

That repo's *trunk* is `group_size 32` with selected 8-bit layers — 19.5 GB, ~18%
slower than ours. **We take only the 0.239 GB sidecar** and pair it with the
`mlx-community` 4-bit trunk. The quantization mismatch can lower draft acceptance but
**cannot affect correctness**, because verification is exact.

## Literature *(lit.)*

- **Prompt Lookup Decoding** (Saxena) / **REST** (arXiv 2311.08252) — draft by
  retrieving n-gram continuations from the context. Zero draft-model cost.
- **Empirical study of speculative decoding on SE tasks** (arXiv 2604.26469) —
  model-based drafting wins on code *generation*; **model-free wins on
  repository-level repair and editing**. Repetitiveness of SE work favours model-free.
- **Component-aware self-speculative decoding in hybrid LMs** (arXiv 2605.01106) —
  using the linear-attention subgraph as a free draft works on *parallel* hybrids
  (Falcon-H1, acceptance 0.68 at k=2) but **fails on sequential hybrids like
  Qwen3.5/3.8: acceptance 0.038**, 18× worse. LayerSkip beat it 12× on sequential
  models. **This direction is ruled out for our architecture before we spend a day
  on it.**
- **Speculative Decoding: Performance or Illusion?** (arXiv 2601.11580) — on
  production vLLM, verification cost dominates and reported speedups do not
  materialize. Its setting is high-batch and compute-bound; ours is batch-1 and
  bandwidth-bound, where verifying k tokens costs nearly the same as one. The critique
  should invert in our favour — **but A1 measures the draft/verify time split rather
  than assuming it.**
- **EAGLE-3 / tree attention / TALON** — richer draft trees; relevant only if A2 needs
  more than `ddtree` provides.
- **SWIFT / CLaSp / LayerSkip** — training-free self-speculative layer skipping,
  1.3–1.7× reported. Lossy for our purposes; held for Phase B.
- **Sub-4-bit quantization** — INT3 costs ~6 points vs ~1.6 at AWQ-4, matching this
  model's reported cliff below Q4. At best ~1.28× for real quality risk. Phase B.
