# results

One markdown file per run: config, environment (macOS build, engine versions,
`iogpu.wired_limit_mb`, power state), raw numbers (3 runs), median, and the gate
decision it informs. Raw dumps go to `results/raw/` (gitignored); summaries are
committed.

Every timing entry must also record the lossless assertion outcome — speculative output
token-identical to dense greedy — and the draft-vs-verify wall-clock split.
