# bench — Phase A1: matched-context timing

Measures wall-clock decode speedup over the dense baseline at 256 / 8K / 16K / 32K.

Every run asserts speculative output is **token-identical to dense greedy** before any
timing number is recorded. A run that fails the assertion is void.

3 runs per cell, median reported; ≥2-min cooldown; plugged in, High Power mode. Records
macOS build, engine versions, `iogpu.wired_limit_mb`, power state, and the
draft-vs-verify wall-clock split.
