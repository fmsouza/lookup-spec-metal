# analysis — Phase A0: offline acceptance

No decode loop. Measures how many tokens a drafter would have won, replayed against a
**frozen** greedy generation from the target model.

| | |
|---|---|
| `build_traces.py` | agentic-coding traces from local Claude Code session logs + WikiText control. Ported from `sparse-spec-metal`; prefix budget raised to ≥24K, tool results untruncated. |
| `generate.py` | one dense greedy continuation per trace, frozen to `generated/` (gitignored). This is what acceptance is measured against — not the human transcript. |
| `replay.py` | pure-function drafter replay + parameter sweep → `summary/*.json` |
| `report.md` | A0 findings and the H1 gate decision |
| `traces/`, `generated/` | **gitignored** — private source code |

```bash
uv run python analysis/build_traces.py --out analysis/traces --max-prefix 24576
uv run python analysis/generate.py
uv run python analysis/replay.py --check     # self-tests, exits 0, no model needed
uv run python analysis/replay.py --sweep
```
