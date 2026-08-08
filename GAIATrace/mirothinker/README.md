# MiroThinker traces

## Layout

```
raw/
└── tool/             measured tool latency + tool_calls/
tracegen/             parser.py → session_traces + transcripts,
                      then replace_summarizer_placeholders.py
traces/               session_traces, merge to miro_random.csv
analysis/             figures and tables, outputs/
agent/                MiroFlow agent that produced the runs (frozen)
```

`parser.py` reads MiroFlow run logs (`raw/agent/*.json`) and `replace_summarizer_placeholders.py` reads the gpt-4o-mini dumps (`raw/summarizer/*.json`). Neither is included in this artifact — see "Source logs" in the root README.

The source logs hold 154 files for 103 tasks: 52 tasks ran once, 51 also have a `format-retry-1` re-issue, and `parser.py` keeps the highest retry per task (103 traces). The retry is not an independent sample: it is seeded with a `=== Previous Attempts Analysis ===` block summarising the first try.

## Pipeline

```bash
./run_all.sh                       # tracegen → traces → analysis
```

`parser.py` tokenizes with the Qwen3 tokenizer and downloads it from HuggingFace on first run
`score_gaia.py` needs the GAIA answers, which are not shipped, pass `--ground-truth`.

## Tool latency

`agent/apps/miroflow-agent/benchmarks/benchmark_mcp_tools_full.py` 
replays each task's calls in order, three times, into `raw/tool/{tool_name}.csv`; 
`merge.py` averages the reps into `inter_request_latency`.

## Provenance and license

`agent/` vendors [MiroThinker](https://github.com/MiroMindAI/MiroThinker),
Apache-2.0; `agent/LICENSE` and the original copyright headers are unmodified.
