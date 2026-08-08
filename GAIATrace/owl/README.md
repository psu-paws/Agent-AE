# OWL (Workforce) traces

## Layout

```
raw/
└── tool/             measured tool latency + tool_calls/
tracegen/             parser.py + replace_oss_rows.py — see "Source logs" below
traces/               session_traces, run_outcomes.csv,
                      selected_set.txt, merge to owl_random.csv
analysis/             figures and tables, outputs/
agent/                OWL agent that produced the runs (frozen)
```

Each run is captured twice, because neither log is sufficient alone. The agent's
stdout (`raw/agent`) carries the requests and their dependency structure, but long
summarization is dispatched asynchronously, so it does not fix the order in which
those requests reached the server. The vLLM log (`raw/vllm`) supplies that order
and the open-weight token ids. `tracegen/` rejoins the two by filename.

Neither log ships with this artifact — see "Source logs" in the root README.
`traces/run_outcomes.csv` carries the one thing the analysis took from them: a
solved/not verdict per run.

## Pipeline

```bash
./run_all.sh                       # everything below, in order
```

## Tool latency

`agent/tool_bench.py` replays every recorded tool call three times and writes
`raw/tool/{tool_name}.csv`. It lives inside the agent tree because it calls the
OWL toolkits directly. Re-measuring needs a `.env` with tool API keys.
`merge.py` fills `inter_request_latency` from these replays, not from the
duration observed during the run: each call's three reps are averaged.

## Provenance and license

`agent/` vendors [OWL](https://github.com/camel-ai/owl) at commit `e398ed2`
(branch `gaia69_workforce_vllm`), including the bundled `camel/` package.
OWL and CAMEL are Apache-2.0; the license is kept at `agent/LICENSE` and the
original copyright headers are unmodified.