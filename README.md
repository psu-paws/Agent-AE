# Artifact

Artifact for *"Characterizing How Complex Agentic AI Systems
Handle General Tasks: A Trace-Based Simulation Study"*.

The artifact is a two-stage pipeline. **GAIATrace** records what two real agent
systems did on the GAIA benchmark; **Vidur-Agent** replays those traces against a
simulated serving cluster.

```
GAIATrace/     stage 1 — agent runs  →  request-level traces   → GAIATrace/README.md
Vidur-Agent/   stage 2 — traces      →  simulation + figures   → Vidur-Agent/README.md
```

## The dependency between the two

Reproducing the paper runs them in order: Vidur-Agent replays the CSVs GAIATrace
produces. The coupling is loose — the traces are plain CSVs, and Vidur-Agent
needs only `num_prefill_tokens` and `num_decode_tokens` — so either side can be used with other inputs.

The run scripts look for the merged traces at these paths:

| produced by | file | consumed by |
|---|---|---|
| `GAIATrace/mirothinker/run_all.sh` | `GAIATrace/mirothinker/traces/miro_random.csv` | `run_miro.sh`, `run_miro70B.sh` |
| `GAIATrace/owl/run_all.sh` | `GAIATrace/owl/traces/owl_random.csv` | `run_owl.sh`, `run_owl70B.sh` |

The run scripts resolve these relative to `Vidur-Agent/`, so the two directories
must stay siblings as laid out above. If your traces live elsewhere, point at them
instead:

```sh
MIRO_TRACE=/path/to/miro_random.csv bash run_miro.sh
OWL_TRACE=/path/to/owl_random.csv   bash run_owl.sh
```

Each script checks its trace exists and stops with a message if not, rather than
failing later inside the simulator.


## Running end to end

```sh
# Stage 1 — generate traces (see GAIATrace/README.md for per-system detail)
cd GAIATrace/owl         && bash run_all.sh && cd ../..
cd GAIATrace/mirothinker && bash run_all.sh && cd ../..

# Stage 2 — simulate and plot (see Vidur-Agent/README.md)
cd Vidur-Agent
uv venv && source .venv/bin/activate && uv sync
bash warmup_predictors.sh          # fit the execution-time predictors once
bash run_miro.sh && bash run_owl.sh && bash run_miro70B.sh && bash run_owl70B.sh
bash analysis/run_gantt_compare.sh
bash analysis/run_plot_ttft_tpot.sh
```

Stage 1 needs the agent systems and their model endpoints; stage 2 is CPU-only
and replays the recorded traces, so it can be run on its own once the CSVs exist.

## What each stage covers

**GAIATrace** records every LLM request two agent systems issued on GAIA —
prompt and output token IDs, inter-request dependencies, and measured tool
latency. OWL (multi-agent, 5,406 requests on GAIA-165-Val) and MiroThinker
(single agent plus summarizer, 2,082 requests on GAIA-103-Text).

**Vidur-Agent** extends [Vidur](https://github.com/microsoft/vidur) at commit
[`25e0082`](https://github.com/microsoft/vidur/commit/25e0082dbbfb206fb0477c3ebbededa7ead78949)
with heterogeneous replica groups, prefill–decode disaggregation with modelled KV
transfer, multi-turn sessions released by a dependency graph, and KV-aware
routing. It reproduces the paper's simulation results and figures.

## Use of AI assistance

The authors used Claude Code in preparing this artifact, for code, comment, and figures. All outputs were reviewed carefully by the authors, the paper carries the same acknowledgement.

## License

Both stages are released under the MIT License. `Vidur-Agent/` derives from
Microsoft's Vidur and retains its copyright notice in `Vidur-Agent/LICENSE`; this
project is not affiliated with or endorsed by Microsoft.
