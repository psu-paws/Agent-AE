# GAIATrace

Request-level execution traces from GAIA runs. Every LLM request a run issued is recorded with its prompt and output token IDs, its dependencies, and
measured tool latency, enough to replay a run against a simulator.

Two agent systems on same benchmark:

| | [OWL](https://github.com/camel-ai/owl/tree/gaia69_workforce_vllm) | [MiroThinker](https://github.com/MiroMindAI/MiroThinker/tree/main) |
|---|---|---|
| shape | multi-agent | single agent + summarizer |
| main LLM | gpt-oss-120b | MiroThinker-mini-1.7 |
| sub LLM | gpt-4o | gpt-4o-mini |
| dataset | GAIA-165-Val | GAIA-103-Text |
| requests | 5,406 | 2,082 |

## Layout

```
owl/             OWL traces, pipeline, analysis   → owl/README.md
mirothinker/     MiroThinker equivalent           → mirothinker/README.md
```

Both trees follow the same shape:

```
raw/tool/    measured tool latency
tracegen/    Generator (source logs → per-task CSVs)
traces/      CSV files, merged file
analysis/    reads traces, writes outputs/
agent/       the agent that produced the runs (from base repo)
```

## Source logs

The agent and server logs that `tracegen/` consumes — OWL's stdout and vLLM
sessions, MiroThinker's MiroFlow run logs and gpt-4o-mini dumps — are **not
included in this artifact**.

They are plain text, and GAIA asks that its validation set not be reshared in a
crawlable format. Inside the traces the same content exists only as `token_ids`,
which is not readable without decoding it. The full corpus will be archived separately under gated access later.

**This does not affect reproduction.** Every number, figure and table in the paper is produced from `traces/`, which is complete here. The analysis scripts have always read the CSVs. `tracegen/` is still included so the derivation from logs to token IDs can be read and audited; it simply cannot be executed without its inputs. The one thing the analysis needed from a log was a Success/Failure verdict, now carried in `owl/traces/run_outcomes.csv`.

With the corpus in hand, drop each tree's logs back into its `raw/` — `owl/raw/{agent,vllm}/` and `mirothinker/raw/{agent,summarizer}/` — and `run_all.sh` picks the tracegen stage up automatically, rebuilding `traces/` from scratch.

## What you get without running anything

The per-run traces are checked in, at each tree's `traces/session_traces/` — one CSV per run, carrying agent roles and dependency structure.

The simulator-ready merged file is not checked in. One `merge.py` call produces it, and `run_all.sh` does that for you:

```bash
cd owl/traces         && python merge.py -o owl_random.csv    # 165 tasks
cd mirothinker/traces && python merge.py -o miro_random.csv   # 103 tasks
```

Columns are simulator-shaped: `arrived_at`, `num_prefill_tokens`, `num_decode_tokens`, `block_size`, `request_id`, `model_id`, `session_id`, `source_file`, `inter_request_latency`, `token_ids` — OWL adds `dep`.

## Rebuilding

Python 3.10+ (the figure scripts use `X | None` annotations).


```bash
pip install -r owl/requirements.txt     # tiktoken, jinja2, matplotlib, transformers

./owl/run_all.sh                        # traces/ → analysis/outputs/
./mirothinker/run_all.sh                # same, per tree
```

Each script regenerates every figure and table in that tree's `analysis/outputs/`.
The tracegen stage is skipped automatically when the source logs are absent, since
`traces/session_traces/` is already its output; with the logs present the same
command rebuilds the traces from scratch too.

The one exception is `score_gaia.py`. It needs the run transcripts and the GAIA
reference answers, neither of which is shipped, and its `score_gaia.txt` output is
a per-task table of predicted answer against reference answer — an answer key — so
that is withheld too. Supply your own copy of the 2023 validation set with
`--ground-truth` to regenerate it; get it from
[gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA).

## Figures and tables

| Output | Script |
|---|---|
| `Figure2.png` | `mirothinker/analysis/figure2.py` — prefill/decode by turn phase |
| `Figure3.png` | `owl/analysis/figure3.py` — prefill/decode by agent role |
| `Figure4.png` | `owl/analysis/figure4.py` — OWL vs MiroThinker on one question |
| `Figure5.png` | `owl/analysis/figure5.py` — per-turn token timeline |
| `table1_owl.txt`, `table1_miro.txt` | token stats and KV hit rate |
| `accuracy.txt` | success rate and token cost by outcome |
| `score_gaia.txt` | MiroThinker equivalent — needs GAIA answers, not shipped (see above) |
| `table2_owl.txt`, `table2_miro.txt` | measured tool latency |

Each lands in that tree's `analysis/outputs/`.

## Reading the numbers

**Successes are not pass@1.** All attempts were run, including repeated failures, to collect enough correct and meaningful traces. Evaluation then keeps one trace per task, preferring successful ones, so that repeated failures on harder tasks do not dominate the study.

**Token counts are consistent, not exact.** The gpt-4o/4o-mini chat template is not available, so every request is rendered with the gpt-oss (harmony) chat template and tokenized with `o200k_harmony`. 
Token counts therefore slightly differ from what the API response, but stay consistent across the corpus. Dates in the system prompt are rewritten to a fixed day, so runs from different days still share a prefix.

## Contents and licensing

The traces record what the agents actually sent and received. `token_ids` decodes back to prompt text with `tiktoken`, so the corpus embeds **GAIA questions** and the **web page content** the agents scraped while working. Images are not included — each sits in the trace as a fixed-width placeholder, and the originals come with the raw logs when those are distributed.

GAIA reference answers are not here in any form: no answer key is shipped, and the outputs that would print one are withheld. What the traces do show is each agent's own answer, which on a correct run is the reference answer.

Please treat it accordingly: do not train on it, and do not re-host the GAIA content. It is here so the runs can be replayed and the numbers checked. GAIA's own [dataset terms](https://huggingface.co/datasets/gaia-benchmark/GAIA) govern its questions and answers.

| | |
|---|---|
| tracegen, traces, analysis (our code) | Apache-2.0 |
| the trace data | CC-BY-4.0 |
| `owl/agent/`, `mirothinker/agent/` | Apache-2.0, upstream — see each tree's `LICENSE` |
| scraped web content inside the traces | remains under its original terms |