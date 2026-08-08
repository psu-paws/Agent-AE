# OWL trace-generation fork

Base: [OWL](https://github.com/camel-ai/owl/tree/gaia69_workforce_vllm) `gaia69_workforce_vllm` @ `e398ed2beecebb39c6ff135f8711678c2b388998`.
Upstream README kept at [README_original.md](README_original.md).

Runs GAIA and prints every model call's request and response to stdout. The
captured log is the trace, downstream tooling parses it into token-level
traces.

Use `run_gaia_workforce_vllm_planner.py`. The other runners
(`run_gaia_workforce.py`, `_claude.py`, `run_workforce_vllm.py`) are inherited
from upstream and are not maintained for trace collection.

## Changes

### Trace capture

| Marker | Source | Contains |
|---|---|---|
| `Request IN: ` | `chat_agent.py`, `enhanced_chat_agent.py` | `openai_messages` + token count |
| `Answer OUT1/2/3` | same | `ChatCompletionMessage`, decoded token list, usage |
| `Answer OUT22/33` | same | fallback form when no logprobs were returned |

- `args_str` (`_types.py`, `func_message.py`, `chat_agent.py`) — raw tool-call
  argument string kept and replayed. Upstream re-serialises the parsed
  dict with `json.dumps()`, changing separators and the token sequence.
- `logprobs` (`openai_config.py`) — required to emit output tokens, not just text.

### Model support

Upstream supports all Models. Token-level traces require the reasoning model
to expose reasoning tokens, so `O3_MINI` is replaced by a locally served
`openai/gpt-oss-120b` (`GPT_OSS`). Hosted reasoning models (o3, Claude, Gemini)
do not disclose reasoning tokens — `CLAUDE_4_0_SONNET`, `GEMINI_2_5_PRO`,
`GEMINI_2_5_FLASH` are registered in `types/enums.py`.

### Latency prints

`Elapsed Time ...`, `Code Run ...`, `Browser Init`, `Browse OUT Time`.
Informational only; nothing branches on them.

### Other

- `huggingface.co/datasets|spaces|blog|...` blocked in `search_toolkit.py`,
  `document_processing_toolkit.py`, `browser_toolkit.py`.
- Firecrawl v2 migration in `document_processing_toolkit.py`.

## Setup

Serve with local model:

```bash
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 --async-scheduling \
    --enable-auto-tool-choice --tool-call-parser openai \
    --gpu-memory-utilization 0.9 --max-model-len 131072 \
    --no-enforce-eager --enable-prefix-caching \
    --enable-prompt-tokens-details
```

Build the agent image (Playwright base — the browser toolkit needs it):

```bash
docker build -t agent-image .
```

Requires `.env` at the repo root (`OPENAI_API_KEY`, `GOOGLE_API_KEY`,
`FIRECRAWL_API_KEY`, `SEARCH_ENGINE_ID`, ...).

The vLLM endpoint is `--host` (default `$VLLM_HOST`, else `localhost`) and
`-p`/`--port` (default 8000).

### GAIA dataset

GAIA is gated: accept the terms at
[gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)
while logged in, then download into `data/gaia/`:

```bash
huggingface-cli login          # newer huggingface_hub calls this `hf auth login`
huggingface-cli download gaia-benchmark/GAIA \
    --repo-type dataset --local-dir data/gaia
```

Same thing from Python — this is what `GAIABenchmark.download()`
(`utils/gaia.py`) does:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset",
                  local_dir="data/gaia")
```

Expected layout (`utils/gaia.py` reads `metadata.jsonl` from both splits):

```
data/gaia/2023/validation/metadata.jsonl   + attachment files
data/gaia/2023/test/metadata.jsonl         + attachment files
```

`data/gaia` is gitignored.

## Running / generating traces

One task per container run, stdout tee'd to a file:

```bash
docker run -it --rm \
    --ipc=host --network=host \
    -e PYTHONUNBUFFERED=1 \
    -e no_proxy="localhost,127.0.0.1" -e NO_PROXY="localhost,127.0.0.1" \
    -v "$(pwd)/run_gaia_workforce_vllm_planner.py:/app/run_gaia_workforce_vllm_planner.py" \
    agent-image \
    -l 1 -t 41 2>&1 | tee L1_41_0.txt
```

The image's entrypoint is `run_gaia_workforce_vllm_planner.py`; the bind mount
lets you edit the runner without rebuilding. Flags: `-l` level, `-t`
comma-separated task indices, `-p` server port, `--model_name` model id.

See `runner.sh` for the batch form (loop over levels and task indices).