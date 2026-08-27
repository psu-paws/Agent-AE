# Artifact: *Vidur-Agent*

Artifact for *&lt;Characterizing How Complex Agentic AI Systems Handle General Tasks: A Trace-Based Simulation Study&gt;*

This artifact reproduces every simulation result and figure in the paper. 
It extends **[Vidur](https://github.com/microsoft/vidur)**, an LLM inference simulator, at commit
[`25e0082`](https://github.com/microsoft/vidur/commit/25e0082dbbfb206fb0477c3ebbededa7ead78949).
original README is preserved as [`README_VIDUR.md`](README_VIDUR.md).

---

## Extension over Vidur

Vidur simulates a homogeneous cluster: one model, one replica configuration, one scheduler. This artifact extends it to study multi-model,  prefill–decode-disaggregated serving of agentic workloads:

| Capability | Summary |
|---|---|
| **Heterogeneous clusters** | Replicas are declared in JSON *replica groups*, each with its own model, parallelism, device, scheduler and cache config. |
| **Prefill–decode (PD) disaggregation** | Replicas take a `prefill` or `decode` role. A request is prefilled on one replica and handed off to another, with the KV transfer cost modelled. |
| **Multi-turn agent sessions** | Traces carry sessions of dependent turns. Turn *k+1* arrives after turn *k* completes plus a tool time, and can reuse turn *k*'s KV. |
| **Scheduling policies** | Session-aware FCFS, shortest-job-first with starvation prevention, and a load-aware / KV-affinity global Dynamo-style router. |
| **Metrics** | Per-session (multi-turn) metrics alongside per-replica ones, plus per-replica KV eviction counts and token-level cache hit rate. |

---

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv venv
source .venv/bin/activate
uv sync
```

Execution times come from the profiling data already in
`data/profiling/`, so the simulator runs entirely on CPU.
No GPU is needed unless running profiling. 

---

## Reproducing the paper

**Generate the traces first.** These runs consume the CSVs produced by the
[GAIATrace](../GAIATrace) stage — see the [top-level README](../README.md) for the
full pipeline. The scripts expect them at

```
../GAIATrace/mirothinker/traces/miro_random.csv
../GAIATrace/owl/traces/owl_random.csv
```

and exit with a message if they are absent. Set `MIRO_TRACE=` / `OWL_TRACE=` to
point elsewhere.

### Step 1 — warm up the execution-time predictors

```sh
bash warmup_predictors.sh
```

Vidur predicts operator execution times with sklearn models fitted to
`data/profiling/`, one per (model, device, parallelism) combination, cached in
`cache/`. Fitting uses every core on the machine; loading from cache is fast and
single-threaded.

`warmup_predictors.sh` trains each configuration used in this artifact, which is everything the four following run scripts need. It takes ~20 min on 64 cores and leaves ~6.5 GB in `cache/`.
`cache/` is not shipped, so a fresh clone must fit them once. The run scripts below launch their runs concurrently.

### Step 2 — run the simulations

Four scripts, each launching its runs in parallel and writing to
`simulator_output/<NAME>/`:

| Script | Workload | Model | Output | Runs |
|---|---|---|---|---|
| `run_miro.sh` | `miro_random.csv` (2,082 requests, 103 sessions) | CodeLlama-34B | `simulator_output/Miro/` | 12 |
| `run_owl.sh` | `owl_random.csv` (5,406 requests, 165 sessions) | CodeLlama-34B | `simulator_output/Owl/` | 8 |
| `run_miro70B.sh` | `miro_random.csv` | Llama-3-70B | `simulator_output/Miro_70B/` | 3 |
| `run_owl70B.sh` | `owl_random.csv` | Llama-3-70B | `simulator_output/Owl_70B/` | 3 |

```sh
bash run_miro.sh
bash run_owl.sh
bash run_miro70B.sh
bash run_owl70B.sh
```

Each script waits for its runs to finish, reports any failures, then builds the
per-request-group latency CSVs via `analysis/build_request_group_latency.py`.

With a warm `cache/` every run is single-threaded, so the scripts above can also
be backgrounded and overlapped with each other. Memory is the binding constraint
rather than cores: each run holds its trace in memory, ~5 GB on the Miro
workload and ~7.5 GB on the larger Owl one, so all 26 at once wants ~150 GB. Run
the scripts one after another where that does not fit; the results are identical,
only the wall-clock differs.

### Step 3 — generate the figures

```sh
bash analysis/run_gantt_compare.sh     # -> analysis/gantt_compare/gantt_<target>_type<N>.pdf
bash analysis/run_plot_ttft_tpot.sh    # -> analysis/ttft_tpot/*.pdf
```


## Running a single simulation

```sh
python -m vidur.main \
  --synthetic_request_generator_config_num_requests 2082 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file ../GAIATrace/mirothinker/traces/miro_random.csv \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 0.1 \
  --global_scheduler_config_type load_aware \
  --replica_scheduler_config_type vllm_v1 \
  --vllm_v1_scheduler_config_chunk_size 4096 \
  --vllm_v1_scheduler_config_batch_size_cap 256 \
  --cache_config_enable_prefix_caching \
  --metrics_config_no_timestamp \
  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_qwen32b.json \
  --metrics_config_output_dir simulator_output/example
```

### Cluster configuration format

```jsonc
{
  "replica_groups": [
    { "role": "prefill", "num_replicas": 1,
      "replica_config": { "model_name": "codellama/CodeLlama-34b-Instruct-hf",
                          "tensor_parallel_size": 4, "device": "a100",
                          "pd_disaggregation": true } }
  ],
  // Per model: which replica indices prefill / decode.
  // Array position is the model_id matching the trace's model_id column.
  "replica_groups_pools": [
    { "prefill": [0], "decode": [2], "cross_node": true }
  ]
}
```

`cross_node` selects the inter-node bandwidth for KV handoff instead of
the profiled intra-node `send_recv` model.

A group can instead take `"role": "agg"` (the default when `role` is omitted),
where each replica both prefills and decodes its own requests using chunked prefill.
Refer to the reference files in `data/replica_groups_configs/`.

### Trace format

Minimum columns are `num_prefill_tokens` and `num_decode_tokens`.

| Column | Purpose |
|---|---|
| `session_id` | Groups turns of one conversation. Absent → each request is single-turn session. |
| `turn_id` | Turn index within the session. Absent → derived from row order. |
| `request_id` | Unique request id |
| `model_id` | Which model (0-indexed) the request targets; indexes `replica_groups_pools`. |
| `token_ids` | Real token ids. Enables token-exact prefix-cache matching. |
| `block_hash_ids` | Pre-computed per-block hashes. Coarser fallback when `token_ids` is absent. |
| `inter_request_latency` | Interval between a turn (e.g. Tool execution time). |
| `dep` | Turn ids that must all complete before this turn is released. |

---

## License

This artifact is derived from [Vidur](https://github.com/microsoft/vidur), released under the MIT License.
The original license is retained in [`LICENSE`](LICENSE) and applies to the upstream code. This project is not affiliated with or endorsed by Microsoft.