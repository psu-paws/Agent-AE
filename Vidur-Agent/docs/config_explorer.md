# Config Explorer

Vidur can be run in a massively parallel fashion, one simulation per CPU core. We regularly run almost all cores of our 100 CPU core machines with a TB of RAM. A common use case is the find the performance of various routing policies simultaneously for several models, traces, and latency SLOs.

Below we describe a way to run Vidur in a massively parallel fashion, using the `capacity_search.py` script. First, we'll create a capacity search YAML config and then run the script.

## YAML Config

```yaml
clusters: # Available node types
  - device: h100
    network_device: h100_dgx
    gpus_per_node: 8
  - device: a100
    network_device: a100_dgx
    gpus_per_node: 8

global_schedulers: # Routing policies to be tested, some policies require their own parameters.
  - scheduler: round_robin
  - scheduler: sticky_lop
  - scheduler: lop_uncached
  - scheduler: tolerant_sticky_lop_uncached
    tolerance_factor: 2.0
  - scheduler: tolerant_sticky_lop_uncached
    tolerance_factor: 3.0
  - scheduler: ranked_sticky_lop_uncached
    top_k: 2
  - scheduler: ranked_sticky_lop_uncached
    top_k: 4

request_queues:
  - name: fcfs
    provider: fcfs

replica_schedulers:
  - scheduler: vllm_v1
    chunk_size: 512
  - scheduler: vllm_v1
    chunk_size: 1024

slo_configs: # Latency SLOs to be tested
  - name: ttft_p99_10s # TTFT p99 should be at max 10s
    type: and
    quantile: 0.99
    slos:
      - metric: prefill_e2e_time
        value: 10
  - name: ttlt_p99_60s # TTLT p99 should be at max 60s
    type: and
    quantile: 0.99
    slos:
      - metric: request_e2e_time
        value: 60
  - name: ttft_p99_10s_and_tpot_p99_100ms # 99% of requests should have TTFT <= 10s and TPOT <= 100ms
    type: and # `or` is also supported
    quantile: 0.99
    slos:
      - metric: prefill_e2e_time
        value: 15
      - metric: decode_time_execution_plus_preemption_normalized
        value: 0.1

traces:
  - name: mooncake_conversation
    trace_file: "./data/processed_traces/mooncake_conversation_trace.csv"
    max_seq_len: 131072
    enable_prefix_caching: true
  - name: mooncake_toolagent
    trace_file: "./data/processed_traces/mooncake_toolagent_trace.csv"
    max_seq_len: 131072
    enable_prefix_caching: true

# allowed batch sizes and TP/PP dimensions
batch_sizes: [512]
tp_dimensions: [1, 4]
pp_dimensions: [1]

models:
  - name: Meta-Llama-3-8B
    identifier: meta-llama/Meta-Llama-3-8B
    exclude_tp_dims: [4] # Do not run TP4 for this model
  - name: Meta-Llama-3-70B
    identifier: meta-llama/Meta-Llama-3-70B
    exclude_tp_dims: [1]

search_configs:
  - trace: mooncake_conversation
    model: Meta-Llama-3-8B
    search_for: qps
    num_replicas: 8
    qps: 4
    num_requests: 12301
    # The above item searches for the maximum QPS (starting at 4.0, can go lower or higher) that can be achieved with 8 replicas of Meta-Llama-3-8B model on mooncake_conversation trace while being under the SLOs defined above.
  - trace: mooncake_toolagent
    model: Meta-Llama-3-8B
    search_for: qps
    num_replicas: 8
    qps: 8
    num_requests: 23608
  - trace: mooncake_conversation
    model: Meta-Llama-3-70B
    search_for: qps
    num_replicas: 16
    qps: 4
    num_requests: 12301
  - trace: mooncake_toolagent
    model: Meta-Llama-3-70B
    search_for: qps
    num_replicas: 16
    qps: 8
    num_requests: 23608

```

## How to run?

1. Ensure that simulator is setup and working. See [README.md](../../README.md) for instructions.
1. Recommend to disable `wandb` logging by setting `WANDB_MODE=disabled` in the shell environment. `wandb` cannot handle such load.
1. Recommend to run the script in a `tmux` or `screen` session as it will run for a long time and you may want to disconnect from the session.
1. Run the following command:

    ```sh
    python -u -m vidur.config_optimizer.config_explorer.main \
    --config-path path/to/config.yml \
    --cache-dir cache \
    --output-dir config_optimizer_output \
    --time-limit 180 \
    ```

1. If the script exits with an error, you can resume the search from the last iteration by running the same command again. The script consults the completed runs in the path specified by `--output_dir` to determine till point it had searched previously and resume from there.
1. Search can be sped up using `tmpfs` filesystem for `cache` and `config_optimizer_output` folders.

    ```sh
    mkdir cache
    sudo mount -t tmpfs tmpfs ./cache -o size=32000m
    mkdir config_optimizer_output
    sudo mount -t tmpfs tmpfs ./config_optimizer_output -o size=32000m
    ```

1. Recommend to backup entire `config_optimizer_output` folder immediately after the search is complete and in between also.
1. Use `htop` command to monitor that all except 2 cores are running with 100% CPU utilization. If not, then the search is not running at full speed.

## Interpreting the results

The script will output the results in the `config_optimizer_output` folder (specified by `--output-dir` argument).
There will be a lot of binary searches each will several iterations in it.

```plain
- config_optimizer_output
    - runs
        - 8bf217e6/
            - r8_q8.0
                - 2024-01-14_19-40-01-606280
                    - plots/
                    - config.json
                    - request_metrics.csv
                - output.log
            - r8_q4.0
                - 2024-01-14_19-40-02-606280
                    - plots/
                    - config.json
                    - request_metrics.csv
                - output.log
        - 9b9581d3/
            ...
    - args.json
    - config.json
```

Each iteration will have uniquely timestamped folder. In it, a `config.yml` file will contain the configuration used for that iteration. The `request_metrics.csv` file will contain the metrics for each request served in that iteration. The `plots` folder will contain the plots and csvs for all the metrics.
Recommend to backup entire `simulator_output` folder after the search is complete and do offline analysis on it using custom scripts.
