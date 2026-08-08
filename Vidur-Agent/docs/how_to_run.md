# How to run?

This guide will enable you to run Vidur for different scenarios such as different models, devices, workloads etc. To run Vidur, these are main things you need to specify:

1. **Model**: The model you want to run (e.g., `meta-llama/Meta-Llama-3-8B`).
1. **Cluster**: Number of replicas and their configuration: device e.g. `h100`, network device e.g. `h100_dgx`, tensor parallel size (TP), and pipeline parallel size (PP).
1. **Workload**: The lengths and the arrival time of inference requests.
1. **Scheduler**: The scheduling algorithms to use for the replicas.

The following command simulates a scenario with a H100 DGX node running 8 replicas of the `Meta-Llama-3-8B` model, with requests generated from the `mooncake_conversation` trace at a QPS of 8. Round robin routing policy is used and the replica scheduler is set to `vllm_v1`.

```sh
python -m vidur.main \
--time_limit 10800 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
--replica_config_device h100 \
--replica_config_network_device h100_dgx \
--cluster_config_num_replicas 8 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 128 \
--length_generator_config_type trace \
--trace_request_length_generator_config_trace_file ./data/processed_traces/mooncake_conversation_trace.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 8.0 \
--global_scheduler_config_type round_robin \
--replica_scheduler_config_type vllm_v1 \
--vllm_v1_scheduler_config_chunk_size 512 \
--vllm_v1_scheduler_config_batch_size_cap 512 \
--cache_config_enable_prefix_caching
```

Next, we'll discuss how to specify these using different parameters and run Vidur.

## Model

Let's explain with examples:

1. Run `Meta-Llama-3-8B`.

    ```sh
    --replica_config_model_name meta-llama/Meta-Llama-3-8B \
    --replica_config_tensor_parallel_size 1 \
    --replica_config_num_pipeline_stages 1 \
    ```

1. Run `Meta-Llama-3-70B` model with TP4 and PP1.

    ```sh
    --replica_config_model_name meta-llama/Meta-Llama-3-70B \
    --replica_config_tensor_parallel_size 4 \
    --replica_config_num_pipeline_stages 1 \
    ```

List of supported models can be found at [model_config.py](vidur/config/model_config.py).

## Cluster

Here we control the device (GPU SKU e.g. `h100`, `a100`), node (`network_device` e.g. `h100_dgx`, `a100_pair_nvlink`), number of replicas, and the TP and PP dimension of each replica.

Let's take a starting example where we are running 8 replicas of a model, each on a single `h100` card as both the TP and PP dimension is set to 1. Replicas are independent and do not have say `all_reduce` between them. So the `network_device` is not relevant here.

```sh
--replica_config_device h100 \
--replica_config_network_device h100_dgx \
--cluster_config_num_replicas 8 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
```

Now say we have a bigger model that we want to run at TP4 and PP1 on a `h100_dgx` node with 8 `h100` GPUs. In this case, each node will have two replicas (8 / 4 = 2). Since we need 8 replicas, 4 nodes are needed. The command will look like this:

```sh
--replica_config_device h100 \
--replica_config_network_device h100_dgx \
--cluster_config_num_replicas 8 \
--replica_config_tensor_parallel_size 4 \
--replica_config_num_pipeline_stages 1 \
```

See [device_sku_config](vidur/config/device_sku_config.py) and [network_sku_config](vidur/config/network_sku_config.py) for the list of supported devices and node configurations.

## Workload

The workload is the set of requests with each request having at minimum its input length (`num_prefill_tokens`), output length (`num_decode_tokens`), and the arrival time. Vidur has several request generators which can be composed together to create a workload. Let's take a look at the most common of them:

### Request Length Generators

1. `trace` is the most common request length generator which reads the request lengths from a trace file. The trace file should be a CSV file with at-least two columns: `num_prefill_tokens` and `num_decode_tokens`.

    ```sh
    --request_generator_config_type trace \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/mooncake_conversation_trace.csv \
    ```

1. `fixed` generates requests each having the same prefill and decode length.

    ```sh
    --request_generator_config_type fixed \
    --fixed_request_length_generator_config_prefill_tokens 2048 \
    --fixed_request_length_generator_config_decode_tokens 512 \
    ```

More request length generators can be found at [request_length_generator_registry.py](vidur/request_generator/request_length_generator_registry.py).

### Request Interval Generators

Interval generators control the arrival time of requests. The first request arrives at time 0, and the subsequent requests arrive at the specified intervals. The most common interval generators are:

1. `poisson` generates requests at a specified QPS (queries per second) using the poisson distribution.

    ```sh
    --interval_generator_config_type poisson \
    --poisson_request_interval_generator_config_qps 8.0 \
    ```

1. `static` makes all requests arrive at start t=0.

    ```sh
    --interval_generator_config_type static \
    ```

More request interval generators can be found at [request_interval_generator_registry.py](vidur/request_generator/request_interval_generator_registry.py).

### Request Generators

1. `synthetic` request generator combines a length generator and a interval (between request arrival) generator to the workload. In the example below, we take requests from the `mooncake_conversation` trace.csv` file and generate requests at a QPS of 8.0.

    ```sh
    --request_generator_config_type synthetic \
    --synthetic_request_generator_config_num_requests 128 \
    --length_generator_config_type trace \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/mooncake_conversation_trace.csv \
    --interval_generator_config_type poisson \
    --poisson_request_interval_generator_config_qps 8.0 \
    ```

1. `trace` request generator reads the request lengths and arrival times from the same trace file. The trace file should be a CSV file with at-least three columns: `num_prefill_tokens`, `num_decode_tokens`, and `arrived_at`. The `arrival_time` column specifies the time at which the request arrives in the system in seconds (float) ideally starting from 0.

    ```sh
    --request_generator_config_type trace \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/mooncake_conversation_trace.csv \
    ```

Note: For prefix caching, the trace file should have two columns `block_hash_ids` and `block_size`. The `block_hash_ids` should be a stringified list of block hash IDs for the entire length (input + output) of each request. `block_size` should be 16 for each request as Vidur only supports block size 16 today. See [mooncake_conversation_trace.csv](data/processed_traces/mooncake_conversation_trace.csv) for an such a trace file.

## Scheduler

Vidur has a hierarchy of schedulers inside it. The top-level scheduler is the global scheduler which schedules requests across replicas. The replica scheduler is responsible for scheduling requests on a specific replica.

### Global Scheduler

This is used use to route requests across replicas. The most common global schedulers are:

1. `round_robin`: schedules requests in a round-robin fashion across replicas.
1. `lor`: The Least Outstanding Requests (LOR) scheduler schedules requests to the replica with the least number of outstanding requests.
1. `lop`: The Least Outstanding Prefill (LOP) scheduler schedules requests to the replica with the least number of outstanding prefill tokens.
1. `sticky`: Each request has a `session_id`. Requests with same `session_id` are routed to the same replica. For a new `session_id`, `lop` policy is used to select the replica. __Note that session_id must be supplied in the trace file using a `session_id` column.__

Several other global schedulers can be found at [global_scheduler_registry.py](vidur/scheduler/global_scheduler/global_scheduler_registry.py).

### Replica Scheduler

The replica scheduler is responsible for scheduling requests on a specific replica. The most common replica schedulers are:

1. `vllm_v1` (recommended): This is the default scheduler which is based on the [vLLM V1 scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py). It supports chunked prefill (sarathi), paged attention, continuous batching, and prefix caching.

    ```sh
    --replica_scheduler_config_type vllm_v1 \
    --vllm_v1_scheduler_config_chunk_size 512 \
    --vllm_v1_scheduler_config_batch_size_cap 512 \
    ```

1. `sarathi`: From [Sarathi-Serve](https://arxiv.org/abs/2403.02310) paper.

    ```sh
    --replica_scheduler_config_type sarathi \
    --sarathi_scheduler_config_chunk_size 512 \
    --sarathi_scheduler_config_batch_size_cap 512 \
    ```

Other replica schedulers including Orca and FasterTransformer can be found at [replica_scheduler_registry.py](vidur/scheduler/replica_scheduler/replica_scheduler_registry.py).


### Metrics

The simulator logs a variety of metrics to help analyze the performance of the system. The metrics are stored in the `simulator_output` directory and can also be logged to wandb. Some important parameters to control the metrics are:

1. `--metrics_config_keep_individual_batch_metrics`: If passed, individual batch metrics are logged. This is useful for analyzing the metrics of each batch (essentially each forward pass) separately at the cost of greatly increased simulation time and disk usage.
1. `--metrics_config_store_operation_metrics`: If passed, operation metrics are logged. This is useful for analyzing the performance of each operation in the model (e.g., MLP, attention) separately.
