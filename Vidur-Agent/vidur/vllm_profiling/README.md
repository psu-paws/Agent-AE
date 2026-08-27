# vLLM-based profiling

Produces the CSVs in `data/profiling/` by timing the kernels a real vLLM
deployment runs, rather than reimplementations of them.

    python -m vidur.vllm_profiling.mlp        --model Qwen/Qwen2.5-32B-Instruct --num_tensor_parallel_workers 2 4 8
    python -m vidur.vllm_profiling.attn       --model Qwen/Qwen2.5-32B-Instruct --num_tensor_parallel_workers 2 4 8 --max_seq_len 131072
    python -m vidur.vllm_profiling.collective --collective all_reduce --devices 0,1 --link-type NVLink

Outputs match the schema `vidur` reads:

    data/profiling/compute/{DEVICE}/{MODEL}/{mlp,attention}.csv
    data/profiling/network/{NETWORK_DEVICE}/{all_reduce,send_recv}.csv

Notes:

- `mlp` and `attn` need only one GPU at any TP degree. Tensor parallelism enters
  as a shard-shape divisor, not as a collective, so a single device measures one
  worker's kernels exactly. Only `collective` needs several GPUs.
- No model weights are downloaded; layers are built from `config.json` with
  random weights.
- `--max_seq_len` must cover the longest context the workload reaches. The
  execution-time predictor is tree-based and returns its boundary value outside
  the training range, so a sweep that stops short silently charges long-context
  work the price of short-context work.
- `attn` records `kv_layout`; keep it matched to the serving configuration.
- `collective` records `--link-type` as given. A node can reach different GPU
  pairs over different fabrics, so profile each class separately and label it.
