#!/bin/bash

# Simulator side of the fidelity table. Pairs with the ground-truth GPU runs
# on real hardware. Ground truth for the comparison ships in data/ground_truth.
#   MIRO_TRACE=/path/to/miro_random.csv bash run_validation.sh
TRACE="${MIRO_TRACE:-../GAIATrace/mirothinker/traces/miro_random.csv}"
if [[ ! -f "$TRACE" ]]; then
    echo "[run_validation.sh] trace not found: $TRACE" >&2
    exit 1
fi

BASE="simulator_output/validate"
PIDFILE="$BASE/pids.txt"
# chunk_size 2048 and max 128 seqs match the deployment's engine settings; the
# characterization scripts use the artifact default of 4096.
COMMON="python -m vidur.main \
  --synthetic_request_generator_config_num_requests 2082 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file $TRACE \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 0.1 \
  --global_scheduler_config_type load_aware \
  --replica_scheduler_config_type vllm_v1 \
  --vllm_v1_scheduler_config_chunk_size 2048 \
  --vllm_v1_scheduler_config_batch_size_cap 128 \
  --cache_config_enable_prefix_caching \
  --metrics_config_no_timestamp"

mkdir -p $BASE
> $PIDFILE  # clear pid file

kill_all() {
    echo "[run_validation.sh] Killing all runs..."
    while read pid; do
        kill "$pid" 2>/dev/null && echo "  killed $pid"
    done < "$PIDFILE"
    exit 1
}

trap kill_all INT TERM

launch() {
    local label=$1; shift
    # Record wall time per run; compare_speedup.py reads these.
    ( start=$(date +%s.%N)
      $COMMON "$@" --metrics_config_output_dir $BASE/$label > $BASE/${label}.txt 2>&1
      rc=$?
      printf '{"case":"%s","sim_wall_s":%.2f,"exit":%d}\n' "$label" \
        "$(echo "$(date +%s.%N) - $start" | bc)" "$rc" > $BASE/${label}.timing.json
      exit $rc ) &
    local pid=$!
    echo $pid >> $PIDFILE
    echo "[run_validation.sh] Started $label (pid $pid)"
}


# All four setups replay the same two-model workload.

# # S1: TP-2 prefill + TP-2 decode, Main-LLM and Sub-LLM sharing the GPUs (4 GPUs)
launch S1-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/validate_S1.json

# # S2: 1 prefill + 1 decode GPU, both models sharing them (2 GPUs)
launch S2-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/validate_S2.json

# # S3: the S1 placement running Qwen2.5-32B instead of Llama-8B (4 GPUs)
launch S3-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/validate_S3.json

# # S4: one GPU each for Main-LLM prefill/decode and Sub-LLM prefill/decode (4 GPUs)
launch S4-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/validate_S4.json

# # S1 at lower arrival rates
launch S1-QPS004 --cluster_config_replica_groups_config data/replica_groups_configs/validate_S1.json \
    --poisson_request_interval_generator_config_qps 0.04
launch S1-QPS001 --cluster_config_replica_groups_config data/replica_groups_configs/validate_S1.json \
    --poisson_request_interval_generator_config_qps 0.01

wait
echo
python analysis/compare_validation.py
