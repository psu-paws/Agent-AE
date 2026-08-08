#!/bin/bash

# Trace produced by the GAIATrace stage. Override if it lives elsewhere:
#   OWL_TRACE=/path/to/owl_random.csv bash run_owl70B.sh
TRACE="${OWL_TRACE:-../GAIATrace/owl/traces/owl_random.csv}"
if [[ ! -f "$TRACE" ]]; then
    echo "[run_owl70B.sh] trace not found: $TRACE" >&2
    echo "  Generate it in GAIATrace first, or set OWL_TRACE=/path/to/owl_random.csv" >&2
    exit 1
fi

BASE="simulator_output/Owl_70B"
PIDFILE="$BASE/pids.txt"
COMMON="python -m vidur.main \
  --synthetic_request_generator_config_num_requests 5407 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file $TRACE \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 0.1 \
  --global_scheduler_config_type load_aware \
  --replica_scheduler_config_type vllm_v1 \
  --vllm_v1_scheduler_config_chunk_size 4096 \
  --vllm_v1_scheduler_config_batch_size_cap 256 \
  --cache_config_enable_prefix_caching \
  --metrics_config_no_timestamp"

mkdir -p $BASE
> $PIDFILE  # clear pid file

kill_all() {
    echo "[run.sh] Killing all runs..."
    while read pid; do
        kill "$pid" 2>/dev/null && echo "  killed $pid"
    done < "$PIDFILE"
    exit 1
}

trap kill_all INT TERM

launch() {
    local label=$1; shift
    $COMMON "$@" --metrics_config_output_dir $BASE/$label \
        2>&1 | tee $BASE/${label}.txt &
    local pid=$!
    echo $pid >> $PIDFILE
    echo "[run.sh] Started $label (pid $pid)"
}

# # Prefix Cache
# launch 4141-4141-NoPC  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --no-cache_config_enable_prefix_caching
# launch 4141-4141-NoPC-SFCFS  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --no-cache_config_enable_prefix_caching  --vllm_v1_scheduler_config_session_priority

# # Arrival Rate
# launch 4141-4141-QPS005 --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --poisson_request_interval_generator_config_qps 0.05
# launch 4141-4141-QPS01  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --poisson_request_interval_generator_config_qps 0.1
# launch 4141-4141-QPS03  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --poisson_request_interval_generator_config_qps 0.3
# launch 4141-4141-QPS05   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --poisson_request_interval_generator_config_qps 0.5

# # Scheduling (QPS 0.05)
# launch 4141-4141-SFCFS-QPS005    --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_session_priority  --poisson_request_interval_generator_config_qps 0.05
# launch 4141-4141-SJF-QPS005    --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --poisson_request_interval_generator_config_qps 0.05
# launch 4141-4141-SJF60-QPS005   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 60.0 \
#     --poisson_request_interval_generator_config_qps 0.05
# launch 4141-4141-SJF60-SFCFS-QPS005   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 60.0 \
#     --poisson_request_interval_generator_config_qps 0.05 --vllm_v1_scheduler_config_sjf_starvation_session_fcfs
# launch 4141-4141-SJF180-QPS005  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 180.0 \
#     --poisson_request_interval_generator_config_qps 0.05
# launch 4141-4141-SJF180-SFCFS-QPS005  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 180.0 \
#     --poisson_request_interval_generator_config_qps 0.05 --vllm_v1_scheduler_config_sjf_starvation_session_fcfs

# # Scheduling (QPS 0.1)
# launch 4141-4141-SFCFS-QPS01    --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_session_priority  
# launch 4141-4141-SJF-QPS01    --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority
# launch 4141-4141-SJF60-QPS01   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 60.0 \
#     --poisson_request_interval_generator_config_qps 0.1
# launch 4141-4141-SJF60-SFCFS-QPS01   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 60.0 \
#     --poisson_request_interval_generator_config_qps 0.1 --vllm_v1_scheduler_config_sjf_starvation_session_fcfs
# launch 4141-4141-SJF180-QPS01  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 180.0 \
#     --poisson_request_interval_generator_config_qps 0.1
# launch 4141-4141-SJF180-SFCFS-QPS01  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 180.0 \
#     --poisson_request_interval_generator_config_qps 0.1 --vllm_v1_scheduler_config_sjf_starvation_session_fcfs

# Scheduling (QPS 0.5)
launch 4141-4141-SFCFS-QPS05    --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
    --vllm_v1_scheduler_config_session_priority  --poisson_request_interval_generator_config_qps 0.5
launch 4141-4141-SJF-QPS05     --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
    --vllm_v1_scheduler_config_sjf_active_priority --poisson_request_interval_generator_config_qps 0.5
# # launch 4141-4141-SJF60-QPS05   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
# #     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 60.0 \
# #     --poisson_request_interval_generator_config_qps 0.5
# # launch 4141-4141-SJF60-SFCFS-QPS05   --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
# #     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 60.0 \
# #     --poisson_request_interval_generator_config_qps 0.5 --vllm_v1_scheduler_config_sjf_starvation_session_fcfs
launch 4141-4141-SJF180-QPS05  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
    --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 180.0 \
    --poisson_request_interval_generator_config_qps 0.5
# launch 4141-4141-SJF180-SFCFS-QPS05  --cluster_config_replica_groups_config data/replica_groups_configs/4141-4141_70B.json \
#     --vllm_v1_scheduler_config_sjf_active_priority --vllm_v1_scheduler_config_sjf_starvation_timeout 180.0 \
#     --poisson_request_interval_generator_config_qps 0.5 --vllm_v1_scheduler_config_sjf_starvation_session_fcfs

# # Model Config
# launch 4141-2321-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/4141-2321.json
# launch 2181-4121-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/2181-4121.json
# launch 2141-8121-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/2141-8121.json
# launch 2181-2141-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/2181-2141.json
# launch 4121-8121-QPS01 --cluster_config_replica_groups_config data/replica_groups_configs/4121-8121.json

# # Model Config
# launch 4141-2321-QPS005 --cluster_config_replica_groups_config data/replica_groups_configs/4141-2321.json --poisson_request_interval_generator_config_qps 0.05
# launch 2181-4121-QPS005 --cluster_config_replica_groups_config data/replica_groups_configs/2181-4121.json --poisson_request_interval_generator_config_qps 0.05
# launch 2141-8121-QPS005 --cluster_config_replica_groups_config data/replica_groups_configs/2141-8121.json --poisson_request_interval_generator_config_qps 0.05
# launch 2181-2141-QPS005 --cluster_config_replica_groups_config data/replica_groups_configs/2181-2141.json --poisson_request_interval_generator_config_qps 0.05
# launch 4121-8121-QPS005 --cluster_config_replica_groups_config data/replica_groups_configs/4121-8121.json --poisson_request_interval_generator_config_qps 0.05

# # Model Config
# launch 4141-2321-QPS05 --cluster_config_replica_groups_config data/replica_groups_configs/4141-2321.json --poisson_request_interval_generator_config_qps 0.5
# launch 2181-4121-QPS05 --cluster_config_replica_groups_config data/replica_groups_configs/2181-4121.json --poisson_request_interval_generator_config_qps 0.5
# launch 2141-8121-QPS05 --cluster_config_replica_groups_config data/replica_groups_configs/2141-8121.json --poisson_request_interval_generator_config_qps 0.5
# launch 2181-2141-QPS05 --cluster_config_replica_groups_config data/replica_groups_configs/2181-2141.json --poisson_request_interval_generator_config_qps 0.5
# launch 4121-8121-QPS05 --cluster_config_replica_groups_config data/replica_groups_configs/4121-8121.json --poisson_request_interval_generator_config_qps 0.5

echo "[run.sh] All launched. To kill everything: kill \$(cat $PIDFILE)"
echo "[run.sh] Waiting for all runs to finish..."

FAILED=()
while read pid; do
    wait "$pid"
    code=$?
    if [ $code -ne 0 ]; then
        echo "[run.sh] FAILED (exit $code): $pid"
        FAILED+=("$pid")
    fi
done < "$PIDFILE"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[run.sh] ${#FAILED[@]} run(s) failed: ${FAILED[*]}"
    exit 1
else
    echo "[run.sh] All runs completed successfully."
fi

echo "[run.sh] Building request group latency CSVs..."
for run_dir in $BASE/*/; do
    echo "  Processing $run_dir"
    python3 analysis/build_request_group_latency.py "$run_dir"
    
done
echo "[run.sh] Done."

echo "[run.sh] Plotting timeline..."
# (cd "$BASE" && python3 ../../analysis/plot_timeline.py)
