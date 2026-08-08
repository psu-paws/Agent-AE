#!/bin/bash
#
# Fit and cache the execution-time predictors the run_*.sh scripts need.
#
# Fitting uses every core, so configurations are trained one at a time. Run this
# once on a fresh clone; the run scripts then load from cache/ (~6.5 GB).
#
#   bash warmup_predictors.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT="simulator_output/warmup"
mkdir -p "$OUT"

# A predictor is keyed on (model, parallelism, device), so these two configs cover
# every combination the run scripts use: CodeLlama-34B at TP2/TP4/TP8, and
# Llama-3-70B at TP4.
CONFIGS=(8121-2141.json 4141-4141_70B.json)
TRACE="data/processed_traces/warmup_trace.csv"

FAILED=()
TIMES=()
START_ALL=$SECONDS

for cfg in "${CONFIGS[@]}"; do
    label="${cfg%.json}"
    echo "── $label"
    start=$SECONDS
    python -m vidur.main \
        --time_limit 1 \
        --length_generator_config_type trace \
        --trace_request_length_generator_config_trace_file "$TRACE" \
        --cluster_config_replica_groups_config "data/replica_groups_configs/$cfg" \
        --metrics_config_output_dir "$OUT/$label" \
        > "$OUT/$label.log" 2>&1
    code=$?
    elapsed=$((SECONDS - start))
    if [ $code -ne 0 ]; then
        echo "   FAILED (exit $code) — see $OUT/$label.log"
        FAILED+=("$label")
    else
        echo "   done in ${elapsed}s"
    fi
    TIMES+=("$label $elapsed")
done

TOTAL=$((SECONDS - START_ALL))
echo
for t in "${TIMES[@]}"; do set -- $t; printf '  %-20s %5ss\n' "$1" "$2"; done
printf '  %-20s %5ss  (%dm%02ds)\n' TOTAL "$TOTAL" $((TOTAL / 60)) $((TOTAL % 60))
echo "  cache/ $(du -sh cache 2>/dev/null | cut -f1)"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[warmup] failed: ${FAILED[*]}" >&2
    exit 1
fi
