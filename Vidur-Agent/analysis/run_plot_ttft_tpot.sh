#!/usr/bin/env bash
# Run plot_ttft_tpot.py and plot_ttft_req_vs_session.py for all target/run-type combos.
# Optionally pass --session to also generate session-level plots from plot_ttft_tpot.py.
#
# Usage:
#   bash analysis/run_plot_ttft_tpot.sh              # request-level only
#   bash analysis/run_plot_ttft_tpot.sh --session    # request + session level

set -euo pipefail

declare -A TARGET_RUN_TYPES
TARGET_RUN_TYPES["Owl"]="0"
TARGET_RUN_TYPES["Miro"]="0 3"
SESSION_FLAGS=("")
if [[ "${1:-}" == "--session" ]]; then
    SESSION_FLAGS=("" "--session-level")
fi

for target in "${!TARGET_RUN_TYPES[@]}"; do
    read -ra RUN_TYPES <<< "${TARGET_RUN_TYPES[$target]}"
    for run_type in "${RUN_TYPES[@]}"; do
        echo ">>> target=$target  run_type=$run_type  (req_vs_session)"
        python analysis/ttft_tpot/plot_ttft_req_vs_session.py \
            --target "$target" \
            --run-type "$run_type"
        for session_flag in "${SESSION_FLAGS[@]}"; do
            echo ">>> target=$target  run_type=$run_type  session=${session_flag:-(none)}"
            python analysis/ttft_tpot/plot_ttft_tpot.py \
                --target "$target" \
                --run-type "$run_type" \
                ${session_flag:+$session_flag}
        done
    done
done

echo "Done. Plots written to analysis/ttft_tpot/"
