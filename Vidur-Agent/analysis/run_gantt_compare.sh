#!/usr/bin/env bash
set -uo pipefail

# `python` is not on PATH in this environment; use the project venv by default.
PYTHON="${PYTHON:-.venv/bin/python}"
# Target folders live under simulator_output.
ROOT="${ROOT:-simulator_output/}"

declare -A TARGET_RUN_TYPES
TARGET_RUN_TYPES["Owl"]="0 3 10"
TARGET_RUN_TYPES["Owl_70B"]="3"
TARGET_RUN_TYPES["Miro"]="0 3 7 9 10"
TARGET_RUN_TYPES["Miro_70B"]="3"

failed=()
for target in "${!TARGET_RUN_TYPES[@]}"; do
    read -ra RUN_TYPES <<< "${TARGET_RUN_TYPES[$target]}"
    for run_type in "${RUN_TYPES[@]}"; do
        echo ">>> target=$target  run_type=$run_type"
        # Keep going after a failure so one missing run does not hide the rest.
        if ! "$PYTHON" analysis/gantt_compare/plot_gantt_comparison.py \
            --root "$ROOT" --target "$target" --run-type "$run_type"; then
            failed+=("$target/type$run_type")
        fi
    done
done

echo "Plots written to analysis/gantt_compare/"
if (( ${#failed[@]} )); then
    printf 'FAILED: %s\n' "${failed[@]}" >&2
    exit 1
fi
