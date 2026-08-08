#!/usr/bin/env bash
#
# run_all.sh — rebuild the OWL results in place.
#
#   traces/session_traces → traces/owl_random.csv → analysis/outputs/
#
# The source logs (raw/{agent,vllm}) are not shipped with this artifact, so the
# tracegen stage is skipped and the checked-in session_traces — which are exactly
# its output — are used as-is. Restore the logs and the same command rebuilds
# raw/ → session_traces first.
#
# Usage:  ./run_all.sh            (or: PY=/path/to/python ./run_all.sh)
#         pip install -r requirements.txt
#
# Overwrites traces/owl_random.csv, traces/selected_set.txt and
# analysis/outputs/. Re-running is safe.
#
# Tool latency is not re-measured (needs API keys and a vLLM server); the stored
# raw/tool/*.csv are read as-is. See agent/tool_bench.py.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PY:-python3}"

step() { printf '\n\033[1m── %s\033[0m\n' "$*"; }

"$PY" -c 'import tiktoken, jinja2, matplotlib' 2>/dev/null || {
    printf 'error: %s is missing deps — pip install -r requirements.txt\n' "$PY" >&2; exit 1; }

step "tracegen  (slow — tokenises every request)"
# The source logs are not shipped with the artifact (see root README); when they
# are absent the shipped traces/session_traces are already the tracegen output.
if [ -d raw/agent ] && [ -d raw/vllm ]; then
    cd tracegen
    "$PY" parser.py
    "$PY" replace_oss_rows.py
    cd ..
else
    printf '  skipping — source logs not in this artifact; starting from traces/\n'
fi

step "dataset"
cd traces
"$PY" select_set.py
"$PY" merge.py -o owl_random.csv
cd ..

step "analysis"
cd analysis
for s in table1_owl accuracy figure3 figure5; do "$PY" "$s.py" >/dev/null; done
# figure4 compares one OWL trace against one MiroThinker trace, so it needs both trees built
if [ -d ../../mirothinker/traces/session_traces ]; then
    "$PY" figure4.py >/dev/null
else
    printf '  skipping figure4.py — run mirothinker/run_all.sh first\n'
fi
cd ../agent
"$PY" tool_bench.py --summary --selected-only > ../analysis/outputs/table2_owl.txt
cd ..
ls analysis/outputs | sed 's|^|  |'

printf '\n\033[32mdone\033[0m\n'