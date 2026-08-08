#!/usr/bin/env bash
#
# run_all.sh — rebuild the MiroThinker results in place.
#
#   traces/session_traces → traces/miro_random.csv → analysis/outputs/
#
# The source logs (raw/{agent,summarizer}) are not shipped with this artifact, so
# the tracegen stage is skipped and the checked-in session_traces — which are
# exactly its output — are used as-is. Restore the logs and the same command
# rebuilds raw/ → session_traces first.
#
# Usage:  ./run_all.sh            (or: PY=/path/to/python ./run_all.sh)
#         pip install -r ../owl/requirements.txt
#
# Overwrites traces/miro_random.csv and analysis/outputs/. Re-running is safe.
#
# parser.py, in the tracegen stage only, needs the Qwen3 tokenizer and downloads
# it from HuggingFace on first run; everything after that is offline.
#
# Tool latency is not re-measured (needs API keys and network); the stored
# raw/tool/*.csv are read as-is. See agent/apps/miroflow-agent/benchmarks/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PY:-python3}"

step() { printf '\n\033[1m── %s\033[0m\n' "$*"; }

"$PY" -c 'import tiktoken, jinja2, matplotlib' 2>/dev/null || {
    printf 'error: %s is missing deps — pip install -r ../owl/requirements.txt\n' "$PY" >&2; exit 1; }

step "tracegen  (slow — tokenises every request)"
# The source logs are not shipped with the artifact (see root README); when they
# are absent the shipped traces/session_traces are already the tracegen output.
if [ -d raw/agent ] && [ -d raw/summarizer ]; then
    cd tracegen
    "$PY" parser.py
    "$PY" replace_summarizer_placeholders.py ../traces/session_traces
    "$PY" extract_tools.py
    cd ..
else
    printf '  skipping — source logs not in this artifact; starting from traces/\n'
fi

step "dataset"
cd traces
"$PY" merge.py -o miro_random.csv
cd ..

step "analysis"
cd analysis
for s in table1_miro table2_miro figure2; do "$PY" "$s.py" >/dev/null; done
# score_gaia needs GAIA answers, which are not shipped with this artifact.
if [ -f "${GAIA_GROUND_TRUTH:-../../data/gaia_text103.jsonl}" ]; then
    "$PY" score_gaia.py --ground-truth "${GAIA_GROUND_TRUTH:-../../data/gaia_text103.jsonl}" >/dev/null
else
    printf '  skipping score_gaia.py — set GAIA_GROUND_TRUTH to the GAIA answers JSONL\n'
fi
cd ..
ls analysis/outputs | sed 's|^|  |'

printf '\n\033[32mdone\033[0m\n'
