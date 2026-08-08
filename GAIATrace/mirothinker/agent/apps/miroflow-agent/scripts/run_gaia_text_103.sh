#!/usr/bin/env bash
# Full GAIA text-103 benchmark. API keys and optional LLM_BASE_URL / LLM_API_KEY come from apps/miroflow-agent/.env
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python benchmarks/common_benchmark.py \
  benchmark=gaia-validation-text-103 \
  agent=mirothinker_v1.5_keep20_max200 \
  llm=mirothinker-mini \
  "$@"