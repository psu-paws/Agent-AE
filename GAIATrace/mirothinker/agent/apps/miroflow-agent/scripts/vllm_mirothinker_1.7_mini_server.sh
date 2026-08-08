#!/usr/bin/env bash
# Start vLLM OpenAI API for MiroThinker-1.7-mini. Run on the GPU machine (not inside miroflow-agent uv).
#
# Align --max-model-len with apps/miroflow-agent/conf/llm/mirothinker-mini.yaml (max_context_length)
# and keep prompt + max_tokens under that budget (Hydra llm.max_tokens).
#
# Default 32k is a sane GAIA/benchmark setting for 2x large GPUs; lower only if OOM (e.g. 16384, 8192).
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec taskset -c "${CPU_AFFINITY:-0-27}" python3 -m vllm.entrypoints.openai.api_server \
  --model miromind-ai/MiroThinker-1.7-mini \
  --dtype float16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --tensor-parallel-size "${VLLM_TP:-2}" \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8080}" \
  --served-model-name "${VLLM_SERVED_MODEL_NAME:-mirothinker-mini}" \
  "$@"
