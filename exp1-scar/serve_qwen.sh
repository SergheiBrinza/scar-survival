#!/usr/bin/env bash
# Launch Qwen2.5-7B-Instruct as a local OpenAI-compatible server on a SINGLE RTX 3090 (GPU 0).
# Port 8002. Once started, hit http://localhost:8002/v1/chat/completions
set -e

export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --port 8002 \
  --served-model-name qwen2.5-7b
