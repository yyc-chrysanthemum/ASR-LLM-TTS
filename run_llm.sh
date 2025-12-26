#!/bin/bash
# run_llm.sh — wrapper to start rkllm_aaa in the right dir and force line buffering

LLM_DIR="/root/rknn-llm/rknn-llm-release-v1.2.2/examples/rkllm_api_demo/deploy"
RKLLM_BIN="./rkllm_aaa"
#MODEL_PATH=~/model_rknnn_llm/xiaoji_3576.rkllm
#MODEL_PATH=~/model_rknnn_llm/qwen2.5-1.5b-instruct_rk3576.rkllm
MODEL_PATH=~/model_rknnn_llm/qwen2.5-0.5b-instruct_w8a8_rk3576.rkllm
MAX_IN=1024
MAX_OUT=1024

export LD_LIBRARY_PATH=/root/rknn-llm/rknn-llm-release-v1.2.2/rkllm-runtime/Linux/librkllm_api/aarch64

cd "$LLM_DIR" || exit 1

# Use stdbuf to try to force line buffering. pexpect will spawn this script.
stdbuf -oL -eL $RKLLM_BIN "$MODEL_PATH" "$MAX_IN" "$MAX_OUT"

