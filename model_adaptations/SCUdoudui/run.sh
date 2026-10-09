#!/bin/bash
# SCUdoudui MiniCPM5-1B official launch script
# Launch contract for tools/ci_pipline/run_ci.py

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime:${PYTHONPATH:-}"
# The current vLLM-SDAA compiler path cannot fake-propagate the tecoops
# pybind pointer ABI. Bind the reviewed operators eagerly at process start.
export TORCH_COMPILE_DISABLE=1

vllm serve /gpfs/model/OpenBMB/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --tensor-parallel-size 1 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype float16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --max-model-len 32768
