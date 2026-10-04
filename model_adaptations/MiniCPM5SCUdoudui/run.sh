#!/bin/bash
# SCUdoudui MiniCPM5-1B official launch script
set -euo pipefail
# Launch contract for tools/ci_pipline/run_ci.py

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${MINICPM5_MODEL:-${MODEL_ROOT}/OpenBMB/MiniCPM5-1B}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ "${MINICPM_OP_PROFILE:-official}" != "official" ]]; then
    echo "FATAL: This export supports only MINICPM_OP_PROFILE=official" >&2
    exit 1
fi
export MINICPM_OP_PROFILE=official
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime:${PYTHONPATH:-}"
# Official kernels are registered as opaque torch.library operations.
# Leave vLLM default compilation enabled.

exec /home/py312/bin/python -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B \
    --tensor-parallel-size 1 \
    --port "${MINICPM_PORT:-8000}" \
    --host "${MINICPM_HOST:-0.0.0.0}" \
    --dtype float16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --max-model-len "${MINICPM_MAX_MODEL_LEN:-32768}" \
    "$@"
