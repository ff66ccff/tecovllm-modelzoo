#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
GEMMA_PATH="${GEMMA_PATH:-${MODEL_ROOT}/google/gemma-4-12B-it}"
PORT="${GEMMA_PORT:-8003}"
HOST="${GEMMA_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-4352}"
MAX_BATCHED="${GEMMA_MAX_BATCHED_TOKENS:-512}"
UTIL="${GEMMA_UTIL:-0.92}"

# Bind the reviewed implementation before vLLM constructs its attention
# classes. No vendor package is modified and no model weights are copied.
export GEMMA4_OVERLAY=1
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONPATH="${SCRIPT_DIR}/overlay:${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# The default serve command below is also the official CI parser contract.
# Resolve environment overrides once, then retain the vendor entrypoint.
vllm() {
    local -a task_argv=("$@")
    local task_index task_host_bound=0 task_port_bound=0
    task_argv[1]="${GEMMA_PATH}"
    for ((task_index=2; task_index<${#task_argv[@]}; task_index++)); do
        case "${task_argv[task_index]}" in
            --host)
                if [ "$task_host_bound" = 0 ]; then
                    task_argv[task_index+1]="${HOST}"
                    task_host_bound=1
                fi ;;
            --port)
                if [ "$task_port_bound" = 0 ]; then
                    task_argv[task_index+1]="${PORT}"
                    task_port_bound=1
                fi ;;
        esac
    done
    # Interpreter entry: overridable + capability-checked + fail-closed.  Resolved here rather
    # than at source time so the official parser contract can still source this launcher
    # unchanged; an explicit PYTHON override must prove the same capability instead of matching
    # a path, and the launcher never falls back to the system python.
    local python_entry="${PYTHON:-/home/py312/bin/python}"
    if [ ! -x "${python_entry}" ]; then
        echo "ERROR: python interpreter ${python_entry} not found or not executable (fail-closed; never falls back to the system python)" >&2
        return 1
    fi
    if ! "${python_entry}" -c 'import torch, torch_sdaa; assert torch.sdaa.is_available()' >/dev/null 2>&1; then
        echo "ERROR: ${python_entry} lacks the required capability (import torch + torch_sdaa + torch.sdaa.is_available()); source /opt/tecoai/setvars.sh, and for py3.11 install the matching tecoops/torch_sdaa assets (docs/bootstrap-py311.md)" >&2
        return 1
    fi
    exec "${python_entry}" -m vllm.entrypoints.cli.main "${task_argv[@]}"

}

vllm serve /gpfs/model/google/gemma-4-12B-it \
    --served-model-name gemma-4-12B-it \
    --tensor-parallel-size "${GEMMA_TP_SIZE:-2}" \
    --gpu-memory-utilization "${UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}' \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8003 \
    "$@"
