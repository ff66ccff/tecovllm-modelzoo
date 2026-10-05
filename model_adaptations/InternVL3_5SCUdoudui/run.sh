#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
source /opt/tecoai/setvars.sh
set -u
export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0
PYTHON=/home/py312/bin/python
if [ "$(readlink -f "$PYTHON")" != /usr/local/python/bin/python3.12 ]; then
    echo 'FATAL: expected vendor /home/py312/bin/python' >&2
    exit 1
fi
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${INTERNVL_MODEL:-${MODEL_ROOT}/OpenGVLab/InternVL3_5-8B}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SDAA_VISIBLE_DEVICES="${SDAA_VISIBLE_DEVICES:-0,1}"
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"

# Normalize the public CLI command once at launch; keep the validated API-server entry.
vllm() {
    local args=("$@")
    local host_seen=0 port_seen=0 i
    if [[ "${args[0]:-}" != serve || ${#args[@]} -lt 2 ]]; then
        echo 'FATAL: expected serve command and model argument' >&2
        return 1
    fi
    args[0]=--model
    args[1]="$MODEL_PATH"
    for ((i=2; i<${#args[@]}; i++)); do
        case "${args[i]}" in
            --host)
                if [[ $host_seen == 0 ]]; then
                    args[i+1]="${INTERNVL_HOST:-0.0.0.0}"
                    host_seen=1
                fi
                ;;
            --port)
                if [[ $port_seen == 0 ]]; then
                    args[i+1]="${INTERNVL_PORT:-8002}"
                    port_seen=1
                fi
                ;;
        esac
    done
    exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${args[@]}"
}

vllm serve /gpfs/model/OpenGVLab/InternVL3_5-8B \
    --served-model-name InternVL3_5-8B \
    --dtype float16 \
    --tensor-parallel-size "${INTERNVL_TP_SIZE:-2}" \
    --gpu-memory-utilization "${INTERNVL_UTIL:-0.85}" \
    --max-model-len "${INTERNVL_MAX_MODEL_LEN:-4352}" \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --host 0.0.0.0 \
    --port 8002 \
    "$@"
