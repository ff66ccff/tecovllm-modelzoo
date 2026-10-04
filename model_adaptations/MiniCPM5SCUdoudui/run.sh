#!/bin/bash
# SCUdoudui MiniCPM5-1B official launch script
set -euo pipefail
# Launch contract for tools/ci_pipline/run_ci.py

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${MINICPM5_MODEL:-${MODEL_ROOT}/OpenBMB/MiniCPM5-1B}"
MINICPM_HOST="${MINICPM_HOST:-0.0.0.0}"
MINICPM_PORT="${MINICPM_PORT:-8000}"
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

vllm() {
    if [[ "${1:-}" != "serve" ]]; then
        printf 'FATAL: local vllm shim only accepts the serve entrypoint\n' >&2
        return 64
    fi
    shift
    local -a serve_args=("$@")
    local i host_seen=0 port_seen=0
    if (( ${#serve_args[@]} == 0 )); then
        printf 'FATAL: local vllm shim received no model path\n' >&2
        return 64
    fi

    serve_args[0]="$MODEL_PATH"
    for ((i = 0; i < ${#serve_args[@]}; i++)); do
        case "${serve_args[i]}" in
            --host)
                if (( host_seen == 0 )); then
                    if (( i + 1 >= ${#serve_args[@]} )); then
                        printf 'FATAL: local vllm shim received no host value\n' >&2
                        return 64
                    fi
                    serve_args[$((i + 1))]="$MINICPM_HOST"
                    host_seen=1
                fi
                ;;
            --port)
                if (( port_seen == 0 )); then
                    if (( i + 1 >= ${#serve_args[@]} )); then
                        printf 'FATAL: local vllm shim received no port value\n' >&2
                        return 64
                    fi
                    serve_args[$((i + 1))]="$MINICPM_PORT"
                    port_seen=1
                fi
                ;;
        esac
    done
    if (( host_seen == 0 || port_seen == 0 )); then
        printf 'FATAL: local vllm shim requires host and port arguments\n' >&2
        return 64
    fi

    exec /home/py312/bin/python -m vllm.entrypoints.cli.main serve "${serve_args[@]}"
}

vllm serve /gpfs/model/OpenBMB/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --tensor-parallel-size 1 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype float16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --max-model-len "${MINICPM_MAX_MODEL_LEN:-32768}" \
    "$@"
