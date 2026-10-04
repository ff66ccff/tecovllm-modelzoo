#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${HY_MT2_MODEL:-${MODEL_ROOT}/Tencent-Hunyuan/Hy-MT2-1.8B}"
HY_MT2_HOST="${HY_MT2_HOST:-0.0.0.0}"
HY_MT2_PORT="${HY_MT2_PORT:-8001}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"

export VLLM_DISABLE_COMPILE_CACHE=1

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
                    serve_args[$((i + 1))]="$HY_MT2_HOST"
                    host_seen=1
                fi
                ;;
            --port)
                if (( port_seen == 0 )); then
                    if (( i + 1 >= ${#serve_args[@]} )); then
                        printf 'FATAL: local vllm shim received no port value\n' >&2
                        return 64
                    fi
                    serve_args[$((i + 1))]="$HY_MT2_PORT"
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

vllm serve /gpfs/model/Tencent-Hunyuan/Hy-MT2-1.8B \
    --served-model-name Hy-MT2-1.8B \
    --tensor-parallel-size "${HY_MT2_TP_SIZE:-1}" \
    --gpu-memory-utilization "${HY_MT2_UTIL:-0.90}" \
    --max-model-len "${HY_MT2_MAX_MODEL_LEN:-8192}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8001 \
    "$@"
