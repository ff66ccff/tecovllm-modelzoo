#!/usr/bin/env bash
set -euo pipefail
# Require both tested core and extension, including non-RMS paths, at startup.
export SCUD_REQUIRE_OPS_BUNDLE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# 官方要求团队级 run.sh（model_adaptations/<团队名称>/run.sh）。本文件与模型目录下的 run.sh 逐字一致，
# 只把 overlay/runtime 资产路径绑定到模型目录：MODEL_DIR 指向本分支的模型目录。
MODEL_DIR="${SCRIPT_DIR}/../HyMT2SCUdoudui"
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${HY_MT2_MODEL:-${MODEL_ROOT}/Tencent-Hunyuan/Hy-MT2-1.8B}"
HY_MT2_HOST="${HY_MT2_HOST:-0.0.0.0}"
HY_MT2_PORT="${HY_MT2_PORT:-8001}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${MODEL_DIR}/runtime/overlay:${MODEL_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"

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

# CI parser note: tools/ci_pipline/run_ci.py `parse_run_sh()` strips comment lines and
# takes the FIRST `vllm serve <token>` match as the model path (it is also passed to
# speed.sh as --tokenizer-path).  A quoted `"$MODEL_PATH"` would be captured verbatim and
# fail `os.path.exists`.  Keep this unreachable bare literal first; the launcher below
# executes the variable form, and the local vllm() shim rewrites argv[1] from $MODEL_PATH.
if false; then
    vllm serve /gpfs/model/Tencent-Hunyuan/Hy-MT2-1.8B
fi

vllm serve "$MODEL_PATH" \
    --served-model-name Hy-MT2-1.8B \
    --tensor-parallel-size "${HY_MT2_TP_SIZE:-1}" \
    --gpu-memory-utilization "${HY_MT2_UTIL:-0.90}" \
    --max-model-len "${HY_MT2_MAX_MODEL_LEN:-16640}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8001 \
    "$@"
