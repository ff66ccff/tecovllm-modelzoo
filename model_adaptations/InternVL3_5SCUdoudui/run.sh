#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
source /opt/tecoai/setvars.sh
set -u
export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0

# 解释器选择：可用 PYTHON=<解释器> 覆盖（官方 py3.11 环境用该入口），默认值仍是
# 厂商 /home/py312/bin/python —— AGENTS.md 硬约束 1 的默认口径不变。
PYTHON="${PYTHON:-/home/py312/bin/python}"

if [ ! -x "${PYTHON}" ]; then
    echo "FATAL: 解释器不存在或不可执行: ${PYTHON}（可用 PYTHON=<解释器> 覆盖）" >&2
    exit 1
fi

# 能力校验（不是路径等式）：必须能 import torch 与 torch_sdaa，且 torch.sdaa
# 报告设备可用。官方 py3.11 环境通过该校验；不具备 torch_sdaa 的解释器被拒绝。
# fail-closed：解释器缺失或能力不足一律非零退出，禁止静默降级到系统 python。
INTERP_PROBE="$("${PYTHON}" - <<'PY' 2>&1
import sys

ok = True


def check(good, msg):
    global ok
    ok = ok and bool(good)
    print(("OK   " if good else "FAIL ") + msg)


try:
    import torch
    check(True, f"import torch -> {torch.__version__}")
    import torch_sdaa  # noqa: F401  (register the SDAA backend)
    check(True, "import torch_sdaa -> OK")
    avail = bool(torch.sdaa.is_available())
    count = int(torch.sdaa.device_count())
    check(avail and count >= 1, f"torch.sdaa.is_available()={avail} device_count={count}")
except Exception as exc:  # diagnostic path only
    check(False, f"{type(exc).__name__}: {exc}")

sys.exit(0 if ok else 1)
PY
)" && INTERP_OK=1 || INTERP_OK=0
printf '%s\n' "${INTERP_PROBE}" | sed 's/^/[interpreter] /'
if [ "${INTERP_OK}" -ne 1 ]; then
    echo "FATAL: 解释器能力校验失败: ${PYTHON}" >&2
    echo "  要求：可 import torch 与 torch_sdaa，且 torch.sdaa.is_available() 为真。" >&2
    echo "  请先 source /opt/tecoai/setvars.sh；官方 py3.11 环境用 PYTHON=<py3.11 解释器>。" >&2
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
