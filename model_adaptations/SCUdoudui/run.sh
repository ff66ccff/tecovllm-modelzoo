#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# 官方要求团队级 run.sh（model_adaptations/<团队名称>/run.sh）。本文件与模型目录下的 run.sh 逐字一致，
# 只把 overlay/runtime 资产路径绑定到模型目录：MODEL_DIR 指向本分支的模型目录。
MODEL_DIR="${SCRIPT_DIR}/../Gemma4SCUdoudui"
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
GEMMA_PATH="${GEMMA_PATH:-${MODEL_ROOT}/google/gemma-4-12B-it}"
# 环境变量（可覆盖）：MODEL_ROOT / GEMMA_PATH 决定加载哪份预置权重；
# GEMMA_HOST / GEMMA_PORT / GEMMA_MAX_MODEL_LEN / GEMMA_MAX_BATCHED_TOKENS /
# GEMMA_TP_SIZE / GEMMA_UTIL / PYTHON 决定服务参数。真实生效的模型路径见文件末尾的
# `vllm serve "$GEMMA_PATH"`：vllm() shim 会在 exec 前把 task_argv[1] 改写成 $GEMMA_PATH。
PORT="${GEMMA_PORT:-8003}"
HOST="${GEMMA_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-16640}"
MAX_BATCHED="${GEMMA_MAX_BATCHED_TOKENS:-512}"
# c52 (2026-10-23): CI 硬上限 14400s；现状 KV=0.5GiB -> 只跑 2 并发 -> precision 4h50m 超时。
# 显式收敛 max_num_seqs=8 使 profiling 激活峰值按 8 序列计：KV=2.06GiB、实测 8 并发、TPOT 不回归。
# 其余参数（0.92 / 512 / 16640 / TP2）保持 CI 已验证值不变；评测命令（prec.sh/speed.sh）未改动。
MAX_SEQS="${GEMMA_MAX_NUM_SEQS:-8}"
# c53 (2026-10-23): 16K 速度阶段（T4_16K_b1，16384-token prompt）在 util 0.92 + mns 8 下
# 于团队自己的 prefill math attention（runtime/custom_ops/prefill_attention/op.py:217 的
# softmax(scores.to(float32)) 需要 ~232 MiB 瞬时激活）处 SDAA OOM：14.01 GiB 已分配 / 403.98 MiB 空闲。
# 实测评测口径：util 0.90 ⇒ KV 1.76 GiB、16,640-token 请求并发 3.56x，T4 通过（648 s/10 req）；
# 精度阶段 8 并发 × 1617 token 只需 ~0.39 GiB KV ⇒ 并发仍为 8，不影响 precision 墙钟。
UTIL="${GEMMA_UTIL:-0.90}"

# Bind the reviewed implementation before vLLM constructs its attention
# classes. No vendor package is modified and no model weights are copied.
export GEMMA4_OVERLAY=1
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONPATH="${MODEL_DIR}/overlay:${MODEL_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# 实现选择在进程初始化期一次完成：vllm() shim 会把 task_argv[1] 改写为 $GEMMA_PATH，
# 并按 $HOST/$PORT 覆盖 --host/--port；其余 serve 参数逐字不动（CI 已固定，勿改参）。
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

# 官方 CI 解析契约（**不执行**）：tools/ci_pipline/run_ci.py:50 用
#   re.search(r"vllm\s+serve\s+(\S+)", run.sh)
# 从本文件抓取"模型路径"，并在 :196 作为 speed.sh 的 --tokenizer-path 传给 evalscope。
# 该位置若写变量，解析器会抓到字符串 '"$GEMMA_PATH"' 而使官方性能步骤失败（设备无关实测，
# 见交付证据 launcher-argv/）。因此这里保留一个字面默认路径，且用 if false 明确标注其不可达；
# 真正启动服务的是文件末尾的 vllm serve "$GEMMA_PATH"。
if false; then
    vllm serve /gpfs/model/google/gemma-4-12B-it
fi

vllm serve "$GEMMA_PATH" \
    --served-model-name gemma-4-12B-it \
    --tensor-parallel-size "${GEMMA_TP_SIZE:-2}" \
    --gpu-memory-utilization "${UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    --max-num-seqs "${MAX_SEQS}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}' \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8003 \
    "$@"
