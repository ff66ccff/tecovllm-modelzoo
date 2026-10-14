#!/usr/bin/env bash
# InternVL3_5-8B 交付自检：run.sh 的模型路径参数解析（零卡、无设备、不复制权重）。
#
# 背景：run.sh 里的 `vllm()` shim 会在 exec 前用 $MODEL_PATH 改写 `--model`，因此
# `vllm serve <字面量路径>` 的旧写法是**死文本**（latent/误导性硬编码，不是功能缺陷）。
# 本测试用「把解释器换成打印 argv 的桩」的方式，**不修改 run.sh 任何逻辑**，直接观测
# shim exec 出来的最终 argv，从而证明 MODEL_ROOT / INTERNVL_MODEL 真正生效。
#
# 用法：  bash tests/verify_runsh_model_root.sh
# 依赖：  仅 bash + coreutils；不 import torch/torch_sdaa，不占用加速卡。
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SH="${RUN_SH:-$HERE/../run.sh}"
FAIL=0
ok()  { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; FAIL=1; }

[ -f "$RUN_SH" ] || { echo "FATAL: run.sh not found: $RUN_SH"; exit 2; }

# 打印 argv 的桩解释器：run.sh 的能力校验只看退出码，因此桩可以通过；随后 exec 被替换成打印。
STUB=$(mktemp -d)/printf-argv
cat > "$STUB" <<'SH'
#!/bin/sh
for a in "$@"; do printf '%s\n' "$a"; done
SH
chmod +x "$STUB"

# 从桩输出里取 `--model` 的取值（argv 中 --model 的下一行）。
model_arg_of() {
  "$@" 2>/dev/null | awk 'prev=="--model"{print; exit} {prev=$0}'
}

echo "===== verify_runsh_model_root.sh ====="
echo "RUN_SH=$RUN_SH"

# ---- 1) 默认：MODEL_ROOT 未设 -> /gpfs/model/OpenGVLab/InternVL3_5-8B ----
got=$(env -u MODEL_ROOT -u INTERNVL_MODEL PYTHON="$STUB" bash "$RUN_SH" 2>/dev/null | awk 'prev=="--model"{print; exit} {prev=$0}')
echo "  默认        -> $got"
[ "$got" = "/gpfs/model/OpenGVLab/InternVL3_5-8B" ] \
  && ok "默认路径 = \$MODEL_ROOT 的默认值 + /OpenGVLab/InternVL3_5-8B" \
  || bad "默认路径异常: $got"

# ---- 2) MODEL_ROOT 覆盖：用 symlink 别名（不复制权重） ----
REAL="${MODEL_ROOT_REAL:-/gpfs/model}/OpenGVLab/InternVL3_5-8B"
if [ -d "$REAL" ]; then
  ALIAS_ROOT=$(mktemp -d)
  mkdir -p "$ALIAS_ROOT/OpenGVLab"
  ln -sfn "$REAL" "$ALIAS_ROOT/OpenGVLab/InternVL3_5-8B"
  got=$(MODEL_ROOT="$ALIAS_ROOT" PYTHON="$STUB" bash "$RUN_SH" 2>/dev/null | awk 'prev=="--model"{print; exit} {prev=$0}')
  echo "  MODEL_ROOT  -> $got"
  [ "$got" = "$ALIAS_ROOT/OpenGVLab/InternVL3_5-8B" ] \
    && ok "MODEL_ROOT 覆盖被 shim 传入的 --model（覆盖生效）" \
    || bad "MODEL_ROOT 未被 honor: $got"
  resolved=$(readlink -f "$got" 2>/dev/null || true)
  [ "$resolved" = "$(readlink -f "$REAL")" ] \
    && ok "别名 readlink -f 回指真实只读权重（未复制权重）: $resolved" \
    || bad "别名未回指真实权重: $resolved"
  rm -rf "$ALIAS_ROOT"
  # ---- 3) INTERNVL_MODEL 覆盖 ----
  got=$(INTERNVL_MODEL="$REAL" PYTHON="$STUB" bash "$RUN_SH" 2>/dev/null | awk 'prev=="--model"{print; exit} {prev=$0}')
  echo "  INTERNVL_MODEL -> $got"
  [ "$got" = "$REAL" ] && ok "INTERNVL_MODEL 优先级高于 MODEL_ROOT" || bad "INTERNVL_MODEL 未生效: $got"
else
  echo "  [skip] 真实权重目录不存在，跳过覆盖用例: $REAL"
fi

# ---- 4) shim 之外的路径型 serve 参数审计：必须一个都没有 ----
bad_flags=""
for f in tokenizer tokenizer-mode mm_processor mm-processor mm-processor-kwargs revision code-revision \
         download-dir model-loader-extra-config lora-modules speculative-model chat-template config; do
  grep -qE -- "^[[:space:]]*--$f([[:space:]]|$|=)" "$RUN_SH" && bad_flags="$bad_flags --$f"
done
if [ -n "$bad_flags" ]; then
  bad "run.sh 出现 shim 改写范围之外的路径型参数:$bad_flags（需一并纳入变量化）"
else
  ok "run.sh 无 shim 范围之外的路径型 serve 参数（--model 是唯一路径参数）"
fi

# ---- 5) run.sh 中残留的 /gpfs 字面量必须包含解析器可见的那一处（model_path 默认值） ----
if grep -q 'MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"' "$RUN_SH"; then
  ok "run.sh 保留了 MODEL_ROOT 的可覆盖默认值 /gpfs/model"
else
  bad "run.sh 缺少 MODEL_ROOT 默认值"
fi

# ---- 6) 官方 CI 解析器契约（tools/ci_pipline/run_ci.py::parse_run_sh）----
# 剥掉 `#` 注释行后，第一个 `vllm serve <X>` 的 X 必须是一个**真实存在**的路径字面量
# （CI 的 speed.sh 会把它当 --tokenizer-path；若抓到 "$MODEL_PATH" 则 os.path.exists 为假）。
CI_PY="${CI_RUN_CI_PY:-}"
if [ -z "$CI_PY" ]; then
  for c in \
      "$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)/tools/ci_pipline/run_ci.py" \
      /root/tmpwork/internvl-readme-refresh/tools/ci_pipline/run_ci.py \
      /root/tmpwork/final-drift-sync/tools/ci_pipline/run_ci.py \
      /root/tmpwork/llmz/tools/ci_pipline/run_ci.py ; do
    [ -f "$c" ] && { CI_PY="$c"; break; }
  done
fi
echo "  CI parser  -> ${CI_PY:-<not found: replica used>}"
/home/py312/bin/python - "$RUN_SH" "$CI_PY" <<'PY'
import importlib.util, os, pathlib, re, sys

run_sh, ci_py = sys.argv[1], (sys.argv[2] or None)
parse = None
origin = None
if ci_py:
    try:
        # run_ci.py imports its siblings (result_analyse) -> put the CI dir on sys.path first.
        sys.path.insert(0, os.path.dirname(os.path.abspath(ci_py)))
        spec = importlib.util.spec_from_file_location("official_run_ci", ci_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parse = mod.parse_run_sh
        origin = "OFFICIAL parse_run_sh from %s" % ci_py
    except Exception as exc:                      # never silently claim the official parser ran
        print("[ci] WARN: 官方模块导入失败(%s: %s) -> 退回逐字复刻实现" % (type(exc).__name__, exc))
if parse is None:
    def parse(path):                              # verbatim replica of parse_run_sh
        lines = pathlib.Path(path).read_text().split("\n")
        active = "\n".join(l for l in lines if not l.strip().startswith("#"))
        mn = re.search(r"--served-model-name\s+(\S+)", active)
        mp = re.search(r"vllm\s+serve\s+(\S+)", active)
        return {"model_path": mp.group(1) if mp else None,
                "model_name": mn.group(1) if mn else None,
                "has_no_prefix_caching": "--no-enable-prefix-caching" in active,
                "port": int((re.search(r"--port\s+(\d+)", active) or [None, "8000"])[1])}
    origin = "verbatim replica of parse_run_sh (official module unavailable)"

info = parse(run_sh)
mp = info.get("model_path")
active = "\n".join(l for l in pathlib.Path(run_sh).read_text().split("\n") if not l.strip().startswith("#"))
print("[ci] parser     =", origin)
print("[ci] keys       =", sorted(info))
print("[ci] model_path =", mp)
print("[ci] model_name =", info.get("model_name"), "| port =", info.get("port"),
      "| host =", info.get("host"),
      "| no-enable-prefix-caching =", "--no-enable-prefix-caching" in active)
assert mp and "$" not in mp, "CI 解析器抓到的 model_path 是变量/带引号: %r" % mp
assert os.path.isabs(mp), "model_path 不是绝对路径: %r" % mp
assert os.path.exists(mp), "CI 解析器抓到的 model_path 不存在: %r" % mp
assert info.get("model_name") == "InternVL3_5-8B", info.get("model_name")
assert "--no-enable-prefix-caching" in active, "run.sh 缺少 --no-enable-prefix-caching（CI 硬要求）"
print("[ci] OK: 官方解析器契约满足（裸字面量、真实存在、served-name/port 与执行行一致）")
PY
[ "${PIPESTATUS[0]:-1}" = "0" ] && ok "官方 CI parse_run_sh 契约满足" || bad "官方 CI parse_run_sh 契约不满足"

echo "===== verdict: $([ "$FAIL" = "0" ] && echo PASS || echo FAIL) ====="
exit "$FAIL"
