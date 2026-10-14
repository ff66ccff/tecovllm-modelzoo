#!/usr/bin/env bash
# InternVL3_5-8B 交付自检：overlay 的 fail-closed 必须是**进程级**的（零卡、无设备）。
#
# 背景（跨线真缺陷，gemma 线实测）：`sitecustomize.py` 由 CPython 的
# `site.execsitecustomize()` 执行，其处理器是 `except Exception`。因此**普通 `raise XxxError`
# 会被吞掉**（只打印一行 "Error in sitecustomize"），解释器照常跑到 main ⇒ "overlay 没绑上也能启动"
# 的静默降级。只有 BaseException 的非 Exception 子类（SystemExit / os._exit）能逃出。
#
# 本脚本用**设备无关**方式证明三件事：
#   [1] 三格对照：RuntimeError 被吞（main 可达、rc=0） vs SystemExit(1) vs os._exit(1)（rc≠0、main 不可达）
#   [2] 真实 overlay + 会失败的 stub custom_ops ⇒ FATAL + rc≠0 + main 不可达（进程级 fail-closed）
#   [3] 静态审计：无裸 raise、含 SystemExit 与非 Exception 兜底；根副本与交付副本逐字节相同
#
# 用法：  bash tests/verify_overlay_fail_closed.sh
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ADAPT=$(cd -- "$HERE/.." && pwd)
REAL_OVERLAY="${REAL_OVERLAY:-$ADAPT/runtime/overlay/sitecustomize.py}"
ROOT_OVERLAY="${ROOT_OVERLAY:-}"
[ -z "$ROOT_OVERLAY" ] && ROOT_OVERLAY="$(cd -- "$ADAPT/../.." && pwd)/custom_ops/overlay/sitecustomize.py"
PY="${PYTHON:-/home/py312/bin/python}"
TMP=$(mktemp -d)
FAIL=0
ok()  { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; FAIL=1; }
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "===== verify_overlay_fail_closed.sh ====="
echo "REAL_OVERLAY=$REAL_OVERLAY"
[ -f "$REAL_OVERLAY" ] || { echo "FATAL: overlay not found"; exit 2; }

probe() {   # probe <extra-pythonpath> -> prints rc + whether main was reached
  local out rc
  out=$(PYTHONPATH="$1" "$PY" -c 'print("MAIN-REACHED")' 2>"$TMP/err.txt"); rc=$?
  printf 'rc=%s main_reached=%s\n' "$rc" "$(printf '%s' "$out" | grep -c 'MAIN-REACHED')"
}

mk() { mkdir -p "$TMP/$1"; printf '%s\n' "$2" > "$TMP/$1/sitecustomize.py"; }

echo
echo "--- [1] 三格对照（合成 sitecustomize，验证 site.py 的吞噬行为） ---"
mk plain 'raise RuntimeError("boom: overlay failed")'
mk sexit 'raise SystemExit(1)'
mk osexit 'import os; os._exit(1)'
r_plain=$(probe "$TMP/plain");  echo "  plain RuntimeError -> $r_plain"
r_sexit=$(probe "$TMP/sexit");  echo "  raise SystemExit(1)-> $r_sexit"
r_os=$(probe "$TMP/osexit");    echo "  os._exit(1)        -> $r_os"
case "$r_plain" in "rc=0 main_reached=1") ok "复现吞噬：普通 raise ⇒ rc=0 且子解释器到达 main（静默降级）";;
  *) bad "未能复现吞噬（预期 rc=0/main_reached=1）: $r_plain";; esac
case "$r_sexit" in "rc=0"*) bad "SystemExit 被吞了（异常）: $r_sexit";;
  *main_reached=0*) ok "SystemExit(1) ⇒ rc≠0 且 main 不可达（进程级）";;
  *) bad "SystemExit 行为异常: $r_sexit";; esac
case "$r_os" in "rc=0"*) bad "os._exit(1) 被吞了（异常）: $r_os";;
  *main_reached=0*) ok "os._exit(1) ⇒ rc≠0 且 main 不可达（进程级）";;
  *) bad "os._exit 行为异常: $r_os";; esac

echo
echo "--- [2] 真实 overlay + 会失败的 stub custom_ops（rms_norm 导入即抛错） ---"
mkdir -p "$TMP/real"; cp -f "$REAL_OVERLAY" "$TMP/real/sitecustomize.py"
mkdir -p "$TMP/stub/custom_ops/rms_norm"
: > "$TMP/stub/custom_ops/__init__.py"
: > "$TMP/stub/custom_ops/rms_norm/__init__.py"
printf '%s\n' 'raise RuntimeError("simulated: tecoops unavailable")' > "$TMP/stub/custom_ops/rms_norm/op.py"
r_real=$(probe "$TMP/real:$TMP/stub"); echo "  real overlay -> $r_real"
grep -q "FATAL: \[custom_ops\]" "$TMP/err.txt" && ok "失败路径打印了可读 FATAL 诊断" || bad "未打印 FATAL 诊断"
case "$r_real" in "rc=0"*) bad "真实 overlay 失败时 rc=0（静默降级）: $r_real";;
  *main_reached=0*) ok "真实 overlay 失败 ⇒ rc≠0 且 main 不可达（进程级 fail-closed）";;
  *) bad "真实 overlay 失败行为异常: $r_real";; esac

echo
echo "--- [3] 负对照：同一个 stub 换成\"无兜底\"的 sitecustomize ⇒ 必须复现静默降级 ---"
mkdir -p "$TMP/nocatch"
{
  printf '%s\n' 'import sys'
  printf '%s\n' 'from custom_ops.rms_norm.op import install_qwen3_rms_norm_overlay'
  printf '%s\n' 'install_qwen3_rms_norm_overlay()'
} > "$TMP/nocatch/sitecustomize.py"
[ -s "$TMP/nocatch/sitecustomize.py" ] || bad "负对照的 sitecustomize 未写成功（否则会假通过）"
r_nocatch=$(probe "$TMP/nocatch:$TMP/stub"); echo "  no-catchall -> $r_nocatch"
case "$r_nocatch" in "rc=0 main_reached=1") ok "负对照成立：无兜底时普通 raise 被吞（说明本测试能识别该类缺陷）";;
  *) bad "负对照未复现吞噬（预期 rc=0/main_reached=1）: $r_nocatch";; esac

echo
echo "--- [4] 静态审计 ---"
if grep -nE '^[[:space:]]*raise [A-Za-z_]*(Error|Exception)' "$REAL_OVERLAY" >/dev/null; then
  bad "overlay 仍存在裸 raise（会被 site.py 吞）"; grep -nE '^[[:space:]]*raise ' "$REAL_OVERLAY" | sed 's/^/      /'
else
  ok "overlay 无裸 raise（失败路径全部 SystemExit/os._exit）"
fi
grep -qE 'raise SystemExit\(1\)|sys\.exit\(1\)|os\._exit\(1\)' "$REAL_OVERLAY" \
  && ok "overlay 含进程级退出（SystemExit/sys.exit/os._exit）" || bad "overlay 未见进程级退出"
grep -q 'except BaseException' "$REAL_OVERLAY" \
  && ok "overlay 含非 Exception 兜底（把意外的普通异常转成 SystemExit）" \
  || bad "overlay 缺少兜底：意外普通异常仍会被 site.py 吞掉"
if [ -f "$ROOT_OVERLAY" ]; then
  if cmp -s "$ROOT_OVERLAY" "$REAL_OVERLAY"; then
    ok "根副本与交付副本逐字节相同（$(sha256sum "$REAL_OVERLAY" | cut -c1-16)…）"
  else
    bad "根副本与交付副本不一致：$ROOT_OVERLAY vs $REAL_OVERLAY"
  fi
else
  echo "  [skip] 未找到根副本: $ROOT_OVERLAY"
fi

echo
echo "===== verdict: $([ "$FAIL" = "0" ] && echo PASS || echo FAIL) ====="
exit "$FAIL"
