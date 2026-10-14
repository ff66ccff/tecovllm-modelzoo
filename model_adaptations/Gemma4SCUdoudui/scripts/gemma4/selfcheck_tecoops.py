#!/usr/bin/env python3
"""Gemma-4-12B-it 交付自检：确认 ``import tecoops`` 解析到的是**本队交付库**，
而不是环境里任意一个已安装的旧 wheel。

**初始化期一次、无热路径分支**（只在启动脚本里跑一次，不参与模型前向）。

为什么需要它（本线实测，不是推断）
----------------------------------
交付包**不含** ``tecoops``：``run.sh`` 只把 ``overlay/`` 与 ``runtime/`` 前置到
``PYTHONPATH``。若启动前没有把本队 ops 的安装目录前置，``import tecoops`` 会落到
环境里任意一个已安装的 wheel —— 本机实测落到 ``site-packages/tecoops``
（``libteco_ops.so`` sha256 ``388dd710a8357ee9…``，278 176 B），**不报错**，
overlay 三个绑定点照常打印并进入 main（静默降级）。

那个旧 wheel 的能力面（实测）：8 个导出，``rms_norm`` / ``weight_stream_gemv`` /
``flash_attn_varlen_func`` 为 ``True``，而 ``rms_norm_compat`` /
``ms_deform_attn_forward`` / ``ms_deform_attn_backward`` / ``rotary_embedding``
为 ``False``，且 ``teco_slave_flash_attention_half_d256`` / ``_d512`` 两个
**交付 flash_attention 特化符号在库里不存在**（交付库分别出现 7 / 1 次）。

因此**不能**用 ``len(tecoops.__all__) > 0`` 或单个 python 名当判据：旧 wheel 也过，
而 flash attention 会**静默绑到旧库的通用 kernel**。本脚本用 **build-root 无关的能力
探针**（python 名 + 交付 flash_attention 特化符号）判定。

退出码：``0`` = 通过；``1`` = 库身份不符（fail-closed）。
"""
import hashlib
import sys

# 当前交付默认档实际依赖的 python 名（缺任一即不可用）。
PY_NAMES = ["rms_norm", "weight_stream_gemv", "flash_attn_varlen_func"]
# 可选：交付包同时提供，但当前默认档不依赖；缺失只提示，不判 FAIL。
PY_NAMES_OPTIONAL = ["rms_norm_compat", "reshape_and_cache"]
# 交付 flash_attention 的 D256/D512 特化符号：旧 wheel 没有，是"跑的是哪个库"的硬判据。
LIB_SYMBOLS = ["teco_slave_flash_attention_half_d256", "teco_slave_flash_attention_half_d512"]


def main() -> int:
    import tecoops

    print("tecoops.__file__ =", tecoops.__file__)

    libs = []
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.split()[-1]
                if path.endswith("libteco_ops.so"):
                    libs.append(path)
    except OSError:  # pragma: no cover
        pass
    libs = sorted(set(libs))
    if not libs:
        print("FAIL: no libteco_ops.so mapped into this process", file=sys.stderr)
        return 1
    blob = b""
    for path in libs:
        with open(path, "rb") as fh:
            data = fh.read()
        blob += data
        print("libteco_ops.so   =", path)
        print("libteco_ops.so   sha256 =", hashlib.sha256(data).hexdigest())

    ok = True
    for name in PY_NAMES:
        present = hasattr(tecoops, name)
        print("python export    %-26s %s" % (name, present))
        ok = ok and present
    for name in PY_NAMES_OPTIONAL:
        print("python export    %-26s %s (optional)" % (name, hasattr(tecoops, name)))
    for symbol in LIB_SYMBOLS:
        present = symbol.encode() in blob
        print("lib symbol       %-40s %s" % (symbol, present))
        ok = ok and present

    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
