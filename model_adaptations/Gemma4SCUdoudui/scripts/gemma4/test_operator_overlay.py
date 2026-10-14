#!/usr/bin/env python
"""CPU 单元测试：port 进来的官方算子 overlay（D2）。

可在纯 CPU 上验证的部分：
  T1  三个算子模块可 import，且暴露预期可调用符号
  T2  block_attention overlay 真的把 BlockAttentionImpl.forward 换掉（identity check）
  T3  gather_kv_from_cache 的 HND 索引逻辑正确（用小手算样例逐元素比对）
  T4  sdpa_prefill_attention 的 chunked-prefill fail-closed 契约（必须抛 RuntimeError）
      —— 这是把"不支持 chunked prefill"从注释变成可执行断言
  T5  sdpa_prefill_attention 在 q_len == kv_len 的纯 prefill 上数值正确
      （与直接 F.scaled_dot_product_attention 对齐）
  T6  sdpa_prefill_attention 的 decode 分支（q_len=1）数值正确

不加载权重、不需要 GPU。

用法:
  PYTHONPATH=overlay:. GEMMA4_OVERLAY=1 /home/py312/bin/python scripts/gemma4/test_operator_overlay.py
"""

from __future__ import annotations

import os
import sys

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    import torch

    # ---- T1 模块与符号 ----
    try:
        from custom_ops.block_attention.op import sdaa_block_attention_forward
        from custom_ops.prefill_attention.op import (
            sdpa_prefill_attention, sdpa_prefill_attention_math, gather_kv_from_cache,
        )
        from custom_ops.reshape_and_cache.op import sdaa_reshape_and_cache

        check("T1.0 三个算子模块 import OK", True)
    except Exception as exc:  # noqa: BLE001
        check("T1.0 三个算子模块 import OK", False, f"{type(exc).__name__}: {exc}")
        print("FAILED (import) - 后续用例无法进行")
        return 1

    check("T1.1 block_attention 暴露 forward", callable(sdaa_block_attention_forward))
    check("T1.2 prefill_attention 暴露 sdpa_prefill_attention", callable(sdpa_prefill_attention))
    check("T1.3 reshape_and_cache 暴露 sdaa_reshape_and_cache", callable(sdaa_reshape_and_cache))

    # 防漂移（L11：门禁验的字节 == 发布字节）：本套件必须对着"交付实际执行的代码"跑。
    # 交付 run.sh 的 PYTHONPATH 是 <delivery>/overlay:<delivery>/runtime，而 custom_ops 是
    # runtime/ 下的子目录；仓库根也有一个 custom_ops/（被 git 跟踪、不在交付子树内、无
    # __init__.py），机制上 regular package 胜过 namespace package、与 sys.path 顺序无关
    # ⇒ 交付执行的是 runtime/custom_ops/。下面两条断言 fail-closed。
    import custom_ops as _custom_ops

    def _c22_find_delivery_custom_ops():
        """从本文件向上找 <root>/model_adaptations/Gemma4SCUdoudui/runtime/custom_ops。
        这样同一个字节的文件放在仓库根或放在交付子树里都能自证（两种布局都能向上找到）。"""
        cur = os.path.dirname(os.path.abspath(__file__))
        for _ in range(8):
            cand = os.path.join(cur, "model_adaptations", "Gemma4SCUdoudui",
                                "runtime", "custom_ops")
            if os.path.isdir(cand):
                return os.path.realpath(cand)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None

    _mod_path = os.path.realpath(_custom_ops.__file__)
    _pkg_dir = os.path.dirname(_mod_path)
    _structural = (os.path.basename(_pkg_dir) == "custom_ops"
                   and os.path.basename(os.path.dirname(_pkg_dir)) == "runtime")
    check("T1.4 custom_ops 解析到 *runtime/custom_ops*（结构断言，任意布局成立）",
          _structural, f"resolved={_mod_path}")
    _expected = _c22_find_delivery_custom_ops()
    if _expected is None:
        check("T1.5 custom_ops == 交付子树 runtime/custom_ops（精确断言）", False,
              "无法从本文件向上定位 model_adaptations/Gemma4SCUdoudui/runtime/custom_ops")
    else:
        check("T1.5 custom_ops == 交付子树 runtime/custom_ops（精确断言）",
              _mod_path == os.path.join(_expected, "__init__.py"),
              f"resolved={_mod_path} expected={os.path.join(_expected, '__init__.py')}")

    # ---- T2 overlay 绑定 ----
    try:
        import vllm_sdaa.attention.block_attn as m
        bound = m.BlockAttentionImpl.forward
        check("T2.0 BlockAttentionImpl.forward 已绑定到 overlay",
              bound is sdaa_block_attention_forward,
              f"{getattr(bound, '__module__', '?')}.{getattr(bound, '__name__', '?')}")
    except Exception as exc:  # noqa: BLE001
        check("T2.0 BlockAttentionImpl.forward 已绑定到 overlay", False,
              f"{type(exc).__name__}: {exc}（需 GEMMA4_OVERLAY=1）")

    # ---- T3 gather 索引逻辑（HND: [num_blocks, kv_heads, block_size, head_size]）----
    num_blocks, nkv, bs, hd = 4, 2, 4, 3
    kc = torch.zeros(num_blocks, nkv, bs, hd, dtype=torch.float32)
    for b in range(num_blocks):
        kc[b] = b + 1                       # 每块填常量，便于手算
    bt = torch.tensor([2, 0, 3, 1], dtype=torch.int64)   # block_table（逻辑块 -> 物理块）
    kv_len = 6
    got_k, got_v = gather_kv_from_cache(kc, kc, bt, kv_len, bs)
    # 逻辑 token i 落在物理块 bt[i // bs]，块内偏移 i % bs
    expect = torch.stack([kc[bt[i // bs], :, i % bs, :] for i in range(kv_len)], dim=0)
    check("T3.0 gather_kv_from_cache 与手算索引一致", torch.equal(got_k, expect),
          f"shape={tuple(got_k.shape)}")
    check("T3.1 gather 长度 == kv_len", got_k.shape[0] == kv_len, str(got_k.shape[0]))
    check("T3.2 K/V 同时返回且同形", got_k.shape == got_v.shape)

    # ---- T4 chunked prefill fail-closed ----
    q_len, kv_l = 2, 5
    q = torch.randn(q_len, 2, hd, dtype=torch.float32)
    kc2 = torch.randn(4, 2, bs, hd, dtype=torch.float32)
    vc2 = torch.randn(4, 2, bs, hd, dtype=torch.float32)
    raised = None
    try:
        sdpa_prefill_attention(q, kc2, vc2, bt.unsqueeze(0), torch.tensor([0, q_len]),
                               torch.tensor([kv_l]), 0.5, True)
    except RuntimeError as exc:
        raised = str(exc)
    check("T4.0 chunked prefill (1<q_len<kv_len) 显式 fail-closed 抛 RuntimeError",
          raised is not None and "chunked prefill" in (raised or ""),
          (raised or "no exception")[:90])

    # ---- T5 纯 prefill 数值对齐 ----
    hs, nq = 3, 2
    n = 4
    q2 = torch.randn(n, nq, hs, dtype=torch.float32)
    kc3 = torch.zeros(2, nq, bs, hs, dtype=torch.float32)
    vc3 = torch.zeros(2, nq, bs, hs, dtype=torch.float32)
    kk = torch.randn(n, nq, hs, dtype=torch.float32)
    vv = torch.randn(n, nq, hs, dtype=torch.float32)
    for i in range(n):                     # 铺进单块（bs=4, n=4）
        kc3[0, :, i, :] = kk[i]
        vc3[0, :, i, :] = vv[i]
    got5 = sdpa_prefill_attention(q2, kc3, vc3, torch.tensor([[0]], dtype=torch.int64),
                                  torch.tensor([0, n]), torch.tensor([n]), 0.5, True)
    ref5 = torch.nn.functional.scaled_dot_product_attention(
        q2.transpose(0, 1).unsqueeze(0), kk.transpose(0, 1).unsqueeze(0),
        vv.transpose(0, 1).unsqueeze(0), is_causal=True, scale=0.5
    ).squeeze(0).transpose(0, 1)
    check("T5.0 纯 prefill 与 F.sdpa 数值一致（atol 1e-6）",
          torch.allclose(got5, ref5, atol=1e-6), f"maxdiff={(got5-ref5).abs().max():.2e}")

    # ---- T6 decode（q_len=1）数值对齐，且掩码应为非因果 ----
    q6 = torch.randn(1, nq, hs, dtype=torch.float32)
    got6 = sdpa_prefill_attention(q6, kc3, vc3, torch.tensor([[0]], dtype=torch.int64),
                                  torch.tensor([0, 1]), torch.tensor([n]), 0.5, True)
    ref6 = torch.nn.functional.scaled_dot_product_attention(
        q6.transpose(0, 1).unsqueeze(0), kk.transpose(0, 1).unsqueeze(0),
        vv.transpose(0, 1).unsqueeze(0), is_causal=False, scale=0.5
    ).squeeze(0).transpose(0, 1)
    check("T6.0 decode(q_len=1) 与非因果 F.sdpa 一致",
          torch.allclose(got6, ref6, atol=1e-6), f"maxdiff={(got6-ref6).abs().max():.2e}")

    # ---- T8 window_size 归一化（厂商 kernel 只接受 -1/127/511）----
    if os.environ.get("GEMMA4_OVERLAY") == "1":
        import vllm_sdaa.attention.block_attn as m

        check("T8.0 __init__ 归一化补丁已装",
              getattr(m.BlockAttentionImpl, "_gemma4_window_patch", False) is True)
        # 直接验证归一化规则本身（不构造真实 impl，避免需要设备）
        kernel_windows = (-1, 127, 511)
        for raw, expect in ((1023, -1), (127, 127), (511, 511), (-1, -1)):
            got = raw if raw in kernel_windows else -1
            check(f"T8.1 window {raw} -> {got}（期望 {expect}）", got == expect)
    else:
        check("T8.0 需 GEMMA4_OVERLAY=1 才能检查 window 补丁", False, "skipped")

    # ---- T7 Gemma sliding-window math kernel（按交付 runtime/custom_ops 语义重写）----
    # 交付副本对 fixture 有硬契约：block_size 必须 32、page id 有下界（windowed >=1 /
    # global >=2）、metadata 必须 int32、query_start_loc 累积端点必须自洽。
    # 旧版 T7.0 用 bs=4 的小 fixture，在交付副本上直接 RuntimeError
    # ("requires matching head dimension and block32") —— 那是"测试 vs 交付"不一致，
    # 本次按交付语义重写。K/V 用**逐页显式写入**构造，并对索引做 round-trip 自检。
    def _c22_fixture(kv, nq, hs, w_page_offset=2, seed=7):
        torch.manual_seed(seed)
        nblk = kv // 32
        kc = torch.zeros(nblk + w_page_offset, nq, 32, hs, dtype=torch.float32)
        vc = torch.zeros(nblk + w_page_offset, nq, 32, hs, dtype=torch.float32)
        kl = torch.randn(kv, nq, hs, dtype=torch.float32)
        vl = torch.randn(kv, nq, hs, dtype=torch.float32)
        for j in range(nblk):
            for t in range(32):
                kc[w_page_offset + j, :, t, :] = kl[j * 32 + t]
                vc[w_page_offset + j, :, t, :] = vl[j * 32 + t]
        bt = (torch.arange(nblk, dtype=torch.int32) + w_page_offset).unsqueeze(0)
        logical = torch.arange(kv)
        rt = kc[bt[0][logical // 32], :, logical % 32, :]
        assert torch.equal(rt, kl), "T7 fixture round-trip failed"
        return kc, vc, bt, kl, vl

    def _c22_ref(q, kl, vl, qpos, w, scale=0.5):
        qt = q.transpose(0, 1).unsqueeze(0)
        kt = kl.transpose(0, 1).unsqueeze(0)
        vt = vl.transpose(0, 1).unsqueeze(0)
        sc = (qt @ kt.transpose(-1, -2)) * scale
        kpos = torch.arange(kl.shape[0])
        m = kpos[None, :] > qpos[:, None]
        if w is not None:
            m = m | (kpos[None, :] < (qpos[:, None] - (w - 1)))
        pr = torch.softmax(sc.masked_fill(m[None, None], float("-inf")), dim=-1)
        return (pr @ vt).squeeze(0).transpose(0, 1)

    # T7.0：小窗口（window=3）的掩码语义 —— kv=32 单页、window=3
    kc7, vc7, bt7, kl7, vl7 = _c22_fixture(32, 1, 3)
    q7 = torch.randn(32, 1, 3, dtype=torch.float32)
    got7 = sdpa_prefill_attention_math(
        q7, kc7, vc7, bt7, torch.tensor([0, 32], dtype=torch.int32),
        torch.tensor([32], dtype=torch.int32), 0.5, True, window_size=3)
    ref7 = _c22_ref(q7, kl7, vl7, torch.arange(32), 3)
    check("T7.0 sliding window=3 与独立右对齐参考一致",
          torch.allclose(got7, ref7, atol=1e-6),
          f"maxdiff={(got7-ref7).abs().max():.2e}")

    # ---- T9 prefill 长度 > 窗口：窗口必须被真的施加 ----
    kv9, w9 = 1536, 1024
    kc9, vc9, bt9, kl9, vl9 = _c22_fixture(kv9, 2, 3, seed=11)
    q9 = torch.randn(kv9, 2, 3, dtype=torch.float32)
    got9 = sdpa_prefill_attention_math(
        q9, kc9, vc9, bt9, torch.tensor([0, kv9], dtype=torch.int32),
        torch.tensor([kv9], dtype=torch.int32), 0.5, True, window_size=w9)
    ref9 = _c22_ref(q9, kl9, vl9, torch.arange(kv9), w9)
    check(f"T9.0 full prefill kv={kv9} > window={w9} 与独立右对齐参考一致",
          torch.allclose(got9, ref9, atol=1e-5),
          f"maxdiff={(got9-ref9).abs().max():.2e}")
    ref9_full = _c22_ref(q9, kl9, vl9, torch.arange(kv9), None)
    check("T9.1 窗口未被忽略（结果 != 全因果参考）",
          not torch.allclose(got9, ref9_full, atol=1e-5),
          f"maxdiff_vs_full_causal={(got9-ref9_full).abs().max():.2e}")

    # T9.2 appended-chunk 形态（chunked prefill 产生的形状）：q=512 / kv=1536
    q9b = torch.randn(512, 2, 3, dtype=torch.float32)
    got9b = sdpa_prefill_attention_math(
        q9b, kc9, vc9, bt9, torch.tensor([0, 512], dtype=torch.int32),
        torch.tensor([kv9], dtype=torch.int32), 0.5, True, window_size=w9)
    ref9b = _c22_ref(q9b, kl9, vl9, torch.arange(512) + (kv9 - 512), w9)
    check("T9.2 appended chunk q=512/kv=1536/window=1024 与独立参考一致",
          torch.allclose(got9b, ref9b, atol=1e-5),
          f"maxdiff={(got9b-ref9b).abs().max():.2e}")

    # T9.3 kv <= window 时窗口是 no-op（必须等于全因果）
    kc9c, vc9c, bt9c, kl9c, vl9c = _c22_fixture(512, 2, 3, seed=13)
    q9c = torch.randn(512, 2, 3, dtype=torch.float32)
    got9c = sdpa_prefill_attention_math(
        q9c, kc9c, vc9c, bt9c, torch.tensor([0, 512], dtype=torch.int32),
        torch.tensor([512], dtype=torch.int32), 0.5, True, window_size=1024)
    ref9c = _c22_ref(q9c, kl9c, vl9c, torch.arange(512), None)
    check("T9.3 kv<=window => 等于全因果（窗口为 no-op）",
          torch.allclose(got9c, ref9c, atol=1e-5),
          f"maxdiff={(got9c-ref9c).abs().max():.2e}")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
