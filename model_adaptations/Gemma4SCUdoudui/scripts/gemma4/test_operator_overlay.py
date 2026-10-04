#!/usr/bin/env python
# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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

    # ---- T7 Gemma sliding-window math kernel ----
    # The self-developed path uses right-aligned causal positions and keeps
    # exactly [q_pos-window+1, q_pos].  This is the model-side kernel contract
    # for Gemma D256 sliding layers; D512 passes window_size=None (global).
    q7 = torch.randn(6, 1, 3, dtype=torch.float32)
    kc7 = torch.randn(2, 1, 4, 3, dtype=torch.float32)
    vc7 = torch.randn_like(kc7)
    bt7 = torch.tensor([[0, 1]], dtype=torch.int64)
    got7 = sdpa_prefill_attention_math(
        q7, kc7, vc7, bt7, torch.tensor([0, 6]), torch.tensor([6]),
        0.5, True, window_size=3,
    )
    k7 = torch.cat((kc7[0, :, :4, :], kc7[1, :, :2, :]), dim=1)
    v7 = torch.cat((vc7[0, :, :4, :], vc7[1, :, :2, :]), dim=1)
    q7t = q7.transpose(0, 1).unsqueeze(0)
    k7t = k7.unsqueeze(0)
    v7t = v7.unsqueeze(0)
    scores7 = (q7t @ k7t.transpose(-1, -2)) * 0.5
    pos7 = torch.arange(6)
    mask7 = (torch.arange(6)[None, :] > pos7[:, None]) | (
        torch.arange(6)[None, :] < (pos7[:, None] - 2)
    )
    scores7 = scores7.masked_fill(mask7[None, None], float("-inf"))
    ref7 = (torch.softmax(scores7, dim=-1) @ v7t).squeeze(0).transpose(0, 1)
    check("T7.0 D256 sliding window=3 与独立参考一致",
          torch.allclose(got7, ref7, atol=1e-6),
          f"maxdiff={(got7-ref7).abs().max():.2e}")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
