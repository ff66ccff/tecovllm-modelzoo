"""受控启动期注入点（overlay）。

用法（不改 site-packages、不改 transformers/vLLM）：

    export GEMMA4_OVERLAY=1
    export PYTHONPATH=/root/SichuanUniversity_AI_accelerator-gemma/overlay:$PYTHONPATH
    /home/py312/bin/python -m vllm.entrypoints.openai.api_server --model ... --config-format gemma4_shim

`site` 模块在解释器启动时自动 import sitecustomize；PYTHONPATH 上的本目录先于
site-packages 命中，因此在 vLLM 读取 config / 构建 tokenizer 之前完成注册。
必须显式设置 GEMMA4_OVERLAY=1 才生效，避免污染同机其他会话进程。
"""

import os


def install_block_attention_overlay() -> None:
    """把厂商未实现的 BlockAttentionImpl.forward 换成 in-repo 实现。

    厂商缺口：`vllm_sdaa/attention/block_attn.py:881` 的
    `BlockAttentionImpl.forward` 无条件 `raise NotImplementedError
    ("flash attention is not implemented!")`，而 `platform.py:263` 又把
    BlockAttentionBackend 作为唯一的默认 attn 后端返回。结果是任何 SDAA vLLM V1
    模型都会在 KV profiling 的前向里直接死掉。

    修复：进程初始化期绑定 `custom_ops.block_attention.op.sdaa_block_attention_forward`
    （转发到 `_C_cache_ops.reshape_and_cache_flash` 与 `_C_sdaa.block_attention`）。
    不改 site-packages；热路径无分支、无 getenv。
    参考实现来源：model/internvl3_5-8b（已在本目录 custom_ops/ 内注明 provenance）。
    """
    import sys

    try:
        import vllm_sdaa.attention.block_attn as block_attn_module
        from custom_ops.block_attention.op import sdaa_block_attention_forward
    except Exception as exc:
        print(f"[gemma4-overlay] FATAL: block attention bind failed: {exc}", file=sys.stderr)
        raise

    impl_cls = block_attn_module.BlockAttentionImpl
    impl_cls.forward = sdaa_block_attention_forward
    if impl_cls.forward is not sdaa_block_attention_forward:
        raise RuntimeError("block attention identity check failed")

    # 第二个厂商缺口：`_C_sdaa.block_attention` 只接受
    # window_size_left ∈ {-1, 127, 511}（见 host_block_multihead_attention.cpp:252 的
    # 断言）。Gemma4 的 sliding_window=1024 会被 BlockAttentionImpl.__init__ 算成
    # `sliding_window - 1 = 1023`，于是 decode 路径直接触发
    #   Assertion `(window_size_left == -1 || window_size_left == 127 ||
    #               window_size_left == 511)` failed
    #   -> RuntimeError: TECOCUSTOM_STATUS_SUCCESS INTERNAL ASSERT FAILED
    #      at aten/native/adaptor/attention_adaptor.cpp:112
    #
    # 归一化在**构造期**一次性完成，不在 forward 热路径加分支（AGENTS.md 规则 6）。
    # 语义说明：1024 无对应的合法窗口值，取 -1（不设窗）。
    #   * 序列 ≤ 1024 token 时与窗口 1024 完全等价（所有 key 本来就在窗内）；
    #   * 序列 > 1024 时 sliding 层会看到超出训练窗口的 key —— 是**近似**，
    #     会偏离训练语义（质量风险），但不崩溃。已知边界见 experiments 记录。
    if not getattr(impl_cls, "_gemma4_window_patch", False):
        _kernel_windows = (-1, 127, 511)
        _orig_init = impl_cls.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            left, right = self.sliding_window
            # vLLM stores a causal window W as W-1. Preserve the semantic
            # window before normalizing the vendor-only field below; the
            # in-repo Gemma kernel consumes the real W (1024 for D256).
            self._gemma4_window_size = left + 1 if left >= 0 else None
            if left not in _kernel_windows:
                self.sliding_window = (-1, right)
            # 第三个厂商缺口：flash SDP 只支持 head_dim <= 256，而 Gemma4 的 8 个
            # full_attention 层 global_head_dim=512。按 head_size 在构造期一次性
            # 绑定 prefill/decode 实现（head_size>256 走显式 math），
            # forward 内不保留后端 if-else（AGENTS.md 规则 6）。
            from custom_ops.block_attention.op import bind_attention_impls

            bind_attention_impls(self)

        impl_cls.__init__ = __init__
        impl_cls._gemma4_window_patch = True

    # 第四个厂商缺口：decode 的 seq_lens 少 1（见 custom_ops/block_attention/metadata_fix.py）
    try:
        from custom_ops.block_attention import metadata_fix

        if metadata_fix.install():
            print("[gemma4-overlay] decode seq_len off-by-one fix installed.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[gemma4-overlay] metadata_fix skipped: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 第五个厂商缺口：BlockAttentionMetadata 不转发 query_start_loc/seq_lens 的 CPU 孪生，
    # 逼得算子逐层做 int(device_tensor)（v1 实测 145 次 D2H、占 span 91.8%）。
    # 见 runtime/custom_ops/block_attention/metadata_cpu_fix.py（只包一层 build，不改 site-packages）。
    try:
        from custom_ops.block_attention import metadata_cpu_fix

        if metadata_cpu_fix.install():
            print("[gemma4-overlay] zero-D2H cpu-twin metadata fix installed.",
                  file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[gemma4-overlay] metadata_cpu_fix skipped: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    print("[gemma4-overlay] BlockAttention overlay bound successfully.", file=sys.stderr)


def install_rms_norm_overlay() -> None:
    """把 Gemma4 的 337 个 RMSNorm 站点绑定到**已交付的 tecoops C++ kernel**（算子交付第 3 层）。

    **默认 = `whole`**：`tecoops.rms_norm` 整链单 kernel（统计量 + epilogue 一次发射、
    一次 fp32 舍入）。冻结库字节上实测模型级 **B/A1 = −26.58 %**（3 轮轮转中位，漂移
    +0.76 %，max(B) < min(A)，9/9 臂 32/32 token IDs == 冻结基线）。初始化期开关
    `GEMMA4_RMSNORM_STATS=compat` 退回下面的参考精确拆分形态（仅作回退）。

    以下为 `compat`（回退档）的原始说明：把 Gemma4 的 RMSNorm 绑定到**参考精确**的 `tecoops.rms_norm_compat`。

    与 2026-10-10 那次绑定（交付 `tecoops.rms_norm`，因 1–4 half-ULP 舍入序差异被否决）的
    唯一区别：统计量（mean/rsqrt）仍由参考算子计算，而 epilogue
    `y = fp16(fp16(fp32(x)*rstd)*weight)`（含 residual 融合分支）由新 kernel 按参考的
    舍入顺序执行 —— 在 85 个真实 Gemma shape / 28 507 904 个元素上 max ULP = 0。

    机制（本分支实测，不是推断）：`vllm...layernorm.RMSNorm` 是 CustomOp，本栈
    `current_platform.is_out_of_tree()` 为真且 `vllm_sdaa` 未注册 OOT `rms_norm`
    （三处 `register_oot` 全注释）⇒ 分派到 `forward_oot` = `forward_native`：
      * 无 residual → `vllm.ir.ops.rms_norm(...)`
      * 有 residual → `RMSNorm.forward_static(...)`
    而 `vllm/ir/ops/layernorm.py` **不是融合 kernel**，是纯 PyTorch 逐算子链
    （to(fp32)/pow/mean/add/rsqrt/mul/to/mul(weight)/to），每次调用约 8 个 device
    kernel + 8 次 host 派发。Gemma4 每 decode step 调用 337 次
    （4*48 hidden-2D + 3*48 qkv-3D + 1 final）；本分支 profile 每 step 恰好
    337 个 `aten::mean` / `aten::rsqrt` / `aten::pow`，并把 5149 次
    `sdaaLaunchKernel` 中的约 2900 次归到 RMSNorm 族。
    绑定后每次调用只剩 1 个 kernel。

    绑定形态：`torch.library.custom_op`（Hy-MT2 overlay 已验证；2026-10-02 的裸
    pybind 调用被 Dynamo 标为 skipped/untraceable）。q/k/v 的 3-D 站点走
    `reshape(-1, head_dim)` 的**连续 2-D 入口**，不新增 `__global__` kernel。

    选择只在进程初始化期做一次（AGENTS.md 规则 6）：forward 热路径无后端分支、
    无 fallback、无 getenv。`GEMMA4_RMSNORM_BIND=0` 只在初始化期读取一次，用于
    同树 A/B/A 的未绑定臂。
    """
    import sys

    try:
        import torch  # noqa: F401
        import torch_sdaa  # noqa: F401
        import tecoops
        from custom_ops.rms_norm.op import (STATS_MODE, TecoopsRMSNorm,
                                             mapped_libteco_ops)
        import vllm.model_executor.models.gemma4 as gemma4
    except Exception as exc:  # pragma: no cover - depends on the SDAA image
        print(f"[gemma4-overlay] FATAL: rms_norm bind deps unavailable: {exc}",
              file=sys.stderr)
        raise

    # fail-closed on the symbol the SELECTED mode needs (GEMMA4_RMSNORM_STATS is
    # read once, inside custom_ops.rms_norm.op, at import time).
    required = "rms_norm" if STATS_MODE == "whole" else "rms_norm_compat"
    if not hasattr(tecoops, required):
        raise RuntimeError(
            f"Gemma4 rms_norm overlay (GEMMA4_RMSNORM_STATS={STATS_MODE!r}) requires "
            f"tecoops.{required} (the reference-exact epilogue specialization for "
            "'compat', the delivered whole chain for 'whole'); rebuild/stage a "
            "libteco_ops.so and _torch_ext that export it"
        )

    # fail-closed：进程实际映射的必须是交付库本身，而不是 site-packages 里的旧库。
    libs = mapped_libteco_ops()
    if not libs:
        raise RuntimeError(
            "Gemma4 rms_norm overlay: no libteco_ops.so mapped into this process"
        )
    expected = os.environ.get("GEMMA4_RMSNORM_EXPECT_LIB")
    if expected:
        if [os.path.realpath(p) for p in libs] != [os.path.realpath(expected)]:
            raise RuntimeError(
                "Gemma4 rms_norm overlay FAIL-CLOSED: mapped "
                f"{libs} != expected {expected}"
            )
    elif len(libs) != 1:
        raise RuntimeError(
            f"Gemma4 rms_norm overlay FAIL-CLOSED: expected exactly one "
            f"libteco_ops.so, mapped {libs}"
        )

    gemma4.RMSNorm = TecoopsRMSNorm
    if gemma4.RMSNorm is not TecoopsRMSNorm:
        raise RuntimeError("Gemma4 rms_norm overlay identity check failed")

    print(f"[gemma4-overlay] RMSNorm bound to tecoops.{required} "
          f"(GEMMA4_RMSNORM_STATS={STATS_MODE}); mapped={libs}", file=sys.stderr)
    print(f"[gemma4-overlay] RMSNorm forward impl = "
          f"{getattr(gemma4.RMSNorm.forward, '__name__', '?')}", file=sys.stderr)

def install_linear_stream_overlay() -> None:
    """把 Gemma4 的 decode 线性层绑定到 ``tecoops.weight_stream_gemv``（算子交付第 3 层）。

    机制（本分支实测，不是推断）：本模型每 decode step 只有 **一行** 激活，线性层
    的耗时 100% 是权重矩阵的 HBM 流式读取。同卡 fp16 M=1 实测：厂商路径
    ``vllm.model_executor.layers.utils.default_unquantized_gemm`` 有效率
    **30.1–77.5 GB/s**（权重字节/中位秒），而同样字节的纯流式读 **92–197 GB/s**、
    读写 copy **206–269 GB/s** —— 差距来自 M=1 GEMM 的形状，不是带宽上限。
    新 kernel 按整行连续 slab 双缓冲流式读权重、激活行常驻 fp32 SPM。

    绑定形态：``torch.library.custom_op``（rms_norm overlay 已证；2026-10-02 的裸
    pybind 调用被 Dynamo 标为 skipped/untraceable），钉在
    ``vllm.model_executor.layers.utils.default_unquantized_gemm`` 这个 vLLM
    ``LinearBase.apply`` 实际调用的函数上。选择只在进程初始化期做一次
    （AGENTS.md 规则 6）：forward 热路径无后端分支、无 fallback、无 getenv；
    唯一的 per-call 判断是算子的**形状契约**（``M==1`` 且 N%256==0 且 K%16==0），
    契约之外的形状留在厂商 GEMM —— 这是特化，不是 fallback。

    ``GEMMA4_LINEAR_STREAM`` 只在初始化期读一次；``GEMMA4_LINEAR_EXPECT_LIB``
    是 fail-closed 的库身份断言（同 rms_norm overlay）。
    """
    import sys

    try:
        import torch  # noqa: F401
        import torch_sdaa  # noqa: F401
        import tecoops
        from custom_ops.linear_stream.op import install_linear_stream, mapped_libteco_ops
    except Exception as exc:  # pragma: no cover - depends on the SDAA image
        print(f"[gemma4-overlay] FATAL: linear_stream bind deps unavailable: {exc}",
              file=sys.stderr)
        raise

    if not hasattr(tecoops, "weight_stream_gemv"):
        raise RuntimeError(
            "Gemma4 linear_stream overlay requires tecoops.weight_stream_gemv "
            "(rebuild/stage a libteco_ops.so and _torch_ext that export it)"
        )

    libs = mapped_libteco_ops()
    if not libs:
        raise RuntimeError(
            "Gemma4 linear_stream overlay: no libteco_ops.so mapped into this process"
        )
    expected = os.environ.get("GEMMA4_LINEAR_EXPECT_LIB")
    if expected:
        if [os.path.realpath(p) for p in libs] != [os.path.realpath(expected)]:
            raise RuntimeError(
                "Gemma4 linear_stream overlay FAIL-CLOSED: mapped "
                f"{libs} != expected {expected}"
            )

    info = install_linear_stream()
    print(f"[gemma4-overlay] linear_stream bound to tecoops.weight_stream_gemv: {info}",
          file=sys.stderr)


import os
import sys
if os.environ.get("SCUD_REQUIRE_OPS_BUNDLE") == "1":
    try:
        from custom_ops.ops_provenance import require_ops_bundle
        require_ops_bundle(['rms_norm', 'rms_norm_compat', 'reshape_and_cache', 'flash_attn_varlen_func', 'weight_stream_gemv', 'linear_stream'])
    except Exception as exc:
        sys.stderr.write(f"FATAL: operator bundle provenance check failed: {exc}\n")
        raise SystemExit(1) from exc

if os.environ.get("GEMMA4_OVERLAY") == "1":
    try:
        import gemma4_shim

        # config shim + tokenizer 兼容 + Gemma4 文本主干 mm 权重跳过
        gemma4_shim.install()
        # 厂商未实现的 SDAA BlockAttention 前向
        install_block_attention_overlay()
        # 已交付 tecoops.rms_norm 的构造期绑定（GEMMA4_RMSNORM_BIND=0 关闭）
        if os.environ.get("GEMMA4_RMSNORM_BIND", "1") == "1":
            install_rms_norm_overlay()
        # decode M=1 线性层的权重流式 GEMV。默认开启（与 rms_norm_compat 一致）；
        # GEMMA4_LINEAR_STREAM=0 只在初始化期读取一次，用于同树 A/B/A 的未绑定臂。
        if os.environ.get("GEMMA4_LINEAR_STREAM", "1") == "1":
            install_linear_stream_overlay()
    except Exception as exc:  # 启动期失败必须可见
        import sys

        print(f"[gemma4-overlay] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        # 注意：sitecustomize 里普通 `raise` 会被 CPython site.py 吞掉
        # （stderr 只留 "Error in sitecustomize"），解释器**照常继续**并把主程序跑到 main
        # ——那样就是"没绑上 overlay 也照样启动"的静默降级，与 fail-closed 承诺不符。
        # 设备无关实测（readme-c16/failclosed/）：普通异常 rc=0 且 MAIN_REACHED；
        # `raise SystemExit(1)` rc=1 且不打印 MAIN_REACHED。故此处显式退出进程。
        raise SystemExit(1)

from custom_ops.diagnostic_shapes import install_decode_census
install_decode_census()
