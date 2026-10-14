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

"""Process-init Hy-MT2 bindings for the tested FP16/TP1 SDAA runtime.

RMSNorm and SwiGLU retain the vendor operations behind compile-safe boundaries.
The attention/cache binding supplies the installed BlockAttention forward stub.
The older Transformers overlays carry separate BF16 correctness/performance
evidence; those results do not establish this vLLM export's performance.
"""

import sys


def install_rotary_embedding_overlay() -> None:
    """Bind vLLM's RotaryEmbedding to the delivered tecoops kernel (fp16)."""
    import os

    # A/B arm selector, read exactly once here (process initialization).  The
    # default is the delivered arm; "off" keeps vLLM's native rotary chain for the
    # baseline arm of a same-build A/B run.  Nothing on the forward path reads the
    # environment.
    if os.environ.get("HY_ROTARY_EMBEDDING", "on").lower() == "off":
        print("[hy-mt2-overlay] HY_ROTARY_EMBEDDING=off: rotary embedding overlay "
              "skipped (A/B baseline arm)", file=sys.stderr)
        return

    try:
        import tecoops
        import vllm.model_executor.layers.rotary_embedding as rope_module
        from custom_ops.hy_mt2.rotary_fused import (
            bind_instance,
            official_rotary_embedding,
            resolve_rotary_base,
        )
    except Exception as exc:  # pragma: no cover - depends on the SDAA image
        raise RuntimeError(f"Hy-MT2 rotary overlay dependencies unavailable: {exc}") from exc

    if not hasattr(tecoops, "rotary_embedding"):
        raise RuntimeError("Hy-MT2 rotary overlay requires tecoops.rotary_embedding")

    op = official_rotary_embedding(tecoops)
    if op is None:
        raise RuntimeError("Hy-MT2 rotary overlay failed to register the opaque op")

    target = resolve_rotary_base(rope_module)
    if target is None or not hasattr(target, "__init__"):
        raise RuntimeError("Hy-MT2 rotary overlay could not resolve RotaryEmbeddingBase "
                           "(searched the package attribute, its `base` submodule and "
                           "the module path)")

    count_calls = os.environ.get("HY_ROTARY_COUNT", "") == "1"
    original_init = target.__init__
    bound = {"n": 0}

    # Seam 1: every rope instance is produced by the get_rope factory.
    factory = getattr(rope_module, "get_rope", None)
    if callable(factory):
        def _get_rope(*args, **kwargs):
            inst = factory(*args, **kwargs)
            if bind_instance(inst, op, count=count_calls):
                bound["n"] += 1
            return inst

        rope_module.get_rope = _get_rope
        if rope_module.get_rope is not _get_rope:
            raise RuntimeError("Hy-MT2 get_rope overlay identity check failed")

    # Seam 2: direct construction (an OOT-registered class would bypass seam 1).
    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if bind_instance(self, op, count=count_calls):
            bound["n"] += 1

    target.__init__ = _init
    if target.__init__ is not _init:
        raise RuntimeError("Hy-MT2 rotary embedding overlay identity check failed")

    import atexit

    def _summary():
        print(f"[hy-mt2-overlay] rotary overlay summary: instances_bound={bound['n']} "
              f"count_calls={count_calls}", file=sys.stderr)

    atexit.register(_summary)
    print("[hy-mt2-overlay] RotaryEmbedding overlay bound to tecoops.rotary_embedding "
          f"(instances bound so far: {bound['n']})", file=sys.stderr)


def install() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch_sdaa  # noqa: F401
        import torch_sdaa.nn  # noqa: F401  registers torch.ops.sdaa.swiglu
        import tecoops
        from custom_ops.hy_mt2.vllm_rms_norm import rms_norm, rms_norm_add
        import vllm.model_executor.models.hunyuan_v1 as hunyuan
    except Exception as exc:  # pragma: no cover - depends on the SDAA image
        raise RuntimeError(f"Hy-MT2 overlay dependencies unavailable: {exc}") from exc

    if not hasattr(tecoops, "rms_norm"):
        raise RuntimeError("Hy-MT2 overlay requires tecoops.rms_norm")
    if getattr(torch.ops.sdaa, "swiglu", None) is None:
        raise RuntimeError("Hy-MT2 overlay requires torch.ops.sdaa.swiglu")

    from custom_ops.hy_mt2.vllm_swiglu import swiglu

    from custom_ops.hy_mt2.vllm_attention import install_attention
    install_attention()

    class TecoopsRMSNorm(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
            var_hidden_size: int | None = None,
            has_weight: bool = True,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            if var_hidden_size not in (None, hidden_size):
                raise ValueError("Hy-MT2 RMSNorm overlay does not support var_hidden_size")
            if not has_weight:
                raise RuntimeError("Hy-MT2 RMSNorm requires a learned weight")
            norm_dtype = dtype or torch.get_default_dtype()
            if norm_dtype != torch.float16 or eps <= 0:
                raise RuntimeError("Hy-MT2 RMSNorm requires FP16 and positive epsilon")
            self.hidden_size = hidden_size
            self.variance_epsilon = eps
            self.has_weight = has_weight
            if has_weight:
                self.weight = nn.Parameter(
                    torch.ones(hidden_size, dtype=dtype or torch.get_default_dtype())
                )

        def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
            x2 = x.reshape(-1, x.shape[-1])
            residual2 = residual.reshape_as(x2) if residual is not None else None
            out2 = torch.empty_like(x2)
            residual_out2 = torch.empty_like(x2) if residual2 is not None else None
            weight = self.weight.data
            if residual2 is None:
                rms_norm(x2, weight, out2, float(self.variance_epsilon))
            else:
                rms_norm_add(
                    x2, weight, residual2, out2, residual_out2,
                    float(self.variance_epsilon),
                )
            out = out2.reshape_as(x)
            if residual_out2 is not None:
                return out, residual_out2.reshape_as(x)
            return out

    def fused_mlp_forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        tokens = gate_up.numel() // gate_up.shape[-1]
        intermediate = gate_up.shape[-1] // 2
        activated = swiglu(
            gate_up.reshape(1, tokens, 1, gate_up.shape[-1]),
        ).reshape(*gate_up.shape[:-1], intermediate)
        output, _ = self.down_proj(activated)
        return output

    hunyuan.RMSNorm = TecoopsRMSNorm
    hunyuan.HunYuanMLP.forward = fused_mlp_forward
    install_rotary_embedding_overlay()
    from custom_ops.hy_mt2.linear_stream import install_linear_stream
    install_linear_stream()
    if hunyuan.RMSNorm is not TecoopsRMSNorm:
        raise RuntimeError("Hy-MT2 RMSNorm binding failed")
    if hunyuan.HunYuanMLP.forward is not fused_mlp_forward:
        raise RuntimeError("Hy-MT2 SwiGLU binding failed")
    print("[hy-mt2-overlay] RMSNorm and SwiGLU bindings installed", file=sys.stderr)


try:
    install()
except Exception as exc:
    print(f"FATAL: Hy-MT2 overlay initialization failed: {exc}", file=sys.stderr)
    # sitecustomize's ordinary exceptions are swallowed by Python startup.
    raise SystemExit(1) from exc
