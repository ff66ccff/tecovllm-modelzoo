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
