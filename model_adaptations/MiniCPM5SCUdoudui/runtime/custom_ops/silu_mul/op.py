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

"""Opaque bridge from MiniCPM SiLU x Mul to vendor fused SWIGLU.

The optional SDAA implementation is bound during process initialization.
The CPU implementation supports reference, FakeTensor, and Dynamo checks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_NAMESPACE = "minicpm_silu_mul"
_OP_NAME = f"{_NAMESPACE}::silu_mul"


def _validate_input(x: torch.Tensor) -> int:
    if not isinstance(x, torch.Tensor):
        raise TypeError("silu_mul expects a torch.Tensor")
    if x.ndim < 2:
        raise ValueError(f"silu_mul expects rank >= 2, got {x.ndim}")
    packed_width = int(x.shape[-1])
    if packed_width <= 0 or packed_width % 2:
        raise ValueError(f"silu_mul expects a positive even packed width, got {packed_width}")
    if x.dtype != torch.float16:
        raise TypeError(f"silu_mul currently accepts FP16 only, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("silu_mul requires contiguous packed input")
    return packed_width // 2


def _cpu_silu_mul(x: torch.Tensor) -> torch.Tensor:
    width = _validate_input(x)
    return F.silu(x[..., :width]) * x[..., width:]


_SDAA_SWIGLU = None


def bind_vendor_silu_mul(kernel) -> None:
    """Bind the vendor callable once during process initialization."""
    global _SDAA_SWIGLU
    if not callable(kernel):
        raise TypeError("vendor SWIGLU binding must be callable")
    from custom_ops.receipt import bind_official_api

    _SDAA_SWIGLU = bind_official_api(kernel, "silu_mul.vendor_swiglu")


def _sdaa_silu_mul(x: torch.Tensor) -> torch.Tensor:
    width = int(x.shape[-1]) // 2
    token_count = x.numel() // int(x.shape[-1])
    packed = x.view(token_count, 1, int(x.shape[-1]))
    output = _SDAA_SWIGLU(packed, token_count, 1, width)
    return output.view(*x.shape[:-1], width)


_LIB_DEF = torch.library.Library(_NAMESPACE, "DEF")
_LIB_DEF.define("silu_mul(Tensor x) -> Tensor")
_LIB_CPU = torch.library.Library(_NAMESPACE, "IMPL", "CPU")
_LIB_CPU.impl("silu_mul", _cpu_silu_mul)
_LIB_SDAA = torch.library.Library(_NAMESPACE, "IMPL", "PrivateUse1")
_LIB_SDAA.impl("silu_mul", _sdaa_silu_mul)


@torch.library.register_fake(_OP_NAME)
def _silu_mul_fake(x: torch.Tensor) -> torch.Tensor:
    width = _validate_input(x)
    return x.new_empty((*x.shape[:-1], width))


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    """Call the opaque operator registered for CPU and SDAA."""
    return torch.ops.minicpm_silu_mul.silu_mul(x)
