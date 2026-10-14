# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
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

"""Hy-MT2 M=1 decode linear -> ``tecoops.linear_stream`` (weight streaming).

Why this exists
---------------
Every decode step of this model is one activation row against a stack of weight
matrices, i.e. pure HBM weight streaming: 3.582 GB of fp16 weights are read per
token.  The vendor M=1 GEMM does not reach the card's streaming bandwidth; the
already-delivered ``weight_stream_gemv`` kernel (L1 C++ kernel in
``teco/ual/kernel/weight_stream_gemv``, L2 binding ``m.def("linear_stream", ...)``
at ``api/torch_ext.cpp``) streams each weight row as one contiguous slab with
double-buffered DMA.  This file is the *model call-site* half of the operator
delivery (AGENTS.md three-layer contract); it changes no operator code.

Binding form (construct-time, NOT per call)
------------------------------------------
``LinearBase.forward`` -> ``quant_method.apply`` -> ``dispatch_unquantized_gemm()``
-> the module-global ``default_unquantized_gemm`` of
``vllm.model_executor.layers.utils``.  Replacing that name once, at process
initialization, is the call-site change.  The replacement contains **no branch,
no shape test, no getenv** (AGENTS.md rule 6): it forwards to one opaque
``torch.library.custom_op``.  The M==1 vs vendor decision lives in the C++ host
function of that operator and therefore survives ``torch.compile`` -- a
Python-level shape test would be folded against the profiling trace
(M = max_num_batched_tokens) and the decode step would silently keep the vendor
branch for ever.

Scope: M == 1 (single decode row) only.  Prefill (M > 1) and any shape that does
not satisfy the kernel contract keep the vendor GEMM untouched, because the
opaque op falls back to ``torch::linear`` in C++.
"""

from __future__ import annotations

import sys

import torch

import tecoops

__all__ = ["weight_stream_gemv_m1", "linear_stream", "install_linear_stream",
           "mapped_libteco_ops"]


def mapped_libteco_ops() -> list[str]:
    """The ``libteco_ops.so`` paths actually mapped into this process."""
    out = set()
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.split()[-1]
                if path.endswith("libteco_ops.so"):
                    out.add(path)
    except OSError:  # pragma: no cover - /proc always present on Linux
        pass
    return sorted(out)


@torch.library.custom_op(
    "hy_mt2_linear_stream::gemv_m1", mutates_args=("out",), device_types="sdaa"
)
def weight_stream_gemv_m1(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> None:
    tecoops.weight_stream_gemv(x, weight, out)


@weight_stream_gemv_m1.register_fake
def _weight_stream_gemv_m1_fake(x, weight, out):
    return None


@torch.library.custom_op(
    "hy_mt2_linear_stream::linear_stream",
    mutates_args=(),
    device_types="sdaa",
)
def linear_stream(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Opaque replacement for ``F.linear``.

    The M==1 vs vendor decision is in the C++ host function
    (``api/torch_ext.cpp: linear_stream_torch``); nothing about it is visible to
    Dynamo, so this operator is a single graph node that always executes and the
    *runtime* shape decides which kernel is launched.
    """
    return tecoops.linear_stream(x, weight, bias)


@linear_stream.register_fake
def _linear_stream_fake(x, weight, bias=None):
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
    )


def install_linear_stream() -> dict:
    """Bind the streaming linear once, at process initialization.

    The A/B selector is read exactly once, here; the forward path never reads the
    environment (AGENTS.md rule 6).
    """
    import os

    if os.environ.get("HY_MT2_LINEAR_STREAM", "1").lower() in ("0", "off", "false"):
        print("[hy-mt2-overlay] HY_MT2_LINEAR_STREAM=off: linear_stream overlay "
              "skipped (A/B baseline arm)", file=sys.stderr)
        return {"skipped": True}

    if not hasattr(tecoops, "linear_stream"):
        raise RuntimeError("Hy-MT2 linear_stream overlay requires "
                           "tecoops.linear_stream")

    import vllm.model_executor.layers.utils as U

    if getattr(U, "_hy_mt2_linear_stream_patched", False):
        return {"already_patched": True}

    orig = U.default_unquantized_gemm

    def stream_unquantized_gemm(layer, x, weight, bias=None):
        return linear_stream(x, weight, bias)

    U.default_unquantized_gemm = stream_unquantized_gemm
    if U.default_unquantized_gemm is not stream_unquantized_gemm:
        raise RuntimeError("linear_stream binding identity check failed")
    U._hy_mt2_linear_stream_patched = True

    info = {
        "bound": "vllm.model_executor.layers.utils.default_unquantized_gemm",
        "orig": getattr(orig, "__name__", repr(orig)),
        "mapped_libteco_ops": mapped_libteco_ops(),
    }
    print("[hy-mt2-overlay] LinearBase linear binding installed "
          "(tecoops.linear_stream; M=1 streams weights, M>1 vendor) "
          f"orig={info['orig']} libs={info['mapped_libteco_ops']}", file=sys.stderr)
    return info
