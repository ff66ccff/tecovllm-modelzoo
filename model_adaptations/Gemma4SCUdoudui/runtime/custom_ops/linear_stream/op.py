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

"""Gemma4 M=1 decode linear -> ``tecoops.weight_stream_gemv`` (weight streaming).

Why this exists (measured on this branch, not assumed)
------------------------------------------------------
Every decode step of this model is one activation row against a stack of weight
matrices, i.e. pure HBM weight streaming.  Measured on one card, fp16, M=1, on the
real Gemma-4-12B-it shapes, the vendor GEMM that ``LinearBase`` reaches through
``vllm.model_executor.layers.utils.default_unquantized_gemm`` runs at
**30.1-77.5 GB/s** effective (weight bytes / time), while a pure streaming read of
the same bytes reaches **92-197 GB/s** and a read+write copy **206-269 GB/s**.
The gap is the M=1 GEMM's shape, not a bandwidth ceiling.

Binding form (construct-time, NOT per call)
------------------------------------------
An earlier form of this file replaced ``default_unquantized_gemm`` with a wrapper
whose *per-call* shape contract decided streaming vs vendor.  That form does not
survive the model's ``torch.compile`` path: the contract is a decision on symbolic
dimensions, so the traced decode graph keeps whichever branch the profiling trace
saw, and materialising a dimension inside the trace conflicts with vLLM's
``mark_dynamic`` dimensions (observed: "You marked L['positions'].size()[0] as
dynamic but your code specialized it to be a constant (2048)").  Marking the
wrapper ``torch._dynamo.disable`` is also rejected by this vLLM
("Skip calling ``torch.compiler.disable()``d function").

So the selection happens **once, at weight-load time**, where shapes are real
Python ints: ``UnquantizedLinearMethod.process_weights_after_loading`` is wrapped
and, for every layer whose weight satisfies the kernel's contract, the layer's
``quant_method`` is replaced by :class:`StreamingLinearMethod`.  The traced
``apply`` therefore contains **no backend branch, no fallback, no getenv and no
shape test at all** (AGENTS.md rule 6) - it is one opaque
``torch.library.custom_op`` call.  Layers that do not satisfy the contract keep the
vendor method untouched.

Scope
-----
M == 1 (decode, one sequence) only.  Prefill (M > 1) keeps the vendor GEMM: the
kernel streams a single activation row, and the decode path is the one measured.
"""

from __future__ import annotations

import json
import os

import torch

import tecoops

__all__ = ["weight_stream_gemv_m1", "linear_stream", "install_linear_stream",
           "mapped_libteco_ops"]

# The kernel splits N into 32 contiguous row blocks and flushes each core's whole
# output block with one DMA: that needs 16 B aligned block starts, hence
# N % 256 == 0.  K is walked as a 16-wide fp32 lane chain, hence K % 16 == 0.
_WSG_N_ALIGN = 256
_WSG_K_ALIGN = 16


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
    "gemma4_linear_stream::gemv_m1", mutates_args=("out",), device_types="sdaa"
)
def weight_stream_gemv_m1(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> None:
    tecoops.weight_stream_gemv(x, weight, out)


@weight_stream_gemv_m1.register_fake
def _weight_stream_gemv_m1_fake(x, weight, out):
    return None


@torch.library.custom_op(
    "gemma4_linear_stream::linear_stream",
    mutates_args=(),
    device_types="sdaa",
)
def linear_stream(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Opaque replacement for ``F.linear``.

    The M==1 vs vendor decision lives in the C++ host function of this one
    operator (``api/torch_ext.cpp: linear_stream_torch``).  Nothing about the
    decision is visible to Dynamo, so the operator is a single graph node that is
    *always executed*, for every shape: the runtime shape decides which kernel the
    op launches.  This is what makes the binding survive ``torch.compile`` --
    a Python-level shape test would be folded against the profiling trace
    (M = max_num_batched_tokens = 2048) and the decode step would keep the vendor
    branch.
    """
    return tecoops.linear_stream(x, weight, bias)


@linear_stream.register_fake
def _linear_stream_fake(x, weight, bias=None):
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
    )


def install_linear_stream() -> dict:
    """Bind the streaming linear once, at process initialization.

    ``LinearBase.forward`` calls ``self.quant_method.apply`` which calls
    ``dispatch_unquantized_gemm()`` -> the module-global
    ``default_unquantized_gemm``; replacing that name is the model call-site
    change required by the operator-delivery contract.  The replacement contains
    no branch at all: it forwards to one opaque custom operator.
    """
    import vllm.model_executor.layers.utils as U

    if getattr(U, "_gemma4_linear_stream_patched", False):
        return {"already_patched": True}

    orig = U.default_unquantized_gemm

    def stream_unquantized_gemm(layer, x, weight, bias=None):
        return linear_stream(x, weight, bias)

    U.default_unquantized_gemm = stream_unquantized_gemm
    if U.default_unquantized_gemm is not stream_unquantized_gemm:
        raise RuntimeError("linear_stream binding identity check failed")
    U._gemma4_linear_stream_patched = True

    # Init-time-gated launch census (C++ side, no hot-path Python work).  It proves
    # the M=1 branch really launched the streaming kernel in this process instead of
    # silently falling back to the vendor GEMM.  Inert unless
    # GEMMA4_LINEAR_STATS_DIR is set at process start.
    stats_dir = os.environ.get("GEMMA4_LINEAR_STATS_DIR")
    if stats_dir:
        import atexit

        os.makedirs(stats_dir, exist_ok=True)
        census_path = os.path.join(stats_dir, f"linear-launches-{os.getpid()}.json")

        def _dump_census() -> None:
            keys = ("kernel_launches", "vendor_fallbacks", "fb_not_m1",
                    "fb_dtype", "fb_w_noncontig", "fb_k_mismatch", "fb_align",
                    "fb_other")
            try:
                from tecoops import _torch_ext as _ext

                values = list(_ext.linear_stream_stats())
            except Exception:  # noqa: BLE001
                try:
                    values = list(tecoops.linear_stream_stats())
                except Exception:  # noqa: BLE001
                    values = [None] * len(keys)
            try:
                with open(census_path, "w") as fh:
                    json.dump({"pid": os.getpid(),
                               **dict(zip(keys, values))}, fh, indent=2)
            except OSError:
                pass

        _dump_census()
        atexit.register(_dump_census)

    return {
        "bound": "vllm.model_executor.layers.utils.default_unquantized_gemm",
        "orig": getattr(orig, "__name__", repr(orig)),
        "mapped_libteco_ops": mapped_libteco_ops(),
    }
