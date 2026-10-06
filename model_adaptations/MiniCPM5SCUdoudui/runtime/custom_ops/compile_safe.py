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

"""FakeTensor-safe opaque registrations for the official ``tecoops`` kernels.

Why this module exists
----------------------
``torch.compiler.allow_in_graph(pybind_fn)`` does **not** make a raw pybind11
extension function opaque to ``torch.compile``.  Dynamo inlines it, dispatches
it under FakeTensor and the kernel dies on ``data_ptr()`` with::

    RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor,
    FunctionalTensor). ... it is likely that we are erroneously tracing into a
    custom kernel. To fix this, please wrap the custom kernel into an opaque
    custom op.

That is a startup blocker for the official service contract, which does *not*
pass ``--enforce-eager``.  The kernels are therefore wrapped here as
``torch.library`` custom ops with explicit mutation annotations, exactly as the
vendor message prescribes.  Registration happens once, during process
initialization; vendor site-packages are never modified.

The opaque signatures deliberately mirror the eager wrappers in
``custom_ops/*/op.py`` so the call site in ``BlockAttentionImpl.forward`` can
bind either implementation once, at import time, with no hot-path branch.
"""

import torch

from custom_ops.receipt import bind_official_api

_NAMESPACE = "tecoops_overlay"
_REGISTERED: dict[str, object] | None = None


class OfficialOps:
    """Opaque, compile-safe bindings of the official operator set."""

    __slots__ = ("rms_norm", "rms_norm_add", "reshape_and_cache", "flash_attn_varlen_func")

    def __init__(self, rms_norm, rms_norm_add, reshape_and_cache, flash_attn_varlen_func) -> None:
        self.rms_norm = rms_norm
        self.rms_norm_add = rms_norm_add
        self.reshape_and_cache = reshape_and_cache
        self.flash_attn_varlen_func = flash_attn_varlen_func


def official_ops(tecoops_module=None) -> "OfficialOps | None":
    """Register (once) and return the opaque bindings, or ``None`` if unbound."""
    global _REGISTERED
    if _REGISTERED is not None:
        return OfficialOps(**_REGISTERED)
    if tecoops_module is None:
        return None

    rms_norm_kernel = getattr(tecoops_module, "rms_norm", None)
    reshape_kernel = getattr(tecoops_module, "reshape_and_cache", None)
    flash_kernel = getattr(tecoops_module, "flash_attn_varlen_func", None)
    if rms_norm_kernel is None or reshape_kernel is None or flash_kernel is None:
        return None

    # Bind accounting inside the opaque implementation so tracing never
    # records a raw vendor invocation. Disabled accounting returns the ABI
    # object unchanged; no runtime backend selection is introduced.
    rms_norm_kernel = bind_official_api(rms_norm_kernel, "tecoops.rms_norm")
    reshape_kernel = bind_official_api(reshape_kernel, "tecoops.reshape_and_cache")
    flash_kernel = bind_official_api(flash_kernel, "tecoops.flash_attn_varlen_func")

    @torch.library.custom_op(f"{_NAMESPACE}::rms_norm", mutates_args=("out",))
    def rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        eps: float,
    ) -> None:
        rms_norm_kernel(
            x.reshape(-1, x.shape[-1]),
            weight,
            None,
            out.reshape(-1, out.shape[-1]),
            None,
            float(eps),
        )

    @rms_norm.register_fake
    def _rms_norm_fake(x, weight, out, eps):
        return None

    @torch.library.custom_op(f"{_NAMESPACE}::rms_norm_add", mutates_args=("out", "residual_out"))
    def rms_norm_add(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        eps: float,
    ) -> None:
        x2 = x.reshape(-1, x.shape[-1])
        rms_norm_kernel(
            x2,
            weight,
            residual.reshape_as(x2),
            out.reshape(-1, out.shape[-1]),
            residual_out.reshape(-1, residual_out.shape[-1]),
            float(eps),
        )

    @rms_norm_add.register_fake
    def _rms_norm_add_fake(x, weight, residual, out, residual_out, eps):
        return None

    @torch.library.custom_op(
        f"{_NAMESPACE}::reshape_and_cache",
        mutates_args=("key_cache", "value_cache"),
    )
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # The official C++ ABI reads key/value as dense half buffers and ignores
        # PyTorch stride metadata, so non-contiguous fused-QKV views are
        # materialized at this boundary (same contract as the eager wrapper).
        reshape_kernel(
            key.contiguous(),
            value.contiguous(),
            slot_mapping,
            key_cache,
            value_cache,
        )

    @reshape_and_cache.register_fake
    def _reshape_and_cache_fake(key, value, key_cache, value_cache, slot_mapping):
        return None

    @torch.library.custom_op(f"{_NAMESPACE}::flash_attn_varlen_func", mutates_args=("out",))
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        placeholder = torch.Tensor()
        flash_kernel(
            q,
            k,
            v,
            int(max_seqlen_q),
            cu_seqlens_q,
            int(max_seqlen_k),
            placeholder,
            seqused_k,
            float(softmax_scale),
            bool(causal),
            placeholder,
            block_table,
            False,
            out,
        )

    @flash_attn_varlen_func.register_fake
    def _flash_attn_varlen_fake(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale,
        causal, seqused_k, block_table, out,
    ):
        return None

    _REGISTERED = {
        "rms_norm": rms_norm,
        "rms_norm_add": rms_norm_add,
        "reshape_and_cache": reshape_and_cache,
        "flash_attn_varlen_func": flash_attn_varlen_func,
    }
    return OfficialOps(**_REGISTERED)


def registered() -> bool:
    return _REGISTERED is not None


# ---------------------------------------------------------------------------
# Dense-SDPA prefill as an opaque op
# ---------------------------------------------------------------------------
_SDPA_REGISTERED: object | None = None


def register_sdpa_prefill(kernel):
    """Register the dense-SDPA prefill path as an opaque custom op.

    The dense path slices per sequence using Python ``int()`` values read off
    the sequence-length tensors, so Dynamo must not trace into it - exactly the
    same reason the tecoops kernels are wrapped.  Declaring ``out`` as mutated
    keeps FakeTensor propagation honest.

    Registration errors propagate; no eager fallback is permitted.
    """
    global _SDPA_REGISTERED
    if _SDPA_REGISTERED is not None:
        return _SDPA_REGISTERED

    @torch.library.custom_op(f"{_NAMESPACE}::sdpa_varlen_prefill", mutates_args=("out",))
    def sdpa_varlen_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool,
    ) -> None:
        kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            float(softmax_scale),
            bool(causal),
            seqused_k,
            None,
            block_table,
            out,
        )

    @sdpa_varlen_prefill.register_fake
    def _sdpa_varlen_prefill_fake(
        q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k, block_table, out,
        max_seqlen_q, max_seqlen_k, softmax_scale, causal,
    ):
        return None

    _SDPA_REGISTERED = sdpa_varlen_prefill
    return _SDPA_REGISTERED


def sdpa_registered() -> bool:
    return _SDPA_REGISTERED is not None
