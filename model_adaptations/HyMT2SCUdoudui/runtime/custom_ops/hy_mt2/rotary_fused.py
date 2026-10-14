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

"""Hy-MT2-1.8B rotary embedding: bind the team's fused tecoops kernel at start-up.

The delivered FP16/TP1 vLLM export runs HunYuanDenseV1 attention through
``vllm/model_executor/models/hunyuan_v1.py`` (``get_rope(..., is_neox_style=True)``),
whose native path is ``RotaryEmbedding.forward_static`` + ``ApplyRotaryEmb.forward_static``
-- 17 device kernel launches per layer on SDAA (1 ``index_select``, 8 fp16
multiplies, 2 subtracts, 2 adds, 4 ``cat``), 544 per forward across 32 layers,
17 408 per 32-token request (measured device-free in ``hy_kernelcount.py``).

Two entry points, both used only during process initialization:

* ``official_rotary_embedding(tecoops_module)`` -- register the kernel once as an
  opaque ``torch.library`` custom op (FakeTensor / ``torch.compile`` safe);
* ``bind_instance(instance, op)`` -- bind ``RotaryEmbeddingBase._forward_method``
  on an instance whose contract the kernel covers.

The bound hot path is two ``torch.empty`` allocations plus one opaque custom-op
call.  No branch selects a backend, no ``getenv`` is called, and there is no
fallback: whether an instance is bound is decided once, at construction, from the
frozen contract in ``contract_ok`` (fp16/fp32 cos/sin cache that the closure
normalises, NeoX half-split layout, full ``rotary_dim``).  Instances outside the
contract keep vLLM's native implementation and are never touched at run time.

Dtype scope: the delivered export is FP16 (``run.sh --dtype float16``), so only
the fp16 kernel entry is used.  A bf16 entry is NOT implemented -- the SDAA slave
SIMD header exposes no bf16 type or conversion, and the delivered runtime never
reaches that dtype; running the native bf16 config would need its own verification.

The bound closure requires ``key`` (the HunYuan q/k contract).  A call with
``key=None`` fails loudly at the pybind boundary instead of silently skipping the
rotation.
"""
from __future__ import annotations

import importlib

import torch

_NAMESPACE = "hy_mt2_rotary_embedding"
_OP_NAME = f"{_NAMESPACE}::rotary_embedding"
_BOUND_OP = None


def official_rotary_embedding(tecoops_module):
    """Register (once) and return the opaque rotary-embedding op, or ``None``."""
    global _BOUND_OP
    if _BOUND_OP is not None:
        return _BOUND_OP
    kernel = getattr(tecoops_module, "rotary_embedding", None)
    if kernel is None:
        return None

    @torch.library.custom_op(_OP_NAME, mutates_args=("q_out", "k_out"))
    def rotary_embedding(
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        is_neox: bool,
    ) -> None:
        kernel(q, k, cos_sin_cache, positions, q_out, k_out, bool(is_neox))

    @rotary_embedding.register_fake
    def _rotary_embedding_fake(q, k, cos_sin_cache, positions, q_out, k_out, is_neox):
        return None

    _BOUND_OP = rotary_embedding
    return _BOUND_OP


def contract_ok(instance) -> bool:
    """True when the delivered kernel covers this rotary-embedding instance."""
    if not bool(getattr(instance, "is_neox_style", False)):
        return False
    head_size = int(getattr(instance, "head_size", 0))
    if head_size <= 0 or head_size % 2:
        return False
    if int(getattr(instance, "rotary_dim", 0)) != head_size:
        return False
    cache = getattr(instance, "cos_sin_cache", None)
    if cache is None or not isinstance(cache, torch.Tensor):
        return False
    # `get_rope` is called with dtype=None at model init, so the cache is float32
    # until the reference path normalises it to the query dtype on the first
    # forward (_match_cos_sin_cache_dtype).  Accept fp32 here and let the bound
    # closure perform that same normalisation; the kernel then enforces fp16.
    if cache.dtype not in (torch.float16, torch.float32):
        return False
    if not cache.is_contiguous():
        return False
    return True


def _make_forward(instance, op, count=False):
    """Build the branch-free hot path for one bound instance."""
    if count:

        def forward(positions, query, key):
            cos_sin_cache = instance._match_cos_sin_cache_dtype(query)
            q_out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
            k_out = torch.empty(key.shape, dtype=key.dtype, device=key.device)
            instance._hy_rotary_calls += 1
            op(query, key, cos_sin_cache, positions.reshape(-1), q_out, k_out, True)
            return q_out, k_out

        forward._hy_rotary_fused = True
        return forward

    def forward(positions, query, key):
        cos_sin_cache = instance._match_cos_sin_cache_dtype(query)
        q_out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        k_out = torch.empty(key.shape, dtype=key.dtype, device=key.device)
        op(query, key, cos_sin_cache, positions.reshape(-1), q_out, k_out, True)
        return q_out, k_out

    forward._hy_rotary_fused = True
    return forward


def bind_instance(instance, op, count=False) -> bool:
    """Bind the kernel on one instance.  Idempotent; True when (already) bound."""
    if getattr(instance, "_hy_rotary_embedding_bound", False):
        return True
    if not contract_ok(instance):
        return False
    instance._forward_method = _make_forward(instance, op, count=count)
    instance._hy_rotary_embedding_bound = True
    instance._hy_rotary_calls = 0
    return True


_ROTARY_BASE_CANDIDATES = (
    "vllm.model_executor.layers.rotary_embedding.base",
)


def resolve_rotary_base(rope_module, import_module=None):
    """Locate ``RotaryEmbeddingBase`` at bind time.

    vLLM's ``rotary_embedding/__init__.py`` re-exports ``RotaryEmbedding`` but not
    always ``RotaryEmbeddingBase``, so looking only at the package attribute fails
    closed in the real process.  Resolution happens once, at install time; nothing
    on the forward path re-resolves or branches.
    """
    if import_module is None:
        import_module = importlib.import_module
    target = getattr(rope_module, "RotaryEmbeddingBase", None)
    if target is not None:
        return target
    sub = getattr(rope_module, "base", None)
    if sub is not None:
        target = getattr(sub, "RotaryEmbeddingBase", None)
        if target is not None:
            return target
    for name in _ROTARY_BASE_CANDIDATES:
        try:
            module = import_module(name)
        except Exception:
            continue
        target = getattr(module, "RotaryEmbeddingBase", None)
        if target is not None:
            return target
    return None
