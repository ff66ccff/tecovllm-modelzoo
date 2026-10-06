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

"""Optional full-sequence paged-KV SDPA prefill; no import-time device changes."""

from typing import Any, Optional
import torch

def _row_block_indices(block_table: Any, index: int, count: int) -> Any:
    """Extract the physical block ids for one request as a flat int64 vector.

    ``attn_metadata.block_table`` reaches the operator in more than one shape:
    the offline harness hands over ``[num_reqs, max_blocks]``, while the model's
    metadata can carry a per-request leading dimension (``[num_reqs, 1,
    max_blocks]``).  Indexing request ``index`` and flattening works for both;
    assuming a rank would break one of them (observed as
    ``RuntimeError: index_select: Index is supposed to be a vector``).
    """
    row = block_table[index]
    return row.reshape(-1)[:count].to(torch.int64)


def _gather_paged(cache: Any, indices: Any, length: int, num_kv_heads: int, head_dim: int) -> Any:
    """Paged cache [blocks, kv_heads, block, dim] -> packed [length, kv_heads, dim].

    The cache is head-major while a packed attention buffer is token-major, so
    this needs an explicit permute; ``reshape(-1, kv_heads, dim)`` would
    interleave heads with tokens and silently produce wrong values.
    """
    blocks = cache.index_select(0, indices)
    per_head = blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
    return per_head[:, :length, :].transpose(0, 1).contiguous()


def sdpa_varlen_prefill(
    q: Any,
    k: Any,
    v: Any,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    seqused_k: Any = None,
    window_size: Any = None,
    block_table: Any = None,
    out: Any = None,
) -> Any:
    """Variable-length prefill attention via dense SDPA with native GQA.

    Same signature as ``_tecoops_flash_attn_varlen_func`` so a caller can bind
    either implementation once, at process initialization.

    Precondition, enforced fail-closed: for every sequence the queries must span
    the whole sequence (``query_len == seq_len``), i.e. no KV prefix and no
    chunked prefill.  The dense KV passed here must therefore contain the full
    sequence.  Silently returning wrong attention for a chunked prefill would be
    far worse than raising.
    """
    import torch.nn.functional as F

    if window_size is not None:
        raise RuntimeError("SDPA prefill does not support sliding windows")
    if not causal:
        raise RuntimeError("MiniCPM fast prefill requires causal attention")
    total_q = int(q.shape[0])
    num_heads_q = int(q.shape[1])
    num_kv_heads = int(k.shape[1]) if k.dim() == 3 else int(k.shape[1])
    head_dim = int(q.shape[2])
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    result = out if out is not None else torch.empty_like(q)
    num_seqs = int(cu_seqlens_q.shape[0]) - 1

    for index in range(num_seqs):
        q_start = int(cu_seqlens_q[index])
        q_end = int(cu_seqlens_q[index + 1])
        query_len = q_end - q_start
        if query_len <= 0:
            continue

        k_start = int(cu_seqlens_k[index])
        k_end = int(cu_seqlens_k[index + 1])
        if seqused_k is not None:
            seq_len = int(seqused_k[index])
        else:
            seq_len = k_end - k_start

        if query_len != seq_len:
            raise RuntimeError(
                "sdpa_varlen_prefill requires query_len == seq_len per sequence "
                f"(sequence {index}: query_len={query_len}, seq_len={seq_len}); "
                "dense SDPA prefill is not valid for chunked prefill or KV prefixes."
            )

        if head_dim and k.dim() == 4:
            # paged cache: [blocks, kv_heads, block, dim]
            blocks_needed = (seq_len + int(k.shape[2]) - 1) // int(k.shape[2])
            indices = _row_block_indices(block_table, index, blocks_needed)
            kd = _gather_paged(k, indices, seq_len, num_kv_heads, head_dim)
            vd = _gather_paged(v, indices, seq_len, num_kv_heads, head_dim)
        else:
            kd = k[k_start:k_end]
            vd = v[k_start:k_end]

        qb = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        kb = kd.transpose(0, 1).unsqueeze(0)
        vb = vd.transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            qb,
            kb,
            vb,
            is_causal=bool(causal),
            scale=float(softmax_scale),
            enable_gqa=num_heads_q != num_kv_heads,
        )
        result[q_start:q_end] = attended.squeeze(0).transpose(0, 1).to(result.dtype)

    return result


# Explicit alternate entry point for A/B measurement and for the opt-in fast
# profile.  Importing this module never switches the default path.
sdaa_flash_attn_varlen_sdpa = sdpa_varlen_prefill


def make_opaque_sdpa_prefill():
    """Return the dense-SDPA prefill as a compile-safe callable.

    The parameter list is deliberately identical to the *opaque* official
    binding (``flash_attn_varlen_func``), i.e. 12 parameters with no
    ``window_size`` slot: the model's call site passes those 12 positionally,
    so an extra parameter shifts ``block_table`` into it and leaves ``out``
    None - which silently discards the attention result and produced token
    salad in the service.  The eager tecoops wrapper does have window_size; this
    adapter must not, because it is bound at the same call site.
    """
    from custom_ops.compile_safe import register_sdpa_prefill

    opaque = register_sdpa_prefill(sdpa_varlen_prefill)
    if opaque is None:
        raise RuntimeError("Failed to register opaque SDPA prefill")

    def bound(
        q: Any,
        k: Any,
        v: Any,
        cu_seqlens_q: Any,
        cu_seqlens_k: Any,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        seqused_k: Any = None,
        block_table: Any = None,
        out: Any = None,
    ) -> Any:
        if out is None:
            out = torch.empty_like(q)
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        opaque(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            block_table,
            out,
            int(max_seqlen_q),
            int(max_seqlen_k),
            float(softmax_scale),
            bool(causal),
        )
        return out

    return bound
