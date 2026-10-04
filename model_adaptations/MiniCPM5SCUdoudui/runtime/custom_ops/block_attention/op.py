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

"""SDAA BlockAttention operator framework adaptation.

Provides in-repo implementation for BlockAttentionImpl.forward to bridge
vLLM V1 attention execution to vendor C++ kernels:
- torch.ops._C_cache_ops.reshape_and_cache_flash
- torch.ops._C_sdaa.block_attention
"""

from typing import Optional
import torch

from custom_ops.compile_safe import official_ops

_OFFICIAL_OPS = official_ops()
if _OFFICIAL_OPS is None:
    raise RuntimeError("Official compile-safe bindings must be registered before BlockAttention import")
_reshape_and_cache = _OFFICIAL_OPS.reshape_and_cache
_flash_attn_varlen = _OFFICIAL_OPS.flash_attn_varlen_func


def sdaa_block_attention_forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    output_block_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute block attention forward on SDAA accelerator.

    Zero runtime branching in hot path: bindings are established at process
    initialization, no fallback or getenv calls per forward iteration.
    """
    if attn_metadata is None:
        # Profiling / dummy run
        if output is not None:
            return output.fill_(0)
        return torch.zeros(
            (query.shape[0], self.num_heads * self.head_size),
            dtype=query.dtype,
            device=query.device,
        )

    num_actual_tokens = attn_metadata.num_actual_tokens
    if num_actual_tokens == 0:
        return output

    key_cache, value_cache = kv_cache[0], kv_cache[1]
    num_blocks_stride = key_cache.stride(0)

    # Cache new key/value entries into paged KV blocks
    if key is not None and value is not None and attn_metadata.slot_mapping is not None:
        _reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
        )

    if output is None:
        output = torch.empty(
            (num_actual_tokens, self.num_heads, self.head_size),
            dtype=query.dtype,
            device=query.device,
        )

    out_view = output[:num_actual_tokens]
    if out_view.ndim == 2:
        out_view = out_view.view(-1, self.num_heads, self.head_size)

    if attn_metadata.max_query_len > 1:
        # Prefill uses the official teco-ops paged-cache flash ABI.
        _flash_attn_varlen(
            query[:num_actual_tokens].contiguous(),
            key_cache,
            value_cache,
            attn_metadata.query_start_loc,
            attn_metadata.query_start_loc,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            out=out_view,
        )
    else:
        # Decode is a single-token path; retain the vendor BlockAttention
        # kernel because the official flash ABI is a prefill contract and is
        # not an efficient decode implementation.
        torch.ops._C_sdaa.block_attention(
            out_view,
            key_cache,
            value_cache,
            query[:num_actual_tokens].contiguous(),
            attn_metadata.seq_lens_pre_cache,
            attn_metadata.seq_lens_pre_cache_cpu,
            attn_metadata.seq_lens_prefill,
            attn_metadata.seq_lens_prefill_cpu,
            attn_metadata.seq_lens_decode,
            attn_metadata.query_start_loc,
            attn_metadata.block_table,
            attn_metadata.max_model_len,
            attn_metadata.max_prefill_len,
            attn_metadata.max_decode_len,
            attn_metadata.version,
            window_size_left=self.sliding_window[0],
            window_size_right=self.sliding_window[1],
            sinks=self.sinks,
            block_stride=num_blocks_stride,
            is_q_only=True,
        )

    return output
