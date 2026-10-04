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

"""Hy-MT2 binding for existing SDAA cache, SDPA and BlockAttention kernels.

The HND layout and scheduling contract follow the validated InternVL binding;
Hy TP1 has the same local Hq16/Hkv4/D128 geometry. No new kernel is introduced.
Chunked prefill is rejected; run.sh disables it during initialization.
"""

import functools
import math
import torch
import torch.nn.functional as F
import tecoops


def _prefill(query, key_cache, value_cache, metadata, scale):
    out = torch.empty_like(query)
    block_size = key_cache.shape[2]
    for i in range(metadata.query_start_loc.numel() - 1):
        start = int(metadata.query_start_loc[i])
        end = int(metadata.query_start_loc[i + 1])
        q_len = end - start
        kv_len = int(metadata.seq_lens[i])
        if q_len <= 0 or kv_len <= 0:
            continue
        if metadata.causal and q_len > 1 and q_len != kv_len:
            raise RuntimeError("Hy-MT2 requires --no-enable-chunked-prefill")
        logical = torch.arange(kv_len, device=key_cache.device)
        blocks = metadata.block_table[i][logical // block_size]
        offsets = logical % block_size
        k = key_cache[blocks, :, offsets, :].transpose(0, 1).unsqueeze(0)
        v = value_cache[blocks, :, offsets, :].transpose(0, 1).unsqueeze(0)
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        result = F.scaled_dot_product_attention(
            q, k, v, is_causal=bool(metadata.causal and q_len > 1),
            scale=scale, enable_gqa=True,
        )
        out[start:end] = result.squeeze(0).transpose(0, 1)
    return out


def attention_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                      output=None, output_scale=None, output_block_scale=None):
    # Dummy runs have no real attention/cache metadata.
    if attn_metadata is None:
        if output is not None:
            return output.fill_(0)
        return torch.zeros((query.shape[0], self.num_heads * self.head_size),
                           dtype=query.dtype, device=query.device)
    n = attn_metadata.num_actual_tokens
    if n == 0:
        return output
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    if key is not None and value is not None and attn_metadata.slot_mapping is not None:
        tecoops.reshape_and_cache(key.contiguous(), value.contiguous(),
                                 attn_metadata.slot_mapping, key_cache, value_cache)
    if output is None:
        output = torch.empty((n, self.num_heads, self.head_size),
                             dtype=query.dtype, device=query.device)
    out_view = output[:n].view(n, self.num_heads, self.head_size)
    # Scheduling distinguishes prefill/mixed and single-token decode phases;
    # each phase has a fixed vendor implementation, without backend selection.
    if attn_metadata.max_query_len > 1:
        out_view.copy_(_prefill(query[:n], key_cache, value_cache, attn_metadata, self.scale))
    else:
        torch.ops._C_sdaa.block_attention(
            out_view, key_cache, value_cache, query[:n].contiguous(),
            attn_metadata.seq_lens_pre_cache, attn_metadata.seq_lens_pre_cache_cpu,
            attn_metadata.seq_lens_prefill, attn_metadata.seq_lens_prefill_cpu,
            attn_metadata.seq_lens_decode, attn_metadata.query_start_loc,
            attn_metadata.block_table, attn_metadata.max_model_len,
            attn_metadata.max_prefill_len, attn_metadata.max_decode_len,
            attn_metadata.version, window_size_left=self.sliding_window[0],
            window_size_right=self.sliding_window[1], sinks=self.sinks,
            block_stride=key_cache.stride(0), is_q_only=True,
        )
    return output


def install_attention() -> None:
    from vllm_sdaa.attention.block_attn import BlockAttentionImpl
    if not hasattr(tecoops, "reshape_and_cache"):
        raise RuntimeError("Hy-MT2 requires tecoops.reshape_and_cache")
    if not hasattr(torch.ops._C_sdaa, "block_attention"):
        raise RuntimeError("Hy-MT2 requires the vendor BlockAttention kernel")
    torch.backends.sdaa.enable_flash_sdp(True)
    original_init = BlockAttentionImpl.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if (self.num_heads, self.num_kv_heads, self.head_size) != (16, 4, 128):
            raise RuntimeError("Hy-MT2 attention binding requires TP1 Hq16/Hkv4/D128")
        if not math.isclose(self.scale, 128 ** -.5, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Hy-MT2 attention binding requires default scale")
        if self.kv_cache_dtype != "auto" or self.kv_sharing_target_layer_name is not None:
            raise RuntimeError("Hy-MT2 requires automatic FP16 cache without KV sharing")
        if self.sliding_window != (-1, -1) or self.logits_soft_cap != 0 or self.sinks is not None:
            raise RuntimeError("Hy-MT2 attention binding requires global attention without softcap/sinks")

    BlockAttentionImpl.__init__ = init
    BlockAttentionImpl.forward = attention_forward
    if BlockAttentionImpl.forward is not attention_forward:
        raise RuntimeError("Hy-MT2 attention binding failed")
