"""SDAA BlockAttention operator framework adaptation.

Provides in-repo implementation for BlockAttentionImpl.forward to bridge
vLLM V1 attention execution to vendor C++ kernels:
- torch.ops._C_cache_ops.reshape_and_cache_flash
- torch.ops._C_sdaa.block_attention

Prefill implementation selection (process-initialization binding):
  INTERNVL_PREFILL_ATTN=official (default) -> official tecoops C++ flash ABI  [delivery form]
  INTERNVL_PREFILL_ATTN=sdpa               -> previous torch-SDPA path          [rollback]
The variable is read exactly once, at module import.  The forward path below contains no
backend/fallback/A-B branch and no getenv() call (AGENTS.md rule 6).
"""

import os
from typing import Optional
import torch

from custom_ops.reshape_and_cache.op import sdaa_reshape_and_cache

# ---------------------------------------------------------------------------------------------
# Bound once, at process initialization.  `_prefill_attention` has a single signature for both
# implementations so the forward can call it unconditionally.
# ---------------------------------------------------------------------------------------------
_PREFILL_IMPL = os.environ.get("INTERNVL_PREFILL_ATTN", "official").strip().lower()

if _PREFILL_IMPL == "official":
    from custom_ops.flash_attn_varlen.op import sdaa_flash_attn_varlen_func as _official_prefill

    _PREFILL_IMPL_SOURCE = "custom_ops.flash_attn_varlen.op:sdaa_flash_attn_varlen_func"

    def _prefill_attention(query, key_cache, value_cache, attn_metadata, scale, out):
        """Official paged-cache flash ABI (eligible delivery form); fills `out` in place."""
        _official_prefill(
            query.contiguous(),
            key_cache,
            value_cache,
            attn_metadata.query_start_loc,
            attn_metadata.query_start_loc,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            softmax_scale=scale,
            causal=attn_metadata.causal,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            out=out,
        )

elif _PREFILL_IMPL == "sdpa":
    from custom_ops.prefill_attention.op import sdpa_prefill_attention as _sdpa_prefill

    _PREFILL_IMPL_SOURCE = "custom_ops.prefill_attention.op:sdpa_prefill_attention"

    def _prefill_attention(query, key_cache, value_cache, attn_metadata, scale, out):
        """Rollback path: paged-cache gather + torch SDPA (NOT eligible for delivery)."""
        out.copy_(_sdpa_prefill(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            scale,
            attn_metadata.causal,
        ))

else:
    raise RuntimeError(
        "INTERNVL_PREFILL_ATTN must be 'official' or 'sdpa', got %r" % _PREFILL_IMPL)


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
        sdaa_reshape_and_cache(
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
        # Prefill / mixed batch: one implementation, bound at import time (see _prefill_attention).
        _prefill_attention(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata,
            self.scale,
            out_view,
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
