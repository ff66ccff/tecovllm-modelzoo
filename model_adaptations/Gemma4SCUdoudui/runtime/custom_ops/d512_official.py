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

"""Initialization-bound D512 global decode on the **delivered** tecoops operator (default tier).

cycle-27 / task-9 (AGENTS §2.3 delivery-runnability): ``tecoops.flash_attn_varlen_func`` (the
D256/D512 paged varlen flash attention delivered by teco-ops main) was previously reachable only
through the isolated extension selected by ``GEMMA_D512_OFFICIAL_EXTENSION``.  That variable is
unset in the delivery configuration, so ``OFFICIAL_DECODE`` was ``None`` and the 8 head_dim-512
global layers stayed on the in-repo math fallback (``_decode_math`` ->
``sdpa_decode_attention_math`` -> ``_attend_math``).

This module binds the delivered operator itself, at **import / construction time**, as the
**default** tier.  The selection happens once, in ``BlockAttentionImpl.__init__`` ->
``bind_attention_impls`` -> ``bind_global_decode``; the forward hot path contains no backend
branch, no fallback and no ``getenv`` (AGENTS.md rule 6).

Fail-closed: the process must map exactly one ``libteco_ops.so`` (optionally pinned by
``GEMMA4_D512_EXPECT_LIB``) and ``tecoops`` must export ``flash_attn_varlen_func``; the legacy
isolated-extension variables must be unset, so the operator identity can never be ambiguous.
Nothing else in the attention path changes: prefill, the D256 sliding layers, cache writes,
RMSNorm and the linear stream are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import tecoops


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapped_libteco_ops():
    """Realpaths of every ``libteco_ops.so`` mapped into this process."""
    out = set()
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.split()[-1]
                if path.endswith("libteco_ops.so"):
                    out.add(str(Path(path).resolve()))
    except OSError:
        pass
    return sorted(out)


def _fail_closed_library_identity():
    libs = _mapped_libteco_ops()
    if not libs:
        raise RuntimeError(
            "D512 delivered decode: no libteco_ops.so mapped into this process")
    expected = os.environ.get("GEMMA4_D512_EXPECT_LIB")
    if expected:
        if libs != [str(Path(expected).resolve())]:
            raise RuntimeError(
                "D512 delivered decode FAIL-CLOSED: mapped %r != expected %r"
                % (libs, expected))
    elif len(libs) != 1:
        raise RuntimeError(
            "D512 delivered decode FAIL-CLOSED: expected exactly one libteco_ops.so, "
            "mapped %r" % (libs,))
    return libs


if os.environ.get("GEMMA_D512_OFFICIAL_EXTENSION") or os.environ.get("GEMMA_D512_OFFICIAL_SHA256"):
    raise RuntimeError(
        "D512 delivered decode: GEMMA_D512_OFFICIAL_EXTENSION/_SHA256 select the isolated "
        "extension tier; unset them when the delivered tecoops binding is the default")

if not hasattr(tecoops, "flash_attn_varlen_func"):
    raise RuntimeError(
        "D512 delivered decode requires tecoops.flash_attn_varlen_func; rebuild/stage a "
        "libteco_ops.so and _torch_ext that export it")

_MAPPED_LIBS = _fail_closed_library_identity()
print("[gemma-d512-delivered] LOAD " + json.dumps({
    "tecoops_file": getattr(tecoops, "__file__", None),
    "mapped_libteco_ops": _MAPPED_LIBS,
    "mapped_sha256": [_sha256(p) for p in _MAPPED_LIBS],
    "kernel": "tecoops.flash_attn_varlen_func",
}), file=sys.stderr, flush=True)


def register_decode(kernel, namespace="gemma_d512_official"):
    execution_recorded = False

    @torch.library.custom_op(namespace + "::decode", mutates_args=("out",))
    def decode(
        query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
        query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
        block_table: torch.Tensor, max_seq_len: int, scale: float,
        out: torch.Tensor,
    ) -> None:
        nonlocal execution_recorded
        n = query.shape[0]
        if query.dtype != torch.float16 or tuple(query.shape[1:]) != (8, 512):
            raise RuntimeError("D512 decode requires FP16 [N,8,512] Q")
        if key_cache.ndim != 4 or tuple(key_cache.shape[1:]) != (1, 32, 512):
            raise RuntimeError("D512 decode requires HND [blocks,1,32,512] cache")
        if value_cache.shape != key_cache.shape:
            raise RuntimeError("D512 K/V cache shapes differ")
        if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise RuntimeError("D512 cache dtype differs from Q")
        if query_start_loc.shape != (n + 1,) or seq_lens.shape != (n,):
            raise RuntimeError("decode requires one query token per request")
        if block_table.ndim != 2 or block_table.shape[0] != n:
            raise RuntimeError("decode requires two-dimensional request block table")
        if any(t.dtype != torch.int32 for t in (query_start_loc, seq_lens, block_table)):
            raise RuntimeError("D512 metadata must be int32")
        if any(t.device != query.device for t in (
            key_cache, value_cache, query_start_loc, seq_lens, block_table, out,
        )):
            raise RuntimeError("D512 inputs and output must share a device")
        if not all(t.is_contiguous() for t in (key_cache, value_cache, out)):
            raise RuntimeError("D512 caches and output must be contiguous")
        if out.shape != query.shape or out.dtype != query.dtype or scale != 1.0:
            raise RuntimeError("D512 output/scale contract mismatch")
        placeholder = torch.Tensor()
        kernel(
            query.contiguous(), key_cache, value_cache,
            1, query_start_loc.contiguous(), int(max_seq_len), placeholder,
            seq_lens.contiguous(), float(scale), True, placeholder,
            block_table.contiguous(), False, out,
        )
        if not execution_recorded:
            stream = (str(torch.sdaa.current_stream(query.device))
                      if query.device.type == "sdaa" else "cpu-mock")
            print("[gemma-d512-official] EXECUTE " + json.dumps({
                "pid": os.getpid(), "query_shape": list(query.shape),
                "key_cache_shape": list(key_cache.shape),
                "value_cache_shape": list(value_cache.shape),
                "out_shape": list(out.shape), "dtype": str(query.dtype),
                "device": str(query.device), "scale": scale, "stream": stream,
            }), file=sys.stderr, flush=True)
            execution_recorded = True

    @decode.register_fake
    def fake(query, key_cache, value_cache, query_start_loc, seq_lens,
             block_table, max_seq_len, scale, out):
        return None

    return decode


OFFICIAL_DECODE = register_decode(tecoops.flash_attn_varlen_func)


def decode_forward(self, out_view, query, key_cache, value_cache,
                   attn_metadata, num_actual_tokens, num_blocks_stride):
    n = num_actual_tokens
    OFFICIAL_DECODE(
        query[:n].contiguous(), key_cache, value_cache,
        attn_metadata.query_start_loc[:n + 1], attn_metadata.seq_lens[:n],
        attn_metadata.block_table[:n], int(attn_metadata.max_model_len),
        self.scale, out_view,
    )


def bind_global_decode(impl):
    if OFFICIAL_DECODE is None or impl.head_size != 512:
        return
    if (getattr(impl, "_gemma4_window_size", None) is not None
            or impl.num_heads != 8 or impl.num_kv_heads != 1
            or impl.logits_soft_cap != 0 or impl.sinks is not None
            or impl.kv_cache_dtype != "auto" or impl.scale != 1.0):
        raise RuntimeError("unsupported D512 global attention construction")
    impl._gemma4_decode_fn = decode_forward
    print("[gemma-d512-official] BIND H8/KV1/D512/global/scale1/decode",
          file=sys.stderr, flush=True)
