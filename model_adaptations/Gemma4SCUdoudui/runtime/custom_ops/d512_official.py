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
import struct
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


# ---------------------------------------------------------------------------
# Fail-closed capability probe on the ACTUAL specialization symbol (c22 L17).
#
# `hasattr(tecoops, "flash_attn_varlen_func")` is NOT a capability probe: it is a Python
# attribute of `_torch_ext`, so an **older** build that still exports the generic
# `teco_slave_flash_attention_half` (ALGO0) also satisfies it -- while lacking
# `teco_slave_flash_attention_half_d512`, which is the branch
# `findFlashAttentionBranch()` selects for `size_per_head == 512`
# (teco/ual/ops/flash_attention/find_flash_attention.cpp: head_dim 512 -> algo 2).  With such a
# library the D512 layers silently take the generic D128 kernel, which is the fail-open this
# probe closes.
#
# The specialization is a device stub compiled into `libteco_ops.so`, so its presence is
# decidable from the library's dynamic symbol table -- which is an **ELF, not a Python**
# question, and is therefore independent of which `tecoops` package happens to be imported.
# This runs once at import (process init) time: it adds no branch to the forward hot path, no
# `getenv` on the hot path, and no new runtime dependency (stdlib `struct` only).  Every parse
# failure is a RuntimeError: "cannot decide" is treated exactly like "absent".
#
# D256 is deliberately NOT required here.  This module binds the D512 global layers only; the
# D256 sliding layers are pinned to the in-repo math path one level up, in
# `custom_ops/block_attention/op.py: base_math = impl_self.head_size >= FLASH_MAX_HEAD_DIM`,
# which never calls the delivered kernel for head_size 256.  Requiring the D256 symbol would
# assert a capability this binding does not use.
_D512_SYMBOL = "teco_slave_flash_attention_half_d512"


def _dynsym_specializations(path, symbols):
    """True for each symbol found in the ELF **dynamic** symbol table of ``path``."""
    for symbol in symbols:
        if not symbol or "\x00" in symbol or len(symbol) > 255:
            raise RuntimeError("D512 delivered decode: bad symbol name %r" % (symbol,))
    found = [False] * len(symbols)
    try:
        with open(path, "rb") as fh:
            ident = fh.read(16)
            if len(ident) != 16 or ident[:4] != b"\x7fELF":
                raise RuntimeError("not an ELF file: %r" % (path,))
            if ident[4] != 2:                     # EI_CLASS: ELFCLASS64
                raise RuntimeError("not a 64-bit ELF library: %r" % (path,))
            if ident[5] != 1:                     # EI_DATA: ELFDATA2LSB
                raise RuntimeError("not a little-endian ELF library: %r" % (path,))
            fh.seek(0)
            header = fh.read(64)
            if len(header) != 64:
                raise RuntimeError("truncated ELF header: %r" % (path,))
            e_shoff, = struct.unpack_from("<Q", header, 0x28)
            e_shentsize, = struct.unpack_from("<H", header, 0x3a)   # Ehdr64: 0x28 shoff
            e_shnum, = struct.unpack_from("<H", header, 0x3c)       #         0x3a shentsize
            e_shstrndx, = struct.unpack_from("<H", header, 0x3e)    #         0x3c shnum / 0x3e shstrndx
            if not e_shoff or not e_shnum:
                raise RuntimeError(
                    "ELF section table absent; the library identity cannot be verified: %r"
                    % (path,))
            fh.seek(e_shoff)
            table = fh.read(e_shentsize * e_shnum)
            if len(table) != e_shentsize * e_shnum:
                raise RuntimeError("truncated ELF section table: %r" % (path,))
            sections = []
            for i in range(e_shnum):
                fields = struct.unpack_from("<IIQQQQIIQQ", table, i * e_shentsize)
                sections.append((fields[0], fields[5], fields[4]))    # sh_name, sh_size, sh_offset
            if e_shstrndx == 0 or e_shstrndx >= e_shnum:
                raise RuntimeError("ELF has no section-name table: %r" % (path,))
            shstr_size, shstr_off = sections[e_shstrndx][1:]
            fh.seek(shstr_off)
            shstr = fh.read(shstr_size)
            if len(shstr) != shstr_size:
                raise RuntimeError("truncated section-name table: %r" % (path,))
            dynsym = dynstr = dynstr_size = None
            for sh_name, size, offset in sections:
                end = shstr.find(b"\x00", sh_name)
                if end < 0:
                    raise RuntimeError("malformed .shstrtab entry: %r" % (path,))
                name = shstr[sh_name:end].decode("ascii", "replace")
                if name == ".dynsym":
                    dynsym, dynsym_size = offset, size
                elif name == ".dynstr":
                    dynstr, dynstr_size = offset, size
            if dynsym is None or dynstr is None:
                raise RuntimeError(
                    "ELF has no .dynsym/.dynstr; the library identity cannot be verified: %r"
                    % (path,))
            fh.seek(dynstr)
            strings = fh.read(dynstr_size)
            if len(strings) != dynstr_size:
                raise RuntimeError("truncated .dynstr: %r" % (path,))
            fh.seek(dynsym)
            entries = fh.read(dynsym_size)
            if len(entries) != dynsym_size:
                raise RuntimeError("truncated .dynsym: %r" % (path,))
            entry_size, count = 24, dynsym_size // 24
            if entry_size * count != dynsym_size:
                raise RuntimeError("malformed .dynsym size: %r" % (path,))
            for i in range(count):
                base = i * entry_size
                st_name, = struct.unpack_from("<I", entries, base)
                if st_name == 0:
                    continue
                end = strings.find(b"\x00", st_name)
                if end < 0:
                    raise RuntimeError("malformed .dynstr offset: %r" % (path,))
                name = strings[st_name:end].decode("ascii", "replace")
                for j, symbol in enumerate(symbols):
                    if not found[j] and symbol in name:
                        found[j] = True
    except RuntimeError:
        raise
    except (OSError, ValueError, struct.error) as exc:
        raise RuntimeError(
            "D512 delivered decode FAIL-CLOSED: %s is not a readable ELF library (%s)"
            % (path, exc))
    return found


def _require_specialization(symbols):
    """Verify every required D512 specialization exists in the mapped library."""
    wanted = tuple(symbols)
    missing = []
    for path in _MAPPED_LIBS:
        found = _dynsym_specializations(path, wanted)
        missing.extend(s for s, ok in zip(wanted, found) if not ok)
    if missing:
        raise RuntimeError(
            "D512 delivered decode FAIL-CLOSED: the mapped library exports "
            "flash_attn_varlen_func but not the D512 specialization symbol(s) %r "
            "(mapped %r). Such a build routes head_dim 512 to the generic "
            "teco_slave_flash_attention_half kernel instead of the delivered D512 branch; "
            "stage a libteco_ops.so built from the delivered teco-ops main."
            % (sorted(set(missing)), _MAPPED_LIBS))
    return {s: True for s in wanted}


_D512_SPECIALIZATIONS = (_D512_SYMBOL,)


if os.environ.get("GEMMA_D512_OFFICIAL_EXTENSION") or os.environ.get("GEMMA_D512_OFFICIAL_SHA256"):
    raise RuntimeError(
        "D512 delivered decode: GEMMA_D512_OFFICIAL_EXTENSION/_SHA256 select the isolated "
        "extension tier; unset them when the delivered tecoops binding is the default")

if not hasattr(tecoops, "flash_attn_varlen_func"):
    raise RuntimeError(
        "D512 delivered decode requires tecoops.flash_attn_varlen_func; rebuild/stage a "
        "libteco_ops.so and _torch_ext that export it")

_MAPPED_LIBS = _fail_closed_library_identity()
_D512_SPECIALIZATION_FOUND = _require_specialization(_D512_SPECIALIZATIONS)
print("[gemma-d512-delivered] LOAD " + json.dumps({
    "tecoops_file": getattr(tecoops, "__file__", None),
    "mapped_libteco_ops": _MAPPED_LIBS,
    "mapped_sha256": [_sha256(p) for p in _MAPPED_LIBS],
    "kernel": "tecoops.flash_attn_varlen_func",
    "required_specializations": list(_D512_SPECIALIZATIONS),
    "specialization_found": _D512_SPECIALIZATION_FOUND,
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
