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

"""Initialization-bound isolated D512 global decode; no global tecoops changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import torch


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_extension(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if path.name != "_torch_ext.cpython-312-loongarch64-linux-gnu.so":
        raise RuntimeError("unexpected isolated extension basename")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise RuntimeError("extension SHA256 must contain 64 hex digits")
    digest = _sha256(path)
    if digest != expected_sha256.lower():
        raise RuntimeError("isolated extension SHA256 mismatch")
    dynamic = subprocess.run(
        ["readelf", "-d", str(path)], check=True, capture_output=True, text=True,
    ).stdout
    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
    cores = [name for name in needed if "teco" in name and "sdaa" not in name]
    if cores != ["libteco_gemma_flash.so"]:
        raise RuntimeError(f"unexpected isolated core dependencies: {cores}")
    core = path.parent / "libteco_gemma_flash.so"
    core = core.resolve(strict=True)
    package_name = "_gemma_d512_official"
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    name = package_name + "._torch_ext"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    maps = Path("/proc/self/maps").read_text()
    loaded_cores = {
        line.split()[-1] for line in maps.splitlines()
        if "libteco_gemma_flash.so" in line
    }
    if loaded_cores != {str(core)}:
        raise RuntimeError(f"unexpected loaded isolated core paths: {loaded_cores}")
    print("[gemma-d512-official] LOAD " + json.dumps({
        "extension": str(path), "extension_sha256": digest,
        "core": str(core), "core_sha256": _sha256(core), "needed": needed,
    }), file=sys.stderr, flush=True)
    return module.flash_attn_varlen_func


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


_path = os.environ.get("GEMMA_D512_OFFICIAL_EXTENSION")
_digest = os.environ.get("GEMMA_D512_OFFICIAL_SHA256")
if bool(_path) != bool(_digest):
    raise RuntimeError("D512 isolated extension path and SHA256 must both be set")
OFFICIAL_DECODE = register_decode(load_extension(_path, _digest)) if _path else None


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
