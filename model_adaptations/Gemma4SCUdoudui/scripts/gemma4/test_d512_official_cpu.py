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

"""Pure CPU/FakeTensor isolated registration and binding checks; no device use."""
import os
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile

os.environ.pop('GEMMA_D512_OFFICIAL_EXTENSION', None)
os.environ.pop('GEMMA_D512_OFFICIAL_SHA256', None)
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from custom_ops import d512_official as op
from custom_ops.block_attention.op import bind_attention_impls, _decode_math

assert op.OFFICIAL_DECODE is None
def impl(head_size=512):
    return SimpleNamespace(head_size=head_size, num_heads=8, num_kv_heads=1,
                           _gemma4_window_size=None, logits_soft_cap=0, sinks=None,
                           kv_cache_dtype='auto', scale=1.0)

baseline = impl()
bind_attention_impls(baseline)
assert baseline._gemma4_decode_fn is _decode_math
calls = []
def kernel(*args):
    calls.append(args)
    args[-1].copy_(args[0] + 1)
registered = op.register_decode(kernel, 'gemma_d512_cpu_test')

q = torch.randn(2, 8, 1024, dtype=torch.float16)[..., ::2]
kc = torch.zeros(4, 1, 32, 512, dtype=torch.float16)
vc = torch.zeros_like(kc)
cu = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
seq = torch.tensor([33, 65, 0], dtype=torch.int32)
bt = torch.tensor([[0, 1, 0], [1, 2, 3], [-1, -1, -1]], dtype=torch.int32)
out = torch.empty(2, 8, 512, dtype=torch.float16)
md = SimpleNamespace(query_start_loc=cu, seq_lens=seq, block_table=bt, max_model_len=2048)
with patch.object(op, 'OFFICIAL_DECODE', registered):
    selected = impl()
    bind_attention_impls(selected)
    assert selected._gemma4_decode_fn is op.decode_forward
    d256 = impl(256)
    d256._gemma4_window_size = 1024
    bind_attention_impls(d256)
    assert d256._gemma4_decode_fn is _decode_math
    for field, value in [('sinks', torch.zeros(8)), ('logits_soft_cap', 1),
                         ('_gemma4_window_size', 1024), ('kv_cache_dtype', 'fp8'),
                         ('num_heads', 16), ('scale', 0.044)]:
        bad = impl()
        setattr(bad, field, value)
        try:
            bind_attention_impls(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(field)
    selected._gemma4_decode_fn(selected, out, q, kc, vc, md, 2, kc.stride(0))
assert torch.equal(out, q + 1)
args = calls[-1]
assert args[0].is_contiguous() and args[3] == 1 and args[5] == 2048
assert torch.equal(args[4], cu[:3]) and torch.equal(args[7], seq[:2])
assert torch.equal(args[11], bt[:2]) and args[8] == 1.0 and args[9] is True

with FakeTensorMode():
    fq = torch.empty(2, 8, 512, dtype=torch.float16)
    fk = torch.empty(4, 1, 32, 512, dtype=torch.float16)
    fcu = torch.empty(3, dtype=torch.int32)
    fs = torch.empty(2, dtype=torch.int32)
    fb = torch.empty(2, 3, dtype=torch.int32)
    fo = torch.empty_like(fq)
    assert registered(fq, fk, fk, fcu, fs, fb, 2048, 1.0, fo) is None
assert len(calls) == 1

def compiled_call(q, kc, vc, cu, seq, bt, out):
    registered(q, kc, vc, cu, seq, bt, 2048, 1.0, out)
    return out
compiled = torch.compile(compiled_call, backend='eager', fullgraph=True)
compiled(q, kc, vc, cu[:3], seq[:2], bt[:2], out)
assert torch.equal(out, q + 1) and len(calls) == 2
try:
    op.load_extension('/missing/_torch_ext.cpython-312-loongarch64-linux-gnu.so', '0' * 64)
except FileNotFoundError:
    pass
else:
    raise AssertionError('loader did not fail closed')
with tempfile.TemporaryDirectory() as directory:
    library = Path(directory) / '_torch_ext.cpython-312-loongarch64-linux-gnu.so'
    library.write_bytes(b'not an extension; reject before import')
    try:
        op.load_extension(library, '0' * 64)
    except RuntimeError as error:
        assert 'SHA256 mismatch' in str(error)
    else:
        raise AssertionError('wrong hash accepted')
    with patch.object(op.subprocess, 'run', return_value=SimpleNamespace(
            stdout='(NEEDED) Shared library: [libteco_ops.so]')):
        try:
            op.load_extension(library, op._sha256(library))
        except RuntimeError as error:
            assert 'core dependencies' in str(error)
        else:
            raise AssertionError('global core accepted')
print('PASS baseline/D512/D256 constructor guards; padding/noncontiguous Q/output mutation; FakeTensor/fullgraph mock; loader missing/hash/DT_NEEDED rejection')
