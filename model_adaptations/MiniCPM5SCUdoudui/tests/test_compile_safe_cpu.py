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

"""Hermetic CPU checks. Never import tecoops, torch_sdaa, vllm or vllm_sdaa.

Run with vendor Python -S; explicitly add its pure Python site-packages.
This is binding/compile evidence only, not numerical vendor-kernel evidence.
"""
from pathlib import Path
import os
import sys
import types

bundle = Path(__file__).resolve().parents[1]
runtime = bundle/'runtime'
# -S on Python 3.12 resets the venv prefix; sysconfig then points at the base
# interpreter. Use the explicitly locked vendor environment, never $HOME.
sys.path.insert(0, '/home/py312/lib/python3.12/site-packages')
sys.path.insert(0, str(runtime))
os.environ.pop('TECOOPS_CALL_RECEIPT', None)
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from custom_ops.compile_safe import official_ops

calls = []
def rms(x, weight, residual, out, residual_out, eps):
    calls.append('rms')
    z = x if residual is None else x + residual
    if residual_out is not None:
        residual_out.copy_(z)
    out.copy_(z.float() * torch.rsqrt(z.float().square().mean(-1, keepdim=True) + eps) * weight.float())

def cache(key, value, slots, key_cache, value_cache):
    calls.append('cache')
    assert key.is_contiguous() and value.is_contiguous()
    assert slots.dtype == torch.int64
    key_cache.copy_(key)
    value_cache.copy_(value)

def flash(q, k, v, maxq, cuq, maxk, placeholder1, used, scale, causal, placeholder2, blocks, return_softmax, out):
    calls.append('flash')
    assert (maxq, maxk, scale, causal, return_softmax) == (4, 4, 0.5, True, False)
    assert placeholder1.numel() == placeholder2.numel() == 0
    out.copy_(q)

stub = types.SimpleNamespace(rms_norm=rms, reshape_and_cache=cache, flash_attn_varlen_func=flash)
assert official_ops(types.SimpleNamespace()) is None
ops = official_ops(stub)
assert official_ops().rms_norm is ops.rms_norm

def forward(x, w, residual, key, value, kc, vc, slots, cu, used, blocks):
    out = torch.empty_like(x)
    out2 = torch.empty_like(x)
    rout = torch.empty_like(x)
    attn_out = torch.empty_like(key)
    ops.rms_norm(x, w, out, 1e-6)
    ops.rms_norm_add(x, w, residual, out2, rout, 1e-6)
    ops.reshape_and_cache(key, value, kc, vc, slots)
    ops.flash_attn_varlen_func(key, kc, vc, cu, cu, 4, 4, 0.5, True, used, blocks, attn_out)
    return out, out2, rout, kc, vc, attn_out

torch.manual_seed(7)
x = torch.randn(4, 8, dtype=torch.float16)
w = torch.randn(8, dtype=torch.float16)
r = torch.randn_like(x)
key = torch.randn(4, 8, 2, dtype=torch.float16).transpose(1, 2)
value = torch.randn_like(key)
args = (x, w, r, key, value, torch.zeros_like(key), torch.zeros_like(value),
        torch.arange(4, dtype=torch.int64), torch.tensor([0, 4], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32), torch.tensor([[0]], dtype=torch.int32))
before = len(calls)
with FakeTensorMode() as mode:
    fake = tuple(mode.from_tensor(t) for t in args)
    results = forward(*fake)
    assert len(results) == 6
assert len(calls) == before, 'FakeTensor propagation entered raw kernels'
print('FAKE_ZERO_RAW_CALLS_OK')

eager = forward(*args)
calls.clear()
compiled = torch.compile(forward, backend='eager', fullgraph=True)(*args)
for got, expected in zip(compiled, eager):
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
assert calls == ['rms', 'rms', 'cache', 'flash'], calls
torch.testing.assert_close(compiled[2], x + r, rtol=0, atol=0)
torch.testing.assert_close(compiled[3], key, rtol=0, atol=0)
torch.testing.assert_close(compiled[4], value, rtol=0, atol=0)
torch.testing.assert_close(compiled[5], key, rtol=0, atol=0)
assert not any(name.startswith(('torch_sdaa', 'vllm', 'tecoops')) for name in sys.modules)
print('FULLGRAPH_CPU_ABI_MUTATION_OK')
