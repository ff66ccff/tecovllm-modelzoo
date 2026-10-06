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


"""CPU semantic/ABI/opaque checks; no vendor backend or device import."""
from pathlib import Path
import os
import sys
import types
import ast
bundle = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/home/py312/lib/python3.12/site-packages')
sys.path.insert(0, str(bundle/'runtime'))
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from custom_ops.flash_attn_varlen.op import sdpa_varlen_prefill, make_opaque_sdpa_prefill
from custom_ops import compile_safe

torch.manual_seed(31)
q=torch.randn(8,4,8)
k=torch.randn(5,2,2,8)
v=torch.randn_like(k)
cu=torch.tensor([0,3,8],dtype=torch.int32)
used=torch.tensor([3,5],dtype=torch.int32)
blocks=torch.tensor([[3,0,-1],[4,2,1]],dtype=torch.int32)
scale=8**-.5
expected=torch.empty_like(q)
for i,(lo,hi) in enumerate(((0,3),(3,8))):
    ids=blocks[i,:((hi-lo+1)//2)].long()
    def dense(cache):
        return cache[ids].permute(1,0,2,3).reshape(2,-1,8)[:,:hi-lo].unsqueeze(0)
    z=torch.nn.functional.scaled_dot_product_attention(q[lo:hi].transpose(0,1).unsqueeze(0),dense(k).repeat_interleave(2,1),dense(v).repeat_interleave(2,1),is_causal=True,scale=scale)
    expected[lo:hi]=z.squeeze(0).transpose(0,1)
for table in (blocks,blocks.unsqueeze(1)):
    out=torch.full_like(q,float('nan'))
    got=sdpa_varlen_prefill(q,k,v,cu,cu,5,5,scale,True,used,None,table,out)
    assert got is out
    torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)
print('PAGED_HND_MIXED_LENGTH_GQA_BLOCKTABLE_2D_3D_OK')

bound=make_opaque_sdpa_prefill()
def forward(q,k,v,cu,used,blocks):
    out=torch.empty_like(q)
    # Exactly the same 12 positional slots used by the official opaque ABI.
    got=bound(q,k,v,cu,cu,5,5,scale,True,used,blocks,out)
    assert got is out
    return out
eager=forward(q,k,v,cu,used,blocks)
torch.testing.assert_close(eager,expected,atol=2e-6,rtol=2e-6)
# Fake must never enter the real kernel body or read metadata tensors.
with FakeTensorMode() as mode:
    fake=[mode.from_tensor(t) for t in (q,k,v,cu,used,blocks)]
    result=forward(*fake)
    assert result.shape==q.shape
compiled=torch.compile(forward,backend='eager',fullgraph=True)(q,k,v,cu,used,blocks)
torch.testing.assert_close(compiled,eager,rtol=0,atol=0)
print('TWELVE_POSITIONAL_OUT_MUTATION_FAKE_FULLGRAPH_OK')

for bad_used in (torch.tensor([4,5],dtype=torch.int32),torch.tensor([3,6],dtype=torch.int32)):
    try:
        forward(q,k,v,cu,bad_used,blocks)
    except RuntimeError as e:
        assert 'query_len == seq_len' in str(e)
    else:
        raise AssertionError('prefix/chunk metadata accepted')
print('CHUNK_PREFIX_FAIL_CLOSED_OK')
old=compile_safe.register_sdpa_prefill
compile_safe.register_sdpa_prefill=lambda kernel:None
try:
    make_opaque_sdpa_prefill()
except RuntimeError:
    pass
else:
    raise AssertionError('registration failure used eager fallback')
finally:
    compile_safe.register_sdpa_prefill=old
print('OPAQUE_REGISTRATION_FAIL_CLOSED_OK')

calls=[]
def raw(*args): pass
stub=types.SimpleNamespace(rms_norm=raw,reshape_and_cache=raw,flash_attn_varlen_func=raw)
compile_safe.official_ops(stub)
torch.backends.sdaa=types.SimpleNamespace(enable_flash_sdp=lambda value:calls.append(value))
source=bundle/'runtime/custom_ops/block_attention/op.py'
# Isolate import-time bindings using the same registered ops and fake backend.
for profile,expected_calls in ((None,0),('official',0),('fast_attention',1),('invalid',0)):
    if profile is None: os.environ.pop('MINICPM_OP_PROFILE',None)
    else: os.environ['MINICPM_OP_PROFILE']=profile
    calls.clear(); namespace={'__name__':'profile_probe'}
    try:
        exec(compile(source.read_text(),str(source),'exec'),namespace)
    except RuntimeError:
        assert profile=='invalid'
    else:
        assert profile!='invalid'
        assert namespace['_PROFILE']==(profile or 'official')
    assert len(calls)==expected_calls
tree=ast.parse(source.read_text())
forward_ast=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sdaa_block_attention_forward')
assert not any(isinstance(n,ast.Name) and n.id=='_PROFILE' for n in ast.walk(forward_ast))
assert not any(isinstance(n,ast.Attribute) and n.attr in ('getenv','environ') for n in ast.walk(forward_ast))
assert not any(n.startswith(('torch_sdaa','vllm','tecoops')) for n in sys.modules)
print('PROFILE_DEFAULT_OPTIN_UNKNOWN_INIT_ONLY_BACKEND_OK')
