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

"""Real official ABI / opaque / fullgraph checks, correctness only."""
from pathlib import Path
import sys
import os
os.environ.pop('TECOOPS_CALL_RECEIPT', None)
bundle = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(bundle/'runtime'))
import numpy as np
import torch
import torch_sdaa
import tecoops
from custom_ops.compile_safe import official_ops
assert torch.sdaa.is_available() and torch.sdaa.device_count() == 1
ops = official_ops(tecoops)
dev = 'sdaa:0'
rng = np.random.default_rng(12)
def tensor(a, dtype=torch.float16): return torch.tensor(a, dtype=dtype, device=dev)
def cpu(t):
    torch.sdaa.synchronize()
    return t.cpu().numpy()
def bits(a, b): assert np.array_equal(a.view(np.uint16), b.view(np.uint16))
def reference_close(a, b, tol, name):
    delta = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    assert np.isfinite(a).all() and delta < tol, (name, delta)
    print(f'{name}: max_abs={delta:.8g} tolerance={tol}')

xn = rng.normal(size=(4,1536)).astype(np.float16)
rn = rng.normal(size=xn.shape).astype(np.float16)
wn = rng.normal(size=(1536,)).astype(np.float16)
x, r, w = tensor(xn), tensor(rn), tensor(wn)
def norm(x, w):
    out = torch.empty_like(x); ops.rms_norm(x,w,out,1e-6); return out
def norm_add(x, w, r):
    out = torch.empty_like(x); rout = torch.empty_like(x)
    ops.rms_norm_add(x,w,r,out,rout,1e-6); return out,rout
raw = torch.empty_like(x); tecoops.rms_norm(x,w,None,raw,None,1e-6)
raw_add = torch.empty_like(x); raw_res = torch.empty_like(x)
tecoops.rms_norm(x,w,r,raw_add,raw_res,1e-6)
for fn, args, expected in ((norm,(x,w),(raw,)),(norm_add,(x,w,r),(raw_add,raw_res))):
    for run in (fn, torch.compile(fn,backend='eager',fullgraph=True)):
        outputs = run(*args)
        if not isinstance(outputs, tuple): outputs = (outputs,)
        for a,b in zip(outputs,expected): bits(cpu(a),cpu(b))
def rms_reference(z):
    z = z.astype(np.float32)
    return z / np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+1e-6) * wn.astype(np.float32)
reference_close(cpu(raw),rms_reference(xn),0.05,'rms_cpu_reference')
summed = xn.astype(np.float32)+rn.astype(np.float32)
reference_close(cpu(raw_add),rms_reference(summed),0.05,'rms_add_cpu_reference')
bits(cpu(raw_res),summed.astype(np.float16))
bits(cpu(x),xn); bits(cpu(r),rn)
print('RMS_AND_FUSED_RAW_OPAQUE_COMPILED_BITS_OK')

kn = rng.normal(size=(4,2,128)).astype(np.float16)
vn = rng.normal(size=kn.shape).astype(np.float16)
slots_n = np.array([1,33,-1,3],dtype=np.int64)
k, v, slots = tensor(kn),tensor(vn),tensor(slots_n,torch.int64)
kinit = np.full((3,2,32,128),-3.25,dtype=np.float16)
vinit = np.full((3,2,32,128),-2.5,dtype=np.float16)
kexpected,vexpected = kinit.copy(),vinit.copy()
for i,slot in enumerate(slots_n):
    if slot < 0: continue
    b,t = divmod(int(slot),32)
    kexpected[b,:,t,:] = kn[i]
    vexpected[b,:,t,:] = vn[i]
def cache(k,v,slots,kc,vc):
    ops.reshape_and_cache(k,v,kc,vc,slots); return kc,vc
for run in (None,cache,torch.compile(cache,backend='eager',fullgraph=True)):
    kc,vc = tensor(kinit),tensor(vinit)
    if run is None: tecoops.reshape_and_cache(k,v,slots,kc,vc)
    else: run(k,v,slots,kc,vc)
    bits(cpu(kc),kexpected); bits(cpu(vc),vexpected)
bits(cpu(k),kn); bits(cpu(v),vn)
print('CACHE_POSITIONS_SKIPPED_UNWRITTEN_BITS_OK')

lens=(8,16); hq,hkv,d,bsize=16,2,128,32
qn=rng.normal(size=(sum(lens),hq,d)).astype(np.float16)
kcn=rng.normal(size=(3,hkv,bsize,d)).astype(np.float16)
vcn=rng.normal(size=kcn.shape).astype(np.float16)
btn=np.array([[2],[0]],dtype=np.int32)
q,kc,vc=tensor(qn),tensor(kcn),tensor(vcn)
cu=tensor(np.array([0,8,24],dtype=np.int32),torch.int32)
used=tensor(np.array(lens,dtype=np.int32),torch.int32)
bt=tensor(btn,torch.int32)
scale=1/d**0.5
def flash(q,kc,vc,cu,used,bt):
    out=torch.empty_like(q)
    ops.flash_attn_varlen_func(q,kc,vc,cu,cu,16,16,scale,True,used,bt,out)
    return out
raw_flash=torch.empty_like(q); placeholder=torch.Tensor()
tecoops.flash_attn_varlen_func(q,kc,vc,16,cu,16,placeholder,used,scale,True,placeholder,bt,False,raw_flash)
for run in (flash,torch.compile(flash,backend='eager',fullgraph=True)):
    bits(cpu(run(q,kc,vc,cu,used,bt)),cpu(raw_flash))
reference=np.empty_like(qn,dtype=np.float32); offset=0
for seq,length in enumerate(lens):
    block=int(btn[seq,0])
    for head in range(hq):
        kh=head//(hq//hkv)
        qs=qn[offset:offset+length,head].astype(np.float32)
        ks=kcn[block,kh,:length].astype(np.float32)
        vs=vcn[block,kh,:length].astype(np.float32)
        scores=np.einsum('id,jd->ij',qs,ks,optimize=False)*scale
        scores[np.triu_indices(length,1)]=-np.inf
        scores-=np.max(scores,axis=-1,keepdims=True)
        weights=np.exp(scores); weights/=np.sum(weights,axis=-1,keepdims=True)
        reference[offset:offset+length,head]=np.einsum('ij,jd->id',weights,vs,optimize=False)
    offset+=length
reference_close(cpu(raw_flash),reference,0.05,'paged_causal_gqa_cpu_reference')
bits(cpu(kc),kcn); bits(cpu(vc),vcn)
print('FLASH_RAW_OPAQUE_COMPILED_BITS_OK')
print('ALL_FOUR_OFFICIAL_BINDINGS_PASS_NO_PERFORMANCE_CLAIM')
