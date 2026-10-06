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

"""Public MiniCPM SDPA focused SDAA correctness; no performance claim.
Run: /home/py312/bin/python tests/test_sdpa_sdaa.py --output result.json
The caller pins exactly one logical device with SDAA_VISIBLE_DEVICES.
"""
import argparse, hashlib, json, os, re, sys, traceback
from pathlib import Path
import numpy as np
import torch
import torch_sdaa

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE/'runtime'))
from custom_ops.flash_attn_varlen.op import sdpa_varlen_prefill, make_opaque_sdpa_prefill
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.profiler import profile, ProfilerActivity

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
report={'passed':False,'performance_claim':False,'bundle':str(BUNDLE),'invoked_python':'/home/py312/bin/python','resolved_python':str(Path(sys.executable).resolve()),'physical_visibility':os.environ.get('SDAA_VISIBLE_DEVICES'),'cases':[],'tolerance_maxabs':0.05,'reference':'CPU NumPy float64 causal softmax, quantized FP16 inputs, per-head chunked oracle'}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def oracle(q,k,v,cu,used,bt):
    result=np.empty_like(q,dtype=np.float64)
    for seq,length in enumerate(used):
        length=int(length);lo=int(cu[seq]);ids=bt[seq].reshape(-1)[:(length+31)//32]
        kd=k[ids].transpose(1,0,2,3).reshape(2,-1,128)[:,:length].astype(np.float64)
        vd=v[ids].transpose(1,0,2,3).reshape(2,-1,128)[:,:length].astype(np.float64)
        for head in range(16):
            kh=head//8
            for begin in range(0,length,128):
                end=min(begin+128,length)
                scores=q[lo+begin:lo+end,head].astype(np.float64)@kd[kh].T/(128**.5)
                mask=np.arange(length)[None,:]>np.arange(begin,end)[:,None]
                scores[mask]=-np.inf
                scores-=scores.max(-1,keepdims=True)
                weights=np.exp(scores);weights/=weights.sum(-1,keepdims=True)
                result[lo+begin:lo+end,head]=weights@vd[kh]
    return result
def build(lengths,seed):
    rng=np.random.default_rng(seed)
    counts=[(n+31)//32 for n in lengths];nblocks=sum(counts)+2
    ids=rng.permutation(nblocks)[:sum(counts)]
    bt=np.full((len(lengths),max(counts)),-1,dtype=np.int32);cursor=0
    for i,count in enumerate(counts):bt[i,:count]=ids[cursor:cursor+count];cursor+=count
    q=rng.normal(size=(sum(lengths),16,128)).astype(np.float16)
    k=rng.normal(size=(nblocks,2,32,128)).astype(np.float16)
    v=rng.normal(size=k.shape).astype(np.float16)
    cu=np.array([0,*np.cumsum(lengths)],dtype=np.int32)
    return q,k,v,cu,np.array(lengths,dtype=np.int32),bt
def device(arr): return torch.tensor(arr,device='sdaa:0')
def cpu(t): torch.sdaa.synchronize();return t.detach().cpu().numpy()
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
report['script_sha256']=digest(Path(__file__))
report['source_sha256']={str(p.relative_to(BUNDLE)):digest(p) for p in (BUNDLE/'runtime/custom_ops/flash_attn_varlen/op.py',BUNDLE/'runtime/custom_ops/compile_safe.py',BUNDLE/'runtime/custom_ops/block_attention/op.py')}
try:
    assert Path('/home/py312/bin/python').resolve()==Path(sys.executable).resolve()
    assert torch.sdaa.is_available() and torch.sdaa.device_count()==1
    torch.backends.sdaa.enable_flash_sdp(True)
    torch.backends.sdaa.enable_math_sdp(False)
    torch.backends.sdaa.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    report['backend_flags']={'flash':torch.backends.sdaa.flash_sdp_enabled(),'math':torch.backends.sdaa.math_sdp_enabled(),'memory_efficient':torch.backends.sdaa.mem_efficient_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}
    assert report['backend_flags']=={'flash':True,'math':False,'memory_efficient':False,'cudnn':False}
    bound=make_opaque_sdpa_prefill()
    def run(q,k,v,cu,used,bt,out,maxlen):
        got=bound(q,k,v,cu,cu,maxlen,maxlen,128**-.5,True,used,bt,out)
        assert got is out
        return out
    compiled=torch.compile(run,backend='eager',fullgraph=True)
    torch.sdaa.reset_peak_memory_stats()
    for case_idx,lengths in enumerate(((7,),(256,),(512,),(1024,),(2048,),(7,256))):
        qn,kn,vn,cun,usedn,btn=build(lengths,71+case_idx)
        reference=oracle(qn,kn,vn,cun,usedn,btn)
        q,k,v,cu,used,bt=map(device,(qn,kn,vn,cun,usedn,btn))
        out=torch.full_like(q,float('nan'));rawout=torch.empty_like(q)
        row={'lengths':list(lengths),'q_shape':list(q.shape),'cache_shape':list(k.shape),'dtype':str(q.dtype),'device':str(q.device),'block_table_shape':list(bt.shape)}
        sdpa_varlen_prefill(q,k,v,cu,cu,max(lengths),max(lengths),128**-.5,True,used,None,bt,rawout)
        run(q,k,v,cu,used,bt,out,max(lengths))
        assert out.device.type=='sdaa' and k.device.type=='sdaa'
        got=cpu(out);raw=cpu(rawout)
        assert np.array_equal(got.view(np.uint16),raw.view(np.uint16))
        delta=np.abs(got.astype(np.float64)-reference)
        row.update(maxabs=float(delta.max()),rmse=float(np.sqrt(np.mean(delta*delta))),opaque_raw_bit_equal=True,out_mutated_finite=bool(np.isfinite(got).all()))
        assert np.isfinite(got).all() and row['maxabs']<0.05,row
        if lengths in ((7,),(7,256)):
            cout=torch.full_like(q,float('nan'))
            compiled(q,k,v,cu,used,bt,cout,max(lengths))
            assert np.array_equal(cpu(cout).view(np.uint16),got.view(np.uint16))
            row['compiled_fullgraph_bit_equal']=True
            with FakeTensorMode() as mode:
                fq,fk,fv,fcu,fused,fbt,fout=[mode.from_tensor(t) for t in (q,k,v,cu,used,bt,out)]
                fake_result=run(fq,fk,fv,fcu,fused,fbt,fout,max(lengths))
                assert fake_result is fout and fake_result.device.type=='sdaa'
            row['fake_out_identity_device']=True
            bt3=bt.unsqueeze(1);out3=torch.empty_like(out);run(q,k,v,cu,used,bt3,out3,max(lengths))
            assert np.array_equal(cpu(out3).view(np.uint16),got.view(np.uint16))
            row['block_table_3d_bit_equal']=True
        assert np.array_equal(cpu(k).view(np.uint16),kn.view(np.uint16))
        assert np.array_equal(cpu(v).view(np.uint16),vn.view(np.uint16))
        row['cache_unchanged']=True;row['passed']=True;report['cases'].append(row)
        print(json.dumps(row),flush=True)
        if lengths==(256,):
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.SDAA]) as prof:
                run(q,k,v,cu,used,bt,out,max(lengths))
                torch.sdaa.synchronize()
            events=prof.events()
            devices=[{'name':e.name,'device_type':str(e.device_type),'device_index':e.device_index} for e in events if 'CPU' not in str(e.device_type)]
            names=[e.name for e in events]
            report['profile_device_events']=devices
            report['profile_attention_operators']=sorted(set(n for n in names if re.search(r'flash|attention|sdp',n,re.I)))
            trace=Path(args.output).with_suffix('.trace.json');prof.export_chrome_trace(str(trace));report['trace_path']=str(trace)
            # Fused operator receipt and real device kernels are both required.
            assert devices,'No SDAA device events: no device execution proof'
            assert any('flash' in n.lower() for n in report['profile_attention_operators']),'No fused flash attention receipt'
            assert not any('_scaled_dot_product_attention_math' in n for n in names),'Math SDP fallback'
            report['fused_flash_device_receipt']=True
    bad=used.clone();bad[0]+=1
    try:run(q,k,v,cu,bad,bt,out,max(lengths))
    except RuntimeError as e:
        assert 'query_len == seq_len' in str(e)
        report['prefix_chunk_rejected']=str(e)
    else:raise AssertionError('prefix/chunk accepted')
    report['peak_allocated_bytes']=int(torch.sdaa.max_memory_allocated())
    report['peak_reserved_bytes']=int(torch.sdaa.max_memory_reserved())
    report['versions']={'torch':torch.__version__,'torch_sdaa':getattr(torch_sdaa,'__version__','unknown')}
    report['source_sha256']={str(p.relative_to(BUNDLE)):digest(p) for p in (BUNDLE/'runtime/custom_ops/flash_attn_varlen/op.py',BUNDLE/'runtime/custom_ops/compile_safe.py',BUNDLE/'runtime/custom_ops/block_attention/op.py')}
    report['passed']=True
except Exception:
    report['error']=traceback.format_exc()
    raise
finally:
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2))
