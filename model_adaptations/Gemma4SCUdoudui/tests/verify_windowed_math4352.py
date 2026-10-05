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

"""Prepared Gemma 4352 capacity prerequisite; explicit future device opt-in."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parent
if Path(__file__).resolve().parent.name == 'tests':
    PUBLIC=Path(__file__).resolve().parent.parent
else:
    PUBLIC=TASK.parents[2]/'model_adaptations/Gemma4SCUdoudui'
CASES=[(0,32)]+[(pr,n) for pr in (31,32,511,512,1023,1024,4095,4096) for n in (1,2,33,257,512) if pr+n<=4352]+[(3840,512),(4096,256)]
ATOL=.006
RTOL=.02
SERIAL_ATOL=.002
SERIAL_RTOL=.02

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-device',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.allow_device:p.error('requires root device release/ownership and --allow-device')
    assert Path(sys.executable).resolve()==Path('/home/py312/bin/python').resolve()
    manifest=json.loads((TASK/'operator4352_manifest.json').read_text())
    assert digest(Path(__file__))==manifest['test_sha256'],'test changed after freeze'
    for name,expected in manifest['sha256'].items():assert digest(PUBLIC/name)==expected,name
    assert digest(PUBLIC/'runtime/custom_ops/prefill_attention/op.py')==manifest['public_source_sha256']
    assert digest(PUBLIC/'runtime/custom_ops/reshape_and_cache/op.py')==manifest['cache_abi_wrapper_sha256']
    sys.path.insert(0,str(PUBLIC/'runtime'))
    import torch
    import torch_sdaa
    from custom_ops.reshape_and_cache import op as cache_op
    assert cache_op.sdaa_reshape_and_cache is cache_op._tecoops_reshape_and_cache,'must use actual vendor ABI'
    spec=importlib.util.spec_from_file_location('window_public_math',PUBLIC/'runtime/custom_ops/prefill_attention/op.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    attend=module.sdpa_prefill_attention_math
    assert Path(attend.__code__.co_filename).resolve()==(PUBLIC/'runtime/custom_ops/prefill_attention/op.py').resolve()
    torch.set_num_threads(4);torch.random.default_generator.manual_seed(20261005)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(passed=False,device='sdaa',gpu_run=True,cases=[],serial=[],rejections=0,thresholds=dict(atol=ATOL,rtol=RTOL,serial_atol=SERIAL_ATOL,serial_rtol=SERIAL_RTOL),prepared_manifest_snapshot=manifest,actual_prefill_filename=attend.__code__.co_filename,cache_abi='official continuous HND',model_run=False,fake_run=False,capture_run=False,performance_claim=False)
    def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
    def progress(row):
        save();print(json.dumps(row),flush=True)
    def oracle(q,k,v,prefix,window):
        # Independent CPU FP32 absolute-position mask, bounded query rows.
        qq=q.transpose(0,1).float();kk=k.transpose(0,1).float().repeat_interleave(8//k.shape[1],0)
        vv=v.transpose(0,1).float().repeat_interleave(8//v.shape[1],0);pieces=[]
        kp=torch.arange(len(k))
        for lo in range(0,len(q),128):
            hi=min(lo+128,len(q));qp=torch.arange(prefix+lo,prefix+hi)
            allowed=kp[None,:]<=qp[:,None]
            if window:allowed &= kp[None,:]>=qp[:,None]-window+1
            logits=qq[:,lo:hi]@kk.transpose(-1,-2)
            pieces.append((torch.softmax(logits.masked_fill(~allowed[None],float('-inf')),dim=-1)@vv).transpose(0,1))
        return torch.cat(pieces)
    def prepare(rows,d,hkv,window,retire=True,retired_mode='null'):
        lengths=[len(k) for q,k,v in rows];pages=[(l+31)//32 for l in lengths];reserved=1 if window else 2
        count=sum(pages)+reserved+2;ids=torch.randperm(count-reserved)+reserved
        table=torch.full((len(rows)+1,max(pages)+2),-1,dtype=torch.int32)
        kc=torch.full((count,hkv,32,d),float('nan'),dtype=torch.float16);vc=torch.full_like(kc,float('nan'))
        all_slots=[];tail_slots=[];tails_k=[];tails_v=[];cursor=0;expected_k=[];expected_v=[]
        for i,(q,k,v) in enumerate(rows):
            table[i,:pages[i]]=ids[cursor:cursor+pages[i]].to(torch.int32);cursor+=pages[i]
            prefix=len(k)-len(q);start=max(0,prefix-window+1) if window else 0
            pos=torch.arange(len(k));slots=table[i,pos//32].long()*32+pos%32
            prefix_pos=pos[:prefix]
            kc[slots[:prefix]//32,:,slots[:prefix]%32,:]=k[:prefix]
            vc[slots[:prefix]//32,:,slots[:prefix]%32,:]=v[:prefix]
            if retire and start:
                old=slots[:start];kc[old//32,:,old%32,:]=float('nan');vc[old//32,:,old%32,:]=float('nan')
                if retired_mode=='null':table[i,:start//32]=0
                else:assert retired_mode=='stale'  # Keep stale IDs; old offsets remain NaN.
            # Current tail must be populated only by the real vendor ABI.
            tail_slots.append(slots[prefix:]);tails_k.append(k[prefix:]);tails_v.append(v[prefix:])
            valid_pos=pos[start:] if retire else pos
            all_slots.append(slots[valid_pos]);expected_k.append(k[valid_pos]);expected_v.append(v[valid_pos])
        counts=[len(q) for q,k,v in rows];starts=torch.tensor([0]+[sum(counts[:i+1]) for i in range(len(rows))]+[sum(counts)],dtype=torch.int32)
        nk=torch.empty(sum(counts),hkv,d+3,dtype=torch.float16)[...,:d];nv=torch.empty_like(torch.empty(sum(counts),hkv,d+3,dtype=torch.float16))[...,:d]
        nk.copy_(torch.cat(tails_k));nv.copy_(torch.cat(tails_v))
        return dict(q=torch.cat([q for q,k,v in rows]),kc=kc,vc=vc,table=table,starts=starts,lens=torch.tensor(lengths+[0],dtype=torch.int32),slots=torch.cat(tail_slots),nk=nk,nv=nv,history=torch.cat(all_slots),expected_k=torch.cat(expected_k),expected_v=torch.cat(expected_v))
    def device_run(plan,window):
        q=plan['q'].to('sdaa');kc=plan['kc'].to('sdaa');vc=plan['vc'].to('sdaa')
        table=plan['table'].to('sdaa');starts=plan['starts'].to('sdaa');lens=plan['lens'].to('sdaa');slots=plan['slots'].to('sdaa')
        # Preserve noncontiguous tail views on device while caches stay real HND contiguous.
        nk=torch.empty(len(slots),plan['nk'].shape[1],plan['nk'].shape[2]+3,dtype=torch.float16,device='sdaa')[...,:-3]
        nv=torch.empty_like(torch.empty(len(slots),plan['nv'].shape[1],plan['nv'].shape[2]+3,dtype=torch.float16,device='sdaa'))[...,:-3]
        nk.copy_(plan['nk']);nv.copy_(plan['nv'])
        assert kc.is_contiguous() and vc.is_contiguous() and not nk.is_contiguous() and not nv.is_contiguous()
        cache_op.sdaa_reshape_and_cache(nk,nv,kc,vc,slots)
        # No .equal/.cpu/synchronize between ABI cache write and attention.
        output=attend(q,kc,vc,table,starts,lens,1.0,True,window)
        torch.sdaa.current_stream().synchronize()
        history=plan['history'].to('sdaa')
        torch.testing.assert_close(kc[history//32,:,history%32,:].cpu(),plan['expected_k'],atol=0,rtol=0)
        torch.testing.assert_close(vc[history//32,:,history%32,:].cpu(),plan['expected_v'],atol=0,rtol=0)
        return output.cpu(),(q,kc,vc,table,starts,lens)
    def serial_run(qq,kk,vv,chunk,d,hkv,window,stream_name):
        # One persistent device cache and fixed physical plan across all chunks.
        total=len(qq);reserved=1 if window else 2;npages=(total+31)//32
        ids=torch.randperm(npages)+reserved;count=npages+reserved
        physical=torch.full((2,npages+2),-1,dtype=torch.int32);physical[0,:npages]=ids.to(torch.int32)
        kc=torch.full((count,hkv,32,d),float('nan'),dtype=torch.float16,device='sdaa');vc=torch.full_like(kc,float('nan'));pieces=[]
        for prefix in range(0,total,chunk):
            end=min(prefix+chunk,total);start=max(0,prefix-window+1) if window else 0
            table_cpu=physical.clone();table_cpu[0,:start//32]=0
            # Poison retired values in their original pages, including old
            # offsets of the retained boundary page. No prefix is reloaded.
            old=torch.arange(start);oldslots=(physical[0,old//32].long()*32+old%32).to('sdaa')
            kc[oldslots//32,:,oldslots%32,:]=float('nan');vc[oldslots//32,:,oldslots%32,:]=float('nan')
            pos=torch.arange(prefix,end);slots=(physical[0,pos//32].long()*32+pos%32).to('sdaa')
            nk=torch.empty(len(pos),hkv,d+3,dtype=torch.float16,device='sdaa')[...,:d];nv=torch.empty_like(torch.empty(len(pos),hkv,d+3,dtype=torch.float16,device='sdaa'))[...,:d]
            nk.copy_(kk[prefix:end]);nv.copy_(vv[prefix:end])
            q=qq[prefix:end].to('sdaa');table=table_cpu.to('sdaa')
            starts=torch.tensor([0,end-prefix,end-prefix],dtype=torch.int32,device='sdaa');lens=torch.tensor([end,0],dtype=torch.int32,device='sdaa')
            cache_op.sdaa_reshape_and_cache(nk,nv,kc,vc,slots)
            value=attend(q,kc,vc,table,starts,lens,1.0,True,window)
            torch.sdaa.current_stream().synchronize()
            needed=torch.arange(start,end);history=(physical[0,needed//32].long()*32+needed%32).to('sdaa')
            torch.testing.assert_close(kc[history//32,:,history%32,:].cpu(),kk[start:end],atol=0,rtol=0)
            torch.testing.assert_close(vc[history//32,:,history%32,:].cpu(),vv[start:end],atol=0,rtol=0)
            pieces.append(value.cpu())
            result['last_progress']=dict(stream=stream_name,phase='persistent_serial',d=d,chunk=chunk,prefix=prefix,end=end)
            progress(result['last_progress'])
        return torch.cat(pieces)
    import contextlib
    contexts=[('default',None),('nondefault',torch.sdaa.Stream())]
    save()
    try:
        for stream_name,stream in contexts:
            with contextlib.nullcontext() if stream is None else torch.sdaa.stream(stream):
                for d,hkv,window in ((256,4,1024),(512,1,None)):
                    for case_index,(prefix,nq) in enumerate(CASES):
                        rows=[]
                        for length,count in ((prefix+nq,nq),(4352,1)):
                            rows.append((torch.randn(count,8,d,dtype=torch.float16)*.2,torch.randn(length,hkv,d,dtype=torch.float16)*.2,torch.randn(length,hkv,d,dtype=torch.float16)*.2))
                        retired_mode='null' if case_index%2==0 else 'stale'
                        plan=prepare(rows,d,hkv,window,retired_mode=retired_mode)
                        output,device_args=device_run(plan,window)
                        expected=torch.cat([oracle(q,k,v,len(k)-len(q),window) for q,k,v in rows])
                        assert torch.isfinite(output).all();torch.testing.assert_close(output.float(),expected,atol=ATOL,rtol=RTOL)
                        q,kc,vc,table,starts,lens=device_args
                        for row,(qi,ki,vi) in enumerate(rows):
                            start=max(0,len(ki)-len(qi)-window+1) if window else 0
                            for null_id in ((0,) if window else (0,1)):
                                bad=table.clone();bad[row,start//32]=null_id
                                try:attend(q,kc,vc,bad,starts,lens,1.0,True,window)
                                except RuntimeError:result['rejections']+=1
                                else:raise AssertionError('required null accepted')
                        row=dict(stream=stream_name,d=d,hq=8,hkv=hkv,window=window,prefix=prefix,q=nq,retired_mode=retired_mode,max_abs=float((output.float()-expected).abs().max()),cache_exact=True)
                        result['cases'].append(row);progress(row)
                    total=4352;qq=torch.randn(total,8,d,dtype=torch.float16)*.2;kk=torch.randn(total,hkv,d,dtype=torch.float16)*.2;vv=torch.randn_like(kk)*.2
                    whole,_=device_run(prepare([(qq,kk,vv)],d,hkv,window,False),window)
                    whole_oracle=oracle(qq,kk,vv,0,window)
                    torch.testing.assert_close(whole.float(),whole_oracle,atol=ATOL,rtol=RTOL)
                    for chunk in (33,257,512):
                        joined=serial_run(qq,kk,vv,chunk,d,hkv,window,stream_name)
                        torch.testing.assert_close(joined,whole,atol=SERIAL_ATOL,rtol=SERIAL_RTOL)
                        torch.testing.assert_close(joined.float(),whole_oracle,atol=ATOL,rtol=RTOL)
                        row=dict(stream=stream_name,d=d,hkv=hkv,chunk=chunk,total=total,max_abs_vs_whole=float((joined-whole).abs().max()))
                        result['serial'].append(row);progress(row)
        for name,expected in manifest['sha256'].items():assert digest(PUBLIC/name)==expected,name
        assert digest(PUBLIC/'runtime/custom_ops/prefill_attention/op.py')==manifest['public_source_sha256']
        counts=manifest['expected_sdaa_counts']
        assert len(result['cases'])==counts['numerical'],('incomplete numerical gate',len(result['cases']),counts)
        assert len(result['serial'])==counts['serial'],('incomplete serial gate',len(result['serial']),counts)
        assert result['rejections']==counts['required_null_rejections'],('incomplete null rejection gate',result['rejections'],counts)
        result['passed']=True
    except Exception as exc:
        result['error']=repr(exc);raise
    finally:save()
    print(json.dumps(dict(passed=True,cases=len(result['cases']),serial=len(result['serial']),rejections=result['rejections'])),flush=True)

if __name__=='__main__':main()
