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

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import torch_sdaa
import torch_sdaa.nn
import tecoops
import faulthandler
faulthandler.enable()
import numpy as np
from custom_ops.hy_mt2.vllm_attention import attention_forward, _prefill, install_attention

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(73051)
install_attention()
from vllm_sdaa.attention.block_attn import BlockAttentionImpl
impl=BlockAttentionImpl(16,128,128**-.5,4,None,None,'auto')
try: BlockAttentionImpl(8,128,128**-.5,4,None,None,'auto')
except RuntimeError: geometry_rejected=True
else: raise AssertionError('Unvalidated geometry accepted')
records=[]
NRMSE=.00390625; MAX_ABS=.06
def tensor(values,device='sdaa',dtype=torch.int32):
    return torch.tensor(values,dtype=dtype,device=device)

with torch.inference_mode():
  for stream_name,stream in [('default',torch.sdaa.default_stream()),('nondefault',torch.sdaa.Stream())]:
    for case,qlens,klens in [('prefill',[9,33],[9,33]),('mixed',[9,1],[9,33]),('decode',[1,1,1],[9,33,129]),('long_prefill',[513],[513])]:
      if len(sys.argv)>2 and case!=sys.argv[2]: continue
      for version in (1,2) if case=='decode' else (1,):
        print('BEGIN',stream_name,case,version,flush=True)
        rows=sum(qlens); block_counts=[(length+31)//32 for length in klens]
        nb=sum(block_counts); max_blocks=max(block_counts)
        table=[]; nxt=0
        # Nonmonotonic physical blocks expose accidental contiguous-cache reads.
        physical=list(reversed(range(nb)))
        for count in block_counts:
            table.append(physical[nxt:nxt+count]+[-1]*(max_blocks-count)); nxt+=count
        q_cpu=torch.randn(rows,16,128,dtype=torch.float16)
        ks=[torch.randn(length,4,128,dtype=torch.float16) for length in klens]
        vs=[torch.randn(length,4,128,dtype=torch.float16) for length in klens]
        kc_cpu=torch.full((nb,4,32,128),.3125,dtype=torch.float16); vc_cpu=kc_cpu.clone()
        slots=[]; current_k=[]; current_v=[]
        for i,(ql,kl) in enumerate(zip(qlens,klens)):
            for token in range(kl-ql):
                kc_cpu[table[i][token//32],:,token%32,:]=ks[i][token]
                vc_cpu[table[i][token//32],:,token%32,:]=vs[i][token]
            for token in range(kl-ql,kl): slots.append(table[i][token//32]*32+token%32)
            current_k.append(ks[i][-ql:]); current_v.append(vs[i][-ql:])
        expected_k=kc_cpu.clone(); expected_v=vc_cpu.clone()
        for slot,key,value in zip(slots,torch.cat(current_k),torch.cat(current_v)):
            expected_k[slot//32,:,slot%32,:]=key; expected_v[slot//32,:,slot%32,:]=value
        qloc=[0]
        for length in qlens: qloc.append(qloc[-1]+length)
        pre=[ql if ql>1 else 0 for ql in qlens]
        decode=[(kl-1 if version==1 else kl) if ql==1 else 0 for ql,kl in zip(qlens,klens)]
        prefix=[kl-ql if ql>1 else 0 for ql,kl in zip(qlens,klens)]
        with torch.sdaa.stream(stream):
            print('COPY',stream_name,case,version,flush=True)
            cache=torch.stack([kc_cpu,vc_cpu]).to('sdaa')
            # Fused-QKV-like strided dense inputs require the ABI contiguous boundary.
            storage=torch.zeros(rows,4,128,2,dtype=torch.float16,device='sdaa')
            storage[...,0].copy_(torch.cat(current_k).to('sdaa'))
            storage[...,1].copy_(torch.cat(current_v).to('sdaa'))
            key,value=storage[...,0],storage[...,1]
            assert not key.is_contiguous() and not value.is_contiguous()
            metadata=SimpleNamespace(num_actual_tokens=rows,max_query_len=max(qlens),
                slot_mapping=tensor(slots,dtype=torch.int64),query_start_loc=tensor(qloc),
                seq_lens=tensor(klens),block_table=tensor(table),causal=True,
                seq_lens_pre_cache=tensor(prefix),seq_lens_pre_cache_cpu=tensor(prefix,'cpu'),
                seq_lens_prefill=tensor(pre),seq_lens_prefill_cpu=tensor(pre,'cpu'),
                seq_lens_decode=tensor(decode),max_model_len=8192,
                max_prefill_len=max([p+s for p,s in zip(prefix,pre)]),max_decode_len=max(decode),version=version)
            output=torch.empty(rows,2048,dtype=torch.float16,device='sdaa')
            print('FORWARD',stream_name,case,version,flush=True)
            actual=attention_forward(impl,None,q_cpu.to('sdaa'),key,value,cache,metadata,output)
        stream.synchronize()
        print('CHECK',stream_name,case,version,flush=True)
        assert actual is output
        assert torch.equal(cache[0].cpu().view(torch.int16),expected_k.view(torch.int16))
        assert torch.equal(cache[1].cpu().view(torch.int16),expected_v.view(torch.int16))
        reference=[]; start=0
        for ql,kl,k,v in zip(qlens,klens,ks,vs):
            q=q_cpu[start:start+ql].numpy().astype(np.float64)
            k=np.repeat(k.numpy().astype(np.float64),4,axis=1)
            v=np.repeat(v.numpy().astype(np.float64),4,axis=1)
            # Independent scalar-reduction CPU oracle; do not invoke the
            # vendor Torch CPU batched GEMM that crashes on 513x513.
            scores=np.einsum('qhd,khd->hqk',q,k,optimize=False)*128**-.5
            mask=np.arange(kl)[None,:]>(np.arange(ql)[:,None]+kl-ql)
            scores=np.where(mask[None],-np.inf,scores)
            probability=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
            probability/=probability.sum(axis=-1,keepdims=True)
            result=np.einsum('hqk,khd->qhd',probability,v,optimize=False)
            reference.append(torch.from_numpy(result).float()); start+=ql
        reference=torch.cat(reference).reshape(rows,2048)
        diff=actual.cpu().float()-reference
        nrmse=float(diff.square().mean().sqrt()/reference.square().mean().sqrt())
        max_abs=float(diff.abs().max())
        record=dict(stream=stream_name,case=case,version=version,q_lens=qlens,kv_lens=klens,
                    cache_bitexact=True,nrmse=nrmse,max_abs=max_abs)
        print('CASE',record,flush=True)
        assert nrmse<=NRMSE and max_abs<=MAX_ABS
        records.append(record)
  # Signed slots: negative destinations must leave all cache guards unchanged.
  key=torch.randn(4,4,128,dtype=torch.float16,device='sdaa')
  value=torch.randn_like(key); cache=torch.full((2,3,4,32,128),.3125,dtype=torch.float16,device='sdaa')
  expected=cache.cpu(); slots=[-1,31,64,-8]
  for i,slot in enumerate(slots):
      if slot>=0:
          expected[0,slot//32,:,slot%32,:]=key[i].cpu(); expected[1,slot//32,:,slot%32,:]=value[i].cpu()
  tecoops.reshape_and_cache(key,value,tensor(slots,dtype=torch.int64),cache[0],cache[1])
  assert torch.equal(cache.cpu().view(torch.int16),expected.view(torch.int16))
  try:
      _prefill(torch.randn(2,16,128,device='sdaa').half(),cache[0],cache[1],
               SimpleNamespace(query_start_loc=tensor([0,2]),seq_lens=tensor([3]),causal=True,
                               block_table=tensor([[0]])),128**-.5)
  except RuntimeError as exc:
      assert 'chunked-prefill' in str(exc); chunk_rejected=True
  else: raise AssertionError('Unsupported chunk accepted')
result=dict(stage='v5_attention_compatibility',backend='vllm-sdaa',pass_count=len(records),records=records,
            cpu_oracle=True,nrmse_tol=NRMSE,max_abs_tol=MAX_ABS,signed_slots_pass=True,
            chunk_rejected=chunk_rejected,geometry_rejected=geometry_rejected,
            claim='framework compatibility, no new kernel or performance claim')
(out/'focused.json').write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',{k:result[k] for k in ('pass_count','signed_slots_pass','chunk_rejected','geometry_rejected')},flush=True)
