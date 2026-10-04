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
import torch
import torch_sdaa
import torch_sdaa.nn
from torch._subclasses.fake_tensor import FakeTensorMode
from custom_ops.hy_mt2.vllm_swiglu import swiglu

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7191)
graphs=[]
def backend(gm, inputs):
    graphs.append(str(gm.graph))
    return gm.forward

compiled=torch.compile(swiglu, backend=backend, fullgraph=True, dynamic=False)
records=[]
with torch.inference_mode():
    for name, stream in [('default', torch.sdaa.default_stream()), ('nondefault', torch.sdaa.Stream())]:
        # Force vLLM-style token constraints before visiting the T1 specialization.
        for tokens in (8,32,1):
            for case in ('signed','zero'):
                cpu=torch.randn(1,tokens,1,12288,dtype=torch.float16)
                if case=='zero': cpu.zero_()
                with torch.sdaa.stream(stream):
                    x=cpu.to('sdaa')
                    if tokens>1: torch._dynamo.mark_dynamic(x,1,min=2,max=2048)
                    eager=torch.ops.sdaa.swiglu(x,tokens,1,6144)
                    actual=compiled(x)
                stream.synchronize()
                assert actual.shape==eager.shape==(tokens,1,6144)
                assert torch.equal(actual.cpu().view(torch.int16),eager.cpu().view(torch.int16))
                reference=(torch.nn.functional.silu(cpu[...,:6144].float())*cpu[...,6144:].float()).half().reshape(tokens,1,6144)
                torch.testing.assert_close(actual.cpu(),reference,atol=.002,rtol=.002)
                assert torch.equal(x.cpu().view(torch.int16),cpu.view(torch.int16))
                if tokens==32 and name=='default': assert len(graphs)==1
                records.append(dict(stream=name,tokens=tokens,case=case,bitexact=True,graph_count=len(graphs),cpu_max_abs=float((actual.cpu().float()-reference.float()).abs().max())))
                print('PASS',records[-1],flush=True)
    with FakeTensorMode():
        x=torch.empty(1,8,1,12288,dtype=torch.float16,device='sdaa')
        y=swiglu(x)
        assert y.shape==(8,1,6144) and y.device==x.device and y.dtype==x.dtype
    try: swiglu(torch.empty(1,8,1,12288,dtype=torch.float16))
    except NotImplementedError: cpu_rejected=True
    else: raise AssertionError('CPU accepted')
assert len(graphs)==2
assert all('hy_mt2_public.swiglu' in graph for graph in graphs)
result=dict(stage='v4_swiglu_tensor_boundary',backend='vllm-sdaa',pass_count=len(records),records=records,
            forced_dynamic_reused=True,graph_count=len(graphs),fake_pass=True,cpu_rejected=cpu_rejected,
            graphs=graphs,atol=.002,rtol=.002,claim='same vendor kernel, dynamic compatibility only')
(out/'focused.json').write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',{k:result[k] for k in ('pass_count','forced_dynamic_reused','graph_count','fake_pass','cpu_rejected')},flush=True)
