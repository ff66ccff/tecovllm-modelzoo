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
import vllm.model_executor.models.hunyuan_v1 as hunyuan
cls=hunyuan.RMSNorm
assert cls.__name__=='TecoopsRMSNorm'
cases=[dict(dtype=torch.bfloat16),dict(has_weight=False,dtype=torch.float16),dict(var_hidden_size=1024,dtype=torch.float16),dict(eps=0,dtype=torch.float16)]
records=[]
for kwargs in cases:
    try: cls(2048,**kwargs)
    except (RuntimeError,ValueError) as exc: records.append({'args':str(kwargs),'rejected':True,'error':str(exc)})
    else: raise RuntimeError('Invalid norm construction accepted')
norm=cls(2048,eps=1e-5,dtype=torch.float16).to('sdaa')
with torch.inference_mode():
    x=torch.randn(8,2048,dtype=torch.float16,device='sdaa')
    y=norm(x)
    yr,res=norm(x,x.clone())
    assert y.shape==yr.shape==res.shape==x.shape
Path(sys.argv[1]).write_text(json.dumps({'guard_records':records,'pass_count':4,'optimized_python':not __debug__,'real_forward_pass':True},indent=2)+'\n')
print('Guard rejects',len(records),'optimized_python',not __debug__,'real_forward_pass',True)
