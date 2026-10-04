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
import torch.nn.functional as F
import vllm.model_executor.models.hunyuan_v1 as hunyuan
torch.manual_seed(4721)
records=[]
with torch.inference_mode():
    for tokens in (1,8,32):
        gu=torch.randn(tokens,12288,dtype=torch.float16,device='sdaa')
        # Projection test doubles isolate the actual patched caller's reshape
        # and return contract, not the separately verified activation function.
        module=SimpleNamespace(gate_up_proj=lambda x:(gu,None),down_proj=lambda x:(x,None))
        actual=hunyuan.HunYuanMLP.forward(module,torch.empty(tokens,2048,dtype=torch.float16,device='sdaa'))
        expected=torch.ops.sdaa.swiglu(gu.reshape(1,tokens,1,12288),tokens,1,6144).reshape(tokens,6144)
        assert actual.shape==(tokens,6144)
        assert torch.equal(actual.cpu().view(torch.int16),expected.cpu().view(torch.int16))
        records.append({'tokens':tokens,'caller_shape_pass':True,'raw_bitexact':True})
Path(sys.argv[1]).write_text(json.dumps({'pass_count':len(records),'records':records},indent=2)+'\n')
print('Actual patched MLP caller',len(records),'cases passed')
