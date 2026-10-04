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
from custom_ops.hy_mt2.vllm_attention import install_attention
install_attention()
from vllm_sdaa.attention.block_attn import BlockAttentionImpl
records=[]
for dtype,sharing in [('fp8',None),('auto','model.layers.0.self_attn')]:
    try:
        # Isolate initialization guard from the framework __new__ hook, which
        # requires distributed groups unavailable in this standalone fixture.
        instance=object.__new__(BlockAttentionImpl)
        BlockAttentionImpl.__init__(instance,16,128,128**-.5,4,None,None,dtype,kv_sharing_target_layer_name=sharing)
    except RuntimeError as exc: records.append({'cache_dtype':dtype,'sharing':sharing,'rejected':True,'error':str(exc)})
    else: raise RuntimeError('Unsupported cache configuration accepted')
Path(sys.argv[1]).write_text(json.dumps({'pass_count':2,'optimized_python':not __debug__,'records':records},indent=2)+'\n')
print('Unvalidated cache/sharing guards passed:',len(records),'optimized Python:',not __debug__)
