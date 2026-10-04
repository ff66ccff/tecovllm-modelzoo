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
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_sdaa
import tecoops

global_package = tecoops.__file__
global_kernel = tecoops.flash_attn_varlen_func
from custom_ops import d512_official as op

torch.set_num_threads(4)
torch.manual_seed(20261004)
random.seed(20261004)
assert op.OFFICIAL_DECODE is not None
rows = []

def invocation(q, k, v, cu, seq, bt, out):
    op.OFFICIAL_DECODE(q, k, v, cu, seq, bt, 2048, 1.0, out)
    return out

compiled = torch.compile(invocation, backend='eager', fullgraph=True)
for lengths in ([33], [65], [1025], [33, 65, 1025]):
    n = len(lengths)
    nb = sum((s + 31) // 32 for s in lengths) + 5
    q_cpu = torch.randn(n, 8, 512).half()
    q_cpu = (q_cpu.float() * torch.rsqrt(q_cpu.float().square().mean(-1, keepdim=True) + 1e-6)).half()
    k_cpu = torch.randn(nb, 1, 32, 512).half()
    k_cpu = (k_cpu.float() * torch.rsqrt(k_cpu.float().square().mean(-1, keepdim=True) + 1e-6)).half()
    v_cpu = torch.randn(nb, 1, 32, 512).half()
    pages = list(range(nb))
    random.shuffle(pages)
    bt_cpu = torch.full((n + 1, (max(lengths) + 31) // 32), -1, dtype=torch.int32)
    refs = []
    cursor = 0
    for i, length in enumerate(lengths):
        count = (length + 31) // 32
        chosen = pages[cursor:cursor + count]
        cursor += count
        bt_cpu[i, :count] = torch.tensor(chosen, dtype=torch.int32)
        kk = k_cpu[chosen, 0].reshape(-1, 512)[:length].float()
        vv = v_cpu[chosen, 0].reshape(-1, 512)[:length].float()
        refs.append(torch.softmax(q_cpu[i].float() @ kk.T, dim=-1) @ vv)
    reference = torch.stack(refs)
    for nondefault in (False, True):
        q_storage = torch.empty(n, 8, 1024, dtype=torch.float16, device='sdaa')
        q = q_storage[..., ::2]
        k = torch.empty_like(k_cpu, device='sdaa')
        v = torch.empty_like(v_cpu, device='sdaa')
        cu = torch.empty(n + 2, dtype=torch.int32, device='sdaa')
        seq = torch.empty(n + 1, dtype=torch.int32, device='sdaa')
        bt = torch.empty_like(bt_cpu, device='sdaa')
        out = torch.empty(n, 8, 512, dtype=torch.float16, device='sdaa')
        stream = torch.sdaa.Stream() if nondefault else torch.sdaa.current_stream()
        stream.wait_stream(torch.sdaa.current_stream())
        with torch.sdaa.stream(stream):
            q.copy_(q_cpu)
            k.copy_(k_cpu)
            v.copy_(v_cpu)
            cu.copy_(torch.tensor(list(range(n + 1)) + [n], dtype=torch.int32))
            seq.copy_(torch.tensor(list(lengths) + [0], dtype=torch.int32))
            bt.copy_(bt_cpu)
            if nondefault:
                # Real opaque execution, with the producer and kernel on the same nondefault stream.
                compiled(q, k, v, cu[:n + 1], seq[:n], bt[:n], out)
            else:
                md = SimpleNamespace(query_start_loc=cu, seq_lens=seq, block_table=bt, max_model_len=2048)
                impl = SimpleNamespace(scale=1.0)
                op.decode_forward(impl, out, q, k, v, md, n, k.stride(0))
        stream.synchronize()
        actual = out.cpu().float()
        error = (actual - reference).abs().max().item()
        assert torch.isfinite(actual).all() and error < 0.02, (lengths, nondefault, error)
        rows.append(dict(lengths=list(lengths), nondefault_stream=nondefault,
                         compiled_fullgraph_eager=nondefault, noncontiguous_query=True,
                         padded_metadata=not nondefault, max_abs_error=error))
        print(json.dumps(rows[-1]), flush=True)

assert tecoops.__file__ == global_package and tecoops.flash_attn_varlen_func is global_kernel
result = dict(passed=True, rows=rows, global_tecoops_unchanged=global_package,
              extension=__import__('_gemma_d512_official._torch_ext', fromlist=['']).__file__,
              max_abs_error=max(row['max_abs_error'] for row in rows))
Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
