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

"""Real-data RMS opaque boundary validation; no model performance timing."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch_sdaa
import tecoops
from torch._subclasses.fake_tensor import FakeTensorMode
from custom_ops.hy_mt2.vllm_rms_norm import rms_norm, rms_norm_add

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(8123)
EPS = 1e-5
ATOL = RTOL = 0.002
records = []


def run_norm(x, w, residual=None):
    out = torch.empty_like(x)
    if residual is None:
        rms_norm(x, w, out, EPS)
        return out
    residual_out = torch.empty_like(x)
    rms_norm_add(x, w, residual, out, residual_out, EPS)
    return out, residual_out


graphs = []
def graph_backend(gm, example_inputs):
    graphs.append(str(gm.graph))
    return gm.forward


compiled = torch.compile(run_norm, backend=graph_backend, fullgraph=True)
with torch.inference_mode():
    for stream_name, stream in [('default', torch.sdaa.default_stream()), ('nondefault', torch.sdaa.Stream())]:
        for tokens in (1, 8, 32):
            for width, rows in ((2048, tokens), (128, tokens * 16), (128, tokens * 4)):
                x_cpu = torch.randn(rows, width, dtype=torch.float16)
                w_cpu = torch.randn(width, dtype=torch.float16)
                r_cpu = torch.randn(rows, width, dtype=torch.float16)
                for with_residual in (False, True):
                    with torch.sdaa.stream(stream):
                        x, w = x_cpu.to('sdaa'), w_cpu.to('sdaa')
                        r = r_cpu.to('sdaa') if with_residual else None
                        raw_out = torch.empty_like(x)
                        raw_r = torch.empty_like(x) if with_residual else None
                        tecoops.rms_norm(x, w, r, raw_out, raw_r, EPS)
                        opaque = run_norm(x, w, r)
                        graph_out = compiled(x, w, r)
                    stream.synchronize()
                    actual = opaque[0] if with_residual else opaque
                    comp = graph_out[0] if with_residual else graph_out
                    assert torch.equal(actual.cpu().view(torch.int16), raw_out.cpu().view(torch.int16))
                    assert torch.equal(comp.cpu().view(torch.int16), raw_out.cpu().view(torch.int16))
                    summed = x_cpu.float() + r_cpu.float() if with_residual else x_cpu.float()
                    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + EPS) * w_cpu.float()).half()
                    torch.testing.assert_close(actual.cpu(), expected, atol=ATOL, rtol=RTOL)
                    if with_residual:
                        for actual_r in (opaque[1], graph_out[1]):
                            assert torch.equal(actual_r.cpu().view(torch.int16), raw_r.cpu().view(torch.int16))
                            torch.testing.assert_close(actual_r.cpu(), summed.half(), atol=ATOL, rtol=RTOL)
                    assert torch.equal(x.cpu().view(torch.int16), x_cpu.view(torch.int16))
                    assert torch.equal(w.cpu().view(torch.int16), w_cpu.view(torch.int16))
                    if with_residual:
                        assert torch.equal(r.cpu().view(torch.int16), r_cpu.view(torch.int16))
                    records.append(dict(stream=stream_name, tokens=tokens, rows=rows, width=width, residual=with_residual,
                                        raw_bitexact=True, compiled_bitexact=True,
                                        reference_max_abs=float((actual.cpu().float()-expected.float()).abs().max()),
                                        residual_cpu_bit_mismatches=int((opaque[1].cpu().view(torch.int16) != summed.half().view(torch.int16)).sum()) if with_residual else 0,
                                        residual_reference_max_abs=float((opaque[1].cpu().float()-summed.half().float()).abs().max()) if with_residual else 0))
                    print('PASS', records[-1], flush=True)
    with FakeTensorMode():
        x = torch.empty(8, 2048, dtype=torch.float16, device='sdaa')
        w = torch.empty(2048, dtype=torch.float16, device='sdaa')
        out = run_norm(x, w)
        assert out.shape == x.shape and out.device == x.device
        out, res = run_norm(x, w, torch.empty_like(x))
        assert out.shape == res.shape == x.shape
    try:
        run_norm(torch.randn(1, 2048).half(), torch.ones(2048).half())
    except NotImplementedError:
        cpu_rejected = True
    else:
        raise AssertionError('SDAA-only opaque op accepted CPU')
assert any('hy_mt2_public.rms_norm' in graph for graph in graphs)
assert any('hy_mt2_public.rms_norm_add' in graph for graph in graphs)
root = Path(__file__).resolve().parents[1]
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*.py') if '__pycache__' not in str(p)}
result = dict(stage='v2_rms_opaque_boundary', backend='vllm-sdaa', pass_count=len(records), records=records,
              graph_count=len(graphs), fake_pass=True, cpu_rejected=cpu_rejected, atol=ATOL, rtol=RTOL,
              graphs=graphs, source_sha256=manifest, claim='compile compatibility only, no speedup claim')
(OUT / 'focused.json').write_text(json.dumps(result, indent=2) + '\n')
print('SUMMARY', {k: result[k] for k in ('pass_count', 'graph_count', 'fake_pass', 'cpu_rejected')}, flush=True)
