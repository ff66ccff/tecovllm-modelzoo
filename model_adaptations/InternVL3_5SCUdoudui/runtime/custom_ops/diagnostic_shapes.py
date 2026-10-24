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

"""Optional metadata-only census, installed once for a diagnostic process.

This intentionally disables compilation only around the census wrapper; it is
never installed in ordinary run.sh / official CI unless SCUD_DECODE_CENSUS is set.
Reported shapes come from real SDAA tensors, not a different model's config.
No weights or tensor data are copied into the report.
"""
import json
import os
from pathlib import Path


def install_decode_census():
    destination=os.environ.get("SCUD_DECODE_CENSUS")
    if not destination:
        return
    import torch
    import vllm.model_executor.layers.utils as utils
    original=utils.default_unquantized_gemm
    seen=set()
    root=Path(destination)
    root.mkdir(parents=True,exist_ok=True)
    output=root / ("decode-shapes-%d.jsonl" % os.getpid())

    @torch._dynamo.disable
    def census(layer,x,weight,bias=None):
        if x.device.type=="sdaa" and x.ndim>=2 and x.numel()//x.shape[-1]==1:
            row=dict(input_shape=list(x.shape), weight_shape=list(weight.shape),
                     weight_stride=list(weight.stride()), dtype=str(x.dtype),
                     layer=type(layer).__name__, bias=bias is not None,
                     provenance="observed SDAA M=1 forward; diagnostic process only")
            key=json.dumps(row,sort_keys=True)
            if key not in seen:
                seen.add(key)
                with output.open("a") as stream:
                    stream.write(key+"\n")
        return original(layer,x,weight,bias)
    utils.default_unquantized_gemm=census
