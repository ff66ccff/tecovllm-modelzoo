# BSD 3-Clause License
# Copyright (c) 2026, Sichuan University SCUdoudui. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
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

"""Compare bounded workspace against an immutable pre-fix inference module."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'sdaa'), required=True)
    parser.add_argument('--non-default-stream', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.baseline.read_bytes()).hexdigest() == (
        '248a96904bd1331f764574a8fbfdcc0a71266e146e5fde3de9097314e6f148e7'
    ), 'baseline must be the immutable pre-fix module from PR head 5ef4690'
    if args.device == 'sdaa':
        import torch_sdaa  # noqa: F401
        assert Path(sys.executable).resolve() == Path('/home/py312/bin/python').resolve()
    source = Path(__file__).resolve().parents[1] / 'runtime/custom_ops/prefill_attention/op.py'
    base = load('workspace_baseline', args.baseline)
    cand = load('workspace_candidate', source)
    result = dict(passed=False, device=args.device, non_default_stream=args.non_default_stream,
                  cases=[], performance_claim=False,
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    backend = torch.sdaa if args.device == 'sdaa' else None
    if args.non_default_stream:
        assert backend is not None, '--non-default-stream requires SDAA'
        stream = backend.Stream()
        backend.set_stream(stream)
    softmax = torch.softmax
    observed = []

    def workspace_softmax(value, *pos, **kw):
        assert kw.get('out') is value, 'softmax must reuse its owned FP32 buffer'
        observed.append(tuple(value.shape))
        return softmax(value, *pos, **kw)

    def measure(function, q, k, v, causal, window):
        if backend:
            backend.synchronize()
            backend.reset_peak_memory_stats()
            before = backend.memory_allocated()
        output = function(q, k, v, causal, .125, True, window)
        if backend:
            backend.synchronize()
            extra = backend.max_memory_allocated() - before
        else:
            extra = None
        return output.cpu(), extra

    try:
        with torch.inference_mode():
            for rows, keys, dim, kvheads, causal, window in [
                (1, 33, 512, 1, False, None),
                (32, 63, 256, 4, True, 1024),
                (63, 129, 256, 4, True, 1024),
                (64, 65, 512, 1, True, None),
                (65, 129, 256, 4, True, 1024),
                (127, 257, 512, 1, True, None),
                (128, 257, 512, 1, True, None),
                (129, 1057, 256, 4, True, 1024),
                (257, 4352, 512, 1, True, None),
                (512, 14848, 512, 1, True, None),
                (512, 16384, 512, 1, True, None),
                (512, 16384, 512, 1, False, None),
                (512, 1535, 256, 4, True, 1024),
            ]:
                q = (torch.randn(1, 8, rows, dim, dtype=torch.float16) * .2).to(args.device)
                k = (torch.randn(1, kvheads, keys, dim, dtype=torch.float16) * .2).to(args.device)
                v = (torch.randn(1, kvheads, keys, dim, dtype=torch.float16) * .2).to(args.device)
                expected, base_peak = measure(base._attend_math, q, k, v, causal, window)
                with patch.object(torch, 'softmax', workspace_softmax):
                    actual, cand_peak = measure(cand._attend_math, q, k, v, causal, window)
                mismatch = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
                row = dict(rows=rows, keys=keys, dim=dim, kvheads=kvheads,
                           causal=causal, window=window, bit_mismatches=mismatch,
                           base_peak_extra_bytes=base_peak, candidate_peak_extra_bytes=cand_peak)
                result['cases'].append(row)
                print(json.dumps(row), flush=True)
                assert mismatch == 0, row
                assert torch.isfinite(actual).all(), row
                if rows == 512 and keys == 16384 and window is None and backend:
                    assert cand_peak < base_peak, row
                del q, k, v, expected, actual
            result['softmax_calls'] = len(observed)
            result['softmax_out_alias_verified'] = True
            result['passed'] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
