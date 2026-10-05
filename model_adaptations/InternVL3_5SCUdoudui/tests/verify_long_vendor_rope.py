#!/usr/bin/env python3
# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
"""Extend the frozen InternVL vendor-RoPE gate to the 4352-token budget."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

SOURCE = Path(__file__).with_name('verify_vendor_rope.py').resolve()
FROZEN_SOURCE_SHA256 = '1c148fb34aef922b793bcc50efbe8b2c36451d4cc1358949abcfc5e5829de2ce'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-device', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_device:
        parser.error('An explicitly scheduled device slot is required')
    assert Path(sys.executable).resolve() == Path('/home/py312/bin/python').resolve()
    frozen = bytes.fromhex(FROZEN_SOURCE_SHA256)
    assert hashlib.sha256(SOURCE.read_bytes()).digest() == frozen, 'Public RoPE verifier changed'
    spec = importlib.util.spec_from_file_location('internvl_frozen_rope_gate', SOURCE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    torch = gate.torch
    assert gate.HQ == 16 and gate.HK == 4 and gate.D == 128
    assert gate.ATOL == .002 and gate.RTOL == .002

    def long_positions(m, mode):
        if isinstance(mode, int):
            return torch.full((m,), mode, dtype=torch.int64)
        pattern = torch.tensor([0, 17, 4095, 4096, 4195, 4351, 4351, 4096], dtype=torch.int64)
        return pattern.repeat((m + len(pattern) - 1) // len(pattern))[:m].contiguous()

    # Only this test module's input factory changes. The installed vendor and
    # public implementation remain untouched; arithmetic/oracle reuse is exact.
    gate.positions_for = long_positions
    result = {
        'passed': False, 'status': 'running',
        'contract': 'TP2 FP16 packed Q16/K4/D128 full Neox, base1e6, cache40960x128',
        'tested_position_range': [0, 4351], 'explicit_positions': [4096, 4195, 4351],
        'rtol': gate.RTOL, 'atol': gate.ATOL, 'raw_cases': [], 'compiled_cases': [],
        'source_path': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'frozen_source_sha256': FROZEN_SOURCE_SHA256,
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_change': False, 'model_run': False, 'performance_claim': False,
        'current_official_PR_head_wheel_equivalence': 'unverified',
    }

    def save():
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + '\n')

    try:
        result['loader'] = gate.manifest()
        torch.sdaa.set_device(args.device)
        device = torch.device(f'sdaa:{args.device}')
        stream = torch.sdaa.Stream(device=device)
        cache = gate.cache_cpu(1000000.)
        cases = [(1, 'random', pos) for pos in (4096, 4195, 4351)]
        cases += [(8, pattern, 'long_mixed') for pattern in ('random', 'signed', 'cancellation', 'subnormal')]
        for m, pattern, mode in cases:
            for selected_stream in (None, stream):
                result['active_case'] = {'phase': 'raw/native/oracle', 'm': m, 'pattern': pattern, 'positions': mode, 'stream': 'nondefault' if selected_stream is not None else 'default'}
                save()
                record = gate.run_case(device, selected_stream, m, pattern, mode, cache)
                result['raw_cases'].append(record)
                save()
                print(json.dumps(result['active_case']), flush=True)
        compiled, schema = gate.compile_wrapper()
        result['opaque_schema'] = schema
        for m, mode in [(1, 4096), (1, 4195), (1, 4351), (8, 'long_mixed')]:
            for selected_stream in (None, stream):
                result['active_case'] = {'phase': 'Fake/fullgraph-eager', 'm': m, 'pattern': 'random', 'positions': mode, 'stream': 'nondefault' if selected_stream is not None else 'default'}
                save()
                record = gate.run_case(device, selected_stream, m, 'random', mode, cache, invoke=compiled, compiled=True)
                result['compiled_cases'].append(record)
                save()
                print(json.dumps(result['active_case']), flush=True)
        assert len(result['raw_cases']) == 14 and len(result['compiled_cases']) == 8
        assert hashlib.sha256(SOURCE.read_bytes()).digest() == frozen
        result.update(passed=True, status='pass', fake_passed=True, public_source_unchanged=True)
        save()
        print(json.dumps({'passed': True, 'raw_cases': 14, 'compiled_cases': 8, 'max_position': 4351}), flush=True)
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if isinstance(exc, gate.GateFailure):
            result['failed_case'] = exc.record
        save()
        raise


if __name__ == '__main__':
    main()
