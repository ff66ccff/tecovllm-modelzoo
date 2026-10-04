#!/usr/bin/env python3
"""Private installed-vendor RoPE capability gate; no timing or model binding."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

import torch
import torch_sdaa  # Device registration only; execution is controlled by the runner.
import vllm_sdaa._C as vendor_extension
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

D, HQ, HK, MAX_POSITION, GUARD = 128, 16, 4, 40960, 32
ATOL = RTOL = 0.002
RAW = torch.ops._C.rotary_embedding.default


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_bits(a, b):
    if a.dtype == torch.float16:
        return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    return torch.equal(a, b)


def manifest():
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if 'libtecovllm_sdaa' in line and line.split()[-1].startswith('/')})
    assert libraries, 'installed vendor core was not loaded'
    return {'torch': torch.__version__, 'vendor_extension': vendor_extension.__file__,
            'extension_sha256': digest(vendor_extension.__file__),
            'core_libraries': [{'path': p, 'sha256': digest(p)} for p in libraries],
            'raw_schema': str(RAW._schema), 'python_realpath': os.path.realpath(os.sys.executable)}


def cache_cpu(base):
    # Invoke the installed cache computation itself, rather than a copied formula.
    spec = SimpleNamespace(base=base, rotary_dim=D, max_position_embeddings=MAX_POSITION)
    spec._compute_inv_freq = lambda b: RotaryEmbedding._compute_inv_freq(spec, b)
    return RotaryEmbedding._compute_cos_sin_cache(spec).to(torch.float16).contiguous()


def oracle(x, positions, cache):
    """Each half multiply narrows before the independent half add/sub."""
    t = x.view(len(positions), -1, D).float()
    selected = cache[positions].float()
    cos, sin = selected[:, :D//2, None], selected[:, D//2:, None]
    cos, sin = cos.transpose(1, 2), sin.transpose(1, 2)
    a, b = t[..., :D//2], t[..., D//2:]
    ac = (a * cos).half().float()
    bs = (b * sin).half().float()
    bc = (b * cos).half().float()
    ass = (a * sin).half().float()
    return torch.cat(((ac-bs).half(), (bc+ass).half()), dim=-1).reshape_as(x)


def positions_for(m, mode):
    if isinstance(mode, int):
        return torch.full((m,), mode, dtype=torch.int64)
    pattern = torch.tensor([0, 17, 4095, 17, 0, 4095], dtype=torch.int64)
    return pattern.repeat((m+len(pattern)-1)//len(pattern))[:m].contiguous()


def inputs(m, pattern, positions, cache):
    generator = torch.Generator().manual_seed(1234+m)
    tensors = []
    for heads in (HQ, HK):
        x = torch.randn((m, heads, D), generator=generator).half()
        if pattern == 'signed':
            values = torch.tensor([0., -0., 1., -1., .5, -.5, 16., -16.]).half()
            x = values.repeat((x.numel()+7)//8)[:x.numel()].reshape_as(x)
        elif pattern == 'cancellation':
            cs = cache[positions].float()
            sign = torch.where(torch.arange(heads).remainder(2)==0, 1., -1.)[None,:,None]
            # Paired terms sin*cos and cos*sin cancel after half-product rounding.
            a = (cs[:,None,D//2:] * sign * 16.).half()
            b = (cs[:,None,:D//2] * sign * 16.).half()
            x = torch.cat((a,b), dim=-1)
        elif pattern == 'subnormal':
            bits = torch.tensor([1, -32767, 1023, -31745, 0, -32768], dtype=torch.int16)
            x = bits.repeat((x.numel()+5)//6)[:x.numel()].view(torch.float16).reshape_as(x)
        tensors.append(x.reshape(m, heads*D).contiguous())
    return tensors


def guarded(cpu, device):
    storage = torch.full((cpu.numel()+2*GUARD,), -19.75, dtype=torch.float16)
    storage[GUARD:-GUARD].copy_(cpu.flatten())
    gpu = storage.to(device)
    view = gpu[GUARD:-GUARD].view_as(cpu)
    assert view.is_contiguous()
    return gpu, view, storage


def check_guard(gpu, before):
    after = gpu.cpu()
    assert same_bits(after[:GUARD], before[:GUARD]), 'prefix guard changed'
    assert same_bits(after[-GUARD:], before[-GUARD:]), 'tail guard changed'


class GateFailure(AssertionError):
    def __init__(self, record):
        self.record = record
        super().__init__(json.dumps(record))


def compare(actual, expected):
    assert torch.isfinite(actual).all(), 'nonfinite output'
    diff = (actual.float()-expected.float()).abs()
    okay = torch.isclose(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)
    return {'max_abs': float(diff.max()), 'failed_elements': int((~okay).sum()),
            'bit_differences': int((actual.view(torch.int16)!=expected.view(torch.int16)).sum()),
            'passed': bool(okay.all())}


def run_case(device, stream, m, pattern, mode, cache, invoke=RAW, compiled=False):
    positions = positions_for(m, mode)
    q, k = inputs(m, pattern, positions, cache)
    expected = [oracle(t, positions, cache) for t in (q,k)]
    context = torch.sdaa.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        pdev, cdev = positions.to(device), cache.to(device)
        qstorage, qdev, qbefore = guarded(q, device)
        kstorage, kdev, kbefore = guarded(k, device)
        qs, ks = qdev.clone(), kdev.clone()  # Bit-preserving same-stream producers.
        if not compiled:
            native = RotaryEmbedding.forward_static(pdev, qs, ks, D, D, cdev, True)
        invoke(pdev, qdev, kdev, D, cdev, True)
    if stream is not None:
        stream.synchronize()
    else:
        torch.sdaa.synchronize()
    assert same_bits(pdev.cpu(), positions), 'positions modified'
    assert same_bits(cdev.cpu(), cache), 'cache modified'
    assert same_bits(qs.cpu(), q) and same_bits(ks.cpu(), k), 'input copy lost bits/native mutated input'
    check_guard(qstorage, qbefore)
    check_guard(kstorage, kbefore)
    results = [compare(qdev.cpu(), expected[0]), compare(kdev.cpu(), expected[1])]
    record = {'m': m, 'pattern': pattern, 'positions': mode,
              'stream': 'nondefault' if stream is not None else 'default',
              'compiled': compiled, 'raw_vs_cpu': results}
    if not compiled:
        record['native_vs_cpu'] = [compare(t.cpu(), gold) for t,gold in zip(native, expected)]
        record['raw_vs_native'] = [compare(t.cpu(), n.cpu()) for t,n in zip((qdev,kdev),native)]
        if not all(r['passed'] for group in ('native_vs_cpu','raw_vs_native') for r in record[group]):
            raise GateFailure(record)
    if not all(r['passed'] for r in results):
        raise GateFailure(record)
    return record


def compile_wrapper():
    @torch.library.custom_op('internvl_rope_probe::apply', mutates_args=('query','key'))
    def opaque(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
               head_size: int, cache: torch.Tensor, is_neox: bool) -> None:
        RAW(positions, query, key, head_size, cache, is_neox)

    @opaque.register_fake
    def fake(positions, query, key, head_size, cache, is_neox):
        return None

    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        p = torch.empty((1,), dtype=torch.int64, device='sdaa')
        q = torch.empty((1,HQ*D), dtype=torch.float16, device='sdaa')
        k = torch.empty((1,HK*D), dtype=torch.float16, device='sdaa')
        c = torch.empty((MAX_POSITION,D), dtype=torch.float16, device='sdaa')
        assert opaque(p,q,k,D,c,True) is None
        assert q.shape == (1,HQ*D) and k.shape == (1,HK*D)
    return torch.compile(opaque, backend='eager', fullgraph=True), str(opaque._schema)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--base', type=float, default=1000000.)
    parser.add_argument('--result', type=Path, default=Path(__file__).with_name('results.json'))
    args = parser.parse_args()
    result = {'status': 'running', 'contract': 'FP16 packed Q16/K4/D128 full Neox; first-order inference',
              'rtol': RTOL, 'atol': ATOL, 'base': args.base, 'cases': []}
    def save():
        args.result.write_text(json.dumps(result, indent=2)+'\n')
    try:
        result['loader'] = manifest()
        result['cache_shape'] = [MAX_POSITION,D]
        result['tested_position_range'] = [0,4095]
        save()
        torch.sdaa.set_device(args.device)
        device = torch.device(f'sdaa:{args.device}')
        nondefault = torch.sdaa.Stream(device=device)
        cache = cache_cpu(args.base)
        # Target gate comes first; failure prevents extended/Fake/compile phases.
        cases = [(1,'random',17), (1,'random',4095)]
        cases += [(m, pattern, 'mixed' if m>1 else 0)
                  for m in (1,8,64,513) for pattern in ('random','signed','cancellation','subnormal')]
        for m,pattern,mode in cases:
            for stream in (None, nondefault):
                result['active_case'] = {'m':m,'pattern':pattern,'positions':mode,'stream':'nondefault' if stream is not None else 'default'}
                save()
                record = run_case(device,stream,m,pattern,mode,cache)
                result['cases'].append(record)
                print(json.dumps(record), flush=True)
                save()
        compiled, schema = compile_wrapper()
        result['opaque_schema'] = schema
        for m in (1,8):
            for stream in (None,nondefault):
                record = run_case(device,stream,m,'random','mixed',cache,invoke=compiled,compiled=True)
                result['cases'].append(record)
                print(json.dumps(record),flush=True)
                save()
        result['status']='passed'
        save()
    except Exception as exc:
        result['status']='failed'
        result['error']=str(exc)
        if isinstance(exc,GateFailure):
            result['failed_case']=exc.record
        result['traceback']=traceback.format_exc()
        save()
        raise

if __name__ == '__main__':
    main()