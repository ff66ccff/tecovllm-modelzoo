#!/usr/bin/env python3
"""Root-run isolation/Fake/fullgraph gate; no model and no timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import torch
import torch_sdaa
import custom_ops.rotary.op as binding
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def expect_rejected(action):
    try:
        action()
    except RuntimeError:
        return
    raise RuntimeError('initialization guard accepted an invalid fixture')


def fixture(cache):
    original=object.__new__(RotaryEmbedding)
    torch.nn.Module.__init__(original)
    original.head_size=original.rotary_dim=128
    original.max_position_embeddings=40960
    original.base=1000000.
    original.dtype=torch.float16
    original.is_neox_style=True
    original.register_buffer('cos_sin_cache',cache,persistent=False)
    original._forward_method=original.forward_native
    return original


def run_cpu_guard_checks():
    cache=torch.empty((40960,128),dtype=torch.float16,device='cpu')
    original=fixture(cache)
    weight=torch.empty((1,),dtype=torch.float16,device='cpu')
    expect_rejected(lambda: binding.private_rope(torch.nn.Identity(),weight))
    for name,value in (('head_size',64),('rotary_dim',64),('is_neox_style',False),
                       ('dtype',torch.float32),('base',10000.),('max_position_embeddings',4096)):
        wrong=fixture(cache)
        setattr(wrong,name,value)
        expect_rejected(lambda: binding.private_rope(wrong,weight))
    expect_rejected(lambda: binding.private_rope(original,weight.float()))
    expect_rejected(lambda: binding.private_rope(original,weight))


def run_device_guard_checks(original,weight):
    cache=original.cos_sin_cache
    for wrong_cache in (cache.float(),cache[:4096],cache.t().contiguous().t(),cache.cpu()):
        wrong=fixture(wrong_cache)
        expect_rejected(lambda: binding.private_rope(wrong,weight))
    # Constructor guards must also fail under -O, after original constructor runs.
    def malformed(self):
        self.num_heads,self.num_kv_heads,self.head_dim=8,4,128
        self.q_size,self.kv_size=1024,512
    expect_rejected(lambda: binding.constructor_hook(malformed)(SimpleNamespace()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--guard-only',action='store_true',help='CPU initialization rejection checks, also under -O')
    parser.add_argument('--result',type=Path,default=Path(__file__).with_name('focused_results.json'))
    args=parser.parse_args()
    result={'status':'running','cases':[]}
    def save():
        args.result.write_text(json.dumps(result,indent=2)+'\n')
    try:
        spec=importlib.util.spec_from_file_location('frozen_rope_capability',
            str(Path(__file__).with_name('verify_vendor_rope.py')))
        cap=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cap)
        run_cpu_guard_checks()
        result['cpu_guard_rejections']='passed'
        if args.guard_only:
            result['status']='passed'
            save()
            print(json.dumps(result),flush=True)
            return
        torch.sdaa.set_device(args.device)
        device=torch.device(f'sdaa:{args.device}')
        cache=cap.cache_cpu(1000000.)
        # Exact vendor class, synthetic construction avoids initializing any model.
        original=object.__new__(RotaryEmbedding)
        torch.nn.Module.__init__(original)
        original.head_size=original.rotary_dim=128
        original.max_position_embeddings=40960
        original.base=1000000.
        original.dtype=torch.float16
        original.is_neox_style=True
        original.register_buffer('cos_sin_cache',cache.to(device),persistent=False)
        original.apply_rotary_emb=torch.nn.Identity()
        original._forward_method=original.forward_native
        before_method=original._forward_method
        before_buffers,before_modules,before_parameters=original._buffers,original._modules,original._parameters
        before_bits=original.cos_sin_cache.cpu().view(torch.int16).clone()
        weight=torch.empty((1,),dtype=torch.float16,device=device)
        run_device_guard_checks(original,weight)
        result['device_guard_rejections']='passed'
        first=binding.private_rope(original,weight)
        second=binding.private_rope(original,weight)
        if not (first is not second and first is not original):
            raise RuntimeError('RoPE initialization/test check failed: first is not second and first is not original')
        if not (first.cos_sin_cache is second.cos_sin_cache is original.cos_sin_cache):
            raise RuntimeError('RoPE initialization/test check failed: first.cos_sin_cache is second.cos_sin_cache is original.cos_sin_cache')
        first._buffers['isolation_marker']=torch.empty((0,),device=device)
        first._modules['isolation_marker']=torch.nn.Identity()
        if not ('isolation_marker' not in original._buffers and 'isolation_marker' not in second._buffers):
            raise RuntimeError("RoPE initialization/test check failed: 'isolation_marker' not in original._buffers and 'isolation_marker' not in second._buffers")
        if not ('isolation_marker' not in original._modules and 'isolation_marker' not in second._modules):
            raise RuntimeError("RoPE initialization/test check failed: 'isolation_marker' not in original._modules and 'isolation_marker' not in second._modules")
        if not (original._forward_method is before_method):
            raise RuntimeError('RoPE initialization/test check failed: original._forward_method is before_method')
        if not (original._buffers is before_buffers and original._modules is before_modules):
            raise RuntimeError('RoPE initialization/test check failed: original._buffers is before_buffers and original._modules is before_modules')
        if not (original._parameters is before_parameters):
            raise RuntimeError('RoPE initialization/test check failed: original._parameters is before_parameters')
        # Exercise the exact hook helper with a tiny constructor stub, not a model.
        attention=SimpleNamespace()
        def stub(self):
            self.original_constructor_completed=True
            self.num_heads,self.num_kv_heads,self.head_dim=16,4,128
            self.q_size,self.kv_size=2048,512
            self.qkv_proj=SimpleNamespace(weight=weight)
            self.rotary_emb=original
        binding.constructor_hook(stub)(attention)
        if not (attention.original_constructor_completed and attention.rotary_emb is not original):
            raise RuntimeError('RoPE initialization/test check failed: attention.original_constructor_completed and attention.rotary_emb is not original')
        if not (attention.rotary_emb.cos_sin_cache is original.cos_sin_cache):
            raise RuntimeError('RoPE initialization/test check failed: attention.rotary_emb.cos_sin_cache is original.cos_sin_cache')
        if not (original._forward_method is before_method):
            raise RuntimeError('RoPE initialization/test check failed: original._forward_method is before_method')
        result['isolation']='passed'
        result['libraries']=binding.libraries()
        result['cache_identity_preserved']=True
        save()
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode(allow_non_fake_inputs=True):
            p=torch.empty((1,),dtype=torch.int64,device=device)
            q=torch.empty((1,2048),dtype=torch.float16,device=device)
            k=torch.empty((1,512),dtype=torch.float16,device=device)
            c=torch.empty((40960,128),dtype=torch.float16,device=device)
            if not (binding.register_op()(p,q,k,128,c,True) is None):
                raise RuntimeError('RoPE initialization/test check failed: binding.register_op()(p,q,k,128,c,True) is None')
            qo,ko=attention.rotary_emb(p,q,k)
            if not (qo is q and ko is k and qo.shape==(1,2048) and ko.shape==(1,512)):
                raise RuntimeError('RoPE initialization/test check failed: qo is q and ko is k and qo.shape==(1,2048) and ko.shape==(1,512)')
        result['fake']='passed'
        result['opaque_schema']=str(binding.register_op()._schema)
        nondefault=torch.sdaa.Stream(device=device)
        def actual(p,q,k,head,c,neox):
            qo,ko=attention.rotary_emb(p,q,k)
            if not (qo is q and ko is k):
                raise RuntimeError('RoPE initialization/test check failed: qo is q and ko is k')
        for m,mode in ((1,17),(1,4095),(8,'mixed')):
            for stream in (None,nondefault):
                record=cap.run_case(device,stream,m,'random',mode,cache,invoke=actual)
                result['cases'].append(record)
                save()
        # Runs only after all actual module wrapper arithmetic/guard cases pass.
        compiled=torch.compile(attention.rotary_emb,backend='eager',fullgraph=True)
        def actual_compiled(p,q,k,head,c,neox):
            qo,ko=compiled(p,q,k)
            if not (qo is q and ko is k):
                raise RuntimeError('RoPE initialization/test check failed: qo is q and ko is k')
        for m,mode in ((1,17),(1,4095),(8,'mixed')):
            for stream in (None,nondefault):
                record=cap.run_case(device,stream,m,'random',mode,cache,invoke=actual_compiled,compiled=True)
                result['cases'].append(record)
                save()
        if not (torch.equal(original.cos_sin_cache.cpu().view(torch.int16),before_bits)):
            raise RuntimeError('RoPE initialization/test check failed: torch.equal(original.cos_sin_cache.cpu().view(torch.int16),before_bits)')
        if not (original._forward_method is before_method):
            raise RuntimeError('RoPE initialization/test check failed: original._forward_method is before_method')
        if not (original._buffers is before_buffers and original._modules is before_modules):
            raise RuntimeError('RoPE initialization/test check failed: original._buffers is before_buffers and original._modules is before_modules')
        result['original_cache_bits']='unchanged'
        result['status']='passed'
        save()
        print(json.dumps(result),flush=True)
    except Exception as exc:
        result['status']='failed'
        result['error']=str(exc)
        result['traceback']=traceback.format_exc()
        save()
        raise

if __name__=='__main__':
    main()