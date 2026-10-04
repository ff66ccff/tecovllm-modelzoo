"""InternVL constructor-only existing-vendor RoPE binding; no new UAL/kernel."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

EXPECTED_EXTENSION = '9ecc2cb57a5c870704580e39112c1e95321bc076507139d8b8855273a90cfec0'
EXPECTED_CORE = 'adf3014a8fd3ced720bbf5cfe666752b8583dc76b17b5c96517fc70716773cb1'
_OP = None
_MANIFEST = None
_CACHE_HASHES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def libraries():
    global _MANIFEST
    if _MANIFEST is not None:
        return _MANIFEST
    import torch_sdaa
    import vllm_sdaa._C as extension
    extension_sha = sha(extension.__file__)
    if not (extension_sha == EXPECTED_EXTENSION):
        raise RuntimeError('vendor extension SHA mismatch')
    paths = sorted({str(Path(line.split()[-1]).resolve())
                    for line in Path('/proc/self/maps').read_text().splitlines()
                    if 'libtecovllm_sdaa._C.so' in line and line.split()[-1].startswith('/')})
    if not (len(paths) == 1):
        raise RuntimeError(f'expected one installed vendor core: {paths}')
    if not (sha(paths[0]) == EXPECTED_CORE):
        raise RuntimeError('vendor core SHA mismatch')
    _MANIFEST = {'extension': extension.__file__, 'extension_sha256': extension_sha,
                 'core': paths[0], 'core_sha256': EXPECTED_CORE,
                 'schema': str(torch.ops._C.rotary_embedding.default._schema)}
    return _MANIFEST


def register_op():
    global _OP
    if _OP is not None:
        return _OP
    libraries()
    raw = torch.ops._C.rotary_embedding.default

    @torch.library.custom_op('internvl_private_rope::apply', mutates_args=('query','key'))
    def opaque(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
               head_size: int, cache: torch.Tensor, is_neox: bool) -> None:
        raw(positions, query, key, head_size, cache, is_neox)

    @opaque.register_fake
    def fake(positions, query, key, head_size, cache, is_neox):
        return None

    _OP = opaque
    return opaque


def fused_forward(self, positions, query, key):
    _OP(positions, query, key, self.head_size, self.cos_sin_cache, self.is_neox_style)
    return query, key


def private_rope(original, weight):
    """Called only at construction; preserve cached original and all tensor data."""
    if not (type(original) is RotaryEmbedding):
        raise RuntimeError('nonstandard RotaryEmbedding')
    if not (original.head_size == original.rotary_dim == 128):
        raise RuntimeError('RoPE initialization/test check failed: original.head_size == original.rotary_dim == 128')
    if not (original.is_neox_style is True and original.dtype == torch.float16):
        raise RuntimeError('RoPE initialization/test check failed: original.is_neox_style is True and original.dtype == torch.float16')
    if not (original.base == 1000000. and original.max_position_embeddings == 40960):
        raise RuntimeError('RoPE initialization/test check failed: original.base == 1000000. and original.max_position_embeddings == 40960')
    if not (weight.dtype == torch.float16 and weight.device.type == 'sdaa'):
        raise RuntimeError("RoPE initialization/test check failed: weight.dtype == torch.float16 and weight.device.type == 'sdaa'")
    if not (weight.device.index == torch.sdaa.current_device()):
        raise RuntimeError('constructor device mismatch')
    cache = original.cos_sin_cache
    if not (cache.device == weight.device and cache.dtype == torch.float16):
        raise RuntimeError('RoPE initialization/test check failed: cache.device == weight.device and cache.dtype == torch.float16')
    if not (tuple(cache.shape) == (40960,128) and cache.is_contiguous()):
        raise RuntimeError('RoPE initialization/test check failed: tuple(cache.shape) == (40960,128) and cache.is_contiguous()')
    if not (cache.stride() == (128,1)):
        raise RuntimeError('RoPE initialization/test check failed: cache.stride() == (128,1)')
    register_op()
    old_method = original._forward_method
    old_buffers, old_modules, old_parameters = original._buffers, original._modules, original._parameters
    private = copy.copy(original)
    private._buffers = dict(old_buffers)
    private._modules = dict(old_modules)
    private._parameters = dict(old_parameters)
    private._forward_method = MethodType(fused_forward, private)
    if not (private is not original and private.cos_sin_cache is cache):
        raise RuntimeError('RoPE initialization/test check failed: private is not original and private.cos_sin_cache is cache')
    if not (original._forward_method is old_method):
        raise RuntimeError('RoPE initialization/test check failed: original._forward_method is old_method')
    if not (original._buffers is old_buffers and original._modules is old_modules):
        raise RuntimeError('RoPE initialization/test check failed: original._buffers is old_buffers and original._modules is old_modules')
    if not (original._parameters is old_parameters):
        raise RuntimeError('RoPE initialization/test check failed: original._parameters is old_parameters')
    if not (private._buffers is not old_buffers and private._modules is not old_modules):
        raise RuntimeError('RoPE initialization/test check failed: private._buffers is not old_buffers and private._modules is not old_modules')
    if not (private._parameters is not old_parameters):
        raise RuntimeError('RoPE initialization/test check failed: private._parameters is not old_parameters')
    return private


def receipt(original, private, attention):
    cache = original.cos_sin_cache
    # Once per distinct original tensor, during initialization only; hash, not data dump.
    key = id(cache)
    if key not in _CACHE_HASHES:
        _CACHE_HASHES[key] = hashlib.sha256(cache.detach().cpu().numpy().tobytes()).hexdigest()
    tp_rank = None
    if torch.distributed.is_initialized():
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
        except (AssertionError, RuntimeError):
            pass  # No TP group in standalone initialization tests.
    record = {'event':'INTERNVL_PRIVATE_ROPE_BIND', 'pid':os.getpid(),
              'tp_rank':tp_rank, 'current_device':torch.sdaa.current_device(), 'attention_id':id(attention),
              'original_rope_id':id(original), 'private_rope_id':id(private),
              'cache_id':key, 'cache_pointer':cache.data_ptr(), 'cache_sha256':_CACHE_HASHES[key],
              'cache_shape':list(cache.shape), 'cache_stride':list(cache.stride()),
              'dtype':str(cache.dtype), 'device':str(cache.device), 'base':original.base,
              'head_size':128, 'rotary_dim':128, 'heads':16, 'kv_heads':4,
              'original_buffers_id':id(original._buffers), 'private_buffers_id':id(private._buffers),
              'original_modules_id':id(original._modules), 'private_modules_id':id(private._modules),
              'original_method':getattr(original._forward_method,'__qualname__',str(type(original._forward_method))),
              'libraries':libraries()}
    sys.stderr.write(json.dumps(record,sort_keys=True)+'\n')


def constructor_hook(original_init):
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not (self.num_heads == 16 and self.num_kv_heads == 4 and self.head_dim == 128):
            raise RuntimeError('RoPE initialization/test check failed: self.num_heads == 16 and self.num_kv_heads == 4 and self.head_dim == 128')
        if not (self.q_size == 2048 and self.kv_size == 512):
            raise RuntimeError('RoPE initialization/test check failed: self.q_size == 2048 and self.kv_size == 512')
        original = self.rotary_emb
        private = private_rope(original, self.qkv_proj.weight)
        self.rotary_emb = private
        receipt(original, private, self)
    return initialize


def install():
    from vllm.model_executor.models.qwen3 import Qwen3Attention
    if not (not getattr(Qwen3Attention, '_internvl_private_rope_installed', False)):
        raise RuntimeError('duplicate private install')
    register_op()
    original_init = Qwen3Attention.__init__
    Qwen3Attention.__init__ = constructor_hook(original_init)
    Qwen3Attention._internvl_private_rope_installed = True
    sys.stderr.write(json.dumps({'event':'INTERNVL_PRIVATE_ROPE_INSTALL', 'pid':os.getpid(), 'libraries':libraries()})+'\n')