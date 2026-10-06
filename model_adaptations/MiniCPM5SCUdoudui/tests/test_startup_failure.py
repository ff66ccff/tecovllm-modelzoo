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

"""Start-up rejection checks with stdlib-only module stubs and -S children."""
from pathlib import Path
import os
import subprocess
import sys

bundle = Path(__file__).resolve().parents[1]
overlay = bundle/'runtime/overlay/sitecustomize.py'
harness = r'''
import runpy, sys, types
mode = sys.argv[1]
compile_mod = types.ModuleType('custom_ops.compile_safe')
def register(module):
    if mode == 'registration': raise RuntimeError('injected registration failure')
    return types.SimpleNamespace(rms_norm=lambda *a: None, rms_norm_add=lambda *a: None)
compile_mod.official_ops = register
sys.modules['custom_ops.compile_safe'] = compile_mod
torch = types.ModuleType('torch'); nn = types.ModuleType('torch.nn')
torch.Tensor = object; torch.nn = nn; nn.Module = object
sys.modules['torch'] = torch; sys.modules['torch.nn'] = nn
llama = types.ModuleType('vllm.model_executor.models.llama')
for name in ('vllm','vllm.model_executor','vllm.model_executor.models'):
    module = types.ModuleType(name); module.__path__ = []; sys.modules[name] = module
sys.modules['vllm.model_executor.models.llama'] = llama
teco = types.ModuleType('tecoops')
for name in ('rms_norm','reshape_and_cache','flash_attn_varlen_func'):
    if mode != 'missing_' + name: setattr(teco, name, lambda *a: None)
sys.modules['tecoops'] = teco
for name in ('vllm_sdaa','vllm_sdaa.attention'):
    module = types.ModuleType(name); module.__path__ = []; sys.modules[name] = module
block = types.ModuleType('vllm_sdaa.attention.block_attn')
if mode == 'identity':
    class Meta(type):
        def __setattr__(cls, name, value): pass
    class Impl(metaclass=Meta):
        def forward(self): pass
elif mode == 'missing_forward':
    class Impl: pass
else:
    class Impl:
        def forward(self): pass
if mode != 'missing_class': block.BlockAttentionImpl = Impl
sys.modules['vllm_sdaa.attention.block_attn'] = block
op = types.ModuleType('custom_ops.block_attention.op')
op.sdaa_block_attention_forward = lambda *a: None
sys.modules['custom_ops.block_attention.op'] = op
runpy.run_path(sys.argv[2])
print('ENTRYPOINT_REACHED')
'''
for case in ('success', 'registration', 'missing_rms_norm', 'missing_reshape_and_cache',
             'missing_flash_attn_varlen_func', 'missing_class', 'missing_forward', 'identity'):
    result = subprocess.run([sys.executable, '-S', '-c', harness, case, str(overlay)], capture_output=True, text=True, timeout=20)
    if case == 'success':
        assert result.returncode == 0 and 'ENTRYPOINT_REACHED' in result.stdout, result.stderr
    else:
        assert result.returncode != 0 and 'FATAL:' in result.stderr and 'ENTRYPOINT_REACHED' not in result.stdout, (case, result)
    print(f'{case}: PASS')

env = dict(os.environ, MINICPM_OP_PROFILE='unknown')
result = subprocess.run(['bash', str(overlay.parents[2]/'run.sh')], env=env, capture_output=True, text=True, timeout=20)
assert result.returncode != 0 and 'unknown MINICPM_OP_PROFILE' in result.stderr, result
print('unsupported_profile: PASS')
