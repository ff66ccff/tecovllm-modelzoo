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

"""Process initialization overlay for TecoVLLM on SDAA accelerator.

Binds custom_ops adaptations before engine initialization to avoid
hot-path branching and keep vendor site-packages strictly read-only.
"""

import sys

try:
    from custom_ops.compile_safe import official_ops
except Exception as exc:
    sys.stderr.write(f"FATAL: [custom_ops] Failed to import compile-safe bindings: {exc}\n")
    sys.exit(1)


def install_llama_rms_norm_overlay() -> None:
    """Bind Llama's RMSNorm construction to the official tecoops API."""
    try:
        import torch
        import torch.nn as nn
        import vllm.model_executor.models.llama as llama_model
        import tecoops
    except Exception as exc:
        sys.stderr.write(f"FATAL: [custom_ops] Failed to import RMSNorm overlay dependencies: {exc}\n")
        sys.exit(1)

    if not hasattr(tecoops, "rms_norm"):
        sys.stderr.write("FATAL: [custom_ops] tecoops.rms_norm is unavailable\n")
        sys.exit(1)
    for required in ("reshape_and_cache", "flash_attn_varlen_func"):
        if not hasattr(tecoops, required):
            sys.stderr.write(f"FATAL: [custom_ops] tecoops.{required} is unavailable\n")
            sys.exit(1)

    # The official wheel exposes raw pybind11 functions.  Registering them with
    # torch.compiler.allow_in_graph is NOT sufficient: Dynamo still inlines them
    # and their data_ptr() access explodes on FakeTensor during torch.compile.
    # Wrap them as opaque torch.library custom ops (registered once, here) so
    # vLLM's default compilation mode is FakeTensor-safe.
    ops = official_ops(tecoops)
    if ops is None:
        sys.stderr.write("FATAL: [custom_ops] Failed to register opaque tecoops custom ops\n")
        sys.exit(1)

    # Bind the official callables exactly once, during process initialization.
    # With TECOOPS_CALL_RECEIPT unset this returns the op object itself, so
    # ``forward`` contains no receipt branch and no getenv call.
    rms_norm_bound = ops.rms_norm
    rms_norm_add_bound = ops.rms_norm_add

    class TecoopsRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6, **_: object) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.variance_epsilon = eps
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.get_default_dtype()))

        def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
            x2 = x.reshape(-1, x.shape[-1])
            out2 = torch.empty_like(x2)
            eps = float(self.variance_epsilon)
            if residual is None:
                rms_norm_bound(x2, self.weight.data, out2, eps)
                return out2.reshape_as(x)
            residual_out2 = torch.empty_like(x2)
            rms_norm_add_bound(
                x2,
                self.weight.data,
                residual.reshape_as(x2),
                out2,
                residual_out2,
                eps,
            )
            return out2.reshape_as(x), residual_out2.reshape_as(x)

    llama_model.RMSNorm = TecoopsRMSNorm
    sys.stderr.write("[custom_ops] Llama RMSNorm overlay bound to tecoops.rms_norm.\n")


def install_block_attention_overlay() -> None:
    """Installs the SDAA BlockAttention forward overlay.

    Fails closed: any import error, missing target attribute, assignment
    failure, or identity check failure immediately terminates the process
    with a non-zero exit code.
    """
    try:
        import vllm_sdaa.attention.block_attn as block_attn_module
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import target module 'vllm_sdaa.attention.block_attn': {exc}\n"
        )
        sys.exit(1)

    if not hasattr(block_attn_module, "BlockAttentionImpl"):
        sys.stderr.write(
            "FATAL: [custom_ops] Target class 'BlockAttentionImpl' not found in 'vllm_sdaa.attention.block_attn'\n"
        )
        sys.exit(1)

    impl_cls = block_attn_module.BlockAttentionImpl
    if not hasattr(impl_cls, "forward"):
        sys.stderr.write(
            "FATAL: [custom_ops] Target attribute 'forward' not found in 'BlockAttentionImpl'\n"
        )
        sys.exit(1)

    try:
        from custom_ops.block_attention.op import sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import 'sdaa_block_attention_forward': {exc}\n"
        )
        sys.exit(1)

    try:
        impl_cls.forward = sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to bind forward method onto BlockAttentionImpl: {exc}\n"
        )
        sys.exit(1)

    if getattr(impl_cls, "forward", None) is not sdaa_block_attention_forward:
        sys.stderr.write(
            "FATAL: [custom_ops] Post-install identity check failed: BlockAttentionImpl.forward is not sdaa_block_attention_forward\n"
        )
        sys.exit(1)

    sys.stderr.write("[custom_ops] BlockAttention overlay bound successfully.\n")


try:
    install_llama_rms_norm_overlay()
    install_block_attention_overlay()
except Exception as exc:
    # Python normally reports ordinary sitecustomize exceptions and continues
    # to the user entrypoint. Convert every installation error to fatal exit.
    sys.stderr.write(f"FATAL: [custom_ops] Overlay initialization failed: {exc}\n")
    sys.exit(1)
