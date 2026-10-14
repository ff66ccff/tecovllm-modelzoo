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

# ---------------------------------------------------------------------------
# L11 DELIBERATE VARIANT (cycle-22 terminal check, 2026-10-13).
# This delivered copy is an INTENTIONAL variant of the repository-root
# `custom_ops/overlay/sitecustomize.py`; it is NOT a stale duplicate and it must
# NOT be "synchronised" into the same blob.  L11's general rule (same filename =>
# same blob) has exactly one recorded exception here, and this is it.  Measured
# differences (diff = 92 lines):
#   1. this BSD-3 header -- the root copy has no header;
#   2. the `custom_ops.compile_safe` import is wrapped so any failure aborts the
#      process with `SystemExit(1)` (fail-closed, L15) instead of raising a plain
#      exception that CPython's site.py would swallow (silent degradation);
#   3. the root copy additionally wraps the bound official callables with
#      `custom_ops.receipt.bind_official_api` and prints the selected operator
#      profile -- both are line-internal diagnostics that must be ABSENT from the
#      delivery, whose hot path stays branch-free and getenv-free (AGENTS rule 6);
#   4. the SiLU path resolves `runtime_custom_ops` as `<delivery>/runtime/custom_ops`
#      rather than the root copy's `<delivery>/runtime`.
# ---------------------------------------------------------------------------

"""Process initialization overlay for TecoVLLM on SDAA accelerator.

Binds custom_ops adaptations before engine initialization to avoid
hot-path branching and keep vendor site-packages strictly read-only.
"""

import sys

try:
    from custom_ops.compile_safe import official_ops
except Exception as exc:
    sys.stderr.write(f"FATAL: [custom_ops] Failed to import compile-safe bindings: {exc}\n")
    raise SystemExit(1)


def install_llama_rms_norm_overlay() -> None:
    """Bind Llama's RMSNorm construction to the official tecoops API."""
    try:
        import torch
        import torch.nn as nn
        import vllm.model_executor.models.llama as llama_model
        import tecoops
    except Exception as exc:
        sys.stderr.write(f"FATAL: [custom_ops] Failed to import RMSNorm overlay dependencies: {exc}\n")
        raise SystemExit(1)

    if not hasattr(tecoops, "rms_norm"):
        sys.stderr.write("FATAL: [custom_ops] tecoops.rms_norm is unavailable\n")
        raise SystemExit(1)
    for required in ("reshape_and_cache", "flash_attn_varlen_func"):
        if not hasattr(tecoops, required):
            sys.stderr.write(f"FATAL: [custom_ops] tecoops.{required} is unavailable\n")
            raise SystemExit(1)

    # The official wheel exposes raw pybind11 functions.  Registering them with
    # torch.compiler.allow_in_graph is NOT sufficient: Dynamo still inlines them
    # and their data_ptr() access explodes on FakeTensor during torch.compile.
    # Wrap them as opaque torch.library custom ops (registered once, here) so
    # vLLM's default compilation mode is FakeTensor-safe.
    ops = official_ops(tecoops)
    if ops is None:
        sys.stderr.write("FATAL: [custom_ops] Failed to register opaque tecoops custom ops\n")
        raise SystemExit(1)

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
        raise SystemExit(1)

    if not hasattr(block_attn_module, "BlockAttentionImpl"):
        sys.stderr.write(
            "FATAL: [custom_ops] Target class 'BlockAttentionImpl' not found in 'vllm_sdaa.attention.block_attn'\n"
        )
        raise SystemExit(1)

    impl_cls = block_attn_module.BlockAttentionImpl
    if not hasattr(impl_cls, "forward"):
        sys.stderr.write(
            "FATAL: [custom_ops] Target attribute 'forward' not found in 'BlockAttentionImpl'\n"
        )
        raise SystemExit(1)

    try:
        from custom_ops.block_attention.op import sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import 'sdaa_block_attention_forward': {exc}\n"
        )
        raise SystemExit(1)

    try:
        impl_cls.forward = sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to bind forward method onto BlockAttentionImpl: {exc}\n"
        )
        raise SystemExit(1)

    if getattr(impl_cls, "forward", None) is not sdaa_block_attention_forward:
        sys.stderr.write(
            "FATAL: [custom_ops] Post-install identity check failed: BlockAttentionImpl.forward is not sdaa_block_attention_forward\n"
        )
        raise SystemExit(1)

    sys.stderr.write("[custom_ops] BlockAttention overlay bound successfully.\n")



def install_silu_mul_vendor_overlay() -> None:
    """Bind optional fused SWIGLU once before vLLM model construction."""
    import os as _os

    profile = _os.environ.get("MINICPM_SILU_MUL_PROFILE", "official")
    if profile == "official":
        return
    if profile != "vendor_swiglu":
        sys.stderr.write(
            f"FATAL: [custom_ops] Unknown MINICPM_SILU_MUL_PROFILE={profile!r}; "
            "expected 'official' or 'vendor_swiglu'\n"
        )
        raise SystemExit(1)

    attention_profile = _os.environ.get("MINICPM_OP_PROFILE", "official")
    if attention_profile != "official":
        sys.stderr.write(
            "FATAL: [custom_ops] vendor_swiglu currently requires MINICPM_OP_PROFILE=official\n"
        )
        raise SystemExit(1)

    validated_dtype = _os.environ.get("MINICPM_SILU_MUL_DTYPE_VALIDATED")
    dtype_values = []
    for index, argument in enumerate(sys.argv):
        if argument == "--dtype":
            if index + 1 >= len(sys.argv):
                sys.stderr.write("FATAL: [custom_ops] vendor_swiglu requires --dtype float16\n")
                raise SystemExit(1)
            dtype_values.append(sys.argv[index + 1].lower())
        elif argument.startswith("--dtype="):
            dtype_values.append(argument.split("=", 1)[1].lower())
    if dtype_values and any(value not in {"float16", "half"} for value in dtype_values):
        sys.stderr.write("FATAL: [custom_ops] vendor_swiglu only supports explicit FP16 dtype\n")
        raise SystemExit(1)
    if validated_dtype not in (None, "fp16"):
        sys.stderr.write("FATAL: [custom_ops] invalid inherited vendor_swiglu dtype validation\n")
        raise SystemExit(1)
    if not dtype_values and validated_dtype != "fp16":
        sys.stderr.write("FATAL: [custom_ops] vendor_swiglu requires explicit FP16 dtype\n")
        raise SystemExit(1)
    if dtype_values:
        # vLLM starts engine workers as new Python processes whose argv may no
        # longer contain the parent server's CLI. Propagate the validated dtype
        # so worker imports can bind the same FP16-only kernel.
        _os.environ["MINICPM_SILU_MUL_DTYPE_VALIDATED"] = "fp16"

    try:
        from pathlib import Path
        import torch
        import torch_sdaa  # noqa: F401
        import vllm.model_executor.layers.activation as activation_module
        import custom_ops
        runtime_custom_ops = str((Path(__file__).resolve().parents[1] / "custom_ops").resolve())
        if runtime_custom_ops in custom_ops.__path__:
            custom_ops.__path__.remove(runtime_custom_ops)
        custom_ops.__path__.insert(0, runtime_custom_ops)
        import importlib
        silu_mul_module = importlib.import_module("custom_ops.silu_mul.op")
        module_file = Path(silu_mul_module.__file__).resolve()
        if Path(runtime_custom_ops) not in module_file.parents:
            raise RuntimeError(
                f"public SiluAndMul op was shadowed by {module_file}; expected under {runtime_custom_ops}"
            )
        from custom_ops.silu_mul.op import bind_vendor_silu_mul, silu_mul
    except Exception as exc:
        sys.stderr.write(f"FATAL: [custom_ops] Failed to prepare vendor SiLU x Mul overlay: {exc}\n")
        raise SystemExit(1)

    if not hasattr(activation_module, "SiluAndMul"):
        sys.stderr.write("FATAL: [custom_ops] vLLM SiluAndMul class is unavailable\n")
        raise SystemExit(1)
    try:
        has_vendor_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(
            "sdaa::swiglu", "PrivateUse1"
        )
    except Exception:
        has_vendor_kernel = False
    if not has_vendor_kernel:
        sys.stderr.write("FATAL: [custom_ops] vendor sdaa::swiglu PrivateUse1 kernel is unavailable\n")
        raise SystemExit(1)
    bind_vendor_silu_mul(torch.ops.sdaa.swiglu)

    def _silu_mul_forward(self, x):
        return silu_mul(x)

    activation_module.SiluAndMul.forward = _silu_mul_forward
    if activation_module.SiluAndMul.forward is not _silu_mul_forward:
        sys.stderr.write("FATAL: [custom_ops] SiluAndMul overlay identity check failed\n")
        raise SystemExit(1)
    sys.stderr.write(f"[custom_ops] MiniCPM SiluAndMul opt-in bound to opaque vendor sdaa::swiglu from {module_file}.\n")


def install_rotary_embedding_overlay() -> None:
    """Bind vLLM's rotary embedding to the delivered tecoops kernel.

    Fails closed: the delivered runtime must carry ``tecoops.rotary_embedding``,
    so a missing symbol aborts start-up instead of silently keeping the 17-kernel
    native path.  The per-instance decision is taken once, at construction.
    """
    import os

    # A/B arm selector, read exactly once here (process initialization), like
    # MINICPM_OP_PROFILE: the default is the delivered arm; "off" keeps vLLM's
    # native rotary path for the baseline arm of a same-build A/B run.  Nothing
    # on the forward path reads the environment.
    if os.environ.get("MINICPM_ROTARY_EMBEDDING", "on").lower() == "off":
        sys.stderr.write(
            "[custom_ops] MINICPM_ROTARY_EMBEDDING=off: rotary embedding overlay skipped "
            "(A/B baseline arm).\n"
        )
        return

    try:
        import torch  # noqa: F401
        import tecoops
        import vllm.model_executor.layers.rotary_embedding as rope_module
        from custom_ops.receipt import bind_official_api
        from custom_ops.rotary_embedding.op import (
            bind_instance,
            official_rotary_embedding,
            resolve_rotary_base,
        )
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import rotary embedding overlay dependencies: {exc}\n"
        )
        raise SystemExit(1)

    if not hasattr(tecoops, "rotary_embedding"):
        sys.stderr.write("FATAL: [custom_ops] tecoops.rotary_embedding is unavailable\n")
        raise SystemExit(1)

    op = official_rotary_embedding(tecoops)
    if op is None:
        sys.stderr.write("FATAL: [custom_ops] Failed to register the opaque rotary embedding op\n")
        raise SystemExit(1)

    # With TECOOPS_CALL_RECEIPT unset this returns the op itself, so the forward
    # path carries no counting branch and no getenv; with it set, the tally is the
    # per-arm proof that the fused kernel really ran (A arm: 0 calls).
    op_bound = bind_official_api(op, "tecoops.rotary_embedding")

    target = resolve_rotary_base(rope_module)
    if target is None or not hasattr(target, "__init__"):
        sys.stderr.write(
            "FATAL: [custom_ops] vllm RotaryEmbeddingBase is unavailable (searched the "
            "package attribute, its `base` submodule and the module path)\n"
        )
        raise SystemExit(1)

    debug = os.environ.get("MINICPM_ROTARY_DEBUG", "") == "1"
    count_calls = os.environ.get("MINICPM_ROTARY_COUNT", "") == "1"
    if debug:
        sys.stderr.write(
            "[custom_ops] rotary debug: target=%s.%s get_rope=%s\n"
            % (target.__module__, target.__name__, getattr(rope_module, "get_rope", None))
        )

    original_init = target.__init__
    bound = {"n": 0}
    if count_calls:
        import inspect
        import traceback

        diag = {"forward": 0, "forward_native": 0, "forward_static": 0, "forward_oot": 0,
                "callsites": []}

        def _mk(name, fn, is_static):
            def wrapper(*a, **k):
                diag[name] += 1
                if len(diag["callsites"]) < 4:
                    diag["callsites"].append(
                        ["%s:%s %s" % (f.filename.split("/")[-1], f.lineno, f.name)
                         for f in traceback.extract_stack()[-14:-1]])
                # inside-the-call instrumentation: which instance, which _forward_method,
                # and what the in-process counter reads before/after the call
                if name == "forward" and a:
                    self_obj = a[0]
                    fm = getattr(self_obj, "_forward_method", None)
                    for key, val in (("self_ids", id(self_obj)),
                                     ("fm_ids", id(fm)),
                                     ("fm_names", getattr(fm, "__name__", type(fm).__name__)),
                                     ("fm_fused", bool(getattr(fm, "_minicpm_rotary_fused", False))),
                                     ("calls_before", getattr(self_obj, "_minicpm_rotary_calls", -1))):
                        diag.setdefault(key, []).append(val)
                result = fn(*a, **k)
                if name == "forward" and a:
                    diag.setdefault("calls_after", []).append(
                        getattr(a[0], "_minicpm_rotary_calls", -1))
                return result

            wrapper.__name__ = name
            return staticmethod(wrapper) if is_static else wrapper

        for _name in ("forward", "forward_native", "forward_static", "forward_oot"):
            _orig_static = inspect.getattr_static(target, _name, None)
            _fn = getattr(target, _name, None)
            if _fn is None:
                continue
            setattr(target, _name, _mk(_name, _fn, isinstance(_orig_static, staticmethod)))
        target._minicpm_rotary_diag = diag
        sys.stderr.write("[custom_ops] rotary diag: wrapped %s\n" % list(diag.keys())[:4])


    # Second, independent binding seam: every vLLM rope instance is produced by the
    # `get_rope` factory (llama and most models go through it), and an OOT-registered
    # class would bypass the class-__init__ seam above.  The factory is patched at
    # process initialization, before any model module imports it.
    factory = getattr(rope_module, "get_rope", None)
    if callable(factory):
        def _get_rope(*args, **kwargs):
            inst = factory(*args, **kwargs)
            if bind_instance(inst, op_bound, count=count_calls):
                bound["n"] += 1
            if debug:
                sys.stderr.write("[custom_ops] get_rope -> %s (bound total %d)\n"
                                 % (type(inst).__name__, bound["n"]))
            return inst

        rope_module.get_rope = _get_rope
        if rope_module.get_rope is not _get_rope:
            sys.stderr.write("FATAL: [custom_ops] get_rope overlay identity check failed\n")
            raise SystemExit(1)

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if bind_instance(self, op_bound, count=count_calls):
            bound["n"] += 1

    target.__init__ = _init
    if target.__init__ is not _init:
        sys.stderr.write("FATAL: [custom_ops] rotary embedding overlay identity check failed\n")
        raise SystemExit(1)

    def _summary():
        sys.stderr.write(
            f"[custom_ops] rotary overlay summary: instances_bound={bound['n']} "
            f"count_calls={count_calls}\n"
        )

    import atexit

    atexit.register(_summary)

    sys.stderr.write(
        "[custom_ops] RotaryEmbedding overlay bound to tecoops.rotary_embedding "
        f"(instances bound so far: {bound['n']}).\n"
    )


try:
    install_llama_rms_norm_overlay()
    install_rotary_embedding_overlay()
    install_block_attention_overlay()
    install_silu_mul_vendor_overlay()
except Exception as exc:
    # Python normally reports ordinary sitecustomize exceptions and continues
    # to the user entrypoint. Convert every installation error to fatal exit.
    sys.stderr.write(f"FATAL: [custom_ops] Overlay initialization failed: {exc}\n")
    raise SystemExit(1)
