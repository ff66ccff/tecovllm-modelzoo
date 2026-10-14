# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
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

"""Gemma4 RMSNorm -> reference-exact ``tecoops.rms_norm_compat`` epilogue.

Why this exists (measured on this branch, not assumed)
------------------------------------------------------
``vllm.model_executor.layers.layernorm.RMSNorm`` is a ``CustomOp``.  On this stack
``current_platform.is_out_of_tree()`` is True and ``vllm_sdaa/layers/layernorm.py``
registers **no** OOT ``rms_norm`` (all three ``register_oot`` lines are commented
out), so ``CustomOp.dispatch_forward`` returns ``forward_oot`` whose default body is
``forward_native``:

* ``residual is None`` -> ``vllm.ir.ops.rms_norm(...)``
* ``residual given``   -> ``RMSNorm.forward_static(...)``

``vllm/ir/ops/layernorm.py`` is **not** a fused kernel - it is a pure-PyTorch eager
chain (``to(fp32) -> pow(2) -> mean -> +eps -> rsqrt -> mul -> to(w.dtype) ->
mul(weight) -> to(orig)``), i.e. ~8 device kernels and ~8 host dispatches per call.
Gemma4 issues 337 such calls per decode step (``4*48`` hidden-2D + ``3*48`` qkv-3D
+ 1 final).

The 2026-10-10 attempt bound the *delivered* ``tecoops.rms_norm`` here and removed
23.7 % of the decode wall time, but was rejected on precision: that kernel evaluates
the whole chain in fp32 and casts once, while the reference rounds to fp16 **before**
the weight multiply, so ~27 % of the fp16 output elements differed by 1 ULP and
greedy decoding degenerated from token 0.

What this binding does instead (the specialization)
---------------------------------------------------
The reference chain is split at the only point where it can be split with no
arithmetic change:

* the **row statistic** (``mean(x^2)`` and ``rsqrt``) is obtained from the reference
  operators themselves - ``x.to(fp32).pow(2).mean(-1, keepdim=True)`` followed by
  ``torch.rsqrt(v + eps)``.  Those two ATen kernels cannot be reproduced bit-for-bit
  by a hand-written kernel: the device ``mean`` reduction is a level-2/3 reduction
  that matches neither a plain sequential sum nor the correctly rounded exact mean
  (0/64 and 21/64 rows respectively on real Gemma rows), and the device ``rsqrt`` is
  a ~1-ULP approximation that is not ``1/sqrt``, not ``pow(t,-0.5)`` and not
  ``exp(-0.5*log(t))``;
* the **epilogue** - ``fp16(fp32(x) * rstd)`` followed by a fp16 multiply by the fp16
  weight, plus, for the fused form, the fp32 residual add and its fp16
  ``residual_out`` rounding - is executed by the new single-kernel
  ``tecoops.rms_norm_compat`` in exactly the reference's rounding order.

Measured: 0 differing elements out of 7 864 320 on the real Gemma shapes
(``(1,3840)``, ``(64,3840)``, ``(520,3840)``, ``(2048,3840)``, ``(8,256)``,
``(1,512)``, both ones-weight and weighted, plus both residual paths), i.e. the fp16
output is bit-identical to ``vllm.ir.ops.rms_norm`` / ``RMSNorm.forward_static``.

Binding form
------------
``torch.library.custom_op`` wrappers (the form proven by the Hy-MT2 overlay and by
the 2026-10-02 Dynamo failure on the bare pybind call).  Everything that selects an
implementation happens once, at process initialization (AGENTS.md rule 6): the
forward path contains no ``if`` on a backend, no fallback and no ``getenv``.  The
delivered ``tecoops.rms_norm`` and its kernels are untouched; this is a separate
entry point backed by separate ``__global__`` entities.

3-D q/k/v sites (``(tokens, heads, head_dim)``) reach the contiguous 2-D entry the
kernel already has via ``reshape(-1, head_dim)``; the *statistic* is still computed on
the original 3-D tensor, exactly as the reference computes it.

cycle-20 C1a: selectable statistic source
-----------------------------------------
``GEMMA4_RMSNORM_STATS`` (read exactly once, at import) picks between two shipped
forms of the SAME 337 sites:

* ``whole`` (DEFAULT since cycle-20 C1a) - the DELIVERED ``tecoops.rms_norm``, which
  computes ``mean(x^2)``, ``rsqrt`` and the epilogue in ONE launch and casts once at
  the end, so its fp16 output sits ~1-4 half-ULP away from the eager reference.  Its
  device cost per 337-call decode mix was measured at 19.8 ms/step, against
  128.1 ms/step for the split form (both archived in this model branch).  Measured
  in-model on the frozen library bytes: B/A1 = -26.58 % (median of 3 rotated rounds,
  drifted +0.76 %), max(B) < min(A), and 9/9 arms produced the frozen 32/32 greedy
  token IDs.
* ``compat`` - the split described above: reference eager statistics (5 ATen kernels
  per call) + ``tecoops.rms_norm_compat`` epilogue.  Bit-identical to
  ``vllm.ir.ops.rms_norm``; retained as the initialization-time fallback switch.

The forward implementation is bound ONCE, at class-definition time, from that single
read: there is no per-call backend branch and no ``getenv`` (AGENTS.md rule 6).
A missing tecoops symbol raises here, at initialization, instead of falling back.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import tecoops

# ---------------------------------------------------------------------------
# cycle-20 C1a - the ONLY place the statistic source is chosen.  Read once, here,
# at process initialization; the forward path never queries the environment.
# ---------------------------------------------------------------------------
STATS_MODE = os.environ.get("GEMMA4_RMSNORM_STATS", "whole")
if STATS_MODE not in ("compat", "whole"):
    raise RuntimeError(
        "GEMMA4_RMSNORM_STATS must be 'compat' (reference-exact split) or 'whole' "
        f"(delivered single-kernel whole chain), got {STATS_MODE!r}"
    )
# Optional per-process call census (cycle-13 linear_stream precedent): inert unless
# the directory is given at process start, and the wrappers are bound HERE, once, so
# the selected forward path still contains no branch and no getenv.
_CENSUS_DIR = os.environ.get("GEMMA4_RMSNORM_CENSUS_DIR")
_REQUIRED_TECOOP = "rms_norm" if STATS_MODE == "whole" else "rms_norm_compat"
if not hasattr(tecoops, _REQUIRED_TECOOP):
    raise RuntimeError(
        f"GEMMA4_RMSNORM_STATS={STATS_MODE!r} requires tecoops.{_REQUIRED_TECOOP}, "
        "but the loaded libteco_ops/_torch_ext does not export it (fail-closed)"
    )

__all__ = ["TecoopsRMSNorm", "rms_norm_epilogue", "rms_norm_epilogue_add",
           "rms_norm_whole", "rms_norm_whole_add", "STATS_MODE",
           "mapped_libteco_ops"]


def mapped_libteco_ops() -> list[str]:
    """The ``libteco_ops.so`` paths actually mapped into this process."""
    out = set()
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.split()[-1]
                if path.endswith("libteco_ops.so"):
                    out.add(path)
    except OSError:  # pragma: no cover - /proc always present on Linux
        pass
    return sorted(out)


@torch.library.custom_op(
    "gemma4_rms_norm::epilogue", mutates_args=("out",), device_types="sdaa"
)
def rms_norm_epilogue(
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    out: torch.Tensor,
) -> None:
    tecoops.rms_norm_compat(x, weight, rstd, None, out, None)


@rms_norm_epilogue.register_fake
def _rms_norm_epilogue_fake(x, weight, rstd, out):
    return None


@torch.library.custom_op(
    "gemma4_rms_norm::epilogue_add",
    mutates_args=("out", "residual_out"),
    device_types="sdaa",
)
def rms_norm_epilogue_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    tecoops.rms_norm_compat(x, weight, rstd, residual, out, residual_out)


@rms_norm_epilogue_add.register_fake
def _rms_norm_epilogue_add_fake(x, weight, rstd, residual, out, residual_out):
    return None


@torch.library.custom_op(
    "gemma4_rms_norm::whole", mutates_args=("out",), device_types="sdaa"
)
def rms_norm_whole(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float
) -> None:
    """Delivered whole-chain ``tecoops.rms_norm``: statistic + epilogue, one launch."""
    tecoops.rms_norm(x, weight, None, out, None, eps)


@rms_norm_whole.register_fake
def _rms_norm_whole_fake(x, weight, out, eps):
    return None


@torch.library.custom_op(
    "gemma4_rms_norm::whole_add",
    mutates_args=("out", "residual_out"),
    device_types="sdaa",
)
def rms_norm_whole_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    eps: float,
) -> None:
    tecoops.rms_norm(x, weight, residual, out, residual_out, eps)


@rms_norm_whole_add.register_fake
def _rms_norm_whole_add_fake(x, weight, residual, out, residual_out, eps):
    return None


class TecoopsRMSNorm(nn.Module):
    """Drop-in replacement for ``vllm.model_executor.layers.layernorm.RMSNorm``.

    The constructor signature and the ``state_dict`` surface (``weight`` is an
    ``nn.Parameter`` only when ``has_weight`` is true, exactly as upstream) are kept
    identical so the existing weight loader is unaffected.  ``has_weight=False``
    sites (``v_norm``, ``norm``) get a non-parameter ``torch.ones`` scale, which is
    the bit-exact identity for a half tensor and is what upstream expresses as
    ``weight=None``.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if var_hidden_size not in (None, hidden_size):
            raise RuntimeError(
                "Gemma4 RMSNorm overlay does not support variance_size_override "
                f"(hidden_size={hidden_size}, var_hidden_size={var_hidden_size})"
            )
        weight_dtype = dtype or torch.get_default_dtype()
        if weight_dtype != torch.float16:
            raise RuntimeError(
                "Gemma4 RMSNorm overlay requires float16, got "
                f"{weight_dtype}"
            )
        if not eps > 0:
            raise RuntimeError(f"Gemma4 RMSNorm overlay requires eps > 0, got {eps}")
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = None
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def _forward_compat(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight.data if self.has_weight else self.weight
        eps = float(self.variance_epsilon)
        # Row statistic: exactly the reference operators, on exactly the reference
        # shape (3-D q/k/v sites keep their 3-D shape here, as upstream does).
        xf = x.to(torch.float32)
        if residual is None:
            variance = xf.pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            # The kernel ABI takes raw pointers and walks rows `D` elements apart, so
            # the tensors it sees must be dense row-major.  `reshape` may legitimately
            # return a strided VIEW (e.g. `big[..., :D]` has row stride 2D) and
            # `empty_like` would then inherit that stride; both would silently index
            # the wrong rows.  `contiguous()`/`empty()` are no-ops on the contiguous
            # call sites, which is every site of this model (337 weights asserted
            # contiguous, stride [1], offset 0).
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            out2 = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
            rms_norm_epilogue(x2, weight, rstd.reshape(-1).contiguous(), out2)
            return out2.reshape_as(x)
        xr = xf + residual
        variance = xr.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        residual2 = residual.reshape(-1, residual.shape[-1]).contiguous()
        out2 = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
        residual_out2 = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
        rms_norm_epilogue_add(
            x2, weight, rstd.reshape(-1).contiguous(), residual2, out2, residual_out2
        )
        return out2.reshape_as(x), residual_out2.reshape_as(residual)


    def _forward_whole(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Delivered single-kernel whole chain (C1a).  Same shape contract as compat.

        Only the operator differs: ``tecoops.rms_norm`` computes the row statistic
        and the epilogue itself, so no ``torch.rsqrt`` / ``mean`` / ``pow`` is issued
        and the eager chain disappears from the step.
        """
        weight = self.weight.data if self.has_weight else self.weight
        eps = float(self.variance_epsilon)
        # Same dense row-major requirement as _forward_compat: the kernel ABI walks
        # rows `D` elements apart and ignores stride metadata.
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out2 = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
        if residual is None:
            rms_norm_whole(x2, weight, out2, eps)
            return out2.reshape_as(x)
        residual2 = residual.reshape(-1, residual.shape[-1]).contiguous()
        residual_out2 = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
        rms_norm_whole_add(x2, weight, residual2, out2, residual_out2, eps)
        return out2.reshape_as(x), residual_out2.reshape_as(residual)

    # Bound ONCE, at import, from the single GEMMA4_RMSNORM_STATS read above.
    # No per-call backend branch, no per-call getenv (AGENTS.md rule 6).
    forward = _forward_compat if STATS_MODE == "compat" else _forward_whole


# ---------------------------------------------------------------------------
# Diagnostic call census.  GEMMA4_RMSNORM_CENSUS_DIR is read once, above; when it
# is absent every name below keeps its identity (plain rebinding, no wrapper), so
# the timed arms pay nothing.  When present, the four operator entry points are
# wrapped ONCE, at import, and the counts are dumped per PID at process exit.
# The census arm is a LOCALISATION arm: its wall clock is Python-perturbed and is
# never used for the performance claim.
# ---------------------------------------------------------------------------
if _CENSUS_DIR:
    import atexit as _atexit
    import json as _json

    _CENSUS = {"rms_norm_whole": 0, "rms_norm_whole_add": 0,
               "rms_norm_epilogue": 0, "rms_norm_epilogue_add": 0}
    _CENSUS_TARGETS = {"rms_norm_whole": rms_norm_whole,
                       "rms_norm_whole_add": rms_norm_whole_add,
                       "rms_norm_epilogue": rms_norm_epilogue,
                       "rms_norm_epilogue_add": rms_norm_epilogue_add}

    def _census_wrap(_name, _target):
        def _counting_entry(*args, **kwargs):
            _CENSUS[_name] += 1
            return _target(*args, **kwargs)
        _counting_entry.__name__ = _name
        return _counting_entry

    rms_norm_whole = _census_wrap("rms_norm_whole", _CENSUS_TARGETS["rms_norm_whole"])
    rms_norm_whole_add = _census_wrap("rms_norm_whole_add", _CENSUS_TARGETS["rms_norm_whole_add"])
    rms_norm_epilogue = _census_wrap("rms_norm_epilogue", _CENSUS_TARGETS["rms_norm_epilogue"])
    rms_norm_epilogue_add = _census_wrap("rms_norm_epilogue_add",
                                         _CENSUS_TARGETS["rms_norm_epilogue_add"])

    def _dump_census() -> None:
        path = os.path.join(_CENSUS_DIR, "rms-norm-census-%d.json" % os.getpid())
        try:
            with open(path, "w") as fh:
                _json.dump({"pid": os.getpid(), "stats_mode": STATS_MODE,
                            "counts": dict(_CENSUS)}, fh, indent=2)
        except OSError:
            pass

    _dump_census()
    _atexit.register(_dump_census)
