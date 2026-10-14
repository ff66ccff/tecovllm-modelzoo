"""InternVL3_5-8B M=1 decode linear -> ``tecoops.linear_stream`` (weight streaming).

Why this exists (mechanism, measured on the delivered ops library)
------------------------------------------------------------------
Every decode step of a 8B-class model with TP=2 is one activation row against a
stack of weight matrices: pure HBM weight streaming, not arithmetic.  Measured on
one card, fp16, M=1, on the Gemma-4-12B shapes, the vendor GEMM that
``LinearBase`` reaches through
``vllm.model_executor.layers.utils.default_unquantized_gemm`` runs at
**30.1-77.5 GB/s** effective (weight bytes / time), while a pure streaming read of
the same bytes reaches **92-197 GB/s** (numbers quoted by the delivered kernel's
own header, ``teco/ual/args/weight_stream_gemv_args.h:44-49``).  The gap is the M=1
GEMM's shape, not a bandwidth ceiling.

Binding form (construct/init time, NOT per call)
-----------------------------------------------
The decision between the streaming kernel and the vendor GEMM lives in the C++
host function of ONE opaque operator (``api/torch_ext.cpp: linear_stream_torch``),
so nothing about it is visible to Dynamo and the operator is a single graph node
that always executes.  This module only replaces the module-global
``default_unquantized_gemm`` (``vllm/model_executor/layers/utils.py:92-98``), which
is the single call site used by ``LinearBase.forward``
(``layers/linear.py:228``) and ``VocabParallelEmbedding.forward``
(``layers/vocab_parallel_embedding.py:75``).  There is no per-forward branch, no
per-forward ``getenv`` and no shape test in Python (AGENTS.md rule 6).

Scope
-----
The operator itself keeps the vendor GEMM for prefill (M > 1) and for any shape
that does not satisfy ``N % 256 == 0 and K % 16 == 0`` -- that fallback is inside
the C++ host function, and it is counted by the launch census of the census host.
"""

from __future__ import annotations

import json
import os

import torch
import torch_sdaa  # noqa: F401  (registers the "sdaa" dispatch key used below)

import tecoops

__all__ = ["linear_stream", "install_linear_stream", "mapped_libteco_ops"]

# The kernel splits N into contiguous row blocks flushed by one DMA (16 B aligned
# block starts => N % 256 == 0) and walks K as a 16-wide fp32 lane chain
# (K % 16 == 0).  Mirrored here only for documentation/tests, never on the hot path.
_WSG_N_ALIGN = 256
_WSG_K_ALIGN = 16


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
    "internvl3_5_linear_stream::linear_stream", mutates_args=(), device_types="sdaa"
)
def linear_stream(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Opaque replacement for ``F.linear`` (M==1 decision lives in C++)."""
    return tecoops.linear_stream(x, weight, bias)


@linear_stream.register_fake
def _linear_stream_fake(x, weight, bias=None):
    return torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)


def install_linear_stream() -> dict:
    """Bind the streaming linear once, at process initialization.

    Fail-closed: every unmet precondition raises ``SystemExit(1)`` so that
    ``site.execsitecustomize`` cannot swallow it into a silently degraded process.
    """
    import vllm.model_executor.layers.utils as U

    if getattr(U, "_internvl_linear_stream_patched", False):
        return {"already_patched": True}

    if not hasattr(tecoops, "linear_stream"):
        raise SystemExit(
            "FATAL: [custom_ops] tecoops.linear_stream is missing; the staged tecoops "
            "package does not carry the delivered M=1 weight-streaming operator"
        )
    orig = getattr(U, "default_unquantized_gemm", None)
    if orig is None or not callable(orig):
        raise SystemExit(
            "FATAL: [custom_ops] vllm.model_executor.layers.utils.default_unquantized_gemm "
            "not found; the vendor layout moved"
        )

    def stream_unquantized_gemm(layer, x, weight, bias=None):
        # No branch, no getenv, no shape test: one opaque operator call.
        return linear_stream(x, weight, bias)

    U.default_unquantized_gemm = stream_unquantized_gemm
    if U.default_unquantized_gemm is not stream_unquantized_gemm:
        raise SystemExit("FATAL: [custom_ops] linear_stream binding identity check failed")
    U._internvl_linear_stream_patched = True

    # Init-time-gated launch census (C++ side, no hot-path Python work).  Inert unless
    # INTERNVL_LINEAR_STATS_DIR is set at process start.  It proves the M==1 branch
    # really launched the streaming kernel in this process instead of silently
    # falling back to the vendor GEMM.
    stats_dir = os.environ.get("INTERNVL_LINEAR_STATS_DIR")
    if stats_dir:
        import atexit

        os.makedirs(stats_dir, exist_ok=True)
        census_path = os.path.join(stats_dir, f"linear-launches-{os.getpid()}.json")
        keys = (
            "kernel_launches",
            "vendor_fallbacks",
            "fb_not_m1",
            "fb_dtype",
            "fb_w_noncontig",
            "fb_k_mismatch",
            "fb_align",
            "fb_other",
        )

        def _dump_census() -> None:
            try:
                values = list(tecoops._torch_ext.linear_stream_stats())
            except Exception:  # noqa: BLE001
                values = [None] * len(keys)
            try:
                with open(census_path, "w") as fh:
                    json.dump(
                        {"pid": os.getpid(), **dict(zip(keys, values))}, fh, indent=2
                    )
            except OSError:
                pass

        _dump_census()
        atexit.register(_dump_census)

    return {
        "bound": "vllm.model_executor.layers.utils.default_unquantized_gemm",
        "orig": getattr(orig, "__name__", repr(orig)),
        "tecoops_file": getattr(tecoops, "__file__", None),
        "mapped_libteco_ops": mapped_libteco_ops(),
        "pid": os.getpid(),
    }
