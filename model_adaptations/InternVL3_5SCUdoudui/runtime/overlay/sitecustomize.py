"""Process initialization overlay for TecoVLLM on SDAA accelerator.

Binds the in-repo BlockAttention compatibility layer before engine
initialization so that ``vllm_sdaa``'s unimplemented attention stub is
replaced without ever writing to vendor site-packages and without leaving a
per-forward branch in the hot path.

It also binds the Qwen3 language tower's ``RMSNorm`` to the official
``tecoops.rms_norm`` operator.  Audit fact that motivated it: before this
overlay the official operator had **0 calls** in a real forward (see
``experiments/internvl3_5-8b.md`` §12.2).  The binding replaces the module-level
``RMSNorm`` name in ``vllm.model_executor.models.qwen3`` **and**
``...models.qwen2`` (the final norm is constructed by ``Qwen2Model``, which
``Qwen3Model`` inherits).  It is installed unconditionally and fails closed: if
the vendor layout moved, the process exits non-zero instead of silently
running without the official operator.

Fail-closed mechanism (important): this module runs inside CPython's
``site.execsitecustomize()``, whose handler is ``except Exception``.  A plain
``raise SomeError`` is therefore **swallowed** -- the interpreter prints
"Error in sitecustomize" and continues to ``main``, i.e. a silent degradation.
Only ``BaseException`` subclasses that are not ``Exception`` escape it.  Every
failure path here therefore ends in ``raise SystemExit(1)`` (equivalently
``sys.exit(1)``/``os._exit(1)``), and the top-level installer additionally
converts any unexpected plain exception into ``SystemExit(1)``.

Scope note (InternVL3_5-8B): the MiniCPM5-1B branch's *Llama-specific* binding
is not ported verbatim; this one targets Qwen3 and is validated by an
init-time identity check plus a call-count receipt.

Loaded automatically by CPython via ``sitecustomize`` when this directory is
placed on ``PYTHONPATH`` (see scripts/serve_internvl.sh).
"""

import sys


def install_rms_norm_overlay() -> None:
    """Bind the Qwen3 language tower RMSNorm to official tecoops.rms_norm."""
    try:
        from custom_ops.rms_norm.op import install_qwen3_rms_norm_overlay
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import the RMSNorm overlay installer: {exc}\n"
        )
        raise SystemExit(1) from exc

    # The installer itself is fail-closed (SystemExit on every unmet precondition).
    install_qwen3_rms_norm_overlay()


def install_block_attention_overlay() -> None:
    """Install the SDAA BlockAttention forward overlay.

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
        raise SystemExit(1) from exc

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
    except (Exception, SystemExit) as exc:
        # SystemExit is included on purpose: the module-level guards in
        # custom_ops/block_attention/op.py (e.g. an invalid INTERNVL_PREFILL_ATTN)
        # raise SystemExit, and we want the readable FATAL line for them too.
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import 'sdaa_block_attention_forward': {exc}\n"
        )
        raise SystemExit(1) from exc

    try:
        impl_cls.forward = sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to bind forward method onto BlockAttentionImpl: {exc}\n"
        )
        raise SystemExit(1) from exc

    if getattr(impl_cls, "forward", None) is not sdaa_block_attention_forward:
        sys.stderr.write(
            "FATAL: [custom_ops] Post-install identity check failed: BlockAttentionImpl.forward is not sdaa_block_attention_forward\n"
        )
        raise SystemExit(1)

    sys.stderr.write("[custom_ops] BlockAttention overlay bound successfully.\n")


def install_rotary_overlay() -> None:
    """Bind the validated existing-vendor language-tower RoPE at initialization."""
    try:
        from custom_ops.rotary.op import install
        install()
    except (Exception, SystemExit) as exc:
        sys.stderr.write(f"FATAL: [custom_ops] Failed to install InternVL RoPE binding: {exc}\n")
        raise SystemExit(1) from exc


def install_linear_stream_overlay() -> None:
    """Bind the language-tower M=1 decode linear to the official tecoops operator.

    Fail-closed like the other four overlays: any failure ends in SystemExit(1) so
    that ``site.execsitecustomize()`` cannot swallow it into a silent degradation.
    """
    try:
        from custom_ops.linear_stream.op import install_linear_stream

        receipt = install_linear_stream()
        sys.stderr.write(f"[custom_ops] linear_stream overlay bound: {receipt}\n")
    except (Exception, SystemExit) as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to install the linear_stream binding: {exc}\n"
        )
        raise SystemExit(1) from exc


def install_all_overlays() -> None:
    """Install every overlay; success path is unchanged, failure is process-level."""
    install_rms_norm_overlay()
    install_block_attention_overlay()
    install_rotary_overlay()
    install_linear_stream_overlay()


import os
import sys
if os.environ.get("SCUD_REQUIRE_OPS_BUNDLE") == "1":
    try:
        from custom_ops.ops_provenance import require_ops_bundle
        require_ops_bundle(['rms_norm', 'reshape_and_cache', 'flash_attn_varlen_func', 'rotary_embedding', 'linear_stream'])
    except Exception as exc:
        sys.stderr.write(f"FATAL: operator bundle provenance check failed: {exc}\n")
        raise SystemExit(1) from exc

try:
    install_all_overlays()
except SystemExit:
    raise
except BaseException as exc:  # noqa: BLE001 -- deliberate: site.py swallows plain Exception
    sys.stderr.write(
        "FATAL: [custom_ops] Overlay installation failed with a non-SystemExit exception "
        f"({type(exc).__name__}: {exc}); converting it to SystemExit(1) so that "
        "site.execsitecustomize() cannot swallow it (silent-degradation guard).\n"
    )
    raise SystemExit(1) from exc

from custom_ops.diagnostic_shapes import install_decode_census
install_decode_census()
