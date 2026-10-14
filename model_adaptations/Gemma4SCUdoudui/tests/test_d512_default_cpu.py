"""Device-free proof for the cycle-27 B-arm overlay (delivered D512 decode = default tier).

Runs on CPU with a fake ``tecoops`` module and a synthetic ``/proc/self/maps``, so no SDAA
device and no built library are needed.  Usage:
    python test_d512_default_cpu.py            # positive suite
    python test_d512_default_cpu.py --neg X    # one fail-closed case (fresh interpreter)
"""
import builtins
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types

HERE = Path(__file__).resolve()
MODULE = HERE.parents[1] / "runtime" / "custom_ops" / "d512_official.py"
FAKE_LIB = Path(tempfile.gettempdir()) / "c27_fake_libteco_ops" / "libteco_ops.so"


def _ensure_fake_lib():
    FAKE_LIB.parent.mkdir(parents=True, exist_ok=True)
    FAKE_LIB.write_bytes(b"\x7fELF-c27-fake-libteco_ops")


_real_open = builtins.open


def _install_fake_maps(path):
    def fake_open(file, *a, **kw):
        if file == "/proc/self/maps":
            return io.StringIO("7f0000000000-7f0000001000 r-xp 00000000 00:00 0 %s\n" % path)
        return _real_open(file, *a, **kw)
    builtins.open = fake_open


def _install_fake_tecoops(calls, export=True):
    mod = types.ModuleType("tecoops")
    mod.__file__ = "/fake/tecoops/__init__.py"

    def flash_attn_varlen_func(*args, **kwargs):
        calls.append((args, kwargs))
        out = args[13]
        out.fill_(0.5)
        return None

    if export:
        mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["tecoops"] = mod
    return mod


def _load(calls, export=True, expect_lib=None):
    _ensure_fake_lib()
    _install_fake_maps(FAKE_LIB)
    _install_fake_tecoops(calls, export=export)
    if expect_lib is None:
        os.environ.pop("GEMMA4_D512_EXPECT_LIB", None)
    else:
        os.environ["GEMMA4_D512_EXPECT_LIB"] = expect_lib
    spec = importlib.util.spec_from_file_location("c27_shadow_d512", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _impl(head_size, window=None, **over):
    ns = dict(head_size=head_size, num_heads=8, num_kv_heads=1, logits_soft_cap=0,
              sinks=None, kv_cache_dtype="auto", scale=1.0, _gemma4_window_size=window)
    ns.update(over)
    return types.SimpleNamespace(**ns)


def positive():
    import torch

    calls = []
    mod = _load(calls, expect_lib=str(FAKE_LIB))
    assert mod.OFFICIAL_DECODE is not None, "default tier must be ON"
    assert mod._MAPPED_LIBS == [str(FAKE_LIB.resolve())], mod._MAPPED_LIBS

    # 1) hot path is getenv-free
    for fn in (mod.decode_forward, mod.register_decode.__wrapped__
               if hasattr(mod.register_decode, "__wrapped__") else mod.decode_forward):
        src = __import__("inspect").getsource(fn)
        for bad in ("getenv", "environ"):
            assert bad not in src, "hot path contains %s in %s" % (bad, fn.__name__)

    # 2) binding is init-time and shape-gated
    g = _impl(512)
    mod.bind_global_decode(g)
    assert g._gemma4_decode_fn is mod.decode_forward, "D512 global layer not bound"
    d256 = _impl(256)
    mod.bind_global_decode(d256)
    assert not hasattr(d256, "_gemma4_decode_fn"), "D256 must stay untouched"
    try:
        mod.bind_global_decode(_impl(512, window=1024))
        raise AssertionError("windowed D512 must be refused")
    except RuntimeError:
        pass
    try:
        mod.bind_global_decode(_impl(512, num_heads=16))
        raise AssertionError("H16 D512 must be refused")
    except RuntimeError:
        pass

    # 3) 14-arg ABI, causal=True, scale1, out mutation
    n, kv, bs, hd = 2, 33, 32, 512
    q = torch.zeros((n, 8, hd), dtype=torch.float16)
    kc = torch.zeros((4, 1, bs, hd), dtype=torch.float16)
    vc = torch.zeros((4, 1, bs, hd), dtype=torch.float16)
    qsl = torch.tensor([0, 1, 2], dtype=torch.int32)
    sl = torch.tensor([kv, kv], dtype=torch.int32)
    bt = torch.zeros((n, 2), dtype=torch.int32)
    out = torch.full((n, 8, hd), -7.0, dtype=torch.float16)
    mod.OFFICIAL_DECODE(q, kc, vc, qsl, sl, bt, 2048, 1.0, out)
    assert len(calls) == 1, calls
    args, _ = calls[0]
    assert len(args) == 14, "expected the 14-arg delivered ABI, got %d" % len(args)
    assert args[3] == 1, "max_seqlen_q must be 1 (single-token decode)"
    assert args[4] is qsl and args[7] is sl and args[11] is bt, "metadata must pass through"
    assert args[8] == 1.0 and args[9] is True, "scale1 + causal=True"
    assert args[12] is False, "return_softmax_lse must be False"
    assert args[13] is out, "out must be written in place"
    assert bool((out == 0.5).all()), "out mutation did not happen"
    print("POSITIVE_OK calls=%d args=%d" % (len(calls), len(args)))
    return 0


def negative(case):
    calls = []
    if case == "missing_symbol":
        _load(calls, export=False)
    elif case == "legacy_env":
        os.environ["GEMMA_D512_OFFICIAL_EXTENSION"] = "/tmp/whatever.so"
        os.environ["GEMMA_D512_OFFICIAL_SHA256"] = "0" * 64
        try:
            _load(calls, expect_lib=str(FAKE_LIB))
        finally:
            os.environ.pop("GEMMA_D512_OFFICIAL_EXTENSION", None)
            os.environ.pop("GEMMA_D512_OFFICIAL_SHA256", None)
    elif case == "wrong_lib":
        _load(calls, expect_lib="/tmp/somewhere/else/libteco_ops.so")
    else:
        raise SystemExit("unknown negative case %r" % case)
    raise AssertionError("negative case %r did NOT fail closed" % case)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--neg":
        try:
            negative(sys.argv[2])
        except AssertionError as exc:
            print("NEGATIVE_FAILED %s: %s" % (sys.argv[2], exc))
            sys.exit(9)
        except RuntimeError as exc:
            assert "D512 delivered decode" in str(exc), str(exc)
            print("NEGATIVE_OK %s: %s" % (sys.argv[2], exc))
            sys.exit(0)
        sys.exit(8)
    sys.exit(positive())
