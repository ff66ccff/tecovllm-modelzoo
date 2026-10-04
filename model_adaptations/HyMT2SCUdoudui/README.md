# Hy-MT2-1.8B (SCU 都队)

This adaptation uses the read-only checkpoint under `MODEL_ROOT` and binds
the existing vendor RMSNorm, SwiGLU, cache and attention APIs during process
initialization. It does not modify the installed vendor packages or download
weights. The tested runtime is FP16, TP1 (Hq16/Hkv4/D128), HND cache blocks32,
global causal attention with the default scale and automatic FP16 cache,
without KV sharing or prefix/chunked prefill.

## Launch

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/tecoai/setvars.sh
export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0 SDAA_VISIBLE_DEVICES=0
MODEL_ROOT=/gpfs/model bash model_adaptations/HyMT2SCUdoudui/run.sh
```

`run.sh` locks `/home/py312/bin/python`, disables stale compile-cache reuse,
and preserves the installed SDAA default vLLM compiler (no enforce-eager).
The required served name is `Hy-MT2-1.8B`; the default port is8001 and maximum
model length8192. `HY_MT2_MODEL`, `HY_MT2_HOST`, `HY_MT2_PORT`, and
`HY_MT2_MAX_MODEL_LEN` configure the local service. TP1/FP16 are the validated
geometry; unsupported attention geometry/window/scale fails at initialization.
Additional vLLM arguments can be supplied to the launcher.

## Runtime bindings

- RMSNorm invokes the original `tecoops.rms_norm` through two SDAA-only
  mutation/Fake boundaries, with and without residual output. This prevents
  Dynamo from tracing the raw pybind call and preserves the output contract.
- The existing merged gate/up projection is retained. A Tensor-only opaque
  SwiGLU boundary calls `torch.ops.sdaa.swiglu` with runtime integer sizes;
  the Fake output uses symbolic Tensor dimensions. Passing token counts to
  the vendor's `int` schema directly would specialize vLLM's dynamic axis.
- The installed `BlockAttentionImpl.forward` stub is bound to the established
  HND cache + vendor SDPA prefill/mixed + vendor BlockAttention decode
  integration. Fused-QKV views are made contiguous only at the cache ABI.
  Full prefill uses causal SDPA; mixed decode uses unmasked SDPA over its
  past cache. Chunked prefill is explicitly disabled and rejected.

The older Transformers BF16 RMSNorm/SwiGLU overlays and their measured
3.16% SwiGLU device-time result remain separate evidence. vLLM already merges
gate/up, so that HF result is not a performance claim for this export.
These changes repair the public runtime entry; they introduce no new UAL.

## Validation

Portable real-device tests, after sourcing the SDK:

```bash
B=model_adaptations/HyMT2SCUdoudui
mkdir -p /tmp/hy-validation
PYTHONPATH="$B/runtime" /home/py312/bin/python "$B/tests/test_vllm_rms_norm.py" /tmp/hy-validation/rms
PYTHONPATH="$B/runtime" /home/py312/bin/python "$B/tests/test_vllm_swiglu.py" /tmp/hy-validation/swiglu
PYTHONPATH="$B/runtime" /home/py312/bin/python "$B/tests/test_vllm_attention.py" /tmp/hy-validation/attention
PYTHONPATH="$B/runtime/overlay:$B/runtime" /home/py312/bin/python "$B/tests/test_vllm_mlp_caller.py" /tmp/hy-validation/mlp.json
```

See `validation/runtime_20261004.json` for source/library identity, actual
export startup, fixed greedy32 and steady reference receipts. Primitive
RMS/SwiGLU boundaries match their original vendor calls bitwise. Independent
CPU arithmetic uses fixed rtol/atol0.002 and retains known FP16 subnormal
flush behavior. Attention uses an independent explicit bottom-right causal
CPU NumPy FP64 reduction oracle (nrmse<=0.00390625, max_abs<=0.06).
The Torch CPU BLAS packing path crashed on the513x513 oracle; the raw failure
and debugger receipt are retained without changing global BLAS libraries.

The model reference checks cover the fixed prompt with32 generated IDs and
concurrency/steady requests; full-length8k generation has not been evaluated.
These tests do not establish official task accuracy or a vLLM speedup. The
installed tecoops wheel has not been proved equivalent to current official
PR35/37 heads. Owner CI and official task evaluation remain separate gates.
