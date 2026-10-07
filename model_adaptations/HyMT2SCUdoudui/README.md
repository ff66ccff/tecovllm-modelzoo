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

## Official CI launch parsing

The launcher exposes its actual literal-default `vllm serve` argv through a
local shell shim so the unchanged official `parse_run_sh` can read the model
path, host and port. The shim executes those arguments through the pinned
Python module entrypoint, replacing only the model path and first host/port
values from the existing environment settings. The official parser is static:
run it with the default `/gpfs/model` path, host `0.0.0.0` and port `8001`.
Environment overrides remain available for direct launcher use.

`validation/ci_contract_20261005.json` records the unchanged official parser,
four argv cases identical to the previous vendor invocation, and two
greedy32 runs at the default configured context 8192. Every output
ID matches this model's reference; runtime/test sources are unchanged.
This validates startup and short requests. Full official accuracy,
performance evaluation and long-context requests remain pending.


## Independent Hy PR37 SIMD epilogue validation (2026-10-07)

This Hy-only FP16 experiment uses official PR37 source revision
`e29b53c256f366e6eee538d656bfbb56e867cefc` and kernel SHA256
`41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be`.
Only the two RMSNorm second-pass output loops are vectorized; reduction order,
residual rounding, DMA, stream and ABI are unchanged. Reusing this official
source/artifact does not reuse another model's correctness or performance.
Default `run.sh`, FP16 TP1 context8192 and runtime/compiler remain unchanged.
The separate native BF16 capability experiment is DEFER and is not an accepted
public profile or a reason to change the default dtype.

Own boundary evidence passes36/36 cases,4graphs, SDAA Fake and real CPU
rejection at fixed atol/rtol0.002. Own profile micro shapes are hidden
[1,2048]/[30,2048] plain/add, q[16,128]/[480,128], k[4,128]/[120,128].
Framework boundary tokens1/8/32 are coverage, not all observed profile shapes.
All8 baseline/candidate output/residual hashes match on identical seeded
inputs. Three device and three wall values plus each median are retained in
`validation/rms_epilogue_simd_20261007.json`. The8groups have stable device
kernel improvements; wall includes a0.126163ms residual-prefill outlier and
small negative changes. Claim device-kernel micro effects only, no wall or
whole-model speedup. Two own fixed prompts each generate32greedy tokens
matching the existing Hy reference and baseline. Official task accuracy is
not established by these regression checks.

The original baseline/candidate model peaks were not captured. Independent
same-profile supplemental baseline/candidate launches now pass both own
prompts, matching promptIDs and all32generated IDs. Before/after worker
inspection counts129actual `sitecustomize.install.<locals>.TecoopsRMSNorm`
modules with SDAA FP16 weights and eps1e-5; the real model is
`HunYuanDenseV1ForCausalLM`. Actual unique mapped extension/core paths and
full hashes match the selected packages for both arms. Initialization binds
`hunyuan_v1.RMSNorm` through the existing public overlay and
`hy_mt2_public::rms_norm` / `hy_mt2_public::rms_norm_add` opaque APIs.

Supplemental process peak allocated is14455050752bytes (13.462315GiB),
reserved14816378880bytes (13.798828GiB), same before/after for both arms.
It includes initialization and the fixed32-token profile; no request-window
reset or original-A/B peak claim is made. This is metadata/peak completion
of the same hypothesis, not a new performance benchmark. Both owned services
stopped gracefully; device-release shows0MB/no process. The proof keeps
original A/B missing-peak fields false and identifies the supplement
separately. Selected modules/mappings and real generation establish the
binding; callability inspection or extra native diagnostic calls do not
measure every model forward. No wall/model speedup is claimed.

For a source-only isolated rebuild, set `B` to this public package's absolute
path, `OPS_SRC` to a fresh official teco-ops checkout in writable RAM,
`HAL_ROOT` to the existing teco-hal0.0.2 dependency, and `PACKAGE_OUT` to a
fresh executable directory. No private task DSO is required:

```bash
git -C "$OPS_SRC" checkout e29b53c256f366e6eee538d656bfbb56e867cefc
# Alternative exact kernel reproduction: checkout8f896f2a9103cc9c684eb4488c9f885a38f15eb6
# then git apply --check and git apply "$B/validation/epilogue-only.patch".
# Do not apply the patch again to the promoted revision.
source /opt/tecoai/setvars.sh
PYTHON=/home/py312/bin/python
test "$(readlink -f "$PYTHON")" = /usr/local/python/bin/python3.12
export WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TMPDIR=/tmp/rn26  # Short and executable; /dev/shm can be noexec.
mkdir -p "$TMPDIR" "$OPS_SRC/thirdparty" "$OPS_SRC/dist"
ln -s "$(readlink -f "$HAL_ROOT")" "$OPS_SRC/thirdparty/teco-hal"
unset PYTHONPATH
(cd "$OPS_SRC" && "$PYTHON" setup.py bdist_wheel --dist-dir "$OPS_SRC/dist")
"$PYTHON" - "$OPS_SRC/dist" "$PACKAGE_OUT" <<'PY'
import hashlib,json,pathlib,sys,zipfile
wdir,out=map(pathlib.Path,sys.argv[1:]);wheel,=wdir.glob('*.whl')
out.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(wheel) as z:
    for name in z.namelist():
        parts=pathlib.Path(name).parts
        if len(parts)==2 and parts[0]=='tecoops' and (name.endswith('.so') or name.endswith('__init__.py')):
            p=out/name;p.parent.mkdir(exist_ok=True);p.write_bytes(z.read(name))
(out/'build-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (out/'tecoops').iterdir()},indent=2))
PY
export PYTHONPATH="$PACKAGE_OUT"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
bash "$B/run.sh"
```

The public launcher adds its existing overlays and binds this package once at
initialization; no per-forward environment or backend selection is added.
Record the locally rebuilt extension/core hashes and actual worker mappings,
then run this Hy's focused and fixed-input model checks. Tested build hashes
identify the measured artifacts; the pinned source identifies what to rebuild
and does not promise identical binaries across build environments. No global
site-packages install or model weight download/copy is required. Baseline
`86576a…/6e10…` and candidate `d267…/c7d4…` are different built artifacts,
with full hashes in the proof. Historical baseline generated compiler-cache
hashes were not captured; shared source/recipe flags are not a measurement of
that missing historical compiler cache.
