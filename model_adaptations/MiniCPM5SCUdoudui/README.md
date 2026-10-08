# MiniCPM5-1B (SCU 都队)

This adaptation binds the existing official `tecoops` RMSNorm, KV-cache write
and paged prefill-attention APIs to MiniCPM5-1B. It uses process-local overlays,
the vendor interpreter and read-only pre-provisioned weights. It contains no new
operator kernel. The optional vendor SwiGLU profile below has bounded
shape-level synchronized wall-time measurements; no model speedup is claimed.

## Launch

From the model-zoo root, with the installed TecoVLLM/Torch-SDAA SDK and a wheel
providing `rms_norm`, `reshape_and_cache` and `flash_attn_varlen_func`:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/tecoai/setvars.sh
export MODEL_ROOT=/gpfs/model
export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0
export SDAA_VISIBLE_DEVICES=1
export MINICPM_MAX_MODEL_LEN=2048
export MINICPM_HOST=127.0.0.1 MINICPM_PORT=8014
b=model_adaptations/MiniCPM5SCUdoudui
bash "$b/run.sh"
```

`run.sh` locks `/home/py312/bin/python`, sets offline weight access and disables
persistent compilation caching. It preserves default vLLM compilation and
does not force `TORCH_COMPILE_DISABLE` or `--enforce-eager`. TP1, FP16 and disabled
prefix caching are fixed. `MINICPM5_MODEL` can override
`$MODEL_ROOT/OpenBMB/MiniCPM5-1B`. `MINICPM_MAX_MODEL_LEN` defaults to 32768;
validation used 2048. Extra CLI arguments are passed through for diagnostics.

`MINICPM_OP_PROFILE=official` remains the default. The overlay registers opaque
`torch.library` operations with explicit output/residual/cache mutation and
FakeTensor implementations, then binds RMSNorm and BlockAttention once during
initialization. Prefill invokes the official paged flash ABI; single-token decode
retains the vendor BlockAttention implementation. Missing interfaces or failed
bindings terminate startup before the main entrypoint. Vendor packages stay
unchanged.

## Optional full-sequence SDPA prefill

For a manual service, explicitly select the accepted model-side alternative:

```bash
MINICPM_OP_PROFILE=fast_attention bash "$b/run.sh"
```

The startup shim disables chunked prefill and prefix caching and fixes
`--max-num-seqs 1` in this profile. It rejects arguments that re-enable those
features or increase the sequence count. Concurrent HTTP requests queue for
one sequence per model step, avoiding mixed prefill/decode batches. Each request must have
query length equal to sequence length; a KV prefix or mixed prefill/decode batch
fails closed instead of using an incorrectly aligned causal mask. The profile
enables SDAA fused flash-SDP once at initialization and registers SDPA as an
opaque mutating operation with a FakeTensor implementation. Registration or
backend setup failure terminates startup; there is no eager fallback.

Fast prefill uses paged-cache gather plus native-GQA SDPA and does **not** call
official `flash_attn_varlen_func`. Official RMSNorm and cache writes and vendor
single-token decode remain in use. The default profile retains all three
official APIs and its literal CI command, compiler mode and launch parameters.
The current public launcher has independent device, greedy32 and 300-second
single-sequence service evidence below. No new kernel or performance result is claimed
by this export. Historical private-profile timing is separate evidence.

```bash
/home/py312/bin/python -S "$b/tests/test_sdpa_cpu.py"
/home/py312/bin/python -S "$b/tests/test_launcher_cpu.py" tools/ci_pipline/run_ci.py
SDAA_VISIBLE_DEVICES=1 /home/py312/bin/python "$b/tests/test_sdpa_sdaa.py" --output /path/to/sdpa.json
```

## Reproduce validation

The following CPU tests isolate imports with the vendor interpreter; the SDAA
test additionally requires the SDK environment loaded above:

```bash
/home/py312/bin/python -S "$b/tests/test_compile_safe_cpu.py"
/home/py312/bin/python -S "$b/tests/test_startup_failure.py"
SDAA_VISIBLE_DEVICES=1 /home/py312/bin/python "$b/tests/test_compile_safe_sdaa.py"
```

Against the separately started service:

```bash
/home/py312/bin/python "$b/tests/verify_model.py" \
  --base http://127.0.0.1:8014 --out /path/to/model.json \
  --expected "$b/validation/baseline_token_ids.json" --steady 300 --concurrency
```

The verifier checks every response's complete 32 greedy token IDs against the
same-stack FP16 reference. To collect actual vendor ABI calls, optionally set
`TECOOPS_CALL_RECEIPT=/path/to/abi_calls.txt` before startup and gracefully stop
the service afterwards. Accounting is selected at initialization; FakeTensor
propagation does not enter the vendor body or count a call. Counters flush on
exit. Without this diagnostic variable, the original ABI callable is bound
directly inside each opaque implementation.

The official evaluation entry remains:

```bash
/home/py312/bin/python tools/ci_pipline/run_ci.py "$b/run.sh"
```

## Validated export (2026-10-04)

- Four official bindings pass CPU FakeTensor/fullgraph mutation checks and real
  SDAA raw/opaque/fullgraph bitwise comparisons. Independent NumPy maximum
  absolute errors are 0.00190449 for RMSNorm, 0.00344181 for fused residual RMSNorm
  and 0.00195313 for causal GQA flash (fixed threshold 0.05). HND block32 cache
  writes, skipped slots and untouched regions match bitwise.
- Nine startup/profile cases pass. Two original public-entry reference runs
  produce identical 32-token IDs. The default-compiled candidate matches all
  482 responses, including concurrency 4/8 and 302.241956 seconds, 108 rounds,
  468 steady requests, with zero errors.
- The worker retains FP16 SDAA weights and reports peak allocated memory
  13.761231 GiB. During the verified requests, actual ABI count deltas are
  RMSNorm 179242, cache writes 87792 and paged flash 4464. These correspond to
  3658 model forward batches (49 norms and 24 cache calls per batch) and 186
  prefill batches (24 flash calls each). Owned services are released.

`validation/runtime_20261004.json` records source hashes, configuration, versions
and limitations. The tested stack uses PyTorch 2.12.0a0+git0d62256, Torch-SDAA
20260623.8.51.dev0+gitd942f23, vLLM 0.20.2.dev0+g132765e35.d20260728 and
Transformers 4.57.6. The installed SDK uses driver/runtime 3.2.0.

This is a compilation compatibility fix and temporary text regression evidence.
Official task accuracy and owner CI remain pending. The default chunked-prefill
setting remains enabled, but these short prompts do not validate long chunks or
full 32768-token generation. The wheel used in this 2026-10-04 receipt was not matched to upstream PR
heads. Model weights and compiled binaries are not included.

## Optional profile validation (2026-10-07)

`validation/sdpa_public_20261007.json` freezes this public package and its
independent MiniCPM evidence. With vendor Python, TP1, FP16 and configured
context 2048 on logical device 1, the original public baseline and the
new default profile each match both complete 32-token reference outputs.
The opt-in profile matches all 119 responses with concurrency 1/4/8 clients queued
under the fixed single-sequence scheduler, including
312.926044 seconds and 117 steady requests, with zero errors.

Six FP16 Hq16/Hkv2/D128, HND/block32 cases cover lengths 7, 256, 512,
1024, 2048 and mixed 7/256. The complete float64 CPU oracle gives maximum
absolute error 0.001618666 (threshold 0.05). Raw/opaque output bits,
12-positional out mutation, FakeTensor/fullgraph, 2D/3D block tables and
unchanged KV contents pass. Profiler records the actual SDAA fused
`tecocustomFlashAttentionForward` kernel with math SDP disabled in the
focused test. The public service binds SDPA at startup; request counters
confirm 24 SDPA prefill calls per request, zero official flash calls, and
the expected official RMSNorm/cache counts. Peak allocated model memory
is 13.749059 GiB.

Default vLLM compilation remains active. The separately audited PR37
wheel is reused without installation; its extension/core hashes are in
the receipt. Full official accuracy, public fast long-context requests
and parallel sequence scheduling remain unvalidated. All test
services were stopped; concurrent VOC training on logical device 0 was
kept running. No current speed comparison is claimed.

## Optional FP16 vendor SwiGLU activation (2026-10-07)

Select this activation independently while retaining official attention:

```bash
MINICPM_OP_PROFILE=official MINICPM_SILU_MUL_PROFILE=vendor_swiglu bash "$b/run.sh"
```

Unset `MINICPM_SILU_MUL_PROFILE` or use `official` to retain the existing default.
The optional profile replaces only `SiluAndMul` with an opaque `torch.library`
bridge to vendor `sdaa::swiglu`. The SDAA forward is `view -> startup-bound
vendor(x,T,1,4608) -> view`. It has no runtime environment lookup or backend
selection. FakeTensor supplies shape metadata. The supported input contract is contiguous FP16 packed input. The startup
shim rejects any explicit non-FP16 `--dtype` even
if an inherited FP16 marker exists, and permits an argument-free worker to
inherit its parent's validated FP16 marker. It requires `MINICPM_OP_PROFILE=official`;
combining this activation with `fast_attention` has not been validated.
The public runtime source is prioritized and a shadowed activation module fails
closed. No installed framework package, model weights, attention/RMSNorm path
or vendor kernel is changed.

A diagnostic eager generation captured real contiguous FP16 inputs rather than
inferring them from configuration: `[1,9216]` occurred 744 times and `[7,9216]`
24 times within the fixed greedy32 generation, 768 activation calls total.
Default-compiler public service validation is separate from that eager capture.

Same-input seed 20261007, warmup 3, 100 calls/run and three alternating A/B trials
measured synchronized host wall time per call:

| Input | Separate SiLU + Mul raw ms / median | Vendor SwiGLU raw ms / median |
| --- | --- | --- |
| `[1,9216]` | `0.174054760/0.173904260/0.173721760` / `0.173904260` | `0.093571370/0.093318760/0.093747360` / `0.093571370` |
| `[7,9216]` | `0.265690130/0.266620630/0.266493620` / `0.266493620` | `0.094014760/0.093581560/0.094851460` / `0.094014760` |

These are shape-level wall-time results (1.85852x/2.83459x median ratios), not
kernel device-time or model-latency results. Private micro and public bridge
forward math/ABI are AST-equivalent up to a local temporary variable name.
Public call accounting is selected once at initialization; when the receipt
variable is unset, the original vendor callable is bound directly. Micro
measurement did not include the diagnostic counter wrapper.

The final checked-in `tests/test_silu_mul_sdaa.py` was actually executed:
two vendor calls, both independent CPU references pass unchanged `rtol=atol=.001`,
and all input bits remain unchanged. The original minimum-normal case is
retained: gate bits `0x0400` times up bits `0x6800` (2048) gives candidate 0.0625
and both CPU FP32-final-FP16 and CPU FP16 separate references 0.0625. The old
separate SDAA path returns 0 from intermediate SiLU flush-to-zero; its comparison
is explicitly false. This candidate is not claimed bitwise equivalent to the
old SDAA path. Initial underflow-case failure and the exact original-case rerun
are preserved without deleting inputs or changing tolerances.

With the same observed public launcher/engine arguments, TP1, configured FP16,
context 2048, default vLLM compiler, official attention and the same fixed prompt,
each of baseline and vendor activation produces two complete greedy32 responses
identical to this model's own `validation/baseline_token_ids.json`. Worker RPC
identifies the actual public activation source and bound vendor callable.
`COUNT silu_mul.vendor_swiglu=1608` is the complete diagnostic process-lifecycle
count, including initialization; it is not a generation-only delta. Peak
allocated/reserved memory is 14776009728/14969470976 bytes (13.761231/13.941406 GiB).
The test services were stopped and their allocated logical device was released.
FP16 is the configured launcher/engine dtype; every model parameter dtype was
not independently enumerated.

Both original exited service processes' native `tecoops`/extension/vendor-DSO
maps and hashes were not retained. The separately timestamped CPU-only current
package/DSO/schema snapshot in the receipt is not evidence of those old process
mappings. Logged argparse configuration/compiler settings match; a separate
baseline launch/environment file was not archived. No current combined-wheel,
owner CI, official task accuracy, BF16, fusion long-context/concurrent/steady
service, or three-run model speedup claim is made. Historical attention or RMS
results are not activation evidence. Full raw values, source hashes, original
boundary vectors and these limits are in
`validation/silu_mul_public_20261007.json`.

```bash
# CPU/Fake/fullgraph/startup tests, in a fresh process.
PYTHONPATH="$b/runtime:${PYTHONPATH:-}" /home/py312/bin/python "$b/tests/test_silu_mul_public_cpu.py"
# Requires a separately assigned device; this command preserves original inputs.
SDAA_VISIBLE_DEVICES=1 PYTHONPATH="$b/runtime:${PYTHONPATH:-}"   /home/py312/bin/python "$b/tests/test_silu_mul_sdaa.py"   --device sdaa:0 --output /path/to/silu-mul-focused.json
```

## Official CI launch parsing

The launcher exposes its actual literal-default `vllm serve` argv through a
local shell shim so the unchanged official `parse_run_sh` can read the model
path, host and port. The shim executes those arguments through the pinned
Python module entrypoint, replacing only the model path and first host/port
values from the existing environment settings. The official parser is static:
run it with the default `/gpfs/model` path, host `0.0.0.0` and port `8000`.
Environment overrides remain available for direct launcher use; start the
official CI runner with its default launch settings because it cannot read
runtime shell overrides.

`validation/ci_contract_20261005.json` records the unchanged official parser,
four argv cases identical to the previous vendor invocation, and two
greedy32 runs at the default configured context 32768. Every output
ID matches this model's reference; runtime/test sources are unchanged.
This validates startup and short requests. Full official accuracy,
performance evaluation and long-context requests remain pending.

## Optional pinned E29 RMS epilogue (2026-10-08)

This independently validated option changes only the official RMS plain/add
second output loops to SIMD. Keep official attention and accepted vendor
SwiGLU. It does not select PR37's current DMA kernel. The measured B reused
preserved E29 binaries; only scalar A was freshly rebuilt in this attempt.
A/B generated flags and compiler hashes match; independent MiniCPM results
and actual runtime mappings are in validation/rms_epilogue_simd_20261008.json.

Build isolated source and a small executable package, without pip installation,
a wheel, global dependency changes or copied weights. Existing SDK/HAL is
required; fail if the pinned source object or local HAL is unavailable.

```bash
set -euo pipefail
b=model_adaptations/MiniCPM5SCUdoudui
rms_source=/dev/shm/minicpm-e29-source
rms_exec=/tmp/minicpm-e29-package
rms_tmp=/tmp/minicpm-e29-build-tmp
test ! -e "$rms_source" && test ! -e "$rms_exec" && test ! -e "$rms_tmp"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u; source /opt/tecoai/setvars.sh; set -u
test "$(readlink -f /home/py312/bin/python)" = /usr/local/python/bin/python3.12
tecocc --version
test -f /root/thirdparty/teco-hal/lib/teco_hal.0.0.2.bc
git clone --no-checkout https://github.com/Tecorigin/teco-ops.git "$rms_source"
git -C "$rms_source" fetch --no-tags origin refs/pull/37/head
git -C "$rms_source" checkout --detach e29b53c256f366e6eee538d656bfbb56e867cefc
test "$(git -C "$rms_source" rev-parse HEAD)" = e29b53c256f366e6eee538d656bfbb56e867cefc
echo '41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be  '"$rms_source/teco/ual/kernel/rms_norm/rms_norm_fp16.scpp" | sha256sum -c -
ln -s /root/thirdparty/teco-hal "$rms_source/thirdparty/teco-hal"
mkdir -p "$rms_tmp" "$rms_exec/tecoops"
(
  cd "$rms_source"
  export PATH=/home/py312/bin:$PATH
  export TMPDIR="$rms_tmp" MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
  export WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF TORCH_DEVICE_BACKEND_AUTOLOAD=0
  unset PYTHONPATH
  /home/py312/bin/python setup.py build_ext --inplace > "$rms_tmp/build.log" 2>&1
)
cp "$rms_source/api/tecoops/__init__.py" "$rms_source/api/tecoops/libteco_ops.so" "$rms_source"/api/tecoops/_torch_ext.cpython-312-*.so "$rms_exec/tecoops/"
sha256sum "$rms_exec"/tecoops/* > "$rms_tmp/package-sha256.txt"
unset TORCH_DEVICE_BACKEND_AUTOLOAD TORCH_COMPILE_DISABLE VLLM_DISABLE_COMPILE VLLM_ENFORCE_EAGER VLLM_TORCH_COMPILE_LEVEL
export PYTHONPATH="$rms_exec:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$rms_exec/tecoops:${LD_LIBRARY_PATH:-}"
export MINICPM_RMS_PACKAGE_ROOT="$rms_exec" SDAA_VISIBLE_DEVICES=1
/home/py312/bin/python "$b/tests/test_rms_epilogue_sdaa.py" --metadata "$b/validation/rms_shapes_20261008.json" --out "$rms_tmp/focused.json"
MINICPM_OP_PROFILE=official MINICPM_SILU_MUL_PROFILE=vendor_swiglu MINICPM_MAX_MODEL_LEN=2048 bash "$b/run.sh"
```

The startup uses existing PYTHONPATH/LD_LIBRARY_PATH initialization binding;
there is no new selector or forward branch. The focused test checks RMS/cache/
flash ABI availability and records actual package/core/extension hashes and
unique mapped RMS core. Vendor SwiGLU comes from vendor sdaa::swiglu, independently
of tecoops. Rebuilt artifact hashes may differ from the measured preserved B;
retain the new build log and hashes, and rerun focused and model verification.

The tested B source snapshot95b0 is not an official Git archive SHA. All559
regular members match the canonical E29 source;558 files including all compiled
sources match bytes, while only noncompiled doc/op_docs/rms_norm.md differs.
Both complete greedy32 outputs match this model's reference, with default
vLLM compile mode3/backend eager/cudagraphNONE, not enforce-eager. Shape-level
speed measurements do not establish model speedup, official accuracy or full
2048-token/long-context generation.
