# MiniCPM5-1B (SCU 都队)

This adaptation binds the existing official `tecoops` RMSNorm, KV-cache write
and paged prefill-attention APIs to MiniCPM5-1B. It uses process-local overlays,
the vendor interpreter and read-only pre-provisioned weights. It contains no new
operator kernel or performance claim.

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
