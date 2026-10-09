# InternVL3_5-8B (SCU 都队)

This adaptation serves `OpenGVLab/InternVL3_5-8B` on SDAA through TecoVLLM.
Weights remain read-only under `MODEL_ROOT`; no vendor site-package or model
copy is required.

## Launch

```bash
MODEL_ROOT=/gpfs/model bash model_adaptations/InternVL3_5SCUdoudui/run.sh
```

The public command is `vllm serve /gpfs/model/OpenGVLab/InternVL3_5-8B` with
host `0.0.0.0` and port `8002`, so the official `parse_run_sh` reads the actual
defaults. A shell-local `vllm` function resolves the model, first host flag and
first port flag from the existing initialization environment, then executes
the same vendor Python `vllm.entrypoints.openai.api_server` entry. Extra CLI
arguments remain last, preserving their override order. No global binary or
vendor package is changed.

The default contract uses FP16, tensor parallel size 2, an explicit port,
`--trust-remote-code`, `--no-enable-prefix-caching`, and
`--no-enable-chunked-prefill`. The launcher sources `/opt/tecoai/setvars.sh`
and explicitly uses `/home/py312/bin/python` (resolved target
`/usr/local/python/bin/python3.12`). `VLLM_DISABLE_COMPILE_CACHE=1` prevents
reuse of graphs from the previous operator binding. Override the local path,
port, host, and sequence limit with `INTERNVL_MODEL`, `INTERNVL_PORT`,
`INTERNVL_HOST`, and `INTERNVL_MAX_MODEL_LEN`.
The official CI entry point `tools/ci_pipline/run_ci.py` drives this tree's
`model_adaptations/InternVL3_5SCUdoudui/run.sh`, which uses FP16 / TP2 /
`--limit-mm-per-prompt '{"image": 1}'` / `--no-enable-chunked-prefill` and the
overridable `--max-model-len` default of 4352. The team's internal launcher
`scripts/serve_internvl.sh` (private accelerator repo, not part of this submission)
states the same contract and the same 4352 default, so both entries state one capacity.
The historical baseline was tested with TP2, FP16, and a sequence limit of 4096.
The candidate changes only the default sequence limit to 4352 for the official
T2 4096-token prompt plus 100-token output. On 2026-10-05, both TP workers
reported max_model_len=4352, max_num_batched_tokens=4352, FP16, and chunked
prefill disabled. The branch's Q16/KV4/D128 capacity prefill, long-position RoPE
gate, fixed text/image greedy32 IDs, and official T2 capacity run passed. T2
plus one same-shape supplemental run recorded 453.985099 seconds of summed
SQLite request spans with 20/20 requests successful. This is capacity and
correctness evidence only; it makes no speedup, official-accuracy, or full-CI
claim. The 12.106 GiB/rank worker peak covers the post-reset T2/steady request
window; startup/profile peak was not measured. See
validation/ci_budget4352_20261005.json for receipts and raw-log hashes. The
public launcher change is tracked by [PR #6](https://github.com/Tecorigin/tecovllm-modelzoo/pull/6); this receipt records its branch-local capacity and correctness evidence.


## Capacity gate reproduction

From the repository root with an available SDAA device, run the public tests
directly from this adaptation. These commands use the public runtime files and
the vendor Python; they do not depend on the internal op_learning helpers.

~~~bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/tecoai/setvars.sh
ADAPTATION="$PWD/model_adaptations/InternVL3_5SCUdoudui"
export SDAA_VISIBLE_DEVICES=0
/home/py312/bin/python "$ADAPTATION/tests/test_capacity_attention.py"
/home/py312/bin/python "$ADAPTATION/tests/verify_long_vendor_rope.py" --allow-device --device 0 --result /tmp/internvl-rope-budget4352.json
~~~

The first test exercises public Q16/KV4/D128 FP16 prefill at sequence length
4352 against a bounded FP32 CPU oracle. The second verifies RoPE at positions
4096, 4195, and 4351 using the unchanged branch tolerance and public
verify_vendor_rope.py implementation.

## Operator path

The process-init overlay binds the Qwen3 language tower's RMSNorm to the
official `tecoops.rms_norm` ABI and replaces the unimplemented SDAA
`BlockAttentionImpl.forward`.  The attention adapter writes paged KV cache via
`reshape_and_cache`, gathers complete per-request K/V blocks, and dispatches
prefill through the fused SDAA SDPA path.  The official flash ABI wrapper is
kept in the bundle for the model's operator contract and independent checks.

Qwen3 language attention also binds to the installed vendor
`torch.ops._C.rotary_embedding` at construction. Each attention receives its
own shallow RoPE module copy and independent module dictionaries, while all
copies retain the original cache Tensor. The original globally cached module
and its data remain unchanged. The opaque mutation declaration and Fake
implementation preserve graph boundaries; the forward path contains no
backend choice or environment lookup.

This integration uses an existing vendor API. It adds no UAL implementation or
binary. Initialization checks require packed Q16/K4, head and rotary dimension
128, full Neox FP16, base 1e6, and a contiguous 40960-by-128 cache on the same
SDAA device as the projection weight. Unsupported contracts and library hashes
raise explicit errors, including under `python -O`; there is no fallback.
The tested extension SHA256 is
`9ecc2cb57a5c870704580e39112c1e95321bc076507139d8b8855273a90cfec0`,
and the vendor core SHA256 is
`adf3014a8fd3ced720bbf5cfe666752b8583dc76b17b5c96517fc70716773cb1`.
The authoritative full hashes are also checked in `runtime/custom_ops/rotary/op.py`;
updating a vendor binary requires repeating the focused and model gates.

## Focused verification

Run from the modelzoo root after sourcing the vendor environment:

```bash
ADAPTATION="$PWD/model_adaptations/InternVL3_5SCUdoudui"
PYTHONPATH="$ADAPTATION/runtime" SDAA_VISIBLE_DEVICES=0,1 \
  /home/py312/bin/python -O "$ADAPTATION/tests/test_model_binding.py" \
  --guard-only --result /tmp/internvl-rope-guards.json
PYTHONPATH="$ADAPTATION/runtime" SDAA_VISIBLE_DEVICES=0,1 \
  /home/py312/bin/python "$ADAPTATION/tests/test_model_binding.py" \
  --device 0 --result /tmp/internvl-rope-focused.json
```

The focused suite checks invalid initialization contracts, instance and cache
isolation, Fake handling, and 12 raw/compiled wrapper cases at M1 nonzero
positions 17/4095 and M8 mixed positions on default and nondefault streams.
Primitive comparison uses fixed `rtol=atol=0.002`; vendor arithmetic is not
bit-exact with the native/CPU reference, and known FP16 subnormal loss remains.

In the previous 4096-context RoPE gate, the exported `run.sh` was tested with the same FP16 TP2 configuration and
fixed text prompt as the native baseline, two warmups, then three consecutive
greedy32 requests. The native reference was recorded before the private
binding experiment and reused for this formal-entry reproduction.
Baseline seconds were 6.898981428 / 6.909929068 / 6.905793745 (median
6.905793745); exported-entry seconds were 6.573132289 / 6.573451439 /
6.573975117 (median 6.573451439), a 4.8125% reduction for this fixed text case. Profiling
was disabled during timing. Across four decode steps per rank, 3600 native
RoPE kernels became 144 fused kernels; all other kernel-name counts were
unchanged. Text/image greedy32, concurrency4/8 and 305.7865 seconds of 82 mixed
requests matched baseline token IDs. This result does not establish official
task accuracy or image performance.
Configuration, timings, source/library identities and trace counts are in
`validation/rope_model_20261004.json`.

The source branch `model/internvl3_5-8b` contains real-shape hardware evidence
for the RMSNorm, reshape/cache, and prefill paths, plus fixed 32-token text and
natural-image smoke logs.  The prefill adapter explicitly disables chunked
prefill because SDAA's non-square causal mask path is not semantically safe.
Full official CI and accuracy evaluation remain the repository's external
evaluation step.


## RMSNorm SIMD epilogue validation and isolated rebuild

The independent 2026-10-07 InternVL gate tests an epilogue-only patch of
[official RMSNorm PR37](https://github.com/Tecorigin/teco-ops/pull/37), starting
from `8f896f2a9103cc9c684eb4488c9f885a38f15eb6`. The patch changes only the two
FP16 output loops: widen, multiply by rstd then weight in FP32, narrow, and
retain the scalar tail. Reduction order, residual rounding, DMA, stream and
ABI remain unchanged. The default launcher/runtime above are unchanged.

The 32 focused cases match baseline output/residual bits, including
hidden4096 model/boundary cases and q/k128 global-head stress, stream and tail
probes. Historical q/k32/8-head rows are not TP2 per-rank measurements.
Three identical micro trials improve the tested long shapes; raw triples, medians and
original CPU oracle limits/differences are in
`validation/rms_epilogue_simd_20261007.json`. Baseline odd-half DMA and CPU
rounding limitations are recorded rather than attributed to this patch.

Independent TP2 public launches match all fixed text/image greedy32 token
IDs; candidate steady318.024 seconds/48requests passed. Request-window peak
is11.713GiB allocated/13.039GiB reserved perworker. Worker mappings/hashes and
145 selected RMSNorm layers confirm package binding. Model timing is only a
single baseline-to-candidate observation: all raw values are retained, with
no causal whole-model speed claim or official accuracy claim.

The public source is pinned to PR37 revision
`e29b53c256f366e6eee538d656bfbb56e867cefc`, whose kernel bytes match the tested
candidate source SHA below. No private task DSO is required. The bundled
`validation/epilogue-only.patch` also reproduces the same kernel from the
exact `8f896f2` baseline; the promoted revision includes documentation, so
its full archive hash differs from the original baseline-plus-patch archive.
For an isolated rebuild, use a fresh official checkout at the promoted revision:

```bash
# ADAPTATION is this public package's absolute directory.
# OPS_SRC is a fresh clean checkout of https://github.com/Tecorigin/teco-ops
# on a writable RAM build area; HAL_ROOT is the existing teco-hal0.0.2 dependency.
git -C "$OPS_SRC" checkout e29b53c256f366e6eee538d656bfbb56e867cefc
# Alternative exact source reproduction: checkout8f896f2a9103cc9c684eb4488c9f885a38f15eb6
# then git apply --check and git apply validation/epilogue-only.patch.
# Do not apply the patch again to the promoted revision.
source /opt/tecoai/setvars.sh
PYTHON=/home/py312/bin/python
test "$(readlink -f "$PYTHON")" = /usr/local/python/bin/python3.12
export WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# TMPDIR must be a short executable path; /dev/shm can be noexec.
export TMPDIR=/tmp/rn26
mkdir -p "$TMPDIR" "$OPS_SRC/thirdparty" "$OPS_SRC/dist"
ln -s "$(readlink -f "$HAL_ROOT")" "$OPS_SRC/thirdparty/teco-hal"
unset PYTHONPATH
(cd "$OPS_SRC" && "$PYTHON" setup.py bdist_wheel --dist-dir "$OPS_SRC/dist")
# PACKAGE_OUT must be a fresh executable directory (not noexec RAM).
"$PYTHON" - "$OPS_SRC/dist" "$PACKAGE_OUT" <<'PY'
import hashlib, json, pathlib, sys, zipfile
wdir, output = map(pathlib.Path, sys.argv[1:])
wheel, = wdir.glob('*.whl')
output.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        parts = pathlib.Path(name).parts
        if len(parts) == 2 and parts[0] == 'tecoops' and (name.endswith('.so') or name.endswith('__init__.py')):
            dest = output / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(archive.read(name))
(output / 'build-sha256.json').write_text(json.dumps({
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (output / 'tecoops').iterdir()}, indent=2))
PY
# Initialization chooses this isolated package; public run.sh adds its overlays.
export PYTHONPATH="$PACKAGE_OUT"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
bash "$ADAPTATION/run.sh"
# No global site-packages installation or model-weight download/copy.
```

Check candidate kernel source SHA256
`41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be`.
For deployment, record each locally rebuilt extension/core hash and verify
actual worker module paths and unique mapped core against that package.
The recorded test-library hashes identify the tested build, while the pinned
source and patch identify what to rebuild. Re-run the independent focused
and fixed-input model gates for a new build; the source commit alone does
not prove another build or model's performance.


## Plain RMSNorm output writeback overlap (2026-10-07)

This isolated patch moves only the plain output-DMA wait to the existing
same-output-buffer reuse point and drains both output handles before freeing
SPM. Input/output ping-pong buffers are separate. The add kernel, reduction
order, FP16 rounding, row layout, stream and ABI remain unchanged; public
run.sh/runtime defaults are unchanged.

[Independent proof](validation/rms_plain_writeback_20261007.json) preserves
all 48 focused cases, CPU oracle differences with original .005/.01 thresholds,
output/residual bit equality against current E29, reuse stress, Fake/fullgraph,
actual TP2 shapes and every A-before/B/A-after raw triple/median. Same public
TP2 FP16 ctx4352 fixed text/image greedy32 passed with two warmups/three trials;
candidate steady passed 317.157 seconds / 16 rounds / 48 requests. Actual worker maps,
145 selected norms per worker and peaks 11.713 allocated / 13.039 reserved GiB/rank
are recorded. Capture-limited call receipts do not count every model forward.

A separate read-only metadata supplement (no generation/timing) verifies
36 attention modules per worker, TP2, 16 query / 4 KV heads, D128, q/k norm weights [128]
FP16 on SDAA and epsilon 1e-6. Historical 32/8-head q/k results are global-head
stress; their original values and precision conclusions remain intact.

The timer is synchronized perf_counter operator-call latency including
Python/dispatch, not pure device-kernel time. Actual TP2 long q/k plain calls
are 12.2–13.2% lower versus both A arms; hidden [1811/4352,4096] calls about 1.7%.
Short-shape/control observations (including minor negatives and changes in
unchanged add) are retained without a speed claim. Model A-to-B timings are
observations only: no E2E gain or official task-accuracy claim.

For isolated rebuilding, start from official PR37
e29b53c256f366e6eee538d656bfbb56e867cefc or documentation head
2949f7061a05af0f4cb2b88fb3344380e3dfc4e0: both have canonical kernel SHA256
41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be.
Apply [the public U0 patch](validation/plain-writeback-only.patch), then use
the preceding vendor-Python/SDK isolated bdist_wheel/extraction/startup
PYTHONPATH recipe. No global package or model weight changes are needed.

    git -C "$OPS_SRC" checkout --detach e29b53c256f366e6eee538d656bfbb56e867cefc
    git -C "$OPS_SRC" apply --unidiff-zero --check "$ADAPTATION/validation/plain-writeback-only.patch"
    git -C "$OPS_SRC" apply --unidiff-zero "$ADAPTATION/validation/plain-writeback-only.patch"
    sha256sum "$OPS_SRC/teco/ual/kernel/rms_norm/rms_norm_fp16.scpp"

Expected candidate SHA256:
93e8e80d08d401fffaa3060bf8e6e24bd989ab59fd8cb09c079228710f66d5b7.
The U0-applied bytes exactly match tested source; tested U3 is retained in raw
evidence. Tested baseline core/extension hashes start d26791a5/c7d4a8c0;
candidate hashes start ed1358d6/e34873d8 (full hashes in proof). Rebuilt binaries
must record their own hashes and re-run gates; source identity does not prove
another build's performance. No binary artifact is included.
