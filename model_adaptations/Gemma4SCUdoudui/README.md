# Gemma 4 12B-it (SCU 都队)

This text-generation adaptation uses the installed TecoVLLM/SDAA stack and
read-only `MODEL_ROOT` weights. Its process-local initialization binds Gemma's
D256 sliding-window decode and D512 global decode; vendor packages are unchanged.

## Launch the validated configuration

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/tecoai/setvars.sh
export PATH="/home/py312/bin:${PATH}"
export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0
export MODEL_ROOT=/gpfs/model
export GEMMA_TP_SIZE=2 GEMMA_UTIL=0.92
export GEMMA_MAX_MODEL_LEN=4352 GEMMA_MAX_BATCHED_TOKENS=512
b=model_adaptations/Gemma4SCUdoudui
bash "$b/run.sh"
```

The current default is the 4352-token capacity profile on port 8003, with
served name `gemma-4-12B-it`. The actual literal serve command is accepted by
the unmodified official `tools/ci_pipline/run_ci.py`; environment overrides
are resolved once before the vendor entrypoint. The official T2 request and
full-context boundary checks recorded below pass at this configuration. They
establish capacity and stability only; official accuracy, T3/T4, and current
owner CI remain pending.

The entry locks `/home/py312/bin/python` and `VLLM_DISABLE_COMPILE_CACHE=1`.
It uses FP16, automatic KV-cache dtype, TP2, disabled prefix caching, enabled
chunked prefill, and the installed default `VLLM_COMPILE` mode with its SDAA eager
backend. It does not select `--enforce-eager`. `GEMMA_HOST`, `GEMMA_PORT`,
`GEMMA_PATH`, `GEMMA_UTIL` and the length variables configure startup; additional
CLI arguments can be passed to `run.sh` for diagnostics.

Without the optional extension below, global D512 decode uses the existing
single-KV-head math implementation. D256 decode retains the trailing 1024 KV
positions and fused non-causal SDPA. Both choices bind during construction.
Math prefill uses the window-aware entry described below; the vendor
`reshape_and_cache` ABI remains unchanged. The overlay also
registers `gemma4_unified`, adapts the installed tokenizer API, and skips
vision/audio-only checkpoint tensors for this text model.

## Optional isolated D512 global decode

This packages the previously accepted combination of official teco-ops
PR [36](https://github.com/Tecorigin/teco-ops/pull/36),
[41](https://github.com/Tecorigin/teco-ops/pull/41) and
[42](https://github.com/Tecorigin/teco-ops/pull/42). It adds no new kernel or
window entry. Its FP16 TP2 contract is Hq8/Hkv1/D512, HND block32, one query
token per sequence and scale 1.0. D256 and prefill still use the paths above.

Use an existing teco-ops Git checkout containing base
`de27305efed0a17ae926d21d5415d8b915614649` and the supplied teco-hal installation.
The output directory must be new; the helper archives that base, applies the
included patch, links teco-hal, and builds without installing a wheel globally.

```bash
export TECO_OPS_SOURCE=/path/to/teco-ops
export TECO_HAL=/path/to/teco-hal
export D512_BUILD="$PWD/build/gemma-d512"
/home/py312/bin/python "$b/scripts/gemma4/build_d512_official.py" \
  --source-repo "$TECO_OPS_SOURCE" --teco-hal "$TECO_HAL" --output "$D512_BUILD"
export GEMMA_D512_OFFICIAL_EXTENSION="$D512_BUILD/api/tecoops/_torch_ext.cpython-312-loongarch64-linux-gnu.so"
export GEMMA_D512_OFFICIAL_SHA256="$(sha256sum "$GEMMA_D512_OFFICIAL_EXTENSION" | cut -d ' ' -f1)"
bash "$b/run.sh"
```

The loader checks the extension digest and unique `libteco_gemma_flash.so`
dependency, records the actually loaded core digest, and binds only supported
global layers. Missing or mismatched selections terminate startup, including
under optimized Python. The output mutation and Fake implementation are explicit.

## Validation

The D512 focused scripts accept `--build-provenance PATH` for a freshly built
library. The receipt is checked against the official base, combined patch SHA,
the realpaths of the recorded and running vendor Python, and the selected
extension/core byte digests. Recorded library paths may differ if the selected
files were copied, provided both digests match. Omitting the option retains
the frozen extension/core SHA checks. After the documented build and library
selection, use the returned receipt for both focused scripts:

```bash
/home/py312/bin/python "$b/tests/test_d512_build_provenance.py"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/verify_d512_4352.py" --out d5124352_operator.json --build-provenance /path/to/build-provenance.json
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/verify_d512_poisoned_tail.py" --out d5124352_poisoned_tail.json --build-provenance /path/to/build-provenance.json
```

The 2026-10-07 fresh isolated public build passes all six provenance unit tests,
eight ordinary operator cases, 12 boundary case/stream records (max absolute
error 0.001107931 against the unchanged 0.02 limit), and six poisoned-tail
records with zero control-to-poison delta and exact cache preservation. The
unchanged public TP2/FP16/4352-context entry completed 83 requests with exact
32-token IDs, including 65 steady requests over 322.430396 seconds. Its three
timing values and worker memory peaks are recorded in
`validation/build_provenance_20261007.json`; no speedup is claimed. The isolated
combination is runtime-equivalent to current operator PRs 36/41/42, but is not a
full merged upstream wheel or a new official accuracy result. This run covers
short model prompts; the existing long-context receipts retain their own scope.

```bash
# CPU/Fake/fullgraph and constructor/loader rejection tests, without selection:
env -u GEMMA_D512_OFFICIAL_EXTENSION -u GEMMA_D512_OFFICIAL_SHA256 \
  PYTHONPATH="$b/runtime" /home/py312/bin/python "$b/scripts/gemma4/test_d512_official_cpu.py"
# After selecting the isolated extension:
PYTHONPATH="$b/runtime" /home/py312/bin/python \
  "$b/op_learning/attention/gemma-d512-model/verify_operator.py" operator.json
# Against the separately started selected server:
/home/py312/bin/python "$b/op_learning/attention/gemma-d512-model/verify_model.py" \
  --base http://127.0.0.1:8003 --out model.json --steady 300 --concurrency
```

`validation/runtime_20261004.json` freezes source identities and results. Eight
real-shape cases across two streams pass (max absolute error 0.001029968),
including fullgraph, padded metadata and noncontiguous queries. Two fixed
greedy32 runs, concurrency 1/4/8 and 302.171015 seconds / 65 steady requests
match every reference token ID. The fixed prompt has 11 input tokens.

With TP2/util0.92/max-length2048, two warmups then three HTTP measurements,
the unselected export takes 13.543721 / 13.553259 / 13.569659 seconds
(median 13.553259); the selected export takes 12.941327 / 12.768881 / 12.942871
(median 12.941327), a 4.515% reduction for this fixed greedy32 case. This
reproduces the accepted combination; it does not isolate PR36, PR41 or PR42,
or assign a separate speedup to packaging. Profile data is excluded from timing.

The public build command also completes in a fresh directory, and its new
library passes all eight operator cases plus a fresh exported greedy32 run.
Worker receipts record source/library hashes, FP16 SDAA weights and peak memory.
The tested stack is PyTorch 2.12.0a0+git0d62256, Torch-SDAA
20260623.8.51.dev0+gitd942f23, vLLM 0.20.2.dev0+g132765e35.d20260728 and
driver/runtime 3.2.0 (custom DNN 3.2.1a0).

This is temporary text regression evidence. Official task accuracy and current
head owner CI remain pending. Full 8192-token generation and multimodal inputs
are untested. The original 302-second gate uses the prior frozen isolated core.
`validation/runtime_20261005.json` adds a separate fresh public build: all
eight operator cases and 83 greedy32 responses pass, including 306.875936
seconds / 65 steady requests. Both workers load its recorded core digest.
Fresh same-input timings are math 13.153049603 / 13.290304534 / 13.146468690
seconds (median 13.153049603) and selected D512 13.006247967 / 13.010500995 /
13.060514411 (median 13.010500995), a 1.084% reduction for this fixed case.
This repeats the combined path comparison; it does not isolate a PR or
establish an independent packaging speedup. Installed teco-ops wheel
equivalence to current upstream heads is not inferred from its version string.
Code retains the included BSD notices and the originating repository licenses.

## Evaluation-client tokenizer compatibility

The installed tokenizer API expects a dict for `extra_special_tokens`, while
this checkpoint supplies a list. Use the explicit client-only overlay before
starting EvalScope; it reuses the existing normalization function without
installing model, attention or config overlays:

```bash
b="$PWD/model_adaptations/Gemma4SCUdoudui"
GEMMA4_EVAL_TOKENIZER=1 PYTHONPATH="$b/eval_overlay" \
  bash tools/ci_pipline/speed.sh gemma-4-12B-it \
  /gpfs/model/google/gemma-4-12B-it 0.0.0.0 8003
/home/py312/bin/python "$b/tests/verify_eval_tokenizer.py"
```

Source the SDK and put `/home/py312/bin` first on PATH as in the launch example.
The client overlay requires explicit activation and changes only process-local
tokenizer initialization. It preserves raw tokenizer.json encoding IDs; config,
weights and installed source files remain unchanged. This focused client gate
is not a model smoke, official accuracy, full CI or performance result. Server
context-budget validation is a separate attempt.

## Historical official T1 input and output budget (2304-token profile)

The former 2304-token context/batch profile reserved room for the official
2048-input / 100-output T1 case. That T1 run is historical; the current
launcher default is 4352/512. Its TP2, FP16, util0.92 attention implementation,
KV layout and prefix/chunked behavior are recorded at their original scope.
This was a capacity enabler, not an accuracy or speedup result.
`validation/ci_budget2304_20261005.json` records eight longer-context D512
operator cases (maximum absolute error 0.0010442734),
same-config math/selected controls with four matching greedy32 responses and
four matching exact-2048-input greedy100 responses, plus five startup argv
cases. KV capacity is 2855 tokens.

With the explicit client tokenizer overlay, the unchanged official T1 arguments
and 1K/10-token warmup complete: all ten T1 database rows succeed and return
100 output tokens. The measured request span is
457.818861 seconds; total validated steady
duration is 457.818861 seconds. Four greedy32 responses
before/after T1 match the committed reference. Both workers load the frozen
isolated D512 core; owned services are released.

The 2048-profile timings above retain their original scope. Reproduce them with
`GEMMA_MAX_MODEL_LEN=2048 GEMMA_MAX_BATCHED_TOKENS=2048 bash "$b/run.sh" --no-enable-chunked-prefill`.
T1 capacity/stability did not establish a new speedup or official accuracy.
The separate 4352 profile now passes the official T2 capacity/stability request
as described below. T3/T4 (8K/16K), multimodal evaluation, the official
accuracy suite and current owner CI remain pending.

After starting the selected server, reproduce the official warmup and T1
individually with vendor Python (all request arguments are unchanged):

```bash
b="$PWD/model_adaptations/Gemma4SCUdoudui"
for task_shape in 1024:10:1 2048:100:10; do
  IFS=: read -r task_prompt task_output task_number <<< "$task_shape"
  GEMMA4_EVAL_TOKENIZER=1 PYTHONPATH="$b/eval_overlay" \
    /home/py312/bin/python /home/py312/bin/evalscope perf \
    --model gemma-4-12B-it --url http://127.0.0.1:8003/v1/chat/completions \
    --api openai --tokenizer-path /gpfs/model/google/gemma-4-12B-it \
    --dataset random --min-prompt-length "$task_prompt" --max-prompt-length "$task_prompt" \
    --max-tokens "$task_output" --min-tokens "$task_output" \
    --extra-args '{"ignore_eos": true}' --parallel 1 --number "$task_number"
done
```

## Window-aware math prefill (2026-10-05)

The math prefill entry accepts contiguous appended chunks with `q_len <= kv_len`.
For local W1024 layers it gathers only
`[max(0, kv_len - q_len - 1024 + 1), kv_len)`; global layers retain full history.
The existing right-aligned math mask, flash entry, shared gather and decoder are
unchanged. Required pages must be valid. Retired prefix pages and poisoned old
values outside that interval are not read, including old offsets in the first
retained boundary page. The global null-page check is specific to this Gemma
TP2 manager64-to-kernel32 contract (reserved subpages 0/1).

The portable focused test checks actual local Q8/K4/D256/W1024 and global
Q8/K1/D512 against an independent CPU FP32 absolute-position oracle. It uses
the vendor cache-write ABI with continuous HND caches, noncontiguous K/V tails,
physical page permutations, mixed q1/padding, required-null rejection and
persistent serial chunks on default/nondefault streams. Run on an owned free
device after exporting `LD_LIBRARY_PATH` and sourcing `/opt/tecoai/setvars.sh`:

```bash
b="$PWD/model_adaptations/Gemma4SCUdoudui"
SDAA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/py312/bin/python "$b/tests/verify_windowed_math.py" \
  --allow-device --output windowed_math_operator.json
```

Public-source validation passes 84 numerical cases, 12 serial cases and 252
required-null rejections; maximum oracle error is 0.000542039 and serial versus
whole error is 0.0000621676, with exact cache values. The unchanged public
launcher passes fixed greedy32, a 2048-token input and mixed 11/1057 requests
against the model's own references. An optional 2304-context chunk512 run also
passes real scheduler metadata and at least 300 seconds of output/cache
stability. `validation/windowed_math_20261005.json` retains the source hashes,
execution/final-test license lineage and independent private three-run values.
These results establish chunked-prefill capability; they do not establish an
official accuracy score, 3492+ context capacity, graph capture or a new speedup.

Those 2304-token measurements used the explicitly configured historical
profile; chunked prefill was enabled only for the recorded chunk512 run. The
current launcher defaults to maxlen4352/batch512 with chunked prefill enabled
and prefix caching disabled. Its independent public-path focused and model
checks are recorded below. The math metadata checks synchronize on the host; graph
capture support was not tested. Runtime diagnostics and private metadata
hooks are excluded from the portable product tests.


## 4352-token capacity profile (2026-10-05)

The current default is TP2, FP16, utilization 0.92, max model length 4352,
max batched tokens 512, chunked prefill enabled, and prefix caching disabled.
D256/window1024 decode, the window-aware math prefill source, cache ABI,
overlay and accepted optional D512 kernel were not changed by this capacity
configuration. The profile covers the official T2 request budget of 4096 input
plus 100 output tokens, as well as a 4320-input plus 32-output endpoint at
4352 total tokens.

The service's actual SDK hybrid-pool receipt reports 511 manager blocks and a
4352-token request-memory lower bound from the live hybrid specs and allocated
pool. This is the SDK capacity calculation, not a maximum inferred from cache
views. The worker memory measurements have separate scopes: startup peak
allocated was 13.745507717 GiB; after resetting the counters before T2, the
request-window peak was 12.236205578 GiB; end allocated was 12.052499771 GiB;
reserved was 14.1328125 GiB. The reset-window peak does not include startup.

The unchanged official warmup and T2 arguments were used: random dataset,
1024 prompt / 10 output / one request for warmup, then fixed 4096 prompt / 100
output, `ignore_eos=true`, parallel 1 and 10 requests for T2. All ten T2 rows
succeeded and produced 100 output tokens; observed prompt counts were 4096–4098.
The SQLite request span from the first request start through the last completion
was 506.650664 seconds. This is a capacity/stability receipt, not a latency
comparison or accuracy score.

Before and after the math-service T2 window, fixed greedy32 IDs matched their
same-configuration reference. Two 4096-input greedy32 responses were exact
same-configuration repeats; the receipt explicitly provides no 2304-profile or
external reference for that prompt. A 4320-input plus 32-output request reached
the 4352 endpoint and matched the saved math reference. The longest MMLU chat
fixture contained 2468 input tokens with a 1024-token output budget; the service
accepted it and naturally stopped after 454 output tokens. That request had no
precision score and is not an accuracy result.

The optional selected D512 service used the same 4352/512 configuration. Its
fixed, 4096-input, concurrent short/long, post-run fixed, and boundary outputs
were checked against this model's math references. The 4320+32 endpoint
completed, followed by 13 successful steady requests over 308.918429 seconds;
the boundary request was repeated afterward. This is capacity and stability
evidence, not a D512 speedup claim. The public-path D512 focused tests also
pass: the main gate records 12 case/stream rows and 24 reported stream
invocations, max absolute error 0.001107931 against the unchanged 0.02 limit,
and exact cache preservation. The poisoned-tail gate records 6 rows / 12
reported invocations, maximum error 0.000973701, control-to-poison output delta
0, and bitwise-exact cache preservation at the same 0.02 limit.

The frozen private and final public-path math entries each pass 160 numerical
cases, 12 serial cases, and 480 required-null rejections, with maximum absolute
error 0.001144115 and serial-versus-whole error 0.0001220703125. Cache contents
are exact. Both routes use the same kernel and unchanged oracle thresholds.

For the focused checks, first source the SDK and supply an idle device owned by
the current run. The math test requires the explicit device opt-in flag. The
D512 scripts use the original extension-path and SHA environment interface;
the extension must be the accepted file and have its matching core beside it.
The D512 SHA values below identify the tested artifact. The public focused
receipts record execution through these relative-path entries:

```bash
b="$PWD/model_adaptations/Gemma4SCUdoudui"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/tecoai/setvars.sh
SDAA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/py312/bin/python "$b/tests/verify_windowed_math4352.py" --allow-device --output math4352_operator.json
export GEMMA_D512_OFFICIAL_EXTENSION=/path/to/accepted/_torch_ext.cpython-312-loongarch64-linux-gnu.so
export GEMMA_D512_OFFICIAL_SHA256=9339b3756b49f9a4e2ec32e98f6b5e2f812d7790ceab360601238f77f517c7cb
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/verify_d512_4352.py" --out d5124352_operator.json
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/verify_d512_poisoned_tail.py" --out d5124352_poisoned_tail.json
```

The consolidated `validation/ci_budget4352_20261005.json` retains source/test
lineage, all ten T2 metric values, memory scopes and raw manifest hashes. All
130 listed raw files were independently hash-verified after local transfer.
T3/T4, the full 200-example repo-CI precision suite, current owner CI, graph
capture, multimodal evaluation and current upstream-head wheel equivalence
remain unexecuted or unverified. This capacity change makes no new speedup or
final competition accuracy claim. Historical 2048/2304 results retain their
original configuration and measurement scope.
