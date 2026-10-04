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
export GEMMA_MAX_MODEL_LEN=2304 GEMMA_MAX_BATCHED_TOKENS=2304
b=model_adaptations/Gemma4SCUdoudui
bash "$b/run.sh"
```

The default is the validated 2304-token profile on port 8003, with served
name `gemma-4-12B-it`. The actual literal serve command is accepted by the
unmodified official `tools/ci_pipline/run_ci.py`; environment overrides
are resolved once before the vendor entrypoint. Parser validation covers
startup metadata, while official accuracy and the full performance suite
remain pending.

The entry locks `/home/py312/bin/python` and `VLLM_DISABLE_COMPILE_CACHE=1`.
It uses FP16, automatic KV-cache dtype, TP2, disabled prefix caching and chunked
prefill, and the installed default `VLLM_COMPILE` mode with its SDAA eager
backend. It does not select `--enforce-eager`. `GEMMA_HOST`, `GEMMA_PORT`,
`GEMMA_PATH`, `GEMMA_UTIL` and the length variables configure startup; additional
CLI arguments can be passed to `run.sh` for diagnostics.

Without the optional extension below, global D512 decode uses the existing
single-KV-head math implementation. D256 decode retains the trailing 1024 KV
positions and fused non-causal SDPA. Both choices bind during construction.
Prefill and vendor `reshape_and_cache` remain unchanged. The overlay also
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

## Official T1 input and output budget

The 2304-token default reserves room for the official 2048-input / 100-output
T1 case. TP2, FP16, util0.92, attention implementations, full KV layout and
prefix/chunked flags are unchanged. This is a capacity enabler.
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

The 2048-profile timings above retain their original scope. Reproduce them by
setting `GEMMA_MAX_MODEL_LEN=2048 GEMMA_MAX_BATCHED_TOKENS=2048` before launch.
T1 capacity/stability does not establish a new speedup or official accuracy.
T2/T3/T4 (4K/8K/16K), multimodal evaluation and owner CI remain pending.

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
