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
export GEMMA_MAX_MODEL_LEN=2048 GEMMA_MAX_BATCHED_TOKENS=2048
b=model_adaptations/Gemma4SCUdoudui
bash "$b/run.sh"
```

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
  --base http://127.0.0.1:8000 --out model.json --steady 300 --concurrency
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
are untested. The 302-second gate uses the prior frozen isolated core; the fresh
public build has focused and greedy32 validation. Installed teco-ops wheel
equivalence to current upstream heads is not inferred from its version string.
Code retains the included BSD notices and the originating repository licenses.
