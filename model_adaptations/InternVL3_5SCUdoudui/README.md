# InternVL3_5-8B (SCU 都队)

This adaptation serves `OpenGVLab/InternVL3_5-8B` on SDAA through TecoVLLM.
Weights remain read-only under `MODEL_ROOT`; no vendor site-package or model
copy is required.

## Launch

```bash
MODEL_ROOT=/gpfs/model bash model_adaptations/InternVL3_5SCUdoudui/run.sh
```

The default contract uses FP16, tensor parallel size 2, an explicit port,
`--trust-remote-code`, `--no-enable-prefix-caching`, and
`--no-enable-chunked-prefill`. The launcher sources `/opt/tecoai/setvars.sh`
and explicitly uses `/home/py312/bin/python` (resolved target
`/usr/local/python/bin/python3.12`). `VLLM_DISABLE_COMPILE_CACHE=1` prevents
reuse of graphs from the previous operator binding. Override the local path,
port, host, and sequence limit with `INTERNVL_MODEL`, `INTERNVL_PORT`,
`INTERNVL_HOST`, and `INTERNVL_MAX_MODEL_LEN`. The validated model checks use
TP2, FP16, and a sequence limit of 4096.

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

The exported `run.sh` was tested with the same FP16 TP2 configuration and
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
Official benchmark and accuracy jobs remain the repository's external
evaluation step.
