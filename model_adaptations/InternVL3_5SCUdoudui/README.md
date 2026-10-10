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

---

> **路径说明（面向模型仓读者）**：下文两节里出现的 `op_learning/**`、`models/deformable-detr/**`
> 与 `scripts/**` 属于本队**私有加速器仓**，不在本包（模型仓提交）内；这里保留与加速器仓交付记录
> **逐字一致**的原文，证据路径仅供追证。`model_adaptations/InternVL3_5SCUdoudui/runtime/custom_ops/**`
> 则**是本包内容**。

### `custom_ops` 双副本与解析来源（2026-10-10 冻结前发现；同日 worker 内实测复核）

本仓同时存在两份 `custom_ops`：根级 `custom_ops/**` 与交付副本
`model_adaptations/InternVL3_5SCUdoudui/runtime/custom_ops/**`（后者是提交给模型仓的那份）。
**本线两者当前逐字节相同**（含 `overlay/sitecustomize.py` `ae0bb38b…`、`__init__.py` `46b713a9…`），
且**绑定发生在 `site` 初始化阶段**：先被搜到的那个 `overlay/sitecustomize.py`
（`<frozen site>` 的 `execsitecustomize` 帧）会立刻 `import custom_ops`，
**早于 `python -m` 把 cwd 插到 `sys.path[0]`**。所以「从仓库根启动」本身不改变绑定，
真正决定来源的是**启动器写进 `PYTHONPATH` 的那两份路径**。

下表两行均经 **TP worker 内 `/collective_rpc`** 实测（`VLLM_SERVER_DEV_MODE=1` +
诊断用 `worker_extension_cls`；只读、不改交付文件、不引入热路径 `getenv`），
每形态两个 worker 结论一致：

| 启动形态（cwd 均为仓库根） | worker 内 `custom_ops.__file__` | 收据 |
| :--- | :--- | :--- |
| `scripts/smoke_internvl.sh` / `scripts/serve_internvl.sh`（现状；`PYTHONPATH=${REPO_ROOT}/custom_ops/overlay:${REPO_ROOT}`） | **仓库根** `custom_ops/__init__.py` | `worker-origin-rpc-B2-serve.json`（pid 785331 / 785332） |
| 交付入口 `model_adaptations/InternVL3_5SCUdoudui/run.sh`（`PYTHONPATH=<delivery>/runtime/overlay:<delivery>/runtime`） | **交付 runtime** `custom_ops/__init__.py` | `worker-origin-rpc-B1-runsh.json`（pid 781058 / 781059） |
| 上表第二行再加 `PYTHONSAFEPATH=1` / `python -P` | 结论不变（`-P` 只影响 cwd/`''`，不影响 `PYTHONPATH` 顺序） | —— |

> **更正记录（2026-10-10，同一冻结窗口内）**：本条第一版矩阵由 `python -c` / `python -` 探针得出。
> 该口径**不代表服务行为**（`-c`/`-` 会在 `site` 处理之前就把 cwd 放进 `sys.path[0]`；
> 服务侧 `sitecustomize` 在 `site` 初始化阶段已导入 `custom_ops`，之后 runpy 再插 cwd 改不了绑定）。
> 现按 worker 内 RPC 结果就地更正：**结论方向不变**（现状⇒根级、`run.sh`⇒交付副本），
> 但**机制**是「启动器 `PYTHONPATH` 决定先命中哪个 overlay，且该导入发生在 `site` 阶段」。
> 另有进程内实证：用 py3.11 启动时 overlay 在 `site` 阶段即因缺 `tecoops` 而 fail-closed，
> 栈帧直接给出当时命中的是 `custom_ops/...` 还是 `runtime/custom_ops/...`
> （`op_learning/runtime/entrypoint-parameterization-internvl-20261010/site-init-resolution-traceback.txt`）。

**影响与建议**：本线因两份内容相同而**无行为差异**，但这是潜在维护风险——一旦两份分叉，
从仓库根启动会静默使用根级副本。需要把模型级证据严格绑定到提交副本时，请使用交付入口或上表
第二行的显式 `PYTHONPATH`；解析矩阵、逐字节一致性证据、worker 内收据与绑定后的模型级重跑见
`op_learning/runtime/custom-ops-resolution-internvl-20261010/` 与
`op_learning/runtime/entrypoint-parameterization-internvl-20261010/`。
**不要**删除根级副本（共享启动器依赖它），也**不要**让模型仓分支携带根级副本。

### 切换到官方 py3.11 环境（解释器可覆盖 + 能力校验，2026-10-10）

三个入口的解释器都可用 `PYTHON=<解释器>` 覆盖，**默认仍是厂商 `/home/py312/bin/python`**
（AGENTS.md 硬约束 1 的默认口径不变）；覆盖仅用于切换到官方环境：

```bash
# 官方环境（ModelZoo py3.11 / PyTorch 2.7.1）；<py3.11> 需自行准备的解释器路径
PYTHON=<py3.11>/bin/python \
    bash scripts/smoke_internvl.sh "${MODEL_ROOT}/OpenGVLab/InternVL3_5-8B" 32 2
PYTHON=<py3.11>/bin/python \
    bash model_adaptations/InternVL3_5SCUdoudui/run.sh --enforce-eager
```

服务入口另修一处**静默换解释器**的漏洞：`scripts/serve_internvl.sh` 原先把 `vllm serve` 交给
shell 按 `PATH` 解析（`PATH` 已前置 `${VENDOR_PY_ROOT}/bin`），因此 `PYTHON=<官方解释器>`
会通过能力校验、服务端却被 `vllm` 脚本的 shebang 换回厂商解释器。现改为
`"${PYTHON}" "${VLLM_CLI}" serve …`（`VLLM_CLI` 由 `command -v vllm` 一次性解析，
缺失即 rc=1），保证**被校验的解释器就是真正跑服务的解释器**。

守卫已从「`readlink -f` 必须等于 `/usr/local/python/bin/python3.12`」改为**能力校验**：
该解释器能 `import torch`、能 `import torch_sdaa`，且 `torch.sdaa.is_available()` 为真、
设备数 ≥ 1。因此官方 py3.11 环境**通过**该校验，而不具备 `torch_sdaa` 的解释器
（例如系统 `/usr/bin/python3`）被明确拒绝。**任何失败都非零退出，禁止静默降级到系统 python。**

实测（2026-10-10，三解释器，CPU 侧即可判定）：

| 解释器 | 能力校验 | 后续行为 |
| :--- | :--- | :--- |
| `/home/py312/bin/python`（厂商，默认） | PASS（torch 2.12.0a0+…、`is_available()=True`、设备 4 / 钉死后 2） | 正常冒烟与服务（本次实测 rc=0、32-token IDs 与冻结基线一致） |
| `/home/py311/bin/python`（官方口径，torch 2.7.1 + torch_sdaa） | PASS（`is_available()=True`、4 设备） | 本机**尚无** py3.11 的 `vllm`/`tecoops`，入口在**缺资产**处 fail-closed 退出（`Failed to import 'tecoops' for the RMSNorm overlay`，rc=1），**不是**静默降级 |
| `/usr/bin/python3`（系统，torch 2.10.0，无 torch_sdaa） | **FAIL → rc=1** | 不启动 |

补齐官方环境的步骤：`tecoops` 必须用该解释器**配对构建**后放到本目录之外的独立路径并前插
`PYTHONPATH`，并以 `tecoops.__file__` 自证来源（防 shadowing）；训练侧同法见 Deformable 线
`models/deformable-detr/docs/bootstrap-py311.md` §2。**本线尚未在 py3.11 上跑通模型**，
故不宣称官方环境适配完成；上表以外所有模型级数字仍来自厂商 py3.12 + Torch-SDAA。
