# MiniCPM5-1B

## 模型概述

MiniCPM5-1B 是面壁智能 MiniCPM5 系列的小尺寸稠密语言模型。本目录是 SCU 都队在太初（SDAA）平台上的推理适配，基于 TecovLLM（vLLM 0.20.2）与自研 `tecoops` 算子库：通过**进程初始化期的 overlay** 把算子绑定进模型，不修改任何 site-packages / torch / vLLM 文件，权重按只读方式引用。

难度系数：**0.5**。

已适配的算子（三层形态：C++ kernel → Python API → 模型调用点）：

| 算子 | 形态 | 说明 |
| --- | --- | --- |
| `rms_norm`（含 residual `+add`） | 复用团队 `tecoops` 算子，启动期绑定 | PR37 形态 |
| `reshape_and_cache` | 复用团队 `tecoops` 算子，启动期绑定 | PR35 形态 |
| `flash_attn_varlen_func` | 复用团队 `tecoops` 算子，启动期绑定 | 官方 paged flash ABI |
| `rotary_embedding` | **本队新增 C++ 融合 kernel**（NeoX / fp16，显式行 stride） | 启动期绑定 vLLM `RotaryEmbedding` |

可选档（**不作为交付主张**，仅作对照与定位）：`MINICPM_OP_PROFILE=fast_attention`（dense-KV + torch SDPA 拼接路径）、`MINICPM_SILU_MUL_PROFILE=vendor_swiglu`（厂商 `sdaa::swiglu`）。

精度口径：组委会测例与达标线尚未下发，本目录只声明**临时回归通过**（见「精度验证」），不声明适配完成。

## 模型权重下载

使用组委会**预置只读**权重，通过 `$MODEL_ROOT` 引用；**不重复下载、不转储副本**。

- `MODEL_ROOT` 默认 `/gpfs/model`；
- 本模型权重目录：`$MODEL_ROOT/OpenBMB/MiniCPM5-1B`；
- 如需指向其它只读挂载点，导出 `MODEL_ROOT=<路径>` 即可，启动命令与 `run.sh` 均使用该变量（已用 symlink 别名做覆盖验证，见下）。

```bash
export MODEL_ROOT=/gpfs/model          # 只读引用，无需下载
ls -ld "$MODEL_ROOT/OpenBMB/MiniCPM5-1B"
```

## 模型注册

本目录随推理仓一起交付，**不需要为本模型单独编译**：

1. 推理仓自身按仓库说明安装（`tecovllm_models/` 提供模型注册），例如 `python setup.py develop` 或 `python setup.py bdist_wheel`；
2. `tecoops` 算子库**不能**沿用组委会镜像预装的 wheel：本交付的 overlay 要求 `tecoops.rotary_embedding`，而预装 wheel（8 个导出）没有该导出 ⇒ 必须**先构建并安装本队算子库**（见下节「必需步骤」，含可复制的自证命令），并把构建树的 `api/` 目录置于 `PYTHONPATH` **首位**（`run.sh` 会把 `runtime/overlay`、`runtime` 前置，其余继承调用者环境）。

3. 算子绑定全部在进程初始化期完成（`runtime/overlay/sitecustomize.py`）：注册 opaque `torch.library` 算子并绑定 RMSNorm、BlockAttention、`RotaryEmbedding` 等；缺接口或绑定身份校验失败即 **fail-closed 终止启动**，不会静默回退到未适配路径。

### 必需步骤：先构建并安装本队 `tecoops` 算子库

> **这一步不是可选项。** 本交付的 `runtime/overlay/sitecustomize.py` 在进程初始化期要求
> `tecoops.rotary_embedding`（本队新增的融合 RoPE kernel）。组委会镜像预装的 wheel
> （`/home/py312/lib/python3.12/site-packages/tecoops`，`libteco_ops.so` sha256
> `388dd710a8357ee9294ccada17fa3f4003a9bc5aa107bec744184fa008cbbcc0`）**只有 8 个导出、没有
> `rotary_embedding`**，直接跑 `run.sh` 会被 overlay **刻意 fail-closed**（`rc=1`、`main`
> 不可达、stderr 打印 `FATAL: [custom_ops] tecoops.rotary_embedding is unavailable`）——
> 这是**设计行为**（缺接口宁可停进程也不静默回退到未适配路径），**不是崩溃**；
> 本交付包**不含** `tecoops`，必须先构建并安装本队算子库。

```bash
# 1) 取得本队算子仓库（另行提交的「自定义算子仓库」）源码
#    源码 commit: 8ef0a5f891603ddc8011cc3c45b127fe988024bf（队内 ops main，含 rotary_embedding kernel）
git clone <teco-ops> /tmp/teco-ops
cd /tmp/teco-ops && git checkout 8ef0a5f891603ddc8011cc3c45b127fe988024bf

# 2) 用厂商 Python 构建 torch-linked 交付产物
#    注意：不要用 bash build.sh —— 它是 WITH_TORCH=OFF 变体，且没有 set -e
set +u; source /opt/tecoai/setvars.sh; set -u
export PATH=/home/py312/bin:/opt/tecoai/bin:$PATH
export WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF
/home/py312/bin/python setup.py build_ext --inplace

# 3) 把构建树的 api/ 目录置于 PYTHONPATH 首位，再启动服务
export PYTHONPATH=/tmp/teco-ops/api:${PYTHONPATH:-}
```

> 本 pin 已前移至 **`8ef0a5f891603ddc8011cc3c45b127fe988024bf`**（自 `4adfd05e` 起依次落地：`79941c5f` ms_deform_attn forward plane → `d84f34f` flash_attention D256 `BM==64` → `54a1d65` rms_norm epilogue 二级展开 → `98e18ca` rotary_embedding CI 修复同步 → `e4e691d` flash_attention `BM==128` NT-form + 转置域 softmax → `beebae2` weight_stream_gemv host 谓词 `n % 256 == 0` → `n % 64 == 0`（按引擎 DMA 校验器实测的 4 B 要求），使 TP 分片的 lm_head 也走上 M=1 快路径 → `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写 → `8ef0a5f` weight_stream_gemv ILP4）。**照旧 pin（`4adfd05e`）复现得到的库不含上述改动**，故复现必须以 `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 为准。

**版本口径**：pin 写成「**源码 commit + 同配方重建**」（本交付自证的源码 tip 为 `8ef0a5f891603ddc8011cc3c45b127fe988024bf`），**不是**某个 `libteco_ops.so` 的字节
——同一个 commit 在不同 build root / 配方下会产出不同字节的 `.so`。自证请用**能力探针**
（不是裸 sha256）：

```bash
# 4) 自证：tecoops 位置 + lib sha256 + 能力探针
/home/py312/bin/python - <<'PY'
import hashlib, pathlib, tecoops
root = pathlib.Path(tecoops.__file__).resolve().parent
print("tecoops.__file__ =", pathlib.Path(tecoops.__file__).resolve())
print("lib sha256       =", hashlib.sha256((root / "libteco_ops.so").read_bytes()).hexdigest())
for name in ("rotary_embedding", "rms_norm", "reshape_and_cache", "flash_attn_varlen_func"):
    print("capability %-24s = %s" % (name, hasattr(tecoops, name)))
PY
```

期望：`tecoops.__file__` 指向 `<teco-ops>/api/tecoops/__init__.py`，四个能力全为 `True`。
本线 2026-10-12 的模型级 A/B/A 窗口就是在这一形态下跑的（lib sha256 `ab7754721d7cd64f…`，
该树是上述 commit 的祖先 `459992a` 加同一个 rotary kernel，行为等价）。

只设 `PYTHONPATH` 而不设 SDK 运行库路径时，`import tecoops` 会以
`ImportError: libsddev.so: cannot open shared object file` 失败（torch 自动加载 torch_sdaa 所致），
必须先 `source /opt/tecoai/setvars.sh` 或手工把 `/opt/tecoai/lib64` 与 torch/torch_sdaa 的 `lib`
目录加进 `LD_LIBRARY_PATH`。

## 启动推理服务

`run.sh` 是唯一启动入口，serve 参数由组委会 CI 固定，**不要改参**：

```bash
export MODEL_ROOT=/gpfs/model
export SDAA_VISIBLE_DEVICES=1
export MINICPM_MAX_MODEL_LEN=2048      # 本队验证口径；run.sh 默认 32768
b=model_adaptations/MiniCPM5SCUdoudui
bash "$b/run.sh"
```

- **前置条件**：本队 `tecoops` 已构建并置于 `PYTHONPATH` 首位（见上节「必需步骤」）；否则进程会在 overlay 处 fail-closed 以 `rc=1` 退出（这是设计行为，不是崩溃）；
- 固定项：TP1、FP16、关闭 prefix caching、保留 vLLM 默认编译（不强制 `--enforce-eager`）、`--trust-remote-code`、`--served-model-name MiniCPM5-1B`；
- 默认 host `0.0.0.0`、port `8000`；可用 `MINICPM_HOST` / `MINICPM_PORT` 覆盖（启动 shim 会把 host/port 替换为这两个变量的值）；
- `MODEL_ROOT` / `MINICPM5_MODEL` 决定模型路径（启动命令使用变量，已做覆盖验证）；
- 官方评测入口：`/home/py312/bin/python tools/ci_pipline/run_ci.py "$b/run.sh"`；
- **CI 解析器约束**：`tools/ci_pipline/run_ci.py:parse_run_sh()` 会剥掉注释行后取**第一个** `vllm serve <token>` 作为模型路径（并作为 `speed.sh` 的 `--tokenizer-path`）。因此 `run.sh` 里保留一个**不可达**的裸字面量块 `if false; then vllm serve /gpfs/model/...; fi`，真正执行的是 `vllm serve "$MODEL_PATH"`（本地 `vllm()` shim 在 exec 前把 argv[1] 改写成 `$MODEL_PATH`）。不要把那行改成带引号的变量，也不要把它挪到注释里。

## 精度验证

> **临时回归通过**：组委会测例与达标线未下发，本节**不给出任何精度分数**，也不声明适配完成。

| 项目 | 口径 | 结果 | 证据 |
| --- | --- | --- | --- |
| 固定 prompt 贪婪生成 | `Explain why reproducible benchmarks matter.`，`max_tokens=32`、`temperature=0`、`top_p=1`、`top_k=-1`、`seed=0` | 32/32 token id 与冻结基线**逐位一致** | `validation/baseline_token_ids.json`（基线）、`validation/runtime_20261004.json`、`validation/ci_contract_20261005.json` |
| 算子级数值 | 官方 `zoo/teco/<op>/test_case/*.prototxt`，DIFF1 | 各算子 focused 用例通过（详见 §性能验证的算子级行） | `validation/rms_shapes_20261008.json`、`validation/silu_mul_public_20261007.json` |
| 启动/绑定 fail-closed | 缺接口、身份校验失败、非 FP16 显式 dtype 等 | 9 例启动/档位用例通过，失败即终止启动 | `tests/test_startup_failure.py`、`tests/test_compile_safe_cpu.py` |

复现命令（在**推理仓根目录**执行，`tools/ci_pipline/` 位于该仓）：

```bash
b=model_adaptations/MiniCPM5SCUdoudui
source /opt/tecoai/setvars.sh
/home/py312/bin/python -S "$b/tests/test_compile_safe_cpu.py"
MINICPM_ROTARY_EMBEDDING=off /home/py312/bin/python -S "$b/tests/test_startup_failure.py"
/home/py312/bin/python -S "$b/tests/test_launcher_cpu.py" tools/ci_pipline/run_ci.py
SDAA_VISIBLE_DEVICES=1 /home/py312/bin/python "$b/tests/test_compile_safe_sdaa.py"
```

- 无 SDAA 的机器上，`test_startup_failure.py` 需要 `MINICPM_ROTARY_EMBEDDING=off`：该用例的正向对照要求
  「能走到入口」，而融合 RoPE overlay 在没有 SDAA（vLLM 无法完整导入）时会按设计 fail-closed 提前退出；
- `test_launcher_cpu.py` 的参数 `tools/ci_pipline/run_ci.py` 是推理仓内路径，必须在推理仓根目录运行。

**官方仓契约精度门禁（非组委会最终测试集）**：mmlu_pro / computer science、5-shot、limit 200，
`accuracy = 0.32 (64/200)`，落在官方仓接受区间 `[0.245, 0.445]`（官方仓 baseline 0.345）；
证据 `op_learning/runtime/minicpm-repo-precision-20261005/runs/minicpm-official-precision-001/`。
该数字**不是**组委会测试集分数，只用于证明评测链路可跑通。

**原件路径（本节每个数字都能追到原始件）**：
- 32-token 逐位基线：`validation/baseline_token_ids.json`（冻结基线）、`validation/runtime_20261004.json`、`validation/ci_contract_20261005.json`；
- 算子级 DIFF1 / `Average Hardware Time`：`validation/rms_shapes_20261008.json`、`validation/rms_epilogue_simd_20261008.json`、`validation/rms_epilogue_simd_model_gate_resolution_20261009.json`、`validation/sdpa_public_20261007.json`、`validation/silu_mul_public_20261007.json`；
- 仓内契约精度门禁（非组委会测试集）：`op_learning/runtime/minicpm-repo-precision-20261005/runs/minicpm-official-precision-001/`；
- 启动 / fail-closed：`tests/test_startup_failure.py`、`tests/test_compile_safe_cpu.py`；
- 2026-10-13 交付默认档精度（本队 1B 线）：`op_learning/runtime/minicpm-precision-c22/`（详见该目录 `RESULT-minicpm-mmlupro-c22.md`）。

> `models/minicpm5-1b/README.md` 是**仓内/上游模型说明**，不是交付件，故不设精度段；交付精度证据只在本节、`validation/`、`op_learning/` 三处。

**已知限制**：仅验证短请求（本队口径上下文 2048），未验证 32768 长上下文与长 chunk；未做官方精度评测与 owner CI；权重与编译产物不随本目录提供。

## 性能验证

口径：**模型级** = 同一 build、同一固定 prompt、贪婪 32 token，每臂独立进程 + 预热 + 3 次计时取中位；**算子级** = 官方测例 + `./build/demo --perf_repeat=50 --warm_repeat=3` 的 `Average Hardware Time`。

### 模型级（融合 `rotary_embedding`，窗口 2026-10-12T110857Z）

| 轮次 | A1（原生链）ms | B（融合）ms | A2 ms | 增益（vs A1） | 漂移 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2112.0 | 1204.0 | 2161.0 | +42.99% | 2.32% |
| 2 | 2446.9 | 1254.9 | 2125.7 | +48.71% | 13.13% |
| 3 | 2499.8 | 1106.5 | 2476.6 | +55.74% | 0.93% |

**中位增益 +48.71%，中位漂移 2.32% ⇒ STRONG_GO**（预注册 ≥+3% GO / ≥+8% 强 GO）。9 臂全部 32/32 token 与冻结基线一致。

**机制（必须与数字同报）**：收益来自 **host 侧 launch 消除** —— 每请求去掉 12 288 次发射（融合 RoPE 把 24 层 × 17 次原生发射降为 24 层 × 1 次，总调用数 26 936 → 14 648/请求），**不是 GPU 计算收益**（kernel 自身 device 时间约 0.0018–0.0039 ms/层）；**不可外推**到把 RoPE 图捕获（torch.compile/CUDA-graph）的栈。

### 算子级

| 算子 | 测例 | 结果 | 口径说明 |
| --- | --- | --- | --- |
| `rotary_embedding`（本队新增） | 自建 3 例（T=1 / T=7 / 行 stride 2560），DIFF1 | `max_error` 0 / 2.11e-08 / 0；AHT 0.0017626 / 0.003934 / 0.0017305 ms | 官方测例未下发 |
| `reshape_and_cache` | 官方 3 例，官方 `Average Hardware Time`，跨树 A1→B→A2 | case_0 0.13336→0.025380 ms（**−80.97%**，漂移 +0.090%）；case_1 0.068506→0.01965（**−71.32%**，−0.069%）；case_2 0.034868→0.01416（**−59.39%**，−0.126%）；三例中位之和 0.236734→0.059190 = **−75.00%**（+0.012%）；同 build lib-swap 复现 −80.86%/−71.27%/−59.30% | **仅算子级主张**：该算子只占 device 时间 **0.3374%**（时间加权，eager 归档口径、分母须声明），且 **96.875%** 的调用是 T=1（744/768）⇒ 模型级上界 0.3374%×74.91% = 0.253%，现实值 <0.01%；正确性 54/54 run PASSED、`max_error = 0`、阈值 1e-3 |
| `rms_norm`（PR37 形态） | 官方 8 例，官方 `Average Hardware Time`（本线候选 `op/rms-norm-ordered-widened-reduce`） | case_0 −5.59%、case_1 −46.56%、case_2 −5.54%、case_3 −45.32%、case_4 −4.80%、case_5 −44.24%、case_6 −1.11%、case_7 −2.94%；8 例中位之和 0.188823→0.128185 = **−32.11%**（漂移 −0.033%） | 自建 D1536 真实 shape 带残差 N=1/9/512/2048 为 −46.73%/−46.12%/−51.35%/−52.37% |
| `flash_attn_varlen_func` | 自建真实 shape（基线）与官方 paged 用例 | 基线中位 **1.651520 ms**；D128 SPE 分工轮 case_2 3.1900→2.4890→3.1901 = **−21.97%/−21.98%**、case_4 −22.92%，但 case_5 **+45.47% 开放负项**、case_0 +0.066% | 该算子 device 占比 **0.1357%**（24 次/1525.122 µs，eager 归档口径） |

**关于 `cur_BM`（口径更正）**：`BM == 32 || BM == 128 → BM == 64` 的放宽属于 **`flash_attention`（D128）** 的
scoping，不是 `rms_norm`。实测 case_4(D64) 中位 2.677600 ms、B64 两方向 −0.01%，case_2 对照 −0.02%/−0.01%，
与同臂 ABA 噪声同量级 ⇒ **该轮为 NO-GO**，不作为收益项。证据
`op_learning/minicpm5-1b/flash-d128-bm64-curbm-20261010/`。

### 端到端 / TTFT / TPOT（已测口径）

| 场景 | 口径 | 结果 |
| --- | --- | --- |
| 官方 Evalscope 契约形态（2K/4K 长 prefill，A/B/A2） | 服务 `run.sh` 原样 + `MINICPM_OP_PROFILE=official` + `MINICPM_SILU_MUL_PROFILE=vendor_swiglu`（**厂商算子档，不计入交付**）+ `MINICPM_MAX_MODEL_LEN=32768`；`--parallel 1`，2K/4K 各 100 次 | TTFT 2K 2623.67→1513.57→2642.74 ms（**−42.31%/−42.73%**，漂移 +0.73%，信噪 58×）；4K 7319.83→3978.67→7346.82 ms（**−45.65%/−45.85%**，漂移 +0.37%，信噪 123×） |
| 同上，解码侧 | 同一批原生 summary | TPOT 2K 65.11/73.26/75.7 ms、4K 64.45/73.96/75.72 ms；output 11.0249/11.4062/9.8634 tok/s（2K）、7.2991/8.8486/6.7370 tok/s（4K）；**账本明示解码侧不作主张**（跨会话 A vs A2 方差 −10.5%(2K)/−7.7%(4K)） |
| 固定 512-token prompt、贪婪 32 token（3 次） | 服务端 HTTP | TTFT 中位 **694.540 ms**、prefill 代理 **625.249 ms**、e2e **5820.262 ms**、输出 **10.9961 tok/s** |
| 并发稳态（内部口径） | p50 延迟 c=1/4/8 | 2362.40 / 2741.56 / 2736.26 ms；吞吐 0.423 / 1.449 / 2.909 req/s（≈13.5 / 46.4 / 93.1 tok/s） |

上表第一行的大额收益来自 **host 侧 launch 消除**（该档为厂商 SwiGLU 档，不计入交付主张），**不是 GPU 计算收益**，
且**不可外推**到图捕获栈；交付档（`official`）的模型级主张以上方融合 RoPE 的 A/B 为准。

## 已知限制

> **已知限制（2026-10-14，交付库 `e4e691d` 测得）**：`e4e691d` 把 `flash_attention` 的 `<128>` 实例化移入**尾部新 TU**，使整库 device 代码布局重排，**`rms_norm` 的 8 个官方 case 因 code-layout 效应变慢**（逐例 `+0.35…+2.6 µs`，合计 `+11.88 µs`；同构建 A/B，`max_error` 逐位不变、`rms_norm_fp16.scpp` 源码 sha256 未变）。该效应**与算子源码无关**，且**换任何结构都只会把税转移给另一个算子**（例如把新 TU 移进 `flash_attention/` 目录会让 `rms_norm` 归零但使 `weight_stream_gemv` 慢 `−35%/−18%`）。本队选择维持 `e4e691d` 的 device 布局（其后 `beebae2` 只改 host 谓词，全部 device 目标码逐字节不变），理由是净账：`flash_attention` 官方 11 例表 `−1.58367 ms`，而 `weight_stream_gemv` 正是 Hy/InternVL 两条线模型级收益所绑定的算子。逐 case 原值与完整 A/B 见本队账本仓 `docs/competition/c25-lead-ledger-20261014.md` §8.4 与 §9.4。

## 许可与出处

- 本队新增与修改的适配代码（`runtime/**`、`README.md`、测试与文档）以 **Apache-2.0** 提交；
- 本仓库根 `LICENSE` 为**上游继承的 BSD 3-Clause**（本目录 fork 自官方推理仓，未改动该许可证文件；`python_api_test` / `teco` 等官方路径亦沿用其 BSD-3 头）；
- 第三方代码出处与许可见各文件头部注释与仓库根 `LICENSE`；本队**不使用** GPL / LGPL / MPL 代码。

## 沿革

本目录此前为英文长文档，2026-10-12 按推理仓官方示例模板重写为中文七段结构；历史逐次实验、旧 SHA、废弃流程与更早的对照证据一律保留在 `experiments/minicpm5-1b.md` 与 `validation/*.json`，不再在 README 中复述。

---

**cycle-31 pin 说明（2026-10-16）**：本交付的 ops pin 由 `beebae220f0e918460d2210e4064030e157d602d` 前移至 **`8ef0a5f891603ddc8011cc3c45b127fe988024bf`**（本队 teco-ops main，含 cycle-31 落地的 `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写、`8ef0a5f` weight_stream_gemv ILP4、`f092863` flash_attention fail-closed 对、`973cbd2` weight_stream_gemv SPM fail-loud、`6d0bde7` 测试面 mock 健壮性）。组合树门禁：单次调用（`--perf_repeat=0 --warm_repeat=0`）`max_error=0`、`python_api_test` focused ALL PASSED、官方全 39 例 `PASSED=39 FAILED=0`（逐例 `max_error` 仅 `weight_stream_gemv` 两例按归约序变化）。旧 pin `beebae220f0e918460d2210e4064030e157d602d` 仍可复现旧库。
