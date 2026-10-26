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
#    当前源码 commit: d28ce49ed2febaac9cfb4d4d9cda661c2bfe150d（队内 ops main，含 rotary_embedding kernel）
git clone https://github.com/tecorigin-caet/teco-ops-SCUdoudui.git /tmp/teco-ops
cd /tmp/teco-ops && git checkout d28ce49ed2febaac9cfb4d4d9cda661c2bfe150d

# 2) 用厂商 Python 构建 torch-linked 交付产物
#    注意：不要用 bash build.sh —— 它是 WITH_TORCH=OFF 变体，且没有 set -e
set +u; source /opt/tecoai/setvars.sh; set -u
export PATH=/home/py312/bin:/opt/tecoai/bin:$PATH
export WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF
/home/py312/bin/python setup.py build_ext --inplace

# 3) 把构建树的 api/ 目录置于 PYTHONPATH 首位，再启动服务
export PYTHONPATH=/tmp/teco-ops/api:${PYTHONPATH:-}
```

> cycle-42 历史 pin 为 **`75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`**（自 `4adfd05e` 起依次落地：`79941c5f` ms_deform_attn forward plane → `d84f34f` flash_attention D256 `BM==64` → `54a1d65` rms_norm epilogue 二级展开 → `98e18ca` rotary_embedding CI 修复同步 → `e4e691d` flash_attention `BM==128` NT-form + 转置域 softmax → `beebae2` weight_stream_gemv host 谓词 `n % 256 == 0` → `n % 64 == 0`（按引擎 DMA 校验器实测的 4 B 要求），使 TP 分片的 lm_head 也走上 M=1 快路径 → `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写 → `8ef0a5f` weight_stream_gemv ILP4 → `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步（算子级 `case_2` 1.829900 → 1.804300 ms，+1.3990%；模型级无可测加速）→ `9f11b33`/`0132061`/`2a1716d`（cycle-36）→ `0d6bd5a`（flash P 直出）→ `7ea7e84`（掩码向量跳过 softmax）→ `37f89d1`/`542a2a1`/`785a784`（cycle-37）→ `3f98d68`/`f748d0a`（cycle-38）→ `9fda597` → `20c245a`（段序 pin）→ `79470cb`/`12271c1`/`fab1fa2`（cycle-39）→ `11cb85c`（已回退）/`60a40d7` → `ce1a2ec`（wsg/gather 段互换）→ `514f800`/`be4eb88`/`731d3d0`/`21dd896`（cycle-40/41）→ `5813b74`/`e6ec6e9`/`35bf8f6`/`75588a9`（cycle-42）。逐笔数字见文末 cycle-36…cycle-42 pin 说明）。**照旧 pin（`4adfd05e`）复现得到的库不含上述改动**，该历史窗口对应 `75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`；当前复现采用文末 2026-10-24 pin 与独立收据。

**版本口径**：pin 写成「**源码 commit + 同配方重建**」（当前源码 tip 为 `d28ce49ed2febaac9cfb4d4d9cda661c2bfe150d`），**不是**某个 `libteco_ops.so` 的字节
——同一个 commit 在不同 build root / 配方下可能产出不同字节的 `.so`。下列能力探针只证明接口存在；
当前启动还强制核验 core/extension 的已验证收据，跨根重建须按文末门禁独立验收。

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
- **CI 解析器约束**：`parse_run_sh()` 读取唯一可执行 serve 命令中的裸字面默认路径，作为 `speed.sh --tokenizer-path`。初始化 `vllm()` shim 在 exec 前绑定实际模型路径与 host/port，故环境覆盖仍生效。脚本仅保留这一条命令；官方评测仍须让默认字面 tokenizer 路径可解析。

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

> **已知限制（2026-10-17，pin `0371909` 复核补充）**：本 pin 的 `flash_attention` ALGO0 档在**算子入口一次性 fail-closed**：
> `findFlashAttentionBranch` 要求 `size_per_head % 32 == 0`（head_dim 128/64 满足）且 **`block_size == 32`**
> （`teco/ual/ops/flash_attention/find_flash_attention.cpp:78-82`；不满足即返回 `-1` ⇒ `Status::NOT_IMPLEMENTED`，
> **硬报错，不会静默算错**）。`block_size` 取自 KV cache 的 `dimA[2]`（`teco/interface/ops/flash_attention.cpp:104`）；
> 引擎默认 `DEFAULT_BLOCK_SIZE = 16`，但 SDAA `FLASH_ATTN` 后端启动时改判为 32（服务日志
> `Setting kv cache block size to 32 for FLASH_ATTN backend.`），而 `vllm_sdaa` 的 `block_attn.py` 又允许
> `32/128/256/512/768/1024` ⇒ **若显式传 `--block-size 128` 等非 32 值，本 pin 下 prefill 会直接失败**。
> 交付口径：**保持后端默认（32），不要显式覆盖 `--block-size`**。

> **算子绑定点（审计用）**：prefill 的 `flash_attn_varlen_func` 绑定点是
> `runtime/custom_ops/compile_safe.py:89`（`tecoops.flash_attn_varlen_func`）；
> `runtime/custom_ops/flash_attn_varlen/op.py` 中只有 `fast_attention`（dense-KV + torch SDPA）**替代档**，
> **不是**交付档 —— 只读该文件会误判绑定点（该审计盲点已在 `experiments/minicpm5-1b.md` 更正，此处补入交付文档）。

## 许可与出处

- 本队新增与修改的适配代码（`runtime/**`、`README.md`、测试与文档）以 **Apache-2.0** 提交；
- 本仓库根 `LICENSE` 为**上游继承的 BSD 3-Clause**（本目录 fork 自官方推理仓，未改动该许可证文件；`python_api_test` / `teco` 等官方路径亦沿用其 BSD-3 头）；
- 第三方代码出处与许可见各文件头部注释与仓库根 `LICENSE`；本队**不使用** GPL / LGPL / MPL 代码。

## 沿革

本目录此前为英文长文档，2026-10-12 按推理仓官方示例模板重写为中文七段结构；历史逐次实验、旧 SHA、废弃流程与更早的对照证据一律保留在 `experiments/minicpm5-1b.md` 与 `validation/*.json`，不再在 README 中复述。

---

**cycle-31 pin 说明（2026-10-16）**：本交付的 ops pin 由 `beebae220f0e918460d2210e4064030e157d602d` 前移至 **`8ef0a5f891603ddc8011cc3c45b127fe988024bf`**（本队 teco-ops main，含 cycle-31 落地的 `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写、`8ef0a5f` weight_stream_gemv ILP4、`f092863` flash_attention fail-closed 对、`973cbd2` weight_stream_gemv SPM fail-loud、`6d0bde7` 测试面 mock 健壮性）。组合树门禁：单次调用（`--perf_repeat=0 --warm_repeat=0`）`max_error=0`、`python_api_test` focused ALL PASSED、官方全 39 例 `PASSED=39 FAILED=0`（逐例 `max_error` 仅 `weight_stream_gemv` 两例按归约序变化）。旧 pin `beebae220f0e918460d2210e4064030e157d602d` 仍可复现旧库。
**cycle-32 pin 说明（2026-10-16）**：本交付的 ops pin 由 `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 前移至 **`0371909a87d5dd11bb928b487afbf6e1e25eb44c`**（本队 teco-ops main，新增 `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步：把 KV tile 内 V 面板的 strided 打包由逐面板同步 `memcpy_stride` 改为 `memcpy_async` 一板预取 + 双 pane ping-pong（两块 pane 落在既有 `ktmp` 内，SPM 占用不变），末板前视按 `MIN(hb + 1, npanel - 1)` 夹紧以免构造零长描述符，循环体保持无实现选择分支。算子级（官方命令形态 `./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3`，同一 build root、仅替换 `libteco_ops.so`）：`flash_attention case_2` 1.829900 → 1.804300 ms（**+1.3990% 加速**，即时间 −1.3990%；落地前复核 1.8300 → 1.8042 ms = +1.410%）；D128 家族 +1.21%/+1.30%/+1.41%、D64 +0.42%、`case_5` +2.48%，未改动 TU 的 D256/D512 家族内对照 −0.037%/+0.118%；11/11 `max_error` 与基线逐位相同（×3 rep），单次调用门禁（`--perf_repeat=0 --warm_repeat=0`）`case_2`/`case_0` = `1.83614e-04`/`1.78913e-04` 且与基线逐位相同。**模型级无可测加速**：MiniCPM5-1B 交付 `run.sh` 下 16 臂每请求 32/32 token 与冻结基线逐位一致、errors=0、无 CPU fallback/OOM、峰值显存与 KV cache 逐位相同，稳态 420.02 s / 344 请求 / 0 错误 / 344-344 逐位一致；墙钟**低于可测门**（pooled +0.464%，A1 自身臂间中位跨度 5.37%）⇒ 不主张模型级加速（亦不构成回退证据）。动态探针 `tecoops.flash_attn_varlen_func = 48`（24 层 × 2 请求）证明该 kernel 真在 MiniCPM prefill 路径执行。V-B（给 `M128_N32` 的 B 侧一个真实 stride、整行消费 V）= **NO-GO**：`M128_N32` 不 honor B stride。旧 pin `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 仍可复现旧库。 **本线可达性**：本线 `official` 档 prefill 走 `tecoops.flash_attn_varlen_func`（Hq16/Hkv2/D128，24 层）⇒ 经过 ALGO0（本次被改 TU）；模型级状态与动态调用计数见 `experiments/minicpm5-1b.md` 的 cycle-32 行。
**cycle-36 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0371909a87d5dd11bb928b487afbf6e1e25eb44c` 前移至 **`2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a`**（三个原子提交：`9f11b33` `weight_stream_gemv` 激活行 fp32 化（官方 case_0/case_1 **−4.55%/−8.28%**）、`0132061` `reshape_and_cache` 跨 token 载入流水线（**−22.4%/−27.5%/−23.2%**）、`2a1716d` `weight_stream_gemv` 尾部冗余 wait 删除（**−2.4%/−3.0%**）；全部为官方固定形态**同窗口同 build root 只换 `.so`** 的 A/B、3 次原值取中位数、`max_error` **逐位不变**、落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §4.2`）。
**cycle-36 round-5 pin 说明（2026-10-17）**：本交付的 ops pin 由 `2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a` 前移至 **`0d6bd5a10eddde4c020bb2367a233c72e005ad65`**（`perf(flash_attention): emit P straight from the softmax pass instead of transposing it` —— ALGO0/D128 每个面板的 `P^T→P` 转置融进 softmax 第二趟：寄存器直写 `p_score`、同一舍入；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：case_0/8/9/10 **−16.1%**、case_1 −9.7%、case_2/3 −7.5%、case_4 −8.5%、**十一例加权 −10.7%**；`max_error` **逐位不变**；**11/11 PASSED**；已知权衡：case_5 **+10.7%**（0.041353→0.045774 ms）。本线可达性：本线 `official` 档 prefill 走 `tecoops.flash_attn_varlen_func`（Hq16/Hkv2/D128，24 层）⇒ 经过 ALGO0（本次被改 TU）；模型级状态与动态调用计数见 `experiments/minicpm5-1b.md` 的 cycle-36 行。证据路径 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §6`。
**cycle-36 round-14 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0d6bd5a10eddde4c020bb2367a233c72e005ad65` 前移至 **`7ea7e84c8aef327d522d3eb318bc4808b15d070c`**（`7ea7e84` `perf(flash_attention): skip the softmax exp on causally-masked score vectors`；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：vs base `0371909` case_0/8/9/10 **−18.32%**、case_1 −9.50%、case_2/3 −6.05%/−6.08%、case_4 −6.85%；**vs 前一交付态 `0d6bd5a`** case_0/8/9/10 **−4.41%**、flash 算子再降 **−1.5%**；`max_error` **逐位不变**、**11/11 PASSED**；证据 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §13/§14`）。
**cycle-37 pin 说明（2026-10-18）**：本交付的 ops pin 由 `7ea7e84c8aef327d522d3eb318bc4808b15d070c` 前移至 **`785a7848169307ea531b52107881b36277737332`**（三个原子提交：`37f89d1` flash_attention D256/ALGO1 的 Q-block `parts` 分裂（官方 `case_6` 同窗 A-B **−14.79% / −14.81%**，两窗，`max_error` 逐位不变）；`542a2a1` reshape_and_cache 的 PUT 槽深两迭代（case_0 −1.33%/−0.89%、case_1 **−7.02%/−6.85%**、case_2 **−7.82%/−7.90%**，两窗，两臂 `max_error` 全 0）；`785a784` 构建期把 weight_stream_gemv 设备段排在 rotary_embedding 之前（官方 case_0 **−37.0%**、case_1 **−25.4%**，i-cache miss 14/3776 ns → 4/38 ns）。落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c37/lead/`）。
**cycle-38 pin 说明（2026-10-18）**：本交付的 ops pin 由 `785a7848169307ea531b52107881b36277737332` 前移至 **`f748d0abc93ab813f6e35feed261795ba850bd3b`**（两个原子提交：`3f98d68` flash_attention ALGO0/D128 的 Q-block 改用入口期 LPT 表分配（两窗复现：`case_2` −30.30%、`case_3` −30.29%、`case_4` −30.82%，flash 集合加权 **−13.6%**；如实披露 `case_0/8/9/10` +1.6%、`case_1` +1.74%、`case_5` +7.35% 的权衡）；`f748d0a` reshape_and_cache 暂存环 8 组（`case_0` −1.8%、`case_1` −5.7%、`case_2` −4.8%，两臂 `max_error` 全 0）。落地库 **39/39 PASSED**。已知待修：`rotary_embedding` 自 `785a784` 起处于慢地址（+133%，0.0017→0.0040 ms），由段序守卫处理。
**cycle-38 round-2 pin 说明（2026-10-18）**：本交付的 ops pin 由 `f748d0abc93ab813f6e35feed261795ba850bd3b` 前移至 **`9fda597f7d8bc5f98893081372a552f411f26e42`**（`9fda597`：flash_attention D256/ALGO1 调度单元 `UQ = BM/2 = 32` 的切分修正；官方固定形态同窗同 build root 只换 `.so`：**`case_6` 0.183760 → 0.116800 ms（−36.44%）**、`case_7` 0.151690 → 0.149830（−1.23%），其余 9 例 −0.1~−0.3% 无回退；`max_error` 逐位不变，**11/11 PASSED、39/39 PASSED**）。
**cycle-38 round-10 pin 说明（2026-10-18）**：本交付的 ops pin 由 `9fda597f7d8bc5f98893081372a552f411f26e42` 前移至 **`20c245a39bbf35fa4dee0c0dbcd3313248490bbf`**（`20c245a` `perf(build): pin the hot device sections into their measured fast addresses`：在 TAX-1 基础上用 5 步纯 list 重排把三个热算子的 `.aitext` 精确钉进实测快类——`weight_stream_gemv` **0x25a08**、`rotary_embedding` **0x28858**、`zzz_flash_attention_d128` **0x2a878**；官方固定形态**同窗同 root 从源码构建**的双臂 A/B、3 次中位数：rotary `case_0/1/2` **0.003998→0.001686 / 0.006323→0.003904 / 0.003994→0.001729（−57.8% / −38.3% / −56.7%）**、wsg `case_0/1` **0.005722→0.003977 / 0.007753→0.007686（−30.5% / −0.9%）**、flash D128 `case_0` **0.828940→0.809140（−2.39%）**、rms_norm `case_6` −0.54%；8 例 SUM `0.87075→0.841684（−3.34%）`；**39/39 PASSED**、`max_error` 不变。原始件 `/root/tmpwork/c38/ops/out-r10/`、`RESULT-R10.md`）。
**cycle-39 pin 说明（2026-10-18）**：本交付的 ops pin 由 `20c245a39bbf35fa4dee0c0dbcd3313248490bbf` 前移至 **`12271c185e3042392d8c9206ceb7535b20a53f50`**（`79470cb` flash D256/D512 的 NT PV 面板异步双 pane：`case_6` **−4.13%/−4.19%**、`case_7` **−4.31%/−4.01%**、`case_5` −5.01%/−4.99%，其余 8 例 −0.2~−0.4% 无回退；`12271c1` weight_stream_gemv 驻留 slab 分裂：`case_0` −1.14%/−2.40%、`case_1` **−11.54%/−12.22%**；两提交均 39/39 PASSED）。已知代价：两内核体积变化使其后锚点平移，wsg `case_0` 从 pin 峰值 0.00396 退到 0.00570（仍比 cycle-37 起点快 17%）；round-12 重推臂因拉回 pre-wsg 微秒算子的旧慢地址而 SUM +0.174%，已 REJECT。
**cycle-39 round-13 pin 说明（2026-10-18）**：本交付的 ops pin 由 `12271c185e3042392d8c9206ceb7535b20a53f50` 前移至 **`11cb85cbdc02470dde0a50d9777bb1e386501b9c`**（两个原子提交：`fab1fa2` flash_attention 的**首 tile rescale 跳过**（每个 Q block 第一个 KV tile 上 `scale = exp(-inf) = 0`，把 0 写成 0 ⇒ 逐位不变；同窗两窗：`case_0/8/9/10` **−1.60%/−1.61%**、`case_2/3` **−1.18%/−1.17%**、`case_5` **−2.31%/−2.28%**、`case_4` −0.89%、`case_1` −0.90%；flash 集合 **−1.20%**；`max_error` 11 例逐位不变、39/39 PASSED）；`11cb85c` D512 **tail** PV 循环的异步双 pane（`79470cb` 只覆盖主循环，tail 占每头 48 个面板中的 32 个；同窗两窗：`case_7` **−3.99%/−3.91%**，`case_6` 无 tail ⇒ +0.12%；`max_error` 逐位不变、39/39 PASSED））。原始件 `/root/tmpwork/c39/`、台账 `docs/competition/c37-lead-ledger-20261018.md`。
**cycle-39 round-15 pin 说明（2026-10-18）**：本交付的 ops pin 由 `11cb85cbdc02470dde0a50d9777bb1e386501b9c` 前移至 **`60a40d756f85fa6c1ac9d4cb3563a5f3ffd9c7db`**——**回退** `11cb85c`（`perf(flash_attention): overlap the D512 tail V-panel staging with its PV GEMMs`）。回退依据（AGENTS 规则 7，稳定负优化）：该提交的**自身 11 例窗口 SUM = +0.153% / +0.151%（两窗，净负）**——目标例 `case_7` 确实 −3.990%/−3.920%，但其余 10 例一致回退 +0.19~0.53%（LTO 重链把 D128 段推离原位），故按逐例/求和双口径均不成立。保留 `fab1fa2`（flash_attention 首 tile rescale 跳过：D128 9 例 −0.89~−2.31%，11 例 SUM **−1.21%**，`max_error` 逐位不变）。
**cycle-39 round-17 pin 说明（2026-10-18）**：本交付的 ops pin 由 `60a40d756f85fa6c1ac9d4cb3563a5f3ffd9c7db` 前移至 **`75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`**（`ce1a2ec` `perf(build): swap the weight_stream_gemv and gather device sections into their measured fast slots`：在纯 TU 顺序上把 `weight_stream_gemv` 与 `zz_ms_deform_attn_backward_ha`(gather) **互换**——wsg 落到实测快地址 `0x23dd4`，而 `flash_d128`/`rms_norm_fp16`/`msda_fwd`/`msda_bwd_fp32`/`memset`/`rotary`/`flatten_rays` **全部保持原址**（这是 K3 做不到的）；官方固定形态**同窗同 root 从源码构建**双臂 A/B：wsg `case_0` 0.005718→**0.003871（−32.30%）**、`case_1` −2.03%、`ms_deform_attn_backward` **−1.15%**、`ms_deform_attn_forward` **−3.66%**、`reshape_and_cache` −0.36%/−0.58%、`memset` ±0.3%、`rotary` ±0.6%；39 例 SUM 9.414836→9.412432（−0.026%）、无 >±0.8% 回退（首窗 1-rep 的 µs 级“越带”已由 3-rep 复核证否）。**说明**：`wsg 快 + gather 原址`经结构性证明不可同时满足（wsg 快地址均低于其 base，而 gather 是其直接前驱、TU 不可拆）⇒ 本折中为 13 臂枚举最优。

**cycle-42 pin 说明（2026-10-19）**：本交付的 ops pin 由 `ce1a2ec00270bb5a519e235166427c255ef2af40` 前移至 **`75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`**（团队 ops main，已合入 main），含四笔：`5813b74` flash D128 `P^T->P` 16-lane 向量化（官方 39 例 SUM -9.30%，两独立窗）；`e6ec6e9` rms_norm decode 热路径 C++ `TORCH_LIBRARY` `out=` 入口（op 级 75.61->26.28 µs，逐位全等、零布局税）；`35bf8f6` 新增 `rms_norm_add_out`（带 residual，76.38->25.08 µs，39 例净零）；`75588a9` flash D128 K cache 异步双缓冲预取 + Step B/E 寄存器融合 + 全可见 tile 掩码跳过 + softmax 栈溢出消除（**长 shape prefill 2K/4K/8K/16K -14.9%~-15.5%，官方 39 例 SUM 7.5999->6.7810 ms = -10.78%**，16/16 长 shape 探针 PASSED、`max_error` 不变）。**本线可达性**：MiniCPM 的 prefill 走团队 `tecoops.flash_attn_varlen_func`（D128/ALGO0）=> 上述 flash 改动在本线真实 forward 生效。**overlay 同步切换**：本线 rms_norm / rms_norm_add 已从 Python `torch.library.custom_op` 切到 `torch.ops.tecoops_ops.rms_norm_out` / `rms_norm_add_out`（进程初始化期一次绑定）；证据：交付 run.sh 原样参数、32-token 贪婪 Prompt1/Prompt2 × 3 轮臂序交替 **32/32 token 逐位相同**，TPOT 36.360 -> 32.766 ms/step（**+9.88%**；3 次原值 base `35.848 / 36.379 / 36.360`、cand `32.766 / 32.037 / 32.784` ms，取中位；同臂 spread base **0.531** / cand **0.747** ms，|Δ| = **3.594** ms ≈ 6.8× base spread ⇒ **GO**；原件现已随包保存在 `validation/c42-rms-out-summary.json`，来源路径与 SHA256 见同目录 `.provenance.json`（历史归档，不冒充本轮重测）），零 traceback/OOM/CPU fallback。**哈希表口径**：本文件上方 `libteco_ops.so` / `_torch_ext` 的 sha256 仍是**旧 pin** 的构建产物，必须按本 pin 从源码重建后重新记录（本轮未重建，不得引用旧值作交付证据）。


## 2026-10-24 复现与库身份门禁

当前源码固定为本队 teco-ops
`d28ce49ed2febaac9cfb4d4d9cda661c2bfe150d`，与本包
`validation/ops-artifacts.json` 的 `source_commit` 一致。前面的 cycle-42 及更早
SHA/收益是历史记录，不能充当本轮安装证据。

`run.sh` 在进程初始化期核验该收据中的 **core 与 host extension 两份 SHA256**，
并核对本模型全部必需 API（含非 RMS 路径）与唯一 mapped core。同名导出不足以证明
实现版本；旧 wheel、混装库或多份 mapped core 会在启动前非零退出。
普通 forward 不做哈希、环境查询或额外实现选择。

这份固定收据仅适用于当时已验证的那一对 core/extension 字节。相同源码在其它构建根
重建后可能生成不同字节：必须独立核对源码 pin，完成对应算子的 focused 正确性测试、
本模型固定 prompt 的 32-token 贪婪门禁，以及 attention/内存等改动所需的 5–10 分钟
稳态验证，保存并审阅新产物哈希与原始日志，再以
`SCUD_OPS_MANIFEST=<已审阅的新收据.json>` 显式绑定。仅将任意 wheel 的新哈希填进
收据不会令其成为已验证产物；不得自动信任或绕过哈希不匹配。

本包的实际证据文件：

- `validation/ops-artifacts.json`：该次构建的源码 pin、core/extension 哈希与构建根；
- `validation/decode-shapes-20261024.json`：本模型独立的真实 decode shape 盘点；
- `tests/test_ops_provenance.py`：身份门禁的 focused CPU 单测；
- `tests/check_ops_capabilities.py`：对当前安装运行同一身份与必需 API 门禁。

在已加载厂商环境并前置已验证 ops 安装目录的 shell 中运行：

```bash
export SCUD_OPS_MANIFEST="$ADAPT/validation/ops-artifacts.json"  # 仅适用于该收据对应产物
/home/py312/bin/python "$ADAPT/tests/check_ops_capabilities.py"
```

其中 `ADAPT` 为本模型目录的绝对路径；跨根重建使用独立验收后形成的新收据路径。
历史 cycle-42 归档仍按原日期保留，不替代本轮门禁或完整 CI。

真实 decode shape 的可选诊断：启动前设 `SCUD_DECODE_CENSUS=<空输出目录>`，启动
脚本会仅在此诊断档添加 `--enforce-eager`，然后跑固定 prompt 的 32-token 贪婪生成。
它仅记录真实 SDAA M=1 张量的 shape/stride，不记录权重值；输出仅作 shape 证据，
不能作性能成绩。正常验证与官方 CI 不设置此变量。


### 2026-10-24 五分钟模型稳态结果（2026-10-25 归档）

使用本节固定的 ops pin db1597e（**历史记录**：该测量发生在 2026-10-24，当时 pin 为 db1597e；当前 pin 见文末前移记录）及 validation/ops-artifacts.json 对应的
core/extension，按本模型原样启动参数运行固定 prompt、greedy-32：
**273 请求 / 300.953 s，全部 32 tokens 且逐位一致，0 请求错误，外层 rc=0**。
所有请求也与随包冻结 token IDs 一致。
原件随包在 validation/steady-20261024/，包含每次 token IDs、结果、启动配置及
受保护官方脚本前后相同的 SHA256。该窗口为现有服务器上的隔离安装验证，
不是干净容器或完整官方 CI，不新增性能提升主张。

## 2026-10-25 单条启动命令复核

本模型专属与团队入口各保留一条可执行 serve 命令。默认路径以裸字面量供官方
解析器读取，初始化 shim 在 exec 前绑定实际权重路径、host/port；服务参数未改。
本模型独立收据见 `validation/launcher-single-serve-20261025.json`：2 个入口、
18 个场景使用真实 Bash 执行基线和修订脚本，完整 argv、退出码及选定环境逐字节
一致，原版 `parse_run_sh()` 的解析字典也一致。收据含各场景原始 argv、脚本和
解析器 SHA256；整个测试将 Python 能力探针与最终 exec 换为桩，不导入设备栈。

本模型专属的新单条命令入口另完成真实 SDAA 固定 prompt 贪婪短生成：2 次请求各
32 tokens，与本模型 2026-10-24 冻结输出逐 token 相同；runner rc=0，受保护
CI 脚本前后哈希相同。原始 result/model-gate/run-configuration/protected 收据、
冻结 row0 比对和 runtime/core 身份见 `validation/launcher-single-serve-20261025/`。
（**历史记录**：该次 2026-10-25 复核时的 pin）实际使用 ops 源码 pin 为 `db1597e`，core SHA256 `531e0caf…77574dc`，
extension `82aa5455…fbbf41`；完整哈希及启动日志原文见收据。

团队共享入口仅做上述无设备 argv 等价验证，未另起设备服务。本次不是干净容器
复现或完整官方 CI，未重跑精度套件或性能测试，不作速度收益主张。

> **pin 前移记录（2026-10-26）**：上一版 pin 为 db1597e1265a5fc3e24207b793c9d1dd94327a24（2026-10-24），本次前移至 d28ce49ed2febaac9cfb4d4d9cda661c2bfe150d（队内 ops main = cycle-55 收口 + 三笔工具/CI 修复；区间 db1597e..d28ce49 共 19 个提交）。与本线相关的是 cycle-55 六笔算子改动（D128 双 pane、多项式 exp、D256/D512 exp、PV 双缓冲、四阶 Chebyshev exp、内联 32×32 butterfly）与三笔工具/CI 修复（get_op.py 语法、自测进 CI、cwd 解析）。本线用新 bundle 跑**官方两阶段 CI**（官方 run_ci.py + 官方 run.sh/prec.sh/speed.sh，200 样本 mmlu_pro）：精度 actual=0.3450 baseline=0.3450 range=[0.2450, 0.445] PASS；TTFT T1–T4 = 396.63 / 1031.91 / 3160.21 / 10838.54 ms，TPOT 32.94 / 33.12 / 33.37 / 34.36 ms。算子级证据：61 例全表 61/61 PASSED（AHT SUM 5.625283 ms），cycle-55 累计 58 例子集 −28.10%。已验证 bundle：libteco_ops.so sha256 = d43a93e7e42b355643153b425986f660ee4821f0ef5adb5fe946c0f44955b013，_torch_ext*.so sha256 = 79ccee58dcaeeb8c5a7e52816da3d39561bcdc8e9d3ce327e099a3b994ea8681（见 validation/ops-artifacts.json 与 validation/ops-pin-d28ce49-20261026.json）。**上一版报文的数字属历史记录，不能充当本次安装证据。**
