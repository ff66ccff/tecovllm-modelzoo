# Hy-MT2-1.8B

## 模型概述

Hy-MT2-1.8B 是腾讯混元系列的稠密翻译语言模型（`HunYuanDenseV1ForCausalLM`）。本目录是 SCU 都队在太初（SDAA）平台上的推理适配，基于 TecovLLM（vLLM 0.20.2）与自研 `tecoops` 算子库：通过**进程初始化期的 overlay** 把算子绑定进模型，不修改任何 site-packages / torch / vLLM 文件，权重按只读方式引用。

难度系数：**0.5**。

已适配的算子（三层形态：C++ kernel → Python API → 模型调用点）：

| 算子 | 形态 | 说明 |
| --- | --- | --- |
| `rms_norm`（含 residual `+add`） | 复用团队 `tecoops` 算子，启动期绑定 | PR37 形态；`+add` 精度余量按 A 档冻结（见「性能验证」） |
| `rotary_embedding` | **本队新增 C++ 融合 kernel**（NeoX / fp16） | 启动期绑定 vLLM `RotaryEmbedding`（`get_rope` 与 `__init__` 两条缝） |
| `linear_stream`（= `weight_stream_gemv`，M=1 decode 权重流式 GEMV） | 复用团队 `tecoops` 算子（上游 PR44），启动期绑定 | 绑定 `vllm.model_executor.layers.utils.default_unquantized_gemm`；M==1 走整行 slab 双缓冲 DMA，M>1 由 C++ host 回落厂商 `torch::linear`；默认开启，`HY_MT2_LINEAR_STREAM=0` 为 A/B 基线臂 |

交付运行态：**FP16、TP1**（`run.sh` 固定 `--dtype float16`）。历史 Transformers/BF16 形态的对照结论属于另一条路径的独立证据，**不构成本交付的结论**。

精度口径：组委会测例与达标线尚未下发，本目录只声明**临时回归通过**（见「精度验证」），不声明适配完成。

## 模型权重下载

使用组委会**预置只读**权重，通过 `$MODEL_ROOT` 引用；**不重复下载、不转储副本**。

- `MODEL_ROOT` 默认 `/gpfs/model`；
- 本模型权重目录：`$MODEL_ROOT/Tencent-Hunyuan/Hy-MT2-1.8B`；
- 如需指向其它只读挂载点，导出 `MODEL_ROOT=<路径>` 即可，启动命令与 `run.sh` 均使用该变量（已用 symlink 别名做覆盖验证）。

```bash
export MODEL_ROOT=/gpfs/model          # 只读引用，无需下载
ls -ld "$MODEL_ROOT/Tencent-Hunyuan/Hy-MT2-1.8B"
```

## 模型注册

本目录随推理仓一起交付，**不需要为本模型单独编译**：

1. 推理仓自身按仓库说明安装（`tecovllm_models/` 提供模型注册），例如 `python setup.py develop` 或 `python setup.py bdist_wheel`；
2. **必须先构建并「更新安装」本队 `tecoops` 算子库，再启动服务**（交付包**不自带** `tecoops`）。

   本模型的 overlay 依赖本队算子库的四个能力：`rms_norm`（含 `+add`；cycle-42 起经 C++ `TORCH_LIBRARY` 的 `torch.ops.tecoops_ops.rms_norm_out` / `rms_norm_add_out` 入口绑定）、`rotary_embedding`、`reshape_and_cache`、`linear_stream`（= `weight_stream_gemv`，默认开启）；SwiGLU 走厂商 `torch.ops.sdaa.swiglu`。镜像**预装**的 wheel（`/home/py312/lib/python3.12/site-packages/tecoops`，其 `libteco_ops.so` sha256 = `388dd710a8357ee9294ccada17fa3f4003a9bc5aa107bec744184fa008cbbcc0`）**不含 `rotary_embedding`**；而本线 overlay 在 cycle-42 之后还硬要求 `torch.ops.tecoops_ops.rms_norm_out` / `rms_norm_add_out`（`runtime/custom_ops/hy_mt2/vllm_rms_norm.py:31-39`，缺失即 `RuntimeError`）⇒ 直接 `bash run.sh` 会被 overlay 的 fail-closed 拦住并以 `SystemExit(1)` 退出（**这是刻意行为，不是崩溃**——宁可拒绝启动，也不静默回退到未适配路径）。本线代码**不使用** `rms_norm_compat`。

   服务库必须来自**本队 ops 仓**（当前源 commit `db1597e1265a5fc3e24207b793c9d1dd94327a24`；历史上自 `4adfd05e` 起已依次落地 `79941c5f` / `d84f34f` / `54a1d65` / `98e18ca` / `e4e691d`，见下方 provenance 句）：

   ```bash
   source /opt/tecoai/setvars.sh            # 提供 libsdaart.so / libsddev.so 等运行库路径
   TEAM_OPS=<本队 teco-ops 检出>             # 内容 = commit db1597e1265a5fc3e24207b793c9d1dd94327a24
   OPS_INSTALL=<隔离安装目录>               # 任意可写目录（交付包不自带），下面自检用它
   git -C "$TEAM_OPS" checkout db1597e1265a5fc3e24207b793c9d1dd94327a24
   bash "$TEAM_OPS/build.sh" --build teco   # 构建 core：build/teco/lib/libteco_ops.so
   cd "$TEAM_OPS" && /home/py312/bin/python setup.py bdist_wheel
   /home/py312/bin/python -m pip install --no-deps --no-compile \
       --target "$OPS_INSTALL" "$TEAM_OPS"/dist/*.whl
   export PYTHONPATH="$OPS_INSTALL:${PYTHONPATH:-}"
   ```

   > cycle-42 历史 pin 为 **`75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`**（自 `4adfd05e` 起依次落地：`79941c5f` ms_deform_attn forward plane → `d84f34f` flash_attention D256 `BM==64` → `54a1d65` rms_norm epilogue 二级展开 → `98e18ca` rotary_embedding CI 修复同步 → `e4e691d` flash_attention `BM==128` NT-form + 转置域 softmax → `beebae2` weight_stream_gemv host 谓词 `n % 256 == 0` → `n % 64 == 0`（按引擎 DMA 校验器实测的 4 B 要求），使 TP 分片的 lm_head 也走上 M=1 快路径 → `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写 → `8ef0a5f` weight_stream_gemv ILP4 → `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步（算子级 `case_2` 1.829900 → 1.804300 ms，+1.3990%；模型级无可测加速）→ `9f11b33`/`0132061`/`2a1716d`（cycle-36：wsg fp32 激活行、reshape_and_cache 跨 token 流水、wsg 尾部 wait 删除）→ `0d6bd5a`（flash P 直出）→ `7ea7e84`（掩码向量跳过 softmax）→ `37f89d1`/`542a2a1`/`785a784`（cycle-37）→ `3f98d68`/`f748d0a`（cycle-38）→ `9fda597` → `20c245a`（段序 pin）→ `79470cb`/`12271c1`/`fab1fa2`（cycle-39）→ `11cb85c`（已回退）/`60a40d7` → `ce1a2ec`（wsg/gather 段互换）→ `514f800`/`be4eb88`/`731d3d0`/`21dd896`（cycle-40/41）→ `5813b74`/`e6ec6e9`/`35bf8f6`/`75588a9`（cycle-42）。逐笔数字见文末 cycle-36…cycle-42 pin 说明）。**照旧 pin（`4adfd05e`）复现得到的库不含上述改动**，该历史窗口对应 `75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`；当前复现采用文末 2026-10-24 pin 与独立收据。

   本线历史实测自建库身份（跨构建根 sha 可能不同；当前启动须核验文末已验证收据，接口能力探针不能代替验收）：

   | 项 | sha256 |
   | --- | --- |
   | `libteco_ops.so` | `42422972ee7f9e42b2c62595a680193db49778ec33f53280351fa205c77cb5ca` |
   | `_torch_ext.cpython-312-loongarch64-linux-gnu.so` | `0fbf071ae5d433d5d74bf305a610d41e484a976f14f3812857a7c142c771c377` |

   **自证命令**（启动前执行；打印包路径、库 sha256 与必需能力项）：

   ```bash
   # 包内脚本；校验本轮映射 core/extension 哈希与全部实际绑定能力。
   /home/py312/bin/python model_adaptations/HyMT2SCUdoudui/tests/check_ops_capabilities.py
   # 成功打印 SCUD_OPS_PROVENANCE；缺能力、旧 wheel、混合 core/extension 均非零退出。
   ```

   注意：只设 `PYTHONPATH` 而不设 SDK 运行库路径时，`import tecoops` 会以 `ImportError: libsdaart.so: cannot open shared object file` 失败（torch 自动加载 torch_sdaa 所致），必须先 `source /opt/tecoai/setvars.sh` 或手工把 `/opt/tecoai/lib64` 与 torch/torch_sdaa 的 `lib` 目录加进 `LD_LIBRARY_PATH`。

   **官方 PR #48 的 wheel 不可单独充当本模型的服务库**：该 PR 从上游 `main`（`de27305e`）发出，只含上游那版 `rms_norm` 与本队 rope 算子（实测其 `libteco_ops.so` sha256 = `d2c8e046147df63b0ae142e176fe1ac151537761f2eb968f3de00b60e1337424`，能力探针判 `NOT_CAPABLE`；本线判据是 `tecoops.rms_norm` / `tecoops.rotary_embedding` / `tecoops.reshape_and_cache` 与 cycle-42 的 `torch.ops.tecoops_ops.rms_norm_out` / `rms_norm_add_out` 入口，**不使用** `rms_norm_compat`）；本模型需要的 `rms_norm`（PR37 / H3a 形态）只存在于本队 ops 仓，必须按上面的 build + install 步骤取自本队 `main`（**当前 pin `db1597e1265a5fc3e24207b793c9d1dd94327a24`**；`0371909` 及其之前的 `8ef0a5f` / `dda33c5b` / `4adfd05e` / `beebae2` 均已被前移取代，见上方 provenance 句与文末 cycle-31…cycle-42 pin 说明）。

3. 算子绑定全部在进程初始化期完成（`runtime/overlay/sitecustomize.py`）：注册 opaque `torch.library` 算子并绑定 RMSNorm、SwiGLU、attention、`RotaryEmbedding` 与 M=1 decode 线性层（`tecoops.linear_stream`）；缺接口或绑定身份校验失败即 **fail-closed 终止启动**（`SystemExit(1)`），不会静默回退到未适配路径；
4. **CI 解析器约束**：`parse_run_sh()` 读取唯一可执行 serve 命令中的裸字面默认路径，作为 `speed.sh --tokenizer-path`。初始化 `vllm()` shim 在 exec 前绑定实际模型路径与 host/port，故环境覆盖仍生效。脚本仅保留这一条命令；官方评测仍须让默认字面 tokenizer 路径可解析。

## 启动推理服务

`run.sh` 是唯一启动入口，serve 参数由组委会 CI 固定，**不要改参**：

```bash
export MODEL_ROOT=/gpfs/model
export SDAA_VISIBLE_DEVICES=0
b=model_adaptations/HyMT2SCUdoudui
bash "$b/run.sh"
```

- 固定项：`--served-model-name Hy-MT2-1.8B`、TP1、`--dtype float16`、`--gpu-memory-utilization 0.90`、`--max-model-len 8192`、`--trust-remote-code`、`--no-enable-prefix-caching`、`--no-enable-chunked-prefill`；
- 默认 host `0.0.0.0`、port `8001`；可用 `HY_MT2_HOST` / `HY_MT2_PORT` 覆盖（启动 shim 会把 host/port 替换为这两个变量的值）；
- 其它可用覆盖：`HY_MT2_TP_SIZE`、`HY_MT2_UTIL`、`HY_MT2_MAX_MODEL_LEN`、`HY_MT2_MODEL`（模型路径）、`MODEL_ROOT`；`--dtype float16` 与两个 `--no-…` 为硬编码，无环境变量可改；
- 官方评测入口：`/home/py312/bin/python tools/ci_pipline/run_ci.py "$b/run.sh"`。

## 精度验证

> **临时回归通过**：组委会测例与达标线未下发，本节**不给出组委会口径的精度分数**，也不声明适配完成。

| 项目 | 口径 | 结果 | 证据 |
| --- | --- | --- | --- |
| 固定 prompt 贪婪生成 | `Explain why reproducible benchmarks matter.`，**chat 模板套 user 消息**（`/v1/chat/completions`，或 `/v1/completions` + `apply_chat_template(..., add_generation_prompt=True)`），`max_tokens=32`、`temperature=0`、`top_p=1`、`top_k=-1`、`seed=0` | 32/32 token id 与冻结基线**逐位一致** | 基线 `validation/runtime_20261004.json → reference.greedy32_ids`；`reference.export_greedy32_match=true`、`each_request_all32_ids_match=true`、`errors=0` |
| 并发/稳态 | 同 build、并发 4/8、303.19 s / 313 请求 / 73 轮 | 全部 32 id 一致、0 错误 | `validation/runtime_20261004.json` |
| 官方**仓契约**精度门禁（**非组委会最终测试集**） | WMT24++ `en-zh_cn` 前 200 条 + `evalscope/wmt22-comet-da` | `mean_comet` **0.8495**（baseline 0.8486），落官方接受区间 `[0.8386, 0.8586]` | `op_learning/runtime/hy-repo-precision-20261005-rev2/` |
| 组委会 `run_ci.py` 端到端（**临时回归**，非 CP2；2026-10-13 实测） | `prec.sh` 原样：`wmt24pp` 200 例 / batch 8 / COMET 打分 | `wmt24pp` **actual=0.8509**（baseline 0.8486，接受区间 `[0.8386, 0.8586]`）**PASS**；同窗 `RUN_CI_EXIT=0` | 原件 `op_learning/hy-mt2-1.8b/c22-hy-ci-capacity-20261013/endtoend-20261013T100839Z/`：`results/precision_result.json`（sha256 `766adbbe5cdb2feaf5021b75b73aea374359f8222129f06f257779bc81606d2f`，内容 `{"wmt24pp": 0.8509}`）、`RUN_CI_EXIT.txt`（内容 `0`）、`run_ci.stdout.txt`（含 `精度对比 (Hy-MT2-1.8B): wmt24pp: actual=0.8509 baseline=0.8486 range=[0.8386, 0.8586] PASS`） |
| 算子级数值 | 官方 `zoo/teco/rms_norm/test_case/*.prototxt`，DIFF3_MAX | 96/96 run ALL PASSED；`+add` 三例 `max_error` 6.06796e-04 / 9.19963e-04 / 8.10373e-04，其余 0，阈值 1.00e-03 | `op_learning/hy-mt2-1.8b/rmsnorm-official-perf-20261010/` |
| 启动/绑定 fail-closed | 缺接口、身份校验失败、非 FP16 显式 dtype | 失败即终止启动（`SystemExit(1)`） | `tests/test_norm_guards.py`、`tests/test_cache_guards.py`、`tests/test_vllm_*.py`（**需 SDAA**：这些用例 import `torch_sdaa`，无卡时会以 `module 'torch' has no attribute 'sdaa'` 失败，属环境要求而非缺陷） |

**请求形式很重要**：把字面 prompt 直接发 `/v1/completions` **不被上述基线覆盖**（实测首 token `185` vs 冻结基线 `6400`）。所有 32-token 门禁都必须按 chat 形式比对，否则会假失败。

复现命令（在**推理仓根目录**执行；SDAA 用例需要设备与 SDK 环境）：

```bash
b=model_adaptations/HyMT2SCUdoudui
source /opt/tecoai/setvars.sh
export PYTHONPATH="$b/runtime:${PYTHONPATH:-}"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/test_norm_guards.py"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/test_cache_guards.py"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/test_vllm_rms_norm.py"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/test_vllm_swiglu.py"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/test_vllm_attention.py"
SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python "$b/tests/test_vllm_mlp_caller.py"
```

**已知限制**：

- 只验证固定输入的短请求；配置上下文 8192，**未评测 8k 全长生成**；未做组委会口径精度评测，owner CI 与官方评测是独立门禁；
- **交付运行时忽略模型 config 的 `rope_scaling`**：vLLM loader 把 `config.rope_parameters` 填成 `{"rope_type":"default"}`（`vllm/transformers_utils/config.py:386-388`），而本模型把动态缩放放在 `rope_scaling`，因此实际建的是 plain cache。这是**交付侧既有语义、非本队算子引入**，对融合算子无影响（算子只消费模型产出的 cache），但**可能与精度口径有关**，如实登记；
- 保留厂商 FP16 次正规数（subnormal）冲刷行为；原始 CPU 算术并非普遍逐位等价（CPU 参考固定 `rtol/atol 0.002`）；
- 镜像内已安装的 `tecoops` wheel（`libteco_ops.so` sha256 `388dd710a8357ee9294ccada17fa3f4003a9bc5aa107bec744184fa008cbbcc0`）**不含 `rotary_embedding`**，且需由本队 ops 库提供 cycle-42 的 `torch.ops.tecoops_ops.rms_norm_out` / `rms_norm_add_out` 入口，不能服务本模型：必须按「模型注册」第 2 步 build + 更新安装**本队** ops 库，否则 overlay 会 fail-closed 退出；官方 PR #48 的 wheel 亦**不可**单独充当服务库（只含上游 `rms_norm` + 本队 rope）。本队库以隔离安装（`pip install --target` / `PYTHONPATH` 前置）方式使用，并逐臂记录 `libteco_ops.so` sha256 与能力探针结果；
- **bf16 未适配**：交付运行态是 FP16，且 SDAA slave SIMD 头无 bf16 向量类型/转换；若将来按 config 原生 bf16 运行需另开独立验证。

## 性能验证

口径：**模型级** = 同一 build、同一固定 prompt、贪婪 32 token，每臂独立进程 + 预热 + 3 次计时取中位；**算子级** = 官方测例 + `./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 --case_path=zoo/teco/rms_norm/test_case/<case>.prototxt` 的 `Average Hardware Time` 中位。

### 模型级（融合 `rotary_embedding`，窗口 2026-10-12T13:01:42Z）

| 轮次 | A1（原生链）ms | B（融合）ms | A2 ms | 增益（vs A1） | 漂移 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3591.8 | 2993.8 | 3688.5 | +16.65% | 2.69% |
| 2 | 3661.4 | 3007.5 | 3627.4 | +17.86% | 0.93% |
| 3 | 3606.0 | 3006.8 | 3270.1 | +16.61% | 9.31% |

**中位增益 +16.65%，中位漂移 2.69% ⇒ STRONG_GO**（预注册 ≥+5.0% GO / <+2.0% NO-GO，且增益绝对值大于漂移）。9 臂全部 32/32 token 与冻结基线一致；探针（进程内）A 档 `bound_modules=0 / rotary_calls=0`、B 档 `bound_modules=1 / rotary_calls=2112`。

**机制（必须与数字同报）**：收益来自 **host 侧 launch 消除** —— 每请求去掉 **16 384** 次发射（原生链每层 17 次发射 → 融合后每层 1 次：17×32 = **544 → 32** launches/forward，−94.1%），**不是 GPU 计算收益**（kernel 自身 device 时间约 0.0018–0.0039 ms/层）；**不可外推**到把 RoPE 图捕获（torch.compile/CUDA-graph）的栈。

### 算子级

| 算子 | 测例（几何） | main → +SIMD → PR37head（AHT 中位 ms） | 相对 main |
| --- | --- | --- | --- |
| `rms_norm` plain | case_0 64×4096 | 0.122900 → 0.040922 → 0.040542 | −66.70% |
| `rms_norm` +res | case_1 64×4096+res | 0.173940 → 0.092706 → 0.092715 | −46.70% |
| `rms_norm` plain | case_2 64×2048 | 0.062402 → 0.021412 → 0.021204 | −65.69% |
| `rms_norm` +res | case_3 64×2048+res | 0.087922 → 0.048013 → 0.048049 | −45.39% |
| `rms_norm` plain | case_4 64×1536 | 0.047207 → 0.016546 → 0.016335 | −64.95% |
| `rms_norm` +res | case_5 64×1536+res | 0.066546 → 0.036937 → 0.036997 | −44.49% |
| `rms_norm` plain | case_6 64×128 (e1e-6) | 0.0059653 → 0.0034655 → 0.0033142 | −41.91% |
| `rms_norm` plain | case_7 64×128 (e1e-5) | 0.0059424 → 0.0034368 → 0.0033251 | −42.16% |

官方口径合计 **−41.9% ~ −66.7%**；收益主体是 SIMD epilogue（plain-DMA overlap + current-stream 在 64×D 上只再贡献 −0.93%/−0.97%/−1.28%，D=128 上 −4.37%/−3.25%；residual 路径 +0.01%~+0.16%，在极差内）。三臂库身份互异（由实际执行 `demo` 的 `ldd` 逐臂证明），96/96 run 全部通过。

**`+add` 精度余量冻结（A 档保留）**：最差 `+add` 用例 case_3（64×2048+res）`output max_error = 9.19963e-04` vs 阈值 `1.00e-03` ⇒ **余量 8.0%**（case_1 39.3%、case_5 19.0%；plain 五例与 residual_out 全 0）。零误差的替代形态 B 在同 build 下会使 `+add` 三例变慢 **+5.22% / +5.98% / +5.97%**，收益与风险不成立，故**保持 A 档**；只有门限收紧到 ≤9.0e-4（或明确按 1e-4/3e-4 判）时才切换。证据：`op_learning/hy-mt2-1.8b/cycle15-rmsnorm-precision-tradeoff-20261011/DECISION.md`、`op_learning/norm/hy-rmsnorm-add-margin-c16-20261011/`。

TTFT / TPOT：**未测**（组委会性能测例与达标线未下发）。已有的端到端数字（32-token 单请求，chat 形式，TP1/FP16/ctx8192）为上述 A/B 的 A1/B/A2 各臂中位；另有一次同分支 baseline 口径中位 **3593.6 ms**（2026-10-07，`op_learning/graph/hy-cudagraph-full-decode-only-20261007/`），仅作记录，不与本 A/B 混用。

## 已知限制

> **已知限制（2026-10-14，交付库 `e4e691d` 测得）**：`e4e691d` 把 `flash_attention` 的 `<128>` 实例化移入**尾部新 TU**，使整库 device 代码布局重排，**`rms_norm` 的 8 个官方 case 因 code-layout 效应变慢**（逐例 `+0.35…+2.6 µs`，合计 `+11.88 µs`；同构建 A/B，`max_error` 逐位不变、`rms_norm_fp16.scpp` 源码 sha256 未变）。该效应**与算子源码无关**，且**换任何结构都只会把税转移给另一个算子**（例如把新 TU 移进 `flash_attention/` 目录会让 `rms_norm` 归零但使 `weight_stream_gemv` 慢 `−35%/−18%`）。本队选择维持 `e4e691d` 的 device 布局（其后 `beebae2` 只改 host 谓词，全部 device 目标码逐字节不变），理由是净账：`flash_attention` 官方 11 例表 `−1.58367 ms`，而 `weight_stream_gemv` 正是 Hy/InternVL 两条线模型级收益所绑定的算子。逐 case 原值与完整 A/B 见本队账本仓 `docs/competition/c25-lead-ledger-20261014.md` §8.4 与 §9.4。

## 许可证

本队改动以 **Apache-2.0** 提交；本目录**不含** `LICENSE` 文件，三仓根 `LICENSE` 为**上游 fork 继承的 BSD-3-Clause**（据终检报告，blob 与上游同值，**未改动**）。该字面差异**待向组委会确认**，本目录不做许可证改写。

## 沿革

本目录此前为英文长文档，2026-10-12 按推理仓官方示例模板重写为中文七段结构；历史逐次实验、旧 SHA、废弃流程与更早的对照证据（含 Transformers/BF16 形态的独立结论）一律保留在 `experiments/hy-mt2-1.8b.md`、`op_learning/**` 与 `validation/*.json`，不再在 README 中复述。

---

**cycle-31 pin 说明（2026-10-16）**：本交付的 ops pin 由 `beebae220f0e918460d2210e4064030e157d602d` 前移至 **`8ef0a5f891603ddc8011cc3c45b127fe988024bf`**（本队 teco-ops main，含 cycle-31 落地的 `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写、`8ef0a5f` weight_stream_gemv ILP4、`f092863` flash_attention fail-closed 对、`973cbd2` weight_stream_gemv SPM fail-loud、`6d0bde7` 测试面 mock 健壮性）。组合树门禁：单次调用（`--perf_repeat=0 --warm_repeat=0`）`max_error=0`、`python_api_test` focused ALL PASSED、官方全 39 例 `PASSED=39 FAILED=0`（逐例 `max_error` 仅 `weight_stream_gemv` 两例按归约序变化）。旧 pin `beebae220f0e918460d2210e4064030e157d602d` 仍可复现旧库。
**cycle-32 pin 说明（2026-10-16）**：本交付的 ops pin 由 `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 前移至 **`0371909a87d5dd11bb928b487afbf6e1e25eb44c`**（本队 teco-ops main，新增 `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步：把 KV tile 内 V 面板的 strided 打包由逐面板同步 `memcpy_stride` 改为 `memcpy_async` 一板预取 + 双 pane ping-pong（两块 pane 落在既有 `ktmp` 内，SPM 占用不变），末板前视按 `MIN(hb + 1, npanel - 1)` 夹紧以免构造零长描述符，循环体保持无实现选择分支。算子级（官方命令形态 `./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3`，同一 build root、仅替换 `libteco_ops.so`）：`flash_attention case_2` 1.829900 → 1.804300 ms（**+1.3990% 加速**，即时间 −1.3990%；落地前复核 1.8300 → 1.8042 ms = +1.410%）；D128 家族 +1.21%/+1.30%/+1.41%、D64 +0.42%、`case_5` +2.48%，未改动 TU 的 D256/D512 家族内对照 −0.037%/+0.118%；11/11 `max_error` 与基线逐位相同（×3 rep），单次调用门禁（`--perf_repeat=0 --warm_repeat=0`）`case_2`/`case_0` = `1.83614e-04`/`1.78913e-04` 且与基线逐位相同。**模型级无可测加速**：MiniCPM5-1B 交付 `run.sh` 下 16 臂每请求 32/32 token 与冻结基线逐位一致、errors=0、无 CPU fallback/OOM、峰值显存与 KV cache 逐位相同，稳态 420.02 s / 344 请求 / 0 错误 / 344-344 逐位一致；墙钟**低于可测门**（pooled +0.464%，A1 自身臂间中位跨度 5.37%）⇒ 不主张模型级加速（亦不构成回退证据）。动态探针 `tecoops.flash_attn_varlen_func = 48`（24 层 × 2 请求）证明该 kernel 真在 MiniCPM prefill 路径执行。V-B（给 `M128_N32` 的 B 侧一个真实 stride、整行消费 V）= **NO-GO**：`M128_N32` 不 honor B stride。旧 pin `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 仍可复现旧库。 **本线可达性**：本线 attention 绑厂商 `torch.ops._C_sdaa.block_attention`、不走团队 flash 入口 ⇒ 本次改动对本线不可达（本线未独立复跑）。
**cycle-36 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0371909a87d5dd11bb928b487afbf6e1e25eb44c` 前移至 **`2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a`**（三个原子提交：`9f11b33` `weight_stream_gemv` 激活行 fp32 化（官方 case_0/case_1 **−4.55%/−8.28%**）、`0132061` `reshape_and_cache` 跨 token 载入流水线（**−22.4%/−27.5%/−23.2%**）、`2a1716d` `weight_stream_gemv` 尾部冗余 wait 删除（**−2.4%/−3.0%**）；全部为官方固定形态**同窗口同 build root 只换 `.so`** 的 A/B、3 次原值取中位数、`max_error` **逐位不变**、落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §4.2`）。
**cycle-36 round-5 pin 说明（2026-10-17）**：本交付的 ops pin 由 `2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a` 前移至 **`0d6bd5a10eddde4c020bb2367a233c72e005ad65`**（`perf(flash_attention): emit P straight from the softmax pass instead of transposing it` —— ALGO0/D128 每个面板的 `P^T→P` 转置融进 softmax 第二趟：寄存器直写 `p_score`、同一舍入；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：case_0/8/9/10 **−16.1%**、case_1 −9.7%、case_2/3 −7.5%、case_4 −8.5%、**十一例加权 −10.7%**；`max_error` **逐位不变**；**11/11 PASSED**；已知权衡：case_5 **+10.7%**（0.041353→0.045774 ms）。本线可达性：本线 attention 绑厂商 `torch.ops._C_sdaa.block_attention`、不走团队 flash 入口 ⇒ 本次改动对本线不可达（本线未独立复跑）。证据路径 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §6`。
**cycle-36 round-14 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0d6bd5a10eddde4c020bb2367a233c72e005ad65` 前移至 **`7ea7e84c8aef327d522d3eb318bc4808b15d070c`**（`7ea7e84` `perf(flash_attention): skip the softmax exp on causally-masked score vectors`；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：vs base `0371909` case_0/8/9/10 **−18.32%**、case_1 −9.50%、case_2/3 −6.05%/−6.08%、case_4 −6.85%；**vs 前一交付态 `0d6bd5a`** case_0/8/9/10 **−4.41%**、flash 算子再降 **−1.5%**；`max_error` **逐位不变**、**11/11 PASSED**；证据 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §13/§14`）。
**cycle-37 pin 说明（2026-10-18）**：本交付的 ops pin 由 `7ea7e84c8aef327d522d3eb318bc4808b15d070c` 前移至 **`785a7848169307ea531b52107881b36277737332`**（三个原子提交：`37f89d1` flash_attention D256/ALGO1 的 Q-block `parts` 分裂（官方 `case_6` 同窗 A-B **−14.79% / −14.81%**，两窗，`max_error` 逐位不变）；`542a2a1` reshape_and_cache 的 PUT 槽深两迭代（case_0 −1.33%/−0.89%、case_1 **−7.02%/−6.85%**、case_2 **−7.82%/−7.90%**，两窗，两臂 `max_error` 全 0）；`785a784` 构建期把 weight_stream_gemv 设备段排在 rotary_embedding 之前（官方 case_0 **−37.0%**、case_1 **−25.4%**，i-cache miss 14/3776 ns → 4/38 ns）。落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c37/lead/`）。
**cycle-38 pin 说明（2026-10-18）**：本交付的 ops pin 由 `785a7848169307ea531b52107881b36277737332` 前移至 **`f748d0abc93ab813f6e35feed261795ba850bd3b`**（两个原子提交：`3f98d68` flash_attention ALGO0/D128 的 Q-block 改用入口期 LPT 表分配（两窗复现：`case_2` −30.30%、`case_3` −30.29%、`case_4` −30.82%，flash 集合加权 **−13.6%**；如实披露 `case_0/8/9/10` +1.6%、`case_1` +1.74%、`case_5` +7.35% 的权衡）；`f748d0a` reshape_and_cache 暂存环 8 组（`case_0` −1.8%、`case_1` −5.7%、`case_2` −4.8%，两臂 `max_error` 全 0）。落地库 **39/39 PASSED**。已知待修：`rotary_embedding` 自 `785a784` 起处于慢地址（+133%，0.0017→0.0040 ms），由段序守卫处理。
**cycle-38 round-2 pin 说明（2026-10-18）**：本交付的 ops pin 由 `f748d0abc93ab813f6e35feed261795ba850bd3b` 前移至 **`9fda597f7d8bc5f98893081372a552f411f26e42`**（`9fda597`：flash_attention D256/ALGO1 调度单元 `UQ = BM/2 = 32` 的切分修正；官方固定形态同窗同 build root 只换 `.so`：**`case_6` 0.183760 → 0.116800 ms（−36.44%）**、`case_7` 0.151690 → 0.149830（−1.23%），其余 9 例 −0.1~−0.3% 无回退；`max_error` 逐位不变，**11/11 PASSED、39/39 PASSED**）。
**cycle-38 round-10 pin 说明（2026-10-18）**：本交付的 ops pin 由 `9fda597f7d8bc5f98893081372a552f411f26e42` 前移至 **`20c245a39bbf35fa4dee0c0dbcd3313248490bbf`**（`20c245a` `perf(build): pin the hot device sections into their measured fast addresses`：在 TAX-1 基础上用 5 步纯 list 重排把三个热算子的 `.aitext` 精确钉进实测快类——`weight_stream_gemv` **0x25a08**、`rotary_embedding` **0x28858**、`zzz_flash_attention_d128` **0x2a878**；官方固定形态**同窗同 root 从源码构建**的双臂 A/B、3 次中位数：rotary `case_0/1/2` **0.003998→0.001686 / 0.006323→0.003904 / 0.003994→0.001729（−57.8% / −38.3% / −56.7%）**、wsg `case_0/1` **0.005722→0.003977 / 0.007753→0.007686（−30.5% / −0.9%）**、flash D128 `case_0` **0.828940→0.809140（−2.39%）**、rms_norm `case_6` −0.54%；8 例 SUM `0.87075→0.841684（−3.34%）`；**39/39 PASSED**、`max_error` 不变。原始件 `/root/tmpwork/c38/ops/out-r10/`、`RESULT-R10.md`）。
**cycle-39 pin 说明（2026-10-18）**：本交付的 ops pin 由 `20c245a39bbf35fa4dee0c0dbcd3313248490bbf` 前移至 **`12271c185e3042392d8c9206ceb7535b20a53f50`**（`79470cb` flash D256/D512 的 NT PV 面板异步双 pane：`case_6` **−4.13%/−4.19%**、`case_7` **−4.31%/−4.01%**、`case_5` −5.01%/−4.99%，其余 8 例 −0.2~−0.4% 无回退；`12271c1` weight_stream_gemv 驻留 slab 分裂：`case_0` −1.14%/−2.40%、`case_1` **−11.54%/−12.22%**；两提交均 39/39 PASSED）。已知代价：两内核体积变化使其后锚点平移，wsg `case_0` 从 pin 峰值 0.00396 退到 0.00570（仍比 cycle-37 起点快 17%）；round-12 重推臂因拉回 pre-wsg 微秒算子的旧慢地址而 SUM +0.174%，已 REJECT。
**cycle-39 round-13 pin 说明（2026-10-18）**：本交付的 ops pin 由 `12271c185e3042392d8c9206ceb7535b20a53f50` 前移至 **`11cb85cbdc02470dde0a50d9777bb1e386501b9c`**（两个原子提交：`fab1fa2` flash_attention 的**首 tile rescale 跳过**（每个 Q block 第一个 KV tile 上 `scale = exp(-inf) = 0`，把 0 写成 0 ⇒ 逐位不变；同窗两窗：`case_0/8/9/10` **−1.60%/−1.61%**、`case_2/3` **−1.18%/−1.17%**、`case_5` **−2.31%/−2.28%**、`case_4` −0.89%、`case_1` −0.90%；flash 集合 **−1.20%**；`max_error` 11 例逐位不变、39/39 PASSED）；`11cb85c` D512 **tail** PV 循环的异步双 pane（`79470cb` 只覆盖主循环，tail 占每头 48 个面板中的 32 个；同窗两窗：`case_7` **−3.99%/−3.91%**，`case_6` 无 tail ⇒ +0.12%；`max_error` 逐位不变、39/39 PASSED））。原始件 `/root/tmpwork/c39/`、台账 `docs/competition/c37-lead-ledger-20261018.md`。
**cycle-39 round-15 pin 说明（2026-10-18）**：本交付的 ops pin 由 `11cb85cbdc02470dde0a50d9777bb1e386501b9c` 前移至 **`60a40d756f85fa6c1ac9d4cb3563a5f3ffd9c7db`**——**回退** `11cb85c`（`perf(flash_attention): overlap the D512 tail V-panel staging with its PV GEMMs`）。回退依据（AGENTS 规则 7，稳定负优化）：该提交的**自身 11 例窗口 SUM = +0.153% / +0.151%（两窗，净负）**——目标例 `case_7` 确实 −3.990%/−3.920%，但其余 10 例一致回退 +0.19~0.53%（LTO 重链把 D128 段推离原位），故按逐例/求和双口径均不成立。保留 `fab1fa2`（flash_attention 首 tile rescale 跳过：D128 9 例 −0.89~−2.31%，11 例 SUM **−1.21%**，`max_error` 逐位不变）。
**cycle-39 round-17 pin 说明（2026-10-18）**：本交付的 ops pin 由 `60a40d756f85fa6c1ac9d4cb3563a5f3ffd9c7db` 前移至 **`ce1a2ec00270bb5a519e235166427c255ef2af40`**（`ce1a2ec` `perf(build): swap the weight_stream_gemv and gather device sections into their measured fast slots`：在纯 TU 顺序上把 `weight_stream_gemv` 与 `zz_ms_deform_attn_backward_ha`(gather) **互换**——wsg 落到实测快地址 `0x23dd4`，而 `flash_d128`/`rms_norm_fp16`/`msda_fwd`/`msda_bwd_fp32`/`memset`/`rotary`/`flatten_rays` **全部保持原址**（这是 K3 做不到的）；官方固定形态**同窗同 root 从源码构建**双臂 A/B：wsg `case_0` 0.005718→**0.003871（−32.30%）**、`case_1` −2.03%、`ms_deform_attn_backward` **−1.15%**、`ms_deform_attn_forward` **−3.66%**、`reshape_and_cache` −0.36%/−0.58%、`memset` ±0.3%、`rotary` ±0.6%；39 例 SUM 9.414836→9.412432（−0.026%）、无 >±0.8% 回退（首窗 1-rep 的 µs 级“越带”已由 3-rep 复核证否）。**说明**：`wsg 快 + gather 原址`经结构性证明不可同时满足（wsg 快地址均低于其 base，而 gather 是其直接前驱、TU 不可拆）⇒ 本折中为 13 臂枚举最优。
**cycle-42 pin 说明（2026-10-19）**：本交付的 ops pin 由 `ce1a2ec00270bb5a519e235166427c255ef2af40` 前移至 **`75588a996c3c1ac6f5239d5bd328bd8d5ea8c97e`**（团队 ops main head，与 `land/c42-flashopt` 同 head，已合入 main）。本轮级联四个重要提交：
1. `5813b74`：flash D128 `P^T→P` 翻转 16-lane 向量化，两独立窗 39 例 SUM 8.3785→7.5996 / 8.3796→7.5999 ms（**−9.30%**），39/39 PASSED，`max_error` 逐位全同；
2. `e6ec6e9`：`api/torch_ext.cpp` decode 热路径改走 C++ `TORCH_LIBRARY` `out=` op，host 派发 75.61→26.28 µs（−65%），3 731 712 halfword 逐位全等，零设备布局税；
3. `35bf8f6`：新增 `rms_norm_add_out`（带 residual 的 `out=` 入口），host 派发 76.38→25.08 µs（−67.2%），7 个真实 shape 36 352 halfword 逐位全等，两窗 39 例 SUM 净零，无布局税；
4. `75588a9`：`zzz_flash_attention_d128.scpp` K cache 异步双缓冲预取 + Step B/E 寄存器融合 + 全可见 tile 掩码跳过 + softmax 栈溢出消除，长 shape 探针 16/16 PASSED（prefill 2K/4K/8K/16K 由 14.42/55.84/219.77/872.02 → 12.27/47.34/185.93/737.10 ms，**−14.9%~−15.5%**），官方 39 例 SUM 7.5999 → 6.7810 ms（**−10.78%**），41 探针×3 rep×2 臂独立复核 −10.744% 且 246/246 PASSED。
本线自建库 sha256（跨构建根 sha 会不同，判据是能力探针，见上文「本线实测自建库身份」）：`libteco_ops.so` = `42422972ee7f9e42b2c62595a680193db49778ec33f53280351fa205c77cb5ca`，`_torch_ext*.so` = `0fbf071ae5d433d5d74bf305a610d41e484a976f14f3812857a7c142c771c377`。

**cycle-42 overlay 切换（2026-10-19）**：本线 rms_norm / rms_norm_add 已从 Python `torch.library.custom_op` 切换到 C++ `TORCH_LIBRARY` 入口 `torch.ops.tecoops_ops.rms_norm_out` / `rms_norm_add_out`（`runtime/custom_ops/hy_mt2/vllm_rms_norm.py:31-39`；进程初始化期一次绑定，热路径零分支零 getenv）。证据：交付 run.sh 原样参数起服务，32-token 贪婪生成 Prompt1/Prompt2 在 3 轮臂序交替中 **32/32 token 逐位相同**；TPOT 48.502 -> 44.621 ms/step（**+8.00%**，|Δ|=3.881 > 同臂 spread 1.982 ⇒ GO）；零 traceback/OOM/CPU fallback。op 级单价 73.9-76.4 µs -> 24.2-25.1 µs（-67.2%），逐 halfword 逐位全等。


## 2026-10-24 复现与库身份门禁

当前源码固定为本队 teco-ops
`db1597e1265a5fc3e24207b793c9d1dd94327a24`，与本包
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

使用本节固定的 ops pin db1597e 及 validation/ops-artifacts.json 对应的
core/extension，按本模型原样启动参数运行固定 prompt、greedy-32：
**196 请求 / 300.614 s，全部 32 tokens 且逐位一致，0 请求错误，外层 rc=0**。
一致性判据为与该窗口首个请求逐 token 相同；本窗口不另作官方精度主张。
原件随包在 validation/steady-20261024/，包含每次 token IDs、结果、启动配置及
受保护官方脚本前后相同的 SHA256。该窗口为现有服务器上的隔离安装验证，
不是干净容器或完整官方 CI，不新增性能提升主张。
