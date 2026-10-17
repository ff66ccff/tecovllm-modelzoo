# Gemma-4-12B-it（SCU 都队）

## 模型概述

Gemma-4-12B-it 是 Google 的文本生成大模型，约 12B 参数（fp16 权重约 23.9 GB），48 层解码器：40 层
sliding attention（head_dim 256，窗口 1024）+ 8 层 full attention（head_dim 512，单 KV head），
hidden 3840，MLP 中间维 15360（GeGLU/tanh 近似），词表 262144，embedding 与 lm_head 共享。
本 checkpoint 另含 vision/audio 侧张量，本适配只使用文本主干。

本适配在**本队 TecovLLM（vLLM 分支）+ Torch-SDAA 软件栈**上完成 TP2 / FP16 的端到端推理，并接入
本队自研的 tecoops C++ 算子。所有模型侧改动都是**进程内的（overlay）**：不修改 site-packages、
不修改厂商包、不复制权重。

- **进程内注册与调用点绑定**（`overlay/` + `runtime/`，进程初始化期一次完成，热路径零分支）：
  `gemma4_unified` config 注册、tokenizer `extra_special_tokens` 兼容、多模态专属权重跳过、
  attention 调用点（D256 sliding decode 走尾随 1024 KV 窗口 + 非因果融合 SDPA；D512 global decode
  走显式 math，或可选的官方隔离 kernel）、**RMSNorm 调用点（默认绑定整链单 kernel 的
  `tecoops.rms_norm`）**、M=1 decode 线性层绑定 `tecoops.weight_stream_gemv`。
- **自研 tecoops C++ 算子**（来自本队 teco-ops 仓）：`rms_norm`、`rms_norm_compat`、
  `reshape_and_cache`、`flash_attn_varlen_func`（D256/D512 特化）、`weight_stream_gemv`。
- **难度系数 1.0**（赛题序号 1）。
- 历史沿革：2026-09-19 立项短探测 → 09-25 打通 CP1（overlay 注册 + 窗口归一化 + math attention）
  → 10-02…10-05 算子与容量档（2048/2304/4352）→ 10-07…10-11 flash_attention 多轮向量化/布局优化
  → 10-11 `rms_norm_compat` → 10-12 `weight_stream_gemv` 与 RMSNorm 整链单 kernel 绑定 + 300 s 稳态门禁。
  逐轮原始记录见模型仓 `experiments/gemma-4-12b-it.md`；本树的 `validation/*.json` 是当时收据。

## 模型权重下载

> **本模型使用组委会预置权重，按只读方式引用：不重复下载、不转储副本**（仓库硬约束 1）。
> 因此本文件**不提供** `modelscope download` 之类命令——权重不是下载来的。

- 默认路径：`${MODEL_ROOT:-/gpfs/model}/google/gemma-4-12B-it`，本机实际为
  `/gpfs/model/google/gemma-4-12B-it`（仅只读引用，适配过程不写入该目录）。
- 目录内容：`config.json`、`generation_config.json`、`chat_template.jinja`、
  `tokenizer.json` / `tokenizer_config.json`、`model.safetensors`（23.9 GB）等。
- 权重位置不在 `/gpfs/model` 下时，用环境变量覆盖，**无需移动或复制权重**：

```bash
export MODEL_ROOT=/your/model/root          # 服务将加载 $MODEL_ROOT/google/gemma-4-12B-it
# 或直接指定：
export GEMMA_PATH=/your/path/to/gemma-4-12B-it
```

`run.sh` 会设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，禁止联网取权重。
> ⚠️ 官方 CI 脚本 `tools/ci_pipline/run_ci.py` 会从 `run.sh` 中**字面**读取该模型路径并作为
> `speed.sh --tokenizer-path` 传给评测客户端（见"已知限制"）。若权重被迁走，请让
> `/gpfs/model` 指向真实根目录（例如软链接），否则服务能正常启动、但官方性能步骤会因该字面
> 路径不存在而失败。

## 模型注册

### 1) 环境

```bash
source /opt/tecoai/setvars.sh
export PATH=/home/py312/bin:$PATH          # 厂商 Python 3.12
```

### 2) tecoops 算子库（本队 teco-ops 仓构建，**不安装进全局 site-packages**）

```bash
set +u; source /opt/tecoai/setvars.sh; set -u   # 提供 /opt/tecoai/lib64（libsdaart.so 等）
cd <teco-ops-repo>
export WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF
/home/py312/bin/python setup.py build_ext --inplace
# 产物：build/teco/lib/libteco_ops.so 与 api/tecoops/_torch_ext*.so
export LD_LIBRARY_PATH=<teco-ops-repo>/build/teco/lib:/opt/tecoai/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=<teco-ops-repo>/api:${PYTHONPATH:-}
# 库身份自检（初始化期一次；fail-closed 退出码）。**不要**用 len(tecoops.__all__) > 0
# 当判据 —— 环境里 cycle-14 的旧 wheel 也通过它，却缺 rms_norm_compat 与
# flash_attention 的 D256/D512 特化符号（实测，见本文件"已知限制"第 8 条）。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/py312/bin/python \
  model_adaptations/Gemma4SCUdoudui/scripts/gemma4/selfcheck_tecoops.py
# 期望输出 tecoops.__file__ + libteco_ops.so sha256 + 能力探针，末行 SELFCHECK PASS、rc=0
```

> **必需步骤**：先按上面 build + 把本队 ops 安装目录前置到 `PYTHONPATH`（或 `pip install
> --target`），再跑 `run.sh`。交付包**不含** `tecoops`；缺包/缺符号时 overlay 刻意
> **fail-closed**（`SystemExit(1)`，不是崩溃）。ops 源码 pin：`teco-ops` 队内 `main`
> `9fda597f7d8bc5f98893081372a552f411f26e42`。
> 库身份与实测口径见 `model_adaptations/Gemma4SCUdoudui/validation/rmsnorm_whole_chain_20261012.json`
> （该文件记录的是**上一版 pin** `dda33c5b` / `4adfd05e` 那一轮构建的
> `libteco_ops_sha256 = 1e756307a94a2eb871856b3830b417f97f26914906f3a75fe51e61cdab2782a5`；
> pin 前移后必须按上面步骤重建并重测本轮的 sha256）。
>
> 本 pin 已前移至 **`9fda597f7d8bc5f98893081372a552f411f26e42`**（自 `4adfd05e` 起依次落地：`79941c5f` ms_deform_attn forward plane → `d84f34f` flash_attention D256 `BM==64` → `54a1d65` rms_norm epilogue 二级展开 → `98e18ca` rotary_embedding CI 修复同步 → `e4e691d` flash_attention `BM==128` NT-form + 转置域 softmax → `beebae2` weight_stream_gemv host 谓词 `n % 256 == 0` → `n % 64 == 0`（按引擎 DMA 校验器实测的 4 B 要求），使 TP 分片的 lm_head 也走上 M=1 快路径 → `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写 → `8ef0a5f` weight_stream_gemv ILP4 → `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步（算子级 `case_2` 1.829900 → 1.804300 ms，+1.3990%；模型级无可测加速））。**照旧 pin（`4adfd05e`）复现得到的库不含上述改动**，故复现必须以 `9fda597f7d8bc5f98893081372a552f411f26e42` 为准。
> **实测边界**：当前默认档只需要 `tecoops.rms_norm` 与 `weight_stream_gemv`，而环境里
> cycle-14 的旧 wheel `388dd710a8357ee9…` 恰好导出这两个 python 名 ⇒ 不前置交付安装目录时
> overlay **会静默降级**到旧库并进入 main（三个绑定点照常打印），**不报错**。因此上面的
> 自检必须作为启动前门禁，不能依赖"缺符号就会 fail-closed"。
>
> **cycle-28 硬化（2026-10-15，交付文件 `runtime/custom_ops/d512_official.py`，提交仓 `model/gemma-4-12b-it` `9ac6beb`）**：
> D512 decode 绑定现在**额外**要求被映射的 `libteco_ops.so` 在其 ELF `.dynsym` 里导出
> `teco_slave_flash_attention_half_d512`（即 `findFlashAttentionBranch()` 对 `head_dim == 512`
> 选中的那条分支）；缺失或无法判定（非 ELF64-LE / 无 `.dynsym` / 截断）时**进程在 import 期即
> `RuntimeError` 失败退出**（`overlay/sitecustomize.py` 转为 `SystemExit(1)`，rc=1，不进 main）。
> 因此上面"不能依赖缺符号就会 fail-closed"这句**只对 `rms_norm` / `weight_stream_gemv` 两个绑定
> 仍然成立**，D512 flash 绑定已不再静默降级。实测：环境里 cycle-14 的旧 wheel `388dd710a8357ee9…`
> 用旧探针会被接受（`PROBE_ACCEPTED (FAIL-OPEN!)`），用本硬化后的探针则 fail-closed。
> 模型级门禁窗口 `c28-gemma-probe2`（2026-10-15 08:48:00Z–08:50:05Z，rc=0，库 `38ad93d8…`）：
> 交付字节 32/32 token 与冻结基线逐位相同、8/8 global D512 层绑定到 `decode_forward`、
> 峰值显存与 cycle-27 各臂逐位相同。设备无关自检：
> `C28_DELIVERED_LIB=<真实 libteco_ops.so> python tests/test_d512_default_cpu.py`（无该变量时显式打印 SKIPPED）。

> 若跳过 `source /opt/tecoai/setvars.sh`，`import tecoops` 会报
> `ImportError: libsdaart.so: cannot open shared object file`——`libteco_ops.so` 依赖厂商运行库，
> 因此 `/opt/tecoai/lib64` 必须在 `LD_LIBRARY_PATH` 上。

### 3) 模型注册（进程内，启动时自动完成）

`run.sh` 已把 `overlay/` 与 `runtime/` 前置到 `PYTHONPATH` 并置 `GEMMA4_OVERLAY=1`，启动时依次完成：
config 注册、tokenizer 兼容、多模态权重跳过、attention/RMSNorm/线性层调用点绑定，以及 `metadata_cpu_fix`（第五个厂商缺口：把 `query_start_loc`/`seq_lens` 的 CPU 孪生经 `.clone()` 冻结后挂到 `BlockAttentionMetadata`，使 prefill/decode 算子不再逐层读设备标量 —— 交付形态实测 D2H 事件 481→97、wall gate 11.947%→9.127%）。
缺**符号**时启动即报错（fail-closed）。**但"库解析到哪一份"必须靠上面的能力探针**：
实测不前置交付安装目录时，`import tecoops` 落到全局 `site-packages` 的旧 wheel，
当前默认档所需的两个符号它都有 ⇒ overlay 三个绑定点照常打印并进入 main（静默降级）。

### 4) 自检

**无卡（不需要加速卡，只需厂商 Python 与源码）**：

```bash
bash -n model_adaptations/Gemma4SCUdoudui/run.sh                      # 语法
/home/py312/bin/python -m py_compile model_adaptations/Gemma4SCUdoudui/overlay/*.py \
    model_adaptations/Gemma4SCUdoudui/runtime/custom_ops/*/*.py       # 语法
```

**CPU 单测（无卡）**：跑的是**交付子树里的**那份测试文件 × 交付 `runtime/custom_ops` 代码
（该文件与仓库根 `scripts/gemma4/test_operator_overlay.py` **blob 逐位相同**，L11 纪律：
门禁验的字节 == 发布字节）：

```bash
cd <repo-root>
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=model_adaptations/Gemma4SCUdoudui/overlay:model_adaptations/Gemma4SCUdoudui/runtime \
  /home/py312/bin/python \
  model_adaptations/Gemma4SCUdoudui/scripts/gemma4/test_operator_overlay.py
# 期望（实测）：T1.0-T1.5 / T3.x / T4.0 / T5.0 / T6.0 / T7.0 / T9.0-T9.3 PASS
#   （T1.4/T1.5 = custom_ops 必须解析到交付 runtime/custom_ops 的防漂移断言）
# 但**套件整体 rc=1**：T2.0（overlay 绑定）与 T8.0（window 归一化）需要
#   GEMMA4_OVERLAY=1 + SDAA 栈，无卡时由 harness 记为 `[FAIL] ... :: skipped` 并汇总
# 2026-10-16 补记：在本机提供 GEMMA_PATH + 交付 tecoops（含 D512 符号）+ GEMMA4_OVERLAY=1 时，
#   同一套件实测 **23/23 PASS、ALL CHECKS PASSED**（含 T5.0/T6.0/T7.0/T9.x 数值面 maxdiff 0.00e+00），
#   且该结果在零 D2H 补丁前后各跑一次、均为全绿。
#   `FAILED (2)`。这是既有的判据形态，不是回归。
```

**需要 SDAA 栈（先 `source /opt/tecoai/setvars.sh`，且会加载 `torch_sdaa`）**：

```bash
# 启动失败必须 fail-closed：选定 D512 扩展缺失/不完整时不得进入 main
cd model_adaptations/Gemma4SCUdoudui && /home/py312/bin/python tests/test_startup_failure.py
```

实测结果（SDAA 栈就绪）：

```json
{"passed": true, "cases": [{"case": "missing_selected_library", "exit_code": 1, "main_unreachable": true},
                          {"case": "incomplete_selection",      "exit_code": 1, "main_unreachable": true}]}
```

> 该用例会以 overlay 启动一个子解释器（`import torch` 时自动加载 `torch_sdaa`），因此
> **不能**在 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 下运行——那样 overlay 会在更早的位置失败，
> 诊断信息不同，断言会报错（本文件上一版把它错列为"无卡"，已更正）。
>
> 另外：该用例曾**失败**，根因不在用例而在 overlay —— `sitecustomize` 里普通的 `raise`
> 会被 CPython `site.py` 吞掉（stderr 只留 `Error in sitecustomize`），**解释器照常继续并跑到
> main**，即"没绑上 overlay 也照样启动"的静默降级。已改为打印诊断后 `raise SystemExit(1)`，
> 使 fail-closed 真正落在进程级；设备无关对照实验（普通异常 rc=0 且打印 MAIN_REACHED；
> `SystemExit(1)` rc=1 且不打印）见 `readme-c16/failclosed/`。

## 启动推理服务

```bash
export MODEL_ROOT=/gpfs/model
bash model_adaptations/Gemma4SCUdoudui/run.sh
```

`run.sh` 内实际下发的 serve 命令（**CI 已固定，请勿改参**；下方为参数清单，实际路径取 `$GEMMA_PATH`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `<模型路径>` | `$GEMMA_PATH`（默认 `$MODEL_ROOT/google/gemma-4-12B-it`） | 只读引用预置权重 |
| `--served-model-name` | `gemma-4-12B-it` | 官方 CI 支持的模型名 |
| `--tensor-parallel-size` | `2`（`GEMMA_TP_SIZE`） | 该 checkpoint 需 TP2 才能放进单卡显存 |
| `--gpu-memory-utilization` | `0.92`（`GEMMA_UTIL`） | |
| `--max-model-len` | `16640`（`GEMMA_MAX_MODEL_LEN`） | 覆盖官方**固定性能用例** T1 2048 / T2 4096 / T3 8192 / T4 16384（各 +100 输出）。`4352` 会让 T3/T4 被服务端 **400 拒绝**，而 `run_ci.py::parse_performance` 要求每例 `Failed==0`（否则 CI FAILED） |
| `--max-num-batched-tokens` | `512`（`GEMMA_MAX_BATCHED_TOKENS`） | |
| `--trust-remote-code` | 开 | 需要 checkpoint 自带实现 |
| `--no-enable-prefix-caching` | 开 | 官方 CI 强制要求 |
| `--enable-chunked-prefill` | 开 | |
| `--hf-overrides` | `{"architectures":["Gemma4ForCausalLM"]}` | 文本主干架构 |
| `--dtype` | `float16` | 本线验证 dtype |
| `--host` / `--port` | `0.0.0.0` / `8003`（`GEMMA_HOST` / `GEMMA_PORT`） | |

启动后就绪与 32-token 冒烟：

```bash
curl -s http://127.0.0.1:8003/health
curl -s http://127.0.0.1:8003/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12B-it","prompt":"Explain in one sentence why the sky appears blue.",
       "max_tokens":32,"temperature":0}'
```

说明：`run.sh` 用一个 `vllm()` shell 函数在 **exec 之前**把模型路径参数改写成 `$GEMMA_PATH`、
并按 `$GEMMA_HOST`/`$GEMMA_PORT` 覆盖 `--host`/`--port`，其余 serve 参数逐字不动；因此
`MODEL_ROOT` / `GEMMA_PATH` 的覆盖**对实际加载的权重生效**（已用无卡 argv 实验验证：默认与
`MODEL_ROOT=<软链接别名>` 两种情况下都能在下发 argv 中看到对应路径，且别名 `readlink -f`
指向同一份只读权重，未复制权重）。服务要求厂商 Python 具备 `torch` + `torch_sdaa` +
`torch.sdaa.is_available()`；缺失时启动即失败（不做系统 Python 回退）。

## 精度验证

> **组委会标准测例与精度达标线尚未下发，因此当前只能给出"临时回归通过"，不是最终精度结论。**
> 待测例下发后补正式精度表。

| 项目 | 口径 | 结果 | 证据 |
|------|------|------|------|
| 固定 prompt 贪婪 32 token（主判据） | 裸文本 prompt `Explain in one sentence why the sky appears blue.`；`temperature=0`、`max_tokens=32`；TP2 / FP16 | **32 个 token ID 与冻结基线逐位一致**，文本连贯无复读；`107, 101, 818, 7217, 7412, 3730, 1547, 529, …` | 冻结基线 `op_learning/attention/gemma-d512-model/baseline_token_ids.json` |
| RMSNorm 整链单 kernel 绑定（交付默认档） | 3 轮 A1(compat)→B(whole)→A2(compat) 轮转，每臂独立进程 + 1 预热 + 1 计时 | **9/9 臂 32/32 token ID 与冻结基线逐位一致** | `validation/rmsnorm_whole_chain_20261012.json` |
| 300 s 连续请求稳态抽检 | 同一 prompt 连续请求，`elapsed=306.3836 s` | **38 请求 / 0 错误 / 38×32 token 全部逐位一致**（首/中/尾亦然） | 同上 |
| 官方算子用例 | `zoo/teco/rms_norm` 8 例，官方固定命令口径 | **16/16 PASSED**（两个 pass 各 8/8），逐 case `max_error` 与交付基线逐位相同（`0` / `6.06796e-04` / `0` / `9.19963e-04` / `0` / `8.10373e-04` / `0` / `0`，阈值 `1.00e-03`） | 同上；换绑定未改动算子库字节 |

另有仓库级临时回归（MMLU-Pro Computer Science 200 行，仓库接受区间 **0.73–0.93**）已完成并留档于模型仓
`logs/gemma-4-12b-it/precision200-20261005/`：`mean_acc = 0.835`（200/200 有效响应、correct 167）见
`evaluator_rev2.stdout.txt` 的 `│ gemma-4-12B-it │ mmlu_pro │ mean_acc │ computer science │ 200 │ 0.835 │ default │` 行，
逐请求原件为 `…/evalscope/predictions/gemma-4-12B-it/mmlu_pro_computer science.jsonl`（sha256 `676c5a29af9cc4eb…`，
与 evaluator 最终回执 `files[".jsonl"]` 记录逐字相同）。**该次运行是 selected-D512 档（设了 `GEMMA_D512_*`）**，
与组委会 CI 的默认档不同；**它不是组委会标准测例**，故本表不列分数、也不作为精度结论（**临时回归、非 CP2**）。

**未执行/未验证**：组委会标准测例、多模态输入、当前 owner CI。
**T3/T4（8K/16K）容量与边界已于 2026-10-13 在交付默认档实测**（每例 `Total==Success==10`、`Failed==0`、completion 恰 100；
16640 接受、16641 = 400）—— 详见本文件「性能验证」段的容量档收据与 `docs/competition/c22-gemma.md` §八；
**精度**（mmlu_pro 等）仍未在默认档做满量验证。

## 性能验证

### 1) 模型级（同会话 A/B/A，3 轮轮转；口径 = 单请求端到端墙钟 / 32-token 贪婪解码）

配置：TP2 / FP16 / `gpu_memory_utilization=0.92` / `max_model_len=2048` /
`max_num_batched_tokens=2048` / 未开 chunked prefill / `enforce_eager=False`；每臂独立进程，1 次预热 + 1 次计时。

| 变更 | A1 三次原值 (s) | B 三次原值 (s) | A2 三次原值 (s) | 中位数 B/A1 | A2/A1 漂移 |
|------|------------------|-----------------|------------------|--------------|-------------|
| RMSNorm 整链单 kernel 绑定（`whole`，交付默认档） | 12.4904 / 11.2930 / 12.7618 | **9.1702 / 8.2073 / 9.2957** | 12.5859 / 12.7229 / 12.5492 | **−26.58 %**（9.1702 vs 12.4904） | +0.76 % |
| `weight_stream_gemv`（M=1 decode 权重流式） | 13.7013 / 13.1179 / 12.8105 | **12.6963 / 12.6556 / 12.6236** | 13.1516 / 13.5271 / 13.1431 | **−3.52 %**（12.6556 vs 13.1179） | +0.26 % |

两条结论的效应均远大于漂移（C1a 为 34.8×），且 9 个 B 值全部低于 9 个 A 值（`max(B) < min(A)`）；
两轮都同时通过了 32-token token-ID 逐位一致检查（不是只看墙钟）。

**诚实边界**：① `−26.58 %` 的来源是**消除每次 decode step 的 5×337 = 1685 次 kernel 发射**
（把 RMSNorm 的 eager 统计量链与 epilogue 合并为一次发射），属**发射开销/延迟地板**类收益，
**不可外推到其它模型、其它形状或其它算子**；② 该实测值**远高于**事前折算预测（7.40 %）——
是"预测被实测覆盖"，不是"预测命中"，说明用 standalone→model 通过率外推会系统性低估此类改动；
③ 上表是**本队自建模型的墙钟口径**，不是官方 `Average Hardware Time`；④ 最长的官方 T3/T4 场景
与当前 owner CI 未执行。

### 2) 服务级时延（历史窗口，配置与当前默认档不同，仅作参考，勿与上表混用）

| 口径 | 输入 / 输出 / 并发 | 配置 | TTFT | TPOT / 单 token | 端到端 |
|------|--------------------|------|------|------------------|--------|
| HTTP 服务（2026-09-25 窗口） | 11 / 32 / 1 | ctx2048、math attention | 0.4870 / 0.4856 / 0.4969 s（中位 **0.487 s**） | — | 13.7134 / 13.9108 / 14.0798 s（中位 **13.91 s**，**2.228 tok/s**） |
| 本队离线 LLM（交付默认档 + 300 s 稳态窗口内 38 次请求） | 11 / 32 / 1 | ctx2048/MBT2048，RMSNorm `whole` | — | **251 ms/token**（中位单请求 8.0414 s ÷ 32） | 单请求中位 **8.0414 s**（min 7.9060 / max 8.4393） |
| 同上，对照档 `compat` | 11 / 32 / 1 | 同上，RMSNorm `compat` | — | 345 ms/token（11.0532 s ÷ 32） | 单请求中位 **11.0532 s** |
| 官方 T2 收据（ctx4352/batch512） | 4096 / 100 / 1 | 默认档 | — | inter-token 中位 **0.4276 s** | — |

未在当前默认档（16640/512）重测 TTFT/TPOT；上表各行的配置与窗口已在表内注明。

**容量档 T1–T4 能力收据（2026-10-13，交付默认档 math + `GEMMA_MAX_MODEL_LEN=16640`，同形态 warmup 1024+10 / max_tokens 100 / `ignore_eos` / parallel 1 / number 10）：****每例 `Total==Success==10`、`Failed==0`、completion 恰 100**；实际 prompt 区间 T1 [2048,2050] / T2 [4096,4097] / T3 [8192,8194] / T4 [16384,16385]；warmup 1/1/0；request span 314.9 / 362.2 / 467.8 / 689.0 s。边界：16640 接受、**16641 = 400**（含 16640 上下文限制）。**这是容量 enabler，不承诺提速**；原始件见 `docs/competition/c22-gemma.md` 索引。

### 3) 算子级（官方固定测例口径：`./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 --case_path=zoo/teco/<op>/test_case/<case>.prototxt | grep "Average Hardware Time"`，报平均值）

| 算子 / 用例 | 结果 |
|-------------|------|
| `rms_norm` case_0…7 | AHT (ms) `0.027797 / 0.034965 / 0.014652 / 0.018166 / 0.011346 / 0.014249 / 0.0026272 / 0.0026868`，8/8 PASSED，阈值 `1.00e-03`，`max_error` 与交付基线逐位相同 |
| 模型内 RMSNorm 段微基准（337 次调用/step，一次同步） | 参考 eager 链 **186.203 ms/step** → `rms_norm_compat`（拆分形态）**128.146 ms/step** → `rms_norm`（整链单 kernel）**19.810 ms/step** |
| `weight_stream_gemv` 有效带宽（自建真实 shape，3 轮 × 5 次轮转中位，GB/s） | gate_up 30720×3840 `75.6 → 124.1`；down 3840×15360 `30.4 → 123.1`；q_sliding `62.2 → 104.6`；k_sliding `51.4 → 87.8`；o_sliding `31.3 → 102.7`；q_global `67.7 → 112.6`；o_global `31.7 → 113.0`；k_global `25.4 → 39.8` |

### 已知限制

1. 组委会标准测例与精度达标线**尚未下发**，CP2 未勾选，本文件的精度部分是**临时回归通过**。
2. 官方 CI 脚本 `tools/ci_pipline/run_ci.py` 用 `re.search(r"vllm\s+serve\s+(\S+)")` 从 `run.sh`
   **字面**读取模型路径（`run_ci.py:50`），并作为 `speed.sh --tokenizer-path`（`run_ci.py:196`）。
   因此 `run.sh` 中保留了一个**不执行**的字面默认路径（`if false; then vllm serve /gpfs/model/...; fi`）：
   若把该位置写成变量，解析器会抓到字符串 `"$GEMMA_PATH"` 而使官方性能步骤失败。权重迁址时请让该
   字面路径可解析。
3. D512 global decode 的官方隔离 kernel 为可选组件（`GEMMA_D512_OFFICIAL_EXTENSION`），
   默认走 math。**滑动窗口在默认档由 overlay 的 math 路径施加**（prefill 与 decode 都施加，
   见**交付实际执行的** `runtime/custom_ops/prefill_attention/op.py` 的
   `_attend_math`/`sdpa_decode_attention_math` 与单测
   `model_adaptations/Gemma4SCUdoudui/scripts/gemma4/test_operator_overlay.py`（交付子树内那份，与仓库根同名文件 blob 逐位相同）的 T7.0/T9）；无窗口的 `sdpa_prefill_attention` 在 Gemma 上不可达（两个 head_dim 都 ≥ 256）。
   **路径口径（2026-10-13 实测更正）**：`run.sh` 的 `PYTHONPATH` 只前置交付子树的
   `overlay/` 与 `runtime/`，`custom_ops` 从 `runtime/custom_ops/` 导入；仓库根的
   `custom_ops/` 被 git 跟踪但**不在交付子树内**，且**没有 `__init__.py`**；交付子树里只有
   `runtime/custom_ops/`。机制：**regular package（有 `__init__.py`）胜过 namespace
   package，与 `sys.path` 顺序无关**（把仓库根放在最前也一样）⇒ 交付执行的是
   `runtime/custom_ops/`，**根副本不执行**。此后一切指向仓库根 `custom_ops/**` 的引用都不代表交付行为。
4. 多模态（vision/audio）为 No-Go：该 checkpoint 只有 `vision_embedder.*` 而没有与 vLLM
   `vision_tower.*` 对齐的塔权重，overlay 无法补出权重。
5. 当前默认档的 TTFT/TPOT 未重测；T3/T4、graph capture、当前 owner CI 未执行。
6. `run.sh` 的模型路径参数是 `"$GEMMA_PATH"`，但官方解析器需要字面量：若权重迁移，请让
   `/gpfs/model` 可解析（见第 2 条）。
7. **许可证口径**：本队对**本交付目录**的改动以 **Apache-2.0** 提交；仓库根 `LICENSE` 是**上游继承的 BSD-3**
   （blob 与上游同值，**不改**）。该差异**待向组委会确认**。
8. 本文件已按"手动执行新 README 的命令"逐条核对，并据此更正了三处：`import tecoops` 必须让
   `/opt/tecoai/lib64` 在 `LD_LIBRARY_PATH` 上；`tests/test_startup_failure.py` 需 SDAA 栈、
   不是无卡用例；overlay 的启动期失败原先只打印诊断（`raise` 被 `site.py` 吞掉），已改为
   `raise SystemExit(1)` 才真正 fail-closed。
9. **交付态"跑的是哪个库"实测（2026-10-13，设备无关）**：交付包不含 `tecoops`，
   `run.sh` 只前置 `overlay:runtime` ⇒ `import tecoops` 落到 `/home/py312/.../site-packages/tecoops`
   （lib sha256 `388dd710a8357ee9294ccada17fa3f4003a9bc5aa107bec744184fa008cbbcc0`，278 176 B）。
   该库 8 个导出，**没有** `rms_norm_compat` / `ms_deform_attn_*` / `rotary_embedding`，且
   `teco_slave_flash_attention_half_d256` / `_d512` 符号计数 = 0（交付库 = 7 / 1）。
   在交付 `run.sh` 的 `PYTHONPATH` 下 overlay 实际行为 = **静默降级并进入 main（rc=0）**，
   不是 fail-closed。反面对照：把交付安装目录前置后同一脚本 `SELFCHECK PASS` / rc=0。
   ⇒ 启动前必须跑 `model_adaptations/Gemma4SCUdoudui/scripts/gemma4/selfcheck_tecoops.py`（本文件「模型注册」第 2 条已写为必需步骤）。
10. **已知限制（2026-10-14，交付库 `e4e691d` 测得）**：`e4e691d` 把 `flash_attention` 的 `<128>` 实例化移入**尾部新 TU**，使整库 device 代码布局重排，**`rms_norm` 的 8 个官方 case 因 code-layout 效应变慢**（逐例 `+0.35…+2.6 µs`，合计 `+11.88 µs`；同构建 A/B，`max_error` 逐位不变、`rms_norm_fp16.scpp` 源码 sha256 未变）。该效应**与算子源码无关**，且**换任何结构都只会把税转移给另一个算子**（例如把新 TU 移进 `flash_attention/` 目录会让 `rms_norm` 归零但使 `weight_stream_gemv` 慢 `−35%/−18%`）。本队选择维持 `e4e691d` 的 device 布局（其后 `beebae2` 只改 host 谓词，全部 device 目标码逐字节不变），理由是净账：`flash_attention` 官方 11 例表 `−1.58367 ms`，而 `weight_stream_gemv` 正是 Hy/InternVL 两条线模型级收益所绑定的算子。逐 case 原值与完整 A/B 见本队账本仓 `docs/competition/c25-lead-ledger-20261014.md` §8.4 与 §9.4。

---

**cycle-31 pin 说明（2026-10-16）**：本交付的 ops pin 由 `beebae220f0e918460d2210e4064030e157d602d` 前移至 **`8ef0a5f891603ddc8011cc3c45b127fe988024bf`**（本队 teco-ops main，含 cycle-31 落地的 `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写、`8ef0a5f` weight_stream_gemv ILP4、`f092863` flash_attention fail-closed 对、`973cbd2` weight_stream_gemv SPM fail-loud、`6d0bde7` 测试面 mock 健壮性）。组合树门禁：单次调用（`--perf_repeat=0 --warm_repeat=0`）`max_error=0`、`python_api_test` focused ALL PASSED、官方全 39 例 `PASSED=39 FAILED=0`（逐例 `max_error` 仅 `weight_stream_gemv` 两例按归约序变化）。旧 pin `beebae220f0e918460d2210e4064030e157d602d` 仍可复现旧库。
**cycle-32 pin 说明（2026-10-16）**：本交付的 ops pin 由 `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 前移至 **`0371909a87d5dd11bb928b487afbf6e1e25eb44c`**（本队 teco-ops main，新增 `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步：把 KV tile 内 V 面板的 strided 打包由逐面板同步 `memcpy_stride` 改为 `memcpy_async` 一板预取 + 双 pane ping-pong（两块 pane 落在既有 `ktmp` 内，SPM 占用不变），末板前视按 `MIN(hb + 1, npanel - 1)` 夹紧以免构造零长描述符，循环体保持无实现选择分支。算子级（官方命令形态 `./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3`，同一 build root、仅替换 `libteco_ops.so`）：`flash_attention case_2` 1.829900 → 1.804300 ms（**+1.3990% 加速**，即时间 −1.3990%；落地前复核 1.8300 → 1.8042 ms = +1.410%）；D128 家族 +1.21%/+1.30%/+1.41%、D64 +0.42%、`case_5` +2.48%，未改动 TU 的 D256/D512 家族内对照 −0.037%/+0.118%；11/11 `max_error` 与基线逐位相同（×3 rep），单次调用门禁（`--perf_repeat=0 --warm_repeat=0`）`case_2`/`case_0` = `1.83614e-04`/`1.78913e-04` 且与基线逐位相同。**模型级无可测加速**：MiniCPM5-1B 交付 `run.sh` 下 16 臂每请求 32/32 token 与冻结基线逐位一致、errors=0、无 CPU fallback/OOM、峰值显存与 KV cache 逐位相同，稳态 420.02 s / 344 请求 / 0 错误 / 344-344 逐位一致；墙钟**低于可测门**（pooled +0.464%，A1 自身臂间中位跨度 5.37%）⇒ 不主张模型级加速（亦不构成回退证据）。动态探针 `tecoops.flash_attn_varlen_func = 48`（24 层 × 2 请求）证明该 kernel 真在 MiniCPM prefill 路径执行。V-B（给 `M128_N32` 的 B 侧一个真实 stride、整行消费 V）= **NO-GO**：`M128_N32` 不 honor B stride。旧 pin `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 仍可复现旧库。 **本线可达性**：本线两个注意力 head_dim 为 256/512，`findFlashAttentionBranch()` 走 algo 1/2，派发层即不进入本次改动的 D128 尾部 TU ⇒ 本次改动对本线不可达（本线未独立复跑）。
**cycle-36 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0371909a87d5dd11bb928b487afbf6e1e25eb44c` 前移至 **`2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a`**（三个原子提交：`9f11b33` `weight_stream_gemv` 激活行 fp32 化（官方 case_0/case_1 **−4.55%/−8.28%**）、`0132061` `reshape_and_cache` 跨 token 载入流水线（**−22.4%/−27.5%/−23.2%**）、`2a1716d` `weight_stream_gemv` 尾部冗余 wait 删除（**−2.4%/−3.0%**）；全部为官方固定形态**同窗口同 build root 只换 `.so`** 的 A/B、3 次原值取中位数、`max_error` **逐位不变**、落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §4.2`）。
**cycle-36 round-5 pin 说明（2026-10-17）**：本交付的 ops pin 由 `2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a` 前移至 **`0d6bd5a10eddde4c020bb2367a233c72e005ad65`**（`perf(flash_attention): emit P straight from the softmax pass instead of transposing it` —— ALGO0/D128 每个面板的 `P^T→P` 转置融进 softmax 第二趟：寄存器直写 `p_score`、同一舍入；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：case_0/8/9/10 **−16.1%**、case_1 −9.7%、case_2/3 −7.5%、case_4 −8.5%、**十一例加权 −10.7%**；`max_error` **逐位不变**；**11/11 PASSED**；已知权衡：case_5 **+10.7%**（0.041353→0.045774 ms）。本线可达性：本线 head_dim 256/512，`findFlashAttentionBranch()` 走 algo 1/2，派发层不进入本次改动的 D128 尾部 TU ⇒ 本次改动对本线不可达（本线未独立复跑）。证据路径 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §6`。
**cycle-36 round-14 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0d6bd5a10eddde4c020bb2367a233c72e005ad65` 前移至 **`7ea7e84c8aef327d522d3eb318bc4808b15d070c`**（`7ea7e84` `perf(flash_attention): skip the softmax exp on causally-masked score vectors`；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：vs base `0371909` case_0/8/9/10 **−18.32%**、case_1 −9.50%、case_2/3 −6.05%/−6.08%、case_4 −6.85%；**vs 前一交付态 `0d6bd5a`** case_0/8/9/10 **−4.41%**、flash 算子再降 **−1.5%**；`max_error` **逐位不变**、**11/11 PASSED**；证据 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §13/§14`）。
**cycle-37 pin 说明（2026-10-18）**：本交付的 ops pin 由 `7ea7e84c8aef327d522d3eb318bc4808b15d070c` 前移至 **`785a7848169307ea531b52107881b36277737332`**（三个原子提交：`37f89d1` flash_attention D256/ALGO1 的 Q-block `parts` 分裂（官方 `case_6` 同窗 A-B **−14.79% / −14.81%**，两窗，`max_error` 逐位不变）；`542a2a1` reshape_and_cache 的 PUT 槽深两迭代（case_0 −1.33%/−0.89%、case_1 **−7.02%/−6.85%**、case_2 **−7.82%/−7.90%**，两窗，两臂 `max_error` 全 0）；`785a784` 构建期把 weight_stream_gemv 设备段排在 rotary_embedding 之前（官方 case_0 **−37.0%**、case_1 **−25.4%**，i-cache miss 14/3776 ns → 4/38 ns）。落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c37/lead/`）。
**cycle-38 pin 说明（2026-10-18）**：本交付的 ops pin 由 `785a7848169307ea531b52107881b36277737332` 前移至 **`f748d0abc93ab813f6e35feed261795ba850bd3b`**（两个原子提交：`3f98d68` flash_attention ALGO0/D128 的 Q-block 改用入口期 LPT 表分配（两窗复现：`case_2` −30.30%、`case_3` −30.29%、`case_4` −30.82%，flash 集合加权 **−13.6%**；如实披露 `case_0/8/9/10` +1.6%、`case_1` +1.74%、`case_5` +7.35% 的权衡）；`f748d0a` reshape_and_cache 暂存环 8 组（`case_0` −1.8%、`case_1` −5.7%、`case_2` −4.8%，两臂 `max_error` 全 0）。落地库 **39/39 PASSED**。已知待修：`rotary_embedding` 自 `785a784` 起处于慢地址（+133%，0.0017→0.0040 ms），由段序守卫处理。
**cycle-38 round-2 pin 说明（2026-10-18）**：本交付的 ops pin 由 `f748d0abc93ab813f6e35feed261795ba850bd3b` 前移至 **`9fda597f7d8bc5f98893081372a552f411f26e42`**（`9fda597`：flash_attention D256/ALGO1 调度单元 `UQ = BM/2 = 32` 的切分修正；官方固定形态同窗同 build root 只换 `.so`：**`case_6` 0.183760 → 0.116800 ms（−36.44%）**、`case_7` 0.151690 → 0.149830（−1.23%），其余 9 例 −0.1~−0.3% 无回退；`max_error` 逐位不变，**11/11 PASSED、39/39 PASSED**）。
