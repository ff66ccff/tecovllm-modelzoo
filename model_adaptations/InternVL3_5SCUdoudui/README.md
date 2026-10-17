# InternVL3_5-8B

## 模型概述

InternVL3.5-8B（`OpenGVLab/InternVL3_5-8B`）是上海 AI Lab InternVL3.5 系列的 8B 多模态模型：InternViT 视觉编码器 + Qwen3-8B 语言塔，支持图像-文本交错输入与中英文对话。组委会给出的适配难度系数为 **0.7**。

本目录是**四川大学"都队"**在 TecoVLLM（SDAA 后端）上对该模型的适配交付：在厂商 Python + Torch-SDAA 下以 FP16、TP=2 启动 OpenAI 兼容推理服务，支持文本与单图输入。

适配内容：

1. **跑通**：TecoVLLM/vLLM 在 SDAA 上加载官方权重完成确定性贪婪生成，无 CPU fallback；
2. **进程初始化期 overlay**（`runtime/overlay/sitecustomize.py`）：把 Qwen3 语言塔的 `RMSNorm` 绑定到官方 `tecoops.rms_norm`，并替换 SDAA 上未实现的 `BlockAttentionImpl.forward`；绑定在 `site` 初始化阶段完成，forward 热路径无分支、无 `getenv`，缺 `tecoops` 即非零退出（fail-closed）；
3. **本队 tecoops C++ 算子接入**（三层形态：teco-ops 仓的 C++ kernel + Python API + 本模型调用点）：`rms_norm`、`reshape_and_cache`、`flash_attention`；
4. **默认档（D11）**：prefill 注意力默认走**可计入**的官方 C++ varlen ABI（`tecoops.flash_attn_varlen_func`）；torch SDPA 路径仅作回退档，**不计入**交付主张。切换 commit `3d62813`（根与交付副本同 blob `943084a0…`）；
5. **权重只读**：只读引用组委会预置权重，不下载、不转储副本；
6. **M=1 decode 线性层绑定**（本队 `tecoops.linear_stream` = `weight_stream_gemv`，上游 PR44）：进程初始化期把 `vllm.model_executor.layers.utils.default_unquantized_gemm` 替换为一个 opaque `torch.library.custom_op`；**M==1 走整行权重 slab 双缓冲 DMA，M>1 与不满足 `n%64==0 且 K%16==0` 的形状由 C++ host 回落厂商 GEMM**；默认开启、**无需环境变量**，热路径零分支零 `getenv`（AGENTS 规则 6），缺符号即 fail-closed。注：`N%256==0` → `n%64==0` 的放宽由 PR #44 落地（host 侧准入谓词），依据是引擎 DMA 校验器只要求 4 B 对齐。

本包是**纯 Python 适配层**，不含二进制；C++ kernel 全部来自本队 teco-ops 仓。

## 模型权重下载

本适配**不下载、不复制**模型权重，直接只读引用组委会预置目录（硬约束 1）：

```bash
export MODEL_ROOT=/gpfs/model                 # 预置权重父目录（可用 MODEL_ROOT 覆盖，run.sh 会真正生效）
ls "${MODEL_ROOT}/OpenGVLab/InternVL3_5-8B"   # 只读查看
```

- 实际路径：`${MODEL_ROOT}/OpenGVLab/InternVL3_5-8B`（默认 `/gpfs/model/OpenGVLab/InternVL3_5-8B`）。
- 权重不在默认位置时，用 `MODEL_ROOT=<父目录>` 或 `INTERNVL_MODEL=<完整模型目录>` 覆盖；**请勿**把权重拷入本目录。
- `run.sh` 已置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，运行期不联网拉取。

## 模型注册

本包无需编译：所谓"注册"= 让服务进程同时能导入 **`tecoops`**（本队算子库）与 **`custom_ops`**（本适配的注意力/RMSNorm/RoPE 绑定）。

依赖：

- 厂商 Python：`/home/py312/bin/python`（`readlink -f` → `/usr/local/python/bin/python3.12`）、`/opt/tecoai/setvars.sh`；
- 本队算子库 `tecoops`：由 teco-ops 仓（含团队 PR37 `rms_norm`、PR#35 `reshape_and_cache` 等）构建。**隔离安装到全局 site-packages 之外**，并把该目录前插 `PYTHONPATH`（不写全局 site-packages）。

```bash
source /opt/tecoai/setvars.sh
export PYTHON=/home/py312/bin/python
ZOO=<模型仓根>                                   # 本包所在仓库的根目录
ADAPT="$ZOO/model_adaptations/InternVL3_5SCUdoudui"
OPS_SRC=<teco-ops 检出目录>                       # 本队算子库源码（pin 到下面的 ops 源码 commit）
OPS_PREFIX=<交付算子库安装目录>                    # 例：$HOME/ivl-tecoops；不要写只在某台机器成立的绝对路径

# 1) 从本队 teco-ops 仓构建 wheel（teco-hal 等依赖见该仓 README）
git -C "$OPS_SRC" checkout 7ea7e84c8aef327d522d3eb318bc4808b15d070c   # 本队 ops main（pin 口径 = 源码 commit）
"$PYTHON" "$OPS_SRC/setup.py" bdist_wheel --dist-dir "$OPS_SRC/dist"

# 2) 隔离安装到独立目录（不覆盖全局 site-packages），并记录本轮 wheel 的 sha256
"$PYTHON" -m pip install --no-deps --target "$OPS_PREFIX" "$OPS_SRC"/dist/*.whl
sha256sum "$OPS_PREFIX"/tecoops/libteco_ops.so    # provenance：记下来随报告一起交

# 3) 前插隔离目录后启动；run.sh 会把本包 runtime/overlay 与 runtime/ 再前置
export PYTHONPATH="$OPS_PREFIX"
MODEL_ROOT=/gpfs/model bash "$ADAPT/run.sh"
```

> 本 pin 已前移至 **`7ea7e84c8aef327d522d3eb318bc4808b15d070c`**（自 `4adfd05e` 起依次落地：`79941c5f` ms_deform_attn forward plane → `d84f34f` flash_attention D256 `BM==64` → `54a1d65` rms_norm epilogue 二级展开 → `98e18ca` rotary_embedding CI 修复同步 → `e4e691d` flash_attention `BM==128` NT-form + 转置域 softmax → `beebae2` weight_stream_gemv host 谓词 `n % 256 == 0` → `n % 64 == 0`（按引擎 DMA 校验器实测的 4 B 要求），使 TP 分片的 lm_head 也走上 M=1 快路径 → `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写 → `8ef0a5f` weight_stream_gemv ILP4 → `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步（算子级 `case_2` 1.829900 → 1.804300 ms，+1.3990%；模型级无可测加速））。**照旧 pin（`4adfd05e`）复现得到的库不含上述改动**，故复现必须以 `7ea7e84c8aef327d522d3eb318bc4808b15d070c` 为准。

**第 3 步不是可选项。** 本包**不含** `tecoops`（只有 Python 适配层）。官方 CI（`tools/ci_pipline/run_ci.py`）**不做任何安装**（它只 `parse_run_sh()` → `bash run.sh` → `prec.sh`/`speed.sh`），安装是交付方责任。若不显式前插交付算子库，`import tecoops` 会静默解析到环境里**任意已安装的 wheel**，且**不报错**。已实测的后果（2026-10-12 本线走查，未前插交付库时）：落到旧 wheel `388dd710…`，图像 total 从 **6.25 s 退化到 14.18 s**（prefill 注意力 216.89 ms/call vs 交付库 14.229 ms/call，36 层折合 **+7.30 s**；与 Step-2 独立记录的旧 wheel ΔTTFT **+7.808 s** 一致）。

### 怎么判定"这次跑的是不是交付库"（三层，按顺序看）

1. **pin 口径 = 源码 commit + 本轮构建产物**：交付库 pin 在本队 teco-ops main **`7ea7e84c8aef327d522d3eb318bc4808b15d070c`**，并记录**本轮** `libteco_ops.so` 的 sha256。
   ⚠ **不得用 lib 字节做跨 build 判等**：lib 字节随 build root 变化（队内实测：同一份源码 `459992a2` 在不同 build root 产出过 `9b0746c7` / `93b75d34` / `a92ad51c` 三种字节）。sha256 只作 provenance 记录。
2. **静默降级的判据用 build-root 无关的能力探针**（不需要任何私有环境变量）：

   ```bash
   LIB="$OPS_PREFIX/tecoops/libteco_ops.so"
   strings -a "$LIB" | grep -c teco_slave_flash_attention_half_d256   # 交付库 = 7；旧 wheel 388dd710 = 0
   nm -D "$LIB"       | grep -c teco_slave_flash_attention_half_d256  # 交付库 = 1；旧 wheel = 0
   ```

   计数 **≥1** 即交付能力在；**0** 说明解析到了旧 wheel（静默降级）。实测前后：交付库 `7` / `1`，旧 wheel `0` / `0`。
3. **`run.sh` 启动时自动打印 provenance**（初始化期一次，零设备开销：只用 `importlib.util.find_spec` 定位路径 + 文件哈希 + 字符串探针，**不 import `tecoops`**、不进热路径、不做 sha 硬门禁）：

   ```text
   [run.sh] tecoops      = <OPS_PREFIX>/tecoops/__init__.py
   [run.sh] libteco_ops  = <OPS_PREFIX>/tecoops/libteco_ops.so
   [run.sh] lib sha256   = <本轮构建的 sha256>
   [run.sh] probe teco_slave_flash_attention_half_d256: count=7 (>=1 表示交付能力在；旧 wheel 388dd710 为 0)
   ```

解析自证（确认 `tecoops` 与 `custom_ops` 的来源；`custom_ops` 必须落在本包 `runtime/`）：

```bash
PYTHONPATH="$OPS_PREFIX:$ADAPT/runtime" "$PYTHON" - \
  "$ADAPT/runtime" <<'PY'
import importlib.util, sys
want_prefix = sys.argv[1].rstrip("/")
spec = importlib.util.find_spec("tecoops")            # find_spec 不 import 模块，零设备开销
print("tecoops    =", spec.origin if spec else "<not found>")
import custom_ops
print("custom_ops =", custom_ops.__file__)
assert custom_ops.__file__.startswith(want_prefix + "/custom_ops"), \
    "custom_ops 未解析到本包 runtime（CWD 影子导入？）"
print("OK: import 来源正确")
PY
```

若 `custom_ops` 解析到仓库根的副本，说明启动目录/PYTHONPATH 不对（本线出现过这类"影子导入"：CPython 会把当前工作目录放在 `sys.path` 之前，排查即上面这条断言）。

初始化期开关（在进程初始化时**只读一次**，forward 热路径零分支、零 `getenv`；非法值在 import 期直接失败）：

| 变量 | 默认 | 含义 |
| :--- | :--- | :--- |
| `INTERNVL_PREFILL_ATTN` | `official` | `official` = 可计入的官方 C++ flash varlen ABI；`sdpa` = 回退到 torch SDPA（**不计入**） |

解释器可用 `PYTHON=<解释器>` 覆盖（默认仍是厂商解释器）；**本线尚未在官方 py3.11 环境跑通该模型**，故不宣称官方环境适配完成。

## 启动推理服务

```bash
MODEL_ROOT=/gpfs/model bash model_adaptations/InternVL3_5SCUdoudui/run.sh
```

`run.sh` 先 `source /opt/tecoai/setvars.sh`，再用**厂商 Python** 启动 vLLM 的 OpenAI 兼容服务。serve 参数由 CI 固定，等同于：

| 参数 | 值 |
| :--- | :--- |
| `--model` | `${INTERNVL_MODEL:-${MODEL_ROOT}/OpenGVLab/InternVL3_5-8B}` |
| `--served-model-name` | `InternVL3_5-8B` |
| `--dtype` | `float16` |
| `--tensor-parallel-size` | `2`（`INTERNVL_TP_SIZE`） |
| `--gpu-memory-utilization` | `0.85`（`INTERNVL_UTIL`） |
| `--max-model-len` | `16640`（`INTERNVL_MAX_MODEL_LEN`） |
| `--limit-mm-per-prompt` | `{"image": 1}` |
| `--trust-remote-code` / `--no-enable-prefix-caching` / `--no-enable-chunked-prefill` | 固定开启 |
| `--host` / `--port` | `0.0.0.0` / `8002`（`INTERNVL_HOST` / `INTERNVL_PORT`） |

> `--max-model-len` 默认 **16640** 是**CI 硬要求**，不是性能取舍：组委会 `tools/ci_pipline/speed.sh` 固定跑
> `warmup 1024` + `T1 2048 / T2 4096 / T3 8192 / T4 16384`（`--max-tokens 100 --min-tokens 100 --ignore_eos`，
> `--parallel 1 --number 10`），而 `result_analyse.py::parse_performance()` 对**每个**非 warmup 用例强制
> `Failed == 0` 且 `Total == Success`（并要求 `Benchmarking summary` / `Percentile results` 存在），
> 且 `run_ci.py` 第 5 步在 `try/except` 之外调用它 ⇒ **任一用例被拒即整个 CI FAILED**。
> T4 需要 `16384 + 100 = 16484` 的上下文上限，故下限 **16484**；默认取 **16640** 留余量。
> 实测（2026-10-13，`--gpu-memory-utilization 0.85` 不变）：`GPU KV cache size 47,104 tokens`
> （对 16,640 有 2.83× 余量）、`teco-smi` 峰值 27,780 MB / 65,536 MB；四例 + warmup 全部 `Failed == 0`。

- `--no-enable-chunked-prefill` 是**必需**的：SDAA 上分块 prefill（`1 < q_len < kv_len`）的因果掩码路径会掉 math 回退。
- 设备可见性由 `SDAA_VISIBLE_DEVICES`（默认 `0,1`）决定；`VLLM_DISABLE_COMPILE_CACHE=1` 避免复用上一次算子绑定产生的图。
- 覆盖项一律用环境变量（`MODEL_ROOT`、`INTERNVL_MODEL`、`INTERNVL_HOST`、`INTERNVL_PORT`、`INTERNVL_TP_SIZE`、`INTERNVL_UTIL`、`INTERNVL_MAX_MODEL_LEN`、`PYTHON`），不改脚本内参数。
- **官方 CI 解析器契约**：`tools/ci_pipline/run_ci.py::parse_run_sh()` 会剥掉 `#` 注释行后取**文件里第一个** `vllm serve <路径>` 的**裸字面量**当 `model_path`（`speed.sh` 还拿它当 `--tokenizer-path`）。因此 `run.sh` 在真正执行行之前保留了一个**非注释行、不可达**（`if false; then … fi`）的 `vllm serve /gpfs/model/OpenGVLab/InternVL3_5-8B …`，其 `--served-model-name`/`--host`/`--port` 与执行行一致；真正执行的是 `vllm serve "$MODEL_PATH"`，故 `MODEL_ROOT`/`INTERNVL_MODEL` 仍然生效。
- 路径解析 + CI 契约自检（零卡、不需设备、不复制权重）：`bash model_adaptations/InternVL3_5SCUdoudui/tests/verify_runsh_model_root.sh`。它把解释器换成打印 argv 的桩观测 shim exec 出的最终 `--model`（默认 / `MODEL_ROOT=<别名>` / `INTERNVL_MODEL` 三种），并**直接导入官方 `parse_run_sh`** 断言 `model_path == "/gpfs/model/OpenGVLab/InternVL3_5-8B"` 且 `os.path.exists()` 为真。原先那行 `vllm serve /gpfs/model/...` 的字面量是 shim 改写前的**死文本**（latent/误导性硬编码，行为本就正确），清理见 commit `a5e82ae`（serve 参数逐字节不变）。

服务就绪后的最简冒烟：

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "InternVL3_5-8B",
  "messages": [{"role": "user", "content": "用一句短话描述这张图。"}],
  "max_tokens": 32, "temperature": 0.0
}' | head -c 400
```

## 精度验证

**状态：临时回归通过。** 组委会标准测例与精度达标线**尚未下发**，本节不作官方精度主张、不填任何分数。

判据与实测（固定 prompt、`temperature=0` 贪婪 32 token）：

| 模态 | 判据 | 结果 |
| :--- | :--- | :--- |
| 文本 | 与冻结基线**逐字符一致** | 一致；内容 sha256(16) = `9e3b639a19dfa124` |
| 图像 | 与冻结基线**逐字符一致** | 一致；内容 sha256(16) = `bb14926ca4041ccd` |

- **300 s 稳态**（attention 属高风险算子，按规则 4 追加）：`elapsed 301.34 s`、47 请求、**0 error**、无 OOM，峰值 `12.666 GiB`、斜率 `0.000000000 GiB/s`；窗口内首/中/尾各 ≥3 次抽样**全部逐位一致**；in-process 调用计数差分 `1692 = 36 层 × 47 请求`（两个 TP worker 均成立）。
- **默认档 A/B 门禁**（改档后重跑）：A/B 两臂各 3 轮、共 9 个 session × 3 次生成，图像/文本**跨臂且与冻结基线逐位一致**；0 error、无残留进程；B 臂官方 flash 调用收据 6 行、A 臂 0 行。
- 算子 focused 测试（本包 `tests/**` 可复跑）：RMSNorm overlay 13/13、reshape/cache 7/7、prefill attention 6/6、overlay fail-closed 4/4。这些是 focused 回归，**不等于**官方精度达标。
- **冻结基线（点名）**：定义与数值出处 = `experiments/internvl3_5-8b.md:1603`（2026-10-12 T1 行）
  与 `docs/competition/lead-delivery-ledger-c20-20261012.md` §4.2.2/§10.2（**本队账本仓，均不在本交付包内**）；
  冻结内容三值：text `9e3b639a19dfa124` / image `bb14926ca4041ccd` / 图像文件 `a77f6ec4…`（图像路径
  `/home/py312/lib/python3.12/site-packages/sklearn/datasets/images/flower.jpg`）。
- **逐位比较的原件**：`/root/tmpwork/c20-internvl/step2-t1/{A1,A2,B}-r{1,2,3}/step2.json`（9 session × 3 生成、
  跨臂逐位）与 `/root/tmpwork/c20-internvl/steady-300s/steady.json`（300 s 稳态、47 请求、
  `probe_counts` 差分 `1692`）+ `steady-300s/frozen-baseline.json`（基线三值）；均为**包外原件**。
- 证据路径：`validation/**`、`tests/**`（本包内）；`/root/tmpwork/c20-internvl/{steady-300s/,step2-d11/,RESULT-d11.md,RESULT-steady.md}`（本队私有加速器仓，**不在本包内**）。
- **未执行**：组委会标准测例、官方 CI（均未下发）。

## 性能验证

口径：同一 build、同一固定输入，3 次独立进程的原值 + 中位数。

### 模型级（固定文本/图像输入，贪婪 32 token）

| 档位 | 图像 total（s） | 图像 TTFT（s） | 文本 total（s） | decode/token（s） |
| :--- | :--- | :--- | :--- | :--- |
| **`official`（默认，可计入）** | **6.2591 / 6.2548 / 6.2548，中位 6.2548** | 2.6641 | 6.3710 | 0.19945 |
| `sdpa`（回退，**不计入**） | 5.7371 / 5.7431 / 5.7304，中位 5.7401 | 2.1493 | 6.4187 | 0.19946 |

- `official` 相对 `sdpa` 的图像 total 差为 **+8.97 %**，其性质是**口径切换，不是提速**：默认档从"**不计入**的 torch SDPA 路径"移到"**可计入**的官方 C++ kernel 路径"。代价**全部在 prefill TTFT**（+0.5147 s，即 in-situ prefill 注意力 **14.301 ms/call**），**decode 差 −0.04 %**（3.5901 / 3.5903 s），文本 −0.74 %。
- 跨窗口稳定性：`official` 的 6.2548 s 与上一轮同口径实测 6.2517 s 差 **+0.05 %**。
- 算子级 ≠ 模型级：官方 demo 的**自建**用例 op 级 AHT 20.371 ms 与 in-situ 14.301 ms/call 相差 **1.43×**，**未归因**，不得用 op 级数字外推模型级。

### 算子级（本队 teco-ops C++ kernel）

| 算子 | 证据 | 结论 |
| :--- | :--- | :--- |
| `rms_norm`（团队 PR37：current-stream 绑定 + SIMD epilogue + plain 输出 DMA 重叠） | `validation/rms_epilogue_simd_20261007.json`、`validation/rms_plain_writeback_20261007.json`、`validation/epilogue-only.patch`、`validation/plain-writeback-only.patch` | focused 32/32 输出位相等；TP2 长 q/k plain 调用较两个 A 臂低 **12.2–13.2 %**，hidden [1811/4352,4096] 约 **1.7 %** |
| `reshape_and_cache`（团队 PR#35） | `validation/ci_contract_20261005.json` | ABI 边界 materialize 非连续 fused-QKV；focused 7/7（native 6/6） |
| `flash_attention`（团队 ALGO0 causal KV-tile 上界） | 私有仓 `op_learning/attention/**`；**自建用例 case_11**（模型真实几何 1811×16/4×128） | AHT **20.371 ms**、`max_error 1.90589e-04`（阈值 `1.00e-02`）。**自建非官方用例，仅作定位，不作官方口径** |
| `linear_stream`（= `weight_stream_gemv`，M=1 decode 权重流式 GEMV；团队 PR44，本轮首次接入本模型） | 本队账本仓 `docs/competition/c25-lead-ledger-20261014.md` §3.2；远程原始证据 `/root/tmpwork/c25/internvl/GATE-REPORT.md` | 9/9 臂 greedy 文本 **32/32 token IDs 与冻结基线逐位相同**；`B/A1` 中位 **−50.09 %**（文本 t32）、decode 斜率 **−51.80 %**、图像请求 **−26.6 %**；`A2/A1` 漂移 +0.027 %；发射清册 `12672 = 88×144`/rank（144 = 36 层 × qkv/o/gate_up/down ⇒ 每个 decode 线性都真进自研 kernel）；307 s 稳态 `errors=0`、峰值显存与对照一致。口径 = 进程内 `vllm.LLM` + `enforce_eager=True`（产生冻结基线的同一配置），**不得与历史 HTTP 数字混用** |

- 另：`validation/ci_budget4352_20261005.json` 记录 4352 上下文容量与正确性复证（T2 4096-token prompt + 100-token 输出，20/20 请求成功）；仅作容量/正确性证据，不含提速主张。
- 本适配 serve 参数由 CI 固定，**未做**任何调参；因此上文数字不含调参收益。

---

**已知限制**：

1. 组委会测例/达标线未下发 ⇒ 精度只能表述为"**临时回归通过**"；
2. 官方 py3.11 环境尚未跑通本模型，全部模型级数字来自厂商 py3.12 + Torch-SDAA；
3. 算子级与模型级的 **1.43× 差已归因**（2026-10-13，`docs/competition/c22-internvl.md` §1/§3.5，本队账本仓）：机制 = 逐层因果 KV 深度差异（`case_11` 取最深一层），链长比 **≈1.43（工作量加权有效口径）/ 1.62（字面口径）**、**残差 ~2% 量级未进一步拆解**；口径与原件见该文。**本条替代原文「未归因」表述**；
4. `−55.4 %` 一类图像收益来自**不计入**的 torch SDPA 路径，**一律不作交付主张**；
5. 官方 kernelbench-sdaa 全量评测未执行（测例未下发）。
6. **出处头覆盖不全（已知差距）**：交付包 26 个代码文件中仅 **5** 个带 BSD 出处头，其余 **21** 个（`run.sh`、`runtime/custom_ops/**`、`tests/**` 等）未带；本轮**未补齐** —— 补齐须**同时**改根副本与交付副本以避免新的 L11 不一致，并需重跑语法检查与 focused 回归，故登记为已知差距（对标：Gemma 25/31、Hy 17/18、MiniCPM 23/27）。
7. **许可字面差异（待组委会确认）**：见下节「许可口径」。
8. **M=1 decode 线性层绑定的已知限制**（2026-10-14 cycle-25 新增）：① **已于 cycle-26（pin `beebae2`）解决**：`fb_align` 桶原为 `92`/rank，来源是 M=1 的 **lm_head**（vLLM 把 LM head 切成 `151936/2 = 75968` 行、`75968 % 256 = 192`，结构性不满足旧的 `n % 256 == 0` 谓词）；把 host 谓词放宽到 **`n % 64 == 0`**（依据引擎自身的 DMA 校验器：它与 `teco_hal_tile.hpp` 都只要求 **4 B** 对齐；旧的「16 B aligned」注释不成立）后，lm_head 走上 M=1 快路径：同窗清册 `fb_align 92 → 0`、`kernel_launches 12672 → 12764`（增量恰等于 lm_head 调用数），模型级文本 32-token 中位 `3.1747 → 2.8005 s = +11.787%`（9/9 臂 32/32 token 逐位一致、305.94 s 稳态 `errors=0`、峰值显存与对照逐位相同）；② 交付库 `libteco_ops.so`（`98e18ca`）**不含** `linear_stream_stats` 计数器，发射清册是用「同源码 + 仅加计数器」的第二份构建（`api/torch_ext.cpp` 只多计数器与一个 `m.def`，M==1 谓词逐字未动）取得的；③ 快路径要求激活与权重同为 fp16，**改 bf16 会静默回落厂商**（算术等价、无加速、无可观测计数器）；④ 未重跑组委会标准精度测例 / 端到端 `run_ci.py`（改动只触 decode 线性调用点，正确性论据 = 冻结 greedy-32 逐位 + 307 s 稳态 `errors=0`）。
9. **已知限制（2026-10-14，交付库 `e4e691d` 测得）**：`e4e691d` 把 `flash_attention` 的 `<128>` 实例化移入**尾部新 TU**，使整库 device 代码布局重排，**`rms_norm` 的 8 个官方 case 因 code-layout 效应变慢**（逐例 `+0.35…+2.6 µs`，合计 `+11.88 µs`；同构建 A/B，`max_error` 逐位不变、`rms_norm_fp16.scpp` 源码 sha256 未变）。该效应**与算子源码无关**，且**换任何结构都只会把税转移给另一个算子**（例如把新 TU 移进 `flash_attention/` 目录会让 `rms_norm` 归零但使 `weight_stream_gemv` 慢 `−35%/−18%`）。本队选择维持 `e4e691d` 的 device 布局（其后 `beebae2` 只改 host 谓词，全部 device 目标码逐字节不变），理由是净账：`flash_attention` 官方 11 例表 `−1.58367 ms`，而 `weight_stream_gemv` 正是 Hy/InternVL 两条线模型级收益所绑定的算子。逐 case 原值与完整 A/B 见本队账本仓 `docs/competition/c25-lead-ledger-20261014.md` §8.4 与 §9.4。

10. **M=1 快路径的准入面（2026-10-14 cycle-26，交付库 `beebae2`）**：放宽后的谓词只可能触及**此前回落厂商**的 shape。官方 `weight_stream_gemv` 两例是 `N = 256/512`（本就 `% 256 == 0`）⇒ **算子级无收益**（`−0.149% / +0.163%`，低于 0.4% 地板；24/24 DIFF1 PASSED 且各臂 `max_error` 逐位不变），收益是**模型级**的。本队另外三条线该谓词**结构不可达**：Hy-MT2-1.8B 每 rank lm_head `N = 120818`(TP1)/`60409`(TP2) 不是 64 的倍数；Gemma-4-12B 每 rank `N = 131072` 本就是 256 的倍数；MiniCPM5-1B 不绑定该算子。**本次未跑**：组委会标准精度测例 / 端到端 `run_ci.py`、HTTP `serve_internvl.sh` 口径、非 eager/编译路径、bf16（仍静默回落厂商）、`N = 75968` 的独立算子级 AHT（官方 case zoo 无该 shape，故以模型级墙钟 + 逐调用清册为证据）。**Lead 独立验收**：本 pin 落地后全表 39 例 **39/39 ALL PASSED**（39 例 `max_error` 与 `e4e691d` 基线逐位相同）。**夹具缺陷已于 2026-10-14 修复**：此前记录的 `causal_conv1d case_0/case_1` `TECOTEST_STATUS_FILE_OPEN_FAULT` 根因是 **demo 内置 python3.11 的 torch↔numpy 桥在本机损坏**（`Failed to initialize NumPy: module 'numpy._globals' has no attribute '_signature_descriptor'`），demo 因此永远生成不了缺失的 `cpu_output_*`；而它走到生成分支时的路径推导（`test/src/suite/executor.cpp:1575` 的 `getcwd() + "../zoo/teco/<op>"`）在「从 `<ops>/test` 跑 demo」下又解析到不存在的 `<ops>/zoo/...`。修法（不改任何受保护文件）：先用**厂商 python 3.12** 跑该 op 自带的生成器预生成参考 —— `TREE=<tree> OP=causal_conv1d bash /root/tmpwork/c16/gen_cpu_ref_all.sh` —— 并把生成的 `cpu_output_*` 放进参考银行，供 `seed_refs.sh` 在所有树上复用。修复后该两例 `max_error 0.00000e+00`、AHT 12.42 / 24.249 ms。**此前 37 PASSED 的表述随之作废。**

---

## 许可口径

本队对 InternVL3_5-8B 适配的全部改动以 **Apache-2.0** 提交；仓库根 `LICENSE`（以及队内
`teco-ops` / `tecovllm` / `teco-modelzoo` 三仓根的 `LICENSE`）为**上游 fork 继承的 BSD-3-Clause**，
本队**未改动任何许可证文件**。该字面差异（队内清单写「Apache 2.0 或 MIT」）**待向组委会确认**。

---

**cycle-31 pin 说明（2026-10-16）**：本交付的 ops pin 由 `beebae220f0e918460d2210e4064030e157d602d` 前移至 **`8ef0a5f891603ddc8011cc3c45b127fe988024bf`**（本队 teco-ops main，含 cycle-31 落地的 `1145625` causal_conv1d 32-SPE×16-lane SIMD 重写、`8ef0a5f` weight_stream_gemv ILP4、`f092863` flash_attention fail-closed 对、`973cbd2` weight_stream_gemv SPM fail-loud、`6d0bde7` 测试面 mock 健壮性）。组合树门禁：单次调用（`--perf_repeat=0 --warm_repeat=0`）`max_error=0`、`python_api_test` focused ALL PASSED、官方全 39 例 `PASSED=39 FAILED=0`（逐例 `max_error` 仅 `weight_stream_gemv` 两例按归约序变化）。旧 pin `beebae220f0e918460d2210e4064030e157d602d` 仍可复现旧库。
**cycle-32 pin 说明（2026-10-16）**：本交付的 ops pin 由 `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 前移至 **`0371909a87d5dd11bb928b487afbf6e1e25eb44c`**（本队 teco-ops main，新增 `0371909` flash_attention ALGO0（head_dim 128/64）V 面板打包异步：把 KV tile 内 V 面板的 strided 打包由逐面板同步 `memcpy_stride` 改为 `memcpy_async` 一板预取 + 双 pane ping-pong（两块 pane 落在既有 `ktmp` 内，SPM 占用不变），末板前视按 `MIN(hb + 1, npanel - 1)` 夹紧以免构造零长描述符，循环体保持无实现选择分支。算子级（官方命令形态 `./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3`，同一 build root、仅替换 `libteco_ops.so`）：`flash_attention case_2` 1.829900 → 1.804300 ms（**+1.3990% 加速**，即时间 −1.3990%；落地前复核 1.8300 → 1.8042 ms = +1.410%）；D128 家族 +1.21%/+1.30%/+1.41%、D64 +0.42%、`case_5` +2.48%，未改动 TU 的 D256/D512 家族内对照 −0.037%/+0.118%；11/11 `max_error` 与基线逐位相同（×3 rep），单次调用门禁（`--perf_repeat=0 --warm_repeat=0`）`case_2`/`case_0` = `1.83614e-04`/`1.78913e-04` 且与基线逐位相同。**模型级无可测加速**：MiniCPM5-1B 交付 `run.sh` 下 16 臂每请求 32/32 token 与冻结基线逐位一致、errors=0、无 CPU fallback/OOM、峰值显存与 KV cache 逐位相同，稳态 420.02 s / 344 请求 / 0 错误 / 344-344 逐位一致；墙钟**低于可测门**（pooled +0.464%，A1 自身臂间中位跨度 5.37%）⇒ 不主张模型级加速（亦不构成回退证据）。动态探针 `tecoops.flash_attn_varlen_func = 48`（24 层 × 2 请求）证明该 kernel 真在 MiniCPM prefill 路径执行。V-B（给 `M128_N32` 的 B 侧一个真实 stride、整行消费 V）= **NO-GO**：`M128_N32` 不 honor B stride。旧 pin `8ef0a5f891603ddc8011cc3c45b127fe988024bf` 仍可复现旧库。 **本线可达性**：本线默认档 `INTERNVL_PREFILL_ATTN=official` 的 prefill 走 `tecoops.flash_attn_varlen_func`，几何为 D128 ⇒ 经过 ALGO0（本次被改 TU）；本线未独立复跑本次改动。
**cycle-36 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0371909a87d5dd11bb928b487afbf6e1e25eb44c` 前移至 **`2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a`**（三个原子提交：`9f11b33` `weight_stream_gemv` 激活行 fp32 化（官方 case_0/case_1 **−4.55%/−8.28%**）、`0132061` `reshape_and_cache` 跨 token 载入流水线（**−22.4%/−27.5%/−23.2%**）、`2a1716d` `weight_stream_gemv` 尾部冗余 wait 删除（**−2.4%/−3.0%**）；全部为官方固定形态**同窗口同 build root 只换 `.so`** 的 A/B、3 次原值取中位数、`max_error` **逐位不变**、落地库 **39/39 PASSED**；原始件 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §4.2`）。
**cycle-36 round-5 pin 说明（2026-10-17）**：本交付的 ops pin 由 `2a1716d7102d15e0ec9ce4aa40438dccbc84cb3a` 前移至 **`0d6bd5a10eddde4c020bb2367a233c72e005ad65`**（`perf(flash_attention): emit P straight from the softmax pass instead of transposing it` —— ALGO0/D128 每个面板的 `P^T→P` 转置融进 softmax 第二趟：寄存器直写 `p_score`、同一舍入；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：case_0/8/9/10 **−16.1%**、case_1 −9.7%、case_2/3 −7.5%、case_4 −8.5%、**十一例加权 −10.7%**；`max_error` **逐位不变**；**11/11 PASSED**；已知权衡：case_5 **+10.7%**（0.041353→0.045774 ms）。本线可达性：本线默认档 `INTERNVL_PREFILL_ATTN=official` 的 prefill 走 `tecoops.flash_attn_varlen_func`，几何为 D128 ⇒ 经过 ALGO0（本次被改 TU）；本线未独立复跑本次改动。证据路径 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §6`。
**cycle-36 round-14 pin 说明（2026-10-17）**：本交付的 ops pin 由 `0d6bd5a10eddde4c020bb2367a233c72e005ad65` 前移至 **`7ea7e84c8aef327d522d3eb318bc4808b15d070c`**（`7ea7e84` `perf(flash_attention): skip the softmax exp on causally-masked score vectors`；官方固定形态**同窗同 build root 只换 `.so`**、3 次中位数、**两次独立窗口逐例复现**：vs base `0371909` case_0/8/9/10 **−18.32%**、case_1 −9.50%、case_2/3 −6.05%/−6.08%、case_4 −6.85%；**vs 前一交付态 `0d6bd5a`** case_0/8/9/10 **−4.41%**、flash 算子再降 **−1.5%**；`max_error` **逐位不变**、**11/11 PASSED**；证据 `/root/tmpwork/c36/lead/LEDGER-DRAFT.md §13/§14`）。
