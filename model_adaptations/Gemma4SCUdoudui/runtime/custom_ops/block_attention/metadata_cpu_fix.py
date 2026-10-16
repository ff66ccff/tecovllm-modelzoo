"""把 CommonAttentionMetadata 的 CPU 孪生张量转发到 BlockAttentionMetadata。

交付形态（model_adaptations/Gemma4SCUdoudui）专用的一代文件，与仓库根那一代的
``custom_ops/block_attention/metadata_cpu_fix.py`` 同设计、不同落点。

## 问题（c33/v1 实测根因）

``runtime/custom_ops/prefill_attention/op.py`` 在**每层**的 per-sequence 循环里做
``int(query_start_loc[i])`` / ``int(query_start_loc[i+1])`` / ``int(seq_lens[i])``
（本代共 12 处行 / 14 个调用点，含 math 入口的端点校验）。设备张量上的 ``int()``
即 ``aten::_local_scalar_dense``，强制 device→host 队列排空。

交付版实测（T1575，1 步 prefill，2 卡 TP）：
  * D2H 次数 ``sdaaMemcpyAsyncD2H`` = 145（= 3 × 48 层 + 1）
  * ``d2h_cpu_self_ms`` = 1606.31，占 span 91.84 %

v1（每层 3 次读合并成 1 次整块 ``.cpu().tolist()``）：
  * D2H 145 → 49、标量读 523 → 235、设备侧 DtoH 9.92 → 3.98 ms，32-token 逐位一致
  * **但** ``d2h_cpu_self_ms`` 1606.31 → 1580.92（−1.6 %），wall gate 8.900 % → 8.807 %
  ⇒ 剩余 49 次调用每次 32.3 ms，是**固定的队列排空代价**：
    少问几次没用，必须让算子**不再需要**这个值（v1 判 NO-GO）。

## 修法：值本来就在 host

``gpu_model_runner._prepare_inputs`` 用 ``CpuGpuBuffer.cpu``（pinned）与
``optimistic_seq_lens_cpu``（pinned）构造 ``CommonAttentionMetadata``，其
``query_start_loc_cpu``（``vllm/v1/attention/backend.py:362``）与 ``_seq_lens_cpu``
（同文件 L413）就是 host 副本，**构造过程零 D2H**。
``vllm_sdaa/attention/block_attn.py:488/490`` 的 ``build()`` 已把它们取成局部变量，
却在唯一的构造点（L568-595）**不转发**给 ``BlockAttentionMetadata``。

本模块在 **build 期**（每步每 KV-cache-group 一次，不在算子热路径）把这两个 host
值 ``.clone()`` 冻结后挂到 ``BlockAttentionMetadata`` 实例上（该 dataclass 无
``__slots__``，可动态加属性），由 ``runtime/custom_ops/block_attention/op.py``
传给注意力算子。算子循环体内不再出现任何 ``int(device_tensor)``。

**残留（必须如实标注）**：``prefill_attention/op.py`` 的 math 入口仍有一处页号校验
``bool(torch.any(required < ...))`` / ``bool(torch.any(required >= key_cache.shape[0]))``，
其 ``required`` 来自**设备** ``block_table``，每请求 2 次 ``_local_scalar_dense``。
本代 ``CommonAttentionMetadata`` 无 ``block_table_cpu`` 字段、``BlockTable.get_cpu_tensor()``
零调用方 ⇒ **无 host 孪生可喂**，故保留该校验以不放宽 ``invalid historical page ID`` 语义。
残留占比见 RESULT.md（窗口静态预算 1.94%）。

### 为什么 ``.clone()`` 冻结而不是直接挂视图

``query_start_loc.cpu`` 与 ``optimistic_seq_lens_cpu`` 都是**常驻 pinned buffer 的切片**，
每步被原地重写（``gpu_model_runner.py:1889-1895`` / ``1902-1907``）。异步执行下若不冻结，
算子读到的是「下一步」的值。冻结把该步数值钉死。

### 为什么用私有 ``_seq_lens_cpu`` 而不是公开属性 ``seq_lens_cpu``

公开属性（``vllm/v1/attention/backend.py:436-439``）在 ``_seq_lens_cpu is None`` 时会
``self.seq_lens.to("cpu")`` —— 那正是一次 D2H。本部署（``run.sh`` 无 async
scheduling / 无投机解码 / 无 DCP）该字段恒有值，故直接取私有字段并**失败即报错**：
宁可构建期显式抛错，也不静默退化成逐层 D2H。

### 语义等价性

设备 ``seq_lens``（``gpu_model_runner.py:1992-1995``）=
``num_computed_tokens + num_scheduled_tokens``；host ``optimistic_seq_lens_cpu``
（同文件 L1902-1905）用同一组操作数、同一算术 ⇒ 逐位相同。
``query_start_loc`` 的 host/gpu 两副本来自同一 pinned buffer 的一次 ``copy_to_gpu()``。
此等价性**未在本机做设备级逐位证明**（无卡窗），仅由卡窗脚本的
``raw_matches_frozen_baseline`` 端到端把关（见 RESULT.md §4/§6）。

不做 ``site-packages`` 修改（AGENTS.md 规则 1）：只包一层 ``build``。
"""

from __future__ import annotations

import inspect
import sys

# 安装回执标记：卡窗脚本据此确认「本次运行确实加载了交付形态的 CPU 孪生绑定」。
REV = "c33-d2h-delivery-cpu-twin-1"

# 加载回执：模块被 import（即 overlay 生效）即打印，与 install() 是否成功无关。
print(f"[gemma-d2h-delivery] LOAD rev={REV}", file=sys.stderr, flush=True)

_FREEZE = None


def _freeze(t):
    """host 侧冻结副本。"""
    return t.clone()


def _resolve_extractor(orig_build):
    """安装期一次性决定「怎么从 build 调用里取出 common_attn_metadata」。

    本代 ``metadata_fix.py`` 不存在（overlay 里的那一块会打印 skipped），所以
    ``orig_build`` 通常就是真实 ``build``（有具名参数）。但为与仓库根那一代保持
    同一实现、并在将来本代补回 ``metadata_fix`` 时仍然正确，这里仍优先用
    ``_gemma4_orig`` 解开包装链（其签名是 ``(self, *args, **kwargs)``，直接取签名
    会取不到参数名），签名不可用时退化为按关键字/位置取值。
    两支都在安装期选定，不进热路径做判断。
    """
    target = getattr(orig_build, "_gemma4_orig", orig_build)
    try:
        params = list(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        params = []

    if "common_attn_metadata" in params:
        sig = inspect.signature(target)

        def extract(self, args, kwargs):
            return sig.bind(self, *args, **kwargs).arguments["common_attn_metadata"]

        return extract

    def extract(self, args, kwargs):
        if "common_attn_metadata" in kwargs:
            return kwargs["common_attn_metadata"]
        if len(args) > 1:
            return args[1]
        return args[0]

    return extract


def install() -> bool:
    """包装 ``BlockAttentionMetadataBuilder.build``，转发 CPU 孪生张量。幂等。"""
    global _FREEZE

    from vllm_sdaa.attention.block_attn import BlockAttentionMetadataBuilder

    if getattr(BlockAttentionMetadataBuilder, "_gemma4_cpu_twin", False):
        return True

    orig_build = BlockAttentionMetadataBuilder.build
    _FREEZE = _freeze
    extract = _resolve_extractor(orig_build)  # bind-once：安装期选定取参方式

    def build(self, *args, **kwargs):
        md = orig_build(self, *args, **kwargs)
        cam = extract(self, args, kwargs)
        qsl_cpu = cam.query_start_loc_cpu
        sl_cpu = cam._seq_lens_cpu  # noqa: SLF001  取 host 驻留副本，避免属性回退 D2H
        # 失败即报错（fail-closed）：结构不变量校验，不是后端/AB 选择。
        # 静默回退会让本线重新落入 v1 已证否的 32.3 ms/次排空。
        if qsl_cpu is None or sl_cpu is None:
            raise RuntimeError(
                "[gemma-d2h-delivery] CommonAttentionMetadata 缺少 host 孪生值 "
                f"(query_start_loc_cpu={type(qsl_cpu).__name__}, "
                f"_seq_lens_cpu={type(sl_cpu).__name__}); "
                "本部署（run.sh：无 async scheduling/投机解码/DCP）不应出现，"
                "拒绝静默回退到逐层 D2H。"
            )
        md.query_start_loc_cpu = _FREEZE(qsl_cpu)
        md.seq_lens_cpu = _FREEZE(sl_cpu)
        return md

    build._gemma4_orig = orig_build
    BlockAttentionMetadataBuilder.build = build
    BlockAttentionMetadataBuilder._gemma4_cpu_twin = True
    print(
        f"[gemma-d2h-delivery] INSTALL rev={REV} cpu-twin-bound",
        file=sys.stderr,
        flush=True,
    )
    return True
