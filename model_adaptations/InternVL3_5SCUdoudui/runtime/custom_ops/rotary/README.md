# InternVL3_5-8B 语言塔 RoPE 模块

本模块把 Qwen3 语言塔的 q/k RoPE 绑定到**厂商已安装**的 `_C.rotary_embedding` 融合算子：不是新增 UAL、kernel 或二进制。

## 支持范围

仅支持本分支已验证的 FP16、TP2 语言塔形状：

- Q16/K4（16 query / 4 KV head）、`head_dim = rotary_dim = 128`；
- 完整 Neox 旋转（`rotary_interleaved=false`）、`rope_theta = 1e6`；
- 原 FP16 packed cache `[40960, 128]`，且 cache 与 qkv weight 位于当前 SDAA 设备。

不支持的 head 配置、cache 格式、量化权重或 CPU 构造一律在初始化期 `RuntimeError` 退出（无 fallback）。

## 绑定方式

本仓 overlay 在 RMSNorm 与 BlockAttention 之后**无条件**安装构造 hook：保留 norm-before-RoPE 语义与原 cache Tensor 的位模式，对每个注意力实例做私有浅复制以隔离 module 字典；forward 只调用固定的 opaque mutating 厂商 API，**无 `getenv`、无后端分支、无缓存格式转换**。

初始化收据记录 PID / TP rank / device、实例与 cache 身份、临时 CPU 缓存 hash 及实际库身份；不持久化 cache 或权重数据。

升级厂商库后必须重新验证，不能直接替换以下 pin（`op.py` 内亦留有完整校验值）：

- 扩展 SHA256 `9ecc2cb57a5c870704580e39112c1e95321bc076507139d8b8855273a90cfec0`
- 厂商 core SHA256 `adf3014a8fd3ced720bbf5cfe666752b8583dc76b17b5c96517fc70716773cb1`

## 验证

本包内可直接复跑（需设备，逐次取锁）：

```bash
ADAPTATION="$PWD/model_adaptations/InternVL3_5SCUdoudui"
PYTHONPATH="$ADAPTATION/runtime" SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python \
  "$ADAPTATION/tests/verify_vendor_rope.py"
PYTHONPATH="$ADAPTATION/runtime" SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python \
  "$ADAPTATION/tests/test_model_binding.py" --device 0
```

正式启动仍使用 `run.sh`（固定 `VLLM_DISABLE_COMPILE_CACHE=1`，与已验证对照一致），不需要私有 `PYTHONPATH` 或 runpy；解释器保持厂商 `/home/py312/bin/python`。

**边界**：厂商算术与 native/CPU 参考**不是位精确**，本模块沿用已验证的 FP16 容差；只声称固定输入 32-token 逐位一致，**不声称官方任务精度**。

- 开发期测试（**包外**，本队账本仓）：`op_learning/rotary/internvl-vendor-fused/test_model_binding_v3.py`（root 串行：先 `--guard-only` 含 `python -O`，再 Fake/M1/M8/fullgraph）。
