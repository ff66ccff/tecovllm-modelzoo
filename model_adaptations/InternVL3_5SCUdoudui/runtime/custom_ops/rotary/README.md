# InternVL vendor fused RoPE 集成

本模块调用已安装vendor `_C.rotary_embedding`，不是新UAL、kernel或二进制。
只支持本分支验证的FP16、TP2语言塔：Q16/K4、head/rotary_dim128、完整Neox、
base1e6、原FP16 packed cache[40960,128]，cache和qkvweight在当前SDAA设备。
不支持其他head配置、cache格式、量化权重或CPU构造；初始化不符即RuntimeError。

现有repo overlay在RMSNorm和BlockAttention后无条件安装构造hook。
保留norm-before-RoPE及原cache Tensor/位模式；私有浅复制隔离module字典。
forward仅调用固定opaque mutating vendor API，无getenv/后端分支/缓存转换。
扩展SHA9ecc2cb57a5c870704580e39112c1e95321bc076507139d8b8855273a90cfec0，
coreSHAadf3014a8fd3ced720bbf5cfe666752b8583dc76b17b5c96517fc70716773cb1；
升级厂商库后须重新验证，不能直接替换pin。初始化收据记录PID/TP rank/device、
实例/缓存身份、临时CPU缓存hash及实际库身份；不持久保存缓存或权重数据。

正式启动继续使用scripts/serve_internvl.sh，它固定VLLM_DISABLE_COMPILE_CACHE=1，
与已验证对照一致；不需要私有PYTHONPATH或runpy。厂商Python保持/home/py312/bin/python。
测试见op_learning/rotary/internvl-vendor-fused/test_model_binding_v3.py：
root串行运行--guard-only（含python -O），再完整Fake/M1/M8/fullgraph及正式模型门禁。
算术不是bit-exact；本模块保留已验证FP16容差和模型32-ID证据，不宣称官方任务精度。
