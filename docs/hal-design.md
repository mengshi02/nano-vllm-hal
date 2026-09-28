# nano-vllm HAL 改造设计（Phase 1 骨架）

## 1. 目标

把 nano-vllm 从「CUDA 深度耦合」改造成**硬件抽象层（HAL）**，支持四类设备接入：

| 厂商 | 运行时 | 进程组后端 | 算子接入点 |
|------|--------|-----------|-----------|
| NVIDIA | `torch.cuda` | `nccl` | triton + flash-attn（现有） |
| 海光 DCU | ROCm（`torch` 的 hip） | `nccl`(ROCm 版) | ROCm 融合算子 |
| 昇腾 NPU | `torch_npu` | `hccl` | `_p_attention` / `_p_moe` 等 |
| 摩尔线程 | `torch_musa` | `mccl` | MUSA fusion kernel |

## 2. 架构决策（为什么这么选）

- **平台模型**：一台硬件 = 一个 `Platform` 类，对齐 vLLM 的 `Platform`/`Hardware` 拆分。引擎与模型代码只认识 `current_platform`，**What（要算什么）与 How（怎么算）分离**。
- **注册机制**：中央 dict + `current_platform.backend_name` 作 key + 平台按需 import。不用 `try/except ImportError`（会掩盖真实编译/环境错误），也不全局 import 所有平台。
- **算子范围**：全家族化，**默认实现即 torch 通用实现**。注意力/KV/采样/线性/激活/归一全部进家族，每家未来只覆写自己需要的家族函数，切换粒度是「家族」而非「全有或全无」。

## 3. 目录结构（已落地）

```
nanovllm/
  platforms/
    __init__.py       # 解析当前设备 → current_platform
    interface.py      # Platform 基类（device/显存/通信/图捕获接口）
    cuda.py           # NVIDIA（参考实现，Phase1 冒烟目标）
    vendors.py        # rocm / ascend / musa 三个 stub
    cpu.py            # 无加速卡时的 fallback
  ops/
    __init__.py       # 家族注册表 + get_xxx_fn() 选择器
    rms_norm.py       # rms_norm_forward / add_rms_norm_forward
    rope.py           # rotary_emb_forward
    kvcache.py        # store_kvcache（triton kernel，默认）
```

## 4. 关键接口

`current_platform` 暴露：`backend_name`、`device_module`、`set_device`、`mem_get_info`、
`empty_cache`、`memory_stats`、`synchronize`、`init_process_group`、`destroy_process_group`、
`graph_class`、`graph_capture_context`、`is_capture_supported`。

算子选择器：`ops.get_rms_norm_fn(backend_name)` / `get_rotary_fn` / `get_kv_cache_fn`。

## 5. 尚未抽象（Phase 2 的 seam）

- **`engine/model_runner.py`** 内联的 `dist.init_process_group("nccl")`、`torch.set_default_device("cuda")`、
  `torch.cuda.CUDAGraph`、`torch.cuda.mem_get_info` 等，需改为走 `current_platform`。
- **`layers/attention.py` 的 `flash_attn_*`** —— 每家最大改动点，Phase 1 仅以平台 key 选出显
  式 attention 家族占位，真实 Ascend/Hygon/Moore 实现在拿到对应 SDK 后补。

## 6. Phase 1 验收

CUDA 冒烟：`example.py` 与 `bench.py` 输出不变（行为零回归），同时 `current_platform.backend_name
== "nvidia"`。其余三家以 stub 形式被 `_resolve()` 识别，但不在本机验证。

## 7. 接入一家新硬件的步骤

1. `platforms/` 新增 `<vendor>.py`，实现 `Platform` 接口。
2. 需覆写算子的，`ops/` 下用 `register(family, "<vendor>")` 挂 ln/jit/自定义 kernel。
3. 若有模型并行/量化差异，在 model 层用 `current_platform.dispatch_key` 分支。
   引擎主流程**不改**。