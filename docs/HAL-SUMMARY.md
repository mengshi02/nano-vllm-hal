# nano-vllm-hal 改造方案与现状总结

> 本文档面向**后续接手的另一个 AI / 开发者**，完整交代：项目目标、已完成的 HAL 改造、当前代码状态、
> 剩余工作（Phase 2 的 seam）、以及新增硬件的标准接入步骤。读完本文件即可继续推进。

---

## 1. 项目定位与目标

把 `nano-vllm`（一个 ~1200 行、CUDA-only、单模型 Qwen3 的轻量 vLLM 复刻）改造成带**硬件抽象层（HAL）**
的框架，支持四类设备接入：

| 厂商 | 运行时 | 进程组(通信)后端 | 注意力/KV 算子接入点 |
|------|--------|-----------------|----------------------|
| NVIDIA | `torch.cuda` | `nccl` | triton + flash-attn（现有，默认实现） |
| 海光 DCU | ROCm（`torch` hip 变体） | `nccl`(ROCm 版) | ROCm 融合算子 |
| 昇腾 NPU | `torch_npu` | `hccl` | `torch_npu` 的 `_p_attention` 等 |
| 摩尔线程 | `torch_musa` | `mccl` | MUSA fusion kernel |

核心原则（与 vLLM 对齐）：**What（要算什么）与 How（在哪算/怎么算）分离**。引擎与模型主流程只认识
`current_platform` 和算子家族选择器，硬件差异全部隔离在 `platforms/` 与 `ops/` 两个目录，新增硬件**不改主流程**。

---

## 2. 已完成的架构决策（重要：不要推翻）

三个关键决策已经拍板并落地，后续请沿用：

1. **抽象粒度 = 完整 Platform 架构**（对齐 vLLM 的 `Platform`/`Hardware` 拆分），但砍掉本项目用不上的
   量化 / MoE / 多架构模型，只保留需要的算子家族。一台硬件 = 一个 `Platform` 类。
2. **注册机制 = 中央 dict + `backend_name` 作 key + 平台按需 import**。
   - 不用 `try/except ImportError`（会掩盖真实编译/环境错误）；
   - 不全局 import 所有平台；
   - 平台探测用 `is_available()` 按优先级 `Ascend → MUSA → ROCm → CUDA → CPU` 解析出 `current_platform`。
3. **算子范围 = 全家族化，默认实现即 torch 通用实现**。注意力 / KV 写入 / RoPE / RMSNorm 已家族化；
   linear / activation / sampler 等复用 torch 通用实现，未来按同法覆写。切换粒度是「家族」，不是「全有或全无」。

---

## 3. 当前目录结构（已落地实景）

```
nano-vllm-hal/
├── README.md               # 仅一行 "# nano-vllm-hal"（本项目自有，勿覆盖）
├── LICENSE                 # Copyright (c) 2026 mengshi（本项目自有，勿覆盖）
├── .gitignore              # 本项目自有
├── pyproject.toml          # name = "nano-vllm-hal"（Python 包名仍是 nanovllm）
├── example.py              # 用法示例（CUDA 冒烟用）
├── bench.py                # 性能基准
├── docs/
│   ├── hal-design.md       # Phase 1 设计文档（旧，略旧于现状）
│   └── HAL-SUMMARY.md       # 本文档（权威现状）
├── assets/logo.png
└── nanovllm/               # Python 包（内核 + HAL 扩展都在此包内，未拆顶层）
    ├── __init__.py          # 只导出 LLM、SamplingParams
    ├── config.py            # Config dataclass（含 max_model_len/gpu_memory_utilization 等）
    ├── llm.py               # class LLM(LLMEngine): pass
    ├── sampling_params.py   # SamplingParams(temperature/max_tokens/ignore_eos)
    ├── engine/              # 内核：调度/块管理/序列/引擎
    │   ├── llm_engine.py
    │   ├── model_runner.py  # ★ 已改：设备调用走 current_platform
    │   ├── scheduler.py
    │   ├── block_manager.py
    │   └── sequence.py
    ├── layers/              # 内核：算子层（含 attention，已改走 ops 家族）
    │   ├── attention.py     # ★ 已改：从 ops 取 get_attention_fn / get_kv_cache_fn
    │   ├── linear.py
    │   ├── layernorm.py
    │   ├── rotary_embedding.py
    │   ├── embed_head.py
    │   ├── activation.py
    │   └── sampler.py
    ├── models/qwen3.py     # 内核：Qwen3ForCausalLM（含 packed_modules_mapping）
    ├── utils/
    │   ├── context.py        # Context 数据结构 + 全局 set/get/reset_context
    │   └── loader.py        # safetensors 权重加载
    ├── ops/                 # ★ HAL：算子家族（扩展）
    │   ├── __init__.py       # 家族注册表 + get_xxx_fn() 选择器
    │   ├── attention.py      # attention_forward（flash-attn，默认=nvidia）
    │   ├── kvcache.py        # store_kvcache（triton 写 KV，默认）
    │   ├── rms_norm.py       # rms_norm_forward / add_rms_norm_forward
    │   └── rope.py           # rotary_emb_forward
    └── platforms/           # ★ HAL：平台抽象（扩展）
        ├── __init__.py       # _resolve() → current_platform
        ├── interface.py      # Platform 基类
        ├── cuda.py           # CUDAPlatform（nvidia，参考实现）
        ├── vendors.py        # ROCmPlatform / AscendPlatform / MUSAPlatform（stub）
        └── cpu.py            # CPUPlatform（fallback，无加速卡时）
```

> 注：`ops/` 与 `platforms/` 当前在 `nanovllm/` **包内**（方案 B，即「保持包内」方案）。之前讨论过是否
> 提到与 `nanovllm` 同级/套 `extensions`/`backends` 壳，最终结论是**保持包内、不套壳**，最贴近 vLLM 真实结构，
> 避免「内核 ↔ 扩展」双向依赖环。

---

## 4. 已完成的代码改造（Phase 1 的具体 diff）

### 4.1 `platforms/interface.py` —— `Platform` 基类

定义了统一接口，方法签名如下（后续新增平台必须实现）：

- 设备：`device_module`(property)、`is_available()`、`device_count()`、`current_device()`、
  `set_device(rank)`、`set_default_device()`
- 显存：`mem_get_info() -> (free, total)`、`empty_cache()`、`reset_peak_memory_stats()`、`memory_stats()`
- 通信：`init_process_group(world_size, rank)`、`destroy_process_group()`
- 图捕获：`is_capture_supported()`（默认 True）、`graph_class()`、`graph_capture_context(graph, pool)`
- 属性：`backend_name`、`device_name`、`dispatch_key`

### 4.2 `platforms/cuda.py` —— 参考实现

把原本散落在 `ModelRunner` 的 `torch.cuda.*` / `dist.init_process_group("nccl")` 原样搬入。
`backend_name="nvidia"`、`graph_class=torch.cuda.CUDAGraph`、`graph_capture_context=torch.cuda.graph`。

### 4.3 `platforms/vendors.py` —— 三个 stub

- `ROCmPlatform(CUDAPlatform)`：海光 DCU，`backend_name="rocm"`，复用 CUDA 运行时，只差异在通信后端。
- `AscendPlatform(Platform)`：昇腾，`torch_npu`，`backend_name="ascend"`，通信 `hccl`，`is_capture_supported()=False`。
- `MUSAPlatform(CUDAPlatform)`：摩尔线程，`torch_musa`，`backend_name="musa"`，通信 `mccl`。
- 三者当前都是 **占位 stub**，`AscendPlatform` 的 `graph_class`/`graph_capture_context` 抛 `NotImplementedError`，
  待拿到对应 SDK 后补真实实现。

### 4.4 `platforms/__init__.py` —— 平台探测

`_resolve()` 按 `AscendPlatform → MUSAPlatform → ROCmPlatform → CUDAPlatform` 顺序，返回第一个
`is_available()` 为真的实例，否则回退 `CPUPlatform`。导出单例 `current_platform`。

### 4.5 `platforms/cpu.py` —— fallback

无加速卡时保证包能干净 import，`device_module=None`、`device_count=0`，通信/图捕获抛未实现。

### 4.6 `ops/__init__.py` —— 算子家族注册表

```python
_RMS_NORM  = {"default": rms_norm.rms_norm_forward}
_ROTARY    = {"default": rope.rotary_emb_forward}
_KV_CACHE  = {"default": kvcache.store_kvcache}
_ATTENTION = {"default": attention.attention_forward}

def register(family, backend_name): ...      # 装饰器，厂商用 @register(_KV_CACHE, "musa")
def get_attention_fn(backend_name): ...      # _pick(_ATTENTION, backend_name)
def get_kv_cache_fn(backend_name): ...
def get_rms_norm_fn(backend_name): ...
def get_rotary_fn(backend_name): ...
```

`_pick` 逻辑是 `family.get(backend_name, family["default"])`：未注册就回退默认实现。

### 4.7 `ops/*.py` —— 四个家族默认实现

- `attention.py`：`attention_forward_flash`，用 `flash_attn_varlen_func`(prefill) 与
  `flash_attn_with_kvcache`(decode)；prefix cache 时 `block_tables` 置空则读 k_cache/v_cache。
  **`flash_attn` 是模块级 import**（保持与原 `layers/attention.py` 一致）。
- `kvcache.py`：`store_kvcache_default`，内联 triton kernel 按 `slot_mapping` 把 K/V 散写到 paged cache；
  triton 用**函数内局部 import** 避免无 triton 平台 import 失败。
- `rms_norm.py` / `rope.py`：torch 通用实现（函数签名见文件），等价于原 `layers/layernorm.py`、
  `layers/rotary_embedding.py` 里的纯 torch 逻辑。

### 4.8 `engine/model_runner.py` —— 引擎接线（关键）

把下列硬编码都改成走 `self.platform = current_platform`：

| 原 | 现 |
|----|----|
| `dist.init_process_group("nccl", ...)` | `self.platform.init_process_group(...)` |
| `torch.cuda.set_device(rank)` | `self.platform.set_device(rank)` |
| `torch.set_default_device("cuda")` | `self.platform.set_default_device()` |
| `torch.cuda.empty_cache()` / `reset_peak_memory_stats()` | `self.platform.empty_cache()` / `reset_peak_memory_stats()` |
| `torch.cuda.mem_get_info()` / `memory_stats()` | `self.platform.mem_get_info()` / `memory_stats()` |
| `torch.cuda.synchronize()` | `self.platform.synchronize()` |
| `torch.cuda.CUDAGraph()` | `self.platform.graph_class()()` |
| `with torch.cuda.graph(graph, pool)` | `with self.platform.graph_capture_context(graph, pool)` |
| `dist.destroy_process_group()` | `self.platform.destroy_process_group()` |

`warmup_model()`、`allocate_kv_cache()`、`capture_cudagraph()`、`exit()` 均已同步改造。

**仍保留的 `dist.*`**（模型并行通用 API，不属于设备抽象，按设计不动）：
`dist.barrier()`（3 处，用于多进程同步）、`torch.distributed` 的 `get_rank/get_world_size/all_reduce/gather`
（在 `linear.py`、`embed_head.py`、`qwen3.py` 里做 tensor parallel）。

### 4.9 `layers/attention.py` —— 注意力接线

删掉内联的 triton `store_kvcache` 与 `flash_attn_*` 调用，改为在 `__init__` 里：

```python
self._store_kvcache = get_kv_cache_fn(current_platform.backend_name)
self._attention    = get_attention_fn(current_platform.backend_name)
```

`forward()` 里统一 `o = self._attention(q, k, v, k_cache, v_cache, self.scale, context)`。
`Context` 结构见 `utils/context.py`。

---

## 5. Phase 2 剩余工作（后继者的任务清单）

按优先级排列，这是下一个 AI/开发者要继续推进的部分：

### P0 —— CUDA 冒烟验证（未做，因当前机器无 torch/GPU）

```bash
cd nano-vllm-hal && python example.py   # 或 python bench.py
```

期望：`current_platform.backend_name == "nvidia"` 且输出与原 nano-vllm 完全一致（**零回归**）。
注意：当前开发机无 torch（`ModuleNotFoundError: No module named 'torch'`），必须在装有
torch + triton + flash-attn 的 CUDA 机器上跑。若失败，优先检查 `ops/attention.py` 的 flash-attn
import 与 `ops/kvcache.py` 的 triton 路径。

### P1 —— 把线性/激活/sampler 也家族化（可选，但建议）

当前 `layers/linear.py`、`layers/activation.py`（`SiluAndMul`）、`layers/sampler.py` 仍是 torch 通用实现，
未纳入 `ops/`。若要彻底对齐 vLLM 的「全家族化」，按同样的 `register` 模式把它们迁入 `ops/`，并保留
默认 torch 实现。`RMSNorm` 与 `RotaryEmbedding` 的 `layers/` 包装类仍在用 `@torch.compile`，未来可让
`layer` 类薄薄一层，重活交给 `ops` 家族。

### P2 —— 三家 vendor 的真实算子实现（拿到 SDK 后）

- **昇腾 Ascend**：实现 `ops/attention.py` 的 `_p_attention` 版本 + KV scatter；`platforms/vendors.py`
  里 `AscendPlatform` 补 `graph_class`/`graph_capture_context`（或保持 `is_capture_supported=False`）。
- **海光 ROCm/DCU**：`ops/attention.py` 走 ROCm flash-attn 变体；`ROCmPlatform` 通信后端确认。
- **摩尔 MUSA**：`ops/kvcache.py`/`ops/attention.py` 走 MUSA fusion kernel；`MUSAPlatform` 通信 `mccl` 确认。

### P3 —— 模型层 TP 的 dispatch 差异

若某家头数无法整除 TP 或量化/ROPE 有差异，在 `models/qwen3.py` 用 `current_platform.dispatch_key`
分支（当前 `dispatch_key` 为 `CUDA`/`ROCM`/`NPU`/`MUSA`/`CPU`）。

---

## 6. 接入一家新硬件的标准步骤（给后继者）

1. 在 `nanovllm/platforms/` 新增 `<vendor>.py`，继承 `Platform`（或 `CUDAPlatform` 若为 CUDA 兼容运行时），
   实现/覆写 `interface.py` 里的方法，设好 `backend_name` / `device_name` / `dispatch_key`。
2. 在 `nanovllm/platforms/__init__.py` 的 `_PLATFORMS` 列表按优先级插入该类；若有 CUDA 兼容运行时
   fallback，务必排在 `CUDAPlatform` 之前。
3. 需覆写算子的，在 `nanovllm/ops/` 里新增/扩展家族，用 `@register(family, "<vendor>")` 注册，或用
   `register` 手动赋值 `family["<vendor>"] = fn`。
4. 若有模型并行/量化差异，在 `models/` 层用 `current_platform.dispatch_key` 分支。
5. **引擎主流程（`engine/`）不改一行**。

---

## 7. 工程/仓库约定（注意）

- **GitHub 仓库**：`github.com:mengshi02/nano-vllm-hal`（remote `origin`）。本文档在此仓库的
  `/Users/mengshi3/py/nano-vllm-hal/`。
- **项目名 / PyPI 包名**：`nano-vllm-hal`（`pyproject.toml` 的 `name`）。**Python import 名仍是 `nanovllm`**
  （`from nanovllm import LLM, SamplingParams`），不要改。
- **`README.md` / `LICENSE` / `.gitignore`**：属于 `nano-vllm-hal` 项目自有，**不要覆盖**。
  - `README.md` 目前只有一行 `# nano-vllm-hal`，尚未补全（可后续单独重写，含 HAL 特性说明）。
  - `LICENSE` 版权是 `Copyright (c) 2026 mengshi`。
- 源工程 `nano-vllm`（上游 `GeeeekExplorer/nano-vllm`）与本工程是**两个独立仓库**，请勿混淆。

---

## 8. 验证与状态记录（时间线）

- ✅ 已建 `platforms/`、`ops/`、`docs/hal-design.md`。
- ✅ 已接线 `engine/model_runner.py`、`layers/attention.py`。
- ✅ 所有改动文件 `python3 -m py_compile` 通过。
- ✅ 已复制到 `/Users/mengshi3/py/nano-vllm-hal/` 并清理 `__pycache__`。
- ✅ 已恢复 `README.md`/`LICENSE`/`.gitignore` 为项目自有版本。
- ⏳ **未做**：CUDA 真机冒烟验证（见 P0）。
- ⏳ **未做**：`git add` / `git commit`（当前改动尚未提交，留待用户决定）。