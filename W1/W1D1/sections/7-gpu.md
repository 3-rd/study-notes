# 七、GPU 加速与分布式

---

## 7-1 单 GPU 迁移

### 基本迁移方法

```python
import torch
import torch.nn as nn

# 判断 GPU 是否可用
print("GPU 可用:", torch.cuda.is_available())           # True/False
print("GPU 数量:", torch.cuda.device_count())           # 几块卡
print("当前设备:", torch.cuda.current_device())         # 0
print("设备名称:", torch.cuda.get_device_name(0))        # NVIDIA GeForce RTX ...

# 指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型迁移
model = MyModel().to(device)

# 数据迁移
x = x.to(device)

# 验证迁移成功
print(next(model.parameters()).device)  # cuda:0
```

### 设备迁移涉及什么

模型和数据是最直观的，还有三样容易被忽略：

| 迁移对象 | 说明 |
|---------|------|
| **模型参数** | `model.to(device)` |
| **数据** | `x.to(device)`, `y.to(device)` |
| **Optimizer state** | 与模型分离，必须在 model.to 之后创建 |
| **自定义 Loss** | 如果是 `nn.Module` 形式的 loss，需要 `.to(device)` |
| **Buffers** | `BatchNorm` 的 running_mean/running_var，model.to 时自动迁移 |

### Optimizer 与模型的关系

**Optimizer 和模型是分开的，不在模型里面。** Optimizer 持有模型参数的引用。

```python
model = MyModel()                              # nn.Module 子类
optimizer = torch.optim.Adam(model.parameters())  # Optimizer 单独创建，引用 model 的参数
```

**Optimizer 必须在 model.to 之后创建**：

```python
# 正确 ✅：model 迁 GPU 后再创建 optimizer
model = MyModel().to('cuda')
optimizer = torch.optim.Adam(model.parameters())  # state 自动在 GPU 上

# 有坑 ❌：model 在 CPU 时先创建了 optimizer
model = MyModel()                               # 参数在 CPU
optimizer = torch.optim.Adam(model.parameters()) # state 在 CPU（记住了参数在 CPU）
model = model.to('cuda')                        # 参数去了 GPU，但 optimizer.state 还指向 CPU
# 训练时报错：Expected all tensors to be on the same device
```

**如果 optimizer 已经创建好了需要迁移**：重建 optimizer 即可。

```python
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters())  # 重建 optimizer
```

### .cuda() vs .to(device)

```python
# 两种写法完全等价
model.cuda()
model.to('cuda')

# 推荐用 .to(device)，更通用（同时支持 cpu/cuda）
model.to(device)
```

### .to() 不修改原对象

```python
# 常见错误 ❌：以为 .to() 会原地修改
model.to(device)
output = model(x)  # model 还在 CPU 上，报错

# 正确 ✅：必须重新赋值
model = model.to(device)
output = model(x)  # ✓
```

### 标量 tensor 默认在 CPU

```python
a = torch.tensor(3)
print(a.device)  # cpu

# 正确做法：创建时就指定 device
a = torch.tensor(3, device=device)
# 或创建后迁移
a = torch.tensor([1, 2, 3]).to(device)
```

### device 常量写法

```python
# ❌ hardcode：如果没有 GPU 就会报错
model = model.to('cuda:0')

# ✅ 自动适配：自动检测有无 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 多卡时选特定 GPU

```python
# 当前进程绑定到 GPU 0（避免 hardcode）
torch.cuda.set_device(0)

# 或者用环境变量，进程只看到第 2 张卡
# CUDA_VISIBLE_DEVICES=1 python train.py
```

---

## 7-2 DataParallel（DP）— 单进程多卡

### DP 用法

```python
import torch.nn as nn

# model 会在内部按 batch 维度自动切分，分发到各 GPU
model = nn.DataParallel(model, device_ids=[0, 1, 2])
model = model.cuda()

# 前向时：batch=32 自动分成 3 份 → [11, 11, 10] 分配到 3 卡
# 反向时：梯度自动汇总到主 GPU（device_ids[0]）
output = model(input)
```

### DP 原理

```
batch=32, 3 GPU
       ↓
GPU0: batch[0:11]   → forward → 梯度
GPU1: batch[11:22]  → forward → 梯度
GPU2: batch[22:32]  → forward → 梯度
       ↓
主 GPU 累加梯度 → optimizer.step()
```

### batch 设定的含义

**batch=32 指的是喂给模型的 batch 总数，DP 自动在 batch 维度切分到各卡**。

- batch=32，3卡 → 每卡 ~11，总量 32
- batch=128，4卡 → 每卡 32，总量 128

**batch 总数和单卡一样，所以学习率不需要调整**，训练效果（模型收敛行为）和单卡 batch=32 完全等价。

### GIL — DP 的核心限制

**GIL（Global Interpreter Lock）** 是 CPython（Python 官方解释器）的特性：同一时刻只能有一个线程执行 Python 字节码。

DP 内部使用 Python **多线程**（不是多进程），所以：
- 虽然有 3 张 GPU，但 Python 这边是串行的
- GPU 计算时会释放 GIL，PyTorch 底层 CUDA 操作不受 GIL 约束
- 但 Python 层的调度和协调仍受 GIL 限制

这就是为什么 DP 效率不如 DDP。

### DP 的问题

| 问题 | 说明 |
|------|------|
| **GIL 限制** | 单进程多线程，Python 层串行，效率受限 |
| **主 GPU 通信瓶颈** | 所有梯度在主 GPU 汇总，主卡通信成为瓶颈 |
| **主 GPU 显存压力** | 所有梯度汇总到主卡，GPU 0 显存占用最高 |
| **无法多机** | 只支持单机多卡，不能跨机器 |
| **官方已标记 legacy** | PyTorch 推荐用 DDP，DP 只适合科研快速实验 |

### device_ids 指定用哪些卡

```python
# 不用全部卡，只用第 0 和第 2 张
model = nn.DataParallel(model, device_ids=[0, 2])
```

### DP vs 单卡的效果对比

| | 单卡 batch=32 | DP 3卡 batch=32 |
|---|---|---|
| 每步样本量 | 32 | 32（每卡 ~11，总量不变） |
| 学习率 | 不变 | 不变 |
| 训练效果 | 等价 | 等价 |
| 速度 | 基准 | 更快（3卡并行计算） |

> DataParallel 适合科研快速实验，工业训练几乎不用 DDP。

---

## 7-3 DistributedDataParallel（DDP）— 工业标准

### DDP vs DP 对比

| | DataParallel (DP) | DistributedDataParallel (DDP) |
|---|---|---|
| 进程数 | 1 个进程（多线程） | 每卡 1 个进程 |
| 通信方式 | 主 GPU 汇总梯度（瓶颈） | 每卡独立反向，梯度 all-reduce 同步 |
| 多机支持 | ❌ 不支持 | ✅ 支持 |
| 效率 | 低（单进程+GIL） | 高（多进程，无 GIL） |
| 工业使用 | 科研快速实验 | 工业级训练 |

### DDP 完整训练流程

**Step 1：每卡启动一个进程（用 torchrun 或 spawn）**

```bash
# torchrun 启动（推荐）
torchrun --nproc_per_node=4 train.py
# 等价于：4 个进程，每个进程用一块 GPU
```

```bash
# 如果多机，每节点 4 卡，2 个节点
# 节点1：
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 train.py
# 节点2：
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 train.py
```

**Step 2：每个进程初始化**

```python
import os
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

# world_size = GPU 总数
# rank = 当前进程编号（0 ~ world_size-1）
```

**Step 3：模型包装 DDP**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])  # 每个进程独立包装

    # 训练循环...
    dist.destroy_process_group()

# 启动 4 个进程（每卡一个）
import torch.multiprocessing as mp
mp.spawn(main, args=(4,), nprocs=4)
```

### DDP 核心概念

**环境变量（torchrun 自动设置）：**

| 变量 | 含义 | 示例 |
|------|------|------|
| `LOCAL_RANK` | 当前节点内的 GPU 编号 | 0, 1, 2, 3 |
| `RANK` | 全局进程编号 | 0~7（2节点×4卡） |
| `WORLD_SIZE` | 总进程数 | 8 |
| `MASTER_ADDR` | 主节点 IP | 192.168.1.1 |
| `MASTER_PORT` | 主节点端口 | 12355 |

**Gradient Buckets — DDP 梯度同步原理：**

DP 的梯度汇总方式：
```
GPU0 梯度 → GPU0
GPU1 梯度 → GPU0
GPU2 梯度 → GPU0  ← 所有梯度在主卡汇合，主卡通信量很大
```

DDP 的梯度同步方式（all-reduce）：
```
GPU0 梯度 ↔ all-reduce ↔ GPU1 ↔ all-reduce ↔ GPU2
```

每个 GPU 同时接收所有其他 GPU 的梯度并做平均（all-reduce），通信量均摊到每张卡，无主卡瓶颈。

### DDP 保存与加载

```python
# 保存：所有进程都执行，但只有 rank=0 实际写入磁盘
if rank == 0:
    torch.save(model.module.state_dict(), 'model.pt')

# 加载：每张卡独立加载
model.load_state_dict(torch.load('model.pt', map_location=f'cuda:{rank}'))
```

---

## 7-4 梯度累加（Gradient Accumulation）

### 为什么需要梯度累加

GPU 显存有限，batch size 太大放不下。梯度累加用小 batch 模拟大 batch。

```python
# 原始：batch=128 → 显存不够
output = model(batch_128)
loss = criterion(output, batch_128_labels)
loss.backward()  # 显存爆炸

# 梯度累加：分 4 次小 batch，累积梯度
accum_steps = 4
output = model(batch_32)
loss = criterion(output, batch_32_labels) / accum_steps  # 先缩小
loss.backward()  # 梯度会累加到 .grad 里

# 4 个 step 后统一更新
if (step + 1) % accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**等效于：**
```python
# 4 个 batch 累加后做一次更新，等价于 batch=128 的效果
# 梯度做了平均，所以要除以 accum_steps
```

### DDP + 梯度累加

```python
model.train()
for step, batch in enumerate(dataloader):
    # 只在本地 GPU 上做前向，不触发 all-reduce
    output = model(batch.to(rank))
    loss = criterion(output, target) / accum_steps
    loss.backward()

    if (step + 1) % accum_steps == 0:
        # 梯度 all-reduce 同步（所有进程同时执行）
        model.require_backward_grad_sync = True
        optimizer.step()
        optimizer.zero_grad()
        model.require_backward_grad_sync = False  # 跳过本轮 all-reduce，继续下一轮累加
```

> 注意：DDP 中梯度同步在 `backward()` 结束时自动触发，梯度累加期间需要跳过这个同步。

---

## 7-5 GPU 显存管理

### 显存占用来源

```
GPU 显存占用 =
  模型参数 (params)
+ 模型中间激活值 (activations) ← 最大头
+ 梯度 (gradients)
+ 优化器状态 (optimizer states，如 Adam 的动量)
+ 临时缓存 (CUDA kernels 中间结果)
```

### 常用显存查看

```python
print(torch.cuda.memory_allocated() / 1024**3)      # 当前占用（GB）
print(torch.cuda.max_memory_allocated() / 1024**3) # 峰值（GB）
print(torch.cuda.memory_reserved() / 1024**3)      # 缓存总量
```

### 显存释放

```python
# 清理临时变量（Python 层面）
del output, loss

# 清理 CUDA 缓存（强制释放显存）
torch.cuda.empty_cache()
```

### 防止 OOM 技巧

```python
# 1. 梯度不要存中间结果
model.eval()  # BatchNorm 等层会切换到 eval 模式，节省显存

# 2. 混合精度训练（AMP）减少显存
# 16 位存储梯度，32 位做参数更新
from torch.cuda.amp import autocast, GradScaler

# 3. 梯度检查点（Gradient Checkpointing）
# 用时间换空间：前向时不保存全部激活值，反向时重新计算
model gradient checkpointing

# 4. 减小 batch size（最简单有效）

# 5. 及时释放
del intermediate_output
torch.cuda.empty_cache()
```

### OOM 排查流程

```python
# 1. 看哪个变量占显存最大
torch.cuda.synchronize()  # 确保 GPU 操作完成
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.numel() * param.grad.element_size() / 1024**2} MB")

# 2. 检查是否真的 OOM
try:
    output = model(input)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM! 当前显存占用:", torch.cuda.memory_allocated() / 1024**3, "GB")
        torch.cuda.empty_cache()
```

---

## 7-6 多节点训练与 NCCL

### NCCL vs Gloo

| 后端 | 适用场景 | 速度 |
|------|---------|------|
| `nccl` | GPU 通信（多 GPU / 多机多卡） | 最快 |
| `gloo` | CPU 通信或无 GPU 环境 | 较慢 |

```python
# GPU 训练用 nccl
dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

# CPU 训练或无 GPU 时用 gloo
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
```

### 多节点训练示例

```bash
# 节点1（IP: 192.168.1.1）
torchrun --nproc_per_node=4 \
  --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.1 \
  --master_port=29500 \
  train.py

# 节点2（IP: 192.168.1.2）
torchrun --nproc_per_node=4 \
  --nnodes=2 --node_rank=1 \
  --master_addr=192.168.1.1 \
  --master_port=29500 \
  train.py
```

所有节点共 8 张卡（WORLD_SIZE=8），每张卡一个进程。

---

## 面试汇总

### Q1: .cuda() 和 .to(device) 区别？
两者完全等价，推荐 `.to(device)` 更通用。

### Q2: DP 和 DDP 区别？
- DP：单进程多线程，主 GPU 汇总梯度，有 GIL 瓶颈，无法多机
- DDP：每卡一个进程，梯度 all-reduce，无 GIL，效率高，支持多机
- 工业训练用 DDP，科研快速实验用 DP

### Q3: DDP 的 rank 和 world_size 是什么？
- `world_size`：总进程数（总 GPU 数）
- `rank`：当前进程的全局编号（0 ~ world_size-1）
- `LOCAL_RANK`：当前节点内的 GPU 编号

### Q4: 梯度累加的原理？
用多个小 batch 累加梯度，累积到一定步数后统一更新一次。等价于 batch size 扩大 N 倍，但显存只增加梯度存储，不增加激活值。

### Q5: 为什么需要 torch.cuda.empty_cache()？
释放 CUDA 缓存的临时显存（Python 层的 `del` 不一定立刻释放显存）。用于 OOM 排查或大模型推理前清理显存。

### Q6: 混合精度训练为什么能省显存？
用 float16 存储激活值和梯度，用 float32 存储参数更新。激活值是显存最大头，减半后显存显著降低。

### Q7: Optimizer 和模型的关系？Optimizer 在模型里面吗？
Optimizer 和模型是分开的，不在模型里面。Optimizer 持有模型参数的引用。**必须先 model.to(device)，再创建 optimizer**，否则 optimizer.state 记住了参数在 CPU，训练时会报错 "Expected all tensors to be on the same device"。

### Q8: device 迁移涉及哪些对象？
模型（model.to(device)）、数据（x.to(device), y.to(device)）、Optimizer（必须在 model.to 之后创建）、自定义 Loss（如果是 nn.Module 需要 to(device)）。Buffers（BatchNorm 等）由 model.to 自动迁移。

### Q9: GIL 是什么？为什么存在？
GIL（Global Interpreter Lock）是 CPython 的特性：同一时刻只有一个线程能执行 Python 字节码。因为 CPython 的内存管理不是线程安全的，加全局锁简化了实现。**Python 多线程对 CPU 密集型任务无效**，但 I/O 等待和 PyTorch CUDA 操作时会释放 GIL，此时其他线程可以执行。

### Q10: DP 中 batch=32 3卡，学习率要调吗？
不需要。batch=32 是 batch 总数，DP 在 batch 维度自动切分到各卡（每卡 ~11），**每步处理的样本总量仍是 32**，和单卡 batch=32 效果等价，学习率不变。DP 和 DDP 在这上面行为不同：DDP 每卡都加载完整 batch。

### Q11: 为什么 DP 效率不如 DDP？
DP 用 Python 多线程，受 GIL 限制，Python 层是串行的；DDP 每卡一个独立进程，无 GIL，效率高。另外 DP 主 GPU 汇总所有梯度是通信瓶颈，DDP 用 all-reduce 均摊到每张卡。
