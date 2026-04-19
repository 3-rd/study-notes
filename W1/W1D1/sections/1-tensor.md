# 一、张量（Tensor）— 最底层数据结构

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| 创建方式 | `torch.tensor()` vs `torch.Tensor()` | 两者区别（是否拷贝数据、类型推断） |
| 数据类型 | `float/int/bool/dtype` | 如何指定 device + dtype 同时创建 |
| GPU 迁移 | `.cuda()` / `.to(device)` | 如何判断 GPU 可用 |
| 形状操作 | `view / reshape / transpose / permute` | view vs reshape 区别（是否连续） |
| 索引切片 | `tensor[mask]` / `torch.masked_select` | 如何按条件筛选 |
| 运算 | `matmul / mm / @` / `torch.sum / mean / max` | 矩阵乘法哪个最快/最安全 |
| 与 NumPy 互转 | `tensor.numpy()` / `torch.from_numpy(nparr)` | 共享内存问题 |

---

## 1.1 创建方式详解

## 四类方法参数对比

| 方法 | device 参数 | requires_grad | dtype 参数 | 数据来源 | 是否拷贝 |
|---|---|---|---|---|---|
| `torch.Tensor(data)` | ❌ 无 | ❌ 无 | ❌ 无（固定float32） | 传入已有数据 | ✅ 拷贝 |
| `torch.tensor(data)` | ✅ 有 | ✅ 有 | ✅ 有（自动推断） | 传入已有数据 | ✅ 拷贝 |
| `torch.from_numpy(arr)` | ❌ 无 | ❌ 无 | ❌ 无（跟随numpy） | 必须是numpy数组 | ❌ 共享内存 |
| `torch.zeros()` / `torch.ones()` | ✅ 有 | ✅ 有（默认False） | ✅ 有（默认float32） | 自己生成 | ✅ 独立 |

## torch.Tensor() — 类构造函数（不推荐）

实际上是 `torch.FloatTensor` 的别名，类型固定 float32，无其他参数。

```python
torch.Tensor([1, 2, 3])  # → [1., 2., 3.] float32
t = torch.Tensor([1., 2.]).to('cuda').requires_grad_(True)
```

## torch.tensor() — 工厂函数（推荐）

自动推断类型，拷贝数据，参数最全。

```python
torch.tensor([1, 2, 3])                            # int64
torch.tensor([1., 2., 3.])                         # float32
torch.tensor([1, 2], dtype=torch.float32)          # 强制 float32
x = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)  # 一步到位
```

## torch.from_numpy() — 共享内存（需注意坑）

必须传 numpy 数组，共享内存修改互相影响。

```python
import numpy as np
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)  # 共享内存！
t[0] = 99.
print(arr)  # [99., 2., 3.] ← numpy 数组被改了！
```

dtype 跟随 numpy，不统一成 PyTorch 默认值。

## torch.zeros() / torch.ones() — 初始化

```python
torch.zeros(3, 4, dtype=torch.float32, device='cuda', requires_grad=True)
torch.ones(2, dtype=torch.int64)
```

## 默认值

所有方法不指定时 → **device=cpu，requires_grad=False**

---

## 1.2 形状操作

### stride 详解

stride 是"跳读规则"，决定按当前维度顺序读取 storage 时，每个维度内跳几个元素才能读完。

```python
x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
x.stride()  # → (12, 4, 1)
```

把 `(2, 3, 4)` 想象成 2 页纸，每页 3 行，每行 4 个元素。

### 连续 stride 的计算公式

```python
stride[i] = shape[i+1] × shape[i+2] × ... × shape[last]
```

## view

**本质**：换一套 shape/stride 来解释同一块 storage，数据完全不动。

```python
x = torch.randn(2, 3, 4)  # (2, 3, 4), stride (12, 4, 1)
x.view(6, 4)   # (6, 4), stride (4, 1) — 数据完全不动
x.view(24,)    # (24,)
x.view(1, 2, 3, 4)  # 任意维度都可以，只要 1×2×3×4 = 24
x.view(-1)     # (24,) — -1 自动推断
```

view 可以任意维度，不限于二维。要求：tensor 必须连续，不连续时报错：

```python
x_t = x.transpose(0, 1)  # stride (4, 12, 1) — 不连续
x_t.view(24)   # ❌ RuntimeError
```

## transpose

**本质**：交换两个维度的位置，数据不动，只换 shape 和 stride。

```python
x = torch.randn(2, 3)        # shape (2, 3), stride (3, 1)
x_t = x.transpose(0, 1)     # shape (3, 2), stride (1, 3)
```

**transpose 后 stride 的算法：直接交换对应位置的值。**

```python
x = torch.randn(2, 3, 4)           # stride (12, 4, 1)
x.transpose(0, 1)                   # stride (4, 12, 1)
```

transpose 后 tensor 必然不连续，因为 stride 顺序和 storage 物理顺序对不上了。

## reshape

**本质**：等价于 `contiguous().view()`。

```python
x.reshape(6, 4)  # 连续 tensor → 直接 view，不复制
x_t.reshape(6,)  # 不连续 tensor → 先复制重排，再 view
```

**优先用 `reshape`**，永远不报错。

## permute

入参语义：新维度 i 来自旧维度 dim_i。

```python
x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
x.permute(2, 0, 1)  # shape: (4, 2, 3)
```

和 transpose 完全一致——按同样索引直接重新排列 stride。

## squeeze / unsqueeze

```python
x = torch.randn(1, 3, 1, 8, 1)  # 5个维度
x.squeeze().shape   # (3, 8) — 删除所有 size=1 的维度
x.squeeze(0).shape  # (3, 1, 8, 1) — 删最外层
x.unsqueeze(0).shape   # (1, 2, 3) — 在最前面插
```

## flatten

```python
x = torch.randn(2, 3, 4)  # (2, 3, 4)
x.flatten().shape           # (24,) — 所有维度展平
x.flatten(1, 2).shape       # (2, 12) — 只展平中间两个维度
```

等价于 `reshape(-1)` 或 `reshape(..., -1)`。

## 三者对比

| | view | transpose | reshape |
|---|---|---|---|
| 数据动了吗？ | ❌ 没动 | ❌ 没动 | ✅ 不连续时会复制重排 |
| 连续性 | 要求连续 | 必然不连续 | 永远成功 |

---

## 1.3 索引切片

### 基础索引 + 切片

```python
x = torch.randn(2, 3, 4)
x[0].shape         # (3, 4) — 取第一个样本
x[0, 1].shape      # (4,) — 取特定元素
x[0:2].shape       # (2, 3, 4) — 切片
x[:, 1:].shape     # (2, 2, 4) — 跨维度切片
```

### None（等价于 unsqueeze）

```python
x = torch.randn(2, 3)
x[None, :].shape   # (1, 2, 3) — 等价于 x.unsqueeze(0)
x[:, None].shape   # (2, 1, 3)
```

### view vs copy 的分界线（核心！）

**返回 view**（共享底层 storage）：基础切片索引、`None` 索引、`squeeze`/`unsqueeze`、`transpose`/`permute`、`view`（连续时）、`expand`

**返回 copy**（独立新 storage）：`clone()`、布尔掩码索引、整数数组索引、`torch.masked_select`

```python
x = torch.randn(2, 3)
y = x[0:1]           # view
z = x[x > 0]          # copy
print(x.storage().data_ptr() == y.storage().data_ptr())  # True → view
print(x.storage().data_ptr() == z.storage().data_ptr())  # False → copy
```

### 布尔掩码

```python
x = torch.randn(3, 4)
mask = x > 0
x[mask].shape              # (?,) — 展平成一维
torch.masked_select(x, mask).shape  # 同上，完全等价
```

布尔掩码返回 **copy**，不共享数据。

### 整数数组索引

```python
x = torch.randn(5, 3)
rows = torch.tensor([0, 2, 3])
x[rows].shape        # (3, 3) — 取指定行
x[rows, 1].shape     # (3,) — 取这些行的第二列
```

整数数组索引返回 **copy**，不共享数据。

### 分步法：不连续行列组合

```python
x = torch.randn(5, 3)
rows = torch.tensor([0, 2, 3])
cols = torch.tensor([0, 2])
x[rows][:, cols].shape  # (3, 2) — 分两步取不连续行列
```

---

## 1.4 常用运算

### 逐元素运算

```python
x + 1      # 加减乘除基本运算
x ** 2     # 幂运算
torch.sqrt(x)  # 开方
x.abs().log()  # 链式调用
```

### 归约运算

```python
x.sum()              # 全部求和 → scalar
x.sum(dim=0)        # 按 dim=0 求和 → (3,)
x.mean()             # 均值
x.max()              # 最大值 → scalar
x.argmax()            # 最大值索引（flatten后）
x.max(dim=0)         # 返回 (values, indices)
```

### 矩阵运算

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)
(a @ b).shape   # (3, 5)
```

**推荐 `@` / `matmul`**，支持多维 + 广播，是全能版。

### 广播机制

从左补1，从右对齐检查。

```python
x = torch.randn(3, 4)
y = torch.randn(4)       # 1D
x + y  # y补成(1,4) → broadcast到(3,4) ✅
```

### torch.where

```python
torch.where(x > 0, x, 0)  # x>0保留x，否则填0（限幅）
```

### in-place 操作

下划线 `_` 结尾 = in-place，直接修改原 tensor。

```python
x.add_(1)           # x = x + 1，原地改
x.zero_()           # 全部变成 0
```

---

## 1.5 GPU 迁移与设备

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(10, device=device)
model = model.to(device)
```

---

## 1.6 与 NumPy 互转

```python
arr = x.numpy()              # Tensor → NumPy（共享内存时有问题）
x = torch.from_numpy(arr)    # NumPy → Tensor（共享内存）
x = tensor.numpy()           # 需要 requires_grad=False 或 detach
```

---

## 1.7 默认值总结

| 方法 | device | requires_grad | dtype |
|---|---|---|---|
| `torch.Tensor()` | cpu | False | float32 |
| `torch.tensor()` | cpu | False | 自动推断 |
| `torch.zeros/ones()` | cpu | False | float32 |
| `torch.from_numpy()` | cpu | False | 跟随numpy |

---

## 代码练习

```python
import torch
import numpy as np

# 1. torch.Tensor() vs torch.tensor() — 类型推断差异
t1 = torch.Tensor([1, 2, 3])       # 固定 float32，值变成 1.0, 2.0, 3.0
t2 = torch.tensor([1, 2, 3])       # 自动推断 int64

# 2. 拷贝验证
lst = [1., 2., 3.]
t = torch.Tensor(lst)
t[0] = 99.
print("原始 list 不变:", lst)  # 不受影响

# 3. from_numpy() 共享内存
arr = np.array([1., 2., 3.])
t_share = torch.from_numpy(arr)
t_share[0] = 99.
print("from_numpy 改了 numpy:", arr)  # [99., 2., 3.]

# 4. view vs reshape
x = torch.randn(2, 3)
x_t = x.transpose(0, 1)
try:
    x_t.view(6)
except RuntimeError as e:
    print("view 不连续报错:", e)
print("reshape 不连续:", x_t.reshape(6))

# 5. stride 验证
x = torch.randn(2, 3, 4)
print("原始 stride:", x.stride())  # (12, 4, 1)

# 6. 连续性判断
x = torch.randn(2, 3)
print("is_contiguous:", x.is_contiguous())
x_t = x.transpose(0, 1)
print("transpose 后 is_contiguous:", x_t.is_contiguous())
```
