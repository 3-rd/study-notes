# 1.1 创建方式详解

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

dtype 跟随 numpy，不统一成 PyTorch 默认值：
- `np.array([1., 2.], dtype=np.float64)` → `torch.float64`（不是常见的 float32）

## torch.zeros() / torch.ones() — 初始化

```python
torch.zeros(3, 4, dtype=torch.float32, device='cuda', requires_grad=True)
torch.ones(2, dtype=torch.int64)
```

## 默认值

所有方法不指定时 → **device=cpu，requires_grad=False**

## 面试必答题

1. `torch.Tensor()` 和 `torch.tensor()` 的区别？
2. `torch.from_numpy()` 的坑？
3. 所有方法输出类型是什么？

# ===== 代码练习 =====

```python
import torch
import numpy as np

# 1. torch.Tensor() vs torch.tensor() — 类型推断差异
t1 = torch.Tensor([1, 2, 3])       # 固定 float32，值变成 1.0, 2.0, 3.0
t2 = torch.tensor([1, 2, 3])       # 自动推断 int64
t3 = torch.tensor([1, 2, 3], dtype=torch.float32)

print("torch.Tensor() dtype:", t1.dtype)  # torch.float32
print("torch.tensor() dtype:", t2.dtype)   # torch.int64
print("强制 float32:", t3.dtype)           # torch.float32

# 2. 拷贝验证
lst = [1., 2., 3.]
t = torch.Tensor(lst)
t[0] = 99.
print("\n原始 list 不变:", lst)  # 不受影响

# 3. from_numpy() 共享内存
arr = np.array([1., 2., 3.])
t_share = torch.from_numpy(arr)
t_share[0] = 99.
print("from_numpy 改了 numpy:", arr)  # [99., 2., 3.]

# 4. dtype 陷阱
arr_f64 = np.array([1., 2., 3.], dtype=np.float64)
t_f64 = torch.from_numpy(arr_f64)
print("\nfrom_numpy float64:", t_f64.dtype)  # torch.float64

# 5. 默认值
t_cpu = torch.tensor([1., 2.])
print("默认 device:", t_cpu.device)         # cpu
print("默认 requires_grad:", t_cpu.requires_grad)  # False

# 6. zeros/ones
z = torch.zeros(3, 4, dtype=torch.int32)
print("zeros dtype:", z.dtype, "shape:", z.shape)
```
