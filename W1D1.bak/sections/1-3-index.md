# 1.3 索引切片

## 基础索引 + 切片

```python
x = torch.randn(2, 3, 4)

x[0].shape         # (3, 4) — 取第一个样本
x[0, 1].shape      # (4,) — 取特定元素（标量）
x[0:2].shape       # (2, 3, 4) — 切片
x[:, 1:].shape     # (2, 2, 4) — 跨维度切片
```

### 负索引

```python
x = torch.randn(5, 3)
x[-1].shape    # (3,) — 最后一行，等价于 x[4]
x[:, -1].shape # (5,) — 最后一列
```

---

## 省略号 `...`

`...` 表示"所有维度全切片"：

```python
x = torch.randn(2, 3, 4, 5)
x[0, ..., 2].shape   # (3, 4) — 固定 dim0，其余全切片
x[..., 2].shape      # (2, 3, 4) — 所有维度全切片，只留最后位置
```

---

## `None`（等价于 unsqueeze）

`None` 在索引里等价于 `unsqueeze`，在指定位置插入 size=1 的维度：

```python
x = torch.randn(2, 3)

x[None, :].shape   # (1, 2, 3) — 等价于 x.unsqueeze(0)
x[:, None].shape   # (2, 1, 3) — 等价于 x.unsqueeze(1)
x[:, None, None].shape  # (2, 1, 1, 3)
```

`None` 返回 **view**，不复制数据。

---

## view vs copy 的分界线（核心！）

**返回 view 的操作（共享底层 storage，修改会影响原 tensor）：**
- 基础切片索引：`x[0:2]`、`x[:, 1:]`、`x[..., 0]`
- `None` 索引：`x[None, :]`
- `squeeze` / `unsqueeze`
- `transpose` / `permute`
- `view`（tensor 连续时）
- `expand`

**返回 copy 的操作（独立新 storage，修改不影响原 tensor）：**
- `clone()`
- 布尔掩码索引：`x[mask]`
- 整数数组索引：`x[rows]`
- `torch.masked_select`
- `torch.index_select`

**验证方法：**
```python
x = torch.randn(2, 3)
y = x[0:1]           # view
z = x[x > 0]          # copy

print(x.storage().data_ptr() == y.storage().data_ptr())  # True → view
print(x.storage().data_ptr() == z.storage().data_ptr())  # False → copy
```

---

## 布尔掩码

```python
x = torch.randn(3, 4)
mask = x > 0

x[mask].shape              # (?,) — 展平成一维
torch.masked_select(x, mask).shape  # 同上，完全等价
```

布尔掩码返回 **copy**，不共享数据。

---

## 整数数组索引

### 基本用法

```python
x = torch.randn(5, 3)
rows = torch.tensor([0, 2, 3])

x[rows].shape        # (3, 3) — 取指定行
x[rows, 1].shape     # (3,) — 取这些行的第二列（标量）
x[rows, 1:2].shape   # (3, 1) — 保持维度
```

整数数组索引返回 **copy**，不共享数据。

### 两个整数数组同时用：配对取标量

```python
x = torch.randn(5, 3)
rows = torch.tensor([0, 2, 3])
cols = torch.tensor([0, 1, 2])

x[rows, cols].shape  # (3,) — [x[0,0], x[2,1], x[3,2]]，配对取标量
```

**长度必须相同**，不同则直接 IndexError（不会自动 broadcast）：

```python
rows = torch.tensor([0, 2, 3])  # size 3
cols = torch.tensor([0, 2])      # size 2

x[rows, cols]
# IndexError: shape mismatch: indexing tensors could not be broadcast together with shapes [3], [2]
```

### 分步法：不连续行列组合（重要！）

**用整数数组取指定行 + 用另一个整数数组取指定列，必须分两步：**

```python
x = torch.randn(5, 3)
rows = torch.tensor([0, 2, 3])
cols = torch.tensor([0, 2])

x[rows][:, cols].shape  # (3, 2) — 取这三行的第一列和第三列
```

**不能用 `x[rows, cols]`，那样会配对取标量，结果是一维。**

原理：
1. `x[rows]` → shape (3, 3)，取了指定行
2. `[:, cols]` → 再从中取指定列

两步各操作一个维度，互不干扰。

---

## in-place 赋值与 view/copy 的关系

```python
x = torch.randn(2, 3)
y = x[0:1, :]      # view，共享 storage
z = x[x > 0]        # copy，独立 storage

y[:] = 0            # ✅ 会修改 x 的底层数据（view 的 in-place 赋值）
z[:] = 0            # ❌ 只修改 z，x 不变（copy 的 in-place 赋值）
```

---

## torch.index_select / torch.gather（补充）

```python
x = torch.randn(5, 3)
idx = torch.tensor([0, 2, 3])

torch.index_select(x, dim=0, index=idx).shape   # (3, 3)
```

`index_select` 是整数数组索引的函数形式，语义一致——返回 **copy**。

---

## 代码练习

```python
import torch

# 1. 基础索引
x = torch.randn(2, 3, 4)
print("x[0]:", x[0].shape)
print("x[0:2]:", x[0:2].shape)
print("x[:, 1:]:", x[:, 1:].shape)

# 2. None（unsqueeze）
x = torch.randn(2, 3)
print("\nx[None, :]:", x[None, :].shape)
print("x[:, None]:", x[:, None].shape)

# 3. 连续性验证：view vs copy
x = torch.randn(2, 3)
y = x[0:1]           # view
z = x[x > 0]          # copy
print("\nview data_ptr 相同:", x.storage().data_ptr() == y.storage().data_ptr())
print("copy data_ptr 相同:", x.storage().data_ptr() == z.storage().data_ptr())

# 4. 整数数组索引
x = torch.randn(5, 3)
rows = torch.tensor([0, 2, 3])
print("\nx[rows]:", x[rows].shape)

# 5. 分步法取不连续行列
cols = torch.tensor([0, 2])
print("x[rows][:, cols]:", x[rows][:, cols].shape)

# 6. 负索引
x = torch.randn(5, 3)
print("\nx[-1]:", x[-1].shape)
print("x[:, -1]:", x[:, -1].shape)
```
