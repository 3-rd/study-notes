# 1.4 常用运算

## 逐元素运算（Element-wise）

```python
x = torch.randn(2, 3)

x + 1      # 加减乘除基本运算
x - 1
x * 2
x / 2
x ** 2     # 幂运算
torch.sqrt(x)  # 开方
x.abs().log()  # 链式调用
```

---

## 归约运算（Reduction）

```python
x = torch.randn(2, 3)

x.sum()              # 全部求和 → scalar
x.sum(dim=0)        # 按 dim=0 求和 → (3,)
x.sum(dim=[0, 1])   # 多个维度 → ()
x.mean()             # 均值
x.mean(dim=0)        # 按维度均值
x.max()              # 最大值 → scalar
x.argmax()            # 最大值索引（flatten后）
x.max(dim=0)         # 返回 (values, indices)
x.min() / x.argmin() # 同理
```

---

## 矩阵运算

### matmul / @（推荐）

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)
(a @ b).shape   # (3, 5)

# 3D batch matmul：只对最后两维做矩阵乘法，前面维度广播
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
(a @ b).shape   # (2, 3, 5)
```

**为什么推荐 `@` / `matmul`：**
- `mm` 只支持 2D，3D 直接报错
- `matmul` 支持多维 + 广播，是全能版

### torch.mm（不推荐）

```python
torch.mm(a, b)  # 只支持 2D，多维报错
```

---

## 比较运算

```python
x = torch.randn(2, 3)

(x > 0).float()   # 转成 0/1
(x > 0).all()     # 全部 True ？
(x > 0).any()     # 存在 True ？
(x == y).sum()    # 统计相等个数
```

---

## 广播机制

### 规则：从左补1，从右对齐检查

**两步走：**
1. 维度数量不同 → 左边补 1
2. 每个维度 → 要么相等，要么有一边为 1，否则报错

```python
# 补1的例子
x = torch.randn(3, 4)
y = torch.randn(4)       # 1D
x + y  # y补成(1,4) → broadcast到(3,4) ✅

# 报错例子
x = torch.randn(3, 4)
y = torch.randn(3)
x + y  # y补成(1,3) → 变成(1,3)，4≠3 ❌ 报错

# 成功的3D例子
x = torch.randn(    3, 4)  # (3,4)
y = torch.randn(2, 4, 5)   # (2,4,5)
(x @ y).shape             # (2,3,5) — x被broadcast
```

---

## 条件选择：torch.where

```python
x = torch.randn(3, 4)
torch.where(x > 0, x, 0)  # x>0保留x，否则填0（限幅）
# 等价于：torch.clamp(x, min=0)
```

---

## 限幅：torch.clamp

```python
x = torch.randn(3, 4)
x.clamp(0, 1)       # 下界0，上界1
x.clamp(min=0)      # 只设下界
x.clamp(max=1)      # 只设上界
```

---

## in-place 操作

**下划线 `_` 结尾 = in-place，直接修改原 tensor，不分配新内存。**

```python
x = torch.randn(2, 3)

x.add_(1)           # x = x + 1，原地改
x.copy_(y)          # 把 y 的值拷贝进 x
x.fill_(0)          # 全部变成 0
x.zero_()           # 全部变成 0
x.mul_(2)           # x = x * 2
x.relu_()           # x = max(x, 0)
```

### 普通操作 vs in-place 对比

| | 普通操作 `x = x.add(1)` | in-place `x.add_(1)` |
|--|------------------------|---------------------|
| 原 tensor 数据 | ❌ 不变（新建） | ✅ 改变 |
| 新内存分配 | ✅ 分配 | ❌ 不分配 |
| 变量 x 引用 | 变了（指向新对象） | 不变 |

```python
x = torch.randn(2, 3)
y = x
x.add_(1)          # x 的底层数据被改，y 也能看到
print(x.sum())     # 值变了

x = torch.randn(2, 3)
y = x
x = x.add(1)       # 新建tensor，x指向新对象，y不变
print(y.sum())      # 值没变
```

### 使用建议

- **用 in-place**：显存紧张时，不需要保留原数据时
- **不用 in-place**：需要保留原数据，或在 autograd 图里（DDP 多卡训练时也慎用）
