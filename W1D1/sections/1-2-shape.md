# 1.2 形状操作

## 核心概念：Tensor 的内存结构

理解形状操作，先要理解 Tensor 在内存里是怎么存的。

### 栈 vs 堆（补充知识）

**为什么需要两个区域？**

物理内存就一块，没有内部结构，但在逻辑上划分成不同区域来管理：

| 区域 | 分配方式 | 速度 | 空间 | 生命周期 |
|---|---|---|---|---|
| 栈（Stack） | 编译器自动分配/释放 | ⚡ 快 | 小（1-8MB） | 函数进出就销毁 |
| 堆（Heap） | 手动(malloc/free)或GC | 🐢 较慢 | 大（几十GB） | 手动/GC管理 |

```java
void foo() {
    int a = 10;           // a 在栈上，存的是值 10
    int[] arr = {1,2,3};  // arr 引用在栈上，数组对象在堆上
    Person p = new Person();  // p 引用在栈上，Person 对象在堆上
}   // 函数结束，栈上的 a 和 arr 和 p 全都没了，堆上的对象如果没人引用则被GC回收
```

**本质**：栈和堆是操作系统和运行时软件层面的逻辑划分，不是硬件的物理分区。

### Tensor 的内存布局

```
栈（Stack）                 堆（Heap）
┌──────────┐              ┌─────────────────────────────┐
│    x     │ ──────────→  │  Tensor 对象                  │
│ (引用)    │              │  ├── shape: (2, 3, 4)         │
└──────────┘              │  ├── stride: (12, 4, 1)      │
                          │  ├── dtype: float32          │
                          │  └── storage ──────────┐     │
                          └────────────────────────┼─────┘
                                                  ↓
                                              ┌────────┐
                                              │ 数据    │
                                              │ (连续)  │
                                              └────────┘
```

**关键理解**：
- 栈上只存**变量引用**（指针），不存真实数据
- Tensor 对象和 storage **都在堆上**
- Tensor 本质 = **一维数组 + 维度解释规则（shape + stride）**
- view 操作 = 换了一套维度解释规则，数据本身**完全没动**

---

## stride 详解

### stride 是什么

stride 是"跳读规则"，决定按当前维度顺序读取 storage 时，每个维度内跳几个元素才能读完。

```python
x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)

x.stride()  # → (12, 4, 1)

# 含义：
# - stride[0] = 12：跳 12 个元素 → 读完当前"页"，进入下一页
# - stride[1] = 4：跳 4 个元素  → 读完当前"行"，进入下一"行"
# - stride[2] = 1：跳 1 个元素  → 读完当前"列"，进入下一个元素
```

把 `(2, 3, 4)` 想象成 2 页纸，每页 3 行，每行 4 个元素：
```
Page 0: [ 0~ 3]  [ 4~ 7]  [ 8~11]   ← 3行
Page 1: [12~15]  [16~19]  [20~23]   ← 3行
```

### 连续 stride 的计算公式

**连续 stride 公式**：
```
stride[i] = shape[i+1] × shape[i+2] × ... × shape[last]

对于 (2, 3, 4)：
  stride[0] = 3 × 4 = 12
  stride[1] = 4     = 4
  stride[2] = 1     = 1
  → (12, 4, 1) ✅ 连续
```

### 理解 view 的读取顺序

**view 无论 reshape 成什么形状，读数据时始终从最后一个维度开始按顺序读**，就像按行优先（row-major）依次扫描 storage。

对于 `(A, B, C)` → view 成 `(X, Y)`：
- 先按 C 个一组读（最内层）
- 然后 B 组 B 组输出
- 最后 A 组 B 组 C 组的这样输出

本质上 view 就是**用新的一套 shape/stride 来重新分格同一块 storage**，数据顺序不变。

---

## view

**本质**：换一套 shape/stride 来解释同一块 storage，**数据完全不动**。

```python
x = torch.randn(2, 3, 4)  # (2, 3, 4), stride (12, 4, 1)
x.view(6, 4)   # (6, 4), stride (4, 1) — 数据完全不动
x.view(24,)    # (24,)
x.view(1, 2, 3, 4)  # 任意维度都可以，只要 1×2×3×4 = 24
x.view(-1)     # (24,) — -1 自动推断
```

**view 可以任意维度**，不限于二维。

**要求**：tensor 必须连续。不连续时报错：
```python
x_t = x.transpose(0, 1)  # stride (4, 12, 1) — 不连续
x_t.view(24)   # ❌ RuntimeError
```

---

## transpose

**本质**：交换两个维度的位置，**数据不动**，只换 shape 和 stride。

```python
x = torch.randn(2, 3)        # shape (2, 3), stride (3, 1)
x_t = x.transpose(0, 1)     # shape (3, 2), stride (1, 3)

# storage 还是 [0, 1, 2, 3, 4, 5]，按新 stride 读要跳着读了
```

### stride 的快速计算方法

**transpose 后 stride 的算法：直接交换对应位置的 stride 值，不是重新算。**

```
原始 stride: (12, 4, 1)
           dim 0, dim 1, dim 2

transpose(0, 1) = 交换 dim 0 和 dim 1 的位置

交换后:      (4, 12, 1)
           dim 1, dim 0, dim 2
```

验证：
```python
x = torch.randn(2, 3, 4)           # stride (12, 4, 1)
x.transpose(0, 1)                   # stride (4, 12, 1) ✅
```

**注意**：transpose 后 tensor 必然不连续，因为 stride 顺序和 storage 物理顺序对不上了。

---

## reshape

**本质**：等价于 `contiguous().view()`。

```python
# 连续 tensor → 直接 view，不复制
x.reshape(6, 4)  # → view(6, 4)

# 不连续 tensor → 先复制重排，再 view
x_t.reshape(6,)  # → contiguous() 复制数据 + view(6,)
```

**优先用 `reshape`**，永远不报错。

### reshape 对不连续 tensor 做了什么（详细拆解）

```python
x = torch.randn(2, 3)       # stride (3, 1), 连续
x_t = x.transpose(0, 1)   # shape (3, 2), stride (1, 3), 不连续
x_t.reshape(6,)            # 做了什么？
```

步骤：
1. **contiguous()**：申请新 storage，按当前遍历顺序重排数据
   - x_t 按顺序读：[x_t[0,0], x_t[0,1], x_t[1,0], x_t[1,1], x_t[2,0], x_t[2,1]]
   - 物理上重新抄一遍，变成连续的排列
2. **view(6,)**：新 shape (6,)，stride (1,)

所以 reshape 的本质：**复制并按遍历顺序重排数据 + 换 shape/stride**。

| | 数据 | stride | shape |
|---|---|---|---|
| 连续 tensor reshape | ❌ 不变 | ✅ 重新计算 | ✅ 变成新的 |
| 不连续 tensor reshape | ✅ 复制重排 | ✅ 变成新的连续 stride | ✅ 变成新的 |

---

## permute

### 入参语义：维度索引序列

`permute(dim0, dim1, ...)` 入参个数 = 维度个数，每个入参的值 = 旧维度索引。

**语义：新维度 i 来自旧维度 dim_i**

```python
x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
#                          dim:    0    1    2

x.permute(2, 0, 1)  # shape: (4, 2, 3)
# 新dim0 = 旧dim2 → 取原来最内层的4
# 新dim1 = 旧dim0 → 取原来的2
# 新dim2 = 旧dim1 → 取原来的3
```

### 和 transpose 的关系

`transpose(dimA, dimB)` 只交换两个维度；`permute` 是多维推广：

```python
# 等价：
x.transpose(0, 2)
x.permute(2, 1, 0)

# permute 一次换多个，transpose 要串用：
x.permute(2, 0, 1)
# 等价于：
x.transpose(0, 1).transpose(0, 2)
```

### stride 变化规律

和 transpose 完全一致——**按同样索引直接重新排列 stride**：

```python
x = torch.randn(2, 3, 4)  # stride: (12, 4, 1)
x.permute(1, 0, 2).stride()  # (4, 12, 1)
```

### 数据完全不动

和 `transpose` 一样，storage 物理顺序不变，只改 shape/stride 元数据。

### ⚠️ 连续性问题：permute 后不能直接接 view

**`permute` 后必然不连续**，直接 `.view()` 会报错：

```python
x = torch.randn(2, 3, 4)
y = x.permute(1, 0, 2)
y.view(24)  # ❌ RuntimeError

# 正确做法：用 reshape
y.reshape(24)  # ✅ reshape 内部自动处理不连续情况
```

### ⚠️ 维度数必须完整匹配

```python
x = torch.randn(2, 3, 4)  # 3个维度
x.permute(1, 0)     # ❌ RuntimeError: number of dims don't match
x.permute(1, 0, 2)   # ✅ 3个参数，一个不能多一个不能少
```

### 实用场景：结合 reshape 换维 + 展平

`permute` 常用于 HWC → CHW 格式转换，然后展平：

```python
# 假设 feature 是 (B, H, W, C) 格式（4维）
B, H, W, C = 2, 7, 7, 64
x = torch.randn(B, H, W, C)  # (2, 7, 7, 64)

# permute 换维：C 移到最后 → (B, C, H, W)
x_perm = x.permute(0, 3, 1, 2)   # (2, 64, 7, 7)

# reshape 展平：永远不报错
x_flat = x_perm.reshape(B, -1)   # (2, 3136)
```

> 注意：这里必须用 `reshape`，不能用 `view`。permute 之后 tensor 不连续，view 会报 RuntimeError。

### 核心结论

| 性质 | 说明 |
|------|------|
| 入参 | 维度索引，新维度i = 旧维度dim_i |
| 数据动了吗？ | ❌ 没动 |
| stride 变了？ | ✅ 按同样索引重新排列 |
| 连续性 | 必然不连续，后续用 `reshape` |

---

## NumPy 形状操作（补充）

### `np.transpose(a, axes=None)`

**默认**：反转所有维度。**指定 axes**：axes[i] = 新维度i 对应到旧维度索引（和 PyTorch `permute` 完全一致）。

```python
import numpy as np

x = np.random.randn(2, 3, 4)  # shape: (2, 3, 4)

x.transpose().shape            # (4, 3, 2) — 默认反转所有维度
x.transpose(2, 0, 1).shape    # (4, 2, 3)
# 新dim0=旧dim2, 新dim1=旧dim0, 新dim2=旧dim1

# 返回视图，不改变原数据
print(x.shape)  # (2, 3, 4) — X 不变
```

### `np.moveaxis(a, source, destination)`

**直观定位**：把 source 位置的轴挪到 destination，其他轴顺序自动调整。

```python
x = np.random.randn(2, 3, 4, 5)  # 4个维度

np.moveaxis(x, 2, 0).shape   # (4, 2, 3, 5) — dim2挪到最前面
np.moveaxis(x, 0, 3).shape   # (3, 4, 5, 2) — dim0挪到最后
```

比 `transpose` 更直观：不需要想"新维度顺序是什么"，只需要说"把哪个轴挪到哪里"。

### `np.swapaxes(a, axis1, axis2)`

只交换两个轴，最简单：

```python
x = np.random.randn(2, 3, 4)

np.swapaxes(x, 0, 2).shape   # (4, 3, 2) — 交换 dim0 和 dim2
np.swapaxes(x, 0, 1).shape   # (3, 2, 4) — 交换 dim0 和 dim1
```

### 三者对比

| 函数 | 入参语义 | 适用场景 |
|------|---------|---------|
| `transpose(axes=(...))` | axes[i] = 新维度i 对应到旧维度索引 | 完整维度重排 |
| `moveaxis(src, dst)` | 把 src 轴挪到 dst 位置 | 单轴移动，最直观 |
| `swapaxes(a, b)` | 交换两个轴 | 两个轴互换 |

**三者都不改变原数据，返回视图。**

## squeeze / unsqueeze

### `squeeze`：删除 size=1 的维度

```python
x = torch.randn(1, 3, 1, 8, 1)  # 5个维度

x.squeeze().shape   # (3, 8) — 删除所有 size=1 的维度
x.squeeze(0).shape  # (3, 1, 8, 1) — 删最外层（size=1）
x.squeeze(2).shape  # (1, 3, 8, 1) — 删中间层（size=1）
x.squeeze(4).shape  # (1, 3, 1, 8) — 删最里层（size=1）
```

**删除指定维度时，若该维度 size≠1，什么都不删（不报错）：**

```python
x = torch.randn(2, 3)   # shape (2, 3)
x.squeeze(0).shape      # (2, 3) — dim0 不是 size=1，原样返回
```

### `unsqueeze`：在指定位置插入 size=1 的维度

```python
x = torch.randn(2, 3)  # ndim=2

x.unsqueeze(0).shape   # (1, 2, 3) — 在最前面插
x.unsqueeze(1).shape   # (2, 1, 3) — 在 dim1 位置插
x.unsqueeze(2).shape   # (2, 3, 1) — 在最后插
```

**dim 参数范围**：`unsqueeze` 接受 `0` 到 `ndim`（末尾之后也是合法插入点），`squeeze(dim)` 只能是 `0` 到 `ndim-1`：

```python
x = torch.randn(2, 3)   # ndim=2

x.unsqueeze(2).shape    # (2, 3, 1) ✅ — dim=ndim 合法
x.unsqueeze(3).shape   # ❌ IndexError — 超出范围

x.squeeze(2).shape     # ❌ IndexError — dim2 不存在
```

### 共同特性

| 特性 | squeeze | unsqueeze |
|------|---------|-----------|
| 返回视图？ | ✅ 是 | ✅ 是 |
| 复制数据？ | ❌ 不复制 | ❌ 不复制 |
| 原数据 X 变吗？ | ❌ 不变 | ❌ 不变 |
| 新 tensor 连续？ | ✅ 连续 | ✅ 连续 |

### NumPy 对应

```python
import numpy as np

x = np.random.randn(1, 3, 1, 8)

np.squeeze(x).shape              # (3, 8) — 删除所有 size=1
np.squeeze(x, axis=0).shape     # (3, 1, 8) — 指定删除

np.expand_dims(x, axis=0).shape # (1, 1, 3, 1, 8) — 等价于 unsqueeze
```

### 常见应用场景

- **`unsqueeze(0)`**：给单个样本加 batch 维度 `(C,) → (1, C)`，DataLoader 格式统一
- **`squeeze()`**：去掉冗余 batch 维度 `(1, N) → (N,)` 或 `(B, 1, C) → (B, C)`

---

## flatten

### 作用：连续展平多个维度为 1 维

```python
x = torch.randn(2, 3, 4)  # (2, 3, 4)

x.flatten().shape           # (24,) — 所有维度展平
x.flatten(0, 1).shape       # (6, 4) — 只展平 dim0 和 dim1
x.flatten(1, 2).shape       # (2, 12) — 只展平中间两个维度
```

### 参数语义

`flatten(start_dim, end_dim)`：把 `[start_dim, end_dim]` 区间内的所有维度合并成 1 维。

```python
x = torch.randn(B, C, H, W)
x.flatten(1, -1).shape  # (B, C*H*W) — 保留 batch 维，展平所有空间和通道
```

### 本质

**`flatten` 等价于 `reshape(-1)`**（展平所有）或 `reshape(..., -1)`（区间展平）：

```python
x = torch.randn(2, 3, 4)
x.flatten().shape        # (24,)
x.reshape(-1).shape      # (24,) — 完全等价

x.flatten(1, 2).shape   # (2, 12)
x.reshape(2, -1).shape  # (2, 12) — 完全等价
```

**内部实现**：调用 `.contiguous().view(...)`，所以永远不会报错。

### 不改变原数据

```python
x = torch.randn(2, 3, 4)
y = x.flatten()
print(x.shape)  # (2, 3, 4) — X 不变
```

---

## 三者对比

| | view | transpose | reshape |
|---|---|---|---|
| 数据动了吗？ | ❌ 没动 | ❌ 没动 | ✅ 不连续时会复制重排 |
| stride 变了吗？ | ✅ 重新计算 | ✅ 直接交换位置 | ✅ 先重排再计算 |
| 连续性 | 要求连续 | 必然不连续 | 永远成功 |
| 适用场景 | 确定连续时 | 换维度视角 | 不确定连续性时 |

---

## is_contiguous() 的判断逻辑

**本质**：检查"按照当前 stride 顺序遍历 storage，能否一次不跳地读完所有元素"。

**判断标准**：当前 stride 是否等于 shape 右侧累积乘积

```
连续条件：stride[i] = shape[i+1] × shape[i+2] × ... × shape[last]  对所有 i 成立
```

```python
x = torch.randn(2, 3)        # stride (3, 1)
# 右侧累积乘积：(1,) → stride[1] = 1 ✅
# → 连续 ✅

x.transpose(0, 1)            # stride (1, 3)
# 右侧累积乘积：(3,) → stride[1] = 3 ✅，stride[0] = 1 ≠ 3 ❌
# → 不连续 ❌
```

### 快速判断方法

**stride[-1] = 1 是必要条件**：最内层维度在 storage 里必须相邻。但**不是充分条件**。

```python
# 假阳性的例子：
x = torch.randn(3, 2, 4)  # stride (8, 4, 1)
# stride[-1] = 1 ✅ 但 stride[0] = 8 ≠ 2×4 = 8 ✅
# 所以其实这个是连续的

# 反例：
x = torch.randn(2, 3).transpose(0, 1)  # stride (1, 3)
# stride[-1] = 3 ≠ 1 ❌ → 不连续
```

总结：完整判断必须检查所有维度，不能只看最后一位。

---

## 面试必答题

1. **view 和 reshape 的区别？**
   - `view` 要求连续，不连续时报错；`reshape` 自动处理不连续情况
2. **transpose 后 tensor 是否连续？**
   - 不连续，因为内存布局和维度顺序不匹配
3. **squeeze 和 unsqueeze 会复制数据吗？**
   - 不会，只是返回改变维度索引的 view

---

# ===== 代码练习 =====

```python
import torch

# 1. view vs reshape
x = torch.randn(2, 3)
print("原 tensor:")
print(x)
print("view(6):", x.view(6))
print("is_contiguous:", x.is_contiguous())

# transpose 后不连续
x_t = x.transpose(0, 1)
print("\ntranspose 后 is_contiguous:", x_t.is_contiguous())

try:
    x_t.view(6)
except RuntimeError as e:
    print("view 不连续报错:", e)

print("reshape 不连续:", x_t.reshape(6))

# 2. transpose 和 permute
x = torch.randn(2, 3, 4)
print("\n原始 shape:", x.shape)
print("原始 stride:", x.stride())

x_t = x.transpose(0, 1)  # 交换 dim 0 和 dim 1
print("\ntranspose(0,1) 后 shape:", x_t.shape)
print("transpose(0,1) 后 stride:", x_t.stride())

x_p = x.permute(1, 0, 2)  # 等价于 transpose(0,1)
print("\npermute(1,0,2) 后 shape:", x_p.shape)
print("permute(1,0,2) 后 stride:", x_p.stride())

# 3. squeeze / unsqueeze
x = torch.randn(1, 3, 1, 8)
print("\n原始 shape:", x.shape)
print("squeeze 后:", x.squeeze().shape)
print("squeeze(0) 后:", x.squeeze(0).shape)
print("unsqueeze(0) 后:", x.unsqueeze(0).shape)

# 4. flatten
x = torch.randn(2, 3, 4)
print("\nflatten 展平:", x.flatten().shape)
print("flatten(1,2) 展平:", x.flatten(1, 2).shape)

# 5. 连续性判断
x = torch.randn(2, 3)
print("\nis_contiguous:", x.is_contiguous())
x_t = x.transpose(0, 1)
print("transpose 后 is_contiguous:", x_t.is_contiguous())
print("transpose 后 contiguous():", x_t.contiguous().is_contiguous())

# 6. stride 验证
x = torch.randn(2, 3, 4)
print("\n原始 stride:", x.stride())
print("右侧累积乘积验证：(4, 1) × 3 = (12, 4, 1) =", (3*4, 4, 1))
```
