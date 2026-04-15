# 1-5 自动求导（Autograd）

## 1-5-1 计算图与反向传播原理

PyTorch 的自动求导基于**动态计算图**（Dynamic Computation Graph）。

**核心概念**：
- 每当对 `requires_grad=True` 的张量进行运算，PyTorch 会构建一个反向传播的 DAG
- 叶子节点（leaf nodes）是用户直接创建的张量（通过 `torch.tensor()` 等）
- 根节点（root）是最终输出标量（scalar），通常是一个 loss

```python
import torch

# 叶子节点：用户创建，requires_grad 默认为 False
x = torch.tensor([1., 2., 3.], requires_grad=True)  # 叶子
y = x ** 2          # 中间节点，grad_fn 记录运算
z = y.sum()         # 根节点，backward 从这里开始

print("x 是叶子:", x.is_leaf)
print("y 是叶子:", y.is_leaf)
print("z 是叶子:", z.is_leaf)
print("z.grad_fn:", z.grad_fn)  # <SumBackward0>
```

**DAG 的构建是动态的**：每次前向传播都会重新构建计算图，修改代码后图结构随之变化。

```python
# 同一个变量两次前向，建两个图
a = torch.tensor([1., 2.], requires_grad=True)
b = a ** 2
c = b * 2
d = c.sum()
print(d.grad_fn)  # <AddBackward1> — 两个图各自独立
```

---

## 1-5-2 backward() 与梯度计算

`backward()` 从当前张量反向传播到所有叶子节点，计算 `∂tensor/∂leaf`。

**必须条件**：调用 `backward()` 的张量必须是**标量**（scalar，形状为空），或者传入 `gradient` 参数。

```python
import torch

# y = x²，求 dy/dx
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2       # y = [1, 4, 9]
z = y.sum()      # z = 14，必须是标量才能直接 backward()

z.backward()     # 等价于 z.backward(torch.tensor(1.))
print(x.grad)    # tensor([2., 4., 6.]) — dy/dx = 2x

# 验证：x=[1,2,3]，2x = [2,4,6] ✓
```

**非标量如何 backward**：需要传入 `gradient` 参数（与输出同形状），表示链式求导的起点。

```python
# J = [J₁, J₂]，每个 Jᵢ 对 x 求偏导
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2       # y = [1, 4, 9]
# y 有 3 个输出，每个都对 x 求偏导构成雅可比矩阵

v = torch.tensor([1., 0., 0.])  # 只想求 J₁ 对 x 的偏导
y.backward(v)
print(x.grad)     # tensor([2., 0., 0.]) = ∂J₁/∂x = 2x
```

---

## 1-5-2.5 标量 backward vs 向量 backward（gradient 参数详解）

### 核心问题：为什么非标量不能直接 backward？

标量 `loss.backward()` 直接能跑，是因为 PyTorch 知道"从 loss（一个数）往回传梯度，起点就是 1"。

但如果 `y` 是向量，`y.backward()` 会遇到歧义：**y 有多个元素，每个元素对上游参数的梯度都不一样**。这时候 PyTorch 不知道该把什么值传回去，所以直接报错：`grad can be implicitly created only for scalar outputs`。

### 用具体例子理解三种梯度

以一个完整的前向计算图为例：

```
x → [matmul] → y → [减法] → [平方] → [求平均] → loss
                                              ↑
                                         y_true
```

**已知数值**：
```
x = [1, 2]
w = [[1, 3], [2, 4]]
y_true = [10, 20]
```

**前向传播（逐步计算）**：

| 步骤 | 计算 | 结果 |
|------|------|------|
| y = x @ w | `y[0]=1×1+2×2, y[1]=1×3+2×4` | y = [5, 11] |
| diff = y - y_true | `5-10, 11-20` | diff = [-5, -9] |
| square = diff² | `(-5)², (-9)²` | square = [25, 81] |
| loss = mean(square) | `(25+81)/2` | loss = 53 |

**反向传播第一步：dloss/dy**

loss 对 y 求偏导——loss 是标量，y 是向量，所以 dloss/dy 也是向量：

```
loss = ((y[0]-10)² + (y[1]-20)²) / 2

∂loss/∂y[0] = 2×(y[0]-10) / 2 = y[0] - 10 = 5 - 10 = -5
∂loss/∂y[1] = 2×(y[1]-20) / 2 = y[1] - 20 = 11 - 20 = -9

所以 dloss/dy = [-5, -9]  ← 就是 diff 本身！
```

**反向传播第二步：dy/dw（雅可比矩阵）**

y 对 w 求偏导——y 是向量，w 是 2×2 矩阵，所以 dy/dw 是一个 2×2 的雅可比矩阵：

```
y[0] = x[0]×w[0,0] + x[1]×w[1,0] = 1×w[0,0] + 2×w[1,0]
y[1] = x[0]×w[0,1] + x[1]×w[1,1] = 1×w[0,1] + 2×w[1,1]

∂y[0]/∂w[0,0] = 1    ∂y[0]/∂w[0,1] = 0
∂y[0]/∂w[1,0] = 2    ∂y[0]/∂w[1,1] = 0
∂y[1]/∂w[0,0] = 0    ∂y[1]/∂w[0,1] = 1
∂y[1]/∂w[1,0] = 0    ∂y[1]/∂w[1,1] = 2

dy/dw = [[1, 0],   ← y[0] 对 w 的偏导
         [2, 0],
         [0, 1],   ← y[1] 对 w 的偏导
         [0, 2]]
```

注意：y[i] 只和 w[:, i]（第 i 列）有关，和 w[:, 1-i] 无关，所以每列只有一个元素有值。

**反向传播第三步：dloss/dw = dloss/dy · dy/dw（链式法则）**

链式法则公式：`∂loss/∂w[i,j] = ∂loss/∂y[j] × ∂y[j]/∂w[i,j]`

| | 第 0 列 (j=0) | 第 1 列 (j=1) |
|--|---------------|---------------|
| 第 0 行 (i=0) | `(-5) × 1 = -5` | `(-9) × 1 = -9` |
| 第 1 行 (i=1) | `(-5) × 2 = -10` | `(-9) × 2 = -18` |

```
dloss/dw = [[-5, -9],
             [-10, -18]]
```

### 标量 backward vs 向量 backward 的本质区别

| | `loss.backward()` | `y.backward(gradient)` |
|--|--|--|
| 起点梯度 | 自动假设 = 1（标量的上游梯度就是 1） | 你手动指定 upstream |
| 传播范围 | 从当前节点一直传到最后（叶子节点） | 只到当前节点为止 |
| 典型场景 | 训练神经网络算 loss 梯度 | 策略梯度、手动梯度构造 |

**`y.backward(gradient=[1,1])` 的实际含义**：

```
y → [matmul] → ... → loss
↑
手动传入 [1,1]，跳过 y 之后的所有计算
```

传入 `[1,1]` 相当于告诉 PyTorch："从 y 往上游传的时候，就传 [1,1] 就行了，不需要再算 loss → y 那一段的梯度了"。

也就是说：**把本应该从 loss 计算到 y 时产生的梯度（-5, -9），替换成你手动输入的值（1, 1）**。

```python
# 等价于：手动替换了 dloss/dy 本来的值 [-5, -9]
y.backward(gradient=torch.tensor([1.0, 1.0]))
# dloss/dw[j,k] = upstream[k] × x[j]
# 结果 = [[1, 1], [2, 2]]（而不是 [[-5,-9], [-10,-18]]）
```

---

## 1-5-3 梯度累加机制（关键！）

**默认行为：梯度是累加的，不是覆盖。**

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)

# 第一次 backward
y = (x ** 2).sum()
y.backward()
print("第1次:", x.grad)   # tensor([2., 4., 6.])

# 第二次 backward — 梯度会累加！
y2 = (x ** 2).sum()
y2.backward()
print("第2次:", x.grad)   # tensor([4., 8., 12.]) — 累加了！

# 正确的多步训练：每次前向之间清梯度
x.grad.zero_()           # 原地清零
y3 = (x ** 2).sum()
y3.backward()
print("清零后:", x.grad)   # tensor([2., 4., 6.]) — 正确
```

**训练循环中的标准写法**：

```python
for data, target in dataloader:
    optimizer.zero_grad()   # Step 1: 清梯度
    output = model(data)     # Step 2: 前向
    loss = criterion(output, target)
    loss.backward()          # Step 3: 反向
    optimizer.step()         # Step 4: 更新参数
```

**为什么不清零会累加？**
- `backward()` 只会**累加**梯度到 `.grad`，不会覆盖
- 这是为了支持**梯度累积**（gradient accumulation）—— 用小 batch 模拟大 batch

```python
# 梯度累积：模拟 batch_size=64，实际用两个 batch_size=32
model.zero_grad()           # 清零
loss1 = model(batch1).sum() / 2  # 32
loss1.backward()             # 梯度 /2 累加
loss2 = model(batch2).sum() / 2  # 32
loss2.backward()             # 梯度 /2 累加
# 等效于 batch_size=64 的梯度
```

---

## 1-5-4 no_grad / set_grad_enabled / eval

### torch.no_grad()

**作用**：禁用梯度计算，不构建计算图，节省显存和计算量。

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2

with torch.no_grad():
    z = y * 2      # 这里不追踪梯度
    print("no_grad 中:", z.requires_grad)  # False

print("no_grad 外:", z.requires_grad)  # False — 不影响外面的变量

# 用于推理/评估阶段
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

**另一个写法**：`@torch.no_grad()` 装饰器，效果相同。

```python
@torch.no_grad()
def evaluate(model, data):
    return model(data)
```

### torch.set_grad_enabled()

**作用**：动态切换梯度追踪状态。

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)

torch.set_grad_enabled(False)
y1 = x * 2
print("关闭后:", y1.requires_grad)  # False

torch.set_grad_enabled(True)   # 重新开启
y2 = x * 2
print("开启后:", y2.requires_grad)  # True
```

### model.eval() vs model.train()

| 模式 | 作用 | 影响 |
|---|---|---|
| `train()` | 训练模式 | BatchNorm 用 batch 统计量，Dropout 生效 |
| `eval()` | 评估模式 | BatchNorm 用全局统计量（moving average），Dropout 不生效 |

```python
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    loss = ...
    loss.backward()

model.eval()                    # 切换
with torch.no_grad():
    for batch in val_loader:
        ...
```

**注意**：`eval()` 不等于 `no_grad()`！两者针对不同问题，要叠加使用：

```python
model.eval()
with torch.no_grad():           # 双重保险
    val_loss = ...
```

---

## 1-5-5 detach() 截断计算图

**作用**：将张量从计算图中分离出来，创建一个**共享存储但不共享梯度追踪**的新张量。

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2                # y 仍在图中
z = y.detach()            # z 与 y 共享数据，但脱离梯度追踪

print("z.requires_grad:", z.requires_grad)  # False
print("y.requires_grad:", y.requires_grad)  # True

# z 可以正常计算，但不会影响 x 的梯度
z_new = z * 2             # 无梯度追踪
```

**典型用途**：需要对一个张量的值进行原地操作，又不想破坏原始计算图。

```python
# 错误示范：直接修改需要梯度的张量
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
x[0] = 10                   # RuntimeError: a view of a variable

# 正确做法：先 detach，再修改
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
x_detached = x.detach()
x_detached[0] = 10          # OK
print(x_detached)           # tensor([10., 2., 3.])
```

**detach() vs no_grad()**：
- `no_grad()`：**整个代码块**都不追踪梯度
- `detach()`：**单个张量**从图中分离出来

---

## 1-5-6 hook 机制（进阶）

**作用**：在不需要修改前向/反向代码的情况下，**拦截**前向传播或反向传播的过程，查看或修改中间张量/梯度。

### 注册前向 hook

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

def forward_hook(module, input, output):
    print(f"输入形状: {input[0].shape}")
    print(f"输出形状: {output.shape}")
    return output  # 可以修改返回值

# 注册：返回 handle，用于取消注册
handle = model.register_forward_hook(forward_hook)

x = torch.randn(2, 10)
y = model(x)

handle.remove()   # 取消注册
```

### 注册反向 hook

```python
def backward_hook(module, grad_input, grad_output):
    print(f"输入梯度形状: {[g.shape for g in grad_input]}")
    print(f"输出梯度形状: {grad_output[0].shape}")
    return grad_input  # 可以修改输入梯度

handle = model.register_backward_hook(backward_hook)

x = torch.randn(2, 10, requires_grad=True)
y = model(x)
loss = y.sum()
loss.backward()
```

### 常用场景

1. **特征提取**：在不修改模型的情况下，获取中间层激活值
2. **梯度检查**：验证梯度计算是否正确
3. **梯度修改**：在反向传播时人为注入噪声、裁剪等

```python
# 完整例子：提取 ResNet 中间层特征
import torchvision.models as models

model = models.resnet18(pretrained=True)
features = {}

def hook_fn(name):
    def fn(module, input, output):
        features[name] = output.detach()
    return fn

model.layer1.register_forward_hook(hook_fn("layer1"))
model.layer2.register_forward_hook(hook_fn("layer2"))
```

---

## 1-5-7 梯度消失与爆炸

### 原因

多层链式求导时，梯度在反向传播中不断连乘：

- **梯度消失**：|∂L/∂W| < 1，连乘后趋近于 0 → 参数几乎不更新
- **梯度爆炸**：|∂L/∂W| > 1，连乘后趋近于 ∞ → 参数大幅震荡

```python
# 演示梯度消失
import torch

x = torch.tensor([0.5], requires_grad=True)
for _ in range(20):
    y = torch.nn.functional.sigmoid(x)
    y = y * 0.01   # 缩小输出
    y.backward()
    print(f"x.grad = {x.grad.item():.6f}")
    x.grad.zero_()
    x = y.detach().requires_grad_(True)
```

### 解决方案

| 方法 | 代码 |
|---|---|
| 梯度裁剪 | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| 残差连接 | `output = x + self.layer(x)` — 梯度直接回传 |
| 归一化 | BatchNorm / LayerNorm — 稳定梯度分布 |
| 激活函数 | ReLU 替代 Sigmoid/Tanh（梯度更稳定） |
| 权重初始化 | `nn.init.kaiming_normal_` / `xavier_uniform_` |
| LSTM/GRU | 门控机制缓解长期依赖的梯度问题 |

```python
# 训练循环中加入梯度裁剪
for epoch in range(E):
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        
        # 裁剪：所有参数梯度的 L2 范数不超过 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
```

---

## 本节面试重点

**Q1: PyTorch 反向传播原理？计算图是怎么构建的？**
- PyTorch 构建动态有向无环图（DAG），`requires_grad=True` 的张量参与运算时自动记录
- `backward()` 从根节点反向遍历，叶子节点的 `.grad` 累加梯度

**Q2: `backward()` 多次调用，梯度累加还是覆盖？**
- 累加！每次 `backward()` 都会把新梯度加到 `.grad` 上
- 所以训练循环中必须 `optimizer.zero_grad()` 清零

**Q3: `no_grad()` vs `eval()` 区别？**
- `no_grad()`：不构建计算图，节省显存
- `eval()`：BN 用全局统计量，Dropout 不生效
- 推理时两者叠加：`model.eval() + with torch.no_grad()`

**Q4: `detach()` 什么时候用？**
- 需要对张量值做原地修改时（叶子节点不能直接修改）
- 需要把计算图中间结果传给不需要梯度的操作时

**Q5: 梯度消失/爆炸的解决？**
- 梯度裁剪（`clip_grad_norm_`）、残差连接、归一化、ReLU 替代 Sigmoid、权重初始化

---

## 1-5-8 梯度本质详解（数学直觉）

### 梯度是什么？

**梯度的物理含义：瞬时变化率（导数）在多维空间的推广。**

对于一元函数 `y = f(x)`，导数 `f'(x)` 描述的是：x 每增加 1 个单位，y 变化多少。

对于多元函数 `L = f(w₁, w₂, ..., wₙ)`，梯度 `∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]` 描述的是：**每个参数 wᵢ 每增加 1 个单位，loss L 变化多少。**

### 梯度为什么能用来更新参数？

**梯度指向的是函数值增大的方向。**

```
L(w) = (w - 5)²，最小值在 w=5
∇L = dL/dw = 2(w - 5)

w=3 时，∇L = -4（负的）
→ 负梯度方向 = 增大方向的反方向
→ w 应该往正方向走（增大）
→ w_new = 3 - lr×(-4) = 3 + lr×4
```

所以：**沿负梯度方向走，函数值下降。** 这就是梯度下降法的核心直觉。

### 链式法则是梯度的核心计算规则

链式法则把复杂函数的梯度拆解为基本运算的梯度组合：

```
L = sin(w²)
设 u = w²，则 L = sin(u)

dL/dw = dL/du · du/dw = cos(u) · 2w = 2w·cos(w²)
```

PyTorch 的 autograd 引擎就是把这个链式法则**自动化执行**：
- 前向传播：真正计算每个节点的数值
- 反向传播：沿着链式法则的路径，把梯度逐级回传

### 梯度的累加本质

梯度是**加性的**——复合函数的梯度等于各部分梯度的和。

```
L = f(g(h(x)))
dL/dx = dL/df · df/dg · dg/dh · dh/dx
```

每一级都在**乘**（链式），最后得到完整梯度。这就是为什么深层网络的梯度容易消失/爆炸——链路上连续乘了很多小于1或大于1的数。

### 为什么梯度是"瞬时"而非"平均"？

```
f(x) = x²，在 x=3 处
- 瞬时导数：f'(3) = 6（切线斜率）
- x从3到5的平均变化率：(25-9)/(5-3) = 8
```

梯度下降用的是瞬时斜率（6），不是平均斜率（8）。这意味着：
- 每一步都假设"当前切线能很好地近似曲线的一小段"
- 步子越小，线性近似的误差越小
- 这也是为什么学习率太大时会发散——步子大到线性近似已经严重失真

### 梯度为负时更新的方向

```
w_new = w - lr × gradient
```

| 梯度方向 | 含义 | 更新动作 | 结果 |
|---|---|---|---|
| 负梯度 | w 增加则 L 减少 | w 增大 | L 减少 ✅ |
| 正梯度 | w 增加则 L 增加 | w 减少 | L 减少 ✅ |

负梯度方向 = 函数值下降最快的方向，这就是**最速下降法**（Steepest Descent）。

---

## 1-5-9 一阶 vs 二阶优化：工业界的现实选择

### 一阶方法的本质

一阶方法只用**一阶导数（梯度）** 来更新参数：

```
w_new = w - lr × ∇L(w)
```

**优点**：
- 计算量小：只需一次前向 + 一次反向，O(n)
- 存储少：只需存梯度向量，O(n)

**缺点**：
- 用线性近似对付非线性曲面，需要多步迭代
- 无法利用曲率信息，不知道步子该迈多大

### 二阶方法的本质

二阶方法利用**二阶导数（Hessian 矩阵）** 来更新参数：

```
w_new = w - H⁻¹ · ∇L(w)
```

其中 `H[i,j] = ∂²L/∂wᵢ∂wⱼ` 是所有二阶偏导数构成的矩阵。

**优点**：
- 对二次函数一步到位（理论最优）
- Hessian 包含曲率信息，步长自然确定，无需学习率

**缺点**：
- Hessian 是 n×n 矩阵，存储 O(n²)
- 计算 Hessian 或其逆是 O(n²) 或更高
- 对于 70B 参数的模型，Hessian 完全无法存储和计算

### 为什么工业界选一阶？

| 维度 | 一阶 | 二阶（Hessian） |
|---|---|---|
| 存储 | O(n) | O(n²) |
| 计算量 | O(n) | O(n²) ~ O(n³) |
| 70B 参数存储 | ~280GB | ~49万亿 GB |
| 可行性 | ✅ | ❌ |

### 工业界的工程 tricks

虽然二阶不可行，但工程 tricks 通过**近似曲率信息**来弥补：

#### 1. 自适应学习率（Adam/RMSProp）

核心思路：用梯度平方的指数移动平均（EMA）来近似每个参数的"有效曲率"：

```python
# Adam 更新公式（简化）
m = β₁·m + (1-β₁)·∇L      # 梯度的一阶矩估计（类似 momentum）
v = β₂·v + (1-β₂)·∇L²     # 梯度平方的 EMA，近似曲率
w = w - lr·m / (√v + ε)
```

本质是用 `∇L²` 的 EMA 作为 Hessian 对角线的近似——不是真正的 Hessian，但计算量仍是一阶。

#### 2. 学习率衰减 / 余弦退火

```
lr(t) = lr_max × cos(t/T_max × π)
```

本质：初期大步探索，后期小步收敛。越接近最优点，曲面越接近二次函数，小步长的线性近似更准确。

#### 3. Warmup

```
前N步：lr 从小慢慢增大
之后：正常衰减
```

防止早期曲率估计不稳定时步子太大。

#### 4. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

防止梯度爆炸，本质是给步长加了一个安全上界。

### 理论的尴尬现状

| 方法 | 理论证明 | 实际情况 |
|---|---|---|
| 梯度下降（凸） | 严格收敛 | 非凸下不保证 |
| Adam | 有收敛证明 | 假设与实际场景不符 |
| AdamW | 无 | 经验性，工业界大量使用 |
| Warmup | 无 | 经验性，效果稳定 |

**一句话总结**：深度学习优化器是"工程经验驱动，理论严重滞后"的领域。一阶方法是当前大规模训练的唯一可行解，工程 tricks 本质上都是在用近似曲率信息弥补二阶信息的缺失。

---

## 1-5-10 PyTorch autograd 核心 API 速查

### 计算图与梯度追踪

| API | 作用 | 典型用法 |
|---|---|---|
| `requires_grad=True` | 开启当前张量的梯度追踪 | `x = torch.tensor([1.], requires_grad=True)` |
| `x.requires_grad` | 查看是否追踪梯度 | 条件判断 |
| `x.is_leaf` | 是否叶子节点（用户创建） | 调试时检查 |
| `x.grad_fn` | 反向传播函数引用 | 调试：`print(y.grad_fn)` |
| `x.grad` | 存储累积的梯度值 | 反向后：`print(x.grad)` |

### 反向传播

| API | 作用 | 典型用法 |
|---|---|---|
| `loss.backward()` | 标量反向传播 | `loss.backward()` |
| `loss.backward(gradient)` | 非标量反向传播 | `y.backward(grad_y)` |
| `retain_graph=True` | 保留计算图供二次反向 | `loss.backward(retain_graph=True)` |
| `create_graph=True` | 在 grad 中构建计算图 | 用于求高阶导 |

```python
# 求二阶导示例
x = torch.tensor([2.], requires_grad=True)
y = x ** 3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]  # 二阶导 = 6x = 12
```

### 梯度控制

| API | 作用 | 典型用法 |
|---|---|---|
| `optimizer.zero_grad()` | 清零所有参数的梯度 | 训练循环必备 |
| `x.grad.zero_()` | 清零特定张量梯度 | 原位操作 |
| `with torch.no_grad():` | 禁用梯度计算 | 推理时 |
| `@torch.no_grad()` | 装饰器版本 | 函数定义 |
| `torch.set_grad_enabled(False)` | 动态开关 | 条件分支推理 |
| `x.detach()` | 分离张量，截断计算图 | 需要非梯度操作时 |
| `x.detach_()` | 原地分离 | 少用 |

### Hook 机制

| API | 作用 | 典型用法 |
|---|---|---|
| `register_forward_hook(fn)` | 拦截前向传播 | 特征提取 |
| `register_backward_hook(fn)` | 拦截反向传播 | 梯度检查/修改 |
| `register_hook(fn)` | 给 grad 注册 hook | 打印或修改梯度 |
| `handle.remove()` | 取消注册 | 防止内存泄漏 |

```python
# 给梯度注册 hook（反向传播时触发）
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
loss = y.sum()

def print_grad(grad):
    print("梯度:", grad)

y.register_hook(print_grad)
loss.backward()  # 打印：梯度: tensor([2., 4., 6.])
```

### 优化器中的梯度

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 完整训练循环
for data, target in dataloader:
    optimizer.zero_grad()        # Step 1: 清梯度
    output = model(data)          # Step 2: 前向（计算图自动构建）
    loss = criterion(output, target)
    loss.backward()              # Step 3: 反向（梯度存到 parameter.grad）
    optimizer.step()             # Step 4: 用梯度更新参数
```

### 梯度检查（Gradient Checking）

用于验证 autograd 计算的梯度是否正确（数值法近似 vs 解析法）：

```python
def gradient_check(model, x, y, eps=1e-5):
    model.eval()
    x.requires_grad = True
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    # 数值梯度
    for p in model.parameters():
        if p.grad is not None:
            numerical = []
            analytical = p.grad.data.clone()
            for i in range(min(5, p.numel())):  # 只检查前5个
                old = p.data.view(-1)[i].item()
                p.data.view(-1)[i] = old + eps
                loss_plus = criterion(model(x), y).item()
                p.data.view(-1)[i] = old - eps
                loss_minus = criterion(model(x), y).item()
                numerical.append((loss_plus - loss_minus) / (2 * eps))
                p.data.view(-1)[i] = old
            print(f"数值: {numerical[:3]}, 解析: {analytical.view(-1)[:3]}")
```
