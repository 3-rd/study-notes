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

### 动态图 vs 静态图

| | PyTorch（动态图） | TensorFlow 1.x（静态图） |
|--|---|---|
| 建图时机 | 前向传播时同步构建 | 先定义图，后执行 |
| 控制流 | 直接用 Python `if/for/while` | 需要 `tf.cond`/`tf.while_loop` |
| 调试 | 直接报错，可 print | 需要 `sess.run` 才能看值 |
| 灵活性 | 极高，代码怎么写图就怎么建 | 需预先定义所有分支 |

**动态图优势**：代码即模型，不用预先声明图结构。

**静态图优势**：图结构已知，可做全局优化（如算子融合、内存规划），性能更好。TensorFlow 2.x 默认 eager execution（动态图），但用 `tf.function` 可以 jit 编译成静态图加速。PyTorch 的 `torch.compile` 也是类似思路——先把动态图"冻结"成静态图再优化。

```python
# 动态图：每行代码立即执行
x = torch.tensor([1., 2.])
y = x * 2        # 立即计算
z = y + 1        # 立即计算

# 静态图（TensorFlow 风格）：
# x = tf.placeholder(tf.float32)   # 先声明
# y = x * 2                        # 只是画边，不计算
# with tf.Session() as sess:
#     result = sess.run(y, feed_dict={x: [1., 2.]})  # 实际执行
```

**Notebook 中的内存问题**：动态图下，每次 cell 跑前向都会创建新的计算图节点，如果不断保存中间结果（图节点被引用），内存会累积。定期 `del` 不需要的变量 + `gc.collect()` 可缓解。静态图因为图结构固定，不存在这个问题。

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

**为什么设计成累加而不是覆盖？**

这是**有意为之**，而不是"忘了清零"。累加设计是为了支持两种场景：

**场景1：梯度累积（大 batch 训练）**
```python
# batch_size=64 显存不够，用4个 batch_size=16 累积
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / 4  # 归一化
    loss.backward()   # 累加到 .grad
    
    if (i + 1) % 4 == 0:
        optimizer.step()      # 用累积的梯度更新
        optimizer.zero_grad()  # 4个batch累积完，清零
```

**场景2：多 loss 多路径回传**
```python
# 一个 shared 参数被多个分支使用
shared_params = ...
out1 = branch1(shared_params)
out2 = branch2(shared_params)

loss1 = criterion(out1, target1)
loss2 = criterion(out2, target2)

loss1.backward()  # shared_params.grad 有了第一份梯度
loss2.backward()  # 累加第二份，两个分支的梯度都被保留
```

**如果设计成"清零覆盖"**：上述两个场景都直接坏掉——要么无法做梯度累积，要么多路径的梯度互相覆盖。累加设计把控制权交给用户，框架不隐式做主。

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

### 共享存储的陷阱

**detach() 出来的张量和原张量共享同一块底层内存**，修改其中一个会影响另一个：

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2      # y = [2, 4, 6]
z = y.detach() # z 和 y 共享底层存储

z[0] = 99      # 原地修改 z
print(y)        # y = [99, 4, 6]  ← y 也被改了！
print(z)        # z = [99, 4, 6]
```

**安全用法**：detach 后立即做只读操作（打印、存列表、画图），不会出问题。

**如果需要修改后不影响原值**，用 `clone()` 深拷贝：

```python
z = y.detach().clone()  # 深拷贝，不共享存储
z[0] = 99
print(y)  # y 不受影响
```

### 典型用途

1. **需要原地修改张量值时**（叶子节点不能直接 in-place 修改）
2. **把张量传给不需要梯度的函数**（如 numpy 操作、打印、画图）
3. **阻止梯度回传到某条分支**（如 Actor-Critic 中 detach baseline）

```python
# 错误示范：直接修改需要梯度的叶子节点
x = torch.tensor([1., 2., 3.], requires_grad=True)
x[0] = 10                   # RuntimeError: a view of a variable

# 正确做法
x_detached = x.detach()
x_detached[0] = 10          # OK
```

**detach() vs no_grad()**：
- `no_grad()`：**整个代码块**都不追踪梯度
- `detach()`：**单个张量**从图中分离出来（但共享存储）

---

## 1-5-6 hook 机制（进阶）

**作用**：在不需要修改前向/反向代码的情况下，**拦截**前向传播或反向传播的过程，查看或修改中间张量/梯度。

**注册层级说明**：hook 绑定的位置决定触发次数：

| 注册位置 | 触发次数 | 说明 |
|---------|---------|------|
| `model`（根模块） | 1次 | 拦截整个模型的最终输入输出 |
| `model.fc1`（子模块） | 1次 | 只在 fc1 算完时触发 |
| `model.fc2`（子模块） | 1次 | 只在 fc2 算完时触发 |

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.Linear(5, 2)
)

# 注册在根模块：只触发1次（最终输出）
handle = model.register_forward_hook(
    lambda m, inp, out: print(f'根模块 hook，output shape: {out.shape}')
)

# 如果想拦截每个层，需要遍历注册
for name, module in model.named_children():
    module.register_forward_hook(
        lambda m, inp, out: print(f'  层 {name} hook')
    )

x = torch.randn(2, 10)
y = model(x)
# 输出：
#   层 0 hook
#   层 1 hook
#   根模块 hook

handle.remove()
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

### 核心机制详解

#### 梯度裁剪（直接修改梯度）

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**原理**：计算所有参数梯度的 L2 范数 `||g||₂ = √(g₁² + g₂² + ...)`，如果超过 `max_norm`，等比例缩放回来。

```python
# 等价逻辑
total_norm = 0.0
for p in model.parameters():
    total_norm += p.grad.norm(2).item() ** 2
total_norm = total_norm ** 0.5

if total_norm > max_norm:
    clip_coef = max_norm / total_norm
    for p in model.parameters():
        p.grad.mul_(clip_coef)  # 直接缩放梯度
```

**特点**：这是**唯一一个直接在反向传播后修改梯度**的方法，属于"事后补救"。只能防止爆炸，不能防止消失。

#### 残差连接（加法保梯度）

```
普通层：output = H(x)
残差层：output = H(x) + x
```

**关键**：加法的反向传播是"分流"的——`∂(H+x)/∂x = ∂H/∂x + 1`。

无论 H(x) 的梯度多小，加个 `+1` 就让梯度永远不会消失。也不存在梯度爆炸的问题（加法不会让梯度相乘变大）。

#### 归一化（稳定激活值，间接稳定梯度）

BatchNorm 前向：
```python
mu = x.mean(dim=0)
var = x.var(dim=0)
x_norm = (x - mu) / sqrt(var + eps)
y = gamma * x_norm + beta
```

**对梯度的影响**：
- **间接**：把激活值压到稳定范围（前向数值稳定 → 反向梯度数值也稳定）
- **直接**：gamma/beta 作为可学习参数，提供稳定的梯度回传路径（形状固定为特征维度，不会随深度指数变化）

#### 激活函数（间接影响梯度）

激活函数在前向时改变激活值分布，间接影响梯度：

| 激活函数 | 前向特点 | 对梯度的影响 |
|---------|---------|-------------|
| Sigmoid | 输出 0~1 | 导数最大 0.25，深层连乘≈0，梯度消失 |
| Tanh | 输出 -1~1 | 导数最大 1，比 Sigmoid 好，但仍有消失问题 |
| ReLU | 负区=0，正区=原值 | 正区导数恒为 1，不消失 |

**ReLU 的梯度（工程定义）**：
```
x > 0: 梯度 = 1
x < 0: 梯度 = 0（截断）
x = 0: 梯度 = 0 或 1（工程规定，不是数学严格定义）
```

**注意**：ReLU 在 x=0 处数学上不可导，但工程上通过**规定 subgradient**（次梯度）让它能参与反向传播。

#### 权重初始化（预防）

权重太小 → 激活值逐层变小 → 梯度消失
权重太大 → 激活值逐层变大 → 梯度爆炸

合理的初始化（如 Kaiming/Xavier）让各层激活值和梯度的方差在合理范围，从源头降低消失/爆炸风险。

#### LSTM/GRU（门控机制）

RNN 循环使用同一个权重矩阵 W：`h_t = h_{t-1} @ W`

梯度回传时：`∂h_T/∂h_t = W^T @ W^T @ ... @ W^T`（T-t 次连乘）

LSTM/GRU 通过**门控**决定保留多少历史、多少新信息：
```
h_t = h_{t-1} * output_gate + new_gate * input_gate
```
梯度可以"抄近道"不经过所有 W 连乘，从根本上缓解消失/爆炸。

### 方法分类总结

| 方法 | 操作阶段 | 机制 |
|------|---------|------|
| 梯度裁剪 | **反向传播后** | 直接修改梯度，治标 |
| 残差连接 | 前向 | 加法+1保梯度，治本 |
| 归一化 | 前向 | 稳定激活值→稳定梯度，治本 |
| 激活函数 | 前向 | 改变激活值分布，间接影响 |
| 权重初始化 | 前向（训练前） | 预防，治本 |
| LSTM/GRU | 前向 | 门控减少连乘，治本 |

**大多数方法都是在"前向阶段预防"梯度问题，只有梯度裁剪是在"反向传播后直接修改梯度"。**

---

## 1-5-7.5 nn.Module 与 nn.Parameter

### 为什么需要 nn.Module？

纯 tensor 的问题：参数需要手动管理。

```python
# 纯 tensor 写法
W = torch.randn(10, 5, requires_grad=True)
b = torch.randn(5, requires_grad=True)
optimizer = torch.optim.SGD([W, b], lr=0.1)  # 手动传入参数列表
```

**nn.Module 的核心价值**：

1. **自动参数收集**：`model.parameters()` 把所有带梯度的参数收拢在一起
2. **设备统一管理**：`model.to('cuda')` 一次移动所有参数
3. **封装复用**：把参数和计算逻辑打包成独立模块

```python
# nn.Module 写法
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weight + self.bias

model = Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 自动收集所有参数
```

### nn.Parameter 是什么？

```python
nn.Parameter(tensor) ≈ torch.tensor(..., requires_grad=True) + 自动注册到父 Module
```

| | 普通 tensor | nn.Parameter |
|--|-----------|--------------|
| `requires_grad` | 默认 False | 默认 True |
| 被 `model.parameters()` 收集 | ❌ | ✅ |
| 能被 optimizer 更新 | ❌ | ✅ |

`nn.Parameter` 就是 `requires_grad=True` 的 tensor，外层包了一层"注册"逻辑。本质上还是个 tensor。

### 计算图视角下的 nn.Module

```
nn.Module
    ├── self.weight (nn.Parameter = requires_grad=True 的 tensor)
    ├── self.bias   (nn.Parameter = requires_grad=True 的 tensor)
    └── forward()
            ↓
        构建计算图（和纯 tensor 完全一样）
```

`nn.Module` 本身不参与计算图，它只是 Python 层的组织结构。真正参与计算图的是 `nn.Parameter`——它们本质上是 tensor。`model(x)` 执行时，实际上是在 tensor 层面做运算，构建计算图。**nn.Module 是组织的壳，nn.Parameter 才是参与梯度追踪的实际的 tensor。**

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

**SGD 的问题**：所有参数用同一个学习率，但不同参数的梯度大小可能差几十倍。

**Adam 解决思路**：让梯度大的时候步子小，梯度小的时候步子大。

```python
# Adam 更新公式
m = β₁·m + (1-β₁)·∇L      # 梯度的一阶矩估计（类似 momentum）
v = β₂·v + (1-β₂)·∇L²     # 梯度平方的 EMA
w = w - lr·m / (√v + ε)
```

| 参数 | 含义 | 初始值 | 固定/可变 |
|------|------|--------|----------|
| `β₁ = 0.9` | m 的衰减率 | 固定 | 超参数，可调 |
| `β₂ = 0.999` | v 的衰减率 | 固定 | 超参数，可调 |
| `lr` | 学习率 | 固定 | 人调，不学习 |
| `ε = 1e-8` | 防止除零 | 固定 | 工程参数 |
| `m` | 梯度 EMA | `0` | 每步更新 |
| `v` | 梯度平方 EMA | `0` | 每步更新 |

**核心机制**：分母 `√v` 自适应调整学习率——梯度大的参数 `√v` 也大，学习率被压小；梯度小的参数 `√v` 也小，学习率被放大。

**与二阶方法的关系**：Newton 法 `w_new = w - H⁻¹·∇L` 用 Hessian 提供最优步长，但 Hessian 存储 O(n²) 不可行。Adam 用 `v` 估计 `E[∇L²]`（Hessian 对角线），用 `m` 估计 `∇L`，本质是**用一阶计算量换二阶效果的部分近似**。这就是 Adam（Adaptive Moment Estimation）名字的由来。

```python
# 直观理解：不同参数自动获得不同学习率
if 梯度大: lr被压小  # 防止震荡
if 梯度小: lr被放大  # 加速收敛
```

RMSProp 和 Adam 思路类似，区别在于 Adam 多了一个 `m`（momentum）来平滑更新方向。

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
