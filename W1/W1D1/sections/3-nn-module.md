# 三、nn.Module — 模型构建核心

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| 继承 `nn.Module` | 必须重写 `__init__` + `forward` | 为什么要继承 |
| `super().__init__()` | 调用父类构造函数 | 不调用会怎样 |
| `named_parameters()` / `parameters()` | 遍历模型参数 | 如何冻结部分层 |
| `state_dict()` / `load_state_dict()` | 模型序列化/加载 | 怎么只加载部分参数 |
| `children()` / `modules()` | 遍历子模块 | 区别是什么 |
| 常见层 | `Linear / Conv2d / BatchNorm / Dropout / LSTM / Embedding` | 参数含义 |

**⚠️ 面试必答题：**
- `nn.Module` 的 `forward` 为什么只需写前向，反向自动搞定？
- `model(img)` 背后发生了什么？（call → forward → hooks）
- `model.train()` vs `model.eval()` 区别？（BN 和 Dropout 的行为差异）

---

# 3. nn.Module — 模型构建核心

## 3-1 模块定义与参数管理

### 为什么要继承 nn.Module？

`nn.Module` 是 PyTorch 的模型组织框架。继承它不是为了"继承方法"，而是为了获得它提供的**基础设施**：

1. **自动参数收集** — `model.parameters()` 把所有 `nn.Parameter` 收拢，optimizer 一行搞定
2. **设备统一管理** — `model.to('cuda')` 一次把所有参数和数据搬到 GPU
3. **状态追踪** — 前向/反向 hooks、buffer 管理、模型结构序列化

---

```python
import torch
import torch.nn as nn

# 不继承 nn.Module：纯手工，每个 tensor 手动管理
W = torch.randn(10, 5, requires_grad=True)
b = torch.randn(5, requires_grad=True)
optimizer = torch.optim.SGD([W, b], lr=0.1)  # 手动传参

# 继承 nn.Module：自动化的基础设施
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()                          # ← 必须调用
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias

model = Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 一行搞定所有参数
```

---

### super().__init__() 为什么必须调用？

`nn.Module.__init__()` 里注册了若干 PyTorch 内部簿记结构——参数注册表、buffer 注册表、forward hook 容器等。不调用 super，`self.weight` 就不会被当成参数收集，`model.parameters()` 里就没有它，`model.to('cuda')` 也不会移动它。

---

```python
class BadLinear(nn.Module):
    def __init__(self, in_f, out_f):
        # 忘记调用 super().__init__()
        self.weight = nn.Parameter(torch.randn(in_f, out_f))  # 不会被注册！

model = BadLinear(10, 5)
print("parameters:", list(model.parameters()))  # [] — 空！weight 丢失了
print("weight 是否在 params:", model.weight in model.parameters())  # False
```

---

### nn.Parameter 是什么？

---

```python
nn.Parameter(tensor) ≈ requires_grad=True + 自动注册到父 Module
```

---

| | 普通 `torch.tensor` | `nn.Parameter` |
|--|---|---|
| `requires_grad` | 默认 False | 默认 True |
| 被 `parameters()` 收集 | ❌ | ✅ |
| 能被 optimizer 更新 | ❌ | ✅ |
| 本质 | 就是 tensor | 就是 tensor（外面包了注册逻辑） |

`nn.Parameter` 本质上就是个 `requires_grad=True` 的 tensor，只是外层套了一层"把自己注册到父 Module"的逻辑。没有其他魔法。

### 计算图视角下的 nn.Module

```
nn.Module (Python 对象)
    ├── self.weight (nn.Parameter = requires_grad=True 的 tensor)
    ├── self.bias   (nn.Parameter = requires_grad=True 的 tensor)
    └── forward()
            ↓
        tensor 运算 → 构建计算图（和纯 tensor 完全一样）
```

**关键**：nn.Module 本身不参与计算图，它只是 Python 层的组织壳。真正在计算图里的是 `nn.Parameter`（tensor）。`model(x)` 执行时，实际上是在 tensor 层面做运算，构建计算图。

---

```python
# 验证：Module 本身不在计算图里
x = torch.tensor([1., 2.], requires_grad=True)
linear = nn.Linear(2, 3)
y = linear(x)  # linear 是个 Python 对象，不是 tensor

print("linear.weight 在图中:", linear.weight.requires_grad)  # True
print("linear.weight.grad_fn:", linear.weight.grad_fn)      # None — 叶子节点！
print("y.grad_fn:", y.grad_fn)                             # <AddmmBackward>
```

---

### forward 为什么只需写前向？

因为 PyTorch 的 autograd 引擎根据**前向运算的每一步**，自动构建反向传播图。`forward` 里写了什么运算，autograd 就自动生成对应的反向函数（grad_fn）。

`model(x)` 背后发生了什么：

---

```python
output = model(x)

# 等价于：
output = nn.Module.__call__(model, x)
# 1. model.__call__() 执行（Python 魔法方法）
# 2. 调用 model.forward(x)                    ← 你的代码在这里
# 3. 执行所有注册的 forward_hooks            ← 可选拦截点
# 4. 返回 output
```

---

`__call__` 是 Python 的语法糖：任何 `obj(args)` 调用，实际是 `obj.__call__(args)`。PyTorch 在 `nn.Module.__call__` 里插入了一些钩子（forward pre-hook、forward post-hook），所以直接写 `forward` 不够——需要经过 `__call__`。这也是为什么 `model(x)` 和 `model.forward(x)` **不完全等价**。

---

## 3-2 遍历与结构（children / modules / named_parameters）

### 四个遍历方法的区别

---

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 8),
    nn.ReLU(),
    nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU()
    )
)
```

---

| 方法 | 返回内容 | 层级 |
|------|---------|------|
| `model.modules()` | 所有模块（包括自己、根） | 深度优先遍历（根 → 子 → 子...） |
| `model.children()` | 直接子模块（不包括嵌套更深） | 仅一层 |
| `model.named_parameters()` | 所有参数的 (name, tensor) | 深度遍历 |
| `model.parameters()` | 所有参数的 tensor | 深度遍历 |

---

```python
print("=== modules() (所有模块，包括嵌套) ===")
for m in model.modules():
    print(f"  {m}")

print("\n=== children() (仅直接子模块) ===")
for m in model.children():
    print(f"  {m}")

print("\n=== named_parameters() ===")
for name, p in model.named_parameters():
    print(f"  {name}: {p.shape}")
```

---

输出：
```
=== modules() ===
  Sequential(
    (0): Linear(in_features=10, out_features=8)
    (1): ReLU()
    (2): Sequential(
      (0): Linear(in_features=8, out_features=4)
      (1): ReLU()
    )
  )
  Linear(in_features=10, out_features=8)
  ReLU()
  Sequential(...)
  Linear(in_features=8, out_features=4)
  ReLU()

=== children() ===
  Linear(in_features=10, out_features=8)
  ReLU()
  Sequential(...)

=== named_parameters() ===
  0.weight: torch.Size([8, 10])
  0.bias: torch.Size([8])
  2.0.weight: torch.Size([4, 8])
  2.0.bias: torch.Size([4])
```

### 实际应用：冻结部分层

---

```python
# 冻结所有 children 中名字包含 'bias' 的参数
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = False

# 或者用 children 遍历，子模块是顺序存储的
for i, child in enumerate(model.children()):
    if i < 2:  # 冻结前两层
        for param in child.parameters():
            param.requires_grad = False

# 验证
trainable = [p for p in model.parameters() if p.requires_grad]
print("可训练参数:", sum(p.numel() for p in trainable))
```

---

### 冻结参数在梯度链中的行为

**冻结参数（`requires_grad=False`）不等于截断梯度**。冻结参数的梯度原封不动继续往上传，冻结只是"不记录该参数的梯度"。

---

```python
import torch

x = torch.tensor([1., 2.], requires_grad=True)
w_frozen = torch.tensor([[1., 0.], [0., 1.]], requires_grad=False)  # 冻结
w_trainable = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)  # 可训练

y = x @ w_frozen @ w_trainable
loss = y.sum()
loss.backward()

print("x.grad:", x.grad)              # 有梯度（w_frozen 没阻断）
print("w_trainable.grad:", w_trainable.grad)  # 有梯度
```

---

梯度路径：`loss → w_trainable → w_frozen → x`

w_frozen 被跳过（不记录梯度），但梯度继续传给它上游的 x。

**冻结 vs detach 的本质区别**：

| 操作 | 梯度流过？ | 记录该参数梯度？ |
|------|-----------|----------------|
| `requires_grad=False` | ✅ 畅通，梯度原封不动 | ❌ 不记录 |
| `detach()` | ❌ **截断**，梯度停止 | — |

**冻结参数的典型应用**：预训练模型 backbone 冻住，只 fine-tune 头部。backbone 的权重作为已知的"常量"，只更新 head 的参数——backbone 本身不更新，但梯度仍然流过它。

---

## 3-3 state_dict 与模型保存加载

### 两种保存方式的区别

---

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)

# 方式一：保存整个模型（不推荐）
torch.save(model, '/tmp/model_full.pth')  # 包含类定义，重载依赖类
loaded1 = torch.load('/tmp/model_full.pth')

# 方式二：保存 state_dict（推荐）
torch.save(model.state_dict(), '/tmp/model_sd.pth')  # 只有参数字典
loaded2 = nn.Linear(10, 2)
loaded2.load_state_dict(torch.load('/tmp/model_sd.pth'))  # 需要先实例化
```

---

**为什么推荐 state_dict**：
- 文件更小（只存参数，不存模型结构）
- 不依赖类定义，迁移性更强
- 断点续训时通常需要同时保存 optimizer 的 state_dict

### 完整断点续训

---

```python
# 保存
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.234,
}
torch.save(checkpoint, '/tmp/ckpt.pth')

# 加载
ckpt = torch.load('/tmp/ckpt.pth')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1
```

---

### 部分加载（strict / 非 strict）

---

```python
# 场景：预训练模型有 100 个参数，但你的模型改了最后层，只有 80 个

# strict=False：忽略不匹配的 key
pretrained = {'layer1.weight': ..., 'layer1.bias': ..., 'layer2.weight': ...}
model = MyModel()  # 只有 layer1
model.load_state_dict(pretrained, strict=False)  # layer2 缺失被忽略

# 精确过滤：只加载部分参数
pretrained = torch.load('/tmp/pretrained.pth')
partial = {k: v for k, v in pretrained.items() if 'fc' not in k}
model = MyModel()
model.load_state_dict(partial, strict=False)
```

---

---

## 3-4 train / eval 模式与 BatchNorm / Dropout 机制

### Dropout 机制

---

```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.5)  # 训练时随机丢弃 50%

x = torch.ones(5)
print("train 模式:", dropout(x))   # 训练时有随机性，部分元素变 0
print("eval 模式:", dropout.eval()(x))  # eval 时所有元素保留原值

# 注意：eval 模式需要用 dropout.eval()，而不是 dropout(x) 在 eval 之后调用
dropout.train()   # 切换到训练模式
dropout.eval()    # 切换到评估模式
```

---

**Dropout 训练 vs 评估的行为差异**：

| 模式 | 行为 | 效果 |
|------|------|------|
| train | 随机置零（p 的概率），剩余元素除以 `(1-p)` 做缩放 | 防止过拟合 |
| eval | 所有元素原样通过 | 确定性输出 |

**缩放的必要性**：`E[dropout(x)] = (1-p) × x/(1-p) = x`，数学期望不变，训练和测试的输出期望一致。

### BatchNorm 机制

---

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(3)  # 3 个特征

x = torch.randn(4, 3)  # batch=4, features=3
print("train 模式:")
out_train = bn(x)
print("  输出:\n", out_train)
print("  mean per feature:", out_train.mean(dim=0))  # ≈ [0, 0, 0]
print("  var per feature:", out_train.var(dim=0))   # ≈ [1, 1, 1]

print("\neval 模式:")
out_eval = bn.eval()(x)
print("  输出:\n", out_eval)  # 和 train 完全不同！
```

---

**BatchNorm 训练 vs 评估的行为差异**：

| 模式 | 计算均值/方差 | 使用均值/方差 |
|------|-------------|-------------|
| train | 用当前 batch 的统计量 | 当前 batch 的统计量 |
| eval | 不计算 | 用**全局移动平均**（moving average，训练时累积的） |

---

```python
# 验证：eval 用的 moving average 是训练时累积的
bn = nn.BatchNorm1d(3)
bn.train()

print("training 模式:")
for i in range(3):
    x = torch.randn(4, 3) * (i + 1)  # 越来越大的方差
    out = bn(x)
    print(f"  batch {i} mean: {out.mean(dim=0)}")

print("\n切换 eval 后（使用 moving average）:")
bn.eval()
x = torch.randn(4, 3)
print("  输出:", bn(x))
print("  running_mean (不更新):", bn.running_mean)
print("  running_var (不更新):", bn.running_var)
```

---

### model.train() vs model.eval() 实际影响

---

```python
model = nn.Sequential(
    nn.Linear(10, 8),
    nn.BatchNorm1d(8),
    nn.ReLU(),
    nn.Dropout(0.3)
)

model.train()
print("train 模式 - BatchNorm 用 batch 统计:")
x = torch.randn(8, 10)
print("  BatchNorm running_mean:", model[1].running_mean[:3])

model.eval()
print("\neval 模式 - BatchNorm 用 moving average:")
print("  BatchNorm running_mean:", model[1].running_mean[:3])
```

---

**总结：哪些层受 train/eval 影响？**

| 层 | train 行为 | eval 行为 |
|---|---|---|
| BatchNorm | 用 batch 统计量 | 用 moving average |
| Dropout | 随机置零 | 全部通过 |
| LayerNorm | 用 batch 统计量 | 用 batch 统计量（不变化） |
| InstanceNorm | 用 batch 统计量 | 用 running average |

---

### 补充：Dropout 的反向传播机制

Dropout 的前向过程是 `out = x * mask / (1-p)`，其中 mask 是随机生成的 0/1 掩码。

**反向传播时，mask 本身不参与梯度计算**（没有梯度），所以：

```
上游梯度 × mask / (1-p)
→ 被 mask=0 置零的位置，梯度 = 0（该神经元这轮不更新）
→ 被 mask=1 保留的位置，梯度正常回传
```

**关键**：Dropout 不阻断梯度流，只是"杀死"被 mask 置零的那些神经元，让它们这轮不更新。下一轮前向时重新生成 mask，神经元可能又活了。

**极端 Bottleneck 场景**：如果网络的窄层只有一个参数连接到前后两部分，该参数被 mask=0 置零时，前后两部分的梯度都会变成 0——这轮训练完全失效。实际网络有冗余路径，所以不严重。

### 补充：Norm 家族深度解析 — CV vs NLP 的本质差异

#### 三种 Norm 的归一化维度对比

| Norm | 归一化维度 | 归一化时参考的样本/位置 | 典型场景 |
|------|-----------|----------------------|---------|
| **BatchNorm** | 每个通道，跨 batch | 同一通道的所有样本同一位置 | 图像分类（CNN） |
| **LayerNorm** | 每个样本，跨所有维度 | 每个样本内部 | Transformer / NLP |
| **InstanceNorm** | 每个样本的每个通道 | 每个样本 × 每通道的空间区域 | 风格迁移 |

#### 核心直觉：在哪个维度做 Norm，就是消除哪个维度的差异

| Norm | 消除的差异 | 保留的差异 |
|------|-----------|-----------|
| **BatchNorm** | 不同样本在同一通道上的绝对水平差异 | 同一样本内部通道间的相对关系 |
| **LayerNorm** | 每个样本内部向量各维度的绝对强度 | 向量在高维空间里的**方向**（语义主要在方向里） |
| **InstanceNorm** | 每个样本每个通道的全局统计量 | 内容结构，去掉风格纹理 |

#### BatchNorm 为什么对图像有效，对 NLP 无效？

**图像**：不同图片在同一个空间位置 (h, w) 具有相同的**视觉语义含义**。所有图片在 (5, 7) 位置都是"某个局部纹理或边缘"，通道 C 在该位置的激活值跨图片具有相似的统计分布——BatchNorm 的假设成立。

**NLP**：第 0 个词和第 0 个词之间没有任何语义对齐。"The" 和 "A" 都出现在句子开头，但含义完全无关。跨样本在同位置做归一化，强行把不同语义的 token 拉成相同分布，破坏了语义信息。

---

```python
# 两个句子，同一位置 pos=0
# 句子A："The cat sits"     → token[0] = "The"
# 句子B："A dog runs"       → token[0] = "A"
# BatchNorm 强制让两个位置的激活值统计分布相同——灾难性的
```

---

#### LayerNorm 保留的是什么？

LayerNorm 对每个 token 的向量做归一化，数学上等价于将向量投影到**高维球面上**（固定模长，消除绝对强度）。语义信息主要编码在**方向**里，而不是模长里——所以 LayerNorm 消除模长的同时保留了语义。

#### 验证：NLP 场景下 BatchNorm 的问题

以 NLP 输入 (batch=2, seq_len=3, embed=4) 为例：

---

```python
# 句子A：[1,2,3,4], [5,6,7,8], [9,10,11,12]
# 句子B：[10,10,10,10], [20,20,20,20], [30,30,30,30]

# LayerNorm 后：句子A token[0] = [-1.34, -0.45, 0.45, 1.34]（递增关系保留）
#              句子B token[0] = [-1.34, -1.34, -1.34, -1.34]（各维度相同，LayerNorm后仍相同）

# BatchNorm 后（embed 维度跨 batch 归一化）：
# 句子A token[0] 和 句子B token[0] → 相同值！
# 两个语义完全不同的句子被强制拉平了
```

---

**结论**：BatchNorm 对 NLP 是有害的，因为它破坏了不同 token 之间的语义差异。

#### 现代大模型为什么不用 BatchNorm？

1. **分布式训练的 batch 统计量不稳定**：多 GPU 分割 batch 后，各 GPU 看到的统计量差异大
2. **序列位置无结构**：前面论证过
3. **训练/推理不一致（TID，Training Inference Discrepancy）**：NLP 任务中 batch 统计量和 running statistics 差异大，导致 BN 在 NLP 表现差
4. **主流架构 Pre-LN 的归一化位置不适用 BN**

现代大模型全部使用 **LayerNorm** 或 **RMSNorm**（只缩放不偏移，计算更快）。

---

## 3-5 常见网络层（Linear / Conv2d / LSTM / Embedding）

### nn.Linear — 全连接层

---

```python
import torch
import torch.nn as nn

# in_features: 输入特征维度
# out_features: 输出特征维度
# bias: 是否有偏置（默认 True）
linear = nn.Linear(10, 5)  # input: (..., 10) → output: (..., 5)

x = torch.randn(2, 10)     # batch=2, 每个样本10维
y = linear(x)              # (2, 5)
print(y.shape)             # torch.Size([2, 5])

# 验证：y = x @ W^T + b
manual = x @ linear.weight.T + linear.bias
print("手动计算一致:", torch.allclose(y, manual))  # True
```

---

**参数数量**：`weight: (5, 10)` + `bias: (5,)` = 55 个参数

### nn.Conv2d — 卷积层

---

```python
import torch
import torch.nn as nn

# in_channels: 输入通道数（灰度图=1，RGB=3）
# out_channels: 输出通道数（卷积核数量）
# kernel_size: 卷积核大小
# stride: 步长（默认1）
# padding: 填充（默认0，same 填充需要 padding=kernel//2）

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

x = torch.randn(1, 3, 32, 32)  # (batch=1, C=3, H=32, W=32)
y = conv(x)                      # (1, 16, 32, 32)
print(y.shape)

# 验证输出尺寸公式
# H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
h_out = (32 + 2*1 - 3) // 1 + 1
print("计算 H_out:", h_out)  # 32 ✓
```

---

**参数数量计算**：`kernel × kernel × in_channels × out_channels + out_channels`
= `3 × 3 × 3 × 16 + 16` = `432 + 16` = 448 个参数

### 池化层

---

```python
# MaxPool2d：取最大值
mp = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半
# AdaptiveAvgPool2d：输出固定尺寸，自适应核大小
ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 输出 (N, C, 1, 1)
# Global Average Pooling: 常见技巧，用自适应池化实现
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 把每个通道压成 1 个值
```

---

### nn.LSTM — 循环层

---

```python
import torch
import torch.nn as nn

# input_size: 每个时间步的输入特征维度
# hidden_size: 隐藏状态的特征维度
# num_layers: LSTM 层数（堆叠）
# batch_first: True → input shape: (batch, seq, feature)

lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)

x = torch.randn(4, 8, 10)  # (batch=4, seq_len=8, input_size=10)
h0 = torch.zeros(2, 4, 32)  # (num_layers, batch, hidden_size)
c0 = torch.zeros(2, 4, 32)  # (num_layers, batch, hidden_size)

output, (hn, cn) = lstm(x, (h0, c0))
print("output:", output.shape)    # (4, 8, 32) — 每个时间步的输出
print("hn:", hn.shape)           # (2, 4, 32)  — 最后一个时间步的隐藏状态
print("cn:", cn.shape)           # (2, 4, 32)  — 最后一个时间步的细胞状态

# 如果只想要最后一个时间步的输出（通常用这个）
last_output = output[:, -1, :]  # (4, 32)
print("last:", last_output.shape)
```

---

### nn.Embedding — 词嵌入层

---

```python
import torch
import torch.nn as nn

# num_embeddings: 词表大小（最大索引 + 1）
# embedding_dim: 每个词的向量维度
emb = nn.Embedding(num_embeddings=10000, embedding_dim=128)

# 输入：整数索引（LongTensor），范围 [0, num_embeddings)
x = torch.tensor([123, 456, 789])  # batch=3 的句子（每个词是一个索引）
vec = emb(x)                         # (3, 128) — 查表得到三个词的向量
print(vec.shape)                      # torch.Size([3, 128])

# 预训练词向量加载
pretrained = nn.Embedding.from_pretrained(torch.randn(10000, 128))
pretrained.weight.requires_grad = False  # 冻结，只fine-tune顶层
```

---

#### Embedding 前向传播：数学等价 vs 工程实现

Embedding 在数学上等价于 one-hot 向量与 embedding 矩阵的矩阵乘法，但工程实现是直接内存索引，两者路径不同但结果相同：

| 视角 | 实现 | 路径 |
|------|------|------|
| **数学上** | `one_hot(token_id) @ embedding_table` | 构造稀疏向量 → 矩阵乘 → 得到向量 |
| **工程上** | `embedding_table[token_id]` | 直接地址寻址，无矩阵运算 |

工程上就是 `array[index]` 的直接寻址，和哈希表查表没有本质区别。数学上写成矩阵乘法是为了理论推导方便。

#### Embedding 反向传播：梯度怎么传

核心逻辑：**谁用过我，谁把梯度传给我**。

---

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=10000, embedding_dim=128)
optimizer = torch.optim.SGD(emb.parameters(), lr=0.01)

# 模拟前向：token_id=42 被使用了两次
input_ids = torch.tensor([42, 7, 42, 99])

# 前向
vec = emb(input_ids)  # shape: (4, 128)
loss = vec.sum()       # 简单 loss

# 反向
loss.backward()

# 验证：token_id=42 被使用了两次，梯度应该累加
print("emb.weight.grad[42]:", emb.weight.grad[42])  # 非零梯度
print("emb.weight.grad[7]:", emb.weight.grad[7])    # 非零梯度
print("emb.weight.grad[99]:", emb.weight.grad[99])  # 非零梯度
print("emb.weight.grad[0]（未使用）:", emb.weight.grad[0])  # 接近零
```

---

**关键行为**：
- 同一个 token 被多次查询（重复词）→ **梯度累加**
- 没被查过的 token → 梯度为 0，不参与更新
- 计算量正比于 token 数（batch × seq_len），不依赖词表大小

#### 预训练词向量加载与微调策略

实际工程中几乎不会从零训练 embedding，而是加载预训练向量：

---

```python
import torch
import torch.nn as nn

# 方式一：from_pretrained 加载
pretrained_vectors = torch.randn(10000, 128)  # 实际从 glove/word2vec 加载
emb = nn.Embedding.from_pretrained(pretrained_vectors, freeze=False)
# freeze=False: 训练时继续更新向量（fine-tune）
# freeze=True:  训练时冻结，当静态特征用
```

---

**四种微调策略**：

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| **Frozen** | `weight.requires_grad = False`，不更新 | 数据量小、领域差异大 |
| **Full Fine-tune** | 全部可训练，正常更新 | 数据量大、领域相近 |
| **Gradual Unfreezing** | 先冻住 → 逐步解冻 | 中等数据量，避免灾难性遗忘 |
| **Adapter** | 冻住主模型，附加小型 MLP 适配器 | 算力有限、保留主模型能力 |

---

```python
# Gradual Unfreezing 示例：先只训练 head，逐步解冻
model = torchvision.models.resnet18(pretrained=True)

# 阶段1：只训练分类头
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True

# 训练若干 epoch 后...
# 阶段2：解冻最后两层
for param in model.layer4.parameters():
    param.requires_grad = True

# 阶段3：更多层...
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3  # 解冻后用较小学习率
)
```

---

---

### Winograd 卷积优化算法

Winograd 算法是深度学习中常用的卷积计算优化技术，核心思想是将卷积中的乘法次数减少。

#### 一维 Winograd F(m, r)：矩阵乘视角

以 F(2, 3) 为例：输入 4 点 `d = [d0, d1, d2, d3]`，滤波器 3 点 `g = [g0, g1, g2]`，输出 2 点：

**直接卷积**（6 次乘）：
```
y0 = d0·g0 + d1·g1 + d2·g2
y1 = d1·g0 + d2·g1 + d3·g2
```

**Winograd 重写形式**：输入组织成 2×3 数据矩阵，滤波器作为 3×1 向量：
```
D = [d0, d1, d2; d1, d2, d3]   (2×3)
y = D · g                        (矩阵乘)
```
这只是一个框架，Winograd 的精妙之处在于：**数据 D 和滤波器 g 都做预变换**，使得主要乘法变成元素乘。

**Winograd 完整公式**：
```
y = A^T · (G·g ⊙ B^T·d)
```

| 步骤 | 操作 | 计算类型 |
|------|------|---------|
| `B^T·d` | 输入变换（4 → 4 点） | 只有加减，无乘 |
| `G·g` | 滤波器变换（3 → 3 点） | 只有加减，无乘 |
| `⊙` | 逐元素乘（4 次） | **只有乘** |
| `A^T` | 输出组合（4 → 2 点） | 只有加减，无乘 |

**总乘法数**：从 6 次降到 **4 次**，代价是十几步加减法。在硬件上乘比加贵得多，所以合算。

#### Winograd F(m, r) 一般形式

Winograd 不是只能 F(2,3)，而是有很多组 (m, r) 可选：

| 形式 | 输出 m | 滤波器 r | 输入点数 | 直接乘法 | Winograd 乘法 |
|------|--------|---------|---------|---------|--------------|
| F(2, 3) | 2 | 3 | 4 | 6 | **4** |
| F(4, 3) | 4 | 3 | 6 | 12 | **6** |
| F(6, 3) | 6 | 3 | 8 | 18 | **8** |
| F(2, 5) | 2 | 5 | 6 | 10 | **6** |

**选大 m 的权衡**：m 越大乘法节省比例越高，但变换矩阵复杂度增加、数值稳定性变差。实际深度学习中选择 F(4,3) 或 F(16,3) 等。

#### 二维 Winograd F(2×2, 3×3)：Khatri-Rao 视角

对于 4×4 输入和 3×3 滤波器，可以分块成 2×3 的 tile，每个 tile 做 2D 卷积：

- 输入 4×4 → 分成重叠的 2×3 块（每个块覆盖一个 2×2 输出区域）
- 每个块是 **2×3 小矩阵**，滤波器是 **3×3 小矩阵**
- 2D 卷积等效为：每个 2×3 块与 3×3 滤波器的矩阵乘

这就是 **Khatri-Rao 乘积**（列对列的逐元素乘）的形式推广。二维 Winograd 的核心洞察：**空间分块 + 滤波器 Khatri-Rao 展开**，使得主要运算变成逐元素乘。

#### 输入不能整除 tile 怎么办

三种处理方式：

**1. 补零（Zero Padding）**：不够的地方补 0，继续用标准 tile，最常用

**2. 剩余处理（Guard / Tail）**：先用标准 tile 覆盖主体，剩下几个点单独用直接卷积（量小，浪费一点乘法无所谓）

**3. 动态选择 tile 大小**：输入长度不是 m 的倍数时，选不规则的最后 tile

#### Winograd 在深度学习中的地位

Winograd 2015-2016 年被工程化后，广泛用于 CNN 推理优化（如 TensorFlow、TensorRT）。但近几年，随着 Tensor Core 等专用矩阵乘硬件的普及，Winograd 在大 tile 场景的优势被削弱——硬件直接做矩阵乘已经足够快。但在**小滤波器（3×3）、中等 batch** 的场景下，Winograd 仍然是重要的优化手段。

---

### 卷积的工程实现：从 for 循环到 im2col 到 cuDNN

#### 为什么 for 循环做卷积极慢

用 for 循环实现单次卷积：

---

```python
# 最 naive 的实现（假设 stride=1, padding=0）
for oh in range(out_h):
    for ow in range(out_w):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                val += input[oh+kh, ow+kw] * kernel[kh, kw]
        output[oh, ow] = val
```

---

四层嵌套 + 大量内存访问 → 极慢，无法利用并行。

#### im2col：把卷积变成矩阵乘法

**核心思想**：把输入的每个滑动窗口"展开"成列，把滤波器展开成行，然后做一次矩阵乘。

```
输入 (H×W) + 滤波器 (K×K)
↓ im2col 展开
矩阵 A (out_h×out_w, K×K)  ×  矩阵 B (K×K, out_channels)
↓                            ↓
每个滑动窗口拉成一列          每个滤波器拉成一行
输出 (out_h×out_w, out_channels)
```

以 4×4 输入、3×3 滤波器、stride=1 为例：
```
输入:
  [d00,d01,d02,d03]
  [d10,d11,d12,d13]
  [d20,d21,d22,d23]
  [d30,d31,d32,d33]

im2col 展开后矩阵 A (4×9):
  [d00,d01,d02,d10,d11,d12,d20,d21,d22]   ← 第1个输出位置的窗口
  [d01,d02,d03,d11,d12,d13,d21,d22,d23]   ← 第2个位置的窗口
  [d10,d11,d12,d20,d21,d22,d30,d31,d32]
  [d11,d12,d13,d21,d22,d23,d31,d32,d33]

滤波器展开成矩阵 B (9×1):
  [k00]
  [k01]
  [k02]
  [k10]
  [k11]
  [k12]
  [k20]
  [k21]
  [k22]

输出 = A @ B  →  矩阵乘!
```

**为什么 im2col 快**：调用高度优化的 BLAS（CPU 上 MKL/OpenBLAS，GPU 上 cuBLAS）矩阵乘，并行度极高。

**内存代价**：im2col 展开后的矩阵 A 有冗余（重叠窗口的数据被重复复制）。但换来的矩阵乘速度提升远大于内存开销。

#### cuDNN：NVIDIA 的卷积底座

cuDNN 是 NVIDIA 提供的深度学习基础库，PyTorch/TensorFlow 等框架的卷积底层都调用它。

**cuDNN 选择算法的方式**：
cuDNN 内部维护多种卷积算法（GEMM-based im2col、Winograd、FFTW 等），每次运行时会对输入 shape 做一次"autotune"——在候选算法上跑一个小算例，选最快的那个缓存下来。

---

```python
# PyTorch 调用 cuDNN 的示意路径
conv = nn.Conv2d(3, 64, 3)
x = torch.randn(1, 3, 224, 224)

# 实际执行路径：
# 1. PyTorch 解析 conv 参数，构造 cudnnConvolutionDescriptor
# 2. cuDNN 根据 shape 调用 cudnnGetConvolutionForwardAlgorithm (autotune)
# 3. cuDNN 执行选中的算法（im2col + cuBLAS / Winograd / FFTW）
# 4. 结果返回 PyTorch tensor
```

---

**cuDNN 7 种卷积算法**（按场景选用）：
| 算法 | 适用场景 |
|------|---------|
| GEMM (im2col) | 通用，k×k 大滤波器、large batch |
| Winograd F(2×2, 3×3) | 小滤波器 (3×3)，中等 batch |
| Winograd F(4×4, 3×3) | 较大 tile，batch 大时更明显 |
| FFT | 极大滤波器 (5×5, 7×7)，但内存开销大 |
| Implicit GEMM | 融合了 im2col，不需显式展开，内存更省 |

**cuDNN vs cuBLAS**：cuBLAS 是通用矩阵乘，cuDNN 在其基础上做了 im2col 打包 + filter 打包，然后调 cuBLAS。cuDNN = cuBLAS + im2col + 算法选择 + 反向传播支持。

#### 三种卷积实现的对比

| 实现方式 | 并行度 | 适用场景 | 内存开销 |
|---------|--------|---------|---------|
| for 循环 | 极低 | 教学/验证 | 低 |
| im2col + cuBLAS | 高（矩阵乘） | CPU / 通用 GPU | 中（需展开） |
| cuDNN (autotune) | 最高 | 实际生产 | 可控 |

实际项目中，永远用 cuDNN，PyTorch 默认就调用它，不需要手动指定。

---

### 池化层详解：MaxPool / AvgPool / AdaptiveAvgPool

#### nn.MaxPool2d — 最大池化

---

```python
import torch.nn as nn

# kernel_size: 池化核大小
# stride: 步长（默认等于 kernel_size）
# padding: 边缘填充
# dilation: 核内空洞间距（膨胀系数）
# ceil_mode: True=向上取整，False=向下取整（默认）

pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
x = torch.randn(1, 3, 32, 32)
y = pool(x)  # (1, 3, 16, 16)
```

---

**前向过程**：在每个 kernel 窗口内取最大值，同时记录最大值位置（indices）。

**反向过程（关键）**：MaxPool 的反向传播**只沿最大值位置传递梯度**，其他位置梯度为 0。这叫 **Max Pooling 的梯度路由**。

---

```python
# 验证 MaxPool 反向传播
import torch

pool = nn.MaxPool2d(2, stride=2)
x = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
y = pool(x)
loss = y.sum()
loss.backward()

print("x.grad:")  # 只有最大值位置(=4)的梯度为1，其他为0
print(x.grad)     # [[[[0., 0.], [0., 1.]]]]
```

---

**dilation（膨胀/空洞）参数**：

---

```python
pool_dilated = nn.MaxPool2d(kernel_size=3, dilation=2)
# 等效感受野 = kernel_size + (kernel_size-1)*(dilation-1) = 3 + 2*1 = 5
# 但实际 kernel 还是 3×3，参数不变，只是采样时跳过了空洞
```

---

#### nn.AvgPool2d — 平均池化

---

```python
avg = nn.AvgPool2d(kernel_size=2, stride=2)
# 前向：取 kernel 内均值
# 反向：梯度平均分配给 kernel 内所有位置（= 1/(k*k)）
```

---

#### Global Average Pooling (GAP)

---

```python
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
x = torch.randn(4, 512, 7, 7)
y = gap(x)  # (4, 512, 1, 1) → squeeze → (4, 512)
```

---

GAP 将每个通道压缩为 1 个值，完全替代 FC 层。优点：**无参数，不易过拟合**，Inception/ResNet 等现代网络最后常用 GAP。

#### AdaptiveAvgPool2d — 自适应输出尺寸

---

```python
# 自适应输出到指定尺寸，框架自动计算 kernel_size 和 stride
adaptive = nn.AdaptiveAvgPool2d(output_size=(3, 3))  # 输出 3×3
adaptive = nn.AdaptiveAvgPool2d(output_size=(1, 1))   # 输出 1×1（GAP）
```

---

无论输入是 7×7 还是 14×14，自适应池化都能输出 3×3，**不需要关心输入尺寸**。

---

### nn.BatchNorm2d — 批归一化深入

#### 训练 vs 推理的统计量差异

---

```python
bn = nn.BatchNorm2d(num_features=64)

bn.train()   # 训练模式
bn.eval()    # 推理模式
```

---

| 模式 | 均值/方差来源 | `running_mean/var` |
|------|------------|-------------------|
| train | 当前 batch 实时计算 | **会更新**（指数移动平均） |
| eval | 不计算，用全局统计量 | **不变**，训练时累积的值 |

**running_mean/var 的更新公式**：
```
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var  = (1 - momentum) * running_var  + momentum * batch_var
```

momentum 默认 0.1（PyTorch 中 `momentum=0.1`），意义：**快速追踪近期 batch 统计量变化**。

#### track_running_stats 参数

---

```python
bn = nn.BatchNorm2d(64, track_running_stats=False)
# 推理时也用 batch 统计量（用于测试集和训练集分布差异大的场景）
```

---

#### 推理时 BN 的确定性行为

eval 模式下，`BatchNorm` 的行为是**完全确定性**的：

---

```python
bn.eval()
y1 = bn(x1)  # 无论 x1 是什么，running_mean/var 都固定
y2 = bn(x2)  # y1 和 y2 不可能相等，因为 x 不同 → 但 running_stats 确实不变
```

---

**关键误解澄清**：eval 模式不是"用 running_mean 代替 batch_mean"，而是：
- **训练阶段**：归一化用 batch 统计量（实时计算）
- **推理阶段**：归一化用 running 统计量（EMA 累积）

两者公式完全一样，只是"均值/方差从哪来"不同。

#### BatchNorm 在什么时候更新 running statistics

---

```python
bn.train()
bn.eval()  # ← 切换到 eval 后，running_mean/var **完全停止更新**

bn.eval()
bn.train()  # ← 切换回 train 后，继续用 batch 统计量更新 running_mean/var
```

---

---

### nn.Dropout — 随机失活深入

#### 训练模式的 mask 生成机制

---

```python
dropout = nn.Dropout(p=0.5)  # p=丢弃概率

x = torch.tensor([1., 2., 3., 4., 5.])
y_train = dropout(x)
# 前向：随机生成 0/1 mask，以 p 概率置 0，剩余 / (1-p)
# 反向：mask=0 的位置梯度=0（不更新），mask=1 的位置正常回传
```

---

**mask 生成过程（PyTorch 内部）**：

---

```python
# 伪代码
mask = (torch.rand(x.shape) > p) / (1 - p)  # 注意除以 (1-p)
y = x * mask
```

---

#### 反向传播：mask 本身不参与梯度计算

Dropout 反向传播的梯度 = 上游梯度 × mask，关键：**mask 在反向时是固定的**（和前向用同一个 mask），mask 本身不产生梯度。

---

```python
# 验证：Dropout 反向传播
x = torch.tensor([1., 2., 3.], requires_grad=True)
dropout = nn.Dropout(p=0.5)
y = dropout(x)
loss = y.sum()
loss.backward()

# x.grad: 某些位置梯度为 0（被 mask 丢弃），某些位置正常
# x.grad != 0 的位置和前向 mask=1 的位置完全一致
```

---

#### Inverted Dropout（PyTorch 使用的方式）

训练时做缩放（除以 1-p），推理时不做任何操作：

---

```python
# Inverted Dropout（PyTorch / TF / 大多数框架）
y = x * mask / (1-p)  # 训练时

# 推理时：所有神经元参与，输出期望 E[y] = x
```

---

优势：**推理代码和训练代码完全一致**，只需切换 mode。相对于"推理时手动缩放"更简洁。

---

### nn.ConvTranspose2d — 转置卷积（反卷积）

转置卷积不是卷积的逆运算，而是**卷积的梯度运算**（conv transpose = gradient with respect to input）。

---

```python
conv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
# stride=2: 输出尺寸 = 输入尺寸 × 2（upsample）
# output_padding=1: 消除 stride>1 时的一个单元偏移
```

---

**用途**：上采样（GAN、U-Net、语义分割）、生成器网络。

**输出尺寸公式**：
```
H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
```

---

```python
# 示例
deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
x = torch.randn(1, 1, 4, 4)
y = deconv(x)  # (1, 1, 8, 8) — 4×2=8
```

---

---

### nn.LSTM — 长短期记忆网络深入

#### LSTM 的四个门控机制

---

```python
import torch.nn as nn

lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True, bidirectional=False)

x = torch.randn(4, 8, 10)  # (batch, seq_len, input_size)
output, (hn, cn) = lstm(x)
```

---

LSTM 内部有 4 个门控，计算公式：

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # 遗忘门：决定丢弃多少旧信息
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # 输入门：决定写入多少新信息
C'_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # 候选记忆：新的候选值
C_t = f_t * C_{t-1} + i_t * C'_t         # 细胞状态：遗忘+输入的组合

o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # 输出门：决定输出多少
h_t = o_t * tanh(C_t)                    # 隐藏状态：输出门 × tanh(细胞状态)
```

#### 为什么 LSTM 能避免梯度消失

关键在于**细胞状态 C_t 的更新方式**：

```
C_t = f_t * C_{t-1} + i_t * C'_t
```

- 遗忘门 f_t 接近 1 时，梯度可以几乎无损地传回很久以前的时间步
- 加法操作（而非连乘）是 LSTM 避免梯度消失的核心：**∂C_t/∂C_{t-1} = f_t**，不是连乘
- 梯度沿细胞状态路径传递时，"*"操作被"+"替代，梯度传播变成加法而非乘法

#### packed_padded_sequence 与 padding mask

处理变长序列时，需要 pad 后一起 batch，但 pad 的位置不应参与计算：

---

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 变长序列
lengths = [5, 3, 7]  # 各序列实际长度
x_padded = torch.nn.utils.rnn.pad_sequence([...], batch_first=True)

# 打包（按长度降序，自动处理 pad）
packed = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=True)
output_packed, (hn, cn) = lstm(packed)

# 解包
output, _ = pad_packed_sequence(output_packed, batch_first=True)
# output[:, 3, :] 在 lengths[0]=5 时，位置3的数据是真实的，位置>3是 pad 的
```

---

---

### nn.Softmax / LogSoftmax — 归一化指数函数

---

```python
import torch.nn as nn

sm = nn.Softmax(dim=-1)
log_sm = nn.LogSoftmax(dim=-1)  # 用于 CrossEntropyLoss（数值稳定）

x = torch.tensor([1., 2., 3.])
y = sm(x)  # [e^1, e^2, e^3] / (e^1+e^2+e^3) ≈ [0.09, 0.24, 0.67]
y_log = log_sm(x)  # log([...]) ≈ [-2.41, -1.41, -0.41]
```

---

**dim 很重要**：在哪个维度做 softmax，就在哪个维度做归一化（和为 1）。

---

```python
# (batch, seq_len, vocab_size) = (2, 5, 10000)
logits = torch.randn(2, 5, 10000)
sm = nn.Softmax(dim=-1)  # ← 在 vocab_size 维度归一化 → 每词概率分布
attn_weights = sm(logits)  # 每个位置一个概率分布
```

---

---

### 激活函数对比总结

| 激活函数 | 公式 | 输出范围 | 优点 | 缺点 |
|---------|------|---------|------|------|
| **ReLU** | max(0, x) | [0, +∞) | 计算快、无梯度饱和 | 神经元死亡问题 |
| **LeakyReLU** | x if x>0 else α·x | (-∞, +∞) | 避免神经元死亡 | 多了超参 α |
| **GELU** | x·Φ(x) | (-∞, +∞) | Transformer 最常用，平滑 | 计算略慢 |
| **SiLU / Swish** | x·sigmoid(x) | (-∞, +∞) | 自门控，比 ReLU 更平滑 | 计算慢 |
| **Sigmoid** | 1/(1+e^{-x}) | (0, 1) | 概率输出 | 梯度易饱和、梯度消失 |
| **Tanh** | (e^x - e^{-x})/(e^x + e^{-x}) | (-1, 1) | 零中心 | 梯度易饱和 |

**工程选择**：
- CNN / 通用 → **ReLU**（最快）
- Transformer / Attention → **GELU**（BERT、ViT、GPT 等）
- 需要概率输出 → **Sigmoid**（多标签分类）
- 非线性输出 → **Tanh**（LSTM 门控内部常用）

**GELU vs ReLU**：GELU 是 ReLU 的平滑近似，梯度在负区间也有小幅非零值，不会完全"死亡"。Transformer 时代 GELU 成为默认选择。

---

## 3-6 Sequential 与模块化设计

### nn.Sequential — 两种定义方式

---

```python
import torch
import torch.nn as nn

# 方式一：直接传入（位置索引，调试不方便）
model = nn.Sequential(
    nn.Linear(10, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)
```

---

```python
# 方式二：OrderedDict 命名（推荐，调试友好）
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1',   nn.Linear(10, 8)),
    ('relu1', nn.ReLU()),
    ('fc2',   nn.Linear(8, 4)),
    ('relu2', nn.ReLU()),
    ('fc3',   nn.Linear(4, 2))
]))
```

---

### Sequential 内部如何工作

Sequential 本身就是一个 `nn.Module`，`forward` 按顺序执行每一层：

---

```python
# Sequential 内部 forward 等价于：
def forward(self, x):
    for layer in self._modules.values():
        x = layer(x)
    return x
```

---

**OrderedDict vs 直接传入的区别**：两者功能完全相同，OrderedDict 只是给每一层起了名字，使得 `named_children()` 和 `state_dict()` 的 key 更清晰。

---

```python
for name, module in model.named_children():
    print(name, '→', module)

# OrderedDict 输出：
# fc1 → Linear(in=10, out=8)
# relu1 → ReLU()
# fc2 → Linear(in=8, out=4)
# relu2 → ReLU()
# fc3 → Linear(in=4, out=2)
```

---

### Sequential 的局限：四种必须自定义的场景

Sequential 只能**线性串联**，以下四种情况必须自定义 Module：

**1. 残差连接**

---

```python
# Sequential 实现不了，必须自定义
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x) + x  # ← 跳跃连接
```

---

**2. 多输入**

---

```python
# 多输入必须自定义
class TextImageFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_net = nn.Linear(768, 256)
        self.img_net  = nn.Linear(512, 256)

    def forward(self, text, img):
        return self.text_net(text) * self.img_net(img)  # ← 多输入
```

---

**3. 多输出**

---

```python
# 多输出必须自定义
class DualOutputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(512, 10)
        self.embedding  = nn.Linear(512, 128)

    def forward(self, x):
        return {
            'logits': self.classifier(x),
            'features': self.embedding(x)
        }
```

---

**4. 共享层**

---

```python
# 同一层在两处使用，必须自定义
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(512, 256)  # ← 两处共享

    def forward(self, x1, x2):
        return self.shared(x1), self.shared(x2)  # ← 同一层走两次
```

---

### Sequential 适用场景对照表

| 场景 | Sequential | 自定义 Module |
|------|-----------|--------------|
| 线性串联：Conv → BN → ReLU → Pool | ✅ | ✅ |
| 快速原型：几行搭一个 MLP | ✅ | ✅ |
| 残差连接（ResNet Block） | ❌ | ✅ |
| 多分支网络（FPN、Inception、U-Net） | ❌ | ✅ |
| 多输入/多输出（多模态） | ❌ | ✅ |
| 共享层（Siamese Network） | ❌ | ✅ |

### 模型设计的分层思维

```
Input → [Conv → BN → ReLU → Pool] × N → FC → Output
              ↑
         用 nn.Sequential 打包
```

---

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        return self.fc(x)
```

---

### 实际工程选择原则

- **简单任务 / 原型验证** → Sequential 够用，几行代码搞定
- **生产级 / 复杂结构** → 自定义 Module（更清晰可控，可读性好）
- **永远不确定** → 自定义 Module，因为后续改起来容易

---

## 面试常考点

**Q1: `nn.Module` 的 `forward` 为什么只需写前向，反向自动搞定？**
- PyTorch autograd 根据前向运算自动构建计算图，每种运算对应一个反向函数（grad_fn）
- `model(x)` 实际调用 `nn.Module.__call__(x)` → 执行 forward → 执行 hooks

**Q2: `model(img)` 背后发生了什么？**
- `nn.Module.__call__` 被调用 → 执行 `forward()` → 执行所有注册的 `forward_hooks` → 返回输出
- `__call__` 和直接调 `forward()` 的区别在于 hooks 是否被执行

**Q3: `model.train()` vs `model.eval()` 区别？**
- `train()`：BatchNorm 用当前 batch 的均值/方差，Dropout 随机置零
- `eval()`：BatchNorm 用全局 moving average，Dropout 全部通过
- 两者都只影响特定层的统计行为，不影响其他层

**Q4: 如何冻结部分层？冻结后梯度怎么传？**

---

```python
for param in model.layer1.parameters():
    param.requires_grad = False
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],  # optimizer 只更新解冻的参数
    lr=0.01
)
```

---

- 冻结（`requires_grad=False`）：梯度**原封不动**往上继续传，只是该参数本身不记录梯度
- 对比 `detach()`：真正截断梯度流，梯度停止传播
- 冻结不等同于"置零"或"常数"，它仍然参与计算图，只是可学习性被关闭

**Q5: `state_dict()` 保存的是什么？**
- 所有 `nn.Parameter` 的字典，key 是参数名（如 `layer1.weight`），value 是张量
- 不包含模型结构（需要先实例化再 load）

**Q6: `children()` vs `modules()` 区别？**
- `children()`：只返回直接子模块（深度1层）
- `modules()`：返回所有模块，包括嵌套的子子模块（深度遍历）