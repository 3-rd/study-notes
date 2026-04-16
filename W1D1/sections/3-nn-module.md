# 3. nn.Module — 模型构建核心

## 3-1 模块定义与参数管理

### 为什么要继承 nn.Module？

`nn.Module` 是 PyTorch 的模型组织框架。继承它不是为了"继承方法"，而是为了获得它提供的**基础设施**：

1. **自动参数收集** — `model.parameters()` 把所有 `nn.Parameter` 收拢，optimizer 一行搞定
2. **设备统一管理** — `model.to('cuda')` 一次把所有参数和数据搬到 GPU
3. **状态追踪** — 前向/反向 hooks、buffer 管理、模型结构序列化

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

### super().__init__() 为什么必须调用？

`nn.Module.__init__()` 里注册了若干 PyTorch 内部簿记结构——参数注册表、buffer 注册表、forward hook 容器等。不调用 super，`self.weight` 就不会被当成参数收集，`model.parameters()` 里就没有它，`model.to('cuda')` 也不会移动它。

```python
class BadLinear(nn.Module):
    def __init__(self, in_f, out_f):
        # 忘记调用 super().__init__()
        self.weight = nn.Parameter(torch.randn(in_f, out_f))  # 不会被注册！

model = BadLinear(10, 5)
print("parameters:", list(model.parameters()))  # [] — 空！weight 丢失了
print("weight 是否在 params:", model.weight in model.parameters())  # False
```

### nn.Parameter 是什么？

```python
nn.Parameter(tensor) ≈ requires_grad=True + 自动注册到父 Module
```

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

```python
# 验证：Module 本身不在计算图里
x = torch.tensor([1., 2.], requires_grad=True)
linear = nn.Linear(2, 3)
y = linear(x)  # linear 是个 Python 对象，不是 tensor

print("linear.weight 在图中:", linear.weight.requires_grad)  # True
print("linear.weight.grad_fn:", linear.weight.grad_fn)      # None — 叶子节点！
print("y.grad_fn:", y.grad_fn)                             # <AddmmBackward>
```

### forward 为什么只需写前向？

因为 PyTorch 的 autograd 引擎根据**前向运算的每一步**，自动构建反向传播图。`forward` 里写了什么运算，autograd 就自动生成对应的反向函数（grad_fn）。

`model(x)` 背后发生了什么：

```python
output = model(x)

# 等价于：
output = nn.Module.__call__(model, x)
# 1. model.__call__() 执行（Python 魔法方法）
# 2. 调用 model.forward(x)                    ← 你的代码在这里
# 3. 执行所有注册的 forward_hooks            ← 可选拦截点
# 4. 返回 output
```

`__call__` 是 Python 的语法糖：任何 `obj(args)` 调用，实际是 `obj.__call__(args)`。PyTorch 在 `nn.Module.__call__` 里插入了一些钩子（forward pre-hook、forward post-hook），所以直接写 `forward` 不够——需要经过 `__call__`。这也是为什么 `model(x)` 和 `model.forward(x)` **不完全等价**。

---

## 3-2 遍历与结构（children / modules / named_parameters）

### 四个遍历方法的区别

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

| 方法 | 返回内容 | 层级 |
|------|---------|------|
| `model.modules()` | 所有模块（包括自己、根） | 深度优先遍历（根 → 子 → 子...） |
| `model.children()` | 直接子模块（不包括嵌套更深） | 仅一层 |
| `model.named_parameters()` | 所有参数的 (name, tensor) | 深度遍历 |
| `model.parameters()` | 所有参数的 tensor | 深度遍历 |

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

## 3-3 state_dict 与模型保存加载

### 两种保存方式的区别

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

**为什么推荐 state_dict**：
- 文件更小（只存参数，不存模型结构）
- 不依赖类定义，迁移性更强
- 断点续训时通常需要同时保存 optimizer 的 state_dict

### 完整断点续训

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

### 部分加载（strict / 非 strict）

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

## 3-4 train / eval 模式与 BatchNorm / Dropout 机制

### Dropout 机制

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

**Dropout 训练 vs 评估的行为差异**：

| 模式 | 行为 | 效果 |
|------|------|------|
| train | 随机置零（p 的概率），剩余元素除以 `(1-p)` 做缩放 | 防止过拟合 |
| eval | 所有元素原样通过 | 确定性输出 |

**缩放的必要性**：`E[dropout(x)] = (1-p) × x/(1-p) = x`，数学期望不变，训练和测试的输出期望一致。

### BatchNorm 机制

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

**BatchNorm 训练 vs 评估的行为差异**：

| 模式 | 计算均值/方差 | 使用均值/方差 |
|------|-------------|-------------|
| train | 用当前 batch 的统计量 | 当前 batch 的统计量 |
| eval | 不计算 | 用**全局移动平均**（moving average，训练时累积的） |

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

### model.train() vs model.eval() 实际影响

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

```python
# 两个句子，同一位置 pos=0
# 句子A："The cat sits"     → token[0] = "The"
# 句子B："A dog runs"       → token[0] = "A"
# BatchNorm 强制让两个位置的激活值统计分布相同——灾难性的
```

#### LayerNorm 保留的是什么？

LayerNorm 对每个 token 的向量做归一化，数学上等价于将向量投影到**高维球面上**（固定模长，消除绝对强度）。语义信息主要编码在**方向**里，而不是模长里——所以 LayerNorm 消除模长的同时保留了语义。

#### 验证：NLP 场景下 BatchNorm 的问题

以 NLP 输入 (batch=2, seq_len=3, embed=4) 为例：

```python
# 句子A：[1,2,3,4], [5,6,7,8], [9,10,11,12]
# 句子B：[10,10,10,10], [20,20,20,20], [30,30,30,30]

# LayerNorm 后：句子A token[0] = [-1.34, -0.45, 0.45, 1.34]（递增关系保留）
#              句子B token[0] = [-1.34, -1.34, -1.34, -1.34]（各维度相同，LayerNorm后仍相同）

# BatchNorm 后（embed 维度跨 batch 归一化）：
# 句子A token[0] 和 句子B token[0] → 相同值！
# 两个语义完全不同的句子被强制拉平了
```

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

**参数数量**：`weight: (5, 10)` + `bias: (5,)` = 55 个参数

### nn.Conv2d — 卷积层

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

**参数数量计算**：`kernel × kernel × in_channels × out_channels + out_channels`
= `3 × 3 × 3 × 16 + 16` = `432 + 16` = 448 个参数

### 池化层

```python
# MaxPool2d：取最大值
mp = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半
# AdaptiveAvgPool2d：输出固定尺寸，自适应核大小
ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 输出 (N, C, 1, 1)
# Global Average Pooling: 常见技巧，用自适应池化实现
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 把每个通道压成 1 个值
```

### nn.LSTM — 循环层

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

### nn.Embedding — 词嵌入层

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

## 3-6 Sequential 与模块化设计

### nn.Sequential — 快速串联

```python
import torch
import torch.nn as nn

# 方式一：直接传入层
model = nn.Sequential(
    nn.Linear(10, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Sigmoid()
)

x = torch.randn(5, 10)
print(model(x).shape)  # (5, 2)
```

```python
# 方式二：使用 OrderedDict 命名层（方便调试）
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20, 20, 5)),
    ('relu2', nn.ReLU()),
    ('pool', nn.MaxPool2d(2, 2)),
]))

for name, module in model.named_children():
    print(f"  {name}: {module}")
```

### 何时不用 Sequential

Sequential 适合**线性串联**的场景。如果有**分支、跳跃连接、多个输入输出**，就必须自定义 Module：

```python
# 残差连接：必须自定义，不能用 Sequential
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x) + x  # 跳跃连接 ← Sequential 实现不了
```

```python
# 多输入：必须自定义
class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_net = nn.Linear(512, 128)
        self.text_net = nn.Linear(768, 128)

    def forward(self, img, text):
        return self.img_net(img) * self.text_net(img)  # 元素乘法
```

### 模型设计的分层思维

```
Input → [Conv → BN → ReLU → Pool] × N → FC → Output
              ↑
         用 nn.Sequential 打包
```

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

**Q4: 如何冻结部分层？**
```python
for param in model.layer1.parameters():
    param.requires_grad = False
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],  # optimizer 只更新解冻的参数
    lr=0.01
)
```

**Q5: `state_dict()` 保存的是什么？**
- 所有 `nn.Parameter` 的字典，key 是参数名（如 `layer1.weight`），value 是张量
- 不包含模型结构（需要先实例化再 load）

**Q6: `children()` vs `modules()` 区别？**
- `children()`：只返回直接子模块（深度1层）
- `modules()`：返回所有模块，包括嵌套的子子模块（深度遍历）
