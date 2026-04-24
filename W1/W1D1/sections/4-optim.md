# 四、torch.optim — 优化器

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| SGD + Momentum | 基础优化器 + 动量加速 | momentum 是什么，为什么能加速 |
| RMSprop / Adagrad | 自适应学习率早期方案 | 解决什么问题 |
| Adam | 自适应学习率 + 偏置校正 | 一阶/二阶矩估计是什么 |
| AdamW | Adam + weight decay 分离 | 为什么比 Adam + L2 好 |
| 学习率调度 | `lr_scheduler` | 常用调度策略及场景 |
| 参数分组 | 不同层不同学习率 | 怎么配 |
| Gradient Clipping | 梯度裁剪 | 防止梯度爆炸 |
| Zero_grad + 状态保存 | 清零梯度 + 保存加载 | 为什么要手动调用 |

**⚠️ 面试必答题：**
- SGD 和 Adam 的区别？各自适用场景？
- AdamW 和 Adam + L2 正则的区别？
- 学习率衰减策略有哪些？
- 为什么梯度要用 `zero_grad()` 清零，不能累加？
- 梯度裁剪具体怎么操作？

---

## 4-1 SGD 与 Momentum

### SGD（随机梯度下降）

PyTorch 中 `SGD` 实际上是大 batch 的梯度下降（不一定是单样本）。

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    loss.backward()
    optimizer.step()
```

**关键参数：**
- `lr`: 学习率
- `momentum`: 动量系数（默认 0）
- `weight_decay`: L2 正则化系数

### Momentum（动量）

物理概念：像滚动的球有惯性，梯度会累积。

**公式：**
```
v_t = γ * v_{t-1} + lr * gradient
param = param - v_t
```

- γ（gamma）: 动量系数，通常 0.9
- lr: 学习率

```python
# 有动量 vs 无动量
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 对比
optimizer_no_mom = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
```

**为什么能加速：**
- 减少振荡：在梯度变化方向一致的维度累积，在振荡方向抵消
- 加速收敛：在损失谷底时，惯性帮助跳出局部极小

### Nesterov 动量

是 Momentum 的改进版，先按动量方向预估，再计算梯度。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, 
                           momentum=0.9, nesterov=True)
```

---

## 4-2 RMSprop / Adagrad

### Adagrad

**特点**：每个参数独立自适应学习率，梯度大的参数学习率衰减快。

**适用**：稀疏特征（如 NLP、Embedding）。

```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, 
                                lr_decay=0.01, weight_decay=0)
```

**问题**：学习率会单调下降，后期训练困难。

### RMSprop

**改进**：引入滑动平均，平滑学习率变化。

**公式：**
```
cache = γ * cache + (1-γ) * gradient²
param = param - lr * gradient / (√cache + ε)
```

- γ: 衰减系数，通常 0.99
- ε: 防止除零，通常 1e-8

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, 
                               alpha=0.99, eps=1e-8)
```

---

## 4-3 Adam 原理

### Adam（Adaptive Moment Estimation）

**目前最流行的优化器**，结合了 Momentum 和 RMSprop。

**核心思想**：用一阶矩估计（类似动量）和二阶矩估计（类似 RMSprop的自适应学习率）分别修正梯度。

**公式：**
```
# 一阶矩（动量估计）
m_t = β1 * m_{t-1} + (1-β1) * gradient

# 二阶矩（梯度的方差估计）
v_t = β2 * v_{t-1} + (1-β2) * gradient²

# 偏置校正（至关重要！）
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)

# 更新参数
param = param - lr * m_hat / (√v_hat + ε)
```

- β1: 一阶矩衰减，通常 0.9
- β2: 二阶矩衰减，通常 0.999
- ε: 防止除零，通常 1e-8
- t: 迭代次数（从 1 开始）

**为什么需要偏置校正？**
- 初始化时 m_0 = 0, v_0 = 0
- 如果不校正：第一次更新 lr * gradient（实际学习率被缩小）
- 校正后：第一轮就是正确学习率

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, 
                          betas=(0.9, 0.999), eps=1e-8)
```

### Adam 适用场景

| 推荐使用 | 不推荐 |
|---|---|
| 深度网络（CNN/Transformer） | 简单线性模型 |
| 非稀疏数据 | 需要精确调参的场景 |
| 快速实验 | lr 敏感的任务 |

---

## 4-4 AdamW vs Adam + L2

### 核心区别

**Adam + L2**：
```
grad = gradient + weight_decay * param
m_t = β1 * m_{t-1} + (1-β1) * grad
```

**AdamW**：
```
m_t = β1 * m_{t-1} + (1-β1) * gradient
param = param - lr * (m_hat / √v_hat + weight_decay * param)
```

区别：**L2 正则化通过梯度更新（影响动量估计）**，而 **AdamW 在参数更新时直接减（ weight_decay * param * lr）**。

### 为什么 AdamW 更好

1. **解耦**：学习率和正则化强度独立
2. **理论保证**：与 L2 正则化的原始目标函数等价
3. **实践效果**：通常 weight_decay 设置为 0.01~0.05（而非 Adam+L2 的 1e-4）

```python
# Adam + L2（传统方式）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# AdamW（推荐方式）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**面试话术**：
> "AdamW 是 Adam 的改进，把 weight_decay 从梯度项移到了参数更新项，避免了动量估计被 L2 正则化污染。实际应用中通常用 0.01-0.05 的 weight_decay，比 Adam+L2 的 1e-4 高两个数量级。"

---

## 4-5 学习率调度（lr_scheduler）

### 常用策略

| 调度器 | 公式 | 适用场景 |
|---|---|---|
| StepLR | 每 N 个 epoch 固定衰减 | 普通训练 |
| MultiStepLR | 指定 epoch 列表衰减 | 已知转折点 |
| CosineAnnealingLR | 余弦曲线 | 图像分类 |
| ReduceLROnPlateau | 指标不降时衰减 | 验证集优化 |
| Warmup + Cosine | 先升后降 | Transformer |

### StepLR

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(50):
    train()
    scheduler.step()
    print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
# Epoch 0-9: 0.001
# Epoch 10-19: 0.0005
# Epoch 20-29: 0.00025
# ...
```

### CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

### Warmup + Cosine（在 PyTorch 2.0+）

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.1,
    epochs=50,
    steps_per_epoch=len(dataloader)
)
```

###ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

for epoch in range(50):
    val_loss = validate()
    scheduler.step(val_loss)  # 根据指标调整
```

**面试话术**：
> "常用的是 StepLR（简单）、CosineAnnealing（效果好）和 ReduceLROnPlateau（自动）。Transformer 类任务一般用 OneCycleLR 先升后降，能获得更好收敛。"

---

## 4-6 参数分组（不同层不同学习率）

### 场景

- 预训练模型微调：特征提取层用小学习率，分类头用大学习率
- 不同层用不同weight_decay

### 实现

```python
# 特征提取层（卷积层）：小学习率，无 weight_decay
base_params = [p for n, p in model.named_parameters() if 'feature' in n]
# 分类头：大学习率，有 weight_decay
head_params = [p for n, p in model.named_parameters() if 'head' in n]

optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': 1e-4, 'weight_decay': 0},
    {'params': head_params, 'lr': 1e-3, 'weight_decay': 0.01}
])
```

### 另一种写法（通过 dict）

```python
optimizer = torch.optim.AdamW([
    {'params': model.feature.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
], weight_decay=0.01)
# 注意：weight_decay 对所有参数生效，可设为 0，再用 params 指定
```

---

## 4-7 Gradient Clipping（梯度裁剪）

### 为什么要裁剪

防止梯度爆炸（尤其 LSTM、Transformer、GAN）。

**常见阈值**：1.0 或 5.0

### 两种方式

#### nn.utils.clip_grad_norm_
按全体参数 L2 范数裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 计算方式
total_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters()))
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in model.parameters():
        p.grad *= scale
```

#### nn.utils.clip_grad_value_
按单个参数值裁剪：

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
# 所有参数的梯度被裁到 [-1.0, 1.0]
```

**推荐**：clip_grad_norm_（更常用）

```python
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

**面试话术**：
> "梯度裁剪防止梯度爆炸，尤其是 RNN/Transformer。我在实际项目中一般用 clip_grad_norm_，阈值设为 1.0 或 5.0，效果很好。"

---

## 4-8 zero_grad() + 状态保存

### zero_grad() 为什么必须手动调用

PyTorch 默认 **累加梯度**（方便大 batch）。

```python
# 错误：梯度会爆炸
for inputs, targets in dataloader:
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()  # 每次都在累加！
    optimizer.step()

# 正确
for inputs, targets in dataloader:
    optimizer.zero_grad()  # 先清零
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
```

**set_to_none=True**（更高效）:

```python
optimizer.zero_grad(set_to_none=True)
# 和 zero_grad() 等价，但内存更省（设为 None 而非 0）
```

### 参数状态保存

optimizer 有内部状态（momentums 等），保存时必须一起保存。

```python
# 保存
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
```

**注意**：不同 optimizer 状态不兼容（Adam 的状态不能加载到 SGD）

---

## 面试汇总

### Q1: SGD 和 Adam 的区别？

| | SGD | Adam |
|---|---|---|
| 学习率 | 固定（需手动调） | 自适应 |
| 收敛速度 | 慢 | 快 |
| 调参难度 | 高 | 低 |
| 适用场景 | 简单模型、精确调参 | 深度网络 |

### Q2: AdamW 和 Adam + L2 的区别？

- Adam + L2：L2 正则化进入梯度，影响动量估计
- AdamW：weight_decay 在参数更新时直接减，独立于动量
- 推荐用 AdamW

### Q3: 常用学习率衰减策略？

- StepLR：每 N 个 epoch 固定衰减
- CosineAnnealing：余弦曲线
- ReduceLROnPlateau：验证集不降时自动降
- OneCycleLR：先升后降（Transformer 推荐）

### Q4: 为什么 zero_grad() 必须手动调用？

- PyTorch 默认梯度累加（支持大 batch）
- 每次 backward 会累加到 .grad，所以要清零

### Q5: 梯度裁剪怎么做？

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 代码练习

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 准备数据
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)

# 2. 模型
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

# 3. 优化器：AdamW + 参数分组
optimizer = optim.AdamW([
    {'params': model[0]..parameters(), 'lr': 1e-3},  # 卷积层
    {'params': model[2].parameters(), 'lr': 1e-4}     # 输出层
], weight_decay=0.01)

# 4. 学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 5. 训练循环
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_x)
        loss = nn.MSELoss()(output, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
```

---

## 参考资料

- PyTorch 官方文档: https://pytorch.org/docs/stable/optim.html
- 论文: Adam: A Method for Stochastic Optimization (2014)
- 论文: Decoupled Weight Decay Regularization (2019) — AdamW