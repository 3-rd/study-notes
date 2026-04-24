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

**核心**：每一步独立更新，看当前梯度，不依赖历史。

```python
Δw = lr × gradient
w = w - Δw
```

**特点**：
- 稳定，但收敛慢（特别是缓坡）
- 学习率调不好容易震荡

### Momentum（动量）

**核心**：历史梯度的指数加权累积，再用来更新参数。

**公式：**
```python
momentum = γ × momentum + lr × gradient
w = w - momentum
```

**展开形式（关键）：**
```
momentum_t = lr × [g_t + γ·g_{t-1} + γ²·g_{t-2} + γ³·g_{t-3} + ...]
```

- γ=0.9 时，10步前的梯度权重只剩约 35%
- 所以 momentum ≈ 过去 ~10 步梯度的加权平均 × lr
- **本质：参数更新量 = lr × 历史梯度的指数加权累积**

**直观例子**（lr=0.1, γ=0.9, momentum=0）：

| Step | 当前梯度 | Momentum | 参数更新 Δw | 累计更新 |
|------|---------|---------|-----------|---------|
| 1 | 3.0 | 0.9×0 + 0.1×3.0 = 0.30 | -0.30 | -0.30 |
| 2 | 2.0 | 0.9×0.30 + 0.1×2.0 = 0.47 | -0.47 | -0.77 |
| 3 | 1.0 | 0.9×0.47 + 0.1×1.0 = 0.52 | -0.52 | -1.29 |
| 4 | 0.5 | 0.9×0.52 + 0.1×0.5 = 0.52 | -0.52 | -1.81 |

**vs 无动量SGD**（同样4步梯度）：累计更新只有 **-0.65**，动量版是 **-1.81**，快了近3倍。

**为什么能加速：**
- 方向一致：速度累积，越走越快
- 方向振荡：正负抵消，自动过滤噪音

**为什么叫"动量"**：物理上像滚球下山，即使坡度变缓，惯性也会推着继续滚。

### Nesterov 动量

**核心**：先按惯性走一步，在那个位置预判梯度，再回来更新。

```python
preview_w = w - γ × momentum          # 预判位置（先按惯性走）
gradient_preview = 在 preview_w 处算的梯度  # 预判梯度
momentum = γ × momentum + lr × gradient_preview
w = w - momentum
```

**直观理解**：
- 普通 Momentum：低头冲，撞到墙才减速
- Nesterov：抬头看路，快到墙了提前减速

**代码：**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, 
                           momentum=0.9, nesterov=True)
```

**三种方式对比**（lr=0.1, γ=0.9）：

| | 无动量 SGD | 带动量 SGD | Nesterov |
|---|---|---|---|
| 每步更新 | lr × g_t | lr × [g_t + γ·g_{t-1} + ...] | 预判位置算梯度，累积同动量 |
| 速度 | 无 | 有累积 | 有累积 |
| 减速机制 | 无 | 靠当前梯度变小 | **提前**感知梯度变小 |
| 4步累计更新 | -0.65 | -1.81 | 更稳（真实场景优于动量） |

### zero_grad 三种方式

梯度清零有三个入口，语义略有不同：

```python
optimizer.zero_grad()    # 优化器清零：只清它管理的那些参数的梯度
model.zero_grad()       # 模型清零：递归清所有子模块的梯度
for p in model.parameters():
    p.grad = None       # 手动清零：最直接
```

**为什么 PyTorch 把 zero_grad 放在 optimizer 上？**
- optimizer 创建时就绑定了要管理的参数
- `optimizer.zero_grad()` = "把管的这批参数的梯度清零"
- 语义自然：谁管谁清

**PyTorch 默认累加梯度**（不是覆盖），方便大 batch：
```python
loss.backward()      # grad += 梯度
loss.backward()      # grad += 新梯度（累加了！）
optimizer.zero_grad()  # 下一步前必须清零
```

---

## 4-2 RMSprop / Adagrad

> 这两个算法的核心都是：**每个参数维护自己的 cache，用 cache 来自适应调节学习率**。
> optimizer 内部为每个参数维护一组独立的 cache 值。

### Adagrad

**核心**：历史梯度平方的**累加和**，作为学习率的分母。

```python
cache_i = cache_i + gradient_i²     # 每个参数单独累积
param_i = param_i - lr × gradient_i / (√cache_i + ε)
```

**展开形式**：
```
cache_i(t) = g₁² + g₂² + g₃² + ... + g_t²
```

**特点**：
- 稀疏特征友好（梯度小的参数 lr 保持较大）
- cache 只增不减，lr 单调下降
- lr_decay 让 lr 衰减**更快**（叠加在 Adagrad 原有的衰减上），不是更慢

**例子**（lr=0.01, lr_decay=0.01）：

| Step | lr_decay=0 | lr_decay=0.01 |
|------|-----------|--------------|
| 1 | 0.0100 | 0.0099 |
| 100 | 0.0100 | 0.0050 |
| 1000 | 0.0100 | 0.0009 |

**问题**：两层衰减叠加，训练后期 lr 接近 0，无法继续学习。

### RMSprop

**核心**：把 Adagrad 的累加和改成**指数移动平均（EMA）**，让历史梯度的影响自然消退。

```python
cache_i = γ × cache_i + (1-γ) × gradient_i²
param_i = param_i - lr × gradient_i / (√cache_i + ε)
```

**展开形式**：
```
cache_i(t) = (1-γ) × [g_t² + γ·g_{t-1}² + γ²·g_{t-2}² + ...]
```

- γ=0.9 时，10 步前的梯度权重只剩约 **3.5%**
- 所以 cache ≈ 最近 ~10 步梯度平方的均值，不是全量累加

**vs Adagrad 对比**：

| | Adagrad | RMSprop |
|---|---|---|
| cache 累积 | 累加和（只增不减） | EMA（γ 衰减权重） |
| lr 变化 | 单调下降 | 趋向稳定值 |
| 需要 lr_decay | 是（额外压制） | 否（EMA 本身是平滑衰减） |
| 适用场景 | 稀疏特征 | 非稀疏 / 深度网络 |

**为什么 RMSprop 不需要 lr_decay**：EMA 天然有"记忆窗口"效果，久远梯度自动被 γ^n 指数衰减压制，不需要额外因子再压 lr。

---

## 4-3 Adam 原理

### Adam（Adaptive Moment Estimation）

**目前最流行的优化器**，结合了 Momentum 和 RMSprop。

## 4-3 Adam 原理

### EMA（指数移动平均）

在讲 Adam 之前，先理解 EMA。

**核心思想**：给近期数据高权重，历史数据权重指数衰减。

```python
EMA_t = γ × EMA_{t-1} + (1-γ) × x_t
```

**展开形式**：
```
EMA_t = (1-γ)×[x_t + γ·x_{t-1} + γ²·x_{t-2} + ...]
```

- γ=0.9 时，当前数据权重 0.1，10 步前权重 ≈ 0.035，50 步前 ≈ 0
- 越近的数据权重越大，但更早的数据不会完全消失，而是指数衰减

**为什么叫"指数"**：权重是 γⁿ（指数衰减）。

### 有偏 vs 无偏估计

**无偏估计**：权重之和等于 1，估计的平均值等于真实值。

```
无偏平均 = (x_1 + x_2 + ... + x_t) / t
权重 = [1/t, 1/t, ..., 1/t]，之和 = 1
```

**有偏估计**：权重之和不等于 1，系统性地偏高或偏低。

EMA 的权重之和：
```
(1-γ) + γ(1-γ) + γ²(1-γ) + ... + γ^{t-1}(1-γ)
= (1-γ)(1 + γ + γ² + ... + γ^{t-1})
= (1-γ)(1-γ^t)/(1-γ)
= 1 - γ^t  < 1
```

所以 EMA 是**有偏估计**——权重之和小于 1，导致系统性偏低。

**偏置校正**：除以真实权重之和。
```
EMA校正 = EMA_t / (1 - γ^t)
```
权重之和变成 1，无偏了。

**为什么 EMA 要有偏**：
- 无偏版本需要存储所有历史数据，O(t) 内存
- EMA 用递归形式，O(1) 内存
- 前几步偏差大，但几步后校正项 ≈ 1，几乎不影响
- **有意为之**：用轻微有偏换取工程可行性

### Adam：Momentum + RMSprop

Adam = **一阶矩 EMA（Momentum 方向）** + **二阶矩 EMA（RMSprop 步长）**。

| 组件 | 来源 | EMA对象 | 作用 |
|---|---|---|---|
| **m**（一阶矩） | Momentum | 历史梯度 | 方向：过滤噪音，知道往哪走 |
| **v**（二阶矩） | RMSprop | 历史梯度² | 步长：自适应缩放，知道走多远 |

**一阶矩 m**（管方向）：
- m = β1 × m + (1-β1) × g
- 基于梯度本身，有正负，加算
- sign(m) = 整体运动方向

**二阶矩 v**（管步长）：
- v = β2 × v + (1-β2) × g²
- 基于梯度平方，只有大小，除算
- 控制 lr 的缩放程度

**参数更新**：
```
m_hat = m / (1 - β1^t)    # 偏置校正
v_hat = v / (1 - β2^t)    # 偏置校正
param = param - lr × m_hat / (√v_hat + ε)
```

**偏置校正的推导**：
```
EMA_t = γ·EMA_{t-1} + (1-γ)·x_t
展开 = (1-γ)[x_t + γ·x_{t-1} + γ²·x_{t-2} + ...]
权重之和 = 1 - γ^t

E[EMA_t] = (1-γ^t) × μ  ← 真实均值 μ 的 (1-γ^t) 倍
E[EMA_t / (1-γ^t)] = μ  ← 除以 (1-γ^t) 变成无偏
```

所以校正项 1/(1-β^t) 是从"EMA 是有偏估计"这个事实推导出来的，不是拍脑袋。

### 为什么 Adam 最流行

**Momentum 的缺陷：只管方向，不管步长**
```
m = γ·m + lr·g
```
- 方向靠历史累加，有过滤
- 但步长 = lr × g，lr 设大振荡，设小收敛慢

**RMSprop 的缺陷：只管步长，不管方向**
```
Δw = lr × g / √v
```
- 步长自适应
- 但方向完全跟着当前梯度 g走，没有过滤噪音

**Adam 两者结合**：
```
m = β1·m + (1-β1)·g          # 方向：历史梯度EMA
v = β2·v + (1-β2)·g²          # 步长：历史梯度²EMA
Δw = lr × m / √v              # 方向+步长同时自适应
```

**直观例子**（梯度 = 5, 4, -3）：

| | Momentum | Adam |
|---|---|---|
| 方向 | 累积后正负抵消 | 同左 |
| 步长 | lr × g，固定 | lr × m/√v，自动缩放 |
| 第3步更新 | lr × 0.465 | lr × 约 2.1（更稳定）|

**为什么实践中最优**：
1. 方向+步长同时自适应，几乎不用调参
2. β1=0.9, β2=0.999 基本固定
3. 前几步偏置校正保证正常
4. 收敛快，适合快速实验

**理论缺点（实践中影响小）**：
- 泛化能力有时不如 SGD+Momentum
- 显存比 SGD 多存两份 state（m 和 v）

### Adam 代码

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, 
                          betas=(0.9, 0.999), eps=1e-8)
```

### Adam 适用场景

| 推荐 | 不推荐 |
|---|---|
| 深度网络（CNN/Transformer） | 简单线性模型 |
| 非稀疏数据 | 需要精确调参的场景 |
| 快速实验阶段 | 对泛化要求极高的任务 |

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

## 4-8 优化器状态保存

optimizer 有内部状态（如 Adam 的 momentum、方差估计），保存 checkpoint 时必须一起保存。

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

**注意**：不同 optimizer 状态结构不同，Adam 的 state_dict 不能加载到 SGD。

**set_to_none=True**（更高效的清零方式）:
```python
optimizer.zero_grad(set_to_none=True)
# 设为 None 而非 0，内存更省
```

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