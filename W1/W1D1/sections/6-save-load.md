# 六、模型保存与加载

---

## 6-1 state_dict 方式

### 模型结构 vs 模型参数

```
model = MyModel()     # 模型结构：类定义 + forward 逻辑
model.weight          # 模型参数：具体数值（tensor）
model.bias
model.running_mean    # BatchNorm 的缓存（也是 state_dict 的一部分）
```

**`model.state_dict()` 返回什么？**

```python
model.state_dict()
# OrderedDict([
#     ('layer1.weight', tensor([...])),
#     ('layer1.bias', tensor([...])),
#     ('layer2.weight', tensor([...])),
#     ...
# ])
```

只包含：**参数值 + Buffer（running_mean/var 等缓存）**，不包含类定义。

**state_dict 包含的是什么？**

```python
model.state_dict().keys()
# 包含两类：
# - layer.weight, layer.bias     ← nn.Parameter（可训练参数）
# - bn.running_mean/var         ← Buffer（不可训练，但会保存和加载）
```

> 注意：`model.parameters()` 和 `model.state_dict()` 不一样。前者只包含可训练参数，后者包含参数 + Buffer。

**为什么要保存 Buffer？**
- `running_mean/var` 是 BatchNorm 在训练时积累的统计量，推理时必须用
- 加载时如果不恢复这些值，BatchNorm 的行为会错乱

---

### torch.load vs load_state_dict（容易混淆）

这是**两个完全不同的操作**，经常被搞混：

```python
# torch.load()：从磁盘文件反序列化为 Python 对象
state = torch.load('checkpoint.pt', weights_only=True)
# state 的类型：dict（包含 model_state_dict、optimizer_state_dict、epoch 等）

# load_state_dict()：把字典里的 tensor 塞进模型参数
model.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])
```

| 方法 | 作用 | 层级 |
|------|------|------|
| `torch.load(path)` | 读文件，返回 Python 对象 | 文件 → 内存 |
| `load_state_dict(dict)` | 把 dict 里的值赋给模型参数 | dict → 模型 |

**典型流程：**
```python
# 1. 加载 checkpoint 文件
checkpoint = torch.load('checkpoint.pt', weights_only=True)

# 2. 把各项塞回对应的地方
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
```

---

### weights_only 参数（安全加载）

```python
# 默认（旧行为）
state = torch.load('checkpoint.pt')
# 可能执行任意 Python 代码（pickle 反序列化风险）

# 推荐写法
state = torch.load('checkpoint.pt', weights_only=True)
# 只允许加载 tensor、数值、字符串、字典、列表等安全类型
```

**weights_only=True 允许什么？**
- tensor ✅
- int / float / str ✅
- dict / list / tuple ✅
- 自定义 Python 类实例 ❌（拒绝）
- lambda 函数 ❌（拒绝）

> Buffer（running_mean 等）是 tensor，不受此限制，正常加载。

**为什么需要这个参数？**
pickle 可以序列化任意 Python 对象，反序列化时可能执行恶意代码。`weights_only=True` 强制 checkpoint 只包含数值，规避这个风险。

> ⚠️ 如果 checkpoint 里保存了自定义对象，用 `weights_only=True` 会报错。**建议永远不要在 checkpoint 里保存自定义对象。**

---

### Config 方式保存模型结构（工业标准）

模型结构和权重**分开存**，这是工业界的主流做法。

**以 HuggingFace 为例：**
```
model/
├── config.json      # 模型结构：层数、隐藏维度、注意力头数...
└── model.safetensors # 模型权重（state_dict）
```

**config.json 例子：**
```json
{
  "architectures": ["BertForMaskedLM"],
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "vocab_size": 30522
}
```

**为什么不用 save 整个模型？**

| | save 整个模型 | save state_dict + config |
|---|---|---|
| 类定义 | ❌ 序列化在文件里，强依赖 | ✅ config.json 纯数据，任何系统都能读 |
| 部署兼容性 | ❌ Python pickle 环境可能不兼容 | ✅ 权重文件独立，任意推理引擎都能用 |
| 工具链支持 | ❌ 很多转换工具不支持 | ✅ ONNX/TensorRT/Triton 都基于 state_dict |

**自己实现一个 Config 类：**

```python
import json
import torch
import torch.nn as nn

class ModelConfig:
    def __init__(self, in_features=10, out_features=2, hidden=[64, 32]):
        self.in_features = in_features
        self.out_features = out_features
        self.hidden = hidden

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        prev = config.in_features
        for h in config.hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, config.out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 保存
config = ModelConfig(in_features=10, out_features=2, hidden=[64, 32])
config.save('config.json')
torch.save(model.state_dict(), 'model.pt')

# 加载（部署时：不需要源码，只需要 config.json + model.pt）
loaded_config = ModelConfig.load('config.json')
model = MLP(loaded_config)
model.load_state_dict(torch.load('model.pt', weights_only=True))
```

**工业流程：**
```
训练侧：save state_dict + config.json
         ↓
部署侧：读 config.json → 建模型结构 → load state_dict → 推理
```

---

### 为什么推荐 state_dict 而非整个模型

**保存整个模型（不推荐）**：

```python
torch.save(model, path)  # 把整个 model 对象序列化
```

- 依赖原始类定义，换个文件/类名就炸
- 包含整个 Python pickle 序列化环境，不干净
- 文件体积和 state_dict 几乎一样（差异只有几 MB，主要来自类定义）

**保存 state_dict（推荐）**：

```python
torch.save(model.state_dict(), path)  # 只保存参数
```

- 权重文件独立，推理引擎（ONNX/TensorRT/Triton）都能用
- 架构用 config.json 单独管理，部署更灵活
- 已经是工业标准，生态工具链都基于此
- 兼容性强，任何相同结构的模型都能加载
- 加载时需要自己先建模型，再塞参数

```python
# 保存
torch.save(model.state_dict(), 'model.pt')

# 加载（必须先有模型结构）
model = MyModel()                          # 先实例化结构
model.load_state_dict(torch.load('model.pt'))  # 再塞参数
```

---

### torch.save / torch.load 基础

```python
# 保存
torch.save(obj, path)          # obj 可以是 state_dict、整个 checkpoint、任意 Python 对象

# 加载
state = torch.load(path)       # 返回之前保存的对象
```

---

## 6-2 断点续训（Checkpoint）

### 为什么需要断点续训

训练中断（宕机、显存爆）后，从头重训代价大。Checkpoint 把训练状态全部存盘。

### checkpoint 包含的内容

```python
checkpoint = {
    'epoch': 5,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),     # 学习率调度器状态
    'scaler': amp_gradScaler.state_dict(),  # 混合精度状态
    'best_loss': 0.32,
}
torch.save(checkpoint, 'checkpoint_epoch5.pt')
```

| 字段 | 必须 | 说明 |
|------|------|------|
| model | ✅ | 模型参数 |
| optimizer | ✅ | 动量、学习率等状态 |
| epoch | 推荐 | 方便断点定位 |
| scheduler | 可选 | 学习率调度器状态 |
| scaler | 可选 | 混合精度训练状态 |
| best_loss | 可选 | 记录最优指标 |

### 完整恢复训练

```python
# 加载
checkpoint = torch.load('checkpoint_epoch5.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
scaler.load_state_dict(checkpoint['scaler'])

start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']

# 继续训练
for epoch in range(start_epoch, num_epochs):
    train(...)
    validate(...)
    scheduler.step()
    save_checkpoint(...)
```

### EMA 的保存与加载

**EMA（指数移动平均）** 维护一套独立于主模型的参数副本，训练时用 EMA 参数做验证通常效果更好。

**EMA 参数是怎么更新的？**

```python
ema_model = AveragedModel(model)  # 包装原始模型，得到一套参数副本

# 每次 optimizer 更新完原模型参数后，立即更新 EMA 副本
ema_model.update_parameters(model)

# 内部公式：
# EMA_w = decay * 旧EMA_w + (1 - decay) * 当前权重
# 默认 decay = 0.999
```

**为什么是"双重动量"？**

| 动量 | 作用对象 | 目的 |
|------|---------|------|
| Optimizer momentum | 梯度更新方向 | 加速收敛、减少振荡 |
| EMA decay | 参数值本身 | 平滑参数曲线、抗过拟合 |

optimizer momentum 让收敛更快，但收敛路径可能有振荡。EMA 在参数层面再做一次平滑，取"均值"而非"终点"，往往泛化更好。

**参数完全一一对应，大小相同：**

```python
print("原模型参数量:", sum(p.numel() for p in model.parameters()))
# 55

print("EMA 参数量:", sum(p.numel() for p in ema_model.parameters()))
# 55（完全一样，EMA 是独立的副本）

# 所以 checkpoint = model + ema = 约 2 倍体积
```

**完整训练示例（含 AMP 混合精度 + EMA + Checkpoint）：**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel

# 初始化
model = MyModel().cuda()
ema_model = AveragedModel(model)          # EMA 副本
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()                      # AMP 缩放器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
criterion = nn.CrossEntropyLoss()

def save_checkpoint(epoch, best_loss):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'ema': ema_model.state_dict(),        # EMA 参数单独保存
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),       # 混合精度状态
        'best_loss': best_loss,
    }
    torch.save(checkpoint, f'checkpoint_epoch{epoch}.pt')

def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    ema_model.load_state_dict(checkpoint['ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint['epoch'], checkpoint['best_loss']

# 训练循环
best_loss = float('inf')
start_epoch = 0

for epoch in range(start_epoch, 30):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        # 混合精度前向
        with autocast():
            output = model(batch['input'])
            loss = criterion(output, batch['target'])

        # 混合精度反向
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 更新 EMA 副本（在 optimizer 更新参数之后立即调用）
        ema_model.update_parameters(model)

    scheduler.step()

    # 验证（用 EMA 模型效果更好）
    model.eval()
    ema_model.eval()
    val_loss = validate(ema_model, val_loader)  # ← 用 EMA 模型验证

    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint(epoch, best_loss)

# 加载 EMA 恢复训练
epoch, best_loss = load_checkpoint('checkpoint_epoch5.pt')
print(f"从 epoch {epoch} 恢复训练，最优 loss: {best_loss}")
```

**EMA 加载后的用法：**

```python
# 验证/推理时用 EMA 模型
ema_model.eval()
with torch.no_grad():
    output = ema_model(input)

# 继续训练时用普通模型（EMA 只用于推理）
```

---

## 6-3 部分加载（strict=False / key 过滤）

### strict=False 的完整加载逻辑

`strict=False` 按 key 名称一一匹配，规则只有三条：

| 情况 | 行为 |
|------|------|
| 两边 key 名称一样 | ✅ 加载进去 |
| 新模型有、pretrain 没有 | 保持随机初始化（不更新） |
| pretrain 有、新模型没有 | 直接丢弃 |

```python
# pretrain: L1, L2, L3
# new model: L1, L2, L4, L3

# strict=False 加载后：
# L1 → ✅ 加载
# L2 → ✅ 加载
# L3 → ✅ 加载（名称相同）
# L4 → 保持随机初始化（pretrain 没有）
# (pretrain 的其他层 → 丢弃)
```

### 迁移学习场景

预训练模型有 100 层，目标任务只需要前 90 层，后 10 层随机初始化：

```python
# 加载预训练权重
pretrained_dict = torch.load('pretrained.pt', weights_only=True)

# 只取前90层的权重（按名称前缀过滤）
filtered_dict = {k: v for k, v in pretrained_dict.items()
                  if k in model.state_dict() and k.startswith('layer')}
model.load_state_dict(filtered_dict, strict=False)
```

### 形状不匹配时（key 存在但 shape 不同）

```python
# pretrained: layer4.weight shape = (64, 512)
# new model:  layer4.weight shape = (64, 256)  ← 输出维度变了

# strict=False 碰到这种情况：
# → 跳过 layer4.weight（形状不一致，不会强制加载）
# → layer4 保持随机初始化

model.load_state_dict(pretrained, strict=False)
```

> 注意：`strict=False` 只解决"有没有这个 key"的问题，不解决"key 有但 shape 不对"的问题。

### key 过滤示例

```python
# 场景：去掉最后的分类头，只保留特征提取部分
pretrained_dict = torch.load('pretrained.pt', weights_only=True)
filtered_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
model.load_state_dict(filtered_dict, strict=False)
```

### 手动映射不同名称的层

如果两个模型 key 名称不同但想手动对应，需要改 pretrained 的 key：

```python
pretrained = torch.load('pretrained.pt', weights_only=True)

# 把 pretrained 里的 L3 → L4（手动重命名）
pretrained_renamed = {}
for k, v in pretrained.items():
    if k.startswith('L3'):
        new_key = k.replace('L3', 'L4')
        pretrained_renamed[new_key] = v
    else:
        pretrained_renamed[k] = v

model.load_state_dict(pretrained_renamed, strict=False)
```

---

## 6-4 跨设备保存与加载

### map_location 原理

`torch.load` 默认在保存的设备上加载。如果 GPU 显存不够，CPU 加载后转 GPU。

```python
# 保存时在 GPU 上
torch.save(model.state_dict(), 'model.pt')

# 加载时在 CPU 上（显存不够时）
state_dict = torch.load('model.pt', map_location='cpu')
model.load_state_dict(state_dict)

# 指定具体设备
state_dict = torch.load('model.pt', map_location='cuda:1')  # 加载到第2张卡
```

**常见设备组合：**

```python
# GPU训练 → CPU加载（用于部署/推理）
state = torch.load('model.pt', map_location='cpu')

# CPU训练 → GPU加载
state = torch.load('model.pt', map_location='cuda:0')

# GPU1 保存 → GPU0 加载
torch.save(model.state_dict(), 'model.pt')  # 在 GPU1 上保存
state = torch.load('model.pt', map_location='cuda:0')  # 在 GPU0 上加载
```

### 保存时 train / eval 模式重要吗？

**训练/推理的断点续训：模式不重要。** 因为保存的是参数值，`model.train()` 和 `model.eval()` 改变的是**层的统计行为**，不改变参数本身的数值。

**真正重要的是：什么时候保存？**

```python
# ✅ 正确：在 epoch 结束时保存
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        train_step(batch)  # running_mean/var 持续更新，越来越稳定
    
    # epoch 结束后保存：running_mean/var 已积累大量 batch，稳定
    torch.save({'model': model.state_dict(), ...}, f'ckpt_epoch{epoch}.pt')

# ❌ 错误：只训了几个 batch 就保存
for i, batch in enumerate(dataloader):
    train_step(batch)
    if i == 10:  # 只训了10个batch就跑路
        torch.save(model.state_dict(), 'ckpt.pt')  # running_mean/var 只基于10个batch，严重不稳定
```

**结论：**
- 断点续训以 **Epoch** 为切分点，running_mean/var 质量由"训练了多少步"决定，和模式无关
- 断点续训时，保存时的模式不重要，加载后切回 `model.train()` 继续训练即可
- 中途保底备份（按 Step 保存）可以存，但只当"保险"，不用于推理

### 单卡保存 / 多卡加载 / DDP

**DDP 训练时，模型会被包装：**

```python
model = DistributedDataParallel(model)  # DDP 包装了一层

model.module  # ← 用 .module 访问被包装的原始模型
```

**多卡保存（去掉 DDP 包装）：**

```python
# DDP 训练后保存，必须用 .module 拿到原始模型
torch.save(model.module.state_dict(), 'model.pt')  # ✅ 正确：保存原始模型
torch.save(model.state_dict(), 'model.pt')          # ❌ 错误：存的是 DDP 包装层状态

# 单卡加载（任意设备）
model = MyModel()
model.load_state_dict(torch.load('model.pt', weights_only=True))
```

**单卡保存 → 多卡加载：**

```python
# 1. 单卡保存
torch.save(model.state_dict(), 'model.pt')

# 2. 多卡加载
model = MyModel()
model.load_state_dict(torch.load('model.pt', weights_only=True))  # 先加载权重
model = DistributedDataParallel(model, device_ids=[0, 1, 2, 3])   # 再包 DDP
# 然后就可以多卡训练了
```

> 注意：权重文件和设备数量无关，同一个文件可以在单卡或多卡上用。


---

## 6-5 大模型与安全

### 分片保存（Sharded Checkpoint）

70B 参数的模型，单文件 `.pt` 可能超过 100GB，写入慢，加载更慢。分片保存把大文件拆成多个小文件。

**HuggingFace 分片保存：**

```python
model.save_pretrained('./model', max_shard_size='5GB')
# 生成：
# model-00001-of-00003.safetensors  (~5GB)
# model-00002-of-00003.safetensors  (~5GB)
# model-00003-of-00003.safetensors  (~剩余)
# config.json
```

**分片加载（核心就是循环读取所有分片，合并成一个 dict）：**

```python
from safetensors import safe_open

state_dict = {}
for shard_file in sorted(glob.glob('model-*-of-*.safetensors')):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

# 合并后正常加载
model = MyModel()
model.load_state_dict(state_dict)
```

> 为什么用 safetensors 而不是 .pt？safetensors 是 HuggingFace 推出的安全格式，加载更快（无需反序列化，直接内存映射），且天然支持分片。

**PyTorch 原生分片（不常用）：**

```python
torch.save(model.state_dict(), 'model.pt',
           _use_new_zipfile_serialization=True)  # zipfile 格式，体积更小
```

### torch.load 的 weights_only 参数

```python
# 旧版本（默认行为）
state = torch.load('model.pt')  # 可能执行任意 Python 代码（pickle 反序列化风险）

# 新版本推荐
state = torch.load('model.pt', weights_only=True)  # 只加载 tensor，禁止执行代码
```

**weights_only=True 允许什么？**

| 类型 | 允许 |
|------|------|
| tensor、数值、字符串 | ✅ |
| dict / list / tuple | ✅ |
| 自定义 Python 类实例 | ❌ |
| lambda 函数 | ❌ |

> Buffer（running_mean 等）是 tensor，不受限制。

**为什么需要这个参数？**
pickle 可以序列化任意 Python 对象，反序列化时可能执行恶意代码。`weights_only=True` 强制 checkpoint 只包含数值。

**weights_only=True 报错怎么办？**
```python
# 原因：checkpoint 里保存了自定义对象（不推荐）
torch.load('model.pt', weights_only=True)
# RuntimeError: unsupported PyTorch type: __main__.MyCustomClass

# 解决：永远不要在 checkpoint 里保存自定义对象
checkpoint = {
    'model': model.state_dict(),    # ✅ 只有数值
    'optimizer': optimizer.state_dict(),  # ✅ 只有数值
    'epoch': 10,                   # ✅ int
    'config': {'lr': 1e-3},       # ✅ dict
    # 'preprocessor': MyPreprocessor()  ❌ 不要放自定义对象
}
```

---

## 面试汇总

### Q1: 模型保存推荐用哪种方式？为什么？

推荐 `torch.save(model.state_dict(), path)`，而不是 `torch.save(model, path)`。

state_dict 优势：
1. 权重文件独立，架构用 config.json 管理，部署到任意推理引擎（ONNX/TensorRT）都能用
2. 不依赖 Python pickle 环境，PyTorch 版本升级后仍能正常加载
3. 是 HuggingFace、Tim m 等主流生态的标准格式，工具链都基于此

### Q2: 断点续训需要保存哪些内容？

| 内容 | 必须 | 说明 |
|------|------|------|
| model state_dict | ✅ | 模型参数 |
| optimizer state_dict | ✅ | 动量、学习率状态 |
| epoch | 推荐 | 方便定位断点 |
| scheduler state_dict | 可选 | 学习率调度器状态 |
| scaler state_dict | 可选 | 混合精度状态 |
| EMA state_dict | 可选 | EMA 参数副本 |

核心是模型参数和优化器状态，其余都是辅助恢复。

### Q3: strict=False 和 key 过滤的区别？

| 方式 | 作用 |
|------|------|
| `strict=False` | 不匹配的 key 跳过（名称不一致、形状不一致都跳过） |
| key 过滤 | 主动筛选要加载的 key（如去掉分类头） |
| 两者结合 | 先过滤再加载 |

### Q4: 为什么加载后要 model.eval()？

BatchNorm 和 Dropout 在 train/eval 模式下行为不同：

- BatchNorm：eval 模式用 **running_mean/var** 归一化；train 模式用 **batch 统计量**
- Dropout：eval 模式全部通过；train 模式随机置零

如果不设 eval()，BatchNorm 会用当前 batch 的统计量（可能只有1个样本），推理结果完全不稳定。

### Q5: map_location 是什么原理？

`torch.load` 默认在保存的设备加载。`map_location` 重新指定加载目标设备，底层是把 tensor 从原设备 copy 到新设备。

```python
# GPU1 保存 → GPU0 加载
torch.save(model.state_dict(), 'model.pt')  # GPU1
state = torch.load('model.pt', map_location='cuda:0')  # 加载到 GPU0
```

### Q6: weights_only=True 解决什么问题？

防止恶意 checkpoint 通过 pickle 反序列化执行任意代码。只允许加载 tensor 等安全类型。

### Q7: EMA 是什么？为什么用它？

EMA（指数移动平均）维护一套独立于主模型的参数副本，用 decay 公式平滑更新：
```
EMA_w = 0.999 * 旧EMA + 0.001 * 当前权重
```

效果：对训练过程的参数轨迹做二次平滑，取"均值"而非"终点"，泛化更好。验证/推理时用 EMA 参数。

### Q8: DDP 训练时保存为什么要用 .module？

`DistributedDataParallel` 包装了原始模型。直接 `model.state_dict()` 会保存 DDP 包装层状态，加载时会出错。用 `model.module.state_dict()` 才能拿到原始模型参数。

---

## 代码练习

```python
import torch
import torch.nn as nn

# ===== 1. 保存和加载 state_dict =====
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 保存（推荐）
torch.save(model.state_dict(), 'model.pt')

# 加载
model2 = SimpleModel()
model2.load_state_dict(torch.load('model.pt', weights_only=True))

# ===== 2. 断点续训 =====
checkpoint = {
    'epoch': 5,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_loss': 0.32,
}
torch.save(checkpoint, 'checkpoint.pt')

# 加载恢复
ckpt = torch.load('checkpoint.pt', weights_only=True)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1

# ===== 3. 部分加载（迁移学习）=====
pretrained_dict = torch.load('pretrained.pt', weights_only=True)
model_dict = model.state_dict()

# 只加载匹配的层
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)

# ===== 4. 跨设备加载 =====
# GPU保存 → CPU加载
state = torch.load('model.pt', map_location='cpu')
model.load_state_dict(state)

# ===== 5. 单卡保存多卡加载 =====
# DDP训练时保存要去掉 .module 包装
torch.save(model.module.state_dict(), 'model.pt')
```
