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

---

### 为什么推荐 state_dict 而非整个模型

**保存整个模型（不推荐）**：

```python
torch.save(model, path)  # 把整个 model 对象序列化
```

- 依赖原始类定义，换个文件/类名就炸
- 包含整个 Python pickle 序列化环境，不干净
- 文件大，加载慢

**保存 state_dict（推荐）**：

```python
torch.save(model.state_dict(), path)  # 只保存参数
```

- 文件小，只保存数值
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

EMA（指数移动平均）维护一套独立于主模型的参数，训练时用 EMA 参数做验证通常效果更好。

```python
from torch.optim.swa_utils import AveragedModel, SWALR

ema_model = AveragedModel(model)

# 训练中更新 EMA
ema_model.update_parameters(model)

# 保存 EMA checkpoint
checkpoint = {
    'model': model.state_dict(),
    'ema': ema_model.state_dict(),   # EMA 参数单独保存
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, 'checkpoint.pt')

# 加载 EMA
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
ema_model.load_state_dict(checkpoint['ema'])
# 验证时用 ema_model，训练时用 model
```

---

## 6-3 部分加载（strict=False / key 过滤）

### 迁移学习场景

预训练模型有 100 层，目标任务只需要前 90 层，后 10 层随机初始化。

```python
# 加载预训练权重
 pretrained_dict = torch.load('pretrained.pt')

# 只取前90层的权重
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith('layer')}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
```

### strict 参数

```python
model.load_state_dict(torch.load('model.pt'), strict=False)
```

- `strict=True`（默认）：key 必须完全匹配，否则抛异常
- `strict=False`：不匹配的 key 被忽略，跳过

### key 过滤示例

```python
# 场景：去掉最后的分类头，只保留特征提取部分
pretrained_dict = torch.load('pretrained.pt')
filtered_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
model.load_state_dict(filtered_dict, strict=False)
```

### 加载部分参数（参数形状不匹配时）

```python
# 如果 key 存在但形状不一致，strict=False 会跳过该层（不报错）
# 通常在迁移学习微调时，新任务的类别数和预训练不同，最后一层 shape 不匹配
# → 需要手动初始化新分类层的权重
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

### 常见设备组合

```python
# GPU训练 → CPU加载（用于部署/推理）
state = torch.load('model.pt', map_location='cpu')

# CPU训练 → GPU加载
state = torch.load('model.pt', map_location='cuda:0')

# GPU1 保存 → GPU0 加载
torch.save(model.state_dict(), 'model.pt')  # 在 GPU1 上保存
state = torch.load('model.pt', map_location='cuda:0')  # 在 GPU0 上加载
```

### 加载后 model.eval() / model.train() 的重要性

```python
# 保存时
model.eval()  # 保存前设为 eval，保证 BatchNorm / Dropout 状态正确
torch.save(model.state_dict(), 'model.pt')

# 加载后
model.eval()  # 推理模式：BN 用保存的 running_mean/var，Dropout 关闭
# 或
model.train()  # 继续训练：BN 用 batch 统计，Dropout 开启
```

**如果加载后忘记设 eval()**：
- BatchNorm：会用 batch 的统计量而不是保存的 running 统计量 → 推理结果异常
- Dropout：随机关闭/开启不固定 → 结果不稳定

### 单卡保存 / 多卡加载

```python
# 保存时在单卡上（去掉 DDP 包装）
torch.save(model.module.state_dict(), 'model.pt')  # model.module 拿到原始模型

# 加载
model = TheModel()
model.load_state_dict(torch.load('model.pt'))
```

---

## 6-5 大模型与安全

### 分片保存（Sharded Checkpoint）

70B 参数的模型，单文件保存 `.pt` 可能超过 100GB，磁盘写入慢，加载也慢。

**PyTorch 原生分片**：

```python
torch.save(model.state_dict(), 'model.pt', _use_new_zipfile_serialization=True)
# 用 zipfile 格式，速度更快，文件更小
```

**HuggingFace 分片保存**：

```python
# transformers 库的 save_pretrained 默认分片
model.save_pretrained('./model', max_shard_size='5GB')
# 生成：
# model.safetensors
# model-00001-of-00002.safetensors
# model-00002-of-00002.safetensors
# config.json
```

**分片加载**：

```python
from safetensors import safe_open

state_dict = {}
for shard_file in ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']:
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
```

### torch.load 的 weights_only 参数

```python
# 旧版本（默认行为）
state = torch.load('model.pt')  # 可能执行任意 Python 代码（pickle 反序列化风险）

# 新版本推荐
state = torch.load('model.pt', weights_only=True)  # 只加载 tensor，禁止执行代码
```

**为什么有安全问题**：pickle 可以序列化任意 Python 对象，反序列化时可能执行恶意代码。

**weights_only=True** 限制：只允许加载 tensor、scalar、字典、列表等安全类型，禁止反序列化自定义类。

### weights_only 的限制

```python
# 如果 checkpoint 里保存了自定义对象（不推荐），weights_only=True 会炸
torch.load('model.pt', weights_only=True)
# RuntimeError: unsupported PyTorch type: __main__.MyCustomClass

# 解决：不要在 checkpoint 里保存自定义对象，或者 weights_only=False（但有安全风险）
```

---

## 面试汇总

### Q1: 模型保存推荐用哪种方式？

推荐 `torch.save(model.state_dict(), path)`，而不是 `torch.save(model, path)`。state_dict 只保存参数，轻量且不依赖类定义。

### Q2: 断点续训需要保存哪些内容？

model state_dict、optimizer state_dict、epoch、scheduler、scaler（混合精度）、best_loss。核心是模型参数和优化器状态。

### Q3: strict=False 和 key 过滤的区别？

strict=False 跳过不匹配的 key（形状不一致或名称不一致）。key 过滤是主动剔除不需要的层（如分类头），两者可结合使用。

### Q4: 为什么加载后要 model.eval()？

BatchNorm 和 Dropout 在 train/eval 模式下行为不同。eval 模式用 running_mean/var 和固定的 dropout 比例。如果加载后忘记设 eval()，BatchNorm 会用 batch 统计量而非保存的统计量，导致推理结果异常。

### Q5: map_location 是什么原理？

`torch.load` 默认在保存的设备加载。map_location 重新指定加载目标设备（cpu、cuda:0 等）。底层是把 tensor 从原设备 copy 到新设备。

### Q6: weights_only=True 解决什么问题？

防止恶意 checkpoint 通过 pickle 反序列化执行任意代码。只允许加载 tensor 等安全类型，避免自定义类带来的安全风险。

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
