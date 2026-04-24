# 五、DataLoader — 数据加载

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| Dataset | 自定义数据抽象，必须实现 __getitem__ + __len__ | 如何自己实现 |
| DataLoader | batch / shuffle / num_workers | 各参数含义 |
| Sampler | Sequential / Random / WeightedRandom | 采样策略是什么 |
| collate_fn | 自定义 batch 拼接 | 什么时候需要重写 |
| pin_memory | 加速 GPU 传输 | 什么原理 |
| drop_last | 丢弃不完整 batch | |
| prefetch_factor + persistent_workers | 多进程预取与保活 | |
| torchvision | 图像领域数据集（MNIST / CIFAR / ImageNet） | |

**⚠️ 面试必答题：**
- DataLoader 的 shuffle 是在哪个层面做的？
- num_workers 设置过大的副作用是什么？
- Sampler 和 shuffle 有什么区别？
- collate_fn 什么时候需要自定义？

---

## 5-1 Dataset

### 核心接口

Dataset 必须实现两个方法：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

- `__len__`：返回数据集大小
- `__getitem__`：根据索引返回单个数据

PyTorch 会自动调用这两个方法，DataLoader 不需要知道具体实现。

### TensorDataset（快捷封装）

如果数据已经是张量，不需要自定义：

```python
from torch.utils.data import TensorDataset

X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
```

### Map-style vs Iterable-style

- **Map-style**（常用）：实现 `__getitem__`，支持索引访问
- **Iterable-style**：实现 `__iter__`，支持流式遍历（如从数据库/网络读取数据）

---

## 5-2 DataLoader 核心参数

### 基本用法

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,        # 每个 epoch 是否打乱
    num_workers=4,      # 多少个子进程加载数据
    pin_memory=True,    # 是否锁页内存
    drop_last=False,    # 是否丢弃不完整 batch
    collate_fn=None,    # 自定义 batch 拼接
)
```

### 逐个参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| batch_size | int | 1 | 每个 batch 多少样本 |
| shuffle | bool | False | 每个 epoch 是否打乱（仅当 sampler=None 时生效） |
| sampler | Sampler | None | 采样策略（与 shuffle 互斥） |
| num_workers | int | 0 | 子进程数，0=主进程加载 |
| pin_memory | bool | False | 锁页内存，加速 GPU 传输 |
| drop_last | bool | False | 丢弃最后一个不完整 batch |
| collate_fn | callable | None | 自定义 batch 拼接 |
| prefetch_factor | int | 2 | 每个 worker 预取多少 batch |
| persistent_workers | bool | False | epoch 间保持 worker 不退出 |

### shuffle 的本质

shuffle=True 时，DataLoader 内部使用 `RandomSampler`，本质是在 **样本层面** 打乱索引顺序，而不是打乱 batch。

```
原始数据：[0, 1, 2, 3, 4, 5, 6, 7]
打乱索引：[3, 7, 1, 5, 0, 6, 2, 4]
按 batch 取：[[3,7,1,5], [0,6,2,4]]
```

shuffle 是在 **Sampler 层面** 做的，不是 DataLoader 本身。

---

## 5-3 Sampler（采样策略）

Sampler 决定**按什么顺序遍历数据集**。

### 内置 Sampler

| Sampler | 说明 | 场景 |
|---------|------|------|
| SequentialSampler | 按顺序遍历 | 验证/测试集 |
| RandomSampler | 随机遍历 | 训练集（配合 shuffle=True）|
| WeightedRandomSampler | 按权重采样 | 不平衡数据集 |
| DistributedSampler | 分布式训练分片 | 多卡训练 |

### RandomSampler

```python
from torch.utils.data import RandomSampler

sampler = RandomSampler(dataset, replacement=False, num_samples=1000)
# replacement=False: 每个样本只能被选一次
# num_samples: 采样多少个（可小于数据集大小）
```

### WeightedRandomSampler（不平衡数据集）

给每个类别设置权重，让模型看到更多少数类：

```python
labels = [0, 0, 0, 0, 1, 1]  # 0多，1少

# 计算每个样本的权重
class_counts = [4, 2]  # 类别0有4个，类别1有2个
weights = [1.0/class_counts[y] for y in labels]
# weights = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
# 类别1的样本权重更高，被采样概率更大

sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
```

### Sampler vs shuffle

| | Sampler | shuffle |
|---|---|---|
| 作用层面 | 样本遍历顺序 | DataLoader 参数 |
| 灵活性 | 高（可自定义权重/分布）| 低（只有 True/False）|
| 互斥 | 和 shuffle 互斥 | 和 Sampler 互斥 |

**shuffle=True 本质**：DataLoader 内部自动创建 RandomSampler。

---

## 5-4 collate_fn（自定义 batch 拼接）

### 默认行为

DataLoader 把 `dataset[i]` 的返回值打包成一个 batch：

```python
# dataset[i] 返回 (image, label)
batch = [dataset[0], dataset[1], ..., dataset[31]]
# 等价于：zip([imgs], [labels]) 后 stack
```

如果所有数据形状相同（如同尺寸图片），默认行为没问题。

### 需要自定义的场景

**场景1：变长序列（如 NLP 的句子，长度不同）**

```python
def collate_fn(batch):
    # batch = [(seq1, label1), (seq2, label2), ...]
    texts = [item[0] for item in batch]    # 变长列表
    labels = [item[1] for item in batch]
    
    # Padding：把不同长度的序列 padding 到一样长
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return padded_texts, torch.tensor(labels)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

**场景2：嵌套结构（如检测任务一个样本有多个框）**

```python
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]  # 每个样本框数不同，不能 stack
    return images, targets
```

### 默认 collate_fn 能处理的情况

```python
# 同一 batch 内形状必须一致
TensorDataset → (batch of tensors)  # 自动 stack
numpy array → (batch of arrays)    # 自动转 tensor 后 stack
```

---

## 5-5 pin_memory（加速 GPU 传输）

### 原理

| | 普通内存 | pin_memory |
|---|---|---|
| 位置 | 可换出到磁盘 | 锁页内存，不会换出 |
| CPU→GPU 传输 | 先拷贝到临时锁页内存，再传 GPU | 直接传，无需中间拷贝 |
| 速度 | 慢 | 快（尤其大 batch）|

```python
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)
```

### 什么时候用

- 训练在 GPU 上时开启
- 单 CPU 训练（不用 GPU）时不需要
- 结合 `torch.cuda.FloatTensor` 使用效果最佳

### num_workers 的副作用

```python
dataloader = DataLoader(dataset, num_workers=8)
```

- num_workers 越大，数据加载越快（并行）
- 但内存占用越高（每个 worker 独立拷贝数据）
- num_workers 过多可能导致：
  - 内存爆炸
  - 进程创建开销大于并行收益
  - 适合的值：2~8，通常 4

---

## 5-6 drop_last / prefetch_factor / persistent_workers

### drop_last（丢弃不完整 batch）

```python
dataloader = DataLoader(dataset, batch_size=32, drop_last=True)
```

最后一个 batch 样本数不足 32，会被丢弃。

**为什么需要**：
- batch 大小不同会导致 BN 层统计不稳定
- 某些模型对 batch 形状有要求

### prefetch_factor（预取队列长度）

```python
dataloader = DataLoader(dataset, num_workers=4, prefetch_factor=2)
```

每个 worker 预取 2 个 batch 到队列。

总预取数 = num_workers × prefetch_factor

**什么时候调大**：
- CPU 处理快，GPU 等待时 → 增大预取
- 网络/存储 IO 慢 → 增大预取

### persistent_workers（保持 worker 不退出）

```python
dataloader = DataLoader(dataset, num_workers=4, persistent_workers=True)
```

epoch 结束后，worker 不退出，保持预加载数据的状态。

**适合场景**：多 epoch 训练，减少 worker 重启开销。

---

## 5-7 torchvision 内置数据集

### 常用数据集

```python
from torchvision import datasets, transforms

# MNIST（手写数字）
datasets.MNIST(root='./data', train=True, download=True,
               transform=transforms.ToTensor())

# CIFAR-10
datasets.CIFAR10(root='./data', train=True, download=True,
                 transform=transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                 ]))

# ImageNet（需要申请）
datasets.ImageNet(root='./data', split='train')
```

### transform 流程

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # 随机裁剪
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ToTensor(),                    # 转为 [0,1] 张量
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])
```

---

## 面试汇总

### Q1: DataLoader 的 shuffle 是在哪个层面做的？

在 **Sampler 层面**。shuffle=True 时，DataLoader 内部使用 RandomSampler，在样本层面打乱索引顺序。

### Q2: num_workers 设置过大的副作用？

- 内存占用高（每个 worker 独立拷贝数据）
- 进程创建开销大
- 适合 2~8

### Q3: Sampler 和 shuffle 有什么区别？

shuffle 是 DataLoader 参数，本质是在创建 RandomSampler。Sampler 决定遍历顺序，shuffle 决定是否打乱。

### Q4: collate_fn 什么时候需要自定义？

- 变长序列（NLP）需要 padding
- 嵌套结构（检测任务）不能 stack
- 非标准数据格式

---

## 代码练习

```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 变长序列的 collate_fn
def collate_fn(batch):
    texts, labels = zip(*batch)
    # Padding 到相同长度
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.tensor(labels)

# 不平衡数据集的 WeightedRandomSampler
labels = torch.randint(0, 2, (1000,)).tolist()
class_counts = [sum(1 for l in labels if l == 0), sum(1 for l in labels if l == 1)]
weights = [1.0 / class_counts[l] for l in labels]
sampler = WeightedRandomSampler(weights, len(weights))

dataloader = DataLoader(
    TextDataset(texts, labels),
    batch_size=32,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

for batch_texts, batch_labels in dataloader:
    print(batch_texts.shape, batch_labels.shape)
```

---

## 参考资料

- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html
- Sampler: https://pytorch.org/docs/stable/data.html#data-samplers
