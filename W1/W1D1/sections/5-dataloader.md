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

- **Map-style**（最常用）：实现 `__getitem__`，支持随机索引访问。数据一次性加载到内存，适合能装进内存的数据。
- **Iterable-style**：实现 `__iter__`，支持流式遍历（如从数据库/网络读取数据）。数据按需读取，不撑爆内存。

### Iterable-style 详解（流式数据）

**为什么需要流式？**  
当数据量极大（如1000万条）无法一次装进内存时，需要"随用随取、用多少取多少"。

**yield 的本质**：不是"暂停"，是"冻结"。函数执行到 yield 时，把值吐出去后卡住，下一次 `next()` 从卡住处继续。

```python
def gen():
    print("A")
    yield 1          # 执行到这里，冻结，吐出 1
    print("B")
    yield 2          # 从这里继续，冻结，吐出 2
    print("C")       # 函数结束，抛 StopIteration
```

```python
g = gen()
next(g)  # A + 返回 1
next(g)  # B + 返回 2
next(g)  # C + 抛 StopIteration
```

**流式数据的自动续接机制**（核心！）：cursor 耗尽后通过异常捕获重新查下一批：

```python
def __iter__(self):
    offset = 0
    FETCH_SIZE = 100000   # 每批从数据库读多少
    TOTAL_LIMIT = 10000000  # 总数据量

    while True:                              # 永真循环，不退出
        cursor = db.execute(
            f"SELECT * FROM logs LIMIT {FETCH_SIZE} OFFSET {offset}"
        )
        try:
            while True:
                row = next(cursor)           # 逐行取
                yield row                    # 有数据，正常吐出
        except StopIteration:                # cursor 耗尽了，抛异常
            offset += FETCH_SIZE             # offset 跳到下一批
            if offset >= TOTAL_LIMIT:
                return                       # 取够了，epoch 结束
            # 没取够，回到外层 while True，重新建 cursor 继续吐
```

**生命周期**：yield 出一条 → DataLoader 打包成 batch → 用了 → 再 next() 取下一条 → cursor 耗尽抛异常 → 捕获后重新查下一批 → 继续 yield

**为什么叫"流式"**：数据不是一次性全量加载到内存，而是边流边用。像水流一样，随用随取，不囤积。

**IterableDataset 适合场景**：日志分析、视频流、数据库大表查询结果。Map-style 适合数据能一次性装进内存的场景。

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

### 完整参数对照表（默认值 / 推荐值 / 利弊）

| 参数 | 默认值 | 默认原因 | 推荐值 | 优点 | 缺点 / 注意 |
|------|--------|----------|--------|------|-------------|
| batch_size | 1 | 最小单元，不影响功能 | 16/32/64（视GPU显存） | 大batch训练稳定，梯度估计准 | 大batch显存压力大，小batch梯度噪声大 |
| shuffle | False | 不强制打乱，用户自己决定 | 训练：True，验证：False | True时每个epoch数据顺序不同，避免过拟合 | 验证集不需要，打乱反而影响可复现性 |
| sampler | None | 配合shuffle使用，不冲突 | 自定义WeightedRandomSampler处理不平衡 | 可控采样策略 | 与shuffle互斥，设置sampler时shuffle必须False |
| num_workers | 0 | 避免多进程开销，简化调试 | 有预处理时：2~8；纯内存索引：0 | 并行加速IO/预处理，GPU不再空等 | 内存占用高（每worker复制一份数据），进程创建开销，macOS有limitation |
| pin_memory | False | 单CPU训练不需要，节省锁页内存 | GPU训练时：True | DMA直传省去中间拷贝，CPU→GPU加速明显 | 锁页内存占用系统可用内存，过多可能导致OOM |
| drop_last | False | 不丢失数据，最大化利用 | 训练时常用True，验证可用False | batch大小一致，BatchNorm统计稳定 | 少量数据被丢弃（最后一个不完整batch） |
| collate_fn | None | 大多数场景直接stack | 变长序列/嵌套结构：自定义 | 灵活处理各种数据格式 | 不需要时别乱设，会覆盖默认stack行为 |
| prefetch_factor | 2 | 平衡预取与内存 | IO慢/预处理重：增大到4~8；高速存储：默认或1 | 队列始终有货，GPU不等待 | 增大则内存占用上升（num_workers × prefetch_factor × batch_size）|
| persistent_workers | False | 避免占用额外内存 | 多epoch训练：True；单epoch或调试：False | 避免每epoch重新创建worker进程，开销小 | epoch间worker一直存活，内存占用持续 |

> **注意**：prefetch_factor 仅在 num_workers > 0 时生效。persistent_workers 仅在 num_workers > 0 时生效。

### shuffle 的本质

shuffle=True 时，DataLoader 内部使用 `RandomSampler`，本质是在 **样本层面** 打乱索引顺序，而不是打乱 batch。

```
原始数据：[0, 1, 2, 3, 4, 5, 6, 7]
打乱索引：[3, 7, 1, 5, 0, 6, 2, 4]
按 batch 取：[[3,7,1,5], [0,6,2,4]]
```

shuffle 是在 **Sampler 层面** 做的，不是 DataLoader 本身。

**shuffle 与 IterableDataset 不兼容**：shuffle=True 会直接抛 `TypeError: shuffle option is not supported with IterableDataset`。因为流式数据没有 `__len__`，无法构建随机索引。流式数据的"打乱"需在数据库查询层面用 `ORDER BY RANDOM()` 实现。

### num_workers 机制详解

**工作流程（Map-style 数据）**：

```
主进程构建 sampler，fork 出 N 个 worker
每个 worker 复制 sampler 状态，按不同步长取索引：
  Worker1: indices[0::N]
  Worker2: indices[1::N]
  Worker3: indices[2::N]
  ...

各 worker 调用 dataset[idx] 取单样本 → 放进共享队列
主进程：从队列取样本，凑满 batch_size → 打包 → 传给 GPU
```

**为什么 num_workers=0 也够用（但有前提）**：如果 `__getitem__` 只是纯内存索引（如 TensorDataset），几乎没有开销，单进程足够。多进程反而增加进程通信开销。

**num_workers 有意义的前提**：`__getitem__` 里有 IO 或 CPU 密集型操作（读文件、数据增强、tokenization等）。

**IterableDataset + num_workers > 0 的重复问题**：每个 worker 都会重新执行 `__iter__()`，导致数据重复。必须通过 `worker_init_fn` 为每个 worker 分配不同的数据分片。

### pin_memory 机制详解

**普通内存（pinned=False）的传输路径**：

```
数据在普通内存页 → 可能被换出到磁盘
                    ↓
              先换回内存（若被换出）
                    ↓
              拷贝到临时锁页缓冲区
                    ↓
              从缓冲区传到 GPU
```

**锁页内存（pin_memory=True）的传输路径**：

```
锁页内存 → 永远不被换出 → DMA 直传 GPU（无需 CPU 介入）
```

**为什么能加速**：省去了"换页回内存 + 中间缓冲区拷贝"两步。锁页内存通过 DMA（直接内存访问）绕过 CPU 直接传到 GPU。

**锁页大小不需要手动设置**：PyTorch 按需分配，用到时临时锁、传完即释放（或被缓存复用）。Linux 锁页内存总量有限（默认≈RAM的一半），一般训练场景不会触达。

### prefetch_factor 与 persistent_workers

**prefetch_factor 控制的是什么**：不是 worker 每次取多少样本，而是队列里最多囤多少个 batch 的样本。

```
num_workers=4, prefetch_factor=2
→ 每个 worker 预取 2 个 batch 的样本
→ 队列里最多有 4×2=8 个 batch 在等待主进程
→ GPU 始终有货可用，主进程不等待
```

**persistent_workers 的作用**：epoch 结束后 worker 不退出，保持预加载状态。

```python
# 无 persistent_workers（默认）
for epoch in range(10):
    # 每个 epoch 重新创建 4 个 worker → 进程创建开销 × 10

# 有 persistent_workers=True
# worker 创建一次，10 个 epoch 复用
```

**典型训练配置推荐**：

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,              # 训练用
    num_workers=4,             # 有预处理时
    pin_memory=True,           # GPU 训练时
    drop_last=True,            # 训练时常用
    persistent_workers=True,    # 多 epoch 训练时
)
```

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

### 代码中使用 Sampler

```python
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

# 方式1：直接设 shuffle=True（最常用）
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# 内部自动创建 RandomSampler，不需要碰 Sampler

# 方式2：手动指定 Sampler（不平衡采样 / 分布式训练时才需要）
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
dataloader = DataLoader(dataset, sampler=sampler)  # 不再设 shuffle
```

注意：`sampler` 和 `shuffle` 互斥，设了 `sampler` 就不能设 `shuffle=True`。

### DistributedSampler 详解

**解决的问题**：多卡训练时，每张卡只跑 1/N 的数据，4张卡合起来覆盖全量数据。

**跳步分区**：

```python
# 10000条数据，4张卡，rank=0 的卡
indices = list(range(len(dataset)))          # [0, 1, 2, ..., 9999]
indices = indices[rank::num_replicas]       # rank=0 → [0, 4, 8, 12, ...]
```

| 卡 | rank | 分到的索引 |
|----|------|-----------|
| 卡0 | 0 | 0, 4, 8, 12, ... |
| 卡1 | 1 | 1, 5, 9, 13, ... |
| 卡2 | 2 | 2, 6, 10, 14, ... |
| 卡3 | 3 | 3, 7, 11, 15, ... |

**epoch 间随机打乱（必须调用 set_epoch）**：

```python
sampler = DistributedSampler(dataset, num_replicas=4, rank=0)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 驱动随机种子变化，改变打乱方式
    dataloader = DataLoader(dataset, sampler=sampler)
    for batch in dataloader:
        train(batch)
```

如果不调用 `set_epoch`，每张卡每轮都拿相同的索引顺序，模型会过拟合到"第0卡总拿0,4,8索引"的模式。

**DistributedSampler 本质**：核心就是跳步分区 + epoch 随机种子管理，padding 处理不整除的情况。

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

### 默认 collate_fn 内部逻辑

PyTorch 默认用的是 `default_collate`，按数据类型自动处理：

```python
def default_collate(batch):
    if all(isinstance(x, torch.Tensor) for x in batch):
        return torch.stack(batch)                # 张量 → stack
    elif all(isinstance(x, np.ndarray) for x in batch):
        return torch.tensor(np.stack(batch))     # numpy → 转tensor后stack
    elif all(isinstance(x, (int, float)) for x in batch):
        return torch.tensor(batch)               # 数字 → 转tensor
    elif all(isinstance(x, str) for x in batch):
        return batch                             # 字符串 → 保持list
    else:
        return batch                             # 其他 → 保持list
```

所以默认 collate_fn 不是简单 stack，而是**按类型分类处理**。

### collate_fn 里不能做数据增强

数据增强是 CPU 密集操作，放在 collate_fn 里会导致问题：

```
Worker1: __getitem__(0) → 数据增强 → yield
Worker2: __getitem__(1) → 数据增强 → yield
...
        ↓
  主进程收集样本
        ↓
  collate_fn() ← 在主进程里执行！
        ↓
  collate_fn 里的增强 → 主进程被阻塞 → GPU空等
```

- collate_fn 在主进程执行，是**串行**的，增强期间 GPU 空闲
- worker 送完样本后阻塞在"等主进程收"，**并行优势浪费**
- 增强结果不可复用，每批都要重算

**正确做法**：数据增强放在 `__getitem__` 里（worker 并行执行），collate_fn 只做组装：

```
Worker1: __getitem__(0) → 数据增强 → yield  ← 并行算
Worker2: __getitem__(1) → 数据增强 → yield  ← 并行算
        ↓
  主进程收集样本
        ↓
  collate_fn()  ← 只做组装，几乎零开销
        ↓
  GPU训练 ← worker 同时已经在算下一批
```

**结论**：collate_fn 应保持轻量，数据增强必须前置到 Dataset 的 `__getitem__` 里。

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

# MNIST（手写数字，0~9，28×28灰度图）
datasets.MNIST(root='./data', train=True, download=True,
               transform=transforms.ToTensor())
# train=True → 6万训练集；train=False → 1万测试集
# download=True 仅在数据不存在时下载，已有缓存则直接使用

# CIFAR-10（10类彩色图：飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车，32×32）
datasets.CIFAR10(root='./data', train=True, download=True,
                 transform=transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                 ]))

# ImageNet（1000类，需申请下载权限，图片尺寸不固定）
datasets.ImageNet(root='./data', split='train')
```

### transform 传进 Dataset，不是 DataLoader

`transform` 在 Dataset 的 `__getitem__` 里应用，DataLoader 不直接管 transform：

```python
# transform 传进 Dataset
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 数据增强
    transforms.ToTensor(),                       # PIL图 → [0,1]张量
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1,1]
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32)  # DataLoader 只管凑 batch，不管 transform

# 训练集和测试集的 transform 通常不同
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 训练时做增强
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),                      # 测试时只归一化，不增强
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
```

### ToTensor() 的作用

`transforms.ToTensor()` 做两件事：
1. 把 PIL.Image 或 numpy array 转成 PyTorch 张量
2. 自动把像素值从 [0,255] 缩放到 [0,1]

### Normalize 归一化

```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# 作用：img = (img - mean) / std
# [0,1] 的图经过 Normalize(mean=0.5, std=0.5) 后 → [-1, 1]
# 目的是让数据分布更稳定，加速收敛
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

### Q5: 为什么数据增强要放在 `__getitem__` 里而不是 collate_fn 里？

collate_fn 在主进程串行执行，数据增强是 CPU 密集操作，会阻塞主进程导致 GPU 空等。`__getitem__` 在 worker 进程并行执行，增强和 GPU 训练可流水线并行。

### Q6: IterableDataset 和 Map-style Dataset 在多 worker 场景下有什么区别？

Map-style 的 sampler 索引会被复制到各 worker，按步长跳取不会重复。IterableDataset 每个 worker 都重新执行 `__iter__()`，会导致数据重复，必须通过 `worker_init_fn` 为各 worker 分配不同数据分片。

### Q7: pin_memory 为什么能加速？

锁页内存不会被换出到磁盘，CPU→GPU 传输可通过 DMA 直传，无需经过临时缓冲区拷贝。普通内存在传输前可能需要先换页回内存，再拷贝到临时锁页缓冲区，再传 GPU，多两步开销。

### Q8: shuffle 可以给 IterableDataset 用吗？

不可以。shuffle 依赖 `__len__` 构建随机索引，IterableDataset 没有长度，PyTorch 直接抛 `TypeError: shuffle option is not supported with IterableDataset`。流式数据的打乱需在数据库查询层面用 `ORDER BY RANDOM()` 实现。

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
