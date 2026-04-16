#!/usr/bin/env python3
"""
build.py — 读取 sections/*.md，生成 W1D1-PyTorch-学习笔记.ipynb

每小节一个 .md 文件，文件内 markdown 说明 + ```python 代码块 交替。
build.py 按 ```python 分隔符拆分：前为 markdown，后为代码。
"""
import glob, re, os, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

DIR = os.path.dirname(os.path.abspath(__file__))

# ── helpers ────────────────────────────────────────────────────────────────

def split_cells(text):
    """按 ```python ... ``` 拆成 [('md', str), ('code', str), ...]"""
    # 要求 closing ``` 前必须有换行，这样能正确截止
    code_pat = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
    matches = list(code_pat.finditer(text))
    cells = []
    prev = 0
    for m in matches:
        md = text[prev:m.start()].strip()
        if md:
            cells.append(('md', md))
        cells.append(('code', m.group(1).rstrip()))
        prev = m.end()
    tail = text[prev:].strip()
    if tail:
        cells.append(('md', tail))
    return cells

# ── 章节结构 ────────────────────────────────────────────────────────────────

SECTIONS = [
    # (section_title, {
    #   'overview': '总览表格 markdown',
    #   'subs': [(sub_key, sub_title, 'sections/{sub_key}.md' or None), ...]
    # })
    ('一、张量（Tensor）— 最底层数据结构',
     'overview',
     [
         ('1-1-create', '1.1 创建方式'),
         ('1-2-shape',  '1.2 形状操作'),
         ('1-3-index',  '1.3 索引切片'),
         ('1-4-ops',    '1.4 常用运算'),
         ('1-5-gpu',    '1.5 GPU 迁移与设备'),
         ('1-6-numpy',  '1.6 与 NumPy 互转'),
         ('1-7-defaults','1.7 默认值总结'),
     ]),
    ('二、自动求导（Autograd）— PyTorch 灵魂', 'sections/1-5-autograd.md', []),
    ('三、nn.Module — 模型构建核心', 'sections/3-nn-module.md', []),
    ('四、torch.optim — 优化器', None, []),
    ('五、DataLoader — 数据加载', None, []),
    ('六、模型保存与加载', None, []),
    ('七、GPU 加速与分布式', None, []),
    ('八、Fine-tuning（微调）', None, []),
    ('九、常见网络层速查', None, []),
    ('十、训练流程全链路（必须能默写）', None, []),
    ('面试常考点（答案要点）', None, []),
    ('学习资源', None, []),
]

STATIC_OVERVIEW = {
    'overview': """**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| 创建方式 | `torch.tensor()` vs `torch.Tensor()` | 两者区别（是否拷贝数据、类型推断） |
| 数据类型 | `float/int/bool/dtype` | 如何指定 device + dtype 同时创建 |
| GPU 迁移 | `.cuda()` / `.to(device)` | 如何判断 GPU 可用 |
| 形状操作 | `view / reshape / transpose / permute` | view vs reshape 区别（是否连续） |
| 索引切片 | `tensor[mask]` / `torch.masked_select` | 如何按条件筛选 |
| 运算 | `matmul / mm / @` / `torch.sum / mean / max` | 矩阵乘法哪个最快/最安全 |
| 与 NumPy 互转 | `tensor.numpy()` / `torch.from_numpy(nparr)` | 共享内存问题 |
""",
}

# ── build ─────────────────────────────────────────────────────────────────

cells = []

# 标题
cells.append(new_markdown_cell(
    '# W1D1｜PyTorch 张量操作 + 自动求导\n\n'
    '> 学习日期：2026-04-10\n'
    '> 目标：掌握 PyTorch 核心 API，理解自动求导机制，夯实 Day 1 基础'
))

# 一、张量
cells.append(new_markdown_cell(f'## 一、张量（Tensor）— 最底层数据结构\n\n{STATIC_OVERVIEW["overview"]}'))
for sub_key, sub_title in SECTIONS[0][2]:
    path = os.path.join(DIR, 'sections', f'{sub_key}.md')
    if os.path.exists(path):
        text = open(path, encoding='utf-8').read()
        parsed = split_cells(text)
        for kind, content in parsed:
            if kind == 'md' and content.strip():
                cells.append(new_markdown_cell(content))
            elif kind == 'code' and content.strip():
                cells.append(new_code_cell(content))
    else:
        cells.append(new_markdown_cell(f'### {sub_title}'))

# 二 ~ 八（静态内容，直接硬编码）

AUTOGRAD_MD = new_markdown_cell("""## 二、自动求导（Autograd）— PyTorch 灵魂

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| `requires_grad` | 默认为 False，设为 True 开启追踪 | 哪些操作会默认开启 |
| `backward()` | 反向传播计算梯度 | 何时调用，梯度会累加还是覆盖 |
| `grad` | 保存梯度值 | 多个 `backward()` 时梯度如何变化 |
| `grad_fn` | 记录创建张量的运算 | 用于什么 |
| `torch.no_grad()` | 前向推理时不追踪梯度 | 与 `eval()` 区别 |
| `detach()` | 截断计算图 | 何时需要 detach |
| `hook` 机制 | 注册前向/反向 hook | 有什么用 |

**⚠️ 面试必答题：**
- PyTorch 反向传播原理？计算图是怎么构建的？
- `backward()` 掉了梯度会怎样？多次 backward 梯度累加还是覆盖？
- 梯度消失/爆炸的原因？在 PyTorch 中如何检测和解决？
""")

cells.append(AUTOGRAD_MD)
# 从 sections/1-5-autograd.md 读取完整内容
_autograd_path = os.path.join(DIR, 'sections', '1-5-autograd.md')
if os.path.exists(_autograd_path):
    _text = open(_autograd_path, encoding='utf-8').read()
    _parsed = split_cells(_text)
    for _kind, _content in _parsed:
        if _kind == 'md' and _content.strip():
            cells.append(new_markdown_cell(_content))
        elif _kind == 'code' and _content.strip():
            cells.append(new_code_cell(_content))

NN_MD = new_markdown_cell("""## 三、`nn.Module` — 模型构建核心

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| 继承 `nn.Module` | 必须重写 `__init__` + `forward` | 为什么要继承 |
| `super().__init__()` | 调用父类构造函数 | 不调用会怎样 |
| `named_parameters()` / `parameters()` | 遍历模型参数 | 如何冻结部分层 |
| `state_dict()` / `load_state_dict()` | 模型序列化/加载 | 怎么只加载部分参数 |
| `children()` / `modules()` | 遍历子模块 | 区别是什么 |
| 常见层 | `Linear / Conv2d / BatchNorm / Dropout / LSTM` | 参数含义 |

**⚠️ 面试必答题：**
- `nn.Module` 的 `forward` 为什么只需写前向，反向自动搞定？
- `model(img)` 背后发生了什么？（call → forward → hooks）
- `model.train()` vs `model.eval()` 区别？（BN 和 Dropout 的行为差异）
""")

cells.append(NN_MD)
# 从 sections/3-nn-module.md 读取完整内容
_nn_path = os.path.join(DIR, 'sections', '3-nn-module.md')
if os.path.exists(_nn_path):
    _text = open(_nn_path, encoding='utf-8').read()
    _parsed = split_cells(_text)
    for _kind, _content in _parsed:
        if _kind == 'md' and _content.strip():
            cells.append(new_markdown_cell(_content))
        elif _kind == 'code' and _content.strip():
            cells.append(new_code_cell(_content))

OPTIM_MD = new_markdown_cell("""## 四、`torch.optim` — 优化器

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| SGD | 随机梯度下降 + momentum | momentum 是什么 |
| Adam / AdamW | 自适应学习率 | Adam 的原理，W 是什么 |
| 学习率调度 | `lr_scheduler` | 常用调度策略 |
| 不同层不同学习率 | optimizer 参数分组 | 怎么配 |
| `zero_grad()` | 清零梯度 | 为什么要手动调用 |

**⚠️ 面试必答题：**
- SGD 和 Adam 的区别？各自适用场景？
- 学习率衰减策略有哪些？
- 为什么梯度要用 `zero_grad()` 清零，不能累加？
""")

cells.append(OPTIM_MD)
cells.append(new_markdown_cell('### 练习：优化器使用'))
cells.append(new_code_cell("""import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 2)
optimizer = optim.SGD([
    {'params': model.weight, 'lr': 0.01},
    {'params': model.bias, 'lr': 0.1}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
print("初始 lr:", scheduler.get_last_lr())

x = torch.randn(1, 10)
target = torch.tensor([1., 0.])
for epoch in range(3):
    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()}")"""))

DL_MD = new_markdown_cell("""## 五、`DataLoader` — 数据加载

**核心知识点：**

| 知识点 | 说明 | 面试高频追问 |
|---|---|---|
| `Dataset` | 自定义数据抽象，必须实现 `__getitem__` + `__len__` | 如何自己实现 |
| `DataLoader` | batch / shuffle / num_workers | 各参数含义 |
| `collate_fn` | 自定义 batch 拼接 | 什么时候需要重写 |
| `pin_memory` | 加速 GPU 传输 | 什么原理 |
| `torchvision` | 图像领域数据集（MNIST / CIFAR / ImageNet） | |

**⚠️ 面试必答题：**
- DataLoader 的 shuffle 是在哪个层面做的？
- num_workers 设置过大的副作用是什么？
""")

cells.append(DL_MD)
cells.append(new_markdown_cell('### 练习：自定义 Dataset 和 DataLoader'))
cells.append(new_code_cell("""import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(size=20)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
print("数据集大小:", len(dataset))
for batch_x, batch_y in loader:
    print("batch x:", batch_x.shape, "batch y:", batch_y.shape)
    break"""))

SAVE_MD = new_markdown_cell("""## 六、模型保存与加载

| 方式 | 代码 | 适用场景 |
|---|---|---|
| 保存 state_dict | `torch.save(model.state_dict(), path)` | **推荐**，轻量 |
| 保存整个模型 | `torch.save(model, path)` | 不推荐，依赖类定义 |
| 加载 | `model.load_state_dict(torch.load(path))` | 常用方式 |
| 只加载部分参数 | `strict=False` / 过滤 key | 迁移学习/微调 |
| 保存优化器状态 | `torch.save({'model': ..., 'optimizer': ...}, path)` | 断点续训 |
""")

cells.append(SAVE_MD)
cells.append(new_markdown_cell('### 练习：模型保存与加载'))
cells.append(new_code_cell("""import torch
import torch.nn as nn

model = nn.Linear(10, 2)
torch.save(model.state_dict(), '/tmp/model.pth')
new_model = nn.Linear(10, 2)
new_model.load_state_dict(torch.load('/tmp/model.pth'))

class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(10, 5)
        self.fc = nn.Linear(5, 3)
    def forward(self, x):
        return self.fc(self.feature(x))

new_model = NewModel()
pretrained = {'feature.weight': torch.randn(5, 10), 'feature.bias': torch.randn(5)}
new_model.load_state_dict(pretrained, strict=False)
print("部分加载成功")"""))

GPU_MD = new_markdown_cell("""## 七、GPU 加速与分布式

| 知识点 | 说明 |
|---|---|
| 单 GPU | `model.cuda()` / `tensor.to(device)` |
| 多 GPU | `nn.DataParallel(model, device_ids=[0,1,2])` |
| 多 GPU 原理 | 按 batch 维度分割 → 各 GPU 独立 forward → 梯度累加到主 GPU |
| 分布式 DDP | `DistributedDataParallel` — 工业级多机多卡 |
| `torch.cuda.is_available()` | 判断 GPU 是否可用 |
""")

cells.append(GPU_MD)
cells.append(new_markdown_cell('### 练习：GPU 检测与迁移'))
cells.append(new_code_cell("""import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())

model = nn.Linear(10, 2).to(device)
x = torch.randn(1, 10).to(device)
out = model(x)
print("输出设备:", out.device)"""))

FT_MD = new_markdown_cell("""## 八、Fine-tuning（微调）

| 方式 | 做法 |
|---|---|
| 局部微调 | 冻结底层参数（`requires_grad=False`），只训练顶层 |
| 全局微调 | 不同层设不同学习率（在 optimizer param_groups 中配置） |
| 加载预训练 | `model = torchvision.models.resnet18(pretrained=True)` |
""")

cells.append(FT_MD)
cells.append(new_markdown_cell('### 练习：冻结层微调'))
cells.append(new_code_cell("""import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet18(weights=None)  # 用 weights=None 避免下载
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
print("可训练参数:", sum(p.numel() for p in model.parameters() if p.requires_grad))"""))

LAYERS_MD = new_markdown_cell("""## 九、常见网络层速查

```
卷积:  nn.Conv2d(in, out, kernel, stride, padding)
池化:  nn.MaxPool2d / nn.AvgPool2d / nn.AdaptiveAvgPool2d
BN:    nn.BatchNorm2d(channels)  — train时用batch统计，eval时用全局统计
Dropout: nn.Dropout(p)           — train时随机置0，eval时全部保留
全连接: nn.Linear(in_features, out_features)
激活:  nn.ReLU / nn.Sigmoid / nn.Tanh
```

**卷积层参数说明：** in_channels / out_channels / kernel_size / stride / padding
""")

cells.append(LAYERS_MD)

TRAIN_MD = new_markdown_cell("""## 十、训练流程全链路（必须能默写）

```
1. 定义 Dataset + DataLoader
2. 定义模型 (继承nn.Module) → 放到GPU
3. 定义损失函数 (CrossEntropyLoss / MSELoss...)
4. 定义优化器 (SGD / Adam)
5. 训练循环:
   for epoch in range(E):
       model.train()
       for batch in train_loader:
           optimizer.zero_grad()
           output = model(input)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()
       model.eval()
       with torch.no_grad():
           ...
```
""")

cells.append(TRAIN_MD)
cells.append(new_markdown_cell('### 练习：完整训练流程'))
cells.append(new_code_cell("""import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(1000, 20)
y = (x.sum(dim=1) > 0).float().unsqueeze(1)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

model = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(3):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, avg_loss: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    preds = model(torch.randn(10, 20))
    print("预测样例:", preds[:3].squeeze())"""))

QA_MD = new_markdown_cell("""## 面试常考点（答案要点）

### Q1: `torch.Tensor()` vs `torch.tensor()` 区别？
- `torch.Tensor()` 是类构造函数，默认 float32，不保证拷贝数据
- `torch.tensor()` 是工厂函数，**拷贝数据**，参数最全

### Q2: `torch.no_grad()` vs `model.eval()` 区别？
- `no_grad()`：**不构建计算图**，节省显存
- `eval()`：**BN 用全局统计量，Dropout 不生效**
- 两者可叠加：`model.eval()` + `with torch.no_grad():`

### Q3: 反向传播原理？
- PyTorch 构建**有向无环图（DAG）**，叶子节点是原始张量
- `backward()` 从输出反向遍历，累加梯度到 `.grad`

### Q4: 梯度消失/爆炸的原因和解决？
- 原因：链式求导连乘效应
- 解决：梯度裁剪、残差连接、归一化、激活函数选择

### Q5: 为什么梯度要用 `zero_grad()` 清零？
- 不清零会**累加**到 `.grad`，导致参数更新错误

### Q6: DataLoader 的 pin_memory 是什么？
- 将数据加载到锁页内存，再传到 GPU，跳过 CPU-GPU 拷贝的同步开销
""")

cells.append(QA_MD)

RES_MD = new_markdown_cell("""## 学习资源

- PyTorch 官方教程：https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- 自动求导官方文档：https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- 动手学深度学习（中文）：https://zh.d2l.ai/
""")

cells.append(RES_MD)

# ── write ─────────────────────────────────────────────────────────────────

nb = new_notebook()
nb.cells = cells

out = os.path.join(DIR, 'W1D1.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Done! {len(cells)} cells -> {out}')
