# 五、DataLoader — 数据加载

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

---

## 5-1 自定义 Dataset

## 5-2 DataLoader 核心参数

## 5-3 collate_fn 与变长序列

## 5-4 pin_memory 与加速

## 5-5 torchvision 内置数据集

---

## 代码练习
