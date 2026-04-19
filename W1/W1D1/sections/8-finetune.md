# 八、Fine-tuning（微调）

| 方式 | 做法 |
|---|---|
| 局部微调 | 冻结底层参数（`requires_grad=False`），只训练顶层 |
| 全局微调 | 不同层设不同学习率（在 optimizer param_groups 中配置） |
| 加载预训练 | `model = torchvision.models.resnet18(pretrained=True)` |

---

## 8-1 预训练模型加载

## 8-2 冻结层微调（Frozen）

## 8-3 不同层不同学习率

## 8-4 Adapter 简介

---

## 代码练习
