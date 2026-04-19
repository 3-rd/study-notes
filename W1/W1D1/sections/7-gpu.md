# 七、GPU 加速与分布式

| 知识点 | 说明 |
|---|---|
| 单 GPU | `model.cuda()` / `tensor.to(device)` |
| 多 GPU | `nn.DataParallel(model, device_ids=[0,1,2])` |
| 多 GPU 原理 | 按 batch 维度分割 → 各 GPU 独立 forward → 梯度累加到主 GPU |
| 分布式 DDP | `DistributedDataParallel` — 工业级多机多卡 |
| `torch.cuda.is_available()` | 判断 GPU 是否可用 |

---

## 7-1 单 GPU 迁移

## 7-2 DataParallel（DP）多卡

## 7-3 DistributedDataParallel（DDP）原理

## 7-4 多卡实验与验证

---

## 代码练习
