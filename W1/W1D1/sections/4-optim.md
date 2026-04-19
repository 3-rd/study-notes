# 四、torch.optim — 优化器

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

---

## 4-1 SGD 与 Momentum

## 4-2 Adam / AdamW 原理

## 4-3 学习率调度（lr_scheduler）

## 4-4 参数分组（不同层不同学习率）

## 4-5 优化器核心 API

---

## 代码练习
