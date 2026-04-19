# 六、模型保存与加载

| 方式 | 代码 | 适用场景 |
|---|---|---|
| 保存 state_dict | `torch.save(model.state_dict(), path)` | **推荐**，轻量 |
| 保存整个模型 | `torch.save(model, path)` | 不推荐，依赖类定义 |
| 加载 | `model.load_state_dict(torch.load(path))` | 常用方式 |
| 只加载部分参数 | `strict=False` / 过滤 key | 迁移学习/微调 |
| 保存优化器状态 | `torch.save({'model': ..., 'optimizer': ...}, path)` | 断点续训 |

---

## 6-1 state_dict 方式

## 6-2 断点续训（Checkpoint）

## 6-3 部分加载（strict=False）

## 6-4 跨设备保存与加载

---

## 代码练习
