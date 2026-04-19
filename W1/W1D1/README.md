# W1D1 — PyTorch 基础

> 本目录为新格式笔记本，W1D1.ipynb 由 sections/*.md + build.py 生成。
> 原始旧格式笔记本见 `../../W1D1.bak/W1D1.ipynb`。

## 状态

| Section | 状态 | 说明 |
|---------|------|------|
| 一、张量 | ✅ | 2026-04-10 讨论 |
| 二、自动求导 | ✅ | 2026-04-10 讨论 |
| 三、nn.Module | ✅ | 2026-04-10 讨论（含 Sequential 局限） |
| 四、优化器 | 📋 | 框架已迁移，待正式讨论 |
| 五、DataLoader | 📋 | 框架已迁移，待正式讨论 |
| 六、模型保存与加载 | 📋 | 框架已迁移，待正式讨论 |
| 七、GPU 加速 | 📋 | 框架已迁移，待正式讨论 |
| 八、Fine-tuning | 📋 | 框架已迁移，待正式讨论 |
| 九、常见网络层速查 | 📋 | 框架已迁移，待正式讨论 |
| 十、训练流程全链路 | 📋 | 框架已迁移，待正式讨论 |

## 目录结构

```
W1D1/
├── W1D1.ipynb       ← 由 sections/*.md + build.py 生成
├── build.py          ← 构建脚本
├── README.md          ← 本文件
└── sections/
    ├── 1-tensor.md
    ├── 2-autograd.md
    ├── 3-nn-module.md
    ├── 4-optim.md
    ├── 5-dataloader.md
    ├── 6-save-load.md
    ├── 7-gpu.md
    ├── 8-finetune.md
    ├── 9-layers-ref.md
    └── 10-training-loop.md
```
