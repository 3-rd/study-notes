# 九、常见网络层速查

```
卷积:  nn.Conv2d(in, out, kernel, stride, padding)
池化:  nn.MaxPool2d / nn.AvgPool2d / nn.AdaptiveAvgPool2d
BN:    nn.BatchNorm2d(channels)  — train时用batch统计，eval时用全局统计
Dropout: nn.Dropout(p)           — train时随机置0，eval时全部保留
全连接: nn.Linear(in_features, out_features)
激活:  nn.ReLU / nn.Sigmoid / nn.Tanh
```

**卷积层参数说明：** in_channels / out_channels / kernel_size / stride / padding

---

## 9-1 卷积层（Conv2d）

## 9-2 池化层（MaxPool / AvgPool / AdaptiveAvgPool）

## 9-3 归一化层（BatchNorm / LayerNorm / InstanceNorm）

## 9-4 Dropout 与正则化

## 9-5 激活函数对比

## 9-6 常用组合

---

## 代码练习
