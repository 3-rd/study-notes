# 十、训练流程全链路（必须能默写）

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

---

## 10-1 完整训练循环

## 10-2 验证与测试流程

## 10-3 训练监控（Loss 曲线、Early Stopping）

## 10-4 常见问题与调试

---

## 代码练习
