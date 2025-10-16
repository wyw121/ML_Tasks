# 二手车价格预测 - 深度学习作业

## 📁 项目结构

```
作业1/
├── used_car/                    # 数据文件夹
│   ├── used_car_train.csv       # 训练数据（40,000条）
│   └── used_car_test.csv        # 测试数据（4,000条）
│
├── 文档/                        # 学习文档（详细教程）
│   ├── 训练成功总结.md          # ⭐ 先看这个！
│   ├── 深度学习作业详解.md      # 理论详解
│   ├── 快速开始指南.md
│   ├── 常见问题解答.md
│   ├── 实验指南.md
│   └── ...
│
├── simple_train.py              # ⭐ 训练代码（极简版）
├── simple_model.pth             # 训练好的模型
├── submission.csv               # 预测结果
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 训练模型
```powershell
python simple_train.py
```

**预计时间**：1-2分钟  
**输出文件**：
- `simple_model.pth` - 训练好的模型
- `submission.csv` - 预测结果

### 2. 查看结果
```powershell
Get-Content submission.csv -Head 10
```

---

## 📊 训练结果

- **训练轮数**：100 epochs
- **最佳损失**：0.0487
- **预测数量**：4,000 条
- **价格范围**：0 ~ 74,000 元

---

## 🎓 学习资料

| 文档 | 内容 | 优先级 |
|-----|------|--------|
| `文档/训练成功总结.md` | 成果展示、核心概念 | ⭐⭐⭐ |
| `文档/深度学习作业详解.md` | 李宏毅课程对应 | ⭐⭐⭐ |
| `文档/常见问题解答.md` | 25个常见问题 | ⭐⭐ |
| `文档/实验指南.md` | 超参数调优实验 | ⭐ |

---

## 💡 代码说明

### 核心训练流程（simple_train.py）

```python
for epoch in range(100):              # 训练100轮
    for batch in 分批(数据, 64):      # Mini-batch
        预测 = 模型(batch)            # 前向传播
        损失 = MSE(预测, 真实)        # 计算损失
        损失.backward()              # 反向传播
        优化器.step()                # 梯度下降
```

**对应李宏毅课程**：
- ✅ 梯度下降（Gradient Descent）
- ✅ Mini-batch 训练
- ✅ Adam 优化器（自适应学习率）
- ✅ MSE 损失函数（回归问题）

---

## 📝 提交作业

需要提交：
1. **代码**：`simple_train.py`
2. **结果**：`submission.csv`
3. **报告**：实验过程和结果分析

---

## 🔧 环境要求

```bash
pip install torch pandas numpy scikit-learn
```

**已安装版本**：
- PyTorch 2.8.0
- Pandas 2.3.1
- NumPy 2.3.2
- Scikit-learn 1.7.2

---

## ✨ 特点

- ✅ **极简代码**：180行，容易理解
- ✅ **稳定训练**：无 nan 或错误
- ✅ **快速运行**：1-2分钟完成
- ✅ **详细注释**：每步都有说明

---

**祝你作业顺利！** 🎉
