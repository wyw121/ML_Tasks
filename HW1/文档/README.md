# 二手车价格预测 - 深度学习作业

## 📋 项目简介

本项目是基于李宏毅老师深度学习课程的实战练习，使用神经网络预测二手车价格。

**数据来源**：阿里天池大数据竞赛

## 📁 文件说明

```
作业1/
├── used_car/
│   ├── used_car_train.csv    # 训练数据（40,001条）
│   └── used_car_test.csv     # 测试数据（4,001条）
├── train_model.py            # 训练代码（详细注释）
├── predict.py                # 预测代码（简化版）
├── 深度学习作业详解.md       # 理论详解文档
└── README.md                 # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### 2. 运行训练

```bash
python train_model.py
```

### 3. 查看结果

训练完成后会生成：
- `best_model.pth` - 最佳模型
- `training_curve.png` - 训练曲线
- `submission.csv` - 预测结果

## 📚 李宏毅课程对应

| 代码部分 | 课程内容 |
|---------|---------|
| `torch.optim.Adam` | Adaptive Learning Rate |
| `DataLoader(batch_size=64)` | Mini-batch 训练 |
| `nn.ReLU()` | 激活函数 |
| `MSELoss()` | 回归损失函数 |
| Early Stopping | 防止过拟合 |

## 🔧 超参数说明

可以在 `train_model.py` 中调整：

```python
BATCH_SIZE = 64      # 批次大小：16/32/64/128
NUM_EPOCHS = 500     # 最大训练轮数
PATIENCE = 20        # Early Stopping 耐心值
learning_rate = 0.001  # 学习率
```

## 📊 模型架构

```
输入层 (28维特征)
    ↓
隐藏层1 (128个神经元) + ReLU
    ↓
隐藏层2 (64个神经元) + ReLU
    ↓
隐藏层3 (32个神经元) + ReLU
    ↓
输出层 (1个值：价格)
```

## 💡 学习建议

1. **先阅读**：`深度学习作业详解.md`
2. **再运行**：`train_model.py`
3. **后实验**：修改超参数，观察效果
4. **最后总结**：对照李宏毅课程，理解原理

## ❓ 常见问题

### Q: 训练很慢怎么办？
A: 增大 `BATCH_SIZE` 到 128，或减少网络层数

### Q: 损失不下降？
A: 检查数据是否标准化，尝试调整学习率

### Q: 过拟合怎么办？
A: Early Stopping 会自动处理，或减少训练轮数

## 📖 参考资料

- [李宏毅机器学习课程 2021](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [天池比赛链接](https://tianchi.aliyun.com/)

## 🎯 作业要求

- [x] 数据预处理
- [x] 构建神经网络
- [x] 训练模型
- [x] 生成预测结果
- [x] 代码注释详细
- [x] 理论解释文档

## ✨ 改进方向

1. 特征工程：创建新特征（如车龄 = 当前年份 - 注册年份）
2. 模型调优：网格搜索最佳超参数
3. 集成学习：多个模型投票预测
4. 数据增强：处理异常值和离群点

---

**作者**：深度学习学习者  
**日期**：2025-10-15  
**版本**：1.0
