# 文档导航中心

## 📚 完整文档列表

本项目包含以下详细文档，帮助您快速上手和深入理解：

### 🚀 快速开始

| 文档 | 内容 | 适合人群 |
|-----|------|--------|
| [QUICKSTART.md](QUICKSTART.md) | 环境配置、安装依赖、快速运行 | 所有初学者 |
| [README.md](README.md) | 项目概览、功能介绍、使用说明 | 所有用户 |

### 📖 深度学习理论

| 文档 | 内容 | 适合人群 |
|-----|------|--------|
| [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) | 实验完整总结、关键技术点、扩展方向 | 想要深入了解的用户 |
| [CNN_ARCHITECTURE.md](CNN_ARCHITECTURE.md) | CNN网络结构详解、参数计算、特征提取过程 | 对CNN感兴趣的用户 |

### 💻 代码文档

#### 模块一：肺炎图像识别

```
pneumonia_recognition/
├── pneumonia_recognition.py      # 主程序（1000+行）
│   ├── 自编码器实现
│   ├── CNN分类器实现
│   ├── 训练循环
│   ├── 模型评估
│   └── 结果可视化
└── data/                          # 数据目录
    ├── train/                     # 训练集
    └── test/                      # 测试集
```

**主要功能**
- 自动生成或加载医学图像数据
- 用自编码器进行图像去噪
- 用CNN进行三分类
- 绘制训练过程曲线
- 保存训练好的模型

**核心代码片段**
```python
# 自编码器
autoencoder = AutoEncoder().to(device)
reconstructed = autoencoder(noisy_images)
loss = mse_loss(reconstructed, clean_images)

# CNN
cnn = CNNClassifier().to(device)
denoised = autoencoder(images)  # 去噪
predictions = cnn(denoised)     # 分类
loss = cross_entropy(predictions, labels)
```

#### 模块二：药物评价情感分析

```
drug_sentiment_analysis/
├── sentiment_analysis.py          # Keras版本实现
│   ├── 数据生成和处理
│   ├── 文本序列化和填充
│   ├── LSTM模型定义
│   ├── 模型训练和评估
│   └── 结果可视化
│
├── sentiment_analysis_pytorch.py  # PyTorch版本实现（选做）
│   ├── 相同的数据处理
│   ├── PyTorch模型定义
│   ├── 手写训练循环
│   ├── 模型评估
│   └── 结果可视化
│
└── data/                          # 数据目录
    ├── train/                     # 训练数据
    └── test/                      # 测试数据
```

**主要功能**
- 生成或加载药物评价数据
- 文本序列化、填充和编码
- LSTM模型构建
- 三分类情感分析
- Keras和PyTorch两个版本对比

**核心代码片段**
```python
# Keras模型
model = Sequential([
    Embedding(vocab_size, 64),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# PyTorch模型
class LSTMSentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm1 = nn.LSTM(64, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 3)
```

### 🔧 工具脚本

| 脚本 | 功能 | 使用方法 |
|-----|-----|--------|
| [run_all_experiments.py](run_all_experiments.py) | 运行所有模块 | `python run_all_experiments.py --module all` |
| [visualize_models.py](visualize_models.py) | 模型架构可视化 | `python visualize_models.py` |

## 🎯 学习路径建议

### 初级（第1天）

1. 阅读 [QUICKSTART.md](QUICKSTART.md)
   - 理解环境配置步骤
   - 按指南安装依赖

2. 运行肺炎识别模块
   ```bash
   cd pneumonia_recognition
   python pneumonia_recognition.py
   ```
   - 理解模块运行流程
   - 查看生成的可视化结果

3. 阅读代码注释
   - 理解自编码器的工作原理
   - 理解CNN的分类流程

### 中级（第2-3天）

1. 阅读 [CNN_ARCHITECTURE.md](CNN_ARCHITECTURE.md)
   - 深入理解CNN的网络结构
   - 计算每层的输出尺寸
   - 理解参数数量计算

2. 运行情感分析模块
   ```bash
   cd drug_sentiment_analysis
   python sentiment_analysis.py
   ```
   - 理解文本处理流程
   - 理解LSTM的工作原理

3. 对比Keras和PyTorch版本
   ```bash
   python sentiment_analysis_pytorch.py
   ```
   - 理解两个框架的差异
   - 理解相同算法的不同实现

### 高级（第4-5天）

1. 阅读 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)
   - 深入理解深度学习原理
   - 学习框架对比分析
   - 了解常见问题和解决方案

2. 修改代码进行实验
   - 调整超参数观察效果
   - 添加新的层或模块
   - 尝试使用真实数据

3. 进行扩展开发
   - 实现模型部署
   - 构建Web应用
   - 尝试迁移学习

## 📊 关键概念速查表

### 模块一：图像处理

**自编码器关键参数**
```
输入: 64×64 灰度图
编码器: Conv → ReLU → Pool (×3)
瓶颈层: 64×8×8 特征
解码器: ConvT → ReLU (×3)
输出: 64×64 重构图
```

**CNN分类器关键参数**
```
输入: 64×64 灰度图
卷积块: Conv → ReLU → Pool (×3)
输出特征: 64×8×8
分类器: FC(4096→128) → FC(128→3)
输出: 3个类别概率
```

### 模块二：文本处理

**LSTM情感分析关键参数**
```
词汇表大小: 5000
序列长度: 100
嵌入维度: 64
LSTM隐藏元: 64 → 32
输出类别: 3 (消极、中性、积极)
```

**文本处理流程**
```
原文本 → 分词 → 序列化(词→ID) → 填充到100 
      → 嵌入(64维) → LSTM → 分类
```

## 🎓 理论复习资源

### 深度学习基础

1. **神经网络**
   - 感知机和多层感知机
   - 激活函数（ReLU, Sigmoid, Softmax）
   - 反向传播算法

2. **卷积神经网络 (CNN)**
   - 卷积操作
   - 池化操作
   - 参数共享和局部连接

3. **循环神经网络 (RNN/LSTM)**
   - LSTM结构（遗忘门、输入门、输出门）
   - 门控机制
   - 长期依赖学习

4. **无监督学习**
   - 自编码器
   - 变分自编码器 (VAE)
   - 生成对抗网络 (GAN)

### 框架选择

| 选择 | PyTorch | Keras |
|------|---------|-------|
| 学术论文实现 | ✓✓✓ | ✓ |
| 快速原型开发 | ✓ | ✓✓✓ |
| 工业生产部署 | ✓✓ | ✓✓✓ |
| 学习难度 | 高 | 低 |
| 社区活跃度 | ✓✓✓ | ✓✓ |

## 🚨 常见问题快速解答

**Q: 运行脚本时内存不足？**
A: 减小 `BATCH_SIZE` 或使用GPU

**Q: 如何使用GPU加速？**
A: 安装CUDA后，脚本会自动检测并使用GPU

**Q: 如何在自己的数据上训练？**
A: 修改数据加载部分，使用 `datasets.ImageFolder` 或 `pd.read_csv`

**Q: 模型准确率低怎么办？**
A: 增加训练轮数、调整学习率、增加模型复杂度

**Q: 两个框架（Keras和PyTorch）哪个更好？**
A: 各有优势，选择适合你的应用场景即可

## 📝 实验提交清单

完成实验前，请检查以下内容：

- [ ] 肺炎识别模块正常运行
- [ ] CNN结构图已绘制
- [ ] 模型保存成功
- [ ] 结果可视化图表已生成
- [ ] 药物评价模块(Keras)正常运行
- [ ] 药物评价模块(PyTorch)正常运行（选做）
- [ ] 框架对比分析已完成
- [ ] 实验报告已准备

## 🔗 相关资源链接

### 官方文档
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [Keras官方文档](https://keras.io/)
- [TensorFlow官方文档](https://www.tensorflow.org/)

### 学习资源
- [Fast.ai深度学习课程](https://www.fast.ai/)
- [Stanford CS231n (CNN)](http://cs231n.stanford.edu/)
- [Stanford CS224n (NLP)](http://cs224n.stanford.edu/)

### 工具和库
- [Jupyter Notebook](https://jupyter.org/)
- [PyCharm IDE](https://www.jetbrains.com/pycharm/)
- [VS Code + Python Extension](https://code.visualstudio.com/)

### 数据集
- [ImageNet](http://www.image-net.org/)
- [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)

## 📞 技术支持

遇到问题？

1. 检查 [QUICKSTART.md](QUICKSTART.md) 中的常见问题
2. 查看代码注释和文档
3. 检查官方文档
4. 搜索相关错误信息

## ✅ 最终检查清单

在提交之前：

```python
# 代码质量检查
- [ ] 代码有适当的注释
- [ ] 没有硬编码的路径
- [ ] 错误处理完善
- [ ] 代码风格一致

# 功能检查
- [ ] 所有模块都能正常运行
- [ ] 生成了所有必要的输出文件
- [ ] 结果准确且合理

# 文档检查
- [ ] README文档完整
- [ ] 代码注释清晰
- [ ] 实验步骤可重现

# 性能检查
- [ ] 模型准确率达到预期
- [ ] 训练时间合理
- [ ] 内存占用可接受
```

---

**最后更新**: 2024-12-12  
**文档版本**: 1.0.0  
**难度等级**: ⭐⭐⭐☆☆  
**预计时间**: 8-10小时  

**Happy Learning! 🎉**
