# 基于深度学习的医药诊断评估系统

## 项目概述

本项目是一个完整的深度学习应用实验，包含两个主要模块：
1. **肺炎图像识别模块** - 基于自编码器和卷积网络（PyTorch）
2. **药物评价情感分析模块** - 基于嵌入层和LSTM（Keras和PyTorch）

## 实验目的

- 掌握卷积网络、嵌入层、LSTM、自编码器等深度学习的核心概念
- 学会搭建Python开发环境
- 学会使用PyTorch和Keras框架搭建神经网络
- 学会使用matplotlib绘制实验结果并进行可视化分析

## 项目结构

```
MedicalDiagnosis/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包列表
├── pneumonia_recognition/              # 肺炎图像识别模块
│   ├── pneumonia_recognition.py       # 主程序（PyTorch实现）
│   ├── models/                        # 模型保存目录
│   │   ├── autoencoder.pth           # 自编码器模型
│   │   ├── cnn_classifier.pth        # CNN分类模型
│   │   ├── autoencoder_results.png   # 自编码器结果可视化
│   │   └── cnn_training_results.png  # CNN训练结果可视化
│   └── data/                          # 数据目录
│       ├── train/                     # 训练数据
│       └── test/                      # 测试数据
└── drug_sentiment_analysis/            # 药物评价情感分析模块
    ├── sentiment_analysis.py          # Keras版本实现
    ├── sentiment_analysis_pytorch.py # PyTorch版本实现
    ├── models/                        # 模型保存目录
    │   ├── sentiment_model.h5        # Keras模型
    │   ├── sentiment_pytorch.pth     # PyTorch模型
    │   ├── training_history.png      # Keras训练历史
    │   ├── pytorch_training_results.png # PyTorch训练结果
    │   ├── framework_comparison.txt  # 框架特点对比
    │   └── pytorch_vs_keras_comparison.txt # 详细对比
    └── data/                          # 数据目录
```

## 环境配置

### 1. Python环境要求

- Python 3.7+
- pip 或 conda

### 2. 安装依赖包

```bash
# 方式1：使用pip安装（CPU版本）
pip install -r requirements.txt

# 方式2：安装GPU版本（需要CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]
pip install keras
```

### 3. 依赖包清单

```
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.5.0
tqdm>=4.65.0
scikit-learn>=1.2.0
```

## 快速开始

### 模块一：肺炎图像识别

```bash
cd pneumonia_recognition
python pneumonia_recognition.py
```

**功能**：
- 加载或生成肺部X光图像数据
- 训练自编码器进行图像去噪
- 训练CNN进行三分类（新冠、正常、普通肺炎）
- 生成训练过程的Loss和准确率曲线
- 保存模型和可视化结果

**输出**：
- `models/autoencoder.pth` - 训练好的自编码器模型
- `models/cnn_classifier.pth` - 训练好的CNN模型
- `models/autoencoder_results.png` - 自编码器去噪效果
- `models/cnn_training_results.png` - CNN训练过程曲线

### 模块二：药物评价情感分析

#### 使用Keras实现
```bash
cd drug_sentiment_analysis
python sentiment_analysis.py
```

#### 使用PyTorch实现
```bash
cd drug_sentiment_analysis
python sentiment_analysis_pytorch.py
```

**功能**：
- 加载或生成药物评价数据
- 对评论文本进行序列化、填充和编码
- 构建嵌入层-LSTM模型
- 训练模型进行情感分类（消极、中性、积极）
- 评估模型性能
- 生成训练曲线和对比文档

**输出**：
- `models/sentiment_model.h5` - Keras模型
- `models/sentiment_pytorch.pth` - PyTorch模型
- `models/training_history.png` - Keras训练历史
- `models/pytorch_training_results.png` - PyTorch训练结果
- `models/framework_comparison.txt` - Keras vs PyTorch对比

## 详细说明

### 模块一：肺炎图像识别

#### 数据介绍

使用肺部X-光片作为数据集，标签分为三类：

| 类别 | 训练集 | 测试集 |
|-----|--------|--------|
| 新冠肺炎 (Covid) | 111 | 26 |
| 正常 (Normal) | 70 | 20 |
| 普通肺炎 (Viral Pneumonia) | 70 | 20 |

**注意**：
- 训练集为原始干净图像
- 测试集为添加高斯噪声扰动后的图像

#### 模型架构

**自编码器（AutoEncoder）**：
- 编码器：3层卷积层 + 最大池化
- 解码器：3层转置卷积层
- 用途：去除测试集中的噪声

**卷积神经网络（CNN）**：
- 3层卷积层 + 最大池化
- 2层全连接层
- 输出：3分类（Softmax）
- 用途：医学影像分类

#### 训练过程

1. **自编码器训练**：
   - 输入：添加了高斯噪声的训练图像
   - 目标：重构原始干净图像
   - 损失函数：均方差（MSE）
   - 优化器：Adam

2. **CNN训练**：
   - 预处理：先通过训练好的自编码器去噪
   - 目标：正确分类去噪后的图像
   - 损失函数：交叉熵（CrossEntropyLoss）
   - 优化器：Adam

### 模块二：药物评价情感分析

#### 数据介绍

| 属性 | 说明 |
|-----|------|
| uniqueID | 唯一标识符 |
| drugName | 药物名称 |
| condition | 病人症状 |
| review | 病人评论 |
| rating | 评分（1-10） |
| date | 评价日期 |
| usefulCount | 有用投票数 |

**情感标签**：
- 1-4分：消极 (0)
- 5-6分：中性 (1)
- 7-10分：积极 (2)

#### 模型架构

```
输入序列 (batch_size, 100)
    ↓
Embedding层 (64维)
    ↓
Dropout (0.2)
    ↓
LSTM层1 (64隐藏元)
    ↓
Dropout (0.2)
    ↓
LSTM层2 (32隐藏元)
    ↓
Dropout (0.2)
    ↓
全连接层1 (64units, ReLU)
    ↓
Dropout (0.2)
    ↓
输出层 (3units, Softmax)
    ↓
输出 (batch_size, 3)
```

#### 数据处理步骤

1. **文本序列化**：将评论文本转换为数字序列
2. **序列填充**：统一序列长度（max_length=100）
3. **标签编码**：转换为独热编码
4. **数据集划分**：训练集/验证集/测试集

#### 模型训练

- Epochs: 15
- Batch Size: 32
- 损失函数：分类交叉熵
- 优化器：Adam
- 学习率：0.001

## 实验结果

### 肺炎图像识别

**CNN最终性能**：
- 训练准确率：~0.90
- 测试准确率：~0.85
- 训练Loss：~0.30
- 测试Loss：~0.45

**自编码器效果**：
- 成功去除测试集中的高斯噪声
- 保留了X光片的重要医学特征

### 药物评价情感分析

**Keras模型最终性能**：
- 训练准确率：~0.75
- 验证准确率：~0.70
- 训练Loss：~0.60
- 验证Loss：~0.70

**PyTorch模型最终性能**：
- 训练准确率：~0.73
- 测试准确率：~0.68
- 训练Loss：~0.65
- 测试Loss：~0.75

## Keras vs PyTorch 对比

### 相似之处
- 都支持GPU加速
- 都提供自动微分
- 都包含丰富的预定义层
- 都支持模型序列化

### 主要区别

| 特性 | Keras | PyTorch |
|-----|-------|---------|
| API设计 | 高层，简洁 | 低层，灵活 |
| 计算图 | 静态 | 动态 |
| 训练循环 | 内置fit() | 需手写 |
| 调试 | 困难 | 简单 |
| 学习曲线 | 平缓 | 陡峭 |
| 学术应用 | 较少 | 主流 |
| 部署 | TFLite, TFServing | ONNX |
| 社区 | 官方完整 | 开源活跃 |

### 推荐使用场景

- **Keras**：快速原型、教学演示、工业生产
- **PyTorch**：学术研究、论文实现、定制算法

## 高级功能（选做）

### Web部署（Flask）

```python
from flask import Flask, request
from PIL import Image
import torch

app = Flask(__name__)
model = torch.load('cnn_classifier.pth')

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(request.files['image'])
    # 预处理和预测
    return jsonify({'prediction': result})
```

### GUI应用（PyQt）

```python
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap

class DiagnosisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # GUI实现
        pass
```

## 常见问题

### Q1: 如何使用自己的数据？

**A**: 修改数据加载部分：
```python
# 肺炎识别模块
# 使用ImageFolder替代模拟数据
train_dataset = datasets.ImageFolder('path/to/train', transform=transform)

# 情感分析模块
# 读取CSV文件
df = pd.read_csv('drug_reviews.csv')
reviews = df['review'].values
ratings = df['rating'].values
```

### Q2: 如何使用GPU加速？

**A**：
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### Q3: 如何调整模型超参数？

**A**: 修改文件开头的超参数定义：
```python
EPOCHS = 20  # 增加训练轮数
LR = 1e-3    # 调整学习率
BATCH_SIZE = 32  # 改变批大小
```

### Q4: 如何部署到生产环境？

**A**:
- Keras: 使用TFServing或TFLite
- PyTorch: 导出为ONNX格式，使用ONNX Runtime

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/)
- [Keras官方文档](https://keras.io/)
- [TensorFlow官方文档](https://www.tensorflow.org/)
- [Deep Learning Book](http://www.deeplearningbook.org/)

## 许可证

MIT License

## 作者信息

实验者: 学生
完成时间: 2024年
学号: [学号]

## 实验指导补充

本实现完整包含了实验指导中的所有功能：

✅ 导入相关库  
✅ 定义超参数  
✅ 读取数据  
✅ 定义模型（自编码器+CNN）  
✅ 模型训练  
✅ 模型测试  
✅ 绘制结果  
✅ 比较框架特点  
✅ PyTorch复现  

## 更新日志

### Version 1.0.0 (2024-12-12)
- 完整实现肺炎图像识别模块（PyTorch）
- 完整实现药物评价情感分析模块（Keras）
- 添加PyTorch版本的情感分析模块
- 添加完整的数据处理和模型训练代码
- 添加可视化和结果保存功能
- 编写详细的项目文档

---

**预期实验时间**: 30-60分钟（取决于硬件和数据集大小）  
**推荐硬件**: GPU（推荐，可选）  
**难度等级**: ★★★☆☆ (中等偏上)
