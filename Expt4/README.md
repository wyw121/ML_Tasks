# 实验四：Keras基础与简单应用

## 📋 实验概述

### 实验目标
学会搭建Keras开发环境，掌握基于TensorFlow的高级API框架Keras的基本用法，通过MNIST手写数字体数据集，学会搭建基于Keras API的神经网络，并用来识别手写数字体。

### 数据集
- **MNIST手写数字数据集**
  - 训练集：60,000张 28×28 灰度图像
  - 测试集：10,000张 28×28 灰度图像
  - 类别：10类（数字0-9）
  - 像素值：0-255（黑白图像）

---

## 🎯 实验内容

### 1. 环境配置

#### 必需依赖
```bash
# 核心依赖
pip install tensorflow>=1.10.0
pip install keras>=2.2.0
pip install numpy>=1.19.0
pip install matplotlib>=3.3.0
pip install h5py>=2.10.0
```

#### 安装步骤

**方式一：使用requirements.txt**
```bash
pip install -r requirements.txt
```

**方式二：手动安装**
```bash
# 1. 安装TensorFlow
pip install tensorflow

# 2. 安装Keras
pip install keras

# 3. 安装其他依赖
pip install numpy matplotlib h5py
```

#### 验证安装
```python
# 在Python中运行以下代码
import tensorflow as tf
import keras
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
```

---

## 📊 实验步骤详解

### 步骤1：数据加载

```python
from keras.datasets import mnist

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 查看数据维度
print(X_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(X_test.shape)   # (10000, 28, 28)
print(y_test.shape)   # (10000,)
```

**数据说明**：
- `X_train`：60,000个训练图像，每个28×28像素
- `y_train`：60,000个训练标签（0-9）
- `X_test`：10,000个测试图像
- `y_test`：10,000个测试标签

---

### 步骤2：数据预处理

#### 2.1 重塑形状

将28×28的二维图像重塑为784维的一维向量：

```python
# 重塑形状
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')

print(X_train.shape)  # (60000, 784)
print(X_train.dtype)  # float32
```

**为什么要重塑？**
- 全连接神经网络需要一维输入
- 28×28 = 784个特征

#### 2.2 归一化

将像素值从0-255归一化到0-1：

```python
# 归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

# 查看归一化后的像素值
print(X_train[1, 100:151])
```

**为什么要归一化？**
- 加快训练速度
- 提高数值稳定性
- 避免梯度消失/爆炸

#### 2.3 One-hot编码

将标签转换为one-hot向量：

```python
from keras.utils import np_utils

# One-hot编码
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print(Y_train[:5])
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]  # 标签5
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]  # 标签0
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]  # 标签4
#  ...]
```

**为什么要One-hot编码？**
- 将类别转换为向量表示
- 适合多分类问题
- 配合softmax激活函数使用

---

### 步骤3：构建神经网络

#### 模型A：基础单层感知机

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

# 创建模型
model = Sequential()

# 添加全连接层（784 -> 10）
model.add(Dense(10, input_shape=(784,)))

# 添加激活层（softmax）
model.add(Activation('softmax'))

# 查看模型结构
model.summary()
```

**模型结构**：
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                7850      
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 7,850
```

**参数计算**：7850 = 784 × 10 + 10（权重 + 偏置）

#### 模型B：改进的多层神经网络

```python
# 创建模型
model = Sequential()

# 输入层 -> 隐藏层1 (784 -> 128, relu)
model.add(Dense(128, input_shape=(784,), activation='relu'))

# 隐藏层1 -> 隐藏层2 (128 -> 128, relu)
model.add(Dense(128, activation='relu'))

# 隐藏层2 -> 输出层 (128 -> 10, softmax)
model.add(Dense(10, activation='softmax'))

# 查看模型结构
model.summary()
```

**模型结构**：
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 118,282
```

**网络层次**：
- 第1层：784 → 128（ReLU激活）
- 第2层：128 → 128（ReLU激活）
- 第3层：128 → 10（Softmax激活）

---

### 步骤4：编译模型

```python
from keras.optimizers import SGD

model.compile(
    loss='categorical_crossentropy',  # 多分类交叉熵损失
    optimizer=SGD(),                  # 随机梯度下降
    metrics=['accuracy']              # 准确率
)
```

**参数说明**：
- `loss`：损失函数，用于优化目标
- `optimizer`：优化器，更新权重
- `metrics`：评估指标，衡量性能

**为什么用categorical_crossentropy？**
- 适合多分类问题
- 配合one-hot编码和softmax
- 数学上等价于最大似然估计

---

### 步骤5：训练模型

#### 基础模型训练（200轮）

```python
history = model.fit(
    X_train, Y_train,
    batch_size=128,
    epochs=200,
    verbose=1,
    validation_split=0.2
)
```

**训练输出**：
```
Train on 48000 samples, validate on 12000 samples
Epoch 1/200
48000/48000 [======] - loss: 1.3633 - acc: 0.6796 - val_loss: 0.8904 - val_acc: 0.8246
Epoch 2/200
48000/48000 [======] - loss: 0.7913 - acc: 0.8272 - val_loss: 0.6572 - val_acc: 0.8546
...
Epoch 200/200
48000/48000 [======] - loss: 0.2761 - acc: 0.9230 - val_loss: 0.2756 - val_acc: 0.9241
```

#### 改进模型训练（20轮）

```python
history = model.fit(
    X_train, Y_train,
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_split=0.2
)
```

**训练输出**：
```
Train on 48000 samples, validate on 12000 samples
Epoch 1/20
48000/48000 [======] - loss: 1.4590 - acc: 0.6352 - val_loss: 0.7348 - val_acc: 0.8351
Epoch 2/20
48000/48000 [======] - loss: 0.5887 - acc: 0.8514 - val_loss: 0.4486 - val_acc: 0.8847
...
Epoch 20/20
48000/48000 [======] - loss: 0.1906 - acc: 0.9453 - val_loss: 0.1885 - val_acc: 0.9476
```

**参数说明**：
- `batch_size=128`：每批128个样本
- `epochs=200/20`：训练轮数
- `validation_split=0.2`：20%用作验证集

---

### 步骤6：评估模型

```python
score = model.evaluate(X_test, Y_test, verbose=1)

print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
```

**评估结果**：

| 模型 | 测试损失 | 测试准确率 | 训练轮数 |
|------|---------|-----------|---------|
| 基础模型 | 0.2774 | 92.13% | 200 |
| 改进模型 | 0.1882 | 94.52% | 20 |

**性能提升**：
- 准确率提升：**2.39%**
- 训练轮数减少：**90%** (200→20)
- 训练时间减少：**大幅降低**

---

## 🚀 使用方法

### 快速开始（3步）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行程序
python mnist_keras.py

# 3. 查看结果
# 会自动生成训练曲线和预测结果图
```

### 选择实验模式

程序运行时会提示选择：

```
请选择实验模式:
1. 基础实验（单层感知机，200轮训练）
2. 改进实验（多层神经网络，20轮训练）
3. 两个实验都运行
```

**推荐**：选择 `2` 或直接回车（运行改进实验）

---

## 📈 实验结果

### 输出文件

运行后会生成以下文件：

1. **训练历史曲线**
   - `basic_training_history.png` - 基础模型
   - `improved_training_history.png` - 改进模型

2. **预测结果**
   - `predictions.png` - 10个随机样本的预测结果

### 性能指标

#### 基础模型（单层感知机）
```
训练准确率：92.30%
验证准确率：92.41%
测试准确率：92.13%
训练轮数：200
```

#### 改进模型（多层神经网络）
```
训练准确率：94.53%
验证准确率：94.76%
测试准确率：94.52%
训练轮数：20
```

---

## 💡 关键知识点

### 1. Keras核心概念

**Sequential模型**：
- 线性堆叠的层
- 适合大多数神经网络
- 简单易用

**层（Layer）**：
- Dense：全连接层
- Activation：激活层
- Conv2D：卷积层（本实验未用）

**激活函数**：
- `relu`：ReLU，隐藏层常用
- `softmax`：输出概率分布，多分类常用
- `sigmoid`：二分类常用

### 2. 损失函数

**categorical_crossentropy**：
```
L = -Σ y_true * log(y_pred)
```

适用场景：
- 多分类问题
- One-hot编码的标签
- Softmax输出

### 3. 优化器

**SGD（随机梯度下降）**：
```
w = w - learning_rate * gradient
```

特点：
- 简单有效
- 可能收敛慢
- 适合教学演示

**其他优化器**：
- Adam：自适应学习率
- RMSprop：适合RNN
- Adagrad：适合稀疏数据

### 4. 数据预处理重要性

| 步骤 | 作用 |
|------|------|
| 重塑 | 适应网络输入 |
| 归一化 | 加快收敛 |
| One-hot | 多分类表示 |

---

## 🔧 常见问题

### Q1: 为什么改进模型只需20轮？

**A**: 更深的网络有更强的表达能力：
- 多层网络可以学习更复杂的特征
- ReLU激活避免梯度消失
- 更快达到更好的性能

### Q2: 如何提高准确率？

**A**: 可以尝试：
```python
# 1. 增加网络层数
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# 2. 使用更好的优化器
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.001), ...)

# 3. 添加Dropout防止过拟合
from keras.layers import Dropout
model.add(Dropout(0.5))

# 4. 数据增强
from keras.preprocessing.image import ImageDataGenerator
```

### Q3: 为什么要用验证集？

**A**: 
- 监控过拟合
- 调整超参数
- 早停（Early Stopping）

### Q4: 如何保存模型？

**A**:
```python
# 保存模型
model.save('mnist_model.h5')

# 加载模型
from keras.models import load_model
model = load_model('mnist_model.h5')
```

### Q5: GPU加速如何启用？

**A**:
```bash
# 安装GPU版TensorFlow
pip install tensorflow-gpu

# Keras会自动使用GPU（如果可用）
```

---

## 📊 实验对比

### 模型对比

| 指标 | 基础模型 | 改进模型 | 提升 |
|------|---------|---------|------|
| 准确率 | 92.13% | 94.52% | +2.39% |
| 训练轮数 | 200 | 20 | -90% |
| 参数量 | 7,850 | 118,282 | +15倍 |
| 训练时间 | ~200s | ~40s | -80% |

### 关键发现

✅ **多层网络优势明显**
- 准确率更高
- 收敛更快
- 总训练时间更短

✅ **深度学习的威力**
- 自动学习特征
- 无需手工特征工程
- 端到端训练

---

## 🎓 学习收获

### 技能掌握

✅ **Keras基础**
- 模型构建（Sequential）
- 层的添加（Dense, Activation）
- 模型编译和训练

✅ **深度学习流程**
- 数据加载和预处理
- 模型设计和训练
- 评估和优化

✅ **实践经验**
- MNIST数据集处理
- 神经网络调参
- 性能分析

---

## 🔍 深入学习

### 推荐阅读
1. Keras官方文档：https://keras.io
2. TensorFlow教程：https://www.tensorflow.org
3. 《Deep Learning with Python》- François Chollet

### 进阶实验
1. 使用CNN卷积神经网络（准确率>99%）
2. 实现其他数据集（CIFAR-10, Fashion-MNIST）
3. 迁移学习和预训练模型

---

## 📝 总结

本实验通过MNIST手写数字识别任务，学习了：

1. ✅ **Keras环境搭建**
2. ✅ **数据预处理技巧**
3. ✅ **神经网络构建**
4. ✅ **模型训练和评估**
5. ✅ **性能优化方法**

**关键结论**：
- 深度神经网络比单层感知机性能更好
- 合适的网络结构可以大幅减少训练时间
- Keras提供了简洁高效的深度学习API

---

**实验完成日期**: 2024-12-12  
**版本**: 1.0  
**状态**: ✅ 完成

祝您学习愉快！🎓✨
