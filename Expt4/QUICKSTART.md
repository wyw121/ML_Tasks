# 🚀 快速开始指南

## ⚡ 3步快速运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行程序
python mnist_keras.py

# 3. 查看结果
# 自动生成 improved_training_history.png 和 predictions.png
```

---

## 📋 环境要求

### Python版本
- Python 3.6+

### 核心依赖
```
tensorflow>=1.10.0
keras>=2.2.0
numpy>=1.19.0
matplotlib>=3.3.0
h5py>=2.10.0
```

---

## 📦 安装依赖

### 方式一：使用requirements.txt（推荐）
```bash
pip install -r requirements.txt
```

### 方式二：手动安装
```bash
pip install tensorflow keras numpy matplotlib h5py
```

### 验证安装
```python
python -c "import tensorflow as tf; import keras; print(f'TF:{tf.__version__}, Keras:{keras.__version__}')"
```

预期输出：
```
TF:2.x.x, Keras:2.x.x
```

---

## 🎯 运行程序

### 默认运行（改进模型）
```bash
python mnist_keras.py
# 直接回车，运行改进的多层神经网络
```

### 选择实验模式
程序会提示：
```
请选择实验模式:
1. 基础实验（单层感知机，200轮训练）
2. 改进实验（多层神经网络，20轮训练）
3. 两个实验都运行
```

**推荐选择**：
- 快速体验：选择 `2`（约40秒）
- 完整对比：选择 `3`（约4分钟）
- 学习基础：选择 `1`（约3分钟）

---

## 📊 输出文件

### 运行改进实验（选项2）生成：
```
improved_training_history.png  # 训练曲线（损失和准确率）
predictions.png                # 10个样本的预测结果
```

### 运行基础实验（选项1）生成：
```
basic_training_history.png     # 训练曲线
predictions.png                # 预测结果
```

### 运行全部实验（选项3）生成：
```
basic_training_history.png     # 基础模型训练曲线
improved_training_history.png  # 改进模型训练曲线
predictions.png                # 预测结果（最后一次运行）
```

---

## 📈 预期结果

### 改进模型（多层神经网络）
```
Epoch 20/20
48000/48000 [======] - loss: 0.1906 - acc: 0.9453 - val_acc: 0.9476

Test accuracy: 0.9452  (约94.5%)
```

### 基础模型（单层感知机）
```
Epoch 200/200
48000/48000 [======] - loss: 0.2761 - acc: 0.9230 - val_acc: 0.9241

Test accuracy: 0.9213  (约92.1%)
```

---

## 🔧 常见问题

### Q1: ModuleNotFoundError: No module named 'tensorflow'
**解决**:
```bash
pip install tensorflow
```

### Q2: ModuleNotFoundError: No module named 'keras'
**解决**:
```bash
pip install keras
```

### Q3: 训练很慢
**原因**: CPU训练较慢  
**解决**:
- 改进模型只需20轮，约40秒
- 如有GPU：`pip install tensorflow-gpu`

### Q4: 无法下载MNIST数据集
**原因**: 网络问题  
**解决**:
```python
# 手动下载并放置到：~/.keras/datasets/mnist.npz
# 或使用镜像源下载
```

### Q5: ImportError: cannot import name 'np_utils'
**解决**: Keras 2.x使用
```python
from keras.utils import to_categorical
```

---

## 💡 快速参数调整

### 修改训练参数
打开 `mnist_keras.py`，找到：

```python
# 调整训练轮数
classifier.train_model(
    batch_size=128,    # 批大小
    epochs=20,         # 训练轮数 ← 修改这里
    validation_split=0.2  # 验证集比例
)
```

### 常用调整
```python
# 快速测试（5轮）
epochs=5

# 标准训练（20轮）
epochs=20

# 深度训练（50轮）
epochs=50
```

---

## 📊 项目结构

```
Expt4/
├── mnist_keras.py                    # 主程序
├── README.md                         # 完整文档
├── QUICKSTART.md                     # 本文件
├── requirements.txt                  # 依赖清单
├── improved_training_history.png     # 训练曲线（运行后生成）
└── predictions.png                   # 预测结果（运行后生成）
```

---

## 🎯 核心代码片段

### 加载数据
```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### 数据预处理
```python
# 重塑
X_train = X_train.reshape(60000, 784).astype('float32')

# 归一化
X_train = X_train / 255.0

# One-hot编码
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
```

### 构建模型
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_shape=(784,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### 编译和训练
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
```

### 评估
```python
score = model.evaluate(X_test, Y_test)
print(f"Test accuracy: {score[1]:.4f}")
```

---

## 🎓 学习建议

### 第一次运行（10分钟）
1. 安装依赖（2分钟）
2. 运行改进实验（选项2，40秒）
3. 查看结果图片（5分钟）

### 深入学习（30分钟）
1. 阅读 README.md 理解原理（20分钟）
2. 修改参数重新训练（5分钟）
3. 分析训练曲线（5分钟）

### 代码研究（1小时）
1. 研读 mnist_keras.py 源代码（30分钟）
2. 尝试修改网络结构（20分钟）
3. 实验不同优化器（10分钟）

---

## 🚀 性能优化建议

### 提高准确率
```python
# 1. 增加隐藏层
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# 2. 使用更好的优化器
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.001), ...)

# 3. 添加Dropout
from keras.layers import Dropout
model.add(Dropout(0.5))
```

### 加快训练
```python
# 增加批大小
batch_size=256

# 使用Adam优化器
optimizer=Adam(lr=0.001)

# GPU加速
pip install tensorflow-gpu
```

---

## 📞 获取帮助

### 文档资源
- [README.md](README.md) - 完整实验文档
- [Keras官方文档](https://keras.io)
- [TensorFlow教程](https://www.tensorflow.org)

### 常见错误
1. **导入错误** → 检查依赖安装
2. **内存不足** → 减小batch_size
3. **训练太慢** → 减少epochs或使用GPU

---

## ✅ 检查清单

运行前确认：
- [ ] Python 3.6+ 已安装
- [ ] 依赖已安装（`pip list`检查）
- [ ] 有稳定网络（首次需下载MNIST）
- [ ] 有足够磁盘空间（约50MB）

---

## 🎉 快速成功体验

```bash
# 复制粘贴这3行命令，1分钟完成实验：

pip install tensorflow keras numpy matplotlib h5py
python mnist_keras.py
# 直接回车，等待40秒，完成！
```

---

**更新日期**: 2024-12-12  
**版本**: 1.0  
**状态**: ✅ 可用

祝您实验顺利！🎓✨
