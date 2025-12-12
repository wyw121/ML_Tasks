# 实验五：基于Keras卷积神经网络实现（MNIST）

## 实验目标
掌握使用 Keras（TensorFlow 高级 API）构建卷积神经网络（CNN），在 MNIST 手写数字数据集上完成分类，并实践：
- 序列模型（Sequential）与功能模型（Functional）
- 训练 / 评估 / 预测
- 模型保存与加载
- 卷积核权重与输出的可视化

## 运行环境
- Python 3.7+
- TensorFlow / Keras（自动兼容 `tensorflow.keras`，若不可用回退 `keras`）
- 依赖见 `requirements.txt`

安装依赖：
```bash
pip install -r requirements.txt
```

## 文件结构
```
Expt5/
├── cnn_keras.py      # 主程序，含序列/功能模型、训练、评估、可视化、保存/加载
├── mnist.py          # MNIST 数据加载与预处理
├── README.md         # 本说明
├── requirements.txt  # 依赖清单
└── 实验5.xmind        # 实验思维导图
```

## 快速开始
运行序列模型（默认 1 epoch）：
```bash
python cnn_keras.py --mode seq --epochs 1 --batch 128
```
运行功能模型：
```bash
python cnn_keras.py --mode func --epochs 1 --batch 128
```
两个模型都跑：
```bash
python cnn_keras.py --mode both --epochs 1 --batch 128
```

## 核心实现概览
- 数据：`mnist.py` 封装加载与预处理，提供展平 (784) 和 one-hot 标签。
- 模型：
  - `build_sequential_model`：两层 Conv2D + MaxPool + Dense(128) + Dense(10, softmax)，Adam(1e-3)
  - `build_functional_model`：同结构，RMSprop(1e-3)
- 训练：`train_and_evaluate` 使用 `fit`，默认 `epochs=1, batch_size=128, val_split=0.2`
- 评估：`evaluate` 输出 loss / accuracy
- 预测：前 9 张测试图像对比真值与预测
- 错误样本：收集全部预测，绘制前 9 张错误样本
- 可视化：
  - 卷积核权重：`plot_conv_weights`
  - 卷积层输出：`plot_conv_output`（方法1：K.function；方法2：Model.predict）
  - 原始图像：`plot_image`
- 模型保存 / 加载：`save_and_load` 使用 `model.save` / `load_model`

## 输出产物
运行后生成：
- `seq_predictions.png` / `func_predictions.png`：前 9 张预测展示
- `seq_errors.png` / `func_errors.png`：错误样本展示
- `seq_conv1_weights.png` / `seq_conv2_weights.png`：卷积核权重
- `seq_conv1_output.png` / `seq_conv2_output.png`：卷积层输出
- `seq_image1.png`：原始输入示例
（功能模型对应前缀 `func_...`）

## 关键代码片段
- 构建序列模型
```python
model = Sequential()
model.add(InputLayer(input_shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))
model.add(Conv2D(5, 1, 16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(5, 1, 36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
```

- 功能模型
```python
inputs = Input(shape=(img_size_flat,))
net = Reshape(img_shape_full)(inputs)
net = Conv2D(5, 1, 16, padding='same', activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(2, 2)(net)
net = Conv2D(5, 1, 36, padding='same', activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(2, 2)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
outputs = Dense(num_classes, activation='softmax')(net)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
```

## 注意事项
- 默认只跑 1 个 epoch 以便快速验证，如需更高精度可提高 `--epochs`。
- 首次运行会自动下载 MNIST（需网络），下载后缓存于 `~/.keras/datasets/mnist.npz`。
- 可视化依赖 matplotlib，已在 `requirements.txt` 中声明。

## 常见问题
- **ImportError: tensorflow/keras 未找到**：`pip install tensorflow` 或直接安装 `requirements.txt`。
- **下载 MNIST 失败**：检查网络，或手动将 `mnist.npz` 放入 `~/.keras/datasets/`。
- **训练慢**：使用 GPU 版 TensorFlow (`pip install tensorflow-gpu`)，或减小 `--epochs`。

## 参考
- Keras 官方文档: https://keras.io
- TensorFlow 教程: https://www.tensorflow.org

**状态**：完成
