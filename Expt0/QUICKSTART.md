# 快速开始指南

## 1. 环境配置

### 1.1 创建虚拟环境（推荐）

```bash
# 使用venv创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 1.2 安装依赖包

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装PyTorch (CPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或安装PyTorch (GPU版本 - 需要CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 1.3 验证安装

```bash
# 检查PyTorch
python -c "import torch; print('PyTorch版本:', torch.__version__); print('GPU可用:', torch.cuda.is_available())"

# 检查TensorFlow
python -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__)"

# 检查Keras
python -c "import keras; print('Keras版本:', keras.__version__)"
```

## 2. 快速运行

### 方式一：运行单个模块

#### 肺炎图像识别模块
```bash
cd pneumonia_recognition
python pneumonia_recognition.py
```
预计运行时间：3-5分钟

#### 药物评价情感分析 (Keras版)
```bash
cd drug_sentiment_analysis
python sentiment_analysis.py
```
预计运行时间：2-4分钟

#### 药物评价情感分析 (PyTorch版)
```bash
cd drug_sentiment_analysis
python sentiment_analysis_pytorch.py
```
预计运行时间：3-5分钟

### 方式二：运行所有模块

```bash
# 在项目根目录创建运行脚本
python run_all_experiments.py
```

## 3. 输出文件说明

### 肺炎识别模块输出
```
pneumonia_recognition/models/
├── autoencoder.pth                    # 自编码器模型（~2MB）
├── cnn_classifier.pth                 # CNN分类模型（~3MB）
├── autoencoder_results.png            # 去噪效果对比图
└── cnn_training_results.png           # 训练过程曲线图
```

### 药物评价模块输出
```
drug_sentiment_analysis/models/
├── sentiment_model.h5                 # Keras模型（~5MB）
├── sentiment_pytorch.pth              # PyTorch模型（~2MB）
├── training_history.png               # Keras训练曲线
├── pytorch_training_results.png       # PyTorch训练曲线
├── framework_comparison.txt           # Keras vs PyTorch对比
└── pytorch_vs_keras_comparison.txt    # 详细框架对比
```

## 4. 修改超参数

### 肺炎识别模块参数调整

编辑 `pneumonia_recognition/pneumonia_recognition.py`：

```python
# 修改这些参数
TRAIN_BATCH_SIZE = 16      # 训练批大小
TEST_BATCH_SIZE = 66       # 测试批大小
EPOCHS = 20                # 训练轮数
LR = 1e-3                  # 学习率
NOISE_FACTOR = 0.5         # 噪声强度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### 药物评价模块参数调整

编辑 `drug_sentiment_analysis/sentiment_analysis.py` 或 `sentiment_analysis_pytorch.py`：

```python
# Keras版本参数
vocab_size = 5000          # 词汇表大小
max_length = 100           # 最大序列长度
batch_size = 32            # 批大小
epochs = 15                # 训练轮数

# PyTorch版本参数
VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
```

## 5. 使用自己的数据

### 肺炎识别 - 使用真实图像数据

```python
# 修改 pneumonia_recognition.py 中的数据加载部分
# 替换这段代码：
train_dataset = PneumoniaDataset(...)

# 使用真实数据：
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(
    'path/to/train/data',
    transform=train_transform
)
```

### 药物评价 - 使用真实评论数据

```python
# 修改 sentiment_analysis.py 中的数据加载部分
import pandas as pd

# 读取CSV文件
df_train = pd.read_csv('drug_reviews_train.csv')
df_test = pd.read_csv('drug_reviews_test.csv')

# 提取评论和评分
train_reviews = df_train['review'].values
train_ratings = df_train['rating'].values
# ... 继续处理
```

## 6. GPU加速

### 检查GPU是否可用

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # 显示GPU型号
print(torch.cuda.get_device_properties(0))  # 显示GPU详细信息
```

### CUDA安装指南

1. 下载CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
2. 下载cuDNN: https://developer.nvidia.com/cudnn
3. 按照官方说明安装
4. 验证安装：
```bash
nvcc --version
```

## 7. 常见问题解决

### 问题1：无法导入PyTorch
**解决**：
```bash
# 确保虚拟环境已激活
# 重新安装PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 问题2：内存不足
**解决**：
- 减小batch_size
- 减少EPOCHS
- 使用更小的模型

```python
TRAIN_BATCH_SIZE = 8   # 从16改为8
EPOCHS = 10            # 从20改为10
```

### 问题3：数据加载缓慢
**解决**：
- 使用num_workers加速
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4  # 添加此行
)
```

### 问题4：GPU显存不足
**解决**：
```python
# 清除缓存
torch.cuda.empty_cache()

# 或使用CPU
DEVICE = torch.device('cpu')
```

## 8. 性能优化建议

### 提高准确率

1. **增加训练轮数**
```python
EPOCHS = 50  # 增加到50个轮次
```

2. **调整学习率**
```python
LR = 1e-4  # 降低学习率以获得更稳定的收敛
```

3. **数据增强**（针对图像）
```python
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
```

4. **增加模型复杂度**
```python
# 在CNN中添加更多层或更多过滤器
self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
```

### 加快训练速度

1. **使用更大的batch size**
```python
TRAIN_BATCH_SIZE = 64  # 增加批大小
```

2. **启用GPU加速**
```python
DEVICE = torch.device("cuda")
```

3. **使用混合精度训练**（高级）
```python
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

## 9. 实验报告生成

项目会自动生成以下报告文件：

### 肺炎识别报告
- `autoencoder_results.png` - 自编码器效果演示
- `cnn_training_results.png` - CNN训练曲线

### 药物评价报告
- `training_history.png` - 训练历史
- `pytorch_training_results.png` - PyTorch训练结果
- `framework_comparison.txt` - 框架对比分析
- `pytorch_vs_keras_comparison.txt` - 详细技术对比

## 10. 下一步建议

### 进阶学习

1. **迁移学习**：使用预训练模型
```python
from torchvision.models import resnet18
model = resnet18(pretrained=True)
```

2. **模型集成**：组合多个模型
3. **超参数优化**：使用Optuna或Ray Tune
4. **部署上线**：使用Flask/FastAPI

### 扩展功能

1. **Web界面**：使用Flask + HTML/CSS/JS
2. **移动应用**：转换为TFLite或ONNX格式
3. **实时预测**：构建REST API

## 联系与支持

- 遇到问题请查看README.md
- 查看代码中的详细注释
- 参考官方文档：
  - PyTorch: https://pytorch.org
  - Keras: https://keras.io
  - TensorFlow: https://tensorflow.org

---

**最后更新**: 2024-12-12  
**难度等级**: ⭐⭐⭐☆☆  
**预计耗时**: 30-60分钟  
**推荐配置**: GPU + 8GB RAM
