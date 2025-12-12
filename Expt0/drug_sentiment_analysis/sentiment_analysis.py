"""
基于嵌入层和LSTM的药物评价情感分析模块
使用Keras/Tensorflow框架实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 导入必需的包
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("正在导入包...")
print("Tensorflow版本:", end=" ")
import tensorflow as tf
print(tf.__version__)

# ============= 1. 创建模拟数据集 =============
print("\n" + "="*60)
print("生成药物评价模拟数据集...")
print("="*60)

np.random.seed(42)

# 生成训练数据
num_train_samples = 1000
num_test_samples = 200

# 模拟评论文本
positive_words = ['excellent', 'wonderful', 'amazing', 'great', 'fantastic', 'love', 'perfect', 'best']
negative_words = ['terrible', 'awful', 'horrible', 'bad', 'waste', 'poor', 'useless', 'worst']
neutral_words = ['okay', 'fine', 'decent', 'average', 'normal', 'regular', 'standard']

def generate_reviews(num_samples, num_positive, num_negative, num_neutral):
    """生成模拟评论"""
    reviews = []
    ratings = []
    
    # 积极评论 (7-10分)
    for _ in range(num_positive):
        review = ' '.join(np.random.choice(positive_words, size=np.random.randint(5, 15)))
        reviews.append(review)
        ratings.append(np.random.randint(7, 11))
    
    # 消极评论 (1-4分)
    for _ in range(num_negative):
        review = ' '.join(np.random.choice(negative_words, size=np.random.randint(5, 15)))
        reviews.append(review)
        ratings.append(np.random.randint(1, 5))
    
    # 中性评论 (5-6分)
    for _ in range(num_neutral):
        review = ' '.join(np.random.choice(neutral_words, size=np.random.randint(5, 15)))
        reviews.append(review)
        ratings.append(np.random.randint(5, 7))
    
    return reviews, ratings

# 生成训练集和测试集
train_reviews, train_ratings = generate_reviews(
    num_train_samples,
    num_positive=400,
    num_negative=300,
    num_neutral=300
)

test_reviews, test_ratings = generate_reviews(
    num_test_samples,
    num_positive=80,
    num_negative=60,
    num_neutral=60
)

print(f"训练集大小: {len(train_reviews)}")
print(f"测试集大小: {len(test_reviews)}")

# ============= 2. 处理数据 =============
print("\n" + "="*60)
print("处理数据...")
print("="*60)

# (1) 根据评分确定情感标签
# 1-4分：消极 (0), 5-6分：中性 (1), 7-10分：积极 (2)

def get_sentiment_label(rating):
    if rating <= 4:
        return 0  # 消极
    elif rating <= 6:
        return 1  # 中性
    else:
        return 2  # 积极

train_labels = [get_sentiment_label(r) for r in train_ratings]
test_labels = [get_sentiment_label(r) for r in test_ratings]

print(f"训练集标签分布: 消极={train_labels.count(0)}, 中性={train_labels.count(1)}, 积极={train_labels.count(2)}")
print(f"测试集标签分布: 消极={test_labels.count(0)}, 中性={test_labels.count(1)}, 积极={test_labels.count(2)}")

# (2) 序列化文本
print("\n正在序列化文本...")

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_reviews)

train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

print(f"词汇表大小: {len(tokenizer.word_index)}")
print(f"第一个序列长度: {len(train_sequences[0])}")
print(f"序列样本: {train_sequences[0][:20]}")

# (3) 填充序列
print("\n正在填充序列...")

max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

print(f"填充后训练集形状: {train_padded.shape}")
print(f"填充后测试集形状: {test_padded.shape}")

# (4) 转换标签为独热编码
print("\n正在转换为独热编码...")

num_classes = 3
train_labels_categorical = to_categorical(train_labels, num_classes=num_classes)
test_labels_categorical = to_categorical(test_labels, num_classes=num_classes)

print(f"独热编码后训练集形状: {train_labels_categorical.shape}")
print(f"独热编码后测试集形状: {test_labels_categorical.shape}")

# ============= 3. 搭建模型 =============
print("\n" + "="*60)
print("搭建LSTM模型...")
print("="*60)

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),  # 嵌入层
    Dropout(0.2),
    LSTM(64, return_sequences=True),  # LSTM层1
    Dropout(0.2),
    LSTM(32),  # LSTM层2
    Dropout(0.2),
    Dense(64, activation='relu'),  # 全连接层1
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # 输出层
])

# 编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ============= 4. 训练模型 =============
print("\n" + "="*60)
print("训练模型...")
print("="*60)

batch_size = 32
epochs = 15

history = model.fit(
    train_padded,
    train_labels_categorical,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(test_padded, test_labels_categorical),
    verbose=1
)

print("训练完成!")

# ============= 5. 评估模型 =============
print("\n" + "="*60)
print("评估模型...")
print("="*60)

test_loss, test_accuracy = model.evaluate(test_padded, test_labels_categorical, verbose=0)
print(f"测试集Loss: {test_loss:.4f}")
print(f"测试集正确率: {test_accuracy:.4f}")

# ============= 6. 保存模型 =============
print("\n正在保存模型...")
save_dir = "d:\\repositories\\ML_Tasks\\MedicalDiagnosis\\drug_sentiment_analysis\\models"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, 'sentiment_model.h5'))
print(f"模型已保存到 {save_dir}")

# ============= 7. 绘制训练结果 =============
print("\n正在生成可视化结果...")

def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    ax1.plot(history.history['loss'], 'b-', label='训练Loss', marker='o')
    ax1.plot(history.history['val_loss'], 'r-', label='验证Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('模型Loss变化')
    ax1.legend()
    ax1.grid(True)
    
    # 正确率曲线
    ax2.plot(history.history['accuracy'], 'b-', label='训练正确率', marker='o')
    ax2.plot(history.history['val_accuracy'], 'r-', label='验证正确率', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('正确率')
    ax2.set_title('模型正确率变化')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"训练结果已保存到 {save_path}")

plot_training_history(history, os.path.join(save_dir, 'training_history.png'))

# ============= 8. 情感预测示例 =============
print("\n" + "="*60)
print("进行预测示例...")
print("="*60)

# 测试几个样本
test_texts = [
    "excellent wonderful amazing fantastic",
    "terrible awful horrible bad",
    "okay fine decent average"
]

sentiment_labels = ['消极', '中性', '积极']

for text in test_texts:
    # 序列化
    seq = tokenizer.texts_to_sequences([text])
    # 填充
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    # 预测
    prediction = model.predict(padded, verbose=0)
    sentiment_idx = np.argmax(prediction[0])
    confidence = prediction[0][sentiment_idx]
    
    print(f"\n评论: {text}")
    print(f"预测情感: {sentiment_labels[sentiment_idx]} (置信度: {confidence:.4f})")
    print(f"各类别概率: 消极={prediction[0][0]:.4f}, 中性={prediction[0][1]:.4f}, 积极={prediction[0][2]:.4f}")

# ============= 9. Keras和PyTorch框架比较 =============
print("\n" + "="*60)
print("Keras和PyTorch框架特点比较")
print("="*60)

comparison = """
1. API设计和易用性:
   - Keras: 高层API，代码简洁，适合快速开发
   - PyTorch: 低层API，更灵活，学习曲线较陡

2. 动态vs静态图:
   - Keras: 支持动态和静态图（取决于后端）
   - PyTorch: 动态计算图，更直观，便于调试

3. 模型定义:
   - Keras: Sequential API和Functional API，结构清晰
   - PyTorch: 需要继承nn.Module，编写forward方法

4. 训练循环:
   - Keras: 内置fit()方法，简化训练过程
   - PyTorch: 需要手写训练循环，更灵活

5. 社区和生态:
   - Keras: Tensorflow生态完整，文档丰富
   - PyTorch: 学术界常用，社区活跃

6. 性能:
   - Keras: 依赖Tensorflow后端，性能优秀
   - PyTorch: 原生实现，性能优秀

7. 部署:
   - Keras: 支持多种部署方式（TFLite, TFServing等）
   - PyTorch: 支持ONNX等标准格式

选择建议:
- 快速原型开发: 使用Keras
- 学术研究: 使用PyTorch
- 工业应用: 根据部署需求选择
"""

print(comparison)

# 保存比较文档
with open(os.path.join(save_dir, 'framework_comparison.txt'), 'w', encoding='utf-8') as f:
    f.write(comparison)

print("\n" + "="*60)
print("药物评价情感分析模块完成!")
print("="*60)
print(f"模型保存路径: {save_dir}")
print(f"最终测试正确率: {test_accuracy:.4f}")
