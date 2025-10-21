"""
🎓 互动学习版 - 二手车价格预测
一步一步运行，理解每个概念

使用方法：
1. 取消注释想要运行的部分（删除 # 号）
2. 运行代码，观察输出
3. 阅读解释，理解概念
4. 继续下一部分
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🎓 欢迎来到深度学习互动教程！")
print("请按步骤取消注释并运行代码")

# ============================================================================
# 🔍 第一步：了解数据
# 取消下面的注释来运行这部分
# ============================================================================

print("\n🔍 第一步：了解数据")
train_data = pd.read_csv("used_car/used_car_train.csv")

# 查看数据基本信息
print(f"数据形状: {train_data.shape}")
print(f"列名: {list(train_data.columns)}")
print("\n前5行数据:")
print(train_data.head())

# 查看数据类型
print(f"\n数据类型:")
print(train_data.dtypes)

# 查看缺失值情况
print(f"\n缺失值统计:")
missing = train_data.isnull().sum()
print(missing[missing > 0])

# 💡 解释：
# - shape: (行数, 列数)
# - head(): 显示前几行数据
# - dtypes: 每列的数据类型
# - isnull().sum(): 统计每列的缺失值数量

# ============================================================================
# 🧹 第二步：数据清洗实验
# 取消下面的注释来运行这部分
# ============================================================================

# print("\n🧹 第二步：数据清洗实验")

# # 创建一个小例子来理解数据清洗
# sample_data = pd.DataFrame({
#     'feature1': [1, 2, '-', 4, 5],
#     'feature2': [10, '-', 30, 40, 50],
#     'price': [1000, 2000, 1500, 3000, 2500]
# })

# print("原始数据:")
# print(sample_data)

# # 替换 '-' 为 NaN
# sample_cleaned = sample_data.replace('-', np.nan)
# print("\n替换'-'为NaN后:")
# print(sample_cleaned)

# # 转换数据类型
# numeric_data = sample_cleaned.astype(float, errors='ignore')
# print(f"\n转换为数值类型后:")
# print(numeric_data)

# # 处理缺失值
# filled_data = numeric_data.fillna(0)
# print(f"\n用0填充缺失值后:")
# print(filled_data)

# 💡 解释：
# - replace(): 替换特定值
# - astype(): 转换数据类型
# - fillna(): 填充缺失值

# ============================================================================
# 📏 第三步：标准化实验
# 取消下面的注释来运行这部分
# ============================================================================

# print("\n📏 第三步：标准化实验")

# # 创建示例数据（不同尺度）
# data = np.array([
#     [1, 1000],      # 年龄, 价格
#     [5, 5000],
#     [10, 10000],
#     [2, 2000]
# ])

# print("原始数据（年龄 vs 价格）:")
# print(data)
# print(f"均值: {data.mean(axis=0)}")
# print(f"标准差: {data.std(axis=0)}")

# # 标准化
# scaler = StandardScaler()
# standardized = scaler.fit_transform(data)

# print(f"\n标准化后:")
# print(standardized)
# print(f"均值: {standardized.mean(axis=0)}")
# print(f"标准差: {standardized.std(axis=0)}")

# 💡 解释：
# - 标准化让所有特征都变成均值0、标准差1
# - 这样不同尺度的特征就在同一量级了

# ============================================================================
# ⚡ 第四步：张量实验
# 取消下面的注释来运行这部分
# ============================================================================

# print("\n⚡ 第四步：张量实验")

# # NumPy数组
# np_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
# print(f"NumPy数组: {np_array}")
# print(f"类型: {type(np_array)}")

# # 转换为PyTorch张量
# tensor = torch.FloatTensor(np_array)
# print(f"\nPyTorch张量: {tensor}")
# print(f"类型: {type(tensor)}")
# print(f"形状: {tensor.shape}")

# # 张量运算
# tensor_squared = tensor ** 2
# print(f"\n张量平方: {tensor_squared}")

# # 转换回NumPy
# back_to_numpy = tensor.numpy()
# print(f"\n转换回NumPy: {back_to_numpy}")

# 💡 解释：
# - 张量是PyTorch的基本数据结构
# - 支持GPU加速和自动求导
# - 可以与NumPy互相转换

# ============================================================================
# 🏗️ 第五步：简单网络实验
# 取消下面的注释来运行这部分
# ============================================================================

# print("\n🏗️ 第五步：简单网络实验")

# # 定义一个超简单的网络
# class TinyModel(nn.Module):
#     def __init__(self):
#         super(TinyModel, self).__init__()
#         self.layer = nn.Linear(2, 1)  # 2个输入 -> 1个输出

#     def forward(self, x):
#         return self.layer(x)

# # 创建模型
# tiny_model = TinyModel()
# print(f"模型结构:")
# print(tiny_model)

# # 创建示例输入
# sample_input = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
# print(f"\n输入数据:")
# print(sample_input)

# # 前向传播
# output = tiny_model(sample_input)
# print(f"\n输出结果:")
# print(output)

# # 查看参数
# print(f"\n网络参数:")
# for name, param in tiny_model.named_parameters():
#     print(f"{name}: {param.data}")

# 💡 解释：
# - nn.Linear(): 全连接层，进行线性变换
# - forward(): 定义数据流向
# - parameters(): 网络的可学习参数

# ============================================================================
# 🎯 第六步：损失函数实验
# 取消下面的注释来运行这部分
# ============================================================================

# print("\n🎯 第六步：损失函数实验")

# # 创建预测值和真实值
# predictions = torch.FloatTensor([2.5, 3.1, 1.8])
# true_values = torch.FloatTensor([2.0, 3.0, 2.0])

# print(f"预测值: {predictions}")
# print(f"真实值: {true_values}")

# # 计算MSE损失
# mse_loss = nn.MSELoss()
# loss = mse_loss(predictions, true_values)

# print(f"\nMSE损失: {loss.item():.4f}")

# # 手动计算验证
# manual_mse = ((predictions - true_values) ** 2).mean()
# print(f"手动计算MSE: {manual_mse.item():.4f}")

# 💡 解释：
# - MSE = (预测值-真实值)²的平均值
# - 损失越小，模型预测越准确

# ============================================================================
# 🔄 第七步：简单训练循环
# 取消下面的注释来运行这部分
# ============================================================================

# print("\n🔄 第七步：简单训练循环")

# # 创建简单的训练数据
# X = torch.FloatTensor([[1], [2], [3], [4]])  # 输入
# y = torch.FloatTensor([2, 4, 6, 8])          # 目标：y = 2*x

# print(f"训练数据:")
# print(f"X: {X.flatten()}")
# print(f"y: {y}")

# # 创建简单模型
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         return self.linear(x).squeeze()

# model = LinearModel()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# print(f"\n开始训练...")
# for epoch in range(100):
#     # 前向传播
#     pred = model(X)
#     loss = criterion(pred, y)

#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # 每20轮打印一次
#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# # 测试训练结果
# test_x = torch.FloatTensor([[5]])
# pred_y = model(test_x)
# print(f"\n测试：x=5时，预测y={pred_y.item():.2f}，真实应该是10")

# 💡 解释：
# - 这是完整的训练循环：前向→计算损失→反向→更新参数
# - 经过训练，模型学会了 y=2*x 的关系

print("\n🎉 教程准备完成！")
print("现在你可以：")
print("1. 逐步取消注释并运行各个部分")
print("2. 观察输出，理解每个概念")
print("3. 尝试修改参数，看看会发生什么")
print("4. 最后回到原始代码，应该就能看懂了！")
