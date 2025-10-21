"""
二手车价格预测 - 详细注释版
每一步都有详细解释，帮助理解深度学习流程
"""

import warnings

import numpy as np  # 数值计算库，用于数组操作、数学运算
import pandas as pd  # 数据处理库，用于读取CSV、数据清洗
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from sklearn.model_selection import train_test_split  # 数据分割工具
from sklearn.preprocessing import StandardScaler  # 数据标准化工具
from torch.utils.data import DataLoader, Dataset  # 数据加载工具

warnings.filterwarnings("ignore")  # 忽略警告信息

print("🚀 开始训练二手车价格预测模型...")

# ============================================================================
# 📁 步骤1：数据加载
# ============================================================================
print("\n📁 正在加载数据...")
train_data = pd.read_csv("used_car/used_car_train.csv")  # 读取训练数据
test_data = pd.read_csv("used_car/used_car_test.csv")  # 读取测试数据

print(f"✅ 训练数据加载完成: {len(train_data)} 条记录")
print(f"✅ 测试数据加载完成: {len(test_data)} 条记录")
print(f"📊 数据列数: {len(train_data.columns)} 列")

# ============================================================================
# 🧹 步骤2：数据清洗
# ============================================================================
print("\n🧹 开始数据清洗...")

# 📌 处理缺失值标记
# 原始数据中用 '-' 表示缺失值，我们需要转换成标准的 NaN
train_data = train_data.replace("-", np.nan)
test_data = test_data.replace("-", np.nan)
print("✅ 已将 '-' 转换为 NaN（标准缺失值标记）")

# 📌 选择特征列
# 去掉不用于预测的列：SaleID（ID号）、price（目标值）、name（车名）
feature_cols = [
    col for col in train_data.columns if col not in ["SaleID", "price", "name"]
]
print(f"✅ 选择了 {len(feature_cols)} 个特征用于训练")

# 📌 提取特征和目标值
X = train_data[feature_cols].values.astype(float)  # 特征矩阵（输入）
y = train_data["price"].values.astype(float)  # 价格向量（目标输出）
X_test = test_data[feature_cols].values.astype(float)  # 测试集特征

# 📌 处理缺失值
# np.nan_to_num() 将 NaN 替换为 0
# 这是一种简单的缺失值处理方法
X = np.nan_to_num(X, 0)
X_test = np.nan_to_num(X_test, 0)
print("✅ 缺失值已用 0 填充")

# 📌 处理异常价格
# np.clip() 确保价格至少为 1，避免负价格或0价格
y = np.clip(y, 1, None)
print("✅ 价格数据已清洗（最小值设为1）")

print(f"📈 特征维度: {X.shape}")  # (样本数, 特征数)
print(f"💰 价格范围: {y.min():.0f} ~ {y.max():.0f} 元")

# ============================================================================
# 📏 步骤3：数据标准化
# ============================================================================
print("\n📏 开始数据标准化...")

# 🔧 创建标准化器
# StandardScaler 会将数据转换为均值0、标准差1的分布
scaler_X = StandardScaler()  # 特征标准化器
scaler_y = StandardScaler()  # 目标值标准化器

# 🔧 标准化特征
X = scaler_X.fit_transform(X)  # 训练集：计算参数并转换
X_test = scaler_X.transform(X_test)  # 测试集：用训练集的参数转换

# 🔧 标准化目标值
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
# reshape(-1, 1): 转换为列向量，因为 StandardScaler 需要2D输入
# flatten(): 转换回1D数组

print("✅ 数据标准化完成")
print("💡 标准化的作用：让不同尺度的特征在同一量级，提升训练效果")

# ============================================================================
# ✂️ 步骤4：数据分割
# ============================================================================
print("\n✂️ 分割训练集和验证集...")

# train_test_split 随机分割数据
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,  # 要分割的数据
    test_size=0.2,  # 20% 作为验证集
    random_state=42,  # 随机种子，保证结果可重复
)

# 🔄 转换为 PyTorch 张量
# PyTorch 需要张量格式，而不是 numpy 数组
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.FloatTensor(y_valid)
X_test = torch.FloatTensor(X_test)

print(f"✅ 训练集: {len(X_train)} 条")
print(f"✅ 验证集: {len(X_valid)} 条")
print("💡 验证集用于评估模型性能，避免过拟合")

# ============================================================================
# 🧠 步骤5：定义神经网络模型
# ============================================================================
print("\n🧠 构建神经网络模型...")


class SimpleModel(nn.Module):
    """
    简单的全连接神经网络

    网络结构：
    输入层 -> 隐藏层1(64神经元) -> ReLU -> 隐藏层2(32神经元) -> ReLU -> 输出层(1神经元)
    """

    def __init__(self, input_size):
        super(SimpleModel, self).__init__()

        # nn.Sequential: 按顺序堆叠网络层
        self.net = nn.Sequential(
            # 第一层：输入 -> 64个神经元
            nn.Linear(input_size, 64),  # 全连接层（线性变换）
            nn.ReLU(),  # ReLU激活函数（引入非线性）
            # 第二层：64 -> 32个神经元
            nn.Linear(64, 32),
            nn.ReLU(),
            # 输出层：32 -> 1个神经元（预测价格）
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """前向传播：数据从输入到输出的计算过程"""
        return self.net(x).squeeze()  # squeeze() 去掉多余的维度


# 🏗️ 创建模型实例
model = SimpleModel(X_train.shape[1])  # 输入维度 = 特征数量
print(f"✅ 模型创建完成")
print(f"📊 输入维度: {X_train.shape[1]}")
print(f"🏗️ 网络结构: {X_train.shape[1]} -> 64 -> 32 -> 1")

# ============================================================================
# 🎯 步骤6：设置训练参数
# ============================================================================
print("\n🎯 设置训练参数...")

# 🎯 损失函数：衡量预测值与真实值的差距
criterion = nn.MSELoss()  # 均方误差 = (预测值-真实值)²的平均值

# 🎯 优化器：根据梯度更新网络参数
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001  # 要优化的参数  # 学习率：参数更新的步长
)

# 🎯 训练超参数
EPOCHS = 100  # 训练轮数：完整遍历数据集的次数
BATCH_SIZE = 64  # 批大小：每次处理的样本数

print(f"✅ 损失函数: MSE（均方误差）")
print(f"✅ 优化器: Adam（学习率: {0.001}）")
print(f"✅ 训练轮数: {EPOCHS}")
print(f"✅ 批大小: {BATCH_SIZE}")

# ============================================================================
# 🚀 步骤7：开始训练
# ============================================================================
print(f"\n🚀 开始训练模型...")
print("=" * 60)

best_loss = float("inf")  # 记录最佳损失值
best_epoch = 0

for epoch in range(EPOCHS):
    # 🔄 设置为训练模式
    model.train()

    # 📦 分批处理训练数据
    # 分批的好处：节省内存，更稳定的梯度
    for i in range(0, len(X_train), BATCH_SIZE):
        # 获取当前批次的数据
        batch_X = X_train[i : i + BATCH_SIZE]
        batch_y = y_train[i : i + BATCH_SIZE]

        # 🔮 前向传播：输入数据，得到预测结果
        pred = model(batch_X)

        # 📊 计算损失：预测值与真实值的差距
        loss = criterion(pred, batch_y)

        # 🔄 反向传播：根据损失计算梯度
        optimizer.zero_grad()  # 清零上次的梯度
        loss.backward()  # 计算新梯度
        optimizer.step()  # 根据梯度更新参数

    # 📈 验证模型性能
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度，节省内存
        valid_pred = model(X_valid)
        valid_loss = criterion(valid_pred, y_valid).item()

    # 💾 保存最佳模型
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), "simple_model.pth")
        best_epoch = epoch + 1

    # 📊 每10轮打印一次进度
    if (epoch + 1) % 10 == 0:
        print(
            f"第 {epoch+1:3d}/{EPOCHS} 轮 | 验证损失: {valid_loss:.4f} | 最佳: {best_loss:.4f}"
        )

print("=" * 60)
print(f"🎉 训练完成！")
print(f"🏆 最佳验证损失: {best_loss:.4f} (第 {best_epoch} 轮)")

# ============================================================================
# 🔮 步骤8：预测测试集
# ============================================================================
print(f"\n🔮 开始预测测试集...")

# 📥 加载最佳模型
model.load_state_dict(torch.load("simple_model.pth"))
model.eval()

# 🔮 进行预测
with torch.no_grad():  # 预测时不需要计算梯度
    predictions = model(X_test).numpy()

# 🔄 反标准化：将预测结果转换回原始价格尺度
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

# 🛡️ 确保预测价格为正数
predictions = np.clip(predictions, 0, None)

print(f"✅ 预测完成！")
print(f"💰 预测价格范围: {predictions.min():.0f} ~ {predictions.max():.0f} 元")

# ============================================================================
# 💾 步骤9：保存预测结果
# ============================================================================
print(f"\n💾 保存预测结果...")

# 📋 创建结果数据框
result = pd.DataFrame(
    {"SaleID": test_data["SaleID"], "price": predictions}  # 车辆ID  # 预测价格
)

# 💾 保存到CSV文件
result.to_csv("submission.csv", index=False)
print("✅ 预测结果已保存到 submission.csv")

# 🎊 训练完成总结
print("\n" + "🎊" * 20)
print("🎉 恭喜！二手车价格预测模型训练完成！")
print("📁 生成的文件：")
print("  - simple_model.pth: 训练好的模型")
print("  - submission.csv: 预测结果")
print("🎊" * 20)

# ============================================================================
# 📚 关键概念总结
# ============================================================================
print(f"\n📚 本次训练涉及的关键概念：")
print("🔹 数据预处理：清洗、标准化")
print("🔹 神经网络：全连接层 + ReLU激活函数")
print("🔹 训练循环：前向传播 -> 计算损失 -> 反向传播 -> 更新参数")
print("🔹 验证评估：防止过拟合，选择最佳模型")
print("🔹 模型预测：加载模型，预测新数据")
