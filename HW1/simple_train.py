"""
二手车价格预测 - 极简版本
只用最基础的方法，容易理解
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("开始训练...")

# ============================================================================
# 步骤1：加载数据
# ============================================================================
train_data = pd.read_csv('used_car/used_car_train.csv')
test_data = pd.read_csv('used_car/used_car_test.csv')

print(f"训练数据: {len(train_data)} 条")

# ============================================================================
# 步骤2：数据清洗和准备
# ============================================================================
# 替换 '-' 为 NaN
train_data = train_data.replace('-', np.nan)
test_data = test_data.replace('-', np.nan)

# 去掉不需要的列
feature_cols = [col for col in train_data.columns 
                if col not in ['SaleID', 'price', 'name']]

# 提取特征和价格
X = train_data[feature_cols].values.astype(float)
y = train_data['price'].values.astype(float)
X_test = test_data[feature_cols].values.astype(float)

# 处理缺失值：替换成0
X = np.nan_to_num(X, 0)
X_test = np.nan_to_num(X_test, 0)

# 处理异常价格
y = np.clip(y, 1, None)  # 价格至少为1

print(f"特征数量: {X.shape[1]}")
print(f"价格范围: {y.min():.0f} ~ {y.max():.0f}")

# ============================================================================
# 步骤3：数据标准化
# ============================================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
X_test = scaler_X.transform(X_test)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print("数据标准化完成")

# ============================================================================
# 步骤4：分割训练集和验证集
# ============================================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.FloatTensor(y_valid)
X_test = torch.FloatTensor(X_test)

print(f"训练集: {len(X_train)} 条")
print(f"验证集: {len(X_valid)} 条")

# ============================================================================
# 步骤5：定义简单的神经网络
# ============================================================================
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

model = SimpleModel(X_train.shape[1])
print("模型创建完成")

# ============================================================================
# 步骤6：训练
# ============================================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100
BATCH_SIZE = 64

print(f"\n开始训练（{EPOCHS}轮）...")
print("-" * 50)

best_loss = float('inf')

for epoch in range(EPOCHS):
    # 训练模式
    model.train()
    
    # 分批训练
    for i in range(0, len(X_train), BATCH_SIZE):
        batch_X = X_train[i:i+BATCH_SIZE]
        batch_y = y_train[i:i+BATCH_SIZE]
        
        # 前向传播
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        valid_pred = model(X_valid)
        valid_loss = criterion(valid_pred, y_valid).item()
    
    # 保存最佳模型
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'simple_model.pth')
        best_epoch = epoch + 1
    
    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - 验证损失: {valid_loss:.4f}")

print("-" * 50)
print(f"训练完成！最佳损失: {best_loss:.4f} (Epoch {best_epoch})")

# ============================================================================
# 步骤7：预测
# ============================================================================
print("\n开始预测...")

# 加载最佳模型
model.load_state_dict(torch.load('simple_model.pth'))
model.eval()

# 预测
with torch.no_grad():
    predictions = model(X_test).numpy()

# 反向标准化（转回原始价格）
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

# 确保价格为正
predictions = np.clip(predictions, 0, None)

print(f"预测完成！")
print(f"预测价格范围: {predictions.min():.0f} ~ {predictions.max():.0f}")

# ============================================================================
# 步骤8：保存结果
# ============================================================================
result = pd.DataFrame({
    'SaleID': test_data['SaleID'],
    'price': predictions
})

result.to_csv('submission.csv', index=False)
print("\n结果已保存到 submission.csv")
print("=" * 50)
print("全部完成！✓")
