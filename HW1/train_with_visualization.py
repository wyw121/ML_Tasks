"""
二手车价格预测 - 完整可视化版本
包含训练过程的多维度可视化分析
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 60)
print("二手车价格预测 - 训练可视化系统")
print("=" * 60)

# ============================================================================
# 训练历史记录类
# ============================================================================
class TrainingHistory:
    """记录训练过程中的各项指标"""
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
        self.learning_rates = []
        self.gradients = []
        self.epoch_times = []
        
    def add_epoch(self, train_loss, valid_loss, lr, gradient_norm, epoch_time):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.learning_rates.append(lr)
        self.gradients.append(gradient_norm)
        self.epoch_times.append(epoch_time)

# ============================================================================
# 步骤1：加载数据
# ============================================================================
print("\n[1/9] 加载数据...")
train_data = pd.read_csv('used_car/used_car_train.csv')
test_data = pd.read_csv('used_car/used_car_test.csv')
print(f"✓ 训练数据: {len(train_data)} 条")
print(f"✓ 测试数据: {len(test_data)} 条")

# ============================================================================
# 步骤2：数据清洗和准备
# ============================================================================
print("\n[2/9] 数据清洗...")
train_data = train_data.replace('-', np.nan)
test_data = test_data.replace('-', np.nan)

feature_cols = [col for col in train_data.columns 
                if col not in ['SaleID', 'price', 'name']]

X = train_data[feature_cols].values.astype(float)
y = train_data['price'].values.astype(float)
X_test = test_data[feature_cols].values.astype(float)

X = np.nan_to_num(X, 0)
X_test = np.nan_to_num(X_test, 0)
y = np.clip(y, 1, None)

print(f"✓ 特征数量: {X.shape[1]}")
print(f"✓ 价格范围: {y.min():.0f} ~ {y.max():.0f}")

# ============================================================================
# 步骤3：数据标准化
# ============================================================================
print("\n[3/9] 数据标准化...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
X_test = scaler_X.transform(X_test)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
print("✓ 标准化完成")

# ============================================================================
# 步骤4：分割训练集和验证集
# ============================================================================
print("\n[4/9] 分割数据集...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)

# 保存原始价格用于可视化
_, _, y_train_orig, y_valid_orig = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.FloatTensor(y_valid)
X_test = torch.FloatTensor(X_test)

print(f"✓ 训练集: {len(X_train)} 条")
print(f"✓ 验证集: {len(X_valid)} 条")

# ============================================================================
# 步骤5：定义神经网络
# ============================================================================
print("\n[5/9] 构建神经网络...")

class ImprovedModel(nn.Module):
    def __init__(self, input_size):
        super(ImprovedModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

model = ImprovedModel(X_train.shape[1])
print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 步骤6：训练配置
# ============================================================================
print("\n[6/9] 配置训练参数...")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

EPOCHS = 100
BATCH_SIZE = 64

history = TrainingHistory()
print("✓ 训练配置完成")

# ============================================================================
# 步骤7：训练循环
# ============================================================================
print(f"\n[7/9] 开始训练 ({EPOCHS} 轮)...")
print("-" * 60)

best_loss = float('inf')
best_epoch = 0

import time

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # 训练模式
    model.train()
    train_losses = []
    
    for i in range(0, len(X_train), BATCH_SIZE):
        batch_X = X_train[i:i+BATCH_SIZE]
        batch_y = y_train[i:i+BATCH_SIZE]
        
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
    # 计算梯度范数
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 验证
    model.eval()
    with torch.no_grad():
        valid_pred = model(X_valid)
        valid_loss = criterion(valid_pred, y_valid).item()
    
    # 更新学习率
    scheduler.step(valid_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录历史
    epoch_time = time.time() - epoch_start
    history.add_epoch(
        np.mean(train_losses), 
        valid_loss, 
        current_lr,
        total_norm,
        epoch_time
    )
    
    # 保存最佳模型
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'best_model_visual.pth')
        best_epoch = epoch + 1
    
    # 打印进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {np.mean(train_losses):.4f} | "
              f"Valid Loss: {valid_loss:.4f} | "
              f"LR: {current_lr:.6f}")

print("-" * 60)
print(f"✓ 训练完成！最佳验证损失: {best_loss:.4f} (Epoch {best_epoch})")

# ============================================================================
# 步骤8：生成预测用于可视化
# ============================================================================
print(f"\n[8/9] 生成预测结果...")

model.load_state_dict(torch.load('best_model_visual.pth'))
model.eval()

with torch.no_grad():
    # 训练集预测
    train_pred = model(X_train).numpy()
    train_pred_orig = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    
    # 验证集预测
    valid_pred = model(X_valid).numpy()
    valid_pred_orig = scaler_y.inverse_transform(valid_pred.reshape(-1, 1)).flatten()
    
    # 测试集预测
    test_pred = model(X_test).numpy()
    test_pred_orig = scaler_y.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    test_pred_orig = np.clip(test_pred_orig, 0, None)

# 计算评估指标
train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
valid_mae = mean_absolute_error(y_valid_orig, valid_pred_orig)
train_r2 = r2_score(y_train_orig, train_pred_orig)
valid_r2 = r2_score(y_valid_orig, valid_pred_orig)

print(f"✓ 训练集 MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
print(f"✓ 验证集 MAE: {valid_mae:.2f}, R²: {valid_r2:.4f}")

# ============================================================================
# 步骤9：创建可视化
# ============================================================================
print(f"\n[9/9] 生成可视化图表...")

# 创建大图
fig = plt.figure(figsize=(20, 12))
fig.suptitle('二手车价格预测模型 - 训练过程可视化分析', 
             fontsize=20, fontweight='bold', y=0.995)

# ============ 图1: 损失函数曲线 ============
ax1 = plt.subplot(3, 3, 1)
epochs_range = range(1, len(history.train_losses) + 1)
ax1.plot(epochs_range, history.train_losses, label='训练损失', linewidth=2, alpha=0.8)
ax1.plot(epochs_range, history.valid_losses, label='验证损失', linewidth=2, alpha=0.8)
ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'最佳模型 (Epoch {best_epoch})')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('损失 (MSE)', fontsize=11)
ax1.set_title('① 训练与验证损失曲线', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============ 图2: 学习率变化 ============
ax2 = plt.subplot(3, 3, 2)
ax2.plot(epochs_range, history.learning_rates, color='orange', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('学习率', fontsize=11)
ax2.set_title('② 学习率动态调整', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# ============ 图3: 梯度范数 ============
ax3 = plt.subplot(3, 3, 3)
ax3.plot(epochs_range, history.gradients, color='green', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('梯度范数', fontsize=11)
ax3.set_title('③ 梯度变化监控', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ============ 图4: 预测vs真实 (训练集) ============
ax4 = plt.subplot(3, 3, 4)
sample_size = min(2000, len(y_train_orig))
indices = np.random.choice(len(y_train_orig), sample_size, replace=False)
ax4.scatter(y_train_orig[indices], train_pred_orig[indices], 
           alpha=0.3, s=10, label='训练集样本')
max_val = max(y_train_orig.max(), train_pred_orig.max())
ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想预测线')
ax4.set_xlabel('真实价格', fontsize=11)
ax4.set_ylabel('预测价格', fontsize=11)
ax4.set_title(f'④ 训练集预测效果 (R²={train_r2:.4f})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============ 图5: 预测vs真实 (验证集) ============
ax5 = plt.subplot(3, 3, 5)
sample_size = min(2000, len(y_valid_orig))
indices = np.random.choice(len(y_valid_orig), sample_size, replace=False)
ax5.scatter(y_valid_orig[indices], valid_pred_orig[indices], 
           alpha=0.3, s=10, color='orange', label='验证集样本')
max_val = max(y_valid_orig.max(), valid_pred_orig.max())
ax5.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想预测线')
ax5.set_xlabel('真实价格', fontsize=11)
ax5.set_ylabel('预测价格', fontsize=11)
ax5.set_title(f'⑤ 验证集预测效果 (R²={valid_r2:.4f})', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============ 图6: 残差分布 ============
ax6 = plt.subplot(3, 3, 6)
residuals = y_valid_orig - valid_pred_orig
ax6.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('残差 (真实值 - 预测值)', fontsize=11)
ax6.set_ylabel('频数', fontsize=11)
ax6.set_title(f'⑥ 残差分布 (验证集)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# ============ 图7: 残差散点图 ============
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(valid_pred_orig, residuals, alpha=0.3, s=10)
ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('预测价格', fontsize=11)
ax7.set_ylabel('残差', fontsize=11)
ax7.set_title('⑦ 残差vs预测值', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# ============ 图8: 损失分布对比 ============
ax8 = plt.subplot(3, 3, 8)
train_errors = np.abs(y_train_orig - train_pred_orig)
valid_errors = np.abs(y_valid_orig - valid_pred_orig)
ax8.hist([train_errors, valid_errors], bins=50, 
         label=['训练集', '验证集'], alpha=0.7, color=['blue', 'orange'])
ax8.set_xlabel('绝对误差', fontsize=11)
ax8.set_ylabel('频数', fontsize=11)
ax8.set_title('⑧ 预测误差分布对比', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# ============ 图9: 训练时间统计 ============
ax9 = plt.subplot(3, 3, 9)
cumulative_time = np.cumsum(history.epoch_times)
ax9.plot(epochs_range, cumulative_time, color='purple', linewidth=2)
ax9.fill_between(epochs_range, cumulative_time, alpha=0.3, color='purple')
ax9.set_xlabel('Epoch', fontsize=11)
ax9.set_ylabel('累计时间 (秒)', fontsize=11)
ax9.set_title(f'⑨ 训练耗时 (总计: {cumulative_time[-1]:.1f}秒)', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# 保存图片
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'training_visualization_{timestamp}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ 可视化图表已保存: {filename}")

plt.show()

# ============================================================================
# 保存提交文件
# ============================================================================
result = pd.DataFrame({
    'SaleID': test_data['SaleID'],
    'price': test_pred_orig
})
result.to_csv('submission.csv', index=False)
print(f"✓ 预测结果已保存: submission.csv")

# ============================================================================
# 打印总结报告
# ============================================================================
print("\n" + "=" * 60)
print("训练总结报告")
print("=" * 60)
print(f"最佳验证损失: {best_loss:.4f} (Epoch {best_epoch})")
print(f"训练集 MAE: {train_mae:.2f} | R²: {train_r2:.4f}")
print(f"验证集 MAE: {valid_mae:.2f} | R²: {valid_r2:.4f}")
print(f"总训练时间: {cumulative_time[-1]:.1f} 秒")
print(f"平均每轮耗时: {np.mean(history.epoch_times):.2f} 秒")
print(f"预测价格范围: {test_pred_orig.min():.0f} ~ {test_pred_orig.max():.0f}")
print("=" * 60)
print("全部完成！✓")
print("=" * 60)
