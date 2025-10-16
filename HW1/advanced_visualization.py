"""
高级可视化分析
包含特征重要性、学习曲线、预测区间等进阶分析
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=" * 60)
print("高级可视化分析系统")
print("=" * 60)

# ============================================================================
# 数据准备
# ============================================================================
print("\n[1/5] 加载数据...")
train_data = pd.read_csv('used_car/used_car_train.csv')
train_data = train_data.replace('-', np.nan)

feature_cols = [col for col in train_data.columns 
                if col not in ['SaleID', 'price', 'name']]

X = train_data[feature_cols].values.astype(float)
y = train_data['price'].values.astype(float)
X = np.nan_to_num(X, 0)
y = np.clip(y, 1, None)

print(f"✓ 特征: {feature_cols}")
print(f"✓ 数据量: {len(X)}")

# ============================================================================
# 特征重要性分析 (使用随机森林)
# ============================================================================
print("\n[2/5] 计算特征重要性...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X[:5000], y[:5000])  # 用部分数据快速训练

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("✓ 特征重要性计算完成")
print("\n前10个重要特征:")
print(feature_importance.head(10))

# ============================================================================
# 学习曲线分析
# ============================================================================
print("\n[3/5] 生成学习曲线...")

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 使用较少的训练样本来加快速度
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, valid_scores = learning_curve(
    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    X_scaled[:10000], y_scaled[:10000],
    train_sizes=train_sizes,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

train_mean = -train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
valid_mean = -valid_scores.mean(axis=1)
valid_std = valid_scores.std(axis=1)

print("✓ 学习曲线生成完成")

# ============================================================================
# 价格分布分析
# ============================================================================
print("\n[4/5] 分析价格分布...")

# 加载已有的预测结果
try:
    # 重新生成预测
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 使用快速模型
    quick_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    quick_model.fit(X_train, y_train)
    predictions = quick_model.predict(X_test)
    
    # 计算预测区间
    estimators_predictions = np.array([tree.predict(X_test) for tree in quick_model.estimators_])
    pred_std = estimators_predictions.std(axis=0)
    
    print("✓ 预测区间计算完成")
except Exception as e:
    print(f"⚠️ 预测计算失败: {e}")
    predictions = y_test = pred_std = None

# ============================================================================
# 创建可视化
# ============================================================================
print("\n[5/5] 生成高级可视化图表...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('二手车价格预测 - 高级分析可视化', 
             fontsize=20, fontweight='bold', y=0.995)

# ============ 图1: 特征重要性 (Top 15) ============
ax1 = plt.subplot(2, 3, 1)
top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'])
ax1.set_xlabel('重要性分数', fontsize=11)
ax1.set_title('① 特征重要性排名 (Top 15)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax1.text(row['importance'], i, f' {row["importance"]:.4f}', 
             va='center', fontsize=9)

# ============ 图2: 学习曲线 ============
ax2 = plt.subplot(2, 3, 2)
ax2.plot(train_sizes_abs, train_mean, 'o-', label='训练集', linewidth=2)
ax2.fill_between(train_sizes_abs, train_mean - train_std, 
                  train_mean + train_std, alpha=0.2)
ax2.plot(train_sizes_abs, valid_mean, 'o-', label='验证集', linewidth=2)
ax2.fill_between(train_sizes_abs, valid_mean - valid_std, 
                  valid_mean + valid_std, alpha=0.2)
ax2.set_xlabel('训练样本数量', fontsize=11)
ax2.set_ylabel('MSE损失', fontsize=11)
ax2.set_title('② 学习曲线 (更多数据的影响)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============ 图3: 价格分布直方图 ============
ax3 = plt.subplot(2, 3, 3)
ax3.hist(y, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
ax3.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {y.mean():.0f}')
ax3.axvline(np.median(y), color='green', linestyle='--', linewidth=2, 
            label=f'中位数: {np.median(y):.0f}')
ax3.set_xlabel('价格', fontsize=11)
ax3.set_ylabel('密度', fontsize=11)
ax3.set_title('③ 真实价格分布', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# ============ 图4: 箱线图 - 检测异常值 ============
ax4 = plt.subplot(2, 3, 4)
box_data = [y, predictions] if predictions is not None else [y]
labels = ['真实价格', '预测价格'] if predictions is not None else ['真实价格']
bp = ax4.boxplot(box_data, labels=labels, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
ax4.set_ylabel('价格', fontsize=11)
ax4.set_title('④ 箱线图 - 异常值检测', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# ============ 图5: 预测置信区间 ============
ax5 = plt.subplot(2, 3, 5)
if predictions is not None and pred_std is not None:
    # 选择一部分样本显示
    sample_indices = np.random.choice(len(predictions), min(200, len(predictions)), replace=False)
    sample_indices = np.sort(sample_indices)
    
    x_axis = range(len(sample_indices))
    ax5.plot(x_axis, y_test[sample_indices], 'o', label='真实值', 
             markersize=4, alpha=0.6)
    ax5.plot(x_axis, predictions[sample_indices], 's', label='预测值', 
             markersize=4, alpha=0.6)
    
    # 置信区间
    ax5.fill_between(x_axis, 
                     predictions[sample_indices] - 2*pred_std[sample_indices],
                     predictions[sample_indices] + 2*pred_std[sample_indices],
                     alpha=0.2, label='95%置信区间')
    
    ax5.set_xlabel('样本索引', fontsize=11)
    ax5.set_ylabel('价格', fontsize=11)
    ax5.set_title('⑤ 预测置信区间 (95%)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, '预测数据不可用', ha='center', va='center',
             fontsize=14, transform=ax5.transAxes)
    ax5.set_title('⑤ 预测置信区间', fontsize=12, fontweight='bold')

# ============ 图6: 相关性热力图 (Top特征) ============
ax6 = plt.subplot(2, 3, 6)
# 选择Top 10特征
top_10_features = feature_importance.head(10)['feature'].tolist()
top_10_indices = [feature_cols.index(f) for f in top_10_features]
corr_matrix = np.corrcoef(X[:, top_10_indices].T)

im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax6.set_xticks(range(len(top_10_features)))
ax6.set_yticks(range(len(top_10_features)))
ax6.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9)
ax6.set_yticklabels(top_10_features, fontsize=9)
ax6.set_title('⑥ 特征相关性热力图 (Top 10)', fontsize=12, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('相关系数', fontsize=10)

# 添加数值
for i in range(len(top_10_features)):
    for j in range(len(top_10_features)):
        text = ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# 保存图片
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'advanced_visualization_{timestamp}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ 高级可视化图表已保存: {filename}")

plt.show()

# ============================================================================
# 生成特征重要性报告
# ============================================================================
print("\n" + "=" * 60)
print("特征重要性报告")
print("=" * 60)
print("\n最重要的5个特征:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:<20s}: {row['importance']:.4f}")

print("\n最不重要的5个特征:")
for idx, row in feature_importance.tail(5).iterrows():
    print(f"  {row['feature']:<20s}: {row['importance']:.4f}")

print("\n" + "=" * 60)
print("分析完成！✓")
print("=" * 60)
