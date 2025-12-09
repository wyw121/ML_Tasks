import pandas as pd
import numpy as np

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=" * 60)
print("训练集信息")
print("=" * 60)
print(f"形状: {train.shape}")
print(f"\n列名 (共{len(train.columns)}列):")
print(train.columns.tolist())

print("\n前5行数据:")
print(train.head())

print("\n数据类型统计:")
print(train.dtypes.value_counts())

print("\n每列的数据类型:")
for col in train.columns:
    print(f"{col:20s}: {str(train[col].dtype):10s} - 唯一值: {train[col].nunique()}")

print("\n缺失值检查:")
missing = train.isnull().sum()
if missing.sum() == 0:
    print("✓ 无缺失值")
else:
    print(missing[missing > 0])

print("\n标签分布:")
print(train['income'].value_counts())
print(f"类别0 (<=50K): {(train['income'] == ' <=50K').sum()} ({(train['income'] == ' <=50K').sum()/len(train)*100:.2f}%)")
print(f"类别1 (>50K):  {(train['income'] == ' >50K').sum()} ({(train['income'] == ' >50K').sum()/len(train)*100:.2f}%)")

print("\n" + "=" * 60)
print("测试集信息")
print("=" * 60)
print(f"形状: {test.shape}")
print(f"是否包含标签列: {'income' in test.columns}")

print("\n" + "=" * 60)
print("连续特征分析")
print("=" * 60)
continuous_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if 'income' in continuous_cols:
    continuous_cols.remove('income')
print(f"连续特征: {continuous_cols}")
print(f"\n连续特征统计:")
print(train[continuous_cols].describe())

print("\n" + "=" * 60)
print("离散特征分析")
print("=" * 60)
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
if 'income' in categorical_cols:
    categorical_cols.remove('income')
print(f"离散特征: {categorical_cols}")
for col in categorical_cols:
    print(f"\n{col}: {train[col].nunique()}个不同值")
    print(train[col].value_counts().head(5))
