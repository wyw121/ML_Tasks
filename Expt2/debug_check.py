import pandas as pd
import numpy as np

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 清洗
train['income'] = train['income'].str.strip()

categorical_features = ['workclass', 'education', 'marital_status', 
                       'occupation', 'relationship', 'race', 
                       'sex', 'native_country']

# 检查每个特征的唯一值
for col in categorical_features:
    train_unique = train[col].nunique()
    test_unique = test[col].nunique()
    print(f"{col:20s}: 训练集{train_unique:3d}个值, 测试集{test_unique:3d}个值")

# 合并编码
train_cat = train[categorical_features]
test_cat = test[categorical_features]
combined = pd.concat([train_cat, test_cat], axis=0)
encoded = pd.get_dummies(combined)

print(f"\nOnehot编码后总列数: {encoded.shape[1]}")
print(f"连续特征: 6列")
print(f"总特征数: {6 + encoded.shape[1]}列")
