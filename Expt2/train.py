"""
实验2: 使用高斯判别分析(GDA)进行收入预测
任务: 二分类 - 预测年收入是否超过50K
方法: 概率生成模型 (不使用梯度下降!)
"""

import numpy as np
import pandas as pd

# ============================================================
# 步骤1: 数据加载
# ============================================================
def load_data():
    """
    加载训练集和测试集
    返回: train_df, test_df
    """
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df


# ============================================================
# 步骤2: 数据清洗
# ============================================================
def clean_data(train_df, test_df):
    """
    数据清洗: 处理标签值前的空格和检查'?'标记
    注: '?'标记在Onehot编码时会自动被当作一个单独的类别处理
    """
    print("\n" + "=" * 60)
    print("步骤2: 数据清洗")
    print("=" * 60)
    
    # 去除标签列的空格
    train_df['income'] = train_df['income'].str.strip()
    
    # 检查'?'标记的分布(用于了解数据质量)
    print("训练集中'?'标记分布:")
    print((train_df == '?').sum())
    print("\n测试集中'?'标记分布:")
    print((test_df == '?').sum())
    
    # 注: 保留'?'标记,在Onehot编码时会自动作为单独类别
    
    print("数据清洗完成!")
    return train_df, test_df


# ============================================================
# 步骤3: 特征工程 - Onehot编码
# ============================================================
def preprocess_features(train_df, test_df):
    """
    特征预处理:
    1. 分离标签
    2. 离散特征 → Onehot编码
    3. 连续特征 → 保持原值
    4. 组合所有特征
    
    返回: X_train, Y_train, X_test (都是numpy数组)
    """
    print("\n" + "=" * 60)
    print("步骤3: 特征工程")
    print("=" * 60)
    
    # 3.1 分离标签并转换为0/1
    Y_train = train_df['income'].values
    Y_train = (Y_train == '>50K').astype(int)  # >50K为1, <=50K为0
    
    # 3.2 定义连续特征和离散特征
    continuous_features = ['age', 'fnlwgt', 'education_num', 
                          'capital_gain', 'capital_loss', 'hours_per_week']
    
    categorical_features = ['workclass', 'education', 'marital_status', 
                           'occupation', 'relationship', 'race', 
                           'sex', 'native_country']
    
    # 3.3 提取连续特征
    X_train_continuous = train_df[continuous_features].values
    X_test_continuous = test_df[continuous_features].values
    print(f"连续特征形状: {X_train_continuous.shape}")
    
    # 3.4 Onehot编码离散特征
    # 重要: 训练集和测试集要一起编码,保证列数一致
    train_categorical = train_df[categorical_features]
    test_categorical = test_df[categorical_features]
    
    combined = pd.concat([train_categorical, test_categorical], axis=0)
    combined_encoded = pd.get_dummies(combined)
    
    X_train_categorical = combined_encoded[:len(train_df)].values
    X_test_categorical = combined_encoded[len(train_df):].values
    
    print(f"离散特征Onehot编码后形状: {X_train_categorical.shape}")
    
    # 3.5 组合连续特征和离散特征
    X_train = np.concatenate([X_train_continuous, X_train_categorical], axis=1)
    X_test = np.concatenate([X_test_continuous, X_test_categorical], axis=1)
    
    print(f"最终特征形状: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"期望特征维度: 106维")
    
    return X_train, Y_train, X_test


# ============================================================
# 步骤4: 数据标准化
# ============================================================
def normalize_features(X_train, X_test):
    """
    标准化特征:
    只标准化前6个连续特征,onehot特征(已经是0/1)不需要标准化
    
    重要: 使用训练集的mean和std来标准化测试集!
    """
    print("\n" + "=" * 60)
    print("步骤4: 数据标准化")
    print("=" * 60)
    
    # 复制数据,避免修改原始数据
    X_train_normalized = X_train.copy()
    X_test_normalized = X_test.copy()
    
    # 只标准化前6列连续特征,Onehot特征(0/1)不需要标准化
    mean = X_train[:, :6].mean(axis=0)
    std = X_train[:, :6].std(axis=0)
    X_train_normalized[:, :6] = (X_train[:, :6] - mean) / std
    X_test_normalized[:, :6] = (X_test[:, :6] - mean) / std  # 使用训练集的统计量!
    
    print("标准化完成!")
    
    return X_train_normalized, X_test_normalized


# ============================================================
# 步骤5: 训练GDA模型 - 计算统计量
# ============================================================
def train_gda(X_train, Y_train):
    """
    训练高斯判别分析模型
    
    这不是梯度下降!这是直接计算统计量!
    
    需要计算:
    1. 先验概率: P(C0), P(C1)
    2. 类别均值: μ0, μ1
    3. 共享协方差矩阵: Σ
    
    返回: 模型参数字典
    """
    print("\n" + "=" * 60)
    print("步骤5: 训练GDA模型(计算统计量)")
    print("=" * 60)
    
    # 5.1 计算先验概率
    n_samples = len(Y_train)
    n_class0 = np.sum(Y_train == 0)
    n_class1 = np.sum(Y_train == 1)
    prior_0 = n_class0 / n_samples
    prior_1 = n_class1 / n_samples
    
    print(f"先验概率: P(C0)={prior_0:.4f}, P(C1)={prior_1:.4f}")
    
    # 5.2 分离两个类别的样本
    X_class0 = X_train[Y_train == 0]
    X_class1 = X_train[Y_train == 1]
    
    print(f"类别0样本数: {len(X_class0)}")
    print(f"类别1样本数: {len(X_class1)}")
    
    # 5.3 计算每个类别的均值向量
    mean_0 = np.mean(X_class0, axis=0)  # 106维向量
    mean_1 = np.mean(X_class1, axis=0)  # 106维向量
    
    print(f"均值向量形状: μ0={mean_0.shape}, μ1={mean_1.shape}")
    
    # 5.4 计算共享协方差矩阵
    print("\n计算共享协方差矩阵...")
    cov_0 = np.cov(X_class0.T)  # 注意要转置!
    cov_1 = np.cov(X_class1.T)
    shared_cov = (n_class0 * cov_0 + n_class1 * cov_1) / n_samples  # 加权平均
    
    # 添加正则化项,防止矩阵奇异导致数值不稳定
    epsilon = 1e-3
    shared_cov += epsilon * np.eye(shared_cov.shape[0])
    
    print(f"协方差矩阵形状: {shared_cov.shape}")
    
    # 返回模型参数
    model = {
        'prior_0': prior_0,
        'prior_1': prior_1,
        'mean_0': mean_0,
        'mean_1': mean_1,
        'shared_cov': shared_cov
    }
    
    return model


# ============================================================
# 步骤6: 预测
# ============================================================
def predict(X_test, model):
    """
    使用训练好的GDA模型进行预测
    
    根据贝叶斯公式化简后的结果:
    P(C1|x) = sigmoid(w^T * x + b)
    
    其中:
    w = Σ^(-1) * (μ1 - μ0)
    b = ... (包含先验概率和均值的二次项)
    """
    print("\n" + "=" * 60)
    print("步骤6: 预测")
    print("=" * 60)
    
    # 提取模型参数
    mean_0 = model['mean_0']
    mean_1 = model['mean_1']
    shared_cov = model['shared_cov']
    prior_0 = model['prior_0']
    prior_1 = model['prior_1']
    
    # 计算决策边界参数 w 和 b
    # 协方差矩阵的逆用于计算决策边界,反映数据在各方向的分布
    cov_inv = np.linalg.inv(shared_cov)
    
    w = cov_inv @ (mean_1 - mean_0)
    # b包含三项: 均值的二次项 + 先验概率项
    b = -0.5 * (mean_1 @ cov_inv @ mean_1 - mean_0 @ cov_inv @ mean_0) + np.log(prior_1 / prior_0)
    
    # 计算线性组合 z = w^T * x + b
    z = X_test @ w + b
    
    # 应用sigmoid函数得到概率,使用clip防止数值溢出
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    probabilities = sigmoid(z)
    
    # 根据概率阈值0.5进行二分类
    predictions = (probabilities > 0.5).astype(int)
    
    print(f"预测完成! 预测了 {len(predictions) if predictions is not None else 'TODO'} 个样本")
    
    return predictions


# ============================================================
# 步骤7: 保存结果
# ============================================================
def save_predictions(predictions):
    """
    将预测结果保存为CSV文件
    格式:
    id,label
    1,0
    2,1
    ...
    """
    print("\n" + "=" * 60)
    print("步骤7: 保存预测结果")
    print("=" * 60)
    
    # 创建结果DataFrame
    result = pd.DataFrame({
        'id': range(1, len(predictions) + 1),
        'label': predictions
    })
    
    # 保存为CSV,不包含索引列
    result.to_csv('predict.csv', index=False)
    
    print(f"结果已保存到 predict.csv (共{len(predictions)}个预测)")
    print(f"预测为类别1(>50K)的样本数: {np.sum(predictions == 1)}")
    print(f"预测为类别0(<=50K)的样本数: {np.sum(predictions == 0)}")


# ============================================================
# 主程序
# ============================================================
def main():
    """
    主流程:
    1. 加载数据
    2. 清洗数据
    3. 特征工程(Onehot编码)
    4. 标准化
    5. 训练GDA模型(计算统计量)
    6. 预测
    7. 保存结果
    """
    print("\n" + "=" * 60)
    print("实验2: 高斯判别分析 - 收入预测")
    print("=" * 60)
    
    # 步骤1: 加载数据
    train_df, test_df = load_data()
    
    # 步骤2: 数据清洗
    train_df, test_df = clean_data(train_df, test_df)
    
    # 步骤3: 特征工程
    X_train, Y_train, X_test = preprocess_features(train_df, test_df)
    
    # 步骤4: 标准化
    X_train, X_test = normalize_features(X_train, X_test)
    
    # 步骤5: 训练模型
    model = train_gda(X_train, Y_train)
    
    # 步骤6: 预测
    predictions = predict(X_test, model)
    
    # 步骤7: 保存结果
    save_predictions(predictions)
    
    print("\n" + "=" * 60)
    print("任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
