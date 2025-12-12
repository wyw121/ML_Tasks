"""
实验三：Logistic回归预测二分类
年薪是否高于50K的二分类预测任务

实验目的：
- 学会使用逻辑回归知识
- 手动实现梯度下降方法
- 完成年薪超过50K的二分类预测

作者：实验学生
完成日期：2024-12-12
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class LogisticRegression:
    """Logistic回归分类器 - 使用梯度下降法"""
    
    def __init__(self, learning_rate=0.01, num_epoch=1000, batch_size=32, 
                 lambda_reg=0.0, validation_split=0.1):
        """
        初始化Logistic回归分类器
        
        参数：
            learning_rate: 学习率
            num_epoch: 训练轮数
            batch_size: 批大小
            lambda_reg: 正则化系数
            validation_split: 验证集比例
        """
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.validation_split = validation_split
        
        self.w = None  # 权重
        self.b = None  # 偏置
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        # 标准化参数
        self.X_mean = None
        self.X_std = None
    
    def _load_data(self, X_path, Y_path=None):
        """加载数据"""
        X = np.genfromtxt(X_path, delimiter=',', skip_header=1)
        
        if Y_path is not None:
            Y = np.genfromtxt(Y_path, delimiter=',', skip_header=1)
            return X, Y
        return X
    
    def _sigmoid(self, z):
        """
        Sigmoid函数
        σ(z) = 1 / (1 + e^(-z))
        
        使用np.clip避免溢出，将数据夹在[1e-6, 1-1e-6]之间
        """
        return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    
    def _get_prob(self, X, w, b):
        """
        获取预测概率
        P(y=1|x) = σ(w·x + b)
        """
        z = np.add(np.matmul(X, w), b)
        return self._sigmoid(z)
    
    def _cross_entropy(self, y_pred, y_true):
        """
        计算交叉熵损失
        L = -[y·log(y_pred) + (1-y)·log(1-y_pred)]
        """
        # 确保y_pred在有效范围内
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
        
        # 交叉熵
        cross_entropy = -np.mean(
            np.multiply(y_true, np.log(y_pred)) + 
            np.multiply(1 - y_true, np.log(1 - y_pred))
        )
        return cross_entropy
    
    def _compute_loss(self, X, Y, w, b):
        """
        计算损失函数
        L(w) = CrossEntropy + λ·||w||²
        """
        y_pred = self._get_prob(X, w, b)
        loss = self._cross_entropy(y_pred, Y)
        
        # 添加L2正则化
        if self.lambda_reg > 0:
            loss += self.lambda_reg * np.sum(np.square(w))
        
        return loss
    
    def _compute_accuracy(self, X, Y, w, b):
        """计算准确率"""
        y_pred = np.round(self._get_prob(X, w, b))
        accuracy = np.mean(y_pred == Y)
        return accuracy
    
    def _gradient(self, X, Y, w, b):
        """
        计算梯度
        对于权重w：w_grad = -mean(pred_error^T · X^T) + λ·w
        对于偏置b：b_grad = -mean(pred_error)
        """
        m = X.shape[0]
        
        # 预测值
        y_pred = self._get_prob(X, w, b)
        
        # 预测误差
        pred_error = y_pred - Y  # (m, 1)
        
        # 权重梯度
        w_grad = -np.mean(np.multiply(pred_error.reshape(-1, 1), X), axis=0)
        
        # 添加L2正则化
        if self.lambda_reg > 0:
            w_grad += self.lambda_reg * w
        
        # 偏置梯度
        b_grad = -np.mean(pred_error)
        
        return w_grad, b_grad
    
    def _normalize_data(self, X, is_training=True):
        """
        标准化数据（正态分布）
        对指定列进行正态标准化：x_norm = (x - mean) / std
        
        选择要标准化的列（连续特征）：
        age, education_num, capital_gain, capital_loss, hours_per_week等
        """
        # 指定要标准化的列（连续特征）
        # 根据特征含义选择：年龄、教育年数、资本增益、资本损失、工作小时数
        standardize_cols = [0, 4, 10, 11, 12]  # age, education-num, capital-gain, capital-loss, hours-per-week
        
        X_normalized = X.copy()
        
        if is_training:
            # 训练集：计算均值和标准差
            self.X_mean = np.zeros(X.shape[1])
            self.X_std = np.ones(X.shape[1])
            
            for col in standardize_cols:
                self.X_mean[col] = np.mean(X[:, col])
                self.X_std[col] = np.std(X[:, col])
                
                # 防止除以0
                if self.X_std[col] == 0:
                    self.X_std[col] = 1.0
                
                X_normalized[:, col] = (X[:, col] - self.X_mean[col]) / self.X_std[col]
        else:
            # 测试集：使用训练集的均值和标准差
            for col in standardize_cols:
                X_normalized[:, col] = (X[:, col] - self.X_mean[col]) / self.X_std[col]
        
        return X_normalized
    
    def train(self, X, Y):
        """
        训练模型
        使用验证集监控过拟合
        """
        print("="*70)
        print("开始训练Logistic回归模型...")
        print("="*70)
        
        # 数据标准化
        print("\n数据标准化...")
        X_normalized = self._normalize_data(X, is_training=True)
        
        # 分割训练集和验证集
        n_samples = X.shape[0]
        n_val = int(n_samples * self.validation_split)
        n_train = n_samples - n_val
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X_normalized[indices]
        Y_shuffled = Y[indices]
        
        X_train = X_shuffled[:n_train]
        Y_train = Y_shuffled[:n_train]
        X_val = X_shuffled[n_train:]
        Y_val = Y_shuffled[n_train:]
        
        print(f"✓ 训练集大小: {X_train.shape[0]}")
        print(f"✓ 验证集大小: {X_val.shape[0]}")
        
        # 初始化权重和偏置
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        print(f"✓ 特征维度: {n_features}")
        print(f"\n训练参数配置:")
        print(f"  学习率: {self.learning_rate}")
        print(f"  训练轮数: {self.num_epoch}")
        print(f"  批大小: {self.batch_size}")
        print(f"  正则化系数λ: {self.lambda_reg}")
        
        print(f"\n开始迭代训练...")
        
        # 训练循环
        n_batches = (n_train + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.num_epoch):
            # 训练集
            epoch_train_loss = 0.0
            n_batches_actual = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_train)
                
                X_batch = X_train[start_idx:end_idx]
                Y_batch = Y_train[start_idx:end_idx]
                
                # 梯度计算
                w_grad, b_grad = self._gradient(X_batch, Y_batch, self.w, self.b)
                
                # 参数更新
                self.w -= self.learning_rate * w_grad
                self.b -= self.learning_rate * b_grad
                
                # 计算损失
                batch_loss = self._compute_loss(X_batch, Y_batch, self.w, self.b)
                epoch_train_loss += batch_loss
                n_batches_actual += 1
            
            # 平均训练损失
            avg_train_loss = epoch_train_loss / n_batches_actual
            
            # 验证集
            val_loss = self._compute_loss(X_val, Y_val, self.w, self.b)
            
            # 准确率
            train_acc = self._compute_accuracy(X_train, Y_train, self.w, self.b)
            val_acc = self._compute_accuracy(X_val, Y_val, self.w, self.b)
            
            # 记录历史
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            # 定期打印
            if (epoch + 1) % max(1, self.num_epoch // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{self.num_epoch}: "
                      f"Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f} | "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        print(f"\n✓ 训练完成")
        print(f"  最终训练损失: {self.train_loss_history[-1]:.6f}")
        print(f"  最终验证损失: {self.val_loss_history[-1]:.6f}")
        print(f"  最终训练准确率: {self.train_acc_history[-1]:.4f}")
        print(f"  最终验证准确率: {self.val_acc_history[-1]:.4f}")
    
    def predict(self, X):
        """预测"""
        if self.w is None or self.b is None:
            raise ValueError("模型未训练。请先调用train方法。")
        
        # 标准化测试数据
        X_normalized = self._normalize_data(X, is_training=False)
        
        # 获取预测概率
        y_prob = self._get_prob(X_normalized, self.w, self.b)
        
        # 二分类：概率>0.5则为1
        y_pred = np.round(y_prob)
        
        return y_pred, y_prob
    
    def plot_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_loss_history) + 1)
        
        # 损失函数
        axes[0].plot(epochs, self.train_loss_history, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, self.val_loss_history, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 准确率
        axes[1].plot(epochs, self.train_acc_history, 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, self.val_acc_history, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 训练历史已保存: {save_path}")
        plt.close()
    
    def get_feature_importance(self, feature_names=None):
        """获取特征重要性（权重绝对值）"""
        if self.w is None:
            raise ValueError("模型未训练。")
        
        importance = np.abs(self.w)
        
        # 排序
        sorted_indices = np.argsort(importance)[::-1]
        
        print("\n" + "="*70)
        print("特征重要性分析（权重绝对值）")
        print("="*70)
        
        if feature_names is not None:
            for i in range(min(15, len(sorted_indices))):
                idx = sorted_indices[i]
                print(f"  {i+1:2d}. {feature_names[idx]:30s}: {importance[idx]:8.6f}")
        else:
            for i in range(min(15, len(sorted_indices))):
                idx = sorted_indices[i]
                print(f"  {i+1:2d}. Feature {idx:3d}: {importance[idx]:8.6f}")
        
        return importance, sorted_indices


def load_and_preprocess_data(data_dir='data'):
    """
    加载和预处理数据
    包括one-hot编码和标准化
    """
    print("="*70)
    print("加载和预处理数据...")
    print("="*70)
    
    X_train_path = f'{data_dir}/X_train'
    Y_train_path = f'{data_dir}/Y_train'
    X_test_path = f'{data_dir}/X_test'
    
    # 加载数据
    print("\n加载训练数据...")
    X_train = np.genfromtxt(X_train_path, delimiter=',', skip_header=1)
    Y_train = np.genfromtxt(Y_train_path, delimiter=',', skip_header=1)
    
    print(f"✓ 训练数据加载完成: X_train {X_train.shape}, Y_train {Y_train.shape}")
    
    print("\n加载测试数据...")
    X_test = np.genfromtxt(X_test_path, delimiter=',', skip_header=1)
    print(f"✓ 测试数据加载完成: X_test {X_test.shape}")
    
    return X_train, Y_train, X_test


def main():
    """主函数"""
    
    print("\n" + "="*80)
    print(" "*20 + "实验三：Logistic回归预测二分类")
    print(" "*15 + "年薪是否高于50K的二分类预测任务")
    print("="*80)
    
    # 1. 加载数据
    try:
        X_train, Y_train, X_test = load_and_preprocess_data('data')
    except FileNotFoundError as e:
        print(f"✗ 错误：{e}")
        print("请确保数据文件在 data 文件夹中")
        return
    
    # 2. 创建和训练模型
    print("\n" + "="*70)
    print("创建Logistic回归模型...")
    print("="*70)
    
    model = LogisticRegression(
        learning_rate=0.01,      # 学习率
        num_epoch=500,           # 训练轮数
        batch_size=64,           # 批大小
        lambda_reg=0.0001,       # L2正则化系数
        validation_split=0.1     # 验证集比例
    )
    
    # 训练
    model.train(X_train, Y_train)
    
    # 3. 绘制训练历史
    print("\n绘制训练历史...")
    model.plot_history('training_history.png')
    
    # 4. 特征重要性分析
    model.get_feature_importance()
    
    # 5. 对测试集进行预测
    print("\n" + "="*70)
    print("对测试集进行预测...")
    print("="*70)
    
    y_test_pred, y_test_prob = model.predict(X_test)
    
    print(f"✓ 预测完成")
    print(f"  预测样本数: {len(y_test_pred)}")
    print(f"  正样本数（年薪>50K）: {int(np.sum(y_test_pred))}")
    print(f"  负样本数（年薪≤50K）: {int(np.sum(1 - y_test_pred))}")
    
    # 6. 保存预测结果
    print("\n" + "="*70)
    print("保存预测结果...")
    print("="*70)
    
    output_path = 'output.csv'
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        
        for i, pred in enumerate(y_test_pred):
            writer.writerow([i, int(pred)])
    
    print(f"✓ 预测结果已保存: {output_path}")
    
    # 打印完成信息
    print("\n" + "="*80)
    print(" "*25 + "✓ 实验完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  - output.csv                : 预测结果")
    print("  - training_history.png      : 训练曲线")
    print("\n预测结果统计:")
    print(f"  - 总样本数: {len(y_test_pred)}")
    print(f"  - 年薪>50K的人数: {int(np.sum(y_test_pred))}")
    print(f"  - 年薪≤50K的人数: {int(np.sum(1 - y_test_pred))}")
    print("="*80)


if __name__ == '__main__':
    main()
