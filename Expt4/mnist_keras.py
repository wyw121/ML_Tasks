#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验四：Keras基础与简单应用
MNIST手写数字识别

功能：
1. 使用Keras加载MNIST数据集
2. 数据预处理（重塑、归一化、one-hot编码）
3. 搭建基础神经网络（单层感知机）
4. 搭建优化后的神经网络（多层神经网络）
5. 训练和评估模型
6. 可视化结果

作者：ML_Tasks
日期：2024-12-12
"""

import numpy as np
import matplotlib.pyplot as plt
# 兼容导入：优先使用tensorflow.keras，若不可用则回退到keras
try:
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.utils import to_categorical
except Exception:
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.optimizers import SGD
    try:
        from keras.utils import to_categorical
    except Exception:
        # 某些版本中通过np_utils提供to_categorical
        from keras.utils import np_utils as _np_utils
        to_categorical = _np_utils.to_categorical


class MNISTKerasClassifier:
    """MNIST手写数字识别分类器（基于Keras）"""
    
    def __init__(self, model_type='basic'):
        """
        初始化分类器
        
        参数:
            model_type: str, 模型类型
                - 'basic': 基础单层感知机
                - 'improved': 改进的多层神经网络
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.Y_train = None  # one-hot编码后的标签
        self.Y_test = None   # one-hot编码后的标签
        
    def load_data(self):
        """
        加载MNIST数据集
        
        返回:
            tuple: (X_train, y_train), (X_test, y_test)
        """
        print("=" * 60)
        print("步骤1: 加载MNIST数据集")
        print("=" * 60)
        
        # 加载数据
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # 查看数据维度
        print(f"\n原始数据维度:")
        print(f"训练集特征: {X_train.shape}")
        print(f"训练集标签: {y_train.shape}")
        print(f"测试集特征: {X_test.shape}")
        print(f"测试集标签: {y_test.shape}")
        
        # 保存原始数据
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        return (X_train, y_train), (X_test, y_test)
    
    def preprocess_data(self):
        """
        数据预处理
        - 重塑形状：28x28 -> 784
        - 归一化：0-255 -> 0-1
        - one-hot编码
        """
        print("\n" + "=" * 60)
        print("步骤2: 数据预处理")
        print("=" * 60)
        
        # 2.1 重塑训练集和测试集的形状
        print("\n2.1 重塑形状 (28x28 -> 784)")
        X_train = self.X_train.reshape(60000, 784).astype('float32')
        X_test = self.X_test.reshape(10000, 784).astype('float32')
        
        print(f"重塑后训练集: {X_train.shape}")
        print(f"重塑后测试集: {X_test.shape}")
        print(f"数据类型: {X_train.dtype}")
        
        # 2.2 归一化 (0-255 -> 0-1)
        print("\n2.2 归一化 (0-255 -> 0-1)")
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        print("归一化后示例 (训练集第2个样本的第100-150个像素):")
        print(X_train[1, 100:151])
        
        # 2.3 one-hot编码
        print("\n2.3 One-hot编码")
        Y_train = to_categorical(self.y_train, 10)
        Y_test = to_categorical(self.y_test, 10)
        
        print("One-hot编码后的前5个标签:")
        print(Y_train[:5])
        
        # 保存预处理后的数据
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
        return X_train, Y_train, X_test, Y_test
    
    def build_basic_model(self):
        """
        构建基础神经网络（单层感知机）
        
        网络结构:
            输入层: 784维
            输出层: 10维 (10个类别)
            激活函数: softmax
        """
        print("\n" + "=" * 60)
        print("步骤3: 构建基础神经网络（单层感知机）")
        print("=" * 60)
        
        # 创建序列模型
        model = Sequential()
        
        # 添加全连接层（输入层 -> 输出层）
        model.add(Dense(10, input_shape=(784,)))
        
        # 添加激活层
        model.add(Activation('softmax'))
        
        # 查看模型结构
        print("\n模型结构:")
        model.summary()
        
        self.model = model
        return model
    
    def build_improved_model(self):
        """
        构建改进的神经网络（多层神经网络）
        
        网络结构:
            输入层: 784维 -> 128维 (relu激活)
            隐藏层1: 128维 -> 128维 (relu激活)
            输出层: 128维 -> 10维 (softmax激活)
        """
        print("\n" + "=" * 60)
        print("步骤3: 构建改进神经网络（多层神经网络）")
        print("=" * 60)
        
        # 创建序列模型
        model = Sequential()
        
        # 添加输入层和第一个隐藏层
        model.add(Dense(128, input_shape=(784,), activation='relu'))
        
        # 添加第二个隐藏层
        model.add(Dense(128, activation='relu'))
        
        # 添加输出层
        model.add(Dense(10, activation='softmax'))
        
        # 查看模型结构
        print("\n模型结构:")
        model.summary()
        
        self.model = model
        return model
    
    def compile_model(self):
        """
        编译模型
        
        配置:
            - 损失函数: categorical_crossentropy (多分类交叉熵)
            - 优化器: SGD (随机梯度下降)
            - 评估指标: accuracy (准确率)
        """
        print("\n" + "=" * 60)
        print("步骤4: 编译模型")
        print("=" * 60)
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(),
            metrics=['accuracy']
        )
        
        print("模型编译完成！")
        print("- 损失函数: categorical_crossentropy")
        print("- 优化器: SGD")
        print("- 评估指标: accuracy")
    
    def train_model(self, batch_size=128, epochs=20, validation_split=0.2):
        """
        训练模型
        
        参数:
            batch_size: int, 批大小
            epochs: int, 训练轮数
            validation_split: float, 验证集比例
        """
        print("\n" + "=" * 60)
        print("步骤5: 训练模型")
        print("=" * 60)
        
        print(f"\n训练参数:")
        print(f"- 批大小: {batch_size}")
        print(f"- 训练轮数: {epochs}")
        print(f"- 验证集比例: {validation_split}")
        
        # 训练模型
        history = self.model.fit(
            self.X_train,
            self.Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=validation_split
        )
        
        self.history = history
        
        print("\n训练完成！")
        return history
    
    def evaluate_model(self):
        """
        评估模型
        
        返回:
            tuple: (loss, accuracy)
        """
        print("\n" + "=" * 60)
        print("步骤6: 评估模型")
        print("=" * 60)
        
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=1)
        
        print(f"\n测试集评估结果:")
        print(f"Test loss: {score[0]:.10f}")
        print(f"Test accuracy: {score[1]:.4f}")
        
        return score
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        绘制训练历史
        
        参数:
            save_path: str, 保存路径
        """
        if self.history is None:
            print("错误：模型尚未训练！")
            return
        
        print("\n绘制训练历史曲线...")
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失曲线
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制准确率曲线（兼容不同Keras版本的键名）
        acc_key = 'accuracy' if 'accuracy' in self.history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in self.history.history else 'val_acc'
        axes[1].plot(self.history.history[acc_key], label='Training Accuracy')
        axes[1].plot(self.history.history[val_acc_key], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史曲线已保存到: {save_path}")
        plt.close()
    
    def predict_samples(self, num_samples=10):
        """
        预测样本并可视化
        
        参数:
            num_samples: int, 预测样本数量
        """
        print(f"\n预测{num_samples}个测试样本...")
        
        # 随机选择样本
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        # 预测
        predictions = self.model.predict(self.X_test[indices])
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.Y_test[indices], axis=1)
        
        # 可视化
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # 重塑回28x28用于显示
            img = self.X_test[indices[i]].reshape(28, 28)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}',
                            color='green' if true_labels[i] == predicted_labels[i] else 'red')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
        print("预测结果已保存到: predictions.png")
        plt.close()
        
        # 打印准确率
        accuracy = np.sum(predicted_labels == true_labels) / num_samples
        print(f"这{num_samples}个样本的准确率: {accuracy:.2%}")


def run_basic_experiment():
    """运行基础实验（单层感知机）"""
    print("\n" + "=" * 60)
    print("运行基础实验：单层感知机")
    print("=" * 60)
    
    # 创建分类器
    classifier = MNISTKerasClassifier(model_type='basic')
    
    # 1. 加载数据
    classifier.load_data()
    
    # 2. 数据预处理
    classifier.preprocess_data()
    
    # 3. 构建模型
    classifier.build_basic_model()
    
    # 4. 编译模型
    classifier.compile_model()
    
    # 5. 训练模型（200轮）
    classifier.train_model(batch_size=128, epochs=200, validation_split=0.2)
    
    # 6. 评估模型
    classifier.evaluate_model()
    
    # 7. 绘制训练历史
    classifier.plot_training_history('basic_training_history.png')
    
    # 8. 预测样本
    classifier.predict_samples(10)
    
    return classifier


def run_improved_experiment():
    """运行改进实验（多层神经网络）"""
    print("\n" + "=" * 60)
    print("运行改进实验：多层神经网络")
    print("=" * 60)
    
    # 创建分类器
    classifier = MNISTKerasClassifier(model_type='improved')
    
    # 1. 加载数据
    classifier.load_data()
    
    # 2. 数据预处理
    classifier.preprocess_data()
    
    # 3. 构建模型
    classifier.build_improved_model()
    
    # 4. 编译模型
    classifier.compile_model()
    
    # 5. 训练模型（20轮）
    classifier.train_model(batch_size=128, epochs=20, validation_split=0.2)
    
    # 6. 评估模型
    classifier.evaluate_model()
    
    # 7. 绘制训练历史
    classifier.plot_training_history('improved_training_history.png')
    
    # 8. 预测样本
    classifier.predict_samples(10)
    
    return classifier


def main():
    """主函数"""
    print("=" * 60)
    print("实验四：Keras基础与简单应用")
    print("MNIST手写数字识别")
    print("=" * 60)
    
    # 选择运行模式
    print("\n请选择实验模式:")
    print("1. 基础实验（单层感知机，200轮训练）")
    print("2. 改进实验（多层神经网络，20轮训练）")
    print("3. 两个实验都运行")
    
    choice = input("请输入选项 (1/2/3，直接回车默认运行改进实验): ").strip()
    
    if choice == '1':
        # 运行基础实验
        basic_classifier = run_basic_experiment()
        print("\n" + "=" * 60)
        print("基础实验完成！")
        print("=" * 60)
        
    elif choice == '2' or choice == '':
        # 运行改进实验
        improved_classifier = run_improved_experiment()
        print("\n" + "=" * 60)
        print("改进实验完成！")
        print("=" * 60)
        
    elif choice == '3':
        # 运行两个实验
        print("\n" + "=" * 60)
        print("开始运行基础实验...")
        print("=" * 60)
        basic_classifier = run_basic_experiment()
        
        print("\n" + "=" * 60)
        print("基础实验完成！开始运行改进实验...")
        print("=" * 60)
        improved_classifier = run_improved_experiment()
        
        # 比较结果
        print("\n" + "=" * 60)
        print("实验对比")
        print("=" * 60)
        
        basic_score = basic_classifier.evaluate_model()
        improved_score = improved_classifier.evaluate_model()
        
        print("\n对比结果:")
        print(f"基础模型 - 测试准确率: {basic_score[1]:.4f}")
        print(f"改进模型 - 测试准确率: {improved_score[1]:.4f}")
        print(f"准确率提升: {(improved_score[1] - basic_score[1]) * 100:.2f}%")
        
        print("\n" + "=" * 60)
        print("所有实验完成！")
        print("=" * 60)
        
    else:
        print("无效选项，运行改进实验...")
        improved_classifier = run_improved_experiment()
        print("\n" + "=" * 60)
        print("改进实验完成！")
        print("=" * 60)


if __name__ == '__main__':
    main()
