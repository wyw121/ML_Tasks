#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MNIST数据加载模块

功能：
- 自动下载和加载MNIST数据集
- 数据预处理和格式化
- 提供训练集和测试集访问接口

作者：ML_Tasks
日期：2024-12-12
"""

import numpy as np
import os
from tensorflow.keras.datasets import mnist as keras_mnist
from tensorflow.keras.utils import to_categorical


class MNIST:
    """MNIST数据集加载器"""
    
    def __init__(self, data_dir="data/MNIST/"):
        """
        初始化MNIST数据集
        
        参数:
            data_dir: str, 数据目录（可选，用于兼容性）
        """
        self.data_dir = data_dir
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 加载数据
        print("加载MNIST数据集...")
        (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
        
        # 数据集参数
        self.img_size = 28
        self.img_size_flat = 784
        self.img_shape = (28, 28)
        self.img_shape_full = (28, 28, 1)
        self.num_classes = 10
        self.num_channel = 1
        
        # 数据预处理
        # 训练集
        self.x_train = x_train.reshape(-1, self.img_size_flat).astype('float32') / 255.0
        self.y_train = y_train
        self.y_train_cls = y_train
        self.y_train_onehot = to_categorical(y_train, self.num_classes)
        
        # 测试集
        self.x_test = x_test.reshape(-1, self.img_size_flat).astype('float32') / 255.0
        self.y_test = y_test
        self.y_test_cls = y_test
        self.y_test_onehot = to_categorical(y_test, self.num_classes)
        
        # 数据集大小
        self.num_train = len(x_train)
        self.num_test = len(x_test)
        
        print(f"训练集大小: {self.num_train}")
        print(f"测试集大小: {self.num_test}")
        print("数据加载完成！")


if __name__ == '__main__':
    # 测试数据加载
    data = MNIST()
    
    print("\n数据集参数:")
    print(f"img_size: {data.img_size}")
    print(f"img_size_flat: {data.img_size_flat}")
    print(f"img_shape: {data.img_shape}")
    print(f"img_shape_full: {data.img_shape_full}")
    print(f"num_classes: {data.num_classes}")
    print(f"num_channel: {data.num_channel}")
    
    print("\n数据形状:")
    print(f"x_train: {data.x_train.shape}")
    print(f"y_train: {data.y_train.shape}")
    print(f"x_test: {data.x_test.shape}")
    print(f"y_test: {data.y_test.shape}")
