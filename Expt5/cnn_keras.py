#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验五：基于Keras卷积神经网络实现（MNIST）

包含：
1) 数据加载（mnist.py）
2) 序列模型 CNN（model_seq）
3) 功能模型 CNN（model_func）
4) 训练、评估、预测、可视化
5) 模型保存与加载

运行方式：
    python cnn_keras.py           # 默认运行序列模型1个epoch示例
    python cnn_keras.py --mode seq --epochs 1 --batch 128
    python cnn_keras.py --mode func --epochs 1 --batch 128
    python cnn_keras.py --mode both --epochs 1 --batch 128

作者：ML_Tasks
日期：2024-12-12
"""

import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt

# 兼容导入 Keras
try:
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (InputLayer, Input, Reshape, MaxPooling2D,
                                         Conv2D, Dense, Flatten)
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras import backend as K
except Exception:
    from keras.models import Sequential, Model, load_model
    from keras.layers import (InputLayer, Input, Reshape, MaxPooling2D,
                              Conv2D, Dense, Flatten)
    from keras.optimizers import Adam, RMSprop
    import keras.backend as K

from mnist import MNIST

# ----------------------
# 可视化辅助函数
# ----------------------

def plot_images(images, cls_true, cls_pred=None, img_shape=(28, 28), save_path=None):
    """绘制前9张图像"""
    assert len(images) >= 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = f"True: {cls_true[i]}"
        else:
            xlabel = f"True: {cls_true[i]}, Pred: {cls_pred[i]}"
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_example_errors(data, cls_pred, correct, save_path=None):
    """绘制错误分类的前9张图像"""
    incorrect = (correct == False)
    images = data.x_test[incorrect]
    cls_pred_err = cls_pred[incorrect]
    cls_true_err = data.y_test_cls[incorrect]
    if len(images) < 9:
        return
    plot_images(images=images[0:9], cls_true=cls_true_err[0:9], cls_pred=cls_pred_err[0:9],
                img_shape=data.img_shape, save_path=save_path)


def plot_conv_weights(weights, input_channel=0, save_path=None):
    """绘制卷积权重"""
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_filters = weights.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_conv_output(values, save_path=None):
    """绘制卷积层输出"""
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_image(image, img_shape=(28, 28), save_path=None):
    """绘制单张图像"""
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------
# 模型构建
# ----------------------

def build_sequential_model(img_size_flat, img_shape_full, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    model.add(Reshape(img_shape_full))
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_functional_model(img_size_flat, img_shape_full, num_classes):
    inputs = Input(shape=(img_size_flat,))
    net = Reshape(img_shape_full)(inputs)
    net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)
    net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)
    net = Flatten()(net)
    net = Dense(128, activation='relu')(net)
    outputs = Dense(num_classes, activation='softmax')(net)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ----------------------
# 训练 / 评估 / 预测
# ----------------------

def train_and_evaluate(model, data, epochs=1, batch_size=128, tag="seq"):
    print(f"\n开始训练模型[{tag}] ...")
    history = model.fit(
        data.x_train, data.y_train_onehot,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2
    )
    print("\n评估测试集...")
    result = model.evaluate(data.x_test, data.y_test_onehot, verbose=1)
    for name, value in zip(model.metrics_names, result):
        print(name, value)
    print(f"测试集准确率: {result[1]:.4f}")
    return history, result


def predict_and_plot(model, data, save_prefix="seq"):
    print("\n预测测试集前9张图片...")
    images = data.x_test[0:9]
    cls_true = data.y_test_cls[0:9]
    y_pred = model.predict(images)
    cls_pred = np.argmax(y_pred, axis=1)
    plot_images(images, cls_true, cls_pred, img_shape=data.img_shape,
                save_path=f"{save_prefix}_predictions.png")
    print(f"预测图已保存: {save_prefix}_predictions.png")


def predict_all_and_errors(model, data, save_prefix="seq"):
    print("\n获取全部测试集预测并绘制错误样本...")
    y_pred = model.predict(data.x_test)
    cls_pred = np.argmax(y_pred, axis=1)
    correct = (cls_pred == data.y_test_cls)
    plot_example_errors(data, cls_pred, correct, save_path=f"{save_prefix}_errors.png")
    print(f"错误样本图已保存: {save_prefix}_errors.png")


def visualize_weights_and_outputs(model, data, save_prefix="seq"):
    print("\n可视化卷积权重与输出...")
    # 获取层
    layer_input = model.layers[0]
    layer_conv1 = model.layers[2]
    layer_conv2 = model.layers[4]
    # 权重
    weights_conv1 = layer_conv1.get_weights()[0]
    weights_conv2 = layer_conv2.get_weights()[0]
    plot_conv_weights(weights_conv1, input_channel=0, save_path=f"{save_prefix}_conv1_weights.png")
    plot_conv_weights(weights_conv2, input_channel=0, save_path=f"{save_prefix}_conv2_weights.png")
    print("卷积核权重图已保存")
    # 输出 - 方法1 (K.function)
    image1 = data.x_test[0]
    output_conv1 = K.function(inputs=[layer_input.input], outputs=[layer_conv1.output])
    layer_output1 = output_conv1([[image1]])[0]
    plot_conv_output(layer_output1, save_path=f"{save_prefix}_conv1_output.png")
    # 输出 - 方法2 (Model)
    output_conv2 = Model(inputs=layer_input.input, outputs=layer_conv2.output)
    layer_output2 = output_conv2.predict(np.array([image1]))
    plot_conv_output(layer_output2, save_path=f"{save_prefix}_conv2_output.png")
    # 原始图
    plot_image(image1, img_shape=data.img_shape, save_path=f"{save_prefix}_image1.png")
    print("卷积层输出与原图已保存")


def save_and_load(model, path_model="model.keras"):
    print(f"\n保存模型到 {path_model} ...")
    model.save(path_model)
    print("删除模型以测试加载...")
    del model
    print("加载模型...")
    model_loaded = load_model(path_model)
    return model_loaded


# ----------------------
# 主流程
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['seq', 'func', 'both'], default='seq', help='选择运行的模型')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch', type=int, default=128, help='批大小')
    args = parser.parse_args()

    # 加载数据
    data = MNIST()

    if args.mode in ['seq', 'both']:
        # 构建并训练序列模型
        model_seq = build_sequential_model(data.img_size_flat, data.img_shape_full, data.num_classes)
        print("\n序列模型结构：")
        model_seq.summary()
        history_seq, result_seq = train_and_evaluate(model_seq, data, epochs=args.epochs, batch_size=args.batch, tag="seq")
        predict_and_plot(model_seq, data, save_prefix="seq")
        predict_all_and_errors(model_seq, data, save_prefix="seq")
        visualize_weights_and_outputs(model_seq, data, save_prefix="seq")
        model_seq = save_and_load(model_seq, path_model="seq_model.keras")
        # 再次验证加载后的模型（可选）
        _ = model_seq.evaluate(data.x_test, data.y_test_onehot, verbose=0)

    if args.mode in ['func', 'both']:
        # 构建并训练功能模型
        model_func = build_functional_model(data.img_size_flat, data.img_shape_full, data.num_classes)
        print("\n功能模型结构：")
        model_func.summary()
        history_func, result_func = train_and_evaluate(model_func, data, epochs=args.epochs, batch_size=args.batch, tag="func")
        predict_and_plot(model_func, data, save_prefix="func")
        predict_all_and_errors(model_func, data, save_prefix="func")
        visualize_weights_and_outputs(model_func, data, save_prefix="func")
        model_func = save_and_load(model_func, path_model="func_model.keras")
        # 再次验证加载后的模型（可选）
        _ = model_func.evaluate(data.x_test, data.y_test_onehot, verbose=0)

    print("\n全部流程完成！")


if __name__ == '__main__':
    main()
