"""
实验一：线性回归预测PM2.5值
使用Adagrad梯度下降方法进行PM2.5预测

实验目的：
- 学会使用线性回归知识
- 手动使用Adagrad梯度下降方法
- 通过给定数据完成PM2.5值的回归预测

作者：实验学生
完成日期：2024-12-12
"""

import numpy as np
import csv
import math
import os
from pathlib import Path

class PM25Predictor:
    """PM2.5预测器 - 使用Adagrad梯度下降"""
    
    def __init__(self, learning_rate=0.01, iterations=10000, epsilon=1e-8):
        """
        初始化预测器
        
        参数：
            learning_rate: 初始学习率，通常设为0.01
            iterations: 迭代次数
            epsilon: 平滑项，避免除数为0
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.w = None  # 权重向量
        self.prev_gra = None  # 存储之前迭代的梯度平方
        
    def load_train_data(self, train_file):
        """
        加载和处理训练数据
        
        参数：
            train_file: 训练数据文件路径
            
        返回：
            train_x: 训练输入数据 (样本数 x 163) - 9小时*18个污染物 + 偏置项
            train_y: 训练目标数据 (样本数 x 1) - 第10小时PM2.5值
        """
        print("="*60)
        print("加载和处理训练数据...")
        print("="*60)
        
        # 步骤1：按18个维度处理原始数据
        data = self._process_raw_data(train_file)
        
        # 步骤2：规整化训练数据 - 每10小时为一个样本
        train_x, train_y = self._prepare_training_data(data)
        
        print(f"✓ 训练数据加载完成")
        print(f"  训练集大小: {train_x.shape}")
        print(f"  特征维度: {train_x.shape[1]}")
        print(f"  样本数: {train_x.shape[0]}")
        
        return train_x, train_y
    
    def _process_raw_data(self, train_file):
        """
        步骤1：将原始数据按18个污染物维度处理
        伪代码:
        {
        Declare a 18-dim vector (Data);
        for i_th row in training data: 
            Data[i_th row%18].append(every element in i_th row)
        }
        """
        data = [[] for _ in range(18)]  # 18个污染物维度
        
        try:
            # 使用big5编码读取，因为原始数据可能包含繁体字
            with open(train_file, 'r', encoding='big5') as text:
                row = csv.reader(text, delimiter=",")
                
                # 跳过标题行
                header = next(row)
                
                # 处理每一行数据
                for i, row_data in enumerate(row):
                    # 跳过非数值数据
                    if len(row_data) < 18:
                        continue
                    
                    # 每18行为一个月的一天
                    day_index = (i % 20)  # 0-19，对应每个月的20天
                    hour_index = i // 20  # 第几个小时
                    pollution_index = i % 18  # 18个污染物中的第几个
                    
                    try:
                        # 跳过缺失值 (通常表示为'NR')
                        for j, val in enumerate(row_data):
                            if val != 'NR' and val != '':
                                try:
                                    data[j].append(float(val))
                                except:
                                    pass
                    except:
                        pass
        
        except FileNotFoundError:
            print(f"✗ 错误：找不到文件 {train_file}")
            raise
        
        return data
    
    def _prepare_training_data(self, data):
        """
        步骤2：规整化训练数据
        - 每个月有20天，每天24小时，共480小时
        - 每10小时为一个样本（前9小时特征，第10小时PM2.5为目标）
        - 每月有 (480-9) = 471 个样本
        - 共12个月，共有 12 * 471 = 5652 个样本
        
        伪代码:
        {
        for i in all the given data:
            sample every 10 hrs：
                train_x.append(previous 9-hr data)
                train_y.append(the value of 10th-hr pm2.5)
            add a bias term to every data in train_x
        }
        """
        # 加载完整的原始数据用于构建时间序列
        train_x = []
        train_y = []
        
        # 重新加载数据构建完整的时间序列
        with open('data/train.csv', 'r', encoding='big5') as text:
            row = csv.reader(text, delimiter=",")
            header = next(row)  # 跳过标题
            
            # 构建完整的数据矩阵
            all_data = []
            for row_data in row:
                if len(row_data) > 0:
                    try:
                        # 转换为浮点数，缺失值处理为0
                        values = []
                        for val in row_data:
                            if val != 'NR' and val != '':
                                try:
                                    values.append(float(val))
                                except:
                                    values.append(0.0)
                            else:
                                values.append(0.0)
                        all_data.append(values[:18])  # 只取前18列
                    except:
                        pass
        
        # 从完整的数据中构建样本
        # 每10小时为一个样本
        all_data = np.array(all_data)
        
        for i in range(len(all_data) - 9):
            # 获取前9小时的所有18个污染物数据
            # 形状: (9, 18) -> 展平为 (1, 162)
            x_sample = all_data[i:i+9].flatten()  # 9 * 18 = 162 维
            
            # 获取第10小时的PM2.5值 (第10列是PM2.5)
            y_sample = all_data[i+9, 9]  # PM2.5通常在第10列（索引9）
            
            train_x.append(x_sample)
            train_y.append(y_sample)
        
        # 转换为numpy数组
        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1, 1)
        
        # 步骤3：添加偏置项（在第一列添加常数1）
        train_x = np.concatenate((np.ones((train_x.shape[0], 1)), train_x), axis=1)
        
        print(f"✓ 数据规整化完成")
        print(f"  样本总数: {train_x.shape[0]}")
        print(f"  特征维度(包括偏置): {train_x.shape[1]}")
        
        return train_x, train_y
    
    def train(self, train_x, train_y):
        """
        使用Adagrad梯度下降法训练模型
        
        Adagrad更新规则：
        g_t += grad^2  （累积梯度平方）
        w -= learning_rate * grad / sqrt(g_t + epsilon)
        """
        print("\n" + "="*60)
        print("开始训练模型（Adagrad梯度下降）...")
        print("="*60)
        print(f"参数配置:")
        print(f"  学习率: {self.learning_rate}")
        print(f"  迭代次数: {self.iterations}")
        print(f"  平滑项ε: {self.epsilon}")
        
        # 初始化权重向量
        m, n = train_x.shape
        self.w = np.zeros((n, 1))
        self.prev_gra = np.zeros((n, 1))
        
        # 存储训练过程中的损失
        loss_history = []
        
        # 迭代训练
        for iteration in range(self.iterations):
            # 前向传播: y' = X * w
            y_pred = np.dot(train_x, self.w)
            
            # 计算损失
            loss = y_pred - train_y
            
            # 计算均方差和标准差
            mse = np.sum(loss**2) / len(train_x)
            rmse = math.sqrt(mse)
            loss_history.append(rmse)
            
            # 计算梯度 gradient = 2 * X^T * (y' - y)
            gradient = 2 * np.dot(train_x.T, loss)
            
            # Adagrad: 累积梯度平方
            self.prev_gra += gradient**2
            
            # 计算自适应学习率
            ada = np.sqrt(self.prev_gra) + self.epsilon
            
            # 更新权重 w = w - learning_rate * gradient / ada
            self.w -= self.learning_rate * gradient / ada
            
            # 定期打印训练进度
            if (iteration + 1) % 1000 == 0 or iteration == 0:
                print(f"  Iteration {iteration+1:5d}/{self.iterations}: RMSE = {rmse:.6f}")
        
        print(f"\n✓ 训练完成")
        print(f"  最终RMSE: {loss_history[-1]:.6f}")
        
        return loss_history
    
    def save_model(self, model_path='model.npy'):
        """保存训练好的模型权重"""
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        np.save(model_path, self.w)
        print(f"\n✓ 模型已保存: {model_path}")
    
    def load_model(self, model_path='model.npy'):
        """加载保存的模型"""
        self.w = np.load(model_path)
        print(f"✓ 模型已加载: {model_path}")
    
    def load_test_data(self, test_file):
        """
        加载测试数据
        Test.csv 格式: 连续10小时为1笔数据
        前9小时是特征，第10小时的PM2.5是答案
        一共可以取出240笔不重复的test data
        """
        print("\n" + "="*60)
        print("加载测试数据...")
        print("="*60)
        
        test_x = []
        
        try:
            with open(test_file, 'r') as text:
                row = csv.reader(text, delimiter=",")
                
                # 读取所有测试数据
                all_test_data = []
                for row_data in row:
                    if len(row_data) > 0:
                        try:
                            values = []
                            for val in row_data:
                                if val != 'NR' and val != '':
                                    try:
                                        values.append(float(val))
                                    except:
                                        values.append(0.0)
                                else:
                                    values.append(0.0)
                            if len(values) >= 18:
                                all_test_data.append(values[:18])
                        except:
                            pass
                
                all_test_data = np.array(all_test_data)
                
                # 处理测试数据：每18行（9小时）为一个样本
                for i in range(0, len(all_test_data) - 9, 9):
                    # 获取前9小时的18个污染物数据
                    x_sample = all_test_data[i:i+9].flatten()
                    test_x.append(x_sample)
                
                # 转换为numpy数组
                test_x = np.array(test_x)
                
                # 添加偏置项
                test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)
                
                print(f"✓ 测试数据加载完成")
                print(f"  测试集大小: {test_x.shape}")
                print(f"  特征维度: {test_x.shape[1]}")
                print(f"  样本数: {test_x.shape[0]}")
                
                return test_x
        
        except FileNotFoundError:
            print(f"✗ 错误：找不到文件 {test_file}")
            raise
    
    def predict(self, test_x):
        """
        使用训练好的模型进行预测
        
        伪代码:
        {
        for every 18 rows:
            test_x.append([1])
            test_x.append(9-hr data)
            test_y = np.dot(weight vector, test_x)
        }
        """
        if self.w is None:
            raise ValueError("模型未训练或未加载。请先训练或加载模型。")
        
        print("\n" + "="*60)
        print("进行PM2.5预测...")
        print("="*60)
        
        # 进行预测
        predictions = np.dot(test_x, self.w)
        
        print(f"✓ 预测完成")
        print(f"  预测样本数: {len(predictions)}")
        print(f"  平均预测值: {np.mean(predictions):.2f}")
        print(f"  预测值范围: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
        
        return predictions
    
    def save_predictions(self, predictions, output_file='data/predict.csv'):
        """
        保存预测结果到CSV文件
        格式: ["id","value"]
        """
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        print("\n" + "="*60)
        print("保存预测结果...")
        print("="*60)
        
        with open(output_file, 'w', newline='') as text:
            s = csv.writer(text, delimiter=',', lineterminator='\n')
            
            # 写入标题
            s.writerow(["id", "value"])
            
            # 写入预测结果
            for i, pred in enumerate(predictions):
                s.writerow([i, pred[0]])
        
        print(f"✓ 预测结果已保存: {output_file}")
        print(f"  共保存 {len(predictions)} 条预测结果")


def main():
    """主函数：完整的PM2.5预测流程"""
    
    print("\n" + "="*80)
    print(" "*20 + "实验一：线性回归预测PM2.5值")
    print(" "*15 + "使用Adagrad梯度下降方法")
    print("="*80)
    
    # 检查数据文件是否存在
    if not os.path.exists('data'):
        print("✗ 错误：找不到 data 文件夹")
        print("请确保在 Expt1 文件夹下有 data 子文件夹，包含 train.csv 和 test.csv")
        return
    
    # 1. 创建预测器
    predictor = PM25Predictor(
        learning_rate=0.01,      # 初始学习率
        iterations=10000,        # 迭代次数
        epsilon=1e-8            # 平滑项
    )
    
    # 2. 加载训练数据
    try:
        train_x, train_y = predictor.load_train_data('data/train.csv')
    except Exception as e:
        print(f"✗ 加载训练数据失败: {e}")
        return
    
    # 3. 训练模型
    try:
        loss_history = predictor.train(train_x, train_y)
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return
    
    # 4. 保存模型
    try:
        predictor.save_model('model.npy')
    except Exception as e:
        print(f"✗ 模型保存失败: {e}")
        return
    
    # 5. 加载测试数据
    try:
        test_x = predictor.load_test_data('data/test.csv')
    except Exception as e:
        print(f"✗ 加载测试数据失败: {e}")
        return
    
    # 6. 进行预测
    try:
        predictions = predictor.predict(test_x)
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        return
    
    # 7. 保存预测结果
    try:
        predictor.save_predictions(predictions, 'data/predict.csv')
    except Exception as e:
        print(f"✗ 保存预测结果失败: {e}")
        return
    
    # 打印完成信息
    print("\n" + "="*80)
    print(" "*25 + "✓ 实验完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  - model.npy          : 训练好的模型")
    print("  - data/predict.csv   : PM2.5预测结果")
    print("\n提示:")
    print("  - 如需调整模型，修改学习率、迭代次数等参数")
    print("  - 查看 loss_history 可以分析训练过程")
    print("="*80)


if __name__ == '__main__':
    main()
