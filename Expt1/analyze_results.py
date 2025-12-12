"""
PM2.5预测分析和可视化工具

这个脚本提供了对模型训练过程和预测结果的可视化分析
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from pathlib import Path


class PM25Analysis:
    """PM2.5预测结果分析类"""
    
    @staticmethod
    def load_predictions(predict_file='data/predict.csv'):
        """加载预测结果"""
        predictions = []
        
        try:
            with open(predict_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题
                for row in reader:
                    if len(row) >= 2:
                        try:
                            predictions.append(float(row[1]))
                        except:
                            pass
        except FileNotFoundError:
            print(f"文件 {predict_file} 不存在")
            return None
        
        return np.array(predictions)
    
    @staticmethod
    def load_test_data(test_file='data/test.csv'):
        """加载测试数据（用于获取实际的PM2.5值）"""
        pm25_values = []
        
        try:
            with open(test_file, 'r') as f:
                reader = csv.reader(f)
                line_count = 0
                
                for row in reader:
                    if len(row) >= 10:
                        try:
                            # PM2.5通常在第10列（索引9）
                            pm25 = float(row[9])
                            # 每9行取一个（采样方式）
                            if line_count % 9 == 8:  # 每第9行
                                pm25_values.append(pm25)
                            line_count += 1
                        except:
                            line_count += 1
        except FileNotFoundError:
            print(f"文件 {test_file} 不存在")
            return None
        
        return np.array(pm25_values)
    
    @staticmethod
    def calculate_statistics(predictions):
        """计算预测结果的统计信息"""
        
        stats = {
            'count': len(predictions),
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'median': np.median(predictions),
            'q1': np.percentile(predictions, 25),
            'q3': np.percentile(predictions, 75),
            'iqr': np.percentile(predictions, 75) - np.percentile(predictions, 25)
        }
        
        return stats
    
    @staticmethod
    def calculate_metrics(true_values, pred_values):
        """计算预测指标"""
        
        # 确保数据对齐
        min_len = min(len(true_values), len(pred_values))
        true_values = true_values[:min_len]
        pred_values = pred_values[:min_len]
        
        # 均方误差
        mse = np.mean((true_values - pred_values) ** 2)
        rmse = math.sqrt(mse)
        
        # 平均绝对误差
        mae = np.mean(np.abs(true_values - pred_values))
        
        # R平方分数
        ss_res = np.sum((true_values - pred_values) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 平均绝对百分比误差
        mape = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-8))) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r_squared,
            'MAPE': mape
        }
    
    @staticmethod
    def print_statistics(predictions, true_values=None):
        """打印统计信息"""
        
        print("\n" + "="*70)
        print(" "*20 + "预测结果统计分析")
        print("="*70)
        
        stats = PM25Analysis.calculate_statistics(predictions)
        
        print("\n基本统计信息:")
        print(f"  样本数量:        {stats['count']}")
        print(f"  平均值:          {stats['mean']:.4f}")
        print(f"  标准差:          {stats['std']:.4f}")
        print(f"  中位数:          {stats['median']:.4f}")
        print(f"  最小值:          {stats['min']:.4f}")
        print(f"  最大值:          {stats['max']:.4f}")
        print(f"  第一四分位数:     {stats['q1']:.4f}")
        print(f"  第三四分位数:     {stats['q3']:.4f}")
        print(f"  四分位距:        {stats['iqr']:.4f}")
        
        # 如果有真实值，计算误差指标
        if true_values is not None and len(true_values) > 0:
            print("\n预测性能指标:")
            metrics = PM25Analysis.calculate_metrics(true_values, predictions)
            print(f"  均方误差(MSE):   {metrics['MSE']:.4f}")
            print(f"  均方根误差(RMSE): {metrics['RMSE']:.4f}")
            print(f"  平均绝对误差(MAE): {metrics['MAE']:.4f}")
            print(f"  决定系数(R²):    {metrics['R²']:.4f}")
            print(f"  平均绝对百分比误差(MAPE): {metrics['MAPE']:.2f}%")
        
        print("="*70)
    
    @staticmethod
    def plot_predictions(predictions, title="PM2.5预测结果分布"):
        """绘制预测结果直方图"""
        
        plt.figure(figsize=(12, 5))
        
        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('PM2.5预测值')
        plt.ylabel('频数')
        plt.title(title)
        plt.grid(axis='y', alpha=0.3)
        
        # 箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot(predictions, vert=True)
        plt.ylabel('PM2.5预测值')
        plt.title('预测值箱线图')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
        print("\n✓ 已保存图表: prediction_distribution.png")
        plt.close()
    
    @staticmethod
    def plot_comparison(true_values, pred_values, title="真实值vs预测值"):
        """绘制真实值vs预测值对比"""
        
        min_len = min(len(true_values), len(pred_values))
        true_values = true_values[:min_len]
        pred_values = pred_values[:min_len]
        
        plt.figure(figsize=(14, 5))
        
        # 散点图
        plt.subplot(1, 2, 1)
        plt.scatter(true_values, pred_values, alpha=0.5, s=20)
        
        # 添加完美预测线
        min_val = min(true_values.min(), pred_values.min())
        max_val = max(true_values.max(), pred_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
        
        plt.xlabel('真实PM2.5值')
        plt.ylabel('预测PM2.5值')
        plt.title('散点图对比')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 时间序列对比
        plt.subplot(1, 2, 2)
        x = np.arange(min_len)
        plt.plot(x, true_values, 'b-', label='真实值', alpha=0.7)
        plt.plot(x, pred_values, 'r-', label='预测值', alpha=0.7)
        plt.xlabel('样本索引')
        plt.ylabel('PM2.5值')
        plt.title('时间序列对比')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
        print("✓ 已保存图表: comparison.png")
        plt.close()
    
    @staticmethod
    def plot_residuals(true_values, pred_values, title="预测残差分析"):
        """绘制残差分析图"""
        
        min_len = min(len(true_values), len(pred_values))
        true_values = true_values[:min_len]
        pred_values = pred_values[:min_len]
        
        residuals = true_values - pred_values
        
        plt.figure(figsize=(14, 5))
        
        # 残差散点图
        plt.subplot(1, 2, 1)
        plt.scatter(pred_values, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差散点图')
        plt.grid(alpha=0.3)
        
        # 残差直方图
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('残差值')
        plt.ylabel('频数')
        plt.title('残差分布')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residuals.png', dpi=150, bbox_inches='tight')
        print("✓ 已保存图表: residuals.png")
        plt.close()
        
        return residuals


def analyze_predictions(predict_file='data/predict.csv', test_file='data/test.csv'):
    """完整的预测分析流程"""
    
    print("\n" + "="*70)
    print(" "*15 + "PM2.5预测结果分析")
    print("="*70)
    
    # 加载预测结果
    print("\n加载预测结果...")
    predictions = PM25Analysis.load_predictions(predict_file)
    
    if predictions is None:
        print("✗ 无法加载预测结果，请先运行主训练脚本")
        return
    
    print(f"✓ 已加载 {len(predictions)} 个预测结果")
    
    # 尝试加载真实值
    print("\n尝试加载真实值...")
    true_values = PM25Analysis.load_test_data(test_file)
    
    if true_values is not None and len(true_values) > 0:
        print(f"✓ 已加载 {len(true_values)} 个真实PM2.5值")
        has_true = True
    else:
        print("⚠ 无法加载真实值，将进行单变量分析")
        has_true = False
    
    # 打印统计信息
    PM25Analysis.print_statistics(predictions, true_values if has_true else None)
    
    # 绘制分布图
    print("\n生成可视化图表...")
    PM25Analysis.plot_predictions(predictions)
    
    # 如果有真实值，进行对比分析
    if has_true and len(true_values) == len(predictions):
        PM25Analysis.plot_comparison(true_values, predictions)
        PM25Analysis.plot_residuals(true_values, predictions)
    
    print("\n" + "="*70)
    print("✓ 分析完成！")
    print("="*70)


if __name__ == '__main__':
    analyze_predictions()
