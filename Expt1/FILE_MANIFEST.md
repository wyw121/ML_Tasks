# 实验一 - 文件清单与使用指南

## 📁 完整文件结构

```
Expt1/
│
├── 📄 pm25_regression.py          ★ 主程序 (800行)
│   └── 完整的PM2.5预测实现
│       • 数据加载与处理
│       • Adagrad优化器实现
│       • 模型训练与保存
│       • 预测结果输出
│
├── 📄 analyze_results.py          ★ 分析工具 (400行)
│   └── 预测结果的可视化分析
│       • 统计信息计算
│       • 图表生成
│       • 性能指标评估
│
├── 📄 README.md                   ★ 完整文档 (400行)
│   └── 详细的实验说明
│       • 实验目的和原理
│       • 详细步骤说明
│       • 参数调整指南
│       • 常见问题解答
│
├── 📄 QUICKSTART.md              ★ 快速开始 (200行)
│   └── 3步快速上手
│       • 数据准备
│       • 运行程序
│       • 查看结果
│
├── 📄 ADAGRAD_DETAILED.md        ★ 算法讲解 (500行)
│   └── Adagrad深度解析
│       • 线性回归基础
│       • 梯度下降原理
│       • Adagrad算法细节
│       • 数学推导
│
├── 📄 requirements.txt            ★ 依赖清单
│   └── numpy >= 1.19.0
│       matplotlib >= 3.3.0 (可选)
│
├── 📁 data/
│   ├── train.csv                  # 训练数据 (12个月)
│   ├── test.csv                   # 测试数据 (240样本)
│   └── predict.csv                # 预测结果 (运行后生成)
│
└── 📄 model.npy                   # 模型文件 (运行后生成)
```

## 📋 文件详细说明

### 1. pm25_regression.py (800行) - 核心程序

**功能**: 完整的PM2.5线性回归预测系统

**类结构**:
```python
class PM25Predictor:
    __init__(learning_rate, iterations, epsilon)
    load_train_data(train_file)        # 加载训练数据
    _process_raw_data(train_file)      # 步骤1：数据处理
    _prepare_training_data(data)       # 步骤2：数据规整化
    train(train_x, train_y)            # 步骤3：模型训练
    save_model(model_path)             # 步骤4：保存模型
    load_test_data(test_file)          # 步骤5：加载测试数据
    predict(test_x)                    # 步骤6：进行预测
    save_predictions(predictions, ...)  # 步骤7：保存结果

def main():                            # 主函数：运行完整流程
```

**使用方法**:
```bash
python pm25_regression.py
```

**输入文件**:
- data/train.csv - 训练集数据
- data/test.csv - 测试集数据

**输出文件**:
- model.npy - 训练好的模型权重
- data/predict.csv - PM2.5预测结果

**关键参数**:
```python
predictor = PM25Predictor(
    learning_rate=0.01,    # 初始学习率
    iterations=10000,      # 迭代次数
    epsilon=1e-8          # 平滑项
)
```

---

### 2. analyze_results.py (400行) - 分析工具

**功能**: 对预测结果进行统计和可视化分析

**类结构**:
```python
class PM25Analysis:
    @staticmethod
    load_predictions(predict_file)         # 加载预测结果
    @staticmethod
    load_test_data(test_file)              # 加载真实值
    @staticmethod
    calculate_statistics(predictions)      # 计算统计信息
    @staticmethod
    calculate_metrics(true_values, pred)   # 计算性能指标
    @staticmethod
    print_statistics(predictions, true)    # 打印统计信息
    @staticmethod
    plot_predictions(predictions)          # 绘制分布图
    @staticmethod
    plot_comparison(true, pred)            # 绘制对比图
    @staticmethod
    plot_residuals(true, pred)             # 绘制残差图
```

**使用方法**:
```bash
python analyze_results.py
```

**生成文件**:
- prediction_distribution.png - 预测分布图
- comparison.png - 真实vs预测对比
- residuals.png - 残差分析图

**包含指标**:
- 基本统计：均值、标准差、最小值、最大值等
- 性能指标：MSE、RMSE、MAE、R²、MAPE
- 可视化图表：直方图、箱线图、散点图、时间序列

---

### 3. README.md (400行) - 完整文档

**章节组织**:
1. 实验概述
2. 实验目的（3个目标）
3. 实验原理（3个部分）
4. 数据描述（来源、污染物列表、数据集划分）
5. 实验环境（系统要求、依赖库）
6. 使用方法（4个步骤）
7. 详细步骤说明（7个步骤）
8. 参数调整指南
9. 算法细节
10. 输出解释
11. 实验验证
12. 常见问题
13. 扩展任务
14. 参考资源
15. 文件列表
16. 总结

**特点**:
- 400+行完整说明
- 包含数学公式
- 提供详细的步骤解释
- 包含常见问题解答
- 提供扩展任务建议

---

### 4. QUICKSTART.md (200行) - 快速开始指南

**核心内容**:

**3步快速上手**:
```bash
# 1. 准备数据 (data文件夹中要有train.csv和test.csv)
# 2. 运行程序
python pm25_regression.py
# 3. 查看结果 (data/predict.csv)
```

**章节**:
- 项目结构
- 快速开始（3步）
- 参数调整
- 性能预期
- 检查清单
- 常见问题
- 概念说明
- 学习路径
- 文件关系
- 优化建议
- 预计耗时

**特点**:
- 简明扼要
- 快速上手
- 包含预期输出
- 提供检查清单

---

### 5. ADAGRAD_DETAILED.md (500行) - 算法深度讲解

**深度讲解内容**:

**1. 线性回归基础**
- 模型定义与矩阵表示
- 损失函数（MSE）
- 梯度计算

**2. 标准梯度下降**
- 算法原理和伪代码
- 更新规则
- 存在的问题

**3. Adagrad优化器**
- 核心思想
- 学习率调整机制
- 权重更新规则
- 向量化实现
- 数学证明
- 优缺点分析

**4. 数据处理**
- 原始数据加载
- 数据规整化
- 特征矩阵构建
- 特征选择

**5. 实现细节**
- 类设计
- 关键函数解析
- 数学验证

**6. 性能分析**
- 时间复杂度：$O(iterations \times m \times n)$
- 空间复杂度：$O(m \times n)$
- 收敛性分析
- 与其他方法的对比

**7. 可视化与理论**
- 梯度下降过程可视化
- RMSE下降曲线
- 数学背景

**特点**:
- 深度的数学推导
- 完整的伪代码
- 详细的复杂度分析
- 与其他优化器的对比

---

### 6. requirements.txt - 依赖说明

```
# 核心依赖
numpy>=1.19.0          # 必需：数值计算

# 可选依赖
matplotlib>=3.3.0      # 用于analyze_results.py

# Python版本
Python >= 3.6
```

**安装方法**:
```bash
pip install -r requirements.txt
```

---

## 📊 数据文件说明

### train.csv - 训练集
```
├── 来源：某监控站12个月的观测数据
├── 内容：每个月前20天的完整资料
├── 大小：12个月 × 20天 × 24小时 × 18污染物
├── 处理后：5652个样本，每个样本包含前9小时的162维特征
└── 格式：CSV，包含18列污染物数据
```

### test.csv - 测试集
```
├── 来源：从剩余数据中取样的连续10小时
├── 样本数：240条独立的测试样本
├── 特征：前9小时的18种污染物观测
├── 目标：第10小时的PM2.5值
└── 格式：CSV，包含18列污染物数据
```

### predict.csv - 预测结果（生成）
```
id,value
0,25.47
1,28.32
...
239,31.15
```

### model.npy - 模型文件（生成）
```
├── 格式：NumPy二进制格式
├── 内容：权重向量，形状为(163, 1)
├── 用途：保存训练好的模型
└── 加载：w = np.load('model.npy')
```

---

## 🔄 工作流程

```
数据准备
  ↓
[train.csv, test.csv] 放在 data/ 目录
  ↓
执行主程序
  ↓
python pm25_regression.py
  ↓
├─→ 加载训练数据 (5652 样本)
├─→ 规整化特征 (163 维)
├─→ Adagrad 训练 (10000 迭代)
├─→ 保存模型 (model.npy)
├─→ 加载测试数据 (240 样本)
├─→ 进行预测
└─→ 保存结果 (predict.csv)
  ↓
查看结果
  ↓
python analyze_results.py (可选)
  ↓
├─→ 生成统计分析
├─→ 生成可视化图表
└─→ 输出性能指标
```

---

## 📈 主要指标含义

### 训练过程输出

| 指标 | 含义 | 期望值 |
|------|------|--------|
| RMSE | 均方根误差 | ↓ 逐渐下降 |
| Iteration | 当前迭代数 | 0~10000 |
| Loss | 预测误差 | ↓ 逐渐减小 |

### 预测结果统计

| 指标 | 含义 | 典型值 |
|------|------|--------|
| Count | 样本数 | 240 |
| Mean | 平均值 | 25~35 |
| Std | 标准差 | 10~20 |
| Min | 最小值 | 0~10 |
| Max | 最大值 | 60~100 |

### 性能指标

| 指标 | 公式 | 含义 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | 平均平方误差 |
| RMSE | $\sqrt{MSE}$ | 均方根误差 |
| MAE | $\frac{1}{n}\sum\|y-\hat{y}\|$ | 平均绝对误差 |
| R² | $1-\frac{SS_{res}}{SS_{tot}}$ | 决定系数（0~1） |
| MAPE | $\frac{1}{n}\sum\|\frac{y-\hat{y}}{y}\|$ | 平均绝对百分比误差 |

---

## 🎯 使用场景

### 场景1：快速验证
```bash
# 只需了解基本概念，快速运行
1. 阅读 QUICKSTART.md
2. 运行 pm25_regression.py
3. 检查 data/predict.csv
```

### 场景2：深入学习
```bash
# 想要理解算法细节
1. 阅读 README.md (完整说明)
2. 阅读 ADAGRAD_DETAILED.md (算法细节)
3. 修改 pm25_regression.py 的参数
4. 运行 analyze_results.py 进行分析
```

### 场景3：参数调优
```bash
# 要改进模型性能
1. 修改学习率、迭代次数等参数
2. 运行实验并记录RMSE
3. 使用 analyze_results.py 分析结果
4. 重复调整直到满意
```

### 场景4：扩展开发
```bash
# 想在此基础上添加新功能
1. 参考 pm25_regression.py 的结构
2. 添加新的方法或功能
3. 运行测试确保正常工作
```

---

## 📞 快速参考

### 常用命令

```bash
# 运行主程序
python pm25_regression.py

# 分析结果
python analyze_results.py

# 查看Python版本
python --version

# 安装依赖
pip install -r requirements.txt

# 查看模型参数
python -c "import numpy as np; w = np.load('model.npy'); print(w.shape)"
```

### 文件打开

```bash
# 查看预测结果
cat data/predict.csv          # Linux/Mac
type data\predict.csv         # Windows

# 编辑主程序
code pm25_regression.py       # VS Code
notepad pm25_regression.py    # Windows记事本
```

### 参数快速调整

```python
# 在 pm25_regression.py 中修改
predictor = PM25Predictor(
    learning_rate=0.01,    # 学习率：更小→更稳定，更大→更快
    iterations=10000,      # 迭代数：更多→效果更好，但耗时更长
    epsilon=1e-8           # 平滑项：很少需要改动
)
```

---

## ✅ 完整性检查清单

运行前检查：
- [ ] Python 3.6+ 已安装
- [ ] NumPy 已安装
- [ ] data/train.csv 存在
- [ ] data/test.csv 存在
- [ ] 当前目录在 Expt1 文件夹中

运行后检查：
- [ ] 没有报错信息
- [ ] model.npy 已生成
- [ ] data/predict.csv 已生成
- [ ] predict.csv 包含240行数据（+1行标题）
- [ ] 预测值不是 NaN 或无穷大

---

## 📚 学习资源

### 文档阅读顺序

1. **初学者** → QUICKSTART.md → README.md
2. **进阶者** → ADAGRAD_DETAILED.md → 修改代码
3. **研究者** → 参考论文 → 实现新的优化器

### 扩展阅读

- 线性回归：https://en.wikipedia.org/wiki/Linear_regression
- 梯度下降：https://www.deeplearningbook.org/
- Adagrad论文：https://jmlr.org/papers/v12/duchi11a.html

---

**最后更新**: 2024-12-12
**版本**: 1.0
**总代码行数**: 1200+ 行
**总文档行数**: 2000+ 行
