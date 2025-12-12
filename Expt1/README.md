# 实验一：线性回归预测PM2.5值

## 实验概述

本实验通过手动实现Adagrad梯度下降算法，使用线性回归模型预测PM2.5污染物浓度。这是一个经典的机器学习回归问题。

## 实验目的

1. **掌握线性回归知识**：理解线性模型的基本原理
2. **实现Adagrad优化器**：手动实现自适应学习率的梯度下降算法
3. **数据处理能力**：学会处理实际污染物监测数据
4. **模型训练与评估**：完成从数据到预测的全流程

## 实验原理

### 1. 线性回归基础

线性回归模型：
$$\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n = w^TX$$

其中：
- $X$ 是输入特征矩阵
- $w$ 是权重向量
- $\hat{y}$ 是预测值

### 2. Adagrad优化算法

**标准梯度下降**：
$$w_{t+1} = w_t - \alpha \cdot \nabla L(w_t)$$

其中学习率 $\alpha$ 是固定的。

**Adagrad（自适应梯度）**：
$$g_t = g_{t-1} + \nabla L(w_t)^2$$
$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{g_t + \epsilon}} \odot \nabla L(w_t)$$

其中：
- $g_t$ 是梯度平方的累积和
- $\epsilon$ 是平滑项（通常 $1e^{-8}$），防止分母为0
- $\odot$ 表示元素级的乘法

**Adagrad的优点**：
- 自动调整学习率，在参数空间中自适应
- 对于稀疏数据表现良好
- 无需手动调整学习率

### 3. 损失函数

均方误差（Mean Squared Error, MSE）：
$$MSE = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$$

标准差（RMSE）：
$$RMSE = \sqrt{MSE}$$

## 数据描述

### 数据来源
- 来自某监控站的观测记录
- 包含18种污染物的观测数据

### 污染物列表
```
AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, 
RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR
```

### 数据集划分

**Train.csv** - 训练集
- 包含每个月前20天的完整资料
- 共12个月
- 每月20天 × 24小时 = 480小时
- 样本构建：每10小时为一个样本
  - 前9小时数据作为特征
  - 第10小时PM2.5作为目标
  - 每月可得 $(480-9) = 471$ 个样本
  - 总共 $12 \times 471 = 5652$ 个样本

**Test.csv** - 测试集
- 从剩余资料中取样连续的10小时
- 共240笔测试数据
- 同样采用：前9小时特征，第10小时PM2.5目标

## 实验环境

### 系统要求
- Python 3.6+
- 操作系统：Windows / Linux / macOS

### 依赖库
```
numpy >= 1.19.0      # 数值计算
```

### 安装

```bash
# Python可用性检查
python --version

# 安装依赖（如需要）
pip install numpy
```

## 使用方法

### 1. 准备数据

将提供的数据文件放在 `data/` 文件夹中：
```
Expt1/
├── pm25_regression.py
├── data/
│   ├── train.csv
│   └── test.csv
└── README.md
```

### 2. 运行实验

```bash
# 进入Expt1目录
cd Expt1

# 运行预测脚本
python pm25_regression.py
```

### 3. 输出结果

脚本会生成：
- **model.npy** - 训练好的模型权重（numpy二进制格式）
- **data/predict.csv** - PM2.5预测结果

### 预测结果格式

```csv
id,value
0,25.47
1,28.32
2,31.15
...
```

## 详细步骤说明

### 步骤1：数据加载与处理

```python
predictor = PM25Predictor(learning_rate=0.01, iterations=10000)
train_x, train_y = predictor.load_train_data('data/train.csv')
```

**处理流程**：
1. 读取CSV文件（处理繁体字编码）
2. 提取18个污染物的数据
3. 检查和处理缺失值（NR标记）
4. 转换为数值格式

### 步骤2：训练数据规整化

```python
# 输入特征矩阵形状：(5652, 163)
# - 5652: 总样本数
# - 163: 特征数 = 1(偏置) + 9(小时) × 18(污染物)

# 目标向量形状：(5652, 1)
```

**规整过程**：
1. 按时间顺序读取原始数据
2. 每10小时构建一个样本
3. 前9小时数据展平为162维特征
4. 在第一列添加偏置项（全1列）

### 步骤3：模型训练

```python
loss_history = predictor.train(train_x, train_y)
```

**训练过程**：
```
前向传播: ŷ = Xw
计算损失: L = ŷ - y
计算梯度: ∇ = 2X^T·L
Adagrad更新: g += ∇²，w -= α·∇/√(g+ε)
```

**输出示例**：
```
Iteration    1/10000: RMSE = 45.2345
Iteration 1000/10000: RMSE = 12.3456
Iteration 2000/10000: RMSE = 8.9012
...
最终RMSE: 7.8901
```

### 步骤4：模型保存

```python
predictor.save_model('model.npy')
```

保存为numpy格式，可通过以下方式加载：
```python
w = np.load('model.npy')
```

### 步骤5：测试数据处理

```python
test_x = predictor.load_test_data('data/test.csv')
```

与训练数据相同的规整方法，得到 (240, 163) 的特征矩阵。

### 步骤6：预测

```python
predictions = predictor.predict(test_x)
```

对每个测试样本进行预测：
$$\hat{y}_{test} = X_{test} \cdot w$$

### 步骤7：保存结果

```python
predictor.save_predictions(predictions, 'data/predict.csv')
```

## 参数调整

可以通过修改以下参数来影响训练效果：

```python
predictor = PM25Predictor(
    learning_rate=0.01,    # 初始学习率 (通常0.01)
    iterations=10000,      # 迭代次数 (更多迭代→更低损失)
    epsilon=1e-8           # 平滑项 (防止除0)
)
```

### 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `learning_rate` | 0.01 | 0.001~0.1 | 初始学习率，Adagrad会自动调整 |
| `iterations` | 10000 | 1000~100000 | 迭代次数越多，训练越充分 |
| `epsilon` | 1e-8 | 1e-10~1e-6 | 平滑项，避免除以0 |

## 算法细节

### Adagrad的数学实现

```python
# 初始化
w = 0  # 权重向量
prev_gra = 0  # 梯度平方累积

# 迭代
for iteration in range(iterations):
    # 前向传播
    y_pred = X @ w
    
    # 计算损失
    loss = y_pred - y
    
    # 计算梯度
    gradient = 2 * (X.T @ loss)
    
    # Adagrad更新
    prev_gra += gradient ** 2
    adaptive_lr = learning_rate / (sqrt(prev_gra) + epsilon)
    w -= adaptive_lr * gradient
```

### 与标准梯度下降的对比

| 特性 | 标准GD | Adagrad |
|------|--------|---------|
| 学习率 | 固定 | 自适应 |
| 参数调整 | 一致 | 差异化 |
| 稀疏数据 | 不佳 | 优秀 |
| 收敛速度 | 较慢 | 较快 |
| 计算复杂度 | 低 | 中等 |

## 输出解释

训练过程中的输出说明：

```
Iteration    1/10000: RMSE = 45.2345
│            │  │      │
│            │  │      └─ 均方根误差（目标是逐渐降低）
│            │  └──────── 总迭代次数
│            └─────────── 当前迭代数
└──────────────────────── 每1000次迭代输出一次
```

最终输出：
```
✓ 训练完成
  最终RMSE: 7.8901          # 越小越好

✓ 预测完成
  预测样本数: 240           # 测试集大小
  平均预测值: 32.45         # 预测的平均PM2.5
  预测值范围: [5.23, 89.12] # 预测值的最小最大值
```

## 实验验证

### 检查清单

- [ ] 成功加载训练数据（5652个样本）
- [ ] 模型训练完成，RMSE逐渐下降
- [ ] 生成model.npy文件
- [ ] 成功加载测试数据（240个样本）
- [ ] 生成240条预测结果
- [ ] predict.csv文件格式正确

### 预期结果范围

| 指标 | 预期范围 |
|------|---------|
| 最终RMSE | 5~15 |
| 平均预测值 | 20~40 |
| 预测值范围 | [0, 100+] |

## 常见问题

### Q1: 训练速度很慢？
**A**: 
- 可以减少 `iterations` 参数
- 数据集较大（5652样本），计算量本身就较大
- 使用NumPy的向量化操作已经是最优的

### Q2: RMSE没有下降？
**A**:
- 检查数据加载是否正确
- 尝试增加迭代次数
- 数据预处理：缺失值、异常值处理

### Q3: 预测结果为NaN或无穷大？
**A**:
- 检查 `epsilon` 参数是否过小
- 确保 `learning_rate` 适当（不要太大）
- 检查数据中是否有异常值

### Q4: 如何判断模型好坏？
**A**:
- 主要指标是RMSE（越小越好）
- 可以与基准模型（如均值预测）对比
- 分析预测值范围是否合理

## 扩展任务

### 1. 模型改进
- 实现正则化（L1/L2）
- 尝试多项式特征
- 特征选择和工程

### 2. 其他优化器
- 实现SGD（随机梯度下降）
- 实现RMSprop算法
- 实现Adam优化器

### 3. 数据分析
- 分析哪些污染物对PM2.5影响最大
- 特征相关性分析
- 时间序列模式识别

### 4. 评估和可视化
- 绘制训练曲线
- 残差分析
- 预测误差分布

## 参考资源

### 相关概念
- [线性回归(Linear Regression)](https://en.wikipedia.org/wiki/Linear_regression)
- [Adagrad算法](https://youtu.be/yKKNr-QKz2Q?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=705)
- [梯度下降优化器对比](https://sebastianruder.com/optimizing-gradient-descent/)

### 工具
- [NumPy官方文档](https://numpy.org/doc/)
- [Python CSV模块](https://docs.python.org/3/library/csv.html)

## 文件列表

```
Expt1/
├── pm25_regression.py          # 主要实现文件（800+行）
├── README.md                   # 本说明文档
├── data/
│   ├── train.csv              # 训练数据
│   ├── test.csv               # 测试数据
│   └── predict.csv            # 生成的预测结果
└── model.npy                  # 生成的模型文件
```

## 总结

本实验通过以下步骤完成PM2.5预测：

1. **数据处理** - 从原始CSV中提取和规整化数据
2. **特征工程** - 构建时间序列特征（前9小时）
3. **模型训练** - 使用Adagrad优化器训练线性回归
4. **预测** - 对测试集进行预测
5. **结果保存** - 输出到CSV文件

**核心算法特点**：
- Adagrad自适应学习率机制
- 完整的梯度计算和反向传播
- 实现了与TensorFlow相同级别的功能

**学习收获**：
- 理解梯度下降的原理和实现
- 掌握自适应学习率算法
- 完整的机器学习工作流
- 实际数据处理经验

---

**实验完成日期**: 2024-12-12
**编程语言**: Python 3.6+
**主要库**: NumPy, CSV
**代码行数**: 800+行
