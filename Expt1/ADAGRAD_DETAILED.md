# Adagrad梯度下降法详解

## 目录
1. [背景介绍](#背景介绍)
2. [线性回归基础](#线性回归基础)
3. [梯度下降算法](#梯度下降算法)
4. [Adagrad优化器](#adagrad优化器)
5. [数据处理流程](#数据处理流程)
6. [实现细节](#实现细节)
7. [性能分析](#性能分析)

## 背景介绍

### 问题定义
给定PM2.5监测站的历史数据，使用前9小时的18种污染物观测数据来预测第10小时的PM2.5浓度。

**问题性质**：监督学习 - 回归问题
**特征维度**：162 (9小时 × 18污染物)
**目标维度**：1 (单个PM2.5值)
**样本数量**：5652 (训练) + 240 (测试)

## 线性回归基础

### 模型定义

线性回归假设特征与目标之间存在线性关系：

$$\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$

或用矩阵形式表示：

$$\hat{y} = Xw$$

其中：
- $X$ 是特征矩阵，形状为 $(m, n)$，$m$为样本数，$n$为特征数
- $w$ 是权重向量，形状为 $(n, 1)$
- $\hat{y}$ 是预测值向量，形状为 $(m, 1)$

### 损失函数

使用均方误差（Mean Squared Error, MSE）作为损失函数：

$$L(w) = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 = \frac{1}{m}||y - Xw||_2^2$$

其中：
- $y$ 是真实值向量
- $\hat{y}$ 是预测值向量
- $m$ 是样本数

### 梯度计算

对损失函数关于权重向量求梯度：

$$\nabla L(w) = -\frac{2}{m}X^T(y - Xw) = \frac{2}{m}X^T(Xw - y)$$

简化形式（不考虑常数）：

$$\nabla L(w) = 2X^T(Xw - y)$$

## 梯度下降算法

### 标准梯度下降（Batch Gradient Descent）

**算法伪代码**：
```python
for iteration in range(num_iterations):
    # 前向传播：计算预测值
    y_pred = X @ w
    
    # 计算损失
    loss = y_pred - y
    
    # 计算梯度
    gradient = 2 * (X.T @ loss)
    
    # 梯度更新
    w = w - learning_rate * gradient
```

**数学表达**：
$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

其中 $\alpha$ 是固定的学习率。

### 梯度下降的问题

1. **学习率难以设置**
   - 学习率过大：震荡，无法收敛
   - 学习率过小：收敛缓慢

2. **对所有参数使用相同学习率**
   - 不适应数据的稀疏性
   - 对于频繁更新的参数应该用更小的学习率

3. **容易陷入局部最优**（对于非凸问题）

## Adagrad优化器

### 核心思想

**Adagrad** = **Adaptive Gradient Descent**

主要特点：
- **自适应学习率**：为不同的参数自动调整学习率
- **基于历史梯度**：参数更新次数多的学习率更小
- **参数独立**：每个参数都有自己的学习率

### 算法原理

#### 学习率调整机制

定义第 $t$ 时刻的累积梯度平方和：

$$g_t = \sum_{i=1}^{t} (\nabla L(w_i))^2$$

然后自适应学习率为：

$$\alpha_t = \frac{\alpha}{\sqrt{g_t + \epsilon}}$$

其中：
- $\alpha$ 是初始学习率（通常0.01）
- $\epsilon$ 是平滑项（通常 $1e^{-8}$），防止分母为0
- $g_t$ 是梯度平方的累积和

#### 权重更新规则

$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{g_t + \epsilon}} \odot \nabla L(w_t)$$

其中 $\odot$ 表示逐元素乘法。

### 向量化实现

在实际实现中，对整个梯度向量进行操作：

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

### 数学证明

**为什么Adagrad更好？**

考虑两个参数的更新：
- 参数1：之前被更新过100次，梯度历史平方和大
- 参数2：之前被更新过1次，梯度历史平方和小

Adagrad会：
- 参数1：使用较小的学习率 $\alpha / \sqrt{100+\epsilon} \approx \alpha/10$
- 参数2：使用较大的学习率 $\alpha / \sqrt{1+\epsilon} \approx \alpha$

这种自适应能力特别适合**稀疏数据**。

### Adagrad的优缺点

**优点**：
- ✅ 自动调整学习率，适应数据
- ✅ 对稀疏梯度表现优异
- ✅ 无需手动调整学习率
- ✅ 收敛速度快

**缺点**：
- ❌ 学习率单调递减，可能过早停止
- ❌ 累积梯度平方可能导致学习率最终趋近于0
- ❌ 计算复杂度略高（需保存累积梯度）

## 数据处理流程

### 步骤1：原始数据加载

**源文件格式**（train.csv 和 test.csv）
```
日期, 时间, 18列污染物观测值
```

**处理方式**：
- 使用 big5 编码读取（支持繁体字）
- 跳过缺失值标记 (NR)
- 转换为浮点数

### 步骤2：数据规整化

**采样策略**：每10小时为一个样本

```
原始时间序列：
时刻:  0   1   2   3   4   5   6   7   8   9  10  11  ...
值:   v0  v1  v2  v3  v4  v5  v6  v7  v8  v9 v10 v11  ...

样本1：x = [v0, v1, ..., v8]，y = v9
样本2：x = [v1, v2, ..., v9]，y = v10
样本3：x = [v2, v3, ..., v10]，y = v11
...
```

### 步骤3：特征矩阵构建

对每个样本：
- 使用前9小时×18污染物 = 162维特征
- 添加偏置项（第一列全为1）
- 最终特征维度 = 163

**矩阵形状**：
```
训练集：X_train (5652, 163)，y_train (5652, 1)
测试集：X_test  (240, 163),  y_test  (240, 1)
```

### 步骤4：特征选择

我们使用所有18种污染物的前9小时数据：
```
污染物列表：
1. AMB_TEMP    (环境温度)
2. CH4         (甲烷)
3. CO          (一氧化碳)
4. NHMC        (非甲烷烃)
5. NO          (一氧化氮)
6. NO2         (二氧化氮)
7. NOx         (氮氧化物)
8. O3          (臭氧)
9. PM10        (PM10)
10. PM2.5      (PM2.5) ← 目标变量
11. RAINFALL   (降雨)
12. RH         (相对湿度)
13. SO2        (二氧化硫)
14. THC        (总烃)
15. WD_HR      (风向小时)
16. WIND_DIREC (风向角度)
17. WIND_SPEED (风速)
18. WS_HR      (风速小时)
```

## 实现细节

### 类设计

```python
class PM25Predictor:
    def __init__(self, learning_rate=0.01, iterations=10000, epsilon=1e-8)
    def load_train_data(train_file) -> (train_x, train_y)
    def train(train_x, train_y) -> loss_history
    def save_model(model_path)
    def load_model(model_path)
    def load_test_data(test_file) -> test_x
    def predict(test_x) -> predictions
    def save_predictions(predictions, output_file)
```

### 关键函数解析

#### 1. 梯度计算

```python
# 前向传播
y_pred = np.dot(train_x, self.w)

# 计算预测误差
loss = y_pred - train_y

# 梯度 = 2 * X^T * (y_pred - y)
gradient = 2 * np.dot(train_x.T, loss)
```

**数学验证**：
- 损失函数：$L = ||Xw - y||_2^2$
- 求导：$\frac{\partial L}{\partial w} = 2X^T(Xw - y)$ ✓

#### 2. Adagrad更新

```python
# 累积梯度平方
self.prev_gra += gradient ** 2

# 自适应学习率
ada = np.sqrt(self.prev_gra) + self.epsilon

# 权重更新
self.w -= self.learning_rate * gradient / ada
```

**数学验证**：
- 累积：$g_t = g_{t-1} + \nabla_t^2$ ✓
- 自适应：$\frac{\alpha}{\sqrt{g_t + \epsilon}}$ ✓

#### 3. 损失计算

```python
# 均方误差
mse = np.sum(loss**2) / len(train_x)

# 均方根误差
rmse = math.sqrt(mse)
```

**数学验证**：
- MSE：$\frac{1}{m}\sum_i (y_i - \hat{y}_i)^2$ ✓
- RMSE：$\sqrt{MSE}$ ✓

## 性能分析

### 时间复杂度

单次迭代的计算：
```
前向传播：O(m*n)
梯度计算：O(m*n)
权重更新：O(n)
整体：O(m*n)
```

总体时间复杂度：$O(iterations \times m \times n)$

对于本实验：
- $iterations = 10000$
- $m = 5652$（样本数）
- $n = 163$（特征数）
- 总操作：$\approx 9.2 \times 10^9$

**预期耗时**：2-5分钟（取决于CPU性能）

### 空间复杂度

```
权重向量 w：O(n) = O(163)
累积梯度 prev_gra：O(n) = O(163)
特征矩阵 X：O(m*n) = O(5652*163) ≈ 1M浮点数
训练数据 y：O(m) = O(5652)

总体：O(m*n) ≈ 1GB（取决于精度）
```

### 收敛性分析

**收敛速度**：

Adagrad相比标准梯度下降：
- 初期收敛快（学习率大）
- 后期收敛慢（学习率递减）
- 总体收敛速度：超线性收敛

**收敛条件**：
- 通常在 5000-10000 次迭代后收敛
- RMSE 梯度接近0
- 损失不再显著下降

### 性能指标

**典型结果**（基于算法分析）：

| 指标 | 值 |
|------|-----|
| 训练集大小 | 5652 |
| 特征维度 | 163 |
| 最终RMSE | 5-15 |
| 收敛迭代 | 8000-10000 |
| 平均PM2.5 | 20-40 |

### 与其他方法的对比

| 方法 | 学习率 | 收敛速度 | 内存 | 适合场景 |
|------|--------|---------|------|---------|
| 标准GD | 固定 | 慢 | 低 | 稠密数据 |
| SGD | 固定 | 中等 | 低 | 大规模数据 |
| Adagrad | 自适应 | 快 | 中等 | 稀疏数据，本实验 |
| RMSprop | 自适应 | 快 | 中等 | 非凸优化 |
| Adam | 自适应 | 最快 | 高 | 深度学习 |

## 梯度下降的可视化

### 1维情况下的更新过程

```
初始权重：w = 0

迭代过程：
Iteration 1:
  y_pred = 0
  loss = 0 - y = -y
  gradient = 2*X^T*(-y)（某个值）
  prev_gra = gradient^2
  w = w - (learning_rate/sqrt(prev_gra)) * gradient

Iteration 2:
  y_pred = X*w（新值）
  loss = y_pred - y（减小）
  gradient = 2*X^T*loss（减小）
  prev_gra += gradient^2（继续累积）
  学习率 = learning_rate / sqrt(prev_gra)（变小）
  w = w - learning_rate * gradient（更新）

...继续迭代...

收敛时：
  gradient ≈ 0
  loss 最小
```

### RMSE下降曲线特征

```
RMSE
 ↑
45├─ 初始状态
  │ ╱
40├ ╱ 快速下降（学习率较大）
  │╱
15├ ┌ 中速下降
  │ └──
10│     ─── 缓慢下降（学习率递减）
  │         ──
 8├         ──┐
  │           ├ 趋于收敛
  │           │
 0└───────────┴─────→ 迭代次数
  0       5000   10000
```

## 数学背景

### 相关理论

1. **凸优化理论**
   - 线性回归的MSE是凸函数
   - Adagrad保证收敛到全局最优

2. **随机矩阵理论**
   - 累积梯度矩阵的条件数变化
   - 解释为什么Adagrad改进收敛

3. **正则化角度**
   - Adagrad中的 $\epsilon$ 类似于L2正则化
   - 防止参数剧烈振荡

## 参考文献

1. Duchi, J., Hazan, E., & Singer, Y. (2011). 
   "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization."
   Journal of Machine Learning Research, 12(7).

2. Bottou, L. (2010).
   "Large-Scale Machine Learning with Stochastic Gradient Descent."
   COMPSTAT 2010.

3. Kingma, D. P., & Ba, J. (2014).
   "Adam: A Method for Stochastic Optimization."
   ArXiv preprint arXiv:1412.6980.

---

**版本**: 1.0
**更新日期**: 2024-12-12
**适用范围**: 实验一 - 线性回归预测PM2.5值
