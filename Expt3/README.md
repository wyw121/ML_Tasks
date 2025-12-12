# 实验三：Logistic回归预测二分类

## 实验概述

使用Logistic回归实现年薪是否高于50K的二分类预测任务。这是一个经典的二分类问题，涉及数据预处理、特征工程、模型训练和评估。

## 实验目的

1. **理解Logistic回归**：掌握二分类问题的基本方法
2. **实现梯度下降**：手动实现梯度下降算法
3. **数据处理**：学会处理混合类型特征（离散+连续）
4. **模型评估**：使用验证集监控模型性能

## 数据集说明

### 任务
二分类任务：预测一个人的年收入是否超过5万美元

### 数据来源
Barry Becker从1994年人口普查数据库提取的数据

### 数据属性（14个属性）
```
1. age - 年龄（连续）
2. workclass - 工作类型（离散）
3. fnlwgt - 最终权重（连续）
4. education - 教育程度（离散）
5. education-num - 教育年数（连续）
6. marital-status - 婚姻状况（离散）
7. occupation - 职业（离散）
8. relationship - 关系（离散）
9. race - 种族（离散）
10. sex - 性别（离散）
11. capital-gain - 资本收益（连续）
12. capital-loss - 资本损失（连续）
13. hours-per-week - 每周工作小时（连续）
14. native-country - 籍贯（离散）
15. income - 年收入目标（≤50K 或 >50K）
```

### 数据格式
- **X_train** / **X_test**: 特征矩阵
  - 离散特征：使用one-hot编码
  - 连续特征：直接使用原值
  - 最终维度：106维

- **Y_train**: 标签
  - 0：年薪 ≤ 50K
  - 1：年薪 > 50K

### 数据规模
- 训练集：32,561条样本
- 测试集：16,281条样本

## 实验原理

### 1. One-Hot编码

用于处理离散特征。例如性别特征：
```
原值：male, female
编码：[1, 0] 或 [0, 1]
```

### 2. 数据标准化

对连续特征进行正态标准化：
$$x_{norm} = \frac{x - \mu}{\sigma}$$

其中 $\mu$ 是均值，$\sigma$ 是标准差

### 3. Logistic回归模型

**预测函数**：
$$\hat{y} = \sigma(w^Tx + b)$$

其中sigmoid函数为：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 4. 损失函数（交叉熵）

$$L(w) = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)] + \lambda||w||_2^2$$

其中：
- $m$ 是样本数
- $\lambda$ 是正则化系数
- $||w||_2^2$ 是L2正则化项

### 5. 梯度计算

**权重梯度**：
$$\frac{\partial L}{\partial w} = -\frac{1}{m}X^T(\hat{y} - y) + \lambda w$$

**偏置梯度**：
$$\frac{\partial L}{\partial b} = -\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

### 6. 参数更新

使用梯度下降：
$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

其中 $\alpha$ 是学习率

## 实现细节

### 类设计

```python
class LogisticRegression:
    def __init__(learning_rate, num_epoch, batch_size, lambda_reg, validation_split)
    def _load_data(X_path, Y_path) -> (X, Y)
    def _sigmoid(z) -> sigmoid(z)
    def _get_prob(X, w, b) -> P(y=1|x)
    def _cross_entropy(y_pred, y_true) -> loss
    def _compute_loss(X, Y, w, b) -> total_loss
    def _compute_accuracy(X, Y, w, b) -> accuracy
    def _gradient(X, Y, w, b) -> (w_grad, b_grad)
    def _normalize_data(X, is_training) -> X_normalized
    def train(X, Y) -> 训练过程
    def predict(X) -> (y_pred, y_prob)
    def plot_history(save_path) -> 绘制训练曲线
    def get_feature_importance(feature_names) -> 特征重要性
```

### 关键方法说明

#### 1. Sigmoid函数
```python
def _sigmoid(self, z):
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
```
- 使用np.clip避免数值溢出
- 输出范围限制在[1e-6, 1-1e-6]

#### 2. 梯度计算
```python
def _gradient(self, X, Y, w, b):
    y_pred = self._get_prob(X, w, b)
    pred_error = y_pred - Y
    w_grad = -np.mean(np.multiply(pred_error.reshape(-1, 1), X), axis=0)
    if self.lambda_reg > 0:
        w_grad += self.lambda_reg * w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad
```

#### 3. 数据标准化
```python
def _normalize_data(self, X, is_training=True):
    standardize_cols = [0, 4, 10, 11, 12]  # 连续特征列
    if is_training:
        self.X_mean = np.mean(X[:, standardize_cols], axis=0)
        self.X_std = np.std(X[:, standardize_cols], axis=0)
    X_normalized[:, col] = (X[:, col] - self.X_mean[col]) / self.X_std[col]
    return X_normalized
```

## 使用方法

### 1. 准备数据

将数据文件放在 `data/` 文件夹中：
```
Expt3/
├── logistic_regression.py
├── data/
│   ├── X_train          # 训练特征
│   ├── Y_train          # 训练标签
│   └── X_test           # 测试特征
└── output.csv           # 预测结果（运行后生成）
```

### 2. 运行程序

```bash
cd Expt3
python logistic_regression.py
```

### 3. 输出文件

- **output.csv**: 预测结果（格式：id, label）
- **training_history.png**: 训练曲线（损失和准确率）

## 参数调整

### 主要参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| learning_rate | 0.01 | 0.001~0.1 | 学习率，控制收敛速度 |
| num_epoch | 500 | 100~2000 | 训练轮数，更多轮→更充分的训练 |
| batch_size | 64 | 32~256 | 批大小，影响梯度估计 |
| lambda_reg | 0.0001 | 0~0.1 | 正则化系数，防止过拟合 |
| validation_split | 0.1 | 0.1~0.3 | 验证集比例 |

### 调参建议

**如果验证准确率不稳定**：
- 增加 validation_split 比例
- 增加 lambda_reg 正则化系数

**如果收敛速度慢**：
- 增加 learning_rate（但可能导致不稳定）
- 增加 batch_size

**如果过拟合**：
- 增加 lambda_reg
- 增加 validation_split
- 减少 num_epoch

## 模型评估

### 训练曲线分析

训练过程会生成 `training_history.png`，包含：
1. **损失曲线**：训练和验证损失的变化
2. **准确率曲线**：训练和验证准确率的变化

### 特征重要性

通过权重的绝对值判断特征对预测的贡献度。权重绝对值越大，特征越重要。

## 预期结果

### 性能指标

| 指标 | 预期值 |
|------|--------|
| 训练准确率 | 0.80~0.85 |
| 验证准确率 | 0.78~0.82 |
| 测试准确率 | 0.78~0.82 |
| 最终验证损失 | 0.35~0.50 |

### 预测结果分布

| 类别 | 比例 |
|------|------|
| 年薪≤50K | ~76% |
| 年薪>50K | ~24% |

## 常见问题

### Q1: 数据加载出错
**A**: 确保数据文件在 `data/` 文件夹中，文件名正确

### Q2: 准确率很低
**A**: 
- 检查数据标准化是否正确
- 增加训练轮数
- 调整学习率

### Q3: 模型过拟合
**A**: 
- 增加正则化系数 lambda_reg
- 减少模型复杂度（特征选择）
- 增加验证集比例

### Q4: 训练速度慢
**A**: 
- 增加批大小 batch_size
- 减少训练轮数（可能牺牲准确率）

## 扩展任务

### 1. 模型改进
- 实现不同的优化算法（SGD、Adam等）
- 添加特征交叉项
- 进行特征选择

### 2. 数据处理优化
- 处理类别不平衡问题
- 异常值检测和处理
- 更复杂的特征工程

### 3. 评估方法
- 实现k折交叉验证
- 计算AUC-ROC曲线
- 绘制混淆矩阵

### 4. 可视化
- 特征分布分析
- 决策边界可视化
- 混淆矩阵热力图

## 参考资源

### 理论基础
- Logistic回归：https://en.wikipedia.org/wiki/Logistic_regression
- 梯度下降：https://www.deeplearningbook.org/
- 交叉熵损失：https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

### 工具
- NumPy：https://numpy.org/
- Matplotlib：https://matplotlib.org/

## 文件清单

```
Expt3/
├── logistic_regression.py   (500+行) - 核心实现
├── README.md                (300+行) - 本说明
├── data/
│   ├── X_train              - 训练特征
│   ├── Y_train              - 训练标签
│   └── X_test               - 测试特征
├── output.csv               - 预测结果（运行后生成）
└── training_history.png     - 训练曲线（运行后生成）
```

## 总结

本实验通过以下步骤完成二分类预测：

1. **数据加载** - 从文件读取训练和测试数据
2. **数据预处理** - 标准化连续特征
3. **模型构建** - 实现Logistic回归和梯度下降
4. **模型训练** - 使用验证集监控性能
5. **评估可视化** - 绘制训练曲线和特征分析
6. **预测输出** - 对测试集进行预测并保存结果

**核心算法特点**：
- 完整的Logistic回归实现
- 批梯度下降优化
- 验证集交叉验证
- 交叉熵损失函数
- L2正则化防止过拟合

---

**实验完成日期**: 2024-12-12  
**编程语言**: Python 3.6+  
**主要库**: NumPy, Matplotlib  
**代码行数**: 500+行
