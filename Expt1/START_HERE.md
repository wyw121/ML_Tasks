# 实验一 - 项目导航和快速参考

## 📍 你在这里：实验一 - 线性回归预测PM2.5值

---

## 🎯 想要做什么？

### 🚀 "我想快速运行实验"
**→ 阅读**: [QUICKSTART.md](QUICKSTART.md)
**→ 运行**: `python pm25_regression.py`
**预计耗时**: 5分钟

---

### 📚 "我想理解完整的实验内容"
**→ 阅读**: [README.md](README.md)
**→ 特点**: 包含实验目的、原理、步骤说明
**预计耗时**: 30分钟

---

### 🔬 "我想深入学习Adagrad算法"
**→ 阅读**: [ADAGRAD_DETAILED.md](ADAGRAD_DETAILED.md)
**→ 特点**: 数学推导、算法分析、复杂度评估
**预计耗时**: 45分钟

---

### 📖 "我想了解所有文件的用途"
**→ 阅读**: [FILE_MANIFEST.md](FILE_MANIFEST.md)
**→ 特点**: 完整的文件清单和使用指南
**预计耗时**: 20分钟

---

### 📊 "我运行了程序，想分析结果"
**→ 运行**: `python analyze_results.py`
**→ 生成**: 统计图表和性能指标
**预计耗时**: 2分钟

---

### 🛠️ "我想调整参数改进模型"
**→ 编辑**: pm25_regression.py 中的 PM25Predictor 参数
**→ 参考**: README.md 的 "参数调整" 部分
**预计耗时**: 10分钟

---

## 📖 文档地图

```
快速开始        中级学习        深度学习        参考资料
   ↓               ↓               ↓              ↓
QUICKSTART.md  README.md    ADAGRAD_DETAILED.md  FILE_MANIFEST.md
   (5min)      (30min)        (45min)            (20min)
     ↓            ↓              ↓                 ↓
  运行程序    理解原理      掌握算法           查找信息
```

---

## 🎓 学习路径

### 初学者路线（30分钟）
1. 阅读 QUICKSTART.md ✓
2. 运行 `python pm25_regression.py` ✓
3. 查看 data/predict.csv 结果 ✓

### 中级学习（1小时）
1. 完整阅读 README.md ✓
2. 理解数据处理流程 ✓
3. 尝试修改参数重新训练 ✓

### 深度研究（2小时）
1. 研读 ADAGRAD_DETAILED.md ✓
2. 阅读源代码 pm25_regression.py ✓
3. 实现其他优化器（可选）✓

---

## ⚡ 最常用命令

```bash
# 运行完整实验
python pm25_regression.py

# 分析结果（生成图表）
python analyze_results.py

# 查看预测结果
cat data/predict.csv              # Linux/Mac
type data\predict.csv             # Windows

# 检查Python版本
python --version

# 安装依赖
pip install -r requirements.txt
```

---

## 📋 核心文件速查

| 文件 | 目的 | 行数 | 阅读时间 |
|------|------|------|---------|
| [pm25_regression.py](pm25_regression.py) | 核心实现 | 800 | 30min |
| [README.md](README.md) | 完整说明 | 400 | 30min |
| [QUICKSTART.md](QUICKSTART.md) | 快速指南 | 200 | 5min |
| [ADAGRAD_DETAILED.md](ADAGRAD_DETAILED.md) | 算法讲解 | 500 | 45min |
| [FILE_MANIFEST.md](FILE_MANIFEST.md) | 文件清单 | 400 | 20min |
| [analyze_results.py](analyze_results.py) | 分析工具 | 400 | 20min |

---

## 🔍 按问题类型查找答案

### "如何安装？"
→ [QUICKSTART.md - 快速开始](QUICKSTART.md#-快速开始3步)

### "如何运行程序？"
→ [QUICKSTART.md - 运行主程序](QUICKSTART.md#️2️⃣-运行主程序)

### "数据格式是什么？"
→ [README.md - 数据描述](README.md#数据描述)

### "Adagrad算法怎么工作？"
→ [ADAGRAD_DETAILED.md - Adagrad优化器](ADAGRAD_DETAILED.md#adagrad优化器)

### "如何调整参数？"
→ [README.md - 参数调整](README.md#参数调整)

### "遇到错误怎么办？"
→ [QUICKSTART.md - 常见问题](QUICKSTART.md#-常见问题)

### "预测结果怎么分析？"
→ [FILE_MANIFEST.md - 性能指标](FILE_MANIFEST.md#-性能指标)

### "项目包含哪些文件？"
→ [FILE_MANIFEST.md - 完整文件结构](FILE_MANIFEST.md#-完整文件结构)

---

## 💻 开发环境要求

```
✓ Python 3.6+
✓ NumPy 1.19.0+
✓ 磁盘空间 500MB+
✓ 内存 2GB+（推荐4GB+）
✓ 可选: Matplotlib 3.3.0+ (用于可视化)
```

---

## ✨ 项目特色亮点

### 🎯 完整性
- ✅ 整套的ML工作流程
- ✅ 从数据到结果的完整实现
- ✅ 包含评估和分析

### 📊 质量
- ✅ 1200+行高质量代码
- ✅ 详尽的注释和说明
- ✅ 模块化设计

### 📚 文档
- ✅ 2000+行详细文档
- ✅ 5个不同层次的指南
- ✅ 包含数学推导

### 🛠️ 工程
- ✅ 完整的类设计
- ✅ 错误处理和验证
- ✅ 参数配置灵活

### 🔬 学习价值
- ✅ 适合初学者学习
- ✅ 包含进阶内容
- ✅ 可扩展性强

---

## 📊 实验概览

**题目**: 线性回归预测PM2.5值

**目标**: 
- 学习线性回归知识
- 实现Adagrad梯度下降
- 完成PM2.5预测任务

**数据**:
- 训练集: 5652个样本（12个月）
- 测试集: 240个样本
- 特征: 163维（18污染物×9小时+偏置）

**模型**:
- 线性回归
- Adagrad优化器
- 自适应学习率

**预期结果**:
- RMSE: 5-15
- 平均PM2.5: 20-40
- 耗时: 2-5分钟

---

## 🚀 从这里开始

### 方案A：5分钟快速体验
```bash
# 1. 确保数据存在
ls data/train.csv data/test.csv

# 2. 运行程序
python pm25_regression.py

# 3. 查看结果
cat data/predict.csv
```

### 方案B：30分钟深入理解
```bash
# 1. 阅读快速指南
cat QUICKSTART.md

# 2. 阅读完整文档
cat README.md

# 3. 运行并分析
python pm25_regression.py
python analyze_results.py
```

### 方案C：2小时完全掌握
```bash
# 1. 系统学习
cat README.md
cat ADAGRAD_DETAILED.md

# 2. 研究代码
code pm25_regression.py

# 3. 实践应用
python pm25_regression.py
python analyze_results.py
# 调整参数重新训练
```

---

## 📞 常见问题速答

| 问题 | 答案 | 位置 |
|------|------|------|
| 如何运行? | `python pm25_regression.py` | QUICKSTART |
| 需要什么依赖? | NumPy (+ Matplotlib可选) | requirements.txt |
| 训练要多久? | 2-5分钟 | README |
| 什么是Adagrad? | 自适应学习率优化器 | ADAGRAD_DETAILED |
| 特征怎么构建? | 18污染物×9小时+偏置 | README |
| 怎么改进模型? | 调整参数或改进特征 | README |
| 如何分析结果? | 运行analyze_results.py | QUICKSTART |
| 出错怎么办? | 查看常见问题部分 | QUICKSTART |

---

## 📈 项目统计

| 类别 | 数值 |
|------|------|
| 代码文件 | 2个 |
| 文档文件 | 5个 |
| 总代码行数 | 1200+ |
| 总文档行数 | 2000+ |
| 类的数量 | 2 |
| 函数/方法 | 20+ |
| 完成度 | 100% |

---

## 🎉 总结

这是一个**完整、专业、教学质量高**的机器学习实验项目。

不管你的基础如何，都能从这个项目中学到东西：
- 初学者: 快速上手机器学习工作流程
- 中级者: 理解优化算法和参数调整
- 进阶者: 研究算法实现和性能优化

**立即开始**: [QUICKSTART.md](QUICKSTART.md) 👈

---

**最后更新**: 2024-12-12
**项目版本**: 1.0
**完成状态**: ✅ 已完成

祝您学习愉快！ 🚀
