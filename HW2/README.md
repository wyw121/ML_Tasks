# HW2 小型视频二分类与文字描述（CNN + Transformer）

## 任务概述
- **视频二分类（必做）**：ResNet18 作为帧级特征提取器 + Transformer Encoder 时序建模 + 线性分类头，标签 0/1。
- **视频文字描述（可选加分）**：使用 BLIP（`Salesforce/blip-image-captioning-base`）对采样帧生成描述并合并为视频级文本。

数据：`Data/labels_train.csv`, `Data/labels_test.csv`, `Data/videos_train/`, `Data/videos_test/`。

## 代码结构
```
HW2/
├── train.py           # 训练/验证/推理主程序，支持BLIP描述
├── modeling.py        # ResNet18 + Transformer 分类模型
├── video_dataset.py   # 视频帧采样与数据集封装（OpenCV均匀采样）
├── requirements.txt   # 依赖清单
└── Data/              # 提供的训练/测试数据与标签
```

## 快速开始
1) 安装依赖（推荐新环境）：
```bash
pip install -r HW2/requirements.txt
```

2) 训练并推理（默认训练 5 epoch，batch=4，帧数=8）：
```bash
python HW2/train.py --data-dir HW2/Data --epochs 5 --batch-size 4 --num-frames 8
```
- 训练集自动按 `--val-ratio` 划分验证集（默认 0.1）。
- 最优模型保存在 `outputs/best_seq.pt`，测试集预测写入 `outputs/pred_test.csv`。

3) 仅推理（使用已有 checkpoint）：
```bash
python HW2/train.py --data-dir HW2/Data --eval --checkpoint outputs/best_seq.pt --output-csv outputs/pred_test.csv
```

4) 生成视频描述（可选加分，需 GPU/充足内存）：
```bash
python HW2/train.py --data-dir HW2/Data --generate-captions --caption-output outputs/captions.jsonl --caption-samples 4
```
输出为 JSONL，每行：`{"video_path": str, "captions": [str, ...]}`。

## 模型与实现要点
- **帧采样**：OpenCV 均匀采样 `num_frames`（默认 8），BGR→RGB，ImageNet 归一化。
- **骨干**：`torchvision.models.resnet18`（ImageNet 预训练），可 `--freeze-cnn` 冻结特征。
- **时序建模**：TransformerEncoder（`nhead=4`, `num_layers=2`, `dim_feedforward=1024`），正弦位置编码。
- **分类头**：时序均值池化 + 线性输出 2 类，`CrossEntropyLoss`，`AdamW` 优化。
- **推理**：`outputs/pred_test.csv`，列：`video_path,label`。
- **可选描述**：HuggingFace BLIP image-to-text pipeline，对每个视频采样若干帧生成描述列表。

## 重要参数
- `--epochs` (默认5) 训练轮数
- `--batch-size` (默认4) batch 大小
- `--num-frames` (默认8) 每个视频采样帧数
- `--image-size` (默认224) ResNet 输入尺寸
- `--val-ratio` (默认0.1) 训练/验证划分比例
- `--lr` (默认1e-4), `--weight-decay` (默认1e-4)

## 依赖
见 `requirements.txt`：
- torch, torchvision
- opencv-python
- tqdm
- transformers（仅在生成描述时需要）
- pandas, numpy

## 输出
- 训练最佳权重：`outputs/best_seq.pt`
- 测试预测：`outputs/pred_test.csv`
- 可选描述：`outputs/captions.jsonl`

## 说明
- 默认训练轮数较小以便复现与快速测试，可根据资源提高 `--epochs`、`--num_frames` 或解冻 CNN 以提升效果。
- 首次运行会下载 BLIP 权重（可选步骤），需网络与较大内存/GPU。仅分类任务无需下载 BLIP。
