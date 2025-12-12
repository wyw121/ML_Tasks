"""
模型架构可视化脚本
生成模型结构图和参数统计
"""

import torch
import torch.nn as nn
import os

# ============= 1. 定义模型（与主程序相同） =============

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 3)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============= 2. 模型统计函数 =============

def count_parameters(model):
    """计算模型参数数量"""
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'层名称':<40} {'参数类型':<15} {'参数数量':>15} {'形状':>20}")
    print("=" * 90)
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params
        
        param_type = "可训练" if param.requires_grad else "固定"
        print(f"{name:<40} {param_type:<15} {params:>15,} {str(list(param.shape)):>20}")
    
    print("=" * 90)
    print(f"{'总参数数':<40} {'':<15} {total_params:>15,}")
    print(f"{'可训练参数':<40} {'':<15} {trainable_params:>15,}")
    print(f"{'固定参数':<40} {'':<15} {total_params - trainable_params:>15,}")
    print("=" * 90)
    
    return total_params, trainable_params

def model_summary(model, input_size):
    """生成模型摘要"""
    print(f"\n{'层编号':<8} {'层类型':<20} {'输出形状':<25} {'参数数量':>15}")
    print("=" * 70)
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            if isinstance(output, (list, tuple)):
                out_shape = [list(o.size()) for o in output]
            else:
                out_shape = list(output.size())
            
            params = sum(p.numel() for p in module.parameters())
            
            print(f"{module_idx:<8} {class_name:<20} {str(out_shape):<25} {params:>15,}")
            
            summary[module_idx] = {
                'output': out_shape,
                'params': params,
                'layer': class_name
            }
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    summary = {}
    hooks = []
    
    model.apply(register_hook)
    
    # 执行一次前向传播
    try:
        if isinstance(input_size, tuple):
            model(torch.randn(1, *input_size))
        else:
            model(torch.randn(*input_size))
    except Exception as e:
        print(f"前向传播错误: {e}")
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
    print("=" * 70)
    return summary

# ============= 3. 生成报告 =============

def generate_model_report(model, model_name, input_size):
    """生成完整的模型报告"""
    print("\n" + "=" * 90)
    print(f"  {model_name} - 模型结构分析")
    print("=" * 90)
    
    # 模型概览
    print(f"\n模型总体信息:")
    print(f"  模型名称: {model_name}")
    print(f"  模型类型: {model.__class__.__name__}")
    print(f"  输入大小: {input_size}")
    
    # 参数统计
    print(f"\n参数统计:")
    total_params, trainable_params = count_parameters(model)
    print(f"  总计: {total_params:,} 参数")
    print(f"  可训练: {trainable_params:,} 参数 ({100*trainable_params/total_params:.1f}%)")
    
    # 内存占用估计
    total_memory = total_params * 4 / (1024 ** 2)  # 假设float32
    print(f"  存储大小: 约 {total_memory:.2f} MB")
    
    # 计算复杂度估计
    print(f"\n模型复杂度估计:")
    flops = estimate_flops(model, input_size)
    print(f"  FLOPs: 约 {flops:.2e}")
    
    # 层级结构
    print(f"\n层级结构分析:")
    model_summary(model, input_size)
    
    return total_params, trainable_params

def estimate_flops(model, input_size):
    """估计模型的浮点运算数"""
    # 简化估计，实际应使用专门的工具如thop
    total_flops = 0
    
    # 这只是粗略估计
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs ≈ 2 × H × W × K × K × C_in × C_out
            total_flops += 2 * 64 * 64 * module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels
        elif isinstance(module, nn.Linear):
            # FLOPs ≈ 2 × in_features × out_features
            total_flops += 2 * module.in_features * module.out_features
    
    return total_flops

# ============= 4. 生成可视化文本表示 =============

def generate_architecture_text(model):
    """生成ASCII艺术风格的架构图"""
    
    ae_architecture = """
╔═══════════════════════════════════════════════════════════════════╗
║                    自编码器 (AutoEncoder)                         ║
╚═══════════════════════════════════════════════════════════════════╝

输入层
  ├─ 尺寸: 1 × 64 × 64 (灰度图)
  └─ 格式: 张量 [batch_size, 1, 64, 64]
                    │
                    ↓
╔═══════════════════════════════════════════════════════════════════╗
║                        编码器 (Encoder)                           ║
╚═══════════════════════════════════════════════════════════════════╝
                    
  ├─ Conv2d(1→16, 3×3)
  │  └─ 输出: 16 × 64 × 64
  │  └─ 参数: 160
  │
  ├─ ReLU激活函数
  │  └─ 输出: 16 × 64 × 64
  │
  ├─ MaxPool2d(2×2)
  │  └─ 输出: 16 × 32 × 32
  │
  ├─ Conv2d(16→32, 3×3)
  │  └─ 输出: 32 × 32 × 32
  │  └─ 参数: 4,640
  │
  ├─ ReLU激活函数
  │  └─ 输出: 32 × 32 × 32
  │
  ├─ MaxPool2d(2×2)
  │  └─ 输出: 32 × 16 × 16
  │
  ├─ Conv2d(32→64, 3×3)
  │  └─ 输出: 64 × 16 × 16
  │  └─ 参数: 18,496
  │
  ├─ ReLU激活函数
  │  └─ 输出: 64 × 16 × 16
  │
  └─ MaxPool2d(2×2)
     └─ 输出: 64 × 8 × 8 ← 瓶颈层 (特征向量)
                    │
                    ↓
╔═══════════════════════════════════════════════════════════════════╗
║                        解码器 (Decoder)                           ║
╚═══════════════════════════════════════════════════════════════════╝

  ├─ ConvTranspose2d(64→32, 2×2)
  │  └─ 输出: 32 × 16 × 16
  │  └─ 参数: 8,224
  │
  ├─ ReLU激活函数
  │  └─ 输出: 32 × 16 × 16
  │
  ├─ ConvTranspose2d(32→16, 2×2)
  │  └─ 输出: 16 × 32 × 32
  │  └─ 参数: 2,064
  │
  ├─ ReLU激活函数
  │  └─ 输出: 16 × 32 × 32
  │
  ├─ ConvTranspose2d(16→1, 2×2)
  │  └─ 输出: 1 × 64 × 64
  │  └─ 参数: 65
  │
  └─ Sigmoid激活函数 (输出值限制在[0,1])
     └─ 输出: 1 × 64 × 64 ← 重构图像

输出层
  └─ 尺寸: 1 × 64 × 64
  └─ 总参数: 33,649
"""

    cnn_architecture = """
╔═══════════════════════════════════════════════════════════════════╗
║                   CNN分类器 (CNNClassifier)                       ║
╚═══════════════════════════════════════════════════════════════════╝

输入层
  ├─ 尺寸: 1 × 64 × 64 (灰度图)
  └─ 格式: 张量 [batch_size, 1, 64, 64]
                    │
                    ↓
╔═══════════════════════════════════════════════════════════════════╗
║                   特征提取 (CNN部分)                              ║
╚═══════════════════════════════════════════════════════════════════╝

  ├─ Conv2d(1→16, 3×3) + ReLU
  │  └─ 输出: 16 × 64 × 64
  │  └─ 参数: 160
  │
  ├─ MaxPool2d(2×2)
  │  └─ 输出: 16 × 32 × 32
  │
  ├─ Conv2d(16→32, 3×3) + ReLU
  │  └─ 输出: 32 × 32 × 32
  │  └─ 参数: 4,640
  │
  ├─ MaxPool2d(2×2)
  │  └─ 输出: 32 × 16 × 16
  │
  ├─ Conv2d(32→64, 3×3) + ReLU
  │  └─ 输出: 64 × 16 × 16
  │  └─ 参数: 18,496
  │
  └─ MaxPool2d(2×2)
     └─ 输出: 64 × 8 × 8 ← 特征图
                    │
                    ↓
         ╔═════════════════════╗
         ║  Flatten(展平)      ║
         ║  4096维向量        ║
         ╚═════════════════════╝
                    │
                    ↓
╔═══════════════════════════════════════════════════════════════════╗
║                   分类器 (FC部分)                                 ║
╚═══════════════════════════════════════════════════════════════════╝

  ├─ Linear(4096→128)
  │  └─ 参数: 524,544
  │
  ├─ ReLU激活函数
  │  └─ 输出: 128维向量
  │
  ├─ Dropout(0.5)
  │  └─ 训练时随机丢弃50%的神经元
  │
  └─ Linear(128→3)
     └─ 参数: 387
                    │
                    ↓
         ╔═════════════════════╗
         ║   Softmax输出       ║
         ║   3个类别概率       ║
         ║  [p_covid, p_normal,║
         ║   p_pneumonia]      ║
         ╚═════════════════════╝

输出层
  ├─ 类别0: COVID肺炎
  ├─ 类别1: 正常
  └─ 类别2: 普通肺炎

总参数: 548,227
可训练参数: 548,227 (100%)
"""
    
    return ae_architecture, cnn_architecture

# ============= 5. 主程序 =============

def main():
    print("=" * 90)
    print("  PyTorch 医学诊断模型 - 架构分析工具")
    print("=" * 90)
    
    # 创建输出目录
    output_dir = "d:\\repositories\\ML_Tasks\\MedicalDiagnosis\\models_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 自编码器分析 =====
    ae_model = AutoEncoder()
    ae_params, ae_trainable = generate_model_report(
        ae_model, 
        "AutoEncoder (自编码器)",
        (1, 1, 64, 64)
    )
    
    # ===== CNN分析 =====
    cnn_model = CNNClassifier()
    cnn_params, cnn_trainable = generate_model_report(
        cnn_model,
        "CNNClassifier (CNN分类器)",
        (1, 1, 64, 64)
    )
    
    # 生成架构可视化
    ae_arch, cnn_arch = generate_architecture_text(None)
    
    print("\n" + "=" * 90)
    print("  自编码器架构")
    print("=" * 90)
    print(ae_arch)
    
    print("\n" + "=" * 90)
    print("  CNN分类器架构")
    print("=" * 90)
    print(cnn_arch)
    
    # 保存到文件
    with open(os.path.join(output_dir, 'model_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  PyTorch 医学诊断模型 - 详细分析报告\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("═" * 90 + "\n")
        f.write("  模型对比统计\n")
        f.write("═" * 90 + "\n")
        f.write(f"{'模型':<40} {'总参数':>20} {'可训练参数':>20}\n")
        f.write("─" * 90 + "\n")
        f.write(f"{'AutoEncoder':<40} {ae_params:>20,} {ae_trainable:>20,}\n")
        f.write(f"{'CNNClassifier':<40} {cnn_params:>20,} {cnn_trainable:>20,}\n")
        f.write("─" * 90 + "\n")
        f.write(f"{'总计':<40} {ae_params + cnn_params:>20,} {ae_trainable + cnn_trainable:>20,}\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("自编码器架构:\n")
        f.write(ae_arch + "\n\n")
        
        f.write("CNN分类器架构:\n")
        f.write(cnn_arch + "\n")
    
    print("\n✓ 模型分析报告已保存到:", os.path.join(output_dir, 'model_analysis.txt'))
    
    # 最终统计
    print("\n" + "=" * 90)
    print("  整体统计")
    print("=" * 90)
    print(f"模型总参数数: {ae_params + cnn_params:,}")
    print(f"整个系统参数数: {ae_params + cnn_params:,}")
    print(f"模型总存储大小: 约 {(ae_params + cnn_params) * 4 / (1024**2):.2f} MB")
    print("=" * 90)

if __name__ == '__main__':
    main()
