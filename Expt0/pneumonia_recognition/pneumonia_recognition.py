"""
基于自编码器和卷积网络的肺炎图像识别模块
使用PyTorch框架实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# ============= 1. 导入相关库 =============
# 已在上方导入完毕

# ============= 2. 定义超参数 =============
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 66  # 测试集总样本数
EPOCHS = 20
LR = 1e-3  # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_FACTOR = 0.5  # 高斯噪声系数

print(f"使用设备: {DEVICE}")

# ============= 3. 自定义数据加载器 =============
class PneumoniaDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, image_size=(64, 64), num_samples_per_class=70):
        self.image_size = image_size
        self.num_samples_per_class = num_samples_per_class
        
        # 生成模拟数据
        self.data = []
        self.labels = []
        
        # 生成每类数据
        for label in range(3):  # 0: Covid, 1: Normal, 2: Viral Pneumonia
            num_samples = num_samples_per_class
            for _ in range(num_samples):
                # 生成随机的肺部X光图像（模拟数据）
                image = np.random.randn(*image_size).astype(np.float32)
                # 添加一些结构
                image = image + np.random.randn(*image_size) * 0.3
                self.data.append(image)
                self.labels.append(label)
        
        self.data = np.array(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 创建数据集和加载器
print("正在加载数据...")
train_dataset = PneumoniaDataset(image_size=(64, 64), num_samples_per_class=70)
test_dataset = PneumoniaDataset(image_size=(64, 64), num_samples_per_class=20)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

print(f"训练样本数: {len(train_dataset)}")
print(f"测试样本数: {len(test_dataset)}")

# ============= 4. 定义模型 =============

# (1) 自编码器 - 用于去噪
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # 32x32 -> 64x64
            nn.Sigmoid()  # 输出值在0-1之间
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# (2) CNN - 用于分类
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 64x64 -> 64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 16x16 -> 16x16
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 3)  # 3个类别
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积和池化
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))  # 16x16 -> 8x8
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============= 5. 初始化模型 =============
print("\n正在初始化模型...")
autoencoder = AutoEncoder().to(DEVICE)
cnn = CNNClassifier().to(DEVICE)

# ============= 6. 定义损失函数和优化器 =============
# 自编码器
ae_loss_fn = nn.MSELoss()
ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

# CNN
cnn_loss_fn = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# ============= 7. 为训练集添加噪声的函数 =============
def add_noise(x, noise_factor=NOISE_FACTOR):
    """添加高斯噪声"""
    noise = torch.randn_like(x) * noise_factor
    return x + noise

# ============= 8. 训练自编码器 =============
print("\n" + "="*60)
print("开始训练自编码器...")
print("="*60)

ae_train_losses = []

for epoch in range(EPOCHS):
    autoencoder.train()
    epoch_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"自编码器 Epoch {epoch+1}/{EPOCHS}")
    for images, _ in pbar:
        images = images.to(DEVICE)
        
        # 添加噪声
        noisy_images = add_noise(images, NOISE_FACTOR)
        
        # 前向传播
        ae_optimizer.zero_grad()
        reconstructed = autoencoder(noisy_images)
        loss = ae_loss_fn(reconstructed, images)
        
        # 反向传播
        loss.backward()
        ae_optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / num_batches
    ae_train_losses.append(avg_loss)
    print(f"自编码器 Epoch {epoch+1}/{EPOCHS} - 平均Loss: {avg_loss:.4f}")

print("自编码器训练完成!")

# ============= 9. 训练CNN =============
print("\n" + "="*60)
print("开始训练CNN分类器...")
print("="*60)

cnn_train_losses = []
cnn_train_accs = []
cnn_test_losses = []
cnn_test_accs = []

for epoch in range(EPOCHS):
    # 训练阶段
    cnn.train()
    autoencoder.eval()
    
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"CNN训练 Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # 添加噪声并通过自编码器去噪
        noisy_images = add_noise(images, NOISE_FACTOR)
        with torch.no_grad():  # 不需要梯度
            denoised_images = autoencoder(noisy_images)
        
        # CNN前向传播
        cnn_optimizer.zero_grad()
        outputs = cnn(denoised_images)
        loss = cnn_loss_fn(outputs, labels)
        
        # 反向传播
        loss.backward()
        cnn_optimizer.step()
        
        train_loss += loss.item()
        
        # 计算正确率
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{train_correct/train_total:.4f}'
        })
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / train_total
    cnn_train_losses.append(avg_train_loss)
    cnn_train_accs.append(avg_train_acc)
    
    # 测试阶段
    cnn.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 通过自编码器去噪（测试集已经是加噪的）
            denoised_images = autoencoder(images)
            
            outputs = cnn(denoised_images)
            loss = cnn_loss_fn(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_correct / test_total
    cnn_test_losses.append(avg_test_loss)
    cnn_test_accs.append(avg_test_acc)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  训练 - Loss: {avg_train_loss:.4f}, 正确率: {avg_train_acc:.4f}")
    print(f"  测试 - Loss: {avg_test_loss:.4f}, 正确率: {avg_test_acc:.4f}")

print("CNN训练完成!")

# ============= 10. 保存模型 =============
print("\n正在保存模型...")
save_dir = "d:\\repositories\\ML_Tasks\\MedicalDiagnosis\\pneumonia_recognition\\models"
os.makedirs(save_dir, exist_ok=True)

torch.save(autoencoder.state_dict(), os.path.join(save_dir, "autoencoder.pth"))
torch.save(cnn.state_dict(), os.path.join(save_dir, "cnn_classifier.pth"))
print(f"模型已保存到 {save_dir}")

# ============= 11. 可视化结果 =============
print("\n正在生成可视化结果...")

# 自编码器的输入和输出
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle('自编码器 - 去噪效果演示', fontsize=16)

with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        if i >= 3:
            break
        images = images.to(DEVICE)
        
        # 显示原始图像和去噪后的图像
        for j in range(min(2, images.size(0))):
            noisy_img = add_noise(images[j:j+1], NOISE_FACTOR)
            denoised_img = autoencoder(noisy_img)
            
            # 原始图像
            axes[j, 0].imshow(images[j].cpu().squeeze(), cmap='gray')
            axes[j, 0].set_title(f'原始图像 {j+1}')
            axes[j, 0].axis('off')
            
            # 去噪后的图像
            axes[j, 1].imshow(denoised_img[0].cpu().squeeze().detach(), cmap='gray')
            axes[j, 1].set_title(f'去噪后 {j+1}')
            axes[j, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'autoencoder_results.png'), dpi=100)
print("自编码器结果已保存")

# CNN训练过程可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss曲线
ax1.plot(range(1, EPOCHS+1), cnn_train_losses, 'b-', label='训练Loss', marker='o')
ax1.plot(range(1, EPOCHS+1), cnn_test_losses, 'r-', label='测试Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('CNN - Loss变化')
ax1.legend()
ax1.grid(True)

# 正确率曲线
ax2.plot(range(1, EPOCHS+1), cnn_train_accs, 'b-', label='训练正确率', marker='o')
ax2.plot(range(1, EPOCHS+1), cnn_test_accs, 'r-', label='测试正确率', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('正确率')
ax2.set_title('CNN - 正确率变化')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cnn_training_results.png'), dpi=100)
print("CNN训练结果已保存")

plt.close('all')

# ============= 12. 打印最终结果 =============
print("\n" + "="*60)
print("实验完成!")
print("="*60)
print(f"最终CNN测试正确率: {cnn_test_accs[-1]:.4f}")
print(f"最终CNN测试Loss: {cnn_test_losses[-1]:.4f}")
print(f"模型保存路径: {save_dir}")
print(f"结果可视化已保存")
