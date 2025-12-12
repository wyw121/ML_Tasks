"""
综合运行脚本 - 执行所有实验模块
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

def print_header(title):
    """打印分隔符和标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def run_command(cmd, cwd=None, description=""):
    """运行命令并处理错误"""
    print(f"\n执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {description} - 成功!")
            return True
        else:
            print(f"✗ {description} - 失败! (返回码: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ 执行出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="基于深度学习的医药诊断评估系统 - 综合运行脚本"
    )
    
    parser.add_argument(
        '--module',
        choices=['all', 'pneumonia', 'sentiment_keras', 'sentiment_pytorch'],
        default='all',
        help='要运行的模块 (默认: all)'
    )
    
    parser.add_argument(
        '--skip-install',
        action='store_true',
        help='跳过依赖检查和安装'
    )
    
    args = parser.parse_args()
    
    print_header("基于深度学习的医药诊断评估系统")
    print(f"项目路径: {PROJECT_ROOT}")
    print(f"运行模式: {args.module}")
    
    # 检查环境
    if not args.skip_install:
        print_header("检查Python环境和依赖")
        check_env(PROJECT_ROOT)
    
    # 运行选定的模块
    results = {}
    
    if args.module in ['all', 'pneumonia']:
        print_header("模块一: 肺炎图像识别 (PyTorch)")
        pneumonia_module = PROJECT_ROOT / 'pneumonia_recognition'
        pneumonia_script = pneumonia_module / 'pneumonia_recognition.py'
        
        if pneumonia_script.exists():
            results['pneumonia'] = run_command(
                [sys.executable, str(pneumonia_script)],
                cwd=str(pneumonia_module),
                description="肺炎图像识别模块"
            )
        else:
            print(f"✗ 找不到脚本: {pneumonia_script}")
            results['pneumonia'] = False
    
    if args.module in ['all', 'sentiment_keras']:
        print_header("模块二: 药物评价情感分析 (Keras)")
        sentiment_module = PROJECT_ROOT / 'drug_sentiment_analysis'
        sentiment_keras = sentiment_module / 'sentiment_analysis.py'
        
        if sentiment_keras.exists():
            results['sentiment_keras'] = run_command(
                [sys.executable, str(sentiment_keras)],
                cwd=str(sentiment_module),
                description="药物评价情感分析模块 (Keras)"
            )
        else:
            print(f"✗ 找不到脚本: {sentiment_keras}")
            results['sentiment_keras'] = False
    
    if args.module in ['all', 'sentiment_pytorch']:
        print_header("模块三: 药物评价情感分析 (PyTorch - 选做)")
        sentiment_module = PROJECT_ROOT / 'drug_sentiment_analysis'
        sentiment_pytorch = sentiment_module / 'sentiment_analysis_pytorch.py'
        
        if sentiment_pytorch.exists():
            results['sentiment_pytorch'] = run_command(
                [sys.executable, str(sentiment_pytorch)],
                cwd=str(sentiment_module),
                description="药物评价情感分析模块 (PyTorch)"
            )
        else:
            print(f"✗ 找不到脚本: {sentiment_pytorch}")
            results['sentiment_pytorch'] = False
    
    # 打印最终结果
    print_header("实验执行结果总结")
    
    for module, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{module:20s}: {status}")
    
    # 打印输出文件位置
    print_header("生成的文件位置")
    
    pneumonia_models = PROJECT_ROOT / 'pneumonia_recognition' / 'models'
    if pneumonia_models.exists():
        print(f"\n肺炎识别模块输出:")
        for file in pneumonia_models.glob('*'):
            print(f"  - {file.name}")
    
    sentiment_models = PROJECT_ROOT / 'drug_sentiment_analysis' / 'models'
    if sentiment_models.exists():
        print(f"\n药物评价模块输出:")
        for file in sentiment_models.glob('*'):
            print(f"  - {file.name}")
    
    # 总体评价
    print_header("最终评估")
    
    total_success = sum(1 for v in results.values() if v)
    total_modules = len(results)
    
    print(f"成功执行: {total_success}/{total_modules} 个模块")
    
    if total_success == total_modules:
        print("✓ 所有模块执行成功!")
        return 0
    elif total_success > 0:
        print("⚠ 部分模块执行成功，请检查失败模块")
        return 1
    else:
        print("✗ 所有模块执行失败，请检查环境配置")
        return 2

def check_env(project_root):
    """检查环境和依赖"""
    
    print("检查Python版本...")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 7):
        print(f"✓ Python {python_version}")
    else:
        print(f"✗ Python版本过低 (需要 >= 3.7, 当前: {python_version})")
        return False
    
    # 检查关键依赖
    print("\n检查关键依赖...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("\n建议安装命令:")
        print(f"  pip install -r {project_root}/requirements.txt")
        
        response = input("\n是否现在安装? (y/n): ").lower()
        if response == 'y':
            install_dependencies(project_root)
        else:
            print("跳过依赖安装")
    
    # 检查GPU
    print("\n检查GPU支持...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✓ GPU不可用 (将使用CPU)")
    except Exception as e:
        print(f"⚠ GPU检查失败: {str(e)}")
    
    return True

def install_dependencies(project_root):
    """安装依赖包"""
    requirements_file = project_root / 'requirements.txt'
    
    if requirements_file.exists():
        print(f"\n安装依赖包 ({requirements_file})...")
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("✓ 依赖安装成功!")
        else:
            print("✗ 依赖安装失败!")
    else:
        print(f"✗ 找不到 requirements.txt: {requirements_file}")

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
