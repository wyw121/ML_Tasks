# 使用国内镜像源快速安装依赖包
# 运行方式：在PowerShell中执行 .\install_packages.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始安装深度学习依赖包（使用清华镜像源）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 方法1：使用清华镜像源（推荐）
Write-Host "正在使用清华大学镜像源安装..." -ForegroundColor Green
pip install torch pandas numpy scikit-learn matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "如果上面的安装失败，可以尝试以下备选镜像源：" -ForegroundColor Yellow
Write-Host ""
Write-Host "# 阿里云镜像源：" -ForegroundColor Yellow
Write-Host "pip install torch pandas numpy scikit-learn matplotlib -i https://mirrors.aliyun.com/pypi/simple/" -ForegroundColor White
Write-Host ""
Write-Host "# 豆瓣镜像源：" -ForegroundColor Yellow
Write-Host "pip install torch pandas numpy scikit-learn matplotlib -i https://pypi.douban.com/simple/" -ForegroundColor White
Write-Host ""
Write-Host "# 腾讯云镜像源：" -ForegroundColor Yellow
Write-Host "pip install torch pandas numpy scikit-learn matplotlib -i https://mirrors.cloud.tencent.com/pypi/simple" -ForegroundColor White
