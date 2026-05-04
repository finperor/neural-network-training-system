@echo off
echo ========================================
echo 化工过程神经网络训练系统
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
python -c "import torch; import pandas; import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在检查依赖...
    pip install torch pandas numpy PyQt6 matplotlib scikit-learn
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络连接
        echo 或者手动运行: pip install torch pandas numpy PyQt6 matplotlib scikit-learn
        pause
        exit /b 1
    )
)

echo [提示] 所有依赖已就绪
echo.
echo [启动] 正在启动程序...
python main.py

if errorlevel 1 (
    echo [错误] 程序启动失败
    pause
)