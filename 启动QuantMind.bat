@echo off
chcp 65001 >nul
title QuantMind 启动器

echo.
echo  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███╗   ███╗██╗███╗   ██╗██████╗
echo  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝████╗ ████║██║████╗  ██║██╔══██╗
echo  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██╔████╔██║██║██╔██╗ ██║██║  ██║
echo  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
echo  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
echo   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
echo.
echo  AI增强量化投资系统  v2.0  全A股5535只  LGBM v6  DashScope + Ollama
echo  ════════════════════════════════════════════════════════════════════
echo.

REM ── 检查 WSL 是否可用 ────────────────────────────────────────────────────────
where wsl >nul 2>&1
if errorlevel 1 (
    echo  [错误] 未找到 WSL，请先安装 Ubuntu WSL
    pause
    exit /b 1
)

echo  [1/3] 启动 FastAPI 后端 (port 8000)...
start "QuantMind API" cmd /k "wsl -e bash -c ""cd /home/lenovo/projects/quantmind && echo 'FastAPI 启动中...' && /home/lenovo/miniforge3/envs/quantmind/bin/python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload && exec bash"""

echo  [等待] API 初始化 (3秒)...
timeout /t 3 /nobreak >nul

echo  [2/3] 启动 Streamlit 前端 (port 8501)...
start "QuantMind UI" cmd /k "wsl -e bash -c ""cd /home/lenovo/projects/quantmind && echo 'Streamlit 启动中...' && /home/lenovo/miniforge3/envs/quantmind/bin/streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false && exec bash"""

echo  [等待] 前端初始化 (5秒)...
timeout /t 5 /nobreak >nul

echo  [3/3] 打开浏览器...
start http://localhost:8501

echo.
echo  ════════════════════════════════════════════════════════════════════
echo.
echo   ✅ QuantMind 已启动！
echo.
echo   前端界面:  http://localhost:8501
echo   API后端:   http://localhost:8000
echo   API文档:   http://localhost:8000/docs
echo.
echo   页面导航:
echo     今日推荐   → 每日三系统选股结果
echo     漏斗选股   → 6层筛选漏斗可视化
echo     单股分析   → 个股四维评分追踪
echo     回测表现   → NAV曲线 + PnL分析
echo     模型管理   → LGBM特征重要性 + IC分析
echo     智能问答   → AI对话 + 命令执行 ← 主交互入口
echo     系统控制台 → 一键运行系统功能 ← 操作中心
echo.
echo   关闭: 分别关闭弹出的两个命令行窗口
echo  ════════════════════════════════════════════════════════════════════
echo.
pause
