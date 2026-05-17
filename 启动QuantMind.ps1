# QuantMind 一键启动脚本 (PowerShell)
# 用法: 右键 → 「用 PowerShell 运行」，或在 PowerShell 中执行 .\启动QuantMind.ps1

$ErrorActionPreference = "SilentlyContinue"

$API_PORT = 8000
$UI_PORT  = 8501
$WSL_ROOT = "/home/lenovo/projects/quantmind"
$PYTHON   = "/home/lenovo/miniforge3/envs/quantmind/bin/python"

Write-Host ""
Write-Host "  ╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║         QuantMind AI 量化投资系统 v2.0        ║" -ForegroundColor Cyan
Write-Host "  ║   全A股5535只 · LGBM v6 · DashScope+Ollama   ║" -ForegroundColor Cyan
Write-Host "  ╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 停止旧进程 ─────────────────────────────────────────────────────────────────
Write-Host "  [准备] 清理旧进程..." -ForegroundColor Yellow
wsl -e bash -c "pkill -f 'uvicorn app.api' 2>/dev/null; pkill -f 'streamlit run' 2>/dev/null; sleep 1; echo done" | Out-Null

# ── 判断是否有 Windows Terminal ──────────────────────────────────────────────
$hasWT = $null -ne (Get-Command "wt" -ErrorAction SilentlyContinue)

if ($hasWT) {
    Write-Host "  [1/3] 使用 Windows Terminal 启动服务..." -ForegroundColor Green

    # 启动 FastAPI（Tab 1）
    $apiCmd = "wsl -e bash -c `"cd $WSL_ROOT && echo '=== QuantMind FastAPI (port $API_PORT) ===' && $PYTHON -m uvicorn app.api.server:app --host 0.0.0.0 --port $API_PORT --reload; exec bash`""
    # 启动 Streamlit（Tab 2）
    $uiCmd  = "wsl -e bash -c `"cd $WSL_ROOT && echo '=== QuantMind Streamlit (port $UI_PORT) ===' && $PYTHON -m streamlit run app/main.py --server.port $UI_PORT --server.address 0.0.0.0 --browser.gatherUsageStats false; exec bash`""

    Start-Process "wt.exe" -ArgumentList "new-tab --title `"QM-API`" cmd /k `"$apiCmd`" ; new-tab --title `"QM-UI`" cmd /k `"$uiCmd`""

} else {
    Write-Host "  [1/3] 启动 FastAPI 后端 (port $API_PORT)..." -ForegroundColor Green
    Start-Process "cmd" -ArgumentList "/k wsl -e bash -c `"cd $WSL_ROOT && $PYTHON -m uvicorn app.api.server:app --host 0.0.0.0 --port $API_PORT --reload`"" -WindowStyle Normal

    Start-Sleep -Seconds 3

    Write-Host "  [2/3] 启动 Streamlit 前端 (port $UI_PORT)..." -ForegroundColor Green
    Start-Process "cmd" -ArgumentList "/k wsl -e bash -c `"cd $WSL_ROOT && $PYTHON -m streamlit run app/main.py --server.port $UI_PORT --server.address 0.0.0.0 --browser.gatherUsageStats false`"" -WindowStyle Normal
}

# ── 等待服务就绪 ──────────────────────────────────────────────────────────────
Write-Host "  [等待] 服务初始化中..." -ForegroundColor Yellow
$maxWait = 20
$apiReady = $false
$uiReady  = $false

for ($i = 1; $i -le $maxWait; $i++) {
    Start-Sleep -Seconds 1
    Write-Host "  ." -NoNewline -ForegroundColor Gray

    if (-not $apiReady) {
        $apiCheck = wsl -e bash -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:$API_PORT/health 2>/dev/null"
        if ($apiCheck -eq "200") { $apiReady = $true; Write-Host " API✓" -NoNewline -ForegroundColor Green }
    }
    if (-not $uiReady) {
        $uiCheck = wsl -e bash -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:$UI_PORT/_stcore/health 2>/dev/null"
        if ($uiCheck -eq "200") { $uiReady = $true; Write-Host " UI✓" -NoNewline -ForegroundColor Green }
    }
    if ($apiReady -and $uiReady) { break }
}
Write-Host ""

# ── 打开浏览器 ────────────────────────────────────────────────────────────────
Write-Host "  [3/3] 打开浏览器..." -ForegroundColor Green
Start-Process "http://localhost:$UI_PORT"

# ── 状态报告 ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║              ✅ 启动完成！                    ║" -ForegroundColor Cyan
Write-Host "  ╠═══════════════════════════════════════════════╣" -ForegroundColor Cyan

$apiStatus = if ($apiReady) { "✅ 在线" } else { "⚠️  启动中" }
$uiStatus  = if ($uiReady)  { "✅ 在线" } else { "⚠️  启动中" }

Write-Host "  ║  FastAPI 后端  http://localhost:$API_PORT   $apiStatus  ║" -ForegroundColor White
Write-Host "  ║  Streamlit前端 http://localhost:$UI_PORT  $uiStatus  ║" -ForegroundColor White
Write-Host "  ║  API 文档      http://localhost:$API_PORT/docs      ║" -ForegroundColor White
Write-Host "  ╠═══════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "  ║  主要入口：                                   ║" -ForegroundColor Yellow
Write-Host "  ║    💬 智能问答  → AI对话 + 自动执行命令       ║" -ForegroundColor Yellow
Write-Host "  ║    🖥️  系统控制台 → 一键运行系统功能          ║" -ForegroundColor Yellow
Write-Host "  ╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  提示：关闭服务请关闭对应命令行窗口" -ForegroundColor Gray
Write-Host ""
