#!/bin/bash
# QuantMind 全栈启动脚本（API + Streamlit）
# 用法: bash start_all.sh

PYTHON="/home/lenovo/miniforge3/envs/quantmind/bin/python"
ROOT="/home/lenovo/projects/quantmind"
cd "$ROOT"

echo "=========================================="
echo "  QuantMind 全栈启动"
echo "  API:      http://localhost:8000"
echo "  前端:     http://localhost:8501"
echo "  API文档:  http://localhost:8000/docs"
echo "=========================================="

# 停止旧进程
for PORT in 8000 8501; do
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "停止端口 $PORT 上的旧进程 (PID=$PID)..."
        kill $PID 2>/dev/null
        sleep 1
    fi
done

# 启动 FastAPI (后台)
echo ""
echo "▶ 启动 FastAPI 后端 (port 8000)..."
nohup $PYTHON -m uvicorn app.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level warning > logs/api.log 2>&1 &
API_PID=$!
echo "  FastAPI PID: $API_PID"
sleep 2

# 检查 API 是否启动
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ FastAPI 启动成功"
else
    echo "  ⚠️  FastAPI 可能未完全就绪（继续启动前端）"
fi

# 启动 Streamlit (前台)
echo ""
echo "▶ 启动 Streamlit 前端 (port 8501)..."
echo "  按 Ctrl+C 停止（API 后台继续运行）"
echo ""

$PYTHON -m streamlit run app/main.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

# Streamlit 退出后停止 API
echo ""
echo "停止 FastAPI (PID=$API_PID)..."
kill $API_PID 2>/dev/null
