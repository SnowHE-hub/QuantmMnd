#!/bin/bash
# QuantMind FastAPI 后端启动脚本
# 用法: bash start_api.sh

PYTHON="/home/lenovo/miniforge3/envs/quantmind/bin/python"
ROOT="/home/lenovo/projects/quantmind"

cd "$ROOT"

echo "=========================================="
echo " QuantMind FastAPI Backend"
echo " http://localhost:8000"
echo " API文档: http://localhost:8000/docs"
echo "=========================================="
echo ""

# 检查端口是否已占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 8000 已被占用，尝试停止旧进程..."
    kill $(lsof -Pi :8000 -sTCP:LISTEN -t) 2>/dev/null
    sleep 1
fi

echo "启动 FastAPI 服务..."
$PYTHON -m uvicorn app.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
