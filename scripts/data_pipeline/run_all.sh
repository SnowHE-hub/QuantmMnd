#!/usr/bin/env bash
# =============================================================================
# QuantMind Alpha Universe 数据获取主流程（一键执行）
#
# 执行顺序：
#   Step1: 构建 Alpha Universe（~1000只，CSI800+行业补位+高波动）
#   Step2: 日线价格 + 复权因子 (2019-2026) ← 高频 API，~15分钟
#   Step3: daily_basic 季末快照 (28期)       ← 高频 API，<1分钟
#   Step4: 财务报表 (fina_indicator等)        ← 自己 API，~40分钟
#
# 总计预计时间：<60 分钟（建议 nohup 后台运行）
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# 加载 API 配置
source "$SCRIPT_DIR/setup_api_config.sh"

echo "========================================================"
echo " QuantMind Alpha Universe 数据获取"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

cd "$ROOT"

# Step 1: 构建 Alpha Universe
echo ""
echo "[Step 1/4] 构建 Alpha Universe (~1000只)..."
python scripts/data_pipeline/step1_build_alpha_universe.py | tee "$LOG_DIR/step1.log"
echo "Step 1 完成 ✓"

# Step 2: 下载日线价格（高频 API）
echo ""
echo "[Step 2/4] 下载日线价格 + 复权因子（预计15分钟）..."
python scripts/data_pipeline/step2_download_prices.py | tee "$LOG_DIR/step2.log"
echo "Step 2 完成 ✓"

# Step 3: 下载 daily_basic（高频 API）
echo ""
echo "[Step 3/4] 下载 daily_basic 季末快照（预计<1分钟）..."
python scripts/data_pipeline/step3_download_daily_basic.py | tee "$LOG_DIR/step3.log"
echo "Step 3 完成 ✓"

# Step 4: 下载财务数据（自己 API，较慢）
echo ""
echo "[Step 4/4] 下载财务报表（预计40分钟，可中断后重跑）..."
python scripts/data_pipeline/step4_download_fundamentals.py | tee "$LOG_DIR/step4.log"
echo "Step 4 完成 ✓"

echo ""
echo "========================================================"
echo " 全部数据获取完成！"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 数据位置: $ROOT/data/alpha_universe/"
echo "========================================================"
