#!/usr/bin/env bash
# =============================================================================
# QuantMind API 配置脚本
# 使用方法：source scripts/data_pipeline/setup_api_config.sh
# =============================================================================

# ── 自己的 Tushare（2000 积分，官方地址，长期有效）────────────────────────────
export TUSHARE_TOKEN="64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56"

# ── 高频 Tushare（15000 积分，180次/分，到期 2026-05-19 17:05:55）────────────
export TUSHARE_TOKEN_HI="5caf9b3022e13d4e915df0af19a076130287cb7837c0b020290691c8"
export TUSHARE_HI_URL="http://tsy.xiaodefa.cn"

echo "[OK] API 环境变量已设置"
echo "  TUSHARE_TOKEN      = ${TUSHARE_TOKEN:0:8}...（2000积分，官方）"
echo "  TUSHARE_TOKEN_HI   = ${TUSHARE_TOKEN_HI:0:8}...（15000积分，高频，到期 2026-05-19）"
echo "  TUSHARE_HI_URL     = $TUSHARE_HI_URL"
