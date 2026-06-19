#!/usr/bin/env bash
# =============================================================================
# QuantMind API 配置脚本
# 用法：source scripts/data_pipeline/setup_api_config.sh
#
# token 不在本脚本硬编码——从仓库根 .env（已 gitignore）读取，单一可信源。
# 首次使用请按 .env.example 在 .env 里填入 TUSHARE_TOKEN / TUSHARE_TOKEN_HI。
# =============================================================================

# 定位仓库根（本脚本位于 scripts/data_pipeline/ 下）
_SELF="${BASH_SOURCE[0]:-$0}"
_ROOT="$(cd "$(dirname "$_SELF")/../.." && pwd)"

# 仅从 .env 提取 TUSHARE_* 三个键并导出（不 source 整个 .env，避免其它值含特殊字符）
if [ -f "$_ROOT/.env" ]; then
    while IFS='=' read -r _k _v; do
        case "$_k" in
            TUSHARE_TOKEN|TUSHARE_TOKEN_HI|TUSHARE_HI_URL)
                _v="${_v%\"}"; _v="${_v#\"}"        # 去掉可能的引号
                _v="${_v%\'}"; _v="${_v#\'}"
                export "$_k=$_v" ;;
        esac
    done < <(grep -E '^(TUSHARE_TOKEN|TUSHARE_TOKEN_HI|TUSHARE_HI_URL)=' "$_ROOT/.env")
else
    echo "[WARN] 未找到 $_ROOT/.env；请先按 .env.example 配置 token"
fi

# 高频代理地址（非密钥；.env 未设时给默认值）
export TUSHARE_HI_URL="${TUSHARE_HI_URL:-http://tsy.xiaodefa.cn}"

# 只确认是否就位，绝不回显 token 明文/片段
echo "[OK] API 环境变量已从 .env 载入"
echo "  TUSHARE_TOKEN      present=$([ -n "${TUSHARE_TOKEN:-}" ] && echo yes || echo NO)"
echo "  TUSHARE_TOKEN_HI   present=$([ -n "${TUSHARE_TOKEN_HI:-}" ] && echo yes || echo NO)"
echo "  TUSHARE_HI_URL     = $TUSHARE_HI_URL"
