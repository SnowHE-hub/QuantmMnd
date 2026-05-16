#!/usr/bin/env bash
# QuantMind 每日更新 cron 配置脚本
# 用法：bash scripts/setup_cron.sh [--dry-run]
#
# 功能：
#   每个交易日 16:30（UTC+8）自动运行 daily_update.py
#   输出日志到 logs/daily_YYYY-MM-DD.log

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/lenovo/miniforge3/envs/quantmind/bin/python"
CRON_CMD="30 16 * * 1-5 cd ${PROJ_DIR} && ${PYTHON} scripts/daily_update.py --universe alpha --no-llm --auto-regime --position-sizing hrp --agent-top 10 --agent-provider none >> logs/daily_\$(date +\\%F).log 2>&1"

echo "QuantMind Cron 配置"
echo "工作目录: ${PROJ_DIR}"
echo "Python: ${PYTHON}"
echo ""
echo "将添加以下 cron 任务（每周一至周五 16:30）："
echo "  ${CRON_CMD}"
echo ""

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY-RUN] 未实际写入，用 --apply 应用"
    exit 0
fi

if [[ "${1:-}" != "--apply" ]]; then
    echo "用法: $0 [--dry-run|--apply]"
    echo "  --dry-run  仅显示命令，不写入"
    echo "  --apply    写入 crontab"
    exit 1
fi

# 保留现有 crontab，追加新任务（去重）
TMPFILE=$(mktemp)
crontab -l 2>/dev/null | grep -v "daily_update.py" > "${TMPFILE}" || true
echo "${CRON_CMD}" >> "${TMPFILE}"
crontab "${TMPFILE}"
rm -f "${TMPFILE}"

echo "✅ Cron 任务已写入。查看：crontab -l"
