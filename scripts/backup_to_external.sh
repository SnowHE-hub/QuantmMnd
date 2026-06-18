#!/usr/bin/env bash
# 外部备份：tar + zstd 打包关键数据/产物/文档到 .env 配置的外部盘，含校验和。
# 用法：bash scripts/backup_to_external.sh
# 依赖：zstd、tar、sha256sum。目标盘路径 = .env 的 BACKUP_TARGET（如 /mnt/d/quantmind_backup）。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 读 BACKUP_TARGET（不打印任何 secret）
BACKUP_TARGET="$(grep -E '^BACKUP_TARGET=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' || true)"
if [ -z "${BACKUP_TARGET:-}" ]; then
  echo "ERROR: .env 未设 BACKUP_TARGET（如 BACKUP_TARGET=/mnt/d/quantmind_backup）" >&2
  exit 1
fi
mkdir -p "$BACKUP_TARGET"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$BACKUP_TARGET/quantmind_backup_${STAMP}.tar.zst"

# 备份范围：数据产物 + 模型 + 关键文档（research 全链可复现所需）
PATHS=(
  data/lake data/panel data/raw
  data/bakeoff/preds data/bakeoff/alpha158_asof_v6.parquet
  data/loss_signals_v4 data/registry
  models
  docs/plans docs/methodology docs/security docs/maintenance docs/data_manifest.md
  scripts/survivorship
)
EXIST=()
for p in "${PATHS[@]}"; do [ -e "$p" ] && EXIST+=("$p"); done

echo "[backup] 目标: $OUT"
echo "[backup] 范围: ${EXIST[*]}"
echo "[backup] 预估源大小: $(du -shc "${EXIST[@]}" 2>/dev/null | tail -1 | cut -f1)"

# 打包（zstd -19 高压缩；-T0 多线程）
tar -cf - "${EXIST[@]}" | zstd -19 -T0 -o "$OUT" -f

# 校验和
sha256sum "$OUT" | tee "${OUT}.sha256"
echo "[backup] 完成: $(du -sh "$OUT" | cut -f1)  →  $OUT"
echo "[backup] 校验: ${OUT}.sha256"
echo "[backup] 还原: zstd -d $OUT -c | tar -xf - -C <目标目录>"
