#!/usr/bin/env bash
# =============================================================================
# run_full_market_pipeline.sh
# 全市场漏斗 + 价格补充 + 6 Agent 完整流程
# 预计耗时：3~8 小时（主要是 Tushare 价格补充，约 300 只 × 5min/只）
# 使用：bash scripts/run_full_market_pipeline.sh
# =============================================================================
set -euo pipefail

PROJECT=/home/lenovo/projects/quantmind
PYTHON=/home/lenovo/miniforge3/envs/quantmind/bin/python
DATE=${1:-2024-12-31}
LOG_DIR=$PROJECT/reports/full_market_run
OUT_DIR=$PROJECT/data/recommendations/$DATE

mkdir -p "$LOG_DIR" "$OUT_DIR" "$PROJECT/reports/selection" \
         "$PROJECT/reports/investment_pipeline/$DATE" \
         "$PROJECT/data/tmp"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/run.log"; }

# ──────────────────────────────────────────────────────────────
# STEP 1: 全市场漏斗（5500只 → 15只）
# ──────────────────────────────────────────────────────────────
log "=== STEP 1: 全市场漏斗选股 ==="
FUNNEL_JSON=$OUT_DIR/funnel_full_a_all.json

export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

$PYTHON -u "$PROJECT/scripts/run_funnel_selection.py" \
  --date "$DATE" \
  --universe full_a \
  --top-n 15 \
  --provider none \
  --output "$FUNNEL_JSON" \
  2>&1 | tee "$LOG_DIR/step1_funnel.log"

# 验收：候选 ≥ 5
$PYTHON -c "
import json, sys
d = json.load(open('$FUNNEL_JSON'))
cands = d.get('candidates', [])
s = d.get('funnel_stats', {})
scores = [round(c.get('score', 0), 3) for c in cands]
unique_scores = len(set(scores))
print(f'  Layer1_in: {s.get(\"layer1_in\",\"?\")}')
print(f'  最终候选: {len(cands)} 只')
print(f'  分数唯一: {unique_scores}/15')
if len(cands) < 5:
    print('FAIL: 候选数 < 5'); sys.exit(1)
print('PASS')
" || { log "STEP 1 FAILED"; exit 1; }

log "STEP 1 完成"

# ──────────────────────────────────────────────────────────────
# STEP 2: 检测并补充非 CSI300 价格数据
# ──────────────────────────────────────────────────────────────
log "=== STEP 2: 检测价格数据缺口 ==="

$PYTHON -c "
import json, pandas as pd, sys

d = json.load(open('$FUNNEL_JSON'))
cands = [c['ticker'] for c in d.get('candidates', [])]

import os
price_file = '$PROJECT/data/raw/daily_prices_panel.parquet'
if os.path.exists(price_file):
    df_p = pd.read_parquet(price_file)
    existing = set(df_p['ts_code'].astype(str).unique())
else:
    existing = set()

missing = [t for t in cands if t not in existing]
print(f'候选 {len(cands)} 只 | 价格面板已有 {len(existing)} 只')
print(f'需补充 {len(missing)} 只: {missing}')

with open('$PROJECT/data/tmp/missing_tickers.txt', 'w') as f:
    f.write('\n'.join(missing))
print('saved to data/tmp/missing_tickers.txt')
"

MISSING_COUNT=$(wc -l < "$PROJECT/data/tmp/missing_tickers.txt" 2>/dev/null || echo 0)
log "缺失价格数据：$MISSING_COUNT 只"

if [ "$MISSING_COUNT" -gt 0 ]; then
  log "=== STEP 2a: 下载缺失股票价格数据（约 $MISSING_COUNT 只 × ~2min）==="
  
  # 检查是否有 Tushare Token
  if [ -z "${TUSHARE_TOKEN:-}" ]; then
    log "警告: 未设置 TUSHARE_TOKEN，跳过价格补充（MomentumAgent 将 fallback 到规则）"
  else
    $PYTHON -u "$PROJECT/scripts/build_daily_price_panel.py" \
      --ticker-file "$PROJECT/data/tmp/missing_tickers.txt" \
      --start-date 2019-01-01 \
      --end-date 2025-12-31 \
      --output "$PROJECT/data/raw/extra_prices_panel.parquet" \
      2>&1 | tee "$LOG_DIR/step2_price_download.log" || {
        log "价格下载部分失败（继续）"
    }

    # 合并
    if [ -f "$PROJECT/data/raw/extra_prices_panel.parquet" ]; then
      log "合并价格面板..."
      $PYTHON -c "
import pandas as pd
main = pd.read_parquet('$PROJECT/data/raw/daily_prices_panel.parquet')
extra = pd.read_parquet('$PROJECT/data/raw/extra_prices_panel.parquet')
before = main['ts_code'].nunique()
combined = pd.concat([main, extra], ignore_index=True).drop_duplicates(['ts_code','trade_date'])
combined.to_parquet('$PROJECT/data/raw/daily_prices_panel.parquet', index=False)
after = combined['ts_code'].nunique()
print(f'价格面板: {before} → {after} 只 (+{after-before})')
"
    fi
  fi
fi

log "STEP 2 完成"

# ──────────────────────────────────────────────────────────────
# STEP 3: 重训 RiskAgent GARCH（可选，若 arch 已安装）
# ──────────────────────────────────────────────────────────────
log "=== STEP 3: 可选 RiskAgent 重训 ==="
if $PYTHON -c "import arch" 2>/dev/null; then
  log "arch 已安装，重训 RiskAgent garch_v2..."
  $PYTHON "$PROJECT/scripts/train_risk_agent_v2.py" \
    2>&1 | tee "$LOG_DIR/step3_risk_train.log" || log "RiskAgent 重训失败（跳过）"
else
  log "arch 未安装，跳过 GARCH 重训（使用 EWMA fallback）"
  log "  安装：conda run -n quantmind pip install arch"
fi

# ──────────────────────────────────────────────────────────────
# STEP 4: 6 Agent 投资分析（15 只候选）
# ──────────────────────────────────────────────────────────────
log "=== STEP 4: 6 Agent 投资分析 ==="

$PYTHON -u "$PROJECT/scripts/run_investment_pipeline.py" \
  --date "$DATE" \
  --tickers-from-file "$FUNNEL_JSON" \
  --output-dir "$PROJECT/reports/investment_pipeline" \
  --top-n 15 \
  2>&1 | tee "$LOG_DIR/step4_pipeline.log"

# 验收
$PYTHON -c "
import json, sys
strats = json.load(open('$PROJECT/reports/investment_pipeline/$DATE/strategies.json'))
print(f'  策略数: {len(strats)}/15')
from collections import Counter
ratings = Counter(s.get('rating') for s in strats)
print(f'  评级: {dict(ratings)}')
sigs = [s.get(\"composite_signal\", 0) for s in strats]
print(f'  信号范围: {min(sigs):.3f} ~ {max(sigs):.3f}')
if len(strats) < 5:
    print('FAIL'); sys.exit(1)
print('PASS')
" || log "STEP 4 验收警告"

log "STEP 4 完成"

# ──────────────────────────────────────────────────────────────
# STEP 5: 验收汇总
# ──────────────────────────────────────────────────────────────
log "=== 最终验收 ==="
$PYTHON -c "
import json, pandas as pd

print('=== 全市场漏斗测试验收 ===')
d = json.load(open('$FUNNEL_JSON'))
s = d.get('funnel_stats', {})
cands = d.get('candidates', [])
scores = [round(c.get('score', 0), 4) for c in cands]

print(f'Layer1_in  : {s.get(\"layer1_in\", \"?\"):>6}  (期望 ≥ 3000 为真全市场)')
print(f'Layer6_out : {len(cands):>6}  (期望 ≤ 15)')
print(f'分数唯一性  : {len(set(scores))}/15  (期望 = 15)')
print()
print('候选名单:')
for c in cands:
    t = c.get('ticker','?')
    n = c.get('name','')
    sc = c.get('score', 0)
    print(f'  {t:<15} {n:<8} score={sc:.4f}')

print()
import os
strat_file = '$PROJECT/reports/investment_pipeline/$DATE/strategies.json'
if os.path.exists(strat_file):
    strats = json.load(open(strat_file))
    print(f'6 Agent 策略: {len(strats)} 份 ✓')
else:
    print('6 Agent 策略: 未生成 ✗')

df_p = pd.read_parquet('$PROJECT/data/raw/daily_prices_panel.parquet')
print(f'价格面板   : {df_p[\"ts_code\"].nunique()} 只')
"

log "=== 全部完成。日志：$LOG_DIR/run.log ==="
