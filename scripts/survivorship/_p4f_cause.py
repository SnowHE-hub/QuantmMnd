"""继续找因：v6 neut IC 高于 v5 0.034 的来源.

Diag A：用当前 harness 重评 v5 的 ridge_full 预测 → 确认 0.034 基线是否仍成立（防止基线 stale）。
Diag B：v6 预测限制到 v5 的 1373 票，UNKNOWN 桶中性化 → 隔离"评估票池组成" vs "模型/特征"。
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import pandas as pd
import scripts.bakeoff.evaluate_bakeoff as EB

# ---- Diag A：v5 基线重评（原始 neutralize，v5 panel）----
v5panel, v5lmc = EB._load_context()
v5p = REPO / "data/bakeoff/preds/ridge_full_12d_quarterly.parquet"
print("v5 preds exists:", v5p.exists())
v5preds = pd.read_parquet(v5p)
mA = EB.score_predictions(v5preds, "12d", panel=v5panel, lmc=v5lmc)
print("[Diag A] v5 RE-EVAL 当前harness: neut_ic={} net={} icir={} n_tickers={}".format(
    mA["neut_ic"], mA["net_excess_cost"], mA["neut_icir"], v5preds.ticker.nunique()))

v5_tickers = set(v5panel.index.get_level_values("ticker").astype(str).unique())
print("[v5 universe] size =", len(v5_tickers))

# ---- Diag B：v6 预测 ∩ v5 的 1373 票（UNKNOWN 桶，v6 ctx）。import 会给 EB 打补丁，放最后 ----
from scripts.survivorship.p4d_eval_v6 import _v6_context, _merge_ctx, _per_asof_neut
p6, l6 = _v6_context()
df6 = _merge_ctx(pd.read_parquet(REPO / "data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet"), p6, l6)
df6["ticker"] = df6["ticker"].astype(str)
ov = df6[df6["ticker"].isin(v5_tickers)]
per = _per_asof_neut(ov)
print("[Diag B] v6预测∩v5的1373票 neut_ic={} 覆盖票={} n_asof={}".format(
    round(float(per["neu_ic"].mean()), 4), ov.ticker.nunique(), len(per)))
