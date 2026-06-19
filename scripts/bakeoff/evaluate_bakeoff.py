#!/usr/bin/env python
"""scripts/bakeoff/evaluate_bakeoff.py — 统一评估 harness（P2，quantmind env，绝不进 qlib）.

输入：任意模型导出的逐 (as_of, ticker, score) 预测（score 已含方向，H-A 在训练侧定）。
输出：raw IC / 中性化 IC / 中性化 ICIR / 含成本净超额 / 最大回撤 / 换手 / pos_frac，
       与 v2 apples-to-apples（中性化逻辑与 scripts/_diag_wf_v2.py 完全一致）。

G2 自检：用 data/loss_signals_v4/wf_v2_oos_predictions.parquet 复现 raw +0.0139 / neut +0.0017。
G2 不过 → 不接深度模型预测。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from quantmind.features.weekly_panel import OUT_PATH
from quantmind.backtest.wf_metrics import net_excess_annualized, per_rebalance_win_rate
from quantmind.backtest.wf_costs import SlippageTiers, amihud_to_quantile, stamp_duty_rate

PERIOD_CFG = {
    "12d": {"label": "forward_return_12d", "holding_td": 12},
    "21d": {"label": "forward_return_21d", "holding_td": 21},   # F-04：robustness 口径（非产品线）
    "63d": {"label": "forward_return_63d", "holding_td": 63},
}


def _spear(a, b):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(d) < 15:
        return np.nan
    return float(stats.spearmanr(d["a"], d["b"])[0])


def _load_context():
    """panel 提供 label/industry/amihud；circ_mv 从 daily_basic 取 log_mktcap（PIT as_of）。"""
    panel = pd.read_parquet(OUT_PATH)
    db = pd.read_parquet(REPO / "data" / "lake" / "daily_basic.parquet",
                         columns=["ticker", "trade_date", "circ_mv"])
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    db = db[db["circ_mv"] > 0]
    db["log_mktcap"] = np.log(db["circ_mv"].astype(float))
    lmc = {pd.Timestamp(d): s.set_index("ticker")["log_mktcap"]
           for d, s in db.groupby("trade_date")}
    return panel, lmc


def _neutralize_date(g):
    """与 _diag_wf_v2.neutralize_date 同口径：score 对 [1, z(logcap), 行业哑变量] 残差。"""
    d = g.dropna(subset=["score", "realized", "industry", "log_mktcap"]).copy()
    if d["industry"].nunique() < 2 or len(d) < 30:
        return None
    D = pd.get_dummies(d["industry"], drop_first=True).astype(float)
    z = (d["log_mktcap"] - d["log_mktcap"].mean()) / (d["log_mktcap"].std() + 1e-12)
    X = np.column_stack([np.ones(len(d)), z.values, D.values])
    y = d["score"].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    raw = _spear(pd.Series(y, index=d.index), d["realized"])
    neu = _spear(pd.Series(resid, index=d.index), d["realized"])
    return pd.Series({"raw_ic": raw, "neu_ic": neu, "n": len(d)})


def score_predictions(preds: pd.DataFrame, period: str = "12d",
                      *, panel=None, lmc=None, cost_bp_oneway: float | None = None) -> dict:
    """preds: columns [as_of, ticker, score]（score 已含方向）。返回指标 dict。

    若 preds 已含 realized/industry/log_mktcap（如 v2 文件），直接用；否则从 panel/lmc 补。
    """
    cfg = PERIOD_CFG[period]
    if panel is None or lmc is None:
        panel, lmc = _load_context()
    df = preds.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    if "realized" not in df.columns:
        lab = panel[cfg["label"]].rename("realized")
        df = df.merge(lab, left_on=["as_of", "ticker"], right_index=True, how="left")
    if "industry" not in df.columns:
        ind = panel["exposure_industry"].rename("industry")
        df = df.merge(ind, left_on=["as_of", "ticker"], right_index=True, how="left")
    if "log_mktcap" not in df.columns:
        rows = []
        for a, sub in df.groupby("as_of"):
            s = lmc.get(pd.Timestamp(a).normalize())
            if s is None:
                continue
            rows.append(sub.assign(log_mktcap=sub["ticker"].map(s)))
        df = pd.concat(rows, ignore_index=True) if rows else df.assign(log_mktcap=np.nan)
    if "amihud" not in df.columns and "amihud_illiquidity" in panel.columns:
        am = panel["amihud_illiquidity"].rename("amihud")
        df = df.merge(am, left_on=["as_of", "ticker"], right_index=True, how="left")

    # ---- IC（raw / 中性化）----
    per = df.groupby("as_of").apply(_neutralize_date, include_groups=False).dropna(how="all")
    per = per.reset_index()
    for c in ("raw_ic", "neu_ic"):
        per[c] = per[c].astype(float)
    raw_ic = float(per["raw_ic"].mean()); neu_ic = float(per["neu_ic"].mean())
    neu_icir = float(per["neu_ic"].mean() / per["neu_ic"].std()) if per["neu_ic"].std() else np.nan
    raw_icir = float(per["raw_ic"].mean() / per["raw_ic"].std()) if per["raw_ic"].std() else np.nan

    # ---- 含成本净超额（holding-period-aware 换手；验收意见②，batch B 前置 must-fix）----
    # 把周频 as_of 子采样到【非重叠持有期】再平衡：step = ceil(H/5) 个 as_of（≥H 交易日，互不重叠）。
    # 每次再平衡：top-quintile 等权；换手 = 与上次持仓的单边 membership churn；成本只计真实换手部分。
    tiers = SlippageTiers()
    holding = cfg["holding_td"]
    step = max(1, int(np.ceil(holding / 5.0)))            # as_of 间隔（周频=5td）
    all_dates = sorted(df["as_of"].unique())
    rebal_dates = all_dates[::step]                        # 非重叠持有期
    port, bench, turn = {}, {}, {}
    prev_top = None
    for a in rebal_dates:
        g = df[(df["as_of"] == a)].dropna(subset=["score", "realized"])
        if len(g) < 20:
            continue
        q80 = g["score"].quantile(0.80)
        top = g[g["score"] >= q80]
        top_set = set(top["ticker"]); n_t = len(top_set)
        port_gross = float(top["realized"].mean())
        bench[pd.Timestamp(a)] = float(g["realized"].mean())
        # 单边换手（等权）：0.5·Σ|w_t − w_{t-1}|；首期=1（建仓）
        if prev_top is None or n_t == 0:
            oneway = 1.0
        else:
            names = top_set | prev_top; n_p = len(prev_top)
            oneway = 0.5 * sum(abs((1.0 / n_t if nm in top_set else 0.0) -
                                   (1.0 / n_p if nm in prev_top else 0.0)) for nm in names)
        prev_top = top_set
        # 滑点：成交名单（top）按 amihud 分位三档均值
        if cost_bp_oneway is not None:
            slip = cost_bp_oneway
        elif "amihud" in g.columns:
            slip = amihud_to_quantile(top["amihud"]).map(tiers.bp_for_quantile).mean()
        else:
            slip = 15.0
        commission, transfer = 3.0, 0.2
        stamp = stamp_duty_rate(a) * 1e4                  # bp（仅卖出）
        # 真实换手成本：买卖两边各计 oneway 部分；印花仅卖出侧
        cost_bp = oneway * (2 * (slip + commission + transfer) + stamp)
        port[pd.Timestamp(a)] = port_gross - cost_bp / 1e4
        turn[pd.Timestamp(a)] = oneway
    port_s = pd.Series(port).sort_index(); bench_s = pd.Series(bench).sort_index()
    turn_s = pd.Series(turn).sort_index()
    ppy = 250.0 / holding                                  # 非重叠持有期 → 每年 250/H 次再平衡
    net_excess = net_excess_annualized(port_s, bench_s, periods_per_year=ppy) if len(port_s) else np.nan
    win = per_rebalance_win_rate(port_s, bench_s) if len(port_s) else np.nan
    excess = (port_s - bench_s)
    cum = (1 + excess).cumprod(); dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = float(abs(dd.min())) if len(dd) else np.nan
    avg_oneway_turn = float(turn_s.mean()) if len(turn_s) else np.nan
    ann_turn = avg_oneway_turn * ppy if avg_oneway_turn == avg_oneway_turn else np.nan

    return {
        "period": period, "n_oos_dates": int(len(per)), "n_rebalances": int(len(port_s)),
        "raw_ic": round(raw_ic, 4), "raw_icir": round(raw_icir, 3),
        "neut_ic": round(neu_ic, 4), "neut_icir": round(neu_icir, 3),
        "neut_ic_posfrac": round(float((per["neu_ic"] > 0).mean()), 3),
        "net_excess_cost": round(float(net_excess), 4) if net_excess == net_excess else None,
        "avg_oneway_turnover": round(avg_oneway_turn, 3) if avg_oneway_turn == avg_oneway_turn else None,
        "annualized_turnover": round(ann_turn, 2) if ann_turn == ann_turn else None,
        "win_rate": round(float(win), 3) if win == win else None,
        "max_drawdown": round(max_dd, 4) if max_dd == max_dd else None,
    }


def g2_selfcheck() -> dict:
    """用 v2 预测复现 raw +0.0139 / neut +0.0017（中性化口径一致性硬门）。"""
    p = REPO / "data" / "loss_signals_v4" / "wf_v2_oos_predictions.parquet"
    v2 = pd.read_parquet(p)  # columns: ticker, pred, realized, industry, log_mktcap, as_of, fold, regime
    v2 = v2.rename(columns={"pred": "score"})
    m = score_predictions(v2[["as_of", "ticker", "score", "realized", "industry", "log_mktcap"]],
                          period="12d")
    ok = (abs(m["raw_ic"] - 0.0139) < 0.0015) and (abs(m["neut_ic"] - 0.0017) < 0.0015)
    return {"g2_pass": bool(ok), "reproduced": m,
            "target": {"raw_ic": 0.0139, "neut_ic": 0.0017}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", action="store_true", help="跑 G2 自检（复现 v2）")
    ap.add_argument("--preds", type=str, default=None, help="预测 parquet 路径")
    ap.add_argument("--period", type=str, default="12d")
    args = ap.parse_args()
    if args.g2:
        r = g2_selfcheck()
        print(json.dumps(r, ensure_ascii=False, indent=2))
        raise SystemExit(0 if r["g2_pass"] else 1)
    if args.preds:
        preds = pd.read_parquet(args.preds)
        print(json.dumps(score_predictions(preds, args.period), ensure_ascii=False, indent=2))
