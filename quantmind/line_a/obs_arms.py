"""quantmind.line_a.obs_arms — 前瞻"观察臂"（T-055：X3 的逆波动加权 W，未通过非劣，只跟踪不入正式荐股）.

W：ENS20 原样选前 300（与参考臂同一研究口径），权重 ∝ 1 / σ̂，截断在等权的 0.5–2 倍再归一；
σ̂ = HAR 预测的未来 20 日年化波动，系数在冻结日之前一次拟合（configs/line_a_v2/x3_har.json），特征只用决策日及以前的价格。
用 T-052 模拟器的研究口径运行（与冻结引擎逐日对照通过）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import SlippageTiers
from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig

HAR_CFG = Path(__file__).resolve().parents[2] / "configs/line_a_v2/x3_har.json"
A = np.sqrt(252)


def har_sigma(prices: pd.DataFrame, dates: list[pd.Timestamp], cfg: Path = HAR_CFG) -> pd.Series:
    c = json.loads(Path(cfg).read_text(encoding="utf-8"))
    b = np.array(c["beta"])
    p = prices[["ts_code", "trade_date", "close", "adj_factor"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p[p["trade_date"] >= min(dates) - pd.Timedelta(days=150)].drop_duplicates(["trade_date", "ts_code"], keep="last")
    adjc = p.pivot(index="trade_date", columns="ts_code", values="close") * \
        p.pivot(index="trade_date", columns="ts_code", values="adj_factor")
    lr = np.log(adjc).diff()
    lr = lr.where(lr.abs() < 0.5)
    f = [np.log(lr.rolling(w, min_periods=m).std() * A) for w, m in ((5, 4), (20, 15), (60, 40))]
    pred = b[0] + b[1] * f[0] + b[2] * f[1] + b[3] * f[2]
    s = np.exp(pred.reindex(pd.DatetimeIndex(dates))).stack()
    s.index.names = ["as_of", "ticker"]
    return s


def inv_vol_weights(sig: pd.Series):
    def fn(d, target):
        s = sig.reindex(pd.MultiIndex.from_arrays([[d] * len(target), target])).to_numpy(dtype=float)
        s = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 1.0)
        w = (1 / s) / np.sum(1 / s)
        eq = 1 / len(target)
        w = np.clip(w, 0.5 * eq, 2 * eq)
        return dict(zip(target, w / w.sum(), strict=True))
    return fn


def run_obs(scores_ens20: pd.DataFrame, prices: pd.DataFrame, amihud: pd.DataFrame | None, out_dir: Path) -> dict:
    """scores_ens20：ens_native_20d 的前瞻网格分数 (as_of, ticker, score)。写 obs_inv_vol_20d__nav.parquet 等，返回摘要。"""
    dates = sorted(pd.to_datetime(scores_ens20["as_of"]).unique())
    start, end = pd.Timestamp(dates[0]), pd.to_datetime(prices["trade_date"]).max()
    sig = har_sigma(prices, dates)
    mk = Market(prices, start, end)
    base = {"aum": 1.0, "top_n": 300, "rebalance_step": 4, "universe_size": 1500, "buffer_mult": None, "drift_band": 0.0,
            "commission_rate": 3e-4, "min_commission": 0.0, "transfer_rate": 2e-5, "tiers": SlippageTiers(),
            "participation": None, "mode": "adj", "dividend_tax": False}
    out = {}
    for name, wf in (("obs_eq_20d", None), ("obs_inv_vol_20d", inv_vol_weights(sig))):   # 同一引擎的等权对照 + W
        r = AccountSimulator(scores_ens20, mk, SimConfig(**base, weight_fn=wf), start=start, end=end, amihud=amihud).run()
        r["nav"].to_parquet(out_dir / f"{name}__nav.parquet")
        if len(r["targets"]):
            t = r["targets"].copy()
            t["targets"] = t["targets"].map(list)
            t.to_parquet(out_dir / f"{name}__targets.parquet")
        out[name] = {k: v for k, v in r["summary"].items() if k != "costs"} | {"status": "observation（T-055，未通过非劣）"}
    return out
