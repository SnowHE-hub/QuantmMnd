"""T-030 P0：把日频特征组合成训练用矩阵（按月分块）.

行 = 日频面板（data/gpu_v1/panel/base 的 (as_of, ticker)，即 v6_1 的 PIT 入池规则）。
股票级特征（F1 量价 45、F2 基本面 27、F3 Alpha158 非全缺失列、F4 行业 9、F5 事件 11）→ 当日截面 gauss-rank（只用当日面板行），
存 float16，缺失保留 NaN。市场级特征（当日截面为常数：v6_1 的 market_* 3 列与 north_bound_30d_net_inflow，以及 F7）
单独存一张逐日表，训练时用该折训练日的统计量标准化。
每行另存：原始 20d / 63d 前向收益（标签，训练时再做当日 gauss-rank）、PIT adv20 前 1500 标记
（与 ExecutableNavEngine 的 PriceStore.universe_topn 同一实现）、申万一级代码（PIT，缺失 = −1）、在 F6 序列数组里的股票下标。
输出 data/gpu_v1/model/v1/{<YYYY-MM>.X.npy, <YYYY-MM>.meta.parquet, features.json, market.parquet}；已存在的月份跳过，从不覆盖。
用法：PYTHONPATH=. python build_model_ready.py <data_root>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantmind.execution.nav_engine import PriceStore
from quantmind.features.weekly_panel import WHITELIST_35
from quantmind.research.gpu.transforms import gauss_rank_by_day

MARKET_IN_BASE = ["market_momentum_60d", "market_volatility_60d", "market_drawdown_60d", "north_bound_30d_net_inflow"]
CATEGORICAL = ["exposure_industry", "exposure_area"]
SECTOR = ["ind_mom_1m", "ind_mom_3m", "ind_mom_6m", "rel_mom_3m", "rel_mom_6m", "ind_dm_roe_ttm", "ind_dm_pb",
          "ind_dm_pe_ttm", "ind_dm_operating_profit_yoy"]


def read(path: Path, columns=None) -> pd.DataFrame:
    if columns is not None:
        names = pq.read_schema(path).names
        columns = [c for c in ("as_of", "ticker") if c in names] + [c for c in columns if c not in ("as_of", "ticker")]
    df = pd.read_parquet(path, columns=columns)
    if "as_of" in df.columns:
        df = df.set_index(["as_of", "ticker"])
    return df


def main() -> None:
    root = Path(sys.argv[1])
    t0 = time.time()
    D = root / "data"
    P = D / "gpu_v1/panel"
    out_dir = D / "gpu_v1/model/v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    months = sorted(p.stem for p in (P / "base").glob("*.parquet"))
    for g in ("sh", "fnd", "a158", "sector", "event"):
        missing = [m for m in months if not (P / g / f"{m}.parquet").exists()]
        assert not missing, f"group {g} missing months {missing[:3]}"
    sh_cols = json.loads((D / "loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"]
    fnd_cols = [c for c in pq.read_schema(P / "fnd" / f"{months[0]}.parquet").names if c not in ("as_of", "ticker")]
    ev_cols = [c for c in pq.read_schema(P / "event" / f"{months[0]}.parquet").names if c not in ("as_of", "ticker")]
    a_all = [c for c in pq.read_schema(P / "a158" / f"{months[0]}.parquet").names if c not in ("as_of", "ticker")]
    feat_path = out_dir / "features.json"
    if feat_path.exists():
        F = json.loads(feat_path.read_text())
    else:
        nn = pd.Series(0, index=a_all)
        for m in months:                                    # 只看特征是否全缺失，不看标签
            nn += read(P / "a158" / f"{m}.parquet", a_all).notna().sum()
        f1 = [c for c in WHITELIST_35 if c not in CATEGORICAL + MARKET_IN_BASE] + sh_cols
        F = {"groups": {"F1": f1, "F2": fnd_cols, "F3": [c for c in a_all if nn[c] > 0], "F4": SECTOR, "F5": ev_cols},
             "a158_all_nan_dropped": [c for c in a_all if nn[c] == 0]}
        F["stock_features"] = [c for g in ("F1", "F2", "F3", "F4", "F5") for c in F["groups"][g]]
        F["market_features"] = MARKET_IN_BASE + json.loads((D / "gpu_v1/seq/meta.json").read_text())["market_columns"]
        feat_path.write_text(json.dumps(F, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = F["stock_features"]
    seq_meta = json.loads((D / "gpu_v1/seq/meta.json").read_text())
    seq_pos = {t: i for i, t in enumerate(seq_meta["tickers"])}

    mkt_path = out_dir / "market.parquet"
    if not mkt_path.exists():
        parts = [read(P / "base" / f"{m}.parquet", MARKET_IN_BASE).groupby(level="as_of").first() for m in months]
        mb = pd.concat(parts)
        f7 = pd.read_parquet(P / "market/market_daily.parquet")
        mk = mb.join(f7, how="left")
        mk.index.name = "as_of"
        mk[F["market_features"]].to_parquet(mkt_path)

    todo = [m for m in months if not (out_dir / f"{m}.meta.parquet").exists()]
    print(f"[model_ready] {len(cols)} stock features, {len(todo)}/{len(months)} months to do", flush=True)
    if not todo:
        return
    pcols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=pcols)
    ps = PriceStore(prices, prices["trade_date"].min() - pd.Timedelta(days=60), prices["trade_date"].max())
    del prices
    l1 = pd.read_parquet(D / "lake/events/index_member_all_20261009.parquet", columns=["l1_name"])["l1_name"]
    sw_codes = {s: i for i, s in enumerate(sorted(l1.dropna().unique()))}
    for m in todo:
        base = read(P / "base" / f"{m}.parquet")
        idx = base.index
        f1 = F["groups"]["F1"]
        px = [c for c in f1 if c in base.columns]
        shc = [c for c in f1 if c not in base.columns]
        parts = [pd.concat([base[px], read(P / "sh" / f"{m}.parquet", shc).reindex(idx)], axis=1)[f1],
                 read(P / "fnd" / f"{m}.parquet", F["groups"]["F2"]).reindex(idx),
                 read(P / "a158" / f"{m}.parquet", F["groups"]["F3"]).reindex(idx),
                 read(P / "sector" / f"{m}.parquet", SECTOR).reindex(idx),
                 read(P / "event" / f"{m}.parquet", F["groups"]["F5"]).reindex(idx)]
        X = pd.concat(parts, axis=1)[cols].to_numpy(dtype=np.float64)
        X[~np.isfinite(X)] = np.nan
        day = idx.get_level_values("as_of")
        codes = pd.factorize(day, sort=True)[0]
        Xg = gauss_rank_by_day(X, codes).astype(np.float16)
        sw = read(P / "sector" / f"{m}.parquet", ["sw_l1"]).reindex(idx)["sw_l1"]
        tick = idx.get_level_values("ticker")
        top = np.zeros(len(idx), dtype=bool)
        for d in np.unique(day):
            u = ps.universe_topn(pd.Timestamp(d), 1500)
            sel = day == d
            top[sel] = np.isin(tick[sel], list(u))
        meta = pd.DataFrame({"as_of": day, "ticker": tick,
                             "y20": base["forward_return_20d"].to_numpy(), "y63": base["forward_return_63d"].to_numpy(),
                             "top1500": top,
                             "sw_l1": [-1 if pd.isna(s) else sw_codes[s] for s in sw],
                             "seq_idx": [seq_pos.get(t, -1) for t in tick]})
        assert (meta["seq_idx"] >= 0).all()
        np.save(out_dir / f"{m}.X.tmp.npy", Xg)
        os.replace(out_dir / f"{m}.X.tmp.npy", out_dir / f"{m}.X.npy")
        tmp = out_dir / f"{m}.meta.tmp"
        meta.to_parquet(tmp, index=False)
        os.replace(tmp, out_dir / f"{m}.meta.parquet")
        print(f"  {m} {Xg.shape} top1500/day={top.sum() / max(1, len(np.unique(day))):.0f} ({time.time() - t0:.0f}s)",
              flush=True)
    (out_dir / "sw_l1_codes.json").write_text(json.dumps(sw_codes, ensure_ascii=False), encoding="utf-8")
    print(f"[model_ready] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
