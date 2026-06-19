"""scripts/survivorship/p2_build_v6.py — P2：建 v6 全市场面板（幸存者修复）.

按已批计划（survivorship_repair_plan.md §9-12）：
- v6 PIT universe：list_date + 120 交易日(seasoning) <= as_of < delist_date；ST 保留。
- 退市标签规则 A：horizon 内退市票 forward_return 算到最后可交易日（真实亏损进标签，
  不 NaN 丢弃）。**新函数实现，不动 v5 的 compute_forward_returns。**
- 35 因子复用 FeaturePipeline 在全市场截面重算；v5 字节不动（md5 守卫）。

模式：
  --industry  补拉 stock_basic L+D+P 的 industry/area 等字段（exposure_* 因子需要）
  --prices    lake daily+adj_factor → data/raw/alpha_prices_panel_v6.parquet（v5 schema）
  --panel     35 因子 + delist-aware 标签 → data/panel/alpha_panel_weekly_v6.parquet（长跑，后台）
  --sanity    v5 md5 守卫复核 + v6 自检统计（票数/退市占比/seasoning 排除/标签均值）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

V5_PRICES = REPO / "data" / "raw" / "alpha_prices_panel.parquet"
V5_PANEL = REPO / "data" / "panel" / "alpha_panel_weekly_v5.parquet"
V6_PRICES = REPO / "data" / "raw" / "alpha_prices_panel_v6.parquet"
V6_PANEL = REPO / "data" / "panel" / "alpha_panel_weekly_v6.parquet"
SB_FULL = REPO / "data" / "lake" / "stock_basic_full.parquet"
GUARD_JSON = REPO / "scripts" / "survivorship" / "_v5_md5_guard.json"
SEASONING_TD = 120
FWD_HORIZONS = (12, 21, 63)


# ============================================================================
# v5 字节守卫
# ============================================================================

def _md5(p: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def record_guard() -> dict:
    g = {str(p.name): _md5(p) for p in (V5_PRICES, V5_PANEL)}
    GUARD_JSON.write_text(json.dumps(g, indent=2))
    return g


def check_guard() -> bool:
    g = json.loads(GUARD_JSON.read_text())
    ok = True
    for name, want in g.items():
        p = V5_PRICES if name == V5_PRICES.name else V5_PANEL
        got = _md5(p)
        print(f"  v5 guard {name}: {'OK' if got == want else 'CHANGED!'}")
        ok &= (got == want)
    return ok


# ============================================================================
# --industry：补拉行业字段（exposure_industry/area + 残差族需要）
# ============================================================================

def cmd_industry() -> int:
    import scripts.backfill_tushare as bf
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:
        pass
    fields = ("ts_code,symbol,name,area,industry,fullname,enname,market,"
              "exchange,curr_type,list_status,list_date,delist_date,is_hs")
    parts = []
    for status in ("L", "D", "P"):
        df = tp._call("stock_basic", list_status=status, fields=fields)
        if df is not None and len(df):
            parts.append(df)
        print(f"  stock_basic({status}): {0 if df is None else len(df)} rows")
    sb = pd.concat(parts, ignore_index=True)
    sb["ts_code"] = sb["ts_code"].astype(str)
    # 只保留 SH+SZ（同 p1 规则）
    def is_sh_sz(s):
        code = s.split(".")[0]
        return (not s.endswith(".BJ")) and code[:1] in ("0", "3", "6") and not code.startswith(("8", "4", "92"))
    sb = sb[sb["ts_code"].map(is_sh_sz)].reset_index(drop=True)
    sb["ticker"] = sb["ts_code"]
    sb.to_parquet(SB_FULL)
    n_ind = sb["industry"].notna().sum()
    by = sb.groupby("list_status")["industry"].apply(lambda s: f"{s.notna().mean():.1%}")
    print(f"[industry] saved {SB_FULL}: {len(sb)} rows, industry non-null={n_ind} ({n_ind/len(sb):.1%})")
    print(f"  industry 覆盖 by status:\n{by.to_string()}")
    return 0


# ============================================================================
# --prices：v6 价格面板（v5 schema + adj_close）
# ============================================================================

def cmd_prices() -> int:
    t0 = time.time()
    daily = pd.read_parquet(REPO / "data/lake/daily.parquet")
    adj = pd.read_parquet(REPO / "data/lake/adj_factor.parquet",
                          columns=["ts_code", "trade_date", "adj_factor"])
    print(f"[prices] daily={len(daily):,} adj={len(adj):,} ({time.time()-t0:.0f}s)")
    df = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    assert df["adj_factor"].notna().all(), "adj_factor 缺口（P1 已验 0，不应发生）"
    df["adj_close"] = df["close"] * df["adj_factor"]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "change", "pct_chg", "vol", "amount", "adj_factor", "adj_close"]
    df = df[cols].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    df.to_parquet(V6_PRICES)
    print(f"[prices] saved {V6_PRICES}: {len(df):,} rows, {df.ts_code.nunique()} tickers, "
          f"{pd.to_datetime(df.trade_date).min().date()}→{pd.to_datetime(df.trade_date).max().date()} "
          f"({time.time()-t0:.0f}s)")
    return 0


# ============================================================================
# delist-aware forward returns（要求 A 的灵魂；新函数，不动 v5）
# ============================================================================

def compute_forward_returns_delist_aware(
    prices_pivot: pd.DataFrame,
    ffill_pivot: pd.DataFrame,
    as_of: pd.Timestamp,
    horizons: tuple[int, ...],
    last_trade_day: pd.Series,
    has_delist: pd.Series,
) -> pd.DataFrame:
    """delist-aware 前向收益。

    规则：
    - 正常票：close(as_of 后第 h 个交易日)/close(as_of) − 1（与 v5 相同）。
    - **已退市票** 且其最后可交易日 <= 目标日：exit = 最后可交易日 close
      （= ffill 到目标日），真实亏损进标签 —— 不 NaN 丢弃。
    - 停牌但未退市：目标日无价 → NaN（与 v5 行为一致）。

    Args:
        prices_pivot:  adj_close wide（index=trade_date, columns=ticker），不 ffill
        ffill_pivot:   prices_pivot.ffill()（外部预计算一次，省每调用重算）
        last_trade_day: ticker -> 该票最后有价交易日
        has_delist:     ticker -> 是否最终退市（list_status==D 且 delist_date 非空）
    """
    target = pd.Timestamp(as_of)
    on_or_before = prices_pivot.index[prices_pivot.index <= target]
    if len(on_or_before) == 0:
        return pd.DataFrame()
    base_idx = on_or_before[-1]
    base = prices_pivot.loc[base_idx].replace(0, np.nan)
    future_idx = prices_pivot.index[prices_pivot.index > base_idx]

    out = pd.DataFrame(index=prices_pivot.columns)
    for h in horizons:
        col = f"forward_return_{h}d"
        if len(future_idx) <= h - 1:
            out[col] = np.nan
            continue
        tgt_day = future_idx[h - 1]
        tgt = prices_pivot.loc[tgt_day]
        # 退市票：最后可交易日 <= 目标日 → 用最后可交易价（真实亏损）
        delist_exit = has_delist & (last_trade_day <= tgt_day)
        exit_px = tgt.where(~(tgt.isna() & delist_exit), ffill_pivot.loc[tgt_day])
        out[col] = exit_px / base - 1.0
    return out


# ============================================================================
# --panel：v6 面板主构建（35 因子 + delist-aware 标签）
# ============================================================================

def _seasoned_stock_basic(cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (sb_true, sb_seasoned)。

    sb_true：真实 list_date/delist_date（喂 snapshot，list_age_years 等因子用真值）。
    sb_seasoned：list_date 替换为 eligible_date = list_date 后第 SEASONING_TD 个交易日
    （喂 pit_universe，使纳入规则 = list+120td <= as_of < delist，PIT 安全）。
    """
    sb = pd.read_parquet(SB_FULL).copy()
    sb["ticker"] = sb["ts_code"].astype(str)
    for c in ("list_date", "delist_date"):
        if not pd.api.types.is_datetime64_any_dtype(sb[c]):
            sb[c] = pd.to_datetime(sb[c], format="%Y%m%d", errors="coerce")
    cal_v = cal.values
    cal0 = pd.Timestamp(cal_v[0])
    def eligible(ld):
        if pd.isna(ld):
            return pd.NaT
        if ld < cal0:
            # 上市早于日历起点（2019 前）：120 交易日 ≈ 175 自然日近似；
            # 绝大多数远早上市票 eligible << 2019，从日历起点即合格（正确）
            return pd.Timestamp(ld) + pd.Timedelta(days=175)
        i = np.searchsorted(cal_v, np.datetime64(ld))
        j = i + SEASONING_TD
        return pd.Timestamp(cal_v[j]) if j < len(cal_v) else pd.NaT  # 历史不足→数据期内永不合格
    sb_seasoned = sb.copy()
    sb_seasoned["list_date"] = sb["list_date"].map(eligible)
    return sb, sb_seasoned


def cmd_panel(n_workers: int = 4, limit_asof: int | None = None) -> int:
    """主构建：仿 weekly_panel.build_weekly_panel，换 universe(seasoned+delist) + 标签(delist-aware)。"""
    from quantmind.features.weekly_panel import (
        WHITELIST_35, LABEL_COLS, build_asof_grid, _PRICE_RENAME, _FACTOR_PRICE_COLS,
        assemble_snapshot, pit_universe, _MIN_TRADING_DAYS, _PER_TICKER_MIN_BARS,
    )
    from quantmind.features.pipeline import FeaturePipeline
    from quantmind.data.lake import load_trading_calendar

    t0 = time.time()
    guard0 = record_guard()
    print(f"[guard] v5 md5 recorded: {guard0}")

    cal = load_trading_calendar()          # 与 v5 同日历 → as_of 网格逐点一致（可比）
    as_of_list = build_asof_grid(cal)
    if limit_asof:
        as_of_list = as_of_list[:limit_asof]

    print("[panel] loading v6 prices ...")
    prices = pd.read_parquet(V6_PRICES)
    prices = prices.rename(columns=_PRICE_RENAME)   # ts_code→ticker, vol→volume, pct_chg→pct_change
    prices["ticker"] = prices["ticker"].astype(str)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values("trade_date").set_index("trade_date", drop=False)
    priced_tickers = set(prices["ticker"].unique())

    sb_true, sb_seasoned = _seasoned_stock_basic(cal)
    print(f"[panel] universe base: {len(sb_true)} tickers "
          f"(D={int((sb_true.list_status=='D').sum())}); as_of={len(as_of_list)}; "
          f"workers={n_workers} ({time.time()-t0:.0f}s)")

    # ---- 标签准备（delist-aware）----
    sub = prices[["trade_date", "ticker", "adj_close"]].dropna(subset=["adj_close"])
    adj_pivot = (sub.drop_duplicates(["trade_date", "ticker"], keep="last")
                 .pivot(index="trade_date", columns="ticker", values="adj_close").sort_index())
    ffill_pivot = adj_pivot.ffill()
    last_trade = adj_pivot.apply(lambda s: s.last_valid_index())
    has_delist = pd.Series(False, index=adj_pivot.columns)
    dmap = sb_true.set_index("ticker")
    common = has_delist.index.intersection(dmap.index)
    has_delist.loc[common] = ((dmap.loc[common, "list_status"] == "D")
                              & dmap.loc[common, "delist_date"].notna()).values
    print(f"[panel] label prep done: pivot {adj_pivot.shape}, delist tickers in prices="
          f"{int(has_delist.sum())} ({time.time()-t0:.0f}s)")

    # ---- 逐 as_of：universe(seasoned) → 35 因子 → 标签 ----
    pipe = FeaturePipeline(groups=["technical", "sentiment", "expansion"], do_standardize=False)
    rows = []
    n_seasoning_excluded = 0
    for k, as_of in enumerate(as_of_list):
        as_of = pd.Timestamp(as_of)
        uni_seasoned = pit_universe(as_of, sb_seasoned, priced_tickers)
        # seasoning 排除统计：用真实 list_date 的 universe 大小 − seasoned 的
        uni_true = pit_universe(as_of, sb_true, priced_tickers)
        n_seasoning_excluded += len(uni_true) - len(uni_seasoned)
        if uni_seasoned.empty:
            continue
        snap = assemble_snapshot(as_of, prices, sb_true, uni_seasoned, cal)
        raw = pipe.compute_raw_from_snapshot(snap, as_of.date())
        missing = [c for c in WHITELIST_35 if c not in raw.columns]
        if missing:
            raise RuntimeError(f"factor cols missing: {missing}")
        out = raw[WHITELIST_35].copy()
        # NaN guards（同 v5：全市场窗口不足 + 逐票 bar 不足）
        px = snap["prices"]; px_days = pd.to_datetime(px["trade_date"]).dt.normalize()
        avail = int(px_days.nunique())
        for col, req in _MIN_TRADING_DAYS.items():
            if avail < req and col in out.columns:
                out[col] = np.nan
        bars = (px.assign(_d=px_days).dropna(subset=["close"])
                .drop_duplicates(["ticker", "_d"]).groupby("ticker")["_d"].size())
        bars = bars.reindex(out.index).fillna(0).astype(int)
        for col, req in _PER_TICKER_MIN_BARS.items():
            if col in out.columns:
                out.loc[bars < req, col] = np.nan
        # 标签（delist-aware）
        fr = compute_forward_returns_delist_aware(
            adj_pivot, ffill_pivot, as_of, FWD_HORIZONS, last_trade, has_delist)
        out = out.join(fr.reindex(out.index), how="left")
        out = out.reset_index().rename(columns={"index": "ticker"})
        out["ticker"] = out["ticker"].astype(str)
        out["as_of"] = as_of
        rows.append(out)
        if k % 25 == 0:
            print(f"  as_of {k+1}/{len(as_of_list)} {as_of.date()} uni={len(uni_seasoned)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    panel = pd.concat(rows, ignore_index=True).set_index(["as_of", "ticker"]).sort_index()
    panel = panel[WHITELIST_35 + list(LABEL_COLS)]
    if V6_PANEL.resolve() in (V5_PANEL.resolve(),):
        raise RuntimeError("refuse to overwrite v5")
    panel.to_parquet(V6_PANEL)
    print(f"[panel] saved {V6_PANEL}: {panel.shape} "
          f"({panel.index.get_level_values('as_of').nunique()} as_of, "
          f"{panel.index.get_level_values('ticker').nunique()} tickers) "
          f"seasoning_excluded_rows={n_seasoning_excluded} ({time.time()-t0:.0f}s)")

    # run manifest
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(REPO / "data/panel", kind="build", data_version="v6",
                       feature_set="whitelist35", label="forward_return_{12,21,63}d",
                       horizon="weekly",
                       artifacts=[V6_PANEL, V6_PRICES],
                       extra={
                           "universe": "full-market SH+SZ, seasoning120td, delist-aware",
                           "n_rows": int(len(panel)),
                           "seasoning_excluded_rows": int(n_seasoning_excluded)})
    ok = check_guard()
    print(f"[guard] v5 unchanged: {ok}")
    return 0 if ok else 1


# ============================================================================
# --sh：16 survivors 在 v6 全市场截面重算 + merge（残差族缺行业显式剔除+计数）
# ============================================================================

def _residual_returns_v6(returns, logcap, industry_map):
    """同 short_horizon_factors.residual_returns，但缺行业票【显式剔除并计数】（用户要求）。

    缺行业票不进截面回归，其 r̃ 列=NaN（残差族因子自然 NaN）；返回 (rtilde, n_dropped)。
    """
    tickers = list(returns.columns)
    ind = industry_map.reindex(tickers)
    missing = ind.isna()
    n_dropped = int(missing.sum())
    ind_dummies = pd.get_dummies(ind.astype("object"), prefix="ind")
    ind_mat_full = ind_dummies.values.astype(float)
    resid = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    rv = returns.values
    lv = logcap.reindex(index=returns.index, columns=returns.columns).values
    cap = np.exp(lv)
    miss_v = missing.values
    for ti in range(rv.shape[0]):
        r = rv[ti]; lc = lv[ti]; cp = cap[ti]
        mask = np.isfinite(r) & np.isfinite(lc) & np.isfinite(cp) & (cp > 0) & (~miss_v)
        if mask.sum() < 50:
            continue
        idx = np.where(mask)[0]
        rr = r[idx]
        w = cp[idx] / cp[idx].sum()
        mkt = float((w * rr).sum())
        ex = rr - mkt
        z = (lc[idx] - lc[idx].mean()) / (lc[idx].std() + 1e-12)
        D = ind_mat_full[idx]
        keep = D.sum(axis=0) > 0
        X = np.column_stack([np.ones(len(idx)), z, D[:, keep]])
        beta, *_ = np.linalg.lstsq(X, ex, rcond=None)
        resid.values[ti, idx] = ex - X @ beta
    return resid, n_dropped


def cmd_sh() -> int:
    from quantmind.features.short_horizon_factors import (
        build_wide_frames, build_short_horizon_factors, sample_at_asof)
    from quantmind.features.weekly_panel import merge_increment
    t0 = time.time()
    surv = json.loads((REPO / "data/loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"]
    pl = pd.read_parquet(V6_PRICES)
    F = build_wide_frames(pl)
    tickers = list(F["adj_close"].columns)
    print(f"[sh] wide frames {F['adj_close'].shape} ({time.time()-t0:.0f}s)")
    # logcap（daily_basic circ_mv，全市场含退市）
    db = pd.read_parquet(REPO / "data/lake/daily_basic.parquet",
                         columns=["ts_code", "trade_date", "circ_mv"])
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    db = db[(db.circ_mv > 0) & db.ts_code.isin(set(tickers))]
    mc = (db.drop_duplicates(["trade_date", "ts_code"], keep="last")
          .pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index())
    lc = np.log(mc).reindex(index=F["adj_close"].index.normalize(), columns=tickers)
    lc.index = F["adj_close"].index
    # 行业（stock_basic_full；退市票大多缺 → 显式剔除计数）
    sb = pd.read_parquet(SB_FULL)
    ind_map = sb.set_index(sb["ts_code"].astype(str))["industry"]
    rt, n_dropped = _residual_returns_v6(F["returns"], lc, ind_map)
    print(f"[sh] residual returns done; 缺行业被剔除票数={n_dropped}/{len(tickers)} "
          f"({time.time()-t0:.0f}s)")
    facs = build_short_horizon_factors(F, rt)
    facs = {k: v for k, v in facs.items() if k in surv}
    print(f"[sh] {len(facs)} survivor factors computed")
    panel = pd.read_parquet(V6_PANEL)
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    shp = sample_at_asof(facs, as_of_list, tickers=tickers).reindex(panel.index)
    out = REPO / "data/panel/short_horizon_factors_v6.parquet"
    shp.to_parquet(out)
    merged = merge_increment(panel, shp)
    merged.to_parquet(V6_PANEL)
    print(f"[sh] saved {out} {shp.shape}; merged v6 panel -> {merged.shape} "
          f"({time.time()-t0:.0f}s)")
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(REPO / "data/panel", kind="build", data_version="v6",
                       feature_set="short_horizon_16_survivors", label="-", horizon="weekly",
                       artifacts=[V6_PANEL, out],
                       extra={"missing_industry_dropped": n_dropped, "n_factors": len(facs)})
    ok = check_guard()
    print(f"[guard] v5 unchanged: {ok}")
    return 0 if ok else 1


# ============================================================================
# --sanity：自检统计
# ============================================================================

def cmd_sanity() -> int:
    ok = check_guard()
    v6 = pd.read_parquet(V6_PANEL)
    v5 = pd.read_parquet(V5_PANEL)
    sb = pd.read_parquet(SB_FULL)
    sb["ticker"] = sb["ts_code"].astype(str)
    dset = set(sb.loc[sb.list_status == "D", "ticker"])
    t6 = v6.index.get_level_values("ticker")
    print(f"v6: rows={len(v6):,} as_of={v6.index.get_level_values('as_of').nunique()} "
          f"tickers={t6.nunique()} (v5: rows={len(v5):,} tickers="
          f"{v5.index.get_level_values('ticker').nunique()})")
    in_d = t6.isin(dset)
    print(f"v6 含退市票行: {in_d.sum():,} ({in_d.mean():.2%}); 退市票数={t6[in_d].nunique()}")
    for col in ("forward_return_12d", "forward_return_63d"):
        m6, m5 = v6[col].mean(), v5[col].mean()
        md = v6.loc[in_d, col].mean()
        print(f"{col}: v6_mean={m6:+.4f} (v5={m5:+.4f}, Δ={m6-m5:+.4f}); "
              f"退市票组均值={md:+.4f} (应显著更低)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--sh", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-asof", type=int, default=None)
    a = ap.parse_args()
    if a.industry:
        return cmd_industry()
    if a.prices:
        return cmd_prices()
    if a.panel:
        return cmd_panel(a.workers, a.limit_asof)
    if a.sh:
        return cmd_sh()
    if a.sanity:
        return cmd_sanity()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
