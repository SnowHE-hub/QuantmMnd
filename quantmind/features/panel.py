"""quantmind.features.panel — 训练集 panel 构建器.

把多时点 × 多 ticker 的因子矩阵拼成一个 ``MultiIndex(as_of, ticker)`` 的
"长格式 panel"，并附上 **forward return 标签**（用于监督学习）。

数据流
======

::

    rebalance_dates = [..., 2023-12-29, 2024-03-29, 2024-06-28, ...]
            │
            ▼
    for as_of in dates:
        build_snapshot(as_of)  ── PIT 严格、缓存复用
        compute_factors()       ── 41 因子
            │
            ▼
    concat → panel[(as_of, ticker)] = factors
            │
            ▼
    fetch future prices    [pivot]
            │
            ▼
    add labels: forward_return_{h}d for h in (21, 63, ...)
            │
            ▼
    data/features/panel_{universe}_{start}_{end}.parquet


关键
====

- PIT：因子层 ``_assert_pit`` 已保证；label 是「未来收益」，不应出现在 features 里
- Forward returns 用 ``adjust='qfq'`` 复权后的 close 计算
- panel 的 columns = 41 因子 + forward_return_21d / forward_return_63d
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from quantmind.core.config import get_settings
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data.snapshot import build_snapshot, list_snapshots
from quantmind.data.sse_calendar import monthly_last_trade_days
from quantmind.data.tushare_provider import TushareProvider
from quantmind.features.pipeline import FeaturePipeline

log = get_logger(__name__)


# ============================================================================
# Rebalance 日期工具
# ============================================================================


def quarter_end_dates(start: date, end: date) -> list[date]:
    """生成 [start, end] 区间内每个季度末的日历日期（3-31 / 6-30 / 9-30 / 12-31）."""
    out = []
    rng = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="QE")
    for ts in rng:
        out.append(ts.date())
    return out


def month_end_dates(start: date, end: date) -> list[date]:
    """月线：每月最后一个 **SSE 交易日**（Tushare trade_cal），适合 A 股月频调仓。"""
    return monthly_last_trade_days(start, end)


def adjust_to_trading_day(d: date, prices_pivot: pd.DataFrame | None = None) -> date:
    """把日期向前回退到最近交易日（用 prices_pivot 的 index 判定）."""
    if prices_pivot is None or prices_pivot.empty:
        return d
    idx = prices_pivot.index
    target = pd.Timestamp(d)
    valid = idx[idx <= target]
    if len(valid) == 0:
        return d
    return valid[-1].date()


# ============================================================================
# 单点：构建 snapshot + 计算特征
# ============================================================================


def build_features_for_date(
    as_of: date,
    *,
    universe_name: str = "csi300",
    max_tickers: int | None = None,
    do_standardize: bool = True,
    skip_if_snapshot_missing: bool = False,
    rebuild_snapshot: bool = False,
) -> pd.DataFrame:
    """为单个 as_of 构建 snapshot 并计算因子，返回 DataFrame(index=ticker)."""
    # 是否需要建 snapshot —— 用 meta.json 判定「构建完成」
    snapshot_dir = Path(get_settings().data.dir) / "snapshots" / as_of.isoformat()
    snapshot_complete = (snapshot_dir / "meta.json").exists()
    if rebuild_snapshot or not snapshot_complete:
        if skip_if_snapshot_missing and not snapshot_complete:
            raise FileNotFoundError(f"snapshot for {as_of} not complete")
        build_snapshot(
            as_of,
            universe_name=universe_name,
            max_tickers=max_tickers,
            overwrite=rebuild_snapshot,
        )

    pipe = FeaturePipeline(do_standardize=do_standardize)
    df = pipe.run_single(as_of) if do_standardize else pipe.compute_raw(as_of)
    return df


# ============================================================================
# Forward returns
# ============================================================================


def fetch_forward_price_pivot(
    tickers: list[str],
    start: date,
    end: date,
    *,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """获取 ``tickers`` 在 [start, end] 的 qfq 复权 close，输出宽表 (date × ticker).

    通过 cache 后的全历史接口加速：每只票只在第一次时拉一次远程。
    """
    tushare = TushareProvider()
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for t in tqdm(sorted(set(tickers)), desc="fetch_prices_for_labels", ncols=80):
        try:
            px = tushare.get_price(t, start=start, end=end, adjust=adjust, with_basic=False)
            if px is None or px.empty:
                failed.append(t)
                continue
            frames.append(px[["trade_date", "ticker", "close"]])
        except Exception as e:  # noqa: BLE001
            log.warning(f"price fetch failed for {t}: {e}")
            failed.append(t)
    if failed:
        log.warning(f"{len(failed)} tickers had no prices: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    long["trade_date"] = pd.to_datetime(long["trade_date"])
    return long.pivot(index="trade_date", columns="ticker", values="close").sort_index()


def compute_forward_returns(
    prices_pivot: pd.DataFrame,
    as_of: date,
    horizons_days: tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """对每只 ticker 在 ``as_of`` 之后 H 个交易日计算累计收益率.

    Returns:
        DataFrame(index=ticker, columns=[forward_return_{H}d for each H])
    """
    if prices_pivot is None or prices_pivot.empty:
        return pd.DataFrame()
    target = pd.Timestamp(as_of)
    # 当日及之后的索引
    after = prices_pivot.index[prices_pivot.index >= target]
    if len(after) == 0:
        return pd.DataFrame()

    # 用 as_of 当天 close 作为基准；如果当天非交易日，回退到前一个交易日
    on_or_before = prices_pivot.index[prices_pivot.index <= target]
    base_idx = on_or_before[-1] if len(on_or_before) > 0 else after[0]
    base = prices_pivot.loc[base_idx]

    out = pd.DataFrame(index=prices_pivot.columns)
    for h in horizons_days:
        col = f"forward_return_{h}d"
        # 从 base 之后第 h 个交易日（含）
        future_idx = prices_pivot.index[prices_pivot.index > base_idx]
        if len(future_idx) <= h - 1:
            out[col] = np.nan
            continue
        target_close = prices_pivot.loc[future_idx[h - 1]]
        out[col] = target_close / base.replace(0, np.nan) - 1.0
    return out


# ============================================================================
# 主入口：build_panel
# ============================================================================


def build_panel(
    rebalance_dates: list[date],
    *,
    universe_name: str = "csi300",
    max_tickers: int | None = None,
    do_standardize: bool = True,
    forward_horizons_days: tuple[int, ...] = (21, 63),
    rebuild_snapshot: bool = False,
    save: bool = True,
    panel_name: str | None = None,
) -> pd.DataFrame:
    """端到端 panel 构建。

    Args:
        rebalance_dates: 时点列表（如季度末），每个时点一行 snapshot
        universe_name: 'csi300' / 'csi500' 等
        max_tickers: 限制每个时点的 ticker 数（小样本验证用）
        do_standardize: 是否横截面标准化（True 适合训练，False 适合分析原始值）
        forward_horizons_days: 前向收益窗口（交易日数）
        rebuild_snapshot: 是否强制重建 snapshot（默认复用已存在的）
        save: 是否保存到 ``data/features/panel_*.parquet``
        panel_name: 自定义文件名（默认 ``{universe}_{first}_{last}``）

    Returns:
        DataFrame，MultiIndex=[as_of, ticker]，columns=[41 因子 + forward_return_*]
    """
    rebalance_dates = sorted(set(rebalance_dates))
    if not rebalance_dates:
        raise ValueError("rebalance_dates is empty")

    log.info(
        f"build_panel: {len(rebalance_dates)} dates, universe={universe_name}, "
        f"max_tickers={max_tickers}, horizons={forward_horizons_days}"
    )

    # ----- 1. 逐时点构建 features -----
    feature_dfs: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for as_of in tqdm(rebalance_dates, desc="features_per_date", ncols=80):
        with operation_logger("panel.date", as_of=str(as_of)):
            try:
                df = build_features_for_date(
                    as_of,
                    universe_name=universe_name,
                    max_tickers=max_tickers,
                    do_standardize=do_standardize,
                    rebuild_snapshot=rebuild_snapshot,
                )
                df = df.reset_index()
                df["as_of"] = pd.Timestamp(as_of)
                feature_dfs.append(df)
                log.info(f"  {as_of}: {df.shape[0]} tickers × {df.shape[1] - 2} factors")
            except Exception as e:  # noqa: BLE001
                log.error(f"  {as_of} FAILED: {e}")
                failures.append((str(as_of), str(e)))

    if not feature_dfs:
        raise RuntimeError("no features built — see log for failures")

    panel = pd.concat(feature_dfs, ignore_index=True)
    log.info(f"raw panel: {panel.shape}, columns: {len(panel.columns)}")

    # ----- 2. 全 ticker 集合（要算 forward return 的票）-----
    all_tickers = sorted(set(panel["ticker"]))
    log.info(f"unique tickers: {len(all_tickers)}")

    # ----- 3. 拉「未来 close」用于计算 forward return -----
    earliest = min(rebalance_dates)
    latest = max(rebalance_dates)
    fwd_end = latest + timedelta(days=int(max(forward_horizons_days) * 1.6) + 10)
    log.info(f"fetching prices for forward returns: {earliest} → {fwd_end}")
    pivot = fetch_forward_price_pivot(all_tickers, start=earliest, end=fwd_end)

    if pivot.empty:
        log.warning("forward price pivot empty; labels will be NaN")
    else:
        log.info(f"forward price pivot: {pivot.shape}")

    # ----- 4. 计算 forward return 并 merge 进 panel -----
    fr_rows = []
    for as_of in rebalance_dates:
        fr = compute_forward_returns(pivot, as_of, forward_horizons_days)
        if fr.empty:
            continue
        fr = fr.reset_index().rename(columns={"index": "ticker"})
        fr["as_of"] = pd.Timestamp(as_of)
        fr_rows.append(fr)
    if fr_rows:
        fr_panel = pd.concat(fr_rows, ignore_index=True)
        panel = panel.merge(fr_panel, on=["as_of", "ticker"], how="left")
    else:
        for h in forward_horizons_days:
            panel[f"forward_return_{h}d"] = np.nan

    # ----- 5. set MultiIndex + sort -----
    panel = panel.set_index(["as_of", "ticker"]).sort_index()

    # ----- 6. 保存 -----
    if save:
        settings = get_settings()
        out_dir = Path(settings.data.dir) / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        if panel_name is None:
            panel_name = f"panel_{universe_name}_{earliest.isoformat()}_{latest.isoformat()}"
        path = out_dir / f"{panel_name}.parquet"
        panel.to_parquet(path)
        meta_path = out_dir / f"{panel_name}.meta.json"
        meta = {
            "universe": universe_name,
            "first_date": earliest.isoformat(),
            "last_date": latest.isoformat(),
            "n_dates": len(rebalance_dates),
            "n_tickers": len(all_tickers),
            "n_rows": int(len(panel)),
            "n_factors": int(panel.shape[1] - len(forward_horizons_days)),
            "factor_cols": [c for c in panel.columns if not c.startswith("forward_return_")],
            "label_cols": [c for c in panel.columns if c.startswith("forward_return_")],
            "standardized": do_standardize,
            "max_tickers_per_date": max_tickers,
            "horizons_days": list(forward_horizons_days),
            "snapshots_present": [d.isoformat() for d in list_snapshots()],
            "failures": failures,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        log.info(f"panel saved to {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return panel


__all__ = [
    "adjust_to_trading_day",
    "build_features_for_date",
    "build_panel",
    "compute_forward_returns",
    "fetch_forward_price_pivot",
    "month_end_dates",
    "quarter_end_dates",
]
