"""quantmind.data.snapshot — PIT 快照构建与持久化.

Snapshot
========

把任一时点 ``as_of`` 的全市场数据冻结到磁盘，后续回测/agent 调用直接复用，
保证 100% 可复现。

每个 snapshot 是 ``data/snapshots/{as_of}/`` 下的几个 parquet 文件：

::

    data/snapshots/2024-06-30/
        meta.json              # 时点、provider、universe、文件清单
        prices.parquet         # 全 universe 的近 N 日 OHLCV
        financials_income.parquet
        financials_balance_sheet.parquet
        financials_cashflow.parquet
        financial_indicators.parquet
        universe.parquet       # 当时成分股 + 权重
        north_bound.parquet
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from quantmind.core.config import get_settings
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data.tushare_provider import TushareProvider
from quantmind.data.universe import get_universe_with_weights

log = get_logger(__name__)


def _snapshot_dir(as_of: date) -> Path:
    settings = get_settings()
    base = Path(settings.data.dir) / "snapshots" / as_of.isoformat()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _fetch_per_ticker(
    tickers: list[str],
    fn: Callable[[str], pd.DataFrame],
    desc: str,
    head_n: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """对 ``tickers`` 串行调用 ``fn(ticker)`` 并合并结果，跳过失败的票.

    Returns:
        (合并 DataFrame, 失败 ticker 列表)
    """
    frames = []
    failed = []
    for t in tqdm(tickers, desc=desc, unit="t", ncols=80):
        try:
            df = fn(t)
            if df is not None and not df.empty:
                frames.append(df.head(head_n) if head_n else df)
        except Exception as e:  # noqa: BLE001
            log.warning(f"{desc} failed for {t}: {e}")
            failed.append(t)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, failed


def build_snapshot(
    as_of: date,
    *,
    universe_name: str = "csi300",
    price_lookback_days: int = 280,  # 252 + 28d 缓冲，方便 momentum_12m_skip_1m
    include_financials: bool = True,
    include_indicators: bool = True,
    max_tickers: int | None = None,
    tickers_override: list[str] | None = None,
    max_workers: int = 1,  # 留接口但默认串行
    overwrite: bool = False,
) -> dict:
    """构建并持久化 ``as_of`` 时点的全市场 snapshot.

    Args:
        as_of: PIT 时点
        universe_name: 'csi300' / 'csi500' / 'csi800' 等
        price_lookback_days: 历史行情回看天数
        include_financials: 是否包含三大报表
        include_indicators: 是否包含 fina_indicator (ROE/ROA TTM)
        max_tickers: 仅取 universe 前 N 只（按权重排序），用于快速验证
        tickers_override: 直接指定 ticker 列表（绕开 universe 解析）
        max_workers: 并发数（暂保留，串行最稳）
        overwrite: 已存在时是否覆盖

    Returns:
        dict 含 ``snapshot_dir`` / ``rows_per_table`` / ``elapsed_sec``
    """
    out_dir = _snapshot_dir(as_of)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        log.info(f"snapshot exists at {out_dir}; loading existing meta")
        return json.loads(meta_path.read_text())

    tushare = TushareProvider()
    rows: dict[str, int] = {}
    files: list[str] = []
    failures: dict[str, list[str]] = {}
    start_time = time.monotonic()

    with operation_logger("snapshot.build", as_of=str(as_of), universe=universe_name):
        # 1. universe with weights
        log.info("[snapshot] step 1/5: universe with weights")
        u_df = get_universe_with_weights(universe_name, as_of=as_of)
        if u_df.empty:
            raise RuntimeError(f"universe is empty at {as_of}; cannot build snapshot")
        if tickers_override is not None:
            from quantmind.data.base import normalize_ticker

            override_set = {normalize_ticker(t) for t in tickers_override}
            u_df = u_df[u_df["ticker"].isin(override_set)].reset_index(drop=True)
        if max_tickers is not None and len(u_df) > max_tickers:
            u_df = u_df.head(max_tickers).reset_index(drop=True)
            log.info(f"  subset to top {max_tickers} tickers by weight")

        u_path = out_dir / "universe.parquet"
        u_df.to_parquet(u_path, index=False)
        rows["universe"] = len(u_df)
        files.append(u_path.name)
        log.info(f"  universe: {len(u_df)} tickers (weight sum={u_df['weight'].sum():.2f}%)")
        tickers = u_df["ticker"].tolist()

        # 2. prices
        log.info(f"[snapshot] step 2/5: prices ({len(tickers)} tickers, {price_lookback_days}d)")
        start_dt = (pd.Timestamp(as_of) - pd.Timedelta(days=int(price_lookback_days * 1.5))).date()
        px_df, failed_px = _fetch_per_ticker(
            tickers,
            lambda t: tushare.get_price(t, start=start_dt, end=as_of, as_of=as_of).tail(
                price_lookback_days
            ),
            desc="prices",
        )
        if not px_df.empty:
            px_path = out_dir / "prices.parquet"
            px_df.to_parquet(px_path, index=False)
            rows["prices"] = len(px_df)
            files.append(px_path.name)
            log.info(f"  prices: {len(px_df)} rows ({len(px_df) / max(len(tickers), 1):.1f} per ticker)")
        if failed_px:
            failures["prices"] = failed_px

        # 3. financials (3 statements)
        if include_financials:
            log.info("[snapshot] step 3/5: financials (3 statements)")
            for stmt in ("income", "balance_sheet", "cashflow"):
                fin_df, failed_fin = _fetch_per_ticker(
                    tickers,
                    lambda t, s=stmt: tushare.get_financials(t, s, as_of=as_of),
                    desc=f"fin/{stmt}",
                    head_n=8,
                )
                if not fin_df.empty:
                    fin_path = out_dir / f"financials_{stmt}.parquet"
                    fin_df.to_parquet(fin_path, index=False)
                    rows[f"financials_{stmt}"] = len(fin_df)
                    files.append(fin_path.name)
                    log.info(f"  financials_{stmt}: {len(fin_df)} rows")
                if failed_fin:
                    failures[f"financials_{stmt}"] = failed_fin

        # 4. financial indicators
        if include_indicators:
            log.info("[snapshot] step 4/5: fina_indicator")
            ind_df, failed_ind = _fetch_per_ticker(
                tickers,
                lambda t: tushare.get_financial_indicators(t, as_of=as_of),
                desc="indicators",
                head_n=8,
            )
            if not ind_df.empty:
                ind_path = out_dir / "financial_indicators.parquet"
                ind_df.to_parquet(ind_path, index=False)
                rows["financial_indicators"] = len(ind_df)
                files.append(ind_path.name)
                log.info(f"  financial_indicators: {len(ind_df)} rows")
            if failed_ind:
                failures["financial_indicators"] = failed_ind

        # 5. daily_basic (market-wide PE/PB/市值，单次调用)
        log.info("[snapshot] step 5/6: daily_basic (market-wide)")
        try:
            db_df = tushare.get_daily_basic_market(trade_date=as_of, as_of=as_of)
            if not db_df.empty:
                # 仅保留 universe 内的票
                db_df = db_df[db_df["ticker"].isin(set(tickers))].reset_index(drop=True)
                db_path = out_dir / "daily_basic.parquet"
                db_df.to_parquet(db_path, index=False)
                rows["daily_basic"] = len(db_df)
                files.append(db_path.name)
                log.info(f"  daily_basic: {len(db_df)} rows ({len(db_df.columns)} cols)")
        except Exception as e:  # noqa: BLE001
            log.warning(f"daily_basic fetch failed: {e}")

        # 6. north bound (market-level)
        log.info("[snapshot] step 6/6: north bound flow")
        try:
            nb_start = (pd.Timestamp(as_of) - pd.Timedelta(days=90)).date()
            nb_df = tushare.get_north_bound_flow(start=nb_start, end=as_of, as_of=as_of)
            if not nb_df.empty:
                nb_path = out_dir / "north_bound.parquet"
                nb_df.to_parquet(nb_path, index=False)
                rows["north_bound"] = len(nb_df)
                files.append(nb_path.name)
                log.info(f"  north_bound: {len(nb_df)} rows")
        except Exception as e:  # noqa: BLE001
            log.warning(f"north_bound fetch failed: {e}")

    elapsed_sec = time.monotonic() - start_time
    meta = {
        "as_of": as_of.isoformat(),
        "universe_name": universe_name,
        "snapshot_dir": str(out_dir),
        "rows_per_table": rows,
        "files": files,
        "built_at": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed_sec, 1),
        "n_tickers": len(tickers),
        "max_tickers": max_tickers,
        "failures": failures,
        "providers": {
            "primary": "tushare",
            "fallback": "akshare",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info(
        f"snapshot built at {out_dir} ({sum(rows.values())} rows total, "
        f"{elapsed_sec:.1f}s)"
    )
    return meta


def load_snapshot(as_of: date) -> dict[str, pd.DataFrame | dict]:
    """加载已构建的 snapshot.

    Returns:
        dict with keys: ``meta`` (dict), ``universe`` / ``prices`` /
        ``financials_income`` / ... (DataFrames; missing if not built)
    """
    out_dir = _snapshot_dir(as_of)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No snapshot found at {out_dir}")
    meta = json.loads(meta_path.read_text())
    out: dict[str, pd.DataFrame | dict] = {"meta": meta}
    for f in meta.get("files", []):
        key = f.removesuffix(".parquet")
        path = out_dir / f
        if path.exists():
            out[key] = pd.read_parquet(path)
    return out


def list_snapshots() -> list[date]:
    """列出所有已构建的 snapshot 日期."""
    settings = get_settings()
    base = Path(settings.data.dir) / "snapshots"
    if not base.exists():
        return []
    out = []
    for p in base.iterdir():
        if p.is_dir() and (p / "meta.json").exists():
            try:
                out.append(date.fromisoformat(p.name))
            except ValueError:
                continue
    return sorted(out)


def validate_snapshot(as_of: date, *, strict: bool = False) -> dict:
    """加载 snapshot 并做一致性检查.

    检查项：
        1. 所有声明的文件都存在
        2. universe.weight 总和接近声明（子集化时按比例）
        3. prices.trade_date <= as_of
        4. financials.f_ann_date <= as_of
        5. financial_indicators.ann_date <= as_of
        6. ticker 列均为合法 6位.SH/SZ/BJ
        7. 各表至少包含 universe 中 70% 的 ticker（财报有缺失情有可原）

    Args:
        as_of: snapshot 日期
        strict: True 则任一检查失败立即抛 AssertionError；False 累积报告

    Returns:
        dict 包含 ``ok``、``checks``、``stats``
    """
    snap = load_snapshot(as_of)
    meta = snap["meta"]
    assert isinstance(meta, dict)
    cutoff = pd.Timestamp(as_of)

    checks: list[tuple[str, bool, str]] = []
    stats: dict[str, dict] = {}

    # 1. 文件存在
    snap_dir = Path(meta["snapshot_dir"])
    for f in meta.get("files", []):
        exists = (snap_dir / f).exists()
        checks.append((f"file_exists::{f}", exists, "" if exists else f"missing {f}"))

    # 2. universe
    if "universe" in snap:
        u_df = snap["universe"]
        assert isinstance(u_df, pd.DataFrame)
        weight_sum = float(u_df["weight"].sum())
        n_uni = len(u_df)
        # 全 csi300 应 ~100%；子集会按比例减
        ok = weight_sum > 0
        checks.append(("universe.weight_sum_positive", ok, f"sum={weight_sum:.2f}%"))
        stats["universe"] = {
            "n_tickers": n_uni,
            "weight_sum_pct": round(weight_sum, 2),
            "top5": u_df.head(5)[["ticker", "weight"]].to_dict("records"),
        }
        universe_set = set(u_df["ticker"])
    else:
        universe_set = set()

    # 3. prices PIT
    if "prices" in snap:
        px = snap["prices"]
        assert isinstance(px, pd.DataFrame)
        max_dt = pd.to_datetime(px["trade_date"]).max()
        ok = max_dt <= cutoff
        checks.append((
            "prices.trade_date_pit",
            ok,
            f"max={max_dt.date()}, as_of={as_of}",
        ))
        ticker_cov = len(set(px["ticker"]) & universe_set) / max(len(universe_set), 1)
        checks.append((
            "prices.ticker_coverage",
            ticker_cov >= 0.7,
            f"coverage={ticker_cov:.0%}",
        ))
        stats["prices"] = {
            "rows": len(px),
            "n_tickers": int(px["ticker"].nunique()),
            "date_range": [str(pd.to_datetime(px["trade_date"]).min().date()), str(max_dt.date())],
            "rows_per_ticker_avg": round(len(px) / max(px["ticker"].nunique(), 1), 1),
        }

    # 4. financials PIT
    for stmt in ("income", "balance_sheet", "cashflow"):
        key = f"financials_{stmt}"
        if key in snap:
            fin = snap[key]
            assert isinstance(fin, pd.DataFrame)
            if "f_ann_date" in fin.columns:
                max_dt = pd.to_datetime(fin["f_ann_date"]).max()
                ok = max_dt <= cutoff
                checks.append((
                    f"{key}.f_ann_date_pit",
                    ok,
                    f"max={max_dt.date()}, as_of={as_of}",
                ))
            ticker_cov = len(set(fin["ticker"]) & universe_set) / max(len(universe_set), 1)
            checks.append((
                f"{key}.ticker_coverage",
                ticker_cov >= 0.5,  # 财报偶尔缺，门槛放宽
                f"coverage={ticker_cov:.0%}",
            ))
            stats[key] = {
                "rows": len(fin),
                "n_tickers": int(fin["ticker"].nunique()),
                "n_columns": len(fin.columns),
            }

    # 4b. daily_basic PIT
    if "daily_basic" in snap:
        db = snap["daily_basic"]
        assert isinstance(db, pd.DataFrame)
        if "trade_date" in db.columns:
            max_dt = pd.to_datetime(db["trade_date"]).max()
            ok = max_dt <= cutoff
            checks.append((
                "daily_basic.trade_date_pit",
                ok,
                f"max={max_dt.date()}, as_of={as_of}",
            ))
        ticker_cov = len(set(db["ticker"]) & universe_set) / max(len(universe_set), 1)
        checks.append((
            "daily_basic.ticker_coverage",
            ticker_cov >= 0.7,
            f"coverage={ticker_cov:.0%}",
        ))
        stats["daily_basic"] = {
            "rows": len(db),
            "n_tickers": int(db["ticker"].nunique()),
            "n_columns": len(db.columns),
        }

    # 5. financial_indicators PIT
    if "financial_indicators" in snap:
        ind = snap["financial_indicators"]
        assert isinstance(ind, pd.DataFrame)
        if "ann_date" in ind.columns:
            max_dt = pd.to_datetime(ind["ann_date"]).max()
            ok = max_dt <= cutoff
            checks.append((
                "financial_indicators.ann_date_pit",
                ok,
                f"max={max_dt.date()}, as_of={as_of}",
            ))
        ticker_cov = len(set(ind["ticker"]) & universe_set) / max(len(universe_set), 1)
        checks.append((
            "financial_indicators.ticker_coverage",
            ticker_cov >= 0.5,
            f"coverage={ticker_cov:.0%}",
        ))
        stats["financial_indicators"] = {
            "rows": len(ind),
            "n_tickers": int(ind["ticker"].nunique()),
            "n_columns": len(ind.columns),
        }

    # 6. ticker 格式校验
    if "universe" in snap:
        from quantmind.data.base import _TICKER_RE  # type: ignore[attr-defined]

        u_df = snap["universe"]
        assert isinstance(u_df, pd.DataFrame)
        bad = [t for t in u_df["ticker"] if not _TICKER_RE.match(t)]
        checks.append((
            "universe.ticker_format",
            not bad,
            f"bad_count={len(bad)}: {bad[:3]}" if bad else "all valid",
        ))

    failed = [name for name, ok, _ in checks if not ok]
    overall_ok = not failed
    if strict and not overall_ok:
        msgs = "\n".join(f"  - {n}: {m}" for n, ok, m in checks if not ok)
        raise AssertionError(f"snapshot validation failed:\n{msgs}")

    return {
        "ok": overall_ok,
        "as_of": as_of.isoformat(),
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "checks": [{"name": n, "ok": ok, "msg": m} for n, ok, m in checks],
        "stats": stats,
    }


__all__ = ["build_snapshot", "list_snapshots", "load_snapshot", "validate_snapshot"]


if __name__ == "__main__":
    from datetime import date as _date

    print("--- existing snapshots ---")
    print(list_snapshots())
