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
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from quantmind.core.config import get_settings
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data.akshare_provider import AkshareProvider
from quantmind.data.tushare_provider import TushareProvider
from quantmind.data.universe import get_universe_with_weights

log = get_logger(__name__)


def _snapshot_dir(as_of: date) -> Path:
    settings = get_settings()
    base = Path(settings.data.dir) / "snapshots" / as_of.isoformat()
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_snapshot(
    as_of: date,
    *,
    universe_name: str = "csi300",
    price_lookback_days: int = 252,
    include_financials: bool = True,
    include_indicators: bool = True,
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
    AkshareProvider()  # 触发 import 以暴露错误（暂未使用）

    rows: dict[str, int] = {}
    files: list[str] = []

    with operation_logger("snapshot.build", as_of=str(as_of), universe=universe_name):
        # 1. universe with weights
        log.info("[snapshot] step 1/5: universe with weights")
        u_df = get_universe_with_weights(universe_name, as_of=as_of)
        if u_df.empty:
            raise RuntimeError(f"universe is empty at {as_of}; cannot build snapshot")
        u_path = out_dir / "universe.parquet"
        u_df.to_parquet(u_path, index=False)
        rows["universe"] = len(u_df)
        files.append(u_path.name)
        log.info(f"  universe: {len(u_df)} tickers")
        tickers = u_df["ticker"].tolist()

        # 2. prices (per ticker, batched serially with cache)
        log.info(f"[snapshot] step 2/5: prices ({len(tickers)} tickers, {price_lookback_days}d)")
        start_dt = (pd.Timestamp(as_of) - pd.Timedelta(days=int(price_lookback_days * 1.5))).date()
        price_frames = []
        for i, t in enumerate(tickers, 1):
            try:
                px = tushare.get_price(t, start=start_dt, end=as_of, as_of=as_of)
                if not px.empty:
                    price_frames.append(px.tail(price_lookback_days))
            except Exception as e:  # noqa: BLE001
                log.warning(f"price fetch failed for {t}: {e}")
            if i % 50 == 0:
                log.info(f"  price progress: {i}/{len(tickers)}")
        if price_frames:
            px_df = pd.concat(price_frames, ignore_index=True)
            px_path = out_dir / "prices.parquet"
            px_df.to_parquet(px_path, index=False)
            rows["prices"] = len(px_df)
            files.append(px_path.name)
            log.info(f"  prices: {len(px_df)} rows")

        # 3. financials (3 statements)
        if include_financials:
            log.info("[snapshot] step 3/5: financials")
            for stmt in ("income", "balance_sheet", "cashflow"):
                fin_frames = []
                for i, t in enumerate(tickers, 1):
                    try:
                        f = tushare.get_financials(t, stmt, as_of=as_of)
                        if not f.empty:
                            fin_frames.append(f.head(8))  # 最近 8 期
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"financials({stmt}) failed for {t}: {e}")
                    if i % 50 == 0:
                        log.info(f"  {stmt} progress: {i}/{len(tickers)}")
                if fin_frames:
                    fin_df = pd.concat(fin_frames, ignore_index=True)
                    fin_path = out_dir / f"financials_{stmt}.parquet"
                    fin_df.to_parquet(fin_path, index=False)
                    rows[f"financials_{stmt}"] = len(fin_df)
                    files.append(fin_path.name)
                    log.info(f"  financials_{stmt}: {len(fin_df)} rows")

        # 4. financial indicators
        if include_indicators:
            log.info("[snapshot] step 4/5: fina_indicator")
            ind_frames = []
            for i, t in enumerate(tickers, 1):
                try:
                    ind = tushare.get_financial_indicators(t, as_of=as_of)
                    if not ind.empty:
                        ind_frames.append(ind.head(8))
                except Exception as e:  # noqa: BLE001
                    log.warning(f"fina_indicator failed for {t}: {e}")
                if i % 50 == 0:
                    log.info(f"  ind progress: {i}/{len(tickers)}")
            if ind_frames:
                ind_df = pd.concat(ind_frames, ignore_index=True)
                ind_path = out_dir / "financial_indicators.parquet"
                ind_df.to_parquet(ind_path, index=False)
                rows["financial_indicators"] = len(ind_df)
                files.append(ind_path.name)
                log.info(f"  financial_indicators: {len(ind_df)} rows")

        # 5. north bound (market-level)
        log.info("[snapshot] step 5/5: north bound flow")
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

    meta = {
        "as_of": as_of.isoformat(),
        "universe_name": universe_name,
        "snapshot_dir": str(out_dir),
        "rows_per_table": rows,
        "files": files,
        "built_at": datetime.now().isoformat(),
        "providers": {
            "primary": "tushare",
            "fallback": "akshare",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info(f"snapshot built at {out_dir} ({sum(rows.values())} rows total)")
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


__all__ = ["build_snapshot", "list_snapshots", "load_snapshot"]


if __name__ == "__main__":
    from datetime import date as _date

    print("--- existing snapshots ---")
    print(list_snapshots())
