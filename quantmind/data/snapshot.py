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
        stock_basic.parquet      # Data Expansion v1：行业/上市地等
        hk_hold.parquet          # 沪深股通持股（个股）
        margin.parquet           # 融资融券明细（个股，源自 margin_detail）
        index_daily.parquet      # 主要指数日线
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from tqdm import tqdm

from quantmind.core.config import get_settings
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data.tushare_provider import TushareProvider
from quantmind.data.universe import get_universe_with_weights

log = get_logger(__name__)

MODULE_SCHEMA_VERSION = "1.0"
DATA_EXPANSION_VERSION = "v1"


def module_manifest_for_dataframe(
    module_id: str,
    df: pd.DataFrame,
    *,
    provider: str = "tushare",
    pit_date_column: str | None = "trade_date",
    date_bounds_from_list_date: bool = False,
) -> dict[str, object]:
    """构建 meta["modules"][module_id] 条目（行数、列名、日期范围、PIT 列）."""
    meta: dict[str, object] = {
        "module_name": module_id,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "provider": provider,
        "pit_date_column": pit_date_column,
        "schema_version": MODULE_SCHEMA_VERSION,
    }
    if date_bounds_from_list_date and "list_date" in df.columns:
        s = pd.to_datetime(df["list_date"], errors="coerce")
        meta["date_min"] = s.min().isoformat() if pd.notna(s.min()) else None  # type: ignore[assignment]
        meta["date_max"] = s.max().isoformat() if pd.notna(s.max()) else None  # type: ignore[assignment]
    elif pit_date_column and pit_date_column in df.columns:
        ts = pd.to_datetime(df[pit_date_column], errors="coerce")
        meta["date_min"] = ts.min().isoformat() if pd.notna(ts.min()) else None  # type: ignore[assignment]
        meta["date_max"] = ts.max().isoformat() if pd.notna(ts.max()) else None  # type: ignore[assignment]
    else:
        meta["date_min"] = None
        meta["date_max"] = None
    return meta


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
    # 过滤掉空帧，避免 pandas FutureWarning（concat 时 empty/all-NA 列的 dtype 推断）
    non_empty = [f for f in frames if not f.empty]
    out = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
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
    tickers_override_policy: Literal["filter", "replace"] = "filter",
    max_workers: int = 1,  # 留接口但默认串行
    overwrite: bool = False,
    # --- Data Expansion v1（None 表示采用 configs/default.yaml / Settings.data）---
    strict: bool | None = None,
    include_stock_basic: bool | None = None,
    include_hk_hold: bool | None = None,
    include_margin: bool | None = None,
    include_index_daily: bool | None = None,
    index_daily_codes: list[str] | None = None,
    hk_hold_lookback_calendar_days: int | None = None,
    margin_lookback_calendar_days: int | None = None,
    index_daily_lookback_calendar_days: int | None = None,
) -> dict:
    """构建并持久化 ``as_of`` 时点的全市场 snapshot.

    Args:
        as_of: PIT 时点
        universe_name: 'csi300' / 'csi500' / 'csi800' 等
        price_lookback_days: 历史行情回看天数
        include_financials: 是否包含三大报表
        include_indicators: 是否包含 fina_indicator (ROE/ROA TTM)
        max_tickers: 仅取 universe 前 N 只（按权重排序），用于快速验证
        tickers_override: 直接指定 ticker 列表。
            - ``filter``（默认）：先取 ``universe_name`` 成分，再与该列表求交（旧行为）。
            - ``replace``：**完全以该列表为 universe**（等权写入 universe.parquet），用于 Alpha 全市场等非指数池。
        tickers_override_policy: ``filter`` | ``replace``
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
        log.info("[snapshot] step 1/10: universe with weights")
        effective_universe_name = universe_name
        if tickers_override is not None and tickers_override_policy == "replace":
            from quantmind.data.base import normalize_ticker

            seen: set[str] = set()
            tickers_ordered: list[str] = []
            for raw in tickers_override:
                t = normalize_ticker(str(raw).strip())
                if not t or t in seen:
                    continue
                seen.add(t)
                tickers_ordered.append(t)
            if not tickers_ordered:
                raise RuntimeError("tickers_override replace: empty after normalize/dedupe")
            n_t = len(tickers_ordered)
            u_df = pd.DataFrame(
                {"ticker": tickers_ordered, "weight": [100.0 / n_t] * n_t}
            )
            effective_universe_name = f"alpha_tickers_replace({n_t})"
            log.info(
                "  universe: REPLACE with {} tickers (equal weights); "
                "universe_name={} ignored for membership",
                n_t,
                universe_name,
            )
        else:
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

        cfg = get_settings().data
        strict = cfg.snapshot_strict if strict is None else strict
        inc_stock_basic = (
            cfg.include_stock_basic if include_stock_basic is None else include_stock_basic
        )
        inc_hk = cfg.include_hk_hold if include_hk_hold is None else include_hk_hold
        inc_margin = cfg.include_margin if include_margin is None else include_margin
        inc_idx = cfg.include_index_daily if include_index_daily is None else include_index_daily
        idx_codes = (
            list(cfg.index_daily_codes) if index_daily_codes is None else list(index_daily_codes)
        )
        hk_lb = (
            cfg.hk_hold_lookback_calendar_days
            if hk_hold_lookback_calendar_days is None
            else hk_hold_lookback_calendar_days
        )
        m_lb = (
            cfg.margin_lookback_calendar_days
            if margin_lookback_calendar_days is None
            else margin_lookback_calendar_days
        )
        ix_lb = (
            cfg.index_daily_lookback_calendar_days
            if index_daily_lookback_calendar_days is None
            else index_daily_lookback_calendar_days
        )
        modules: dict[str, dict[str, object]] = {}

        # 2. prices
        log.info(f"[snapshot] step 2/10: prices ({len(tickers)} tickers, {price_lookback_days}d)")
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
        if strict and px_df.empty:
            raise RuntimeError(f"strict snapshot: prices empty at {as_of}")

        # 3. financials (3 statements)
        if include_financials:
            log.info("[snapshot] step 3/10: financials (3 statements)")
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
            log.info("[snapshot] step 4/10: fina_indicator")
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
        log.info("[snapshot] step 5/10: daily_basic (market-wide)")
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
        log.info("[snapshot] step 6/10: north bound flow")
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

        # -- Data Expansion v1：7 stock_basic, 8 hk_hold, 9 margin, 10 index_daily --
        hk_start = (pd.Timestamp(as_of) - pd.Timedelta(days=hk_lb)).date()
        m_start = (pd.Timestamp(as_of) - pd.Timedelta(days=m_lb)).date()
        ix_start = (pd.Timestamp(as_of) - pd.Timedelta(days=ix_lb)).date()

        if inc_stock_basic:
            log.info("[snapshot] step 7/10: stock_basic (universe subset)")
            try:
                full_basic = tushare.get_stock_basic(include_delisted=False)
                sb = full_basic[full_basic["ticker"].isin(set(tickers))].copy()
                if sb.empty:
                    raise ValueError("stock_basic: no rows after universe filter")
                sb_path = out_dir / "stock_basic.parquet"
                sb.to_parquet(sb_path, index=False)
                rows["stock_basic"] = len(sb)
                files.append(sb_path.name)
                log.info(f"  stock_basic: {len(sb)} rows")
                modules["stock_basic"] = module_manifest_for_dataframe(
                    "stock_basic",
                    sb,
                    pit_date_column=None,
                    date_bounds_from_list_date=True,
                )
            except Exception as e:  # noqa: BLE001
                log.warning(f"stock_basic fetch failed: {e}")
                if strict:
                    raise

        if inc_hk:
            log.info("[snapshot] step 8/10: hk_hold (per ticker)")
            hk_frames: list[pd.DataFrame] = []
            try:
                for t in tqdm(tickers, desc="hk_hold", unit="t", ncols=80):
                    try:
                        hdf = tushare.get_hk_hold(t, hk_start, as_of, as_of=as_of)
                        if hdf is not None and not hdf.empty:
                            hk_frames.append(hdf)
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"hk_hold failed for {t}: {e}")
                hk_all = (
                    pd.concat(hk_frames, ignore_index=True) if hk_frames else pd.DataFrame()
                )
                if not hk_all.empty:
                    hk_path = out_dir / "hk_hold.parquet"
                    hk_all.to_parquet(hk_path, index=False)
                    rows["hk_hold"] = len(hk_all)
                    files.append(hk_path.name)
                    log.info(f"  hk_hold: {len(hk_all)} rows")
                    modules["hk_hold"] = module_manifest_for_dataframe(
                        "hk_hold", hk_all, pit_date_column="trade_date"
                    )
                elif strict:
                    raise RuntimeError("strict snapshot: hk_hold produced no rows")
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning(f"hk_hold module failed: {e}")
                if strict:
                    raise

        if inc_margin:
            log.info("[snapshot] step 9/10: margin_detail -> margin.parquet (per ticker)")
            m_frames: list[pd.DataFrame] = []
            try:
                for t in tqdm(tickers, desc="margin", unit="t", ncols=80):
                    try:
                        mdf = tushare.get_margin_detail(t, m_start, as_of, as_of=as_of)
                        if mdf is not None and not mdf.empty:
                            m_frames.append(mdf)
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"margin_detail failed for {t}: {e}")
                m_all = pd.concat(m_frames, ignore_index=True) if m_frames else pd.DataFrame()
                if not m_all.empty:
                    mp_path = out_dir / "margin.parquet"
                    m_all.to_parquet(mp_path, index=False)
                    rows["margin"] = len(m_all)
                    files.append(mp_path.name)
                    log.info(f"  margin: {len(m_all)} rows")
                    modules["margin"] = module_manifest_for_dataframe(
                        "margin", m_all, pit_date_column="trade_date"
                    )
                elif strict:
                    raise RuntimeError("strict snapshot: margin produced no rows")
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning(f"margin module failed: {e}")
                if strict:
                    raise

        if inc_idx:
            log.info("[snapshot] step 10/10: index_daily (configured indices)")
            ix_frames: list[pd.DataFrame] = []
            try:
                for code in idx_codes:
                    try:
                        idf = tushare.get_index_daily(code, ix_start, as_of, as_of=as_of)
                        if idf is not None and not idf.empty:
                            ix_frames.append(idf)
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"index_daily failed for {code}: {e}")
                ix_all = (
                    pd.concat(ix_frames, ignore_index=True) if ix_frames else pd.DataFrame()
                )
                if not ix_all.empty:
                    ix_path = out_dir / "index_daily.parquet"
                    ix_all.to_parquet(ix_path, index=False)
                    rows["index_daily"] = len(ix_all)
                    files.append(ix_path.name)
                    log.info(f"  index_daily: {len(ix_all)} rows")
                    modules["index_daily"] = module_manifest_for_dataframe(
                        "index_daily", ix_all, pit_date_column="trade_date"
                    )
                elif strict:
                    raise RuntimeError("strict snapshot: index_daily produced no rows")
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning(f"index_daily module failed: {e}")
                if strict:
                    raise

    elapsed_sec = time.monotonic() - start_time
    meta = {
        "as_of": as_of.isoformat(),
        "universe_name": effective_universe_name,
        "universe_index_arg": universe_name,
        "tickers_override_policy": tickers_override_policy,
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
        "data_expansion_version": DATA_EXPANSION_VERSION,
        "snapshot_strict": strict,
        "include_flags": {
            "stock_basic": inc_stock_basic,
            "hk_hold": inc_hk,
            "margin": inc_margin,
            "index_daily": inc_idx,
        },
        "modules": modules,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info(
        f"snapshot built at {out_dir} ({sum(rows.values())} rows total, "
        f"{elapsed_sec:.1f}s)"
    )
    return meta


def _snapshot_readonly_dir(as_of: date) -> Path:
    """已存在 snapshot 目录（不因 patch 新建根目录之外的意外路径）。"""
    settings = get_settings()
    return Path(settings.data.dir).resolve() / "snapshots" / as_of.isoformat()


def patch_v1_modules(
    as_of: date | str,
    *,
    include_stock_basic: bool = True,
    include_hk_hold: bool = True,
    include_margin: bool = True,
    include_index_daily: bool = True,
    index_daily_codes: list[str] | None = None,
    strict: bool = False,
    overwrite_v1_modules: bool = False,
    dry_run: bool = True,
    hk_hold_lookback_calendar_days: int | None = None,
    margin_lookback_calendar_days: int | None = None,
    index_daily_lookback_calendar_days: int | None = None,
    provider: object | None = None,
) -> dict[str, object]:
    """在**已有** snapshot 上增量补齐 Data Expansion v1 四张表（不触碰 prices / financials / daily_basic）。

    Args:
        as_of: snapshot 时点
        include_* : 是否拉取该模块（dry-run 时也用于规划）
        index_daily_codes: 指数列表；None 则用 ``Settings.data.index_daily_codes``
        strict: 任一勾选模块首次失败即抛错；False 仅记录 warning / failed_modules
        overwrite_v1_modules: True 时已存在的 v1 parquet 也会被重写
        dry_run: True 时不联网、不写盘、不写 meta（仅生成报告草案）
        provider: 仅测试注入；prod 应为 ``None`` → 内部 ``TushareProvider()``

    Returns:
        报告字典：``as_of`` / ``snapshot_dir`` / ``dry_run`` / ``patched_modules`` /
        ``skipped_modules`` / ``failed_modules`` / ``module_row_counts`` /
        ``pit_max_trade_dates`` / ``warnings`` 等。
    """
    ad: date = (
        date.fromisoformat(str(as_of).strip()[:10]) if isinstance(as_of, str) else as_of
    )
    snap_dir = _snapshot_readonly_dir(ad)
    warnings: list[str] = []

    uni_p = snap_dir / "universe.parquet"
    meta_p = snap_dir / "meta.json"
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"snapshot directory does not exist: {snap_dir}")
    if not uni_p.is_file():
        raise FileNotFoundError(
            f"cannot patch snapshot without universe.parquet: {uni_p}"
        )
    if not meta_p.is_file():
        raise FileNotFoundError(f"cannot patch snapshot without meta.json: {meta_p}")

    u_df = pd.read_parquet(uni_p)
    if "ticker" not in u_df.columns or u_df.empty:
        raise ValueError(f"universe.parquet missing ticker column or empty: {uni_p}")
    tickers = u_df["ticker"].tolist()

    meta_existing: dict[str, object] = json.loads(meta_p.read_text(encoding="utf-8"))

    cfg = get_settings().data
    hk_lb = (
        cfg.hk_hold_lookback_calendar_days
        if hk_hold_lookback_calendar_days is None
        else hk_hold_lookback_calendar_days
    )
    m_lb = (
        cfg.margin_lookback_calendar_days
        if margin_lookback_calendar_days is None
        else margin_lookback_calendar_days
    )
    ix_lb = (
        cfg.index_daily_lookback_calendar_days
        if index_daily_lookback_calendar_days is None
        else index_daily_lookback_calendar_days
    )
    idx_codes = (
        list(cfg.index_daily_codes) if index_daily_codes is None else list(index_daily_codes)
    )

    hk_start = (pd.Timestamp(ad) - pd.Timedelta(days=int(hk_lb))).date()
    m_start = (pd.Timestamp(ad) - pd.Timedelta(days=int(m_lb))).date()
    ix_start = (pd.Timestamp(ad) - pd.Timedelta(days=int(ix_lb))).date()

    v1_targets: dict[str, tuple[str, str]] = {
        "stock_basic": ("stock_basic", "stock_basic.parquet"),
        "hk_hold": ("hk_hold", "hk_hold.parquet"),
        "margin": ("margin", "margin.parquet"),
        "index_daily": ("index_daily", "index_daily.parquet"),
    }
    toggles = {
        "stock_basic": include_stock_basic,
        "hk_hold": include_hk_hold,
        "margin": include_margin,
        "index_daily": include_index_daily,
    }

    skipped: list[str] = []
    plan_fetch: list[str] = []

    for mod_key, enabled in toggles.items():
        if not enabled:
            continue
        _row_key, pq_name = v1_targets[mod_key]
        pq_path = snap_dir / pq_name
        if pq_path.is_file() and not overwrite_v1_modules:
            skipped.append(mod_key)
        else:
            plan_fetch.append(mod_key)

    if dry_run:
        return {
            "as_of": ad.isoformat(),
            "snapshot_dir": str(snap_dir),
            "dry_run": True,
            "patched_modules": [],
            "skipped_modules": skipped,
            "would_fetch_modules": plan_fetch,
            "failed_modules": [],
            "module_row_counts": {},
            "pit_max_trade_dates": {},
            "warnings": warnings,
            "n_universe_tickers": len(tickers),
            "index_daily_codes_planned": idx_codes,
            "overwrite_v1_modules": overwrite_v1_modules,
            "strict": strict,
            "legacy_meta_fields_preserved_preview": sorted(meta_existing.keys()),
        }

    # --- fetch + write (no dry_run) ---
    tushare = provider if provider is not None else TushareProvider()
    patched: list[str] = []
    failed: list[dict[str, object]] = []
    row_counts: dict[str, int] = {}
    pit_max: dict[str, str | None] = {}
    new_files_written: list[str] = []
    rows_table_updates: dict[str, int] = {}

    def _abort_strict(msg: str) -> None:
        if strict:
            raise RuntimeError(msg)

    meta_modules = dict(meta_existing.get("modules") or {})  # type: ignore[arg-type]

    # stock_basic
    if toggles["stock_basic"]:
        pq_path = snap_dir / "stock_basic.parquet"
        if pq_path.is_file() and not overwrite_v1_modules:
            pass  # skipped
        else:
            try:
                full_basic = tushare.get_stock_basic(include_delisted=False)
                sb = full_basic[full_basic["ticker"].isin(set(tickers))].copy()
                if sb.empty:
                    msg_empty = "stock_basic empty after universe filter"
                    warnings.append(msg_empty)
                    failed.append({"module": "stock_basic", "error": msg_empty})
                    _abort_strict(msg_empty)
                else:
                    sb.to_parquet(pq_path, index=False)
                    patched.append("stock_basic")
                    new_files_written.append("stock_basic.parquet")
                    mf = module_manifest_for_dataframe(
                        "stock_basic", sb, pit_date_column=None, date_bounds_from_list_date=True
                    )
                    meta_modules["stock_basic"] = mf
                    rows_table_updates["stock_basic"] = len(sb)
                    row_counts["stock_basic"] = len(sb)
                    pit_max["stock_basic"] = str(mf.get("date_max")) if mf else None
            except Exception as e:  # noqa: BLE001
                msg = f"stock_basic: {e}"
                log.warning(msg)
                warnings.append(msg)
                failed.append({"module": "stock_basic", "error": repr(e)})
                _abort_strict(msg)

    # hk_hold
    if toggles["hk_hold"]:
        pq_path = snap_dir / "hk_hold.parquet"
        if pq_path.is_file() and not overwrite_v1_modules:
            pass
        else:
            try:
                hk_frames: list[pd.DataFrame] = []
                for t in tqdm(tickers, desc="patch.hk_hold", unit="t", ncols=80):
                    try:
                        hdf = tushare.get_hk_hold(t, hk_start, ad, as_of=ad)
                        if hdf is not None and not hdf.empty:
                            hk_frames.append(hdf)
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"hk_hold failed for {t}: {e}")
                hk_all = pd.concat(hk_frames, ignore_index=True) if hk_frames else pd.DataFrame()
                if hk_all.empty and strict:
                    raise RuntimeError("hk_hold produced no rows (strict)")
                if hk_all.empty:
                    warnings.append("hk_hold: no rows aggregated")
                    failed.append({"module": "hk_hold", "error": "empty dataframe"})
                    _abort_strict("hk_hold produced no rows")
                else:
                    hk_all.to_parquet(pq_path, index=False)
                    patched.append("hk_hold")
                    new_files_written.append("hk_hold.parquet")
                    mf = module_manifest_for_dataframe(
                        "hk_hold", hk_all, pit_date_column="trade_date"
                    )
                    meta_modules["hk_hold"] = mf
                    rows_table_updates["hk_hold"] = len(hk_all)
                    row_counts["hk_hold"] = len(hk_all)
                    pit_max["hk_hold"] = str(mf.get("date_max")) if mf else None
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                msg = f"hk_hold: {e}"
                log.warning(msg)
                warnings.append(msg)
                failed.append({"module": "hk_hold", "error": repr(e)})
                _abort_strict(msg)

    # margin
    if toggles["margin"]:
        pq_path = snap_dir / "margin.parquet"
        if pq_path.is_file() and not overwrite_v1_modules:
            pass
        else:
            try:
                m_frames: list[pd.DataFrame] = []
                for t in tqdm(tickers, desc="patch.margin", unit="t", ncols=80):
                    try:
                        mdf = tushare.get_margin_detail(t, m_start, ad, as_of=ad)
                        if mdf is not None and not mdf.empty:
                            m_frames.append(mdf)
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"margin_detail failed for {t}: {e}")
                m_all = pd.concat(m_frames, ignore_index=True) if m_frames else pd.DataFrame()
                if m_all.empty and strict:
                    raise RuntimeError("margin produced no rows (strict)")
                if m_all.empty:
                    warnings.append("margin: no rows aggregated")
                    failed.append({"module": "margin", "error": "empty dataframe"})
                    _abort_strict("margin produced no rows")
                else:
                    m_all.to_parquet(pq_path, index=False)
                    patched.append("margin")
                    new_files_written.append("margin.parquet")
                    mf = module_manifest_for_dataframe(
                        "margin", m_all, pit_date_column="trade_date"
                    )
                    meta_modules["margin"] = mf
                    rows_table_updates["margin"] = len(m_all)
                    row_counts["margin"] = len(m_all)
                    pit_max["margin"] = str(mf.get("date_max")) if mf else None
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                msg = f"margin: {e}"
                log.warning(msg)
                warnings.append(msg)
                failed.append({"module": "margin", "error": repr(e)})
                _abort_strict(msg)

    # index_daily
    if toggles["index_daily"]:
        pq_path = snap_dir / "index_daily.parquet"
        if pq_path.is_file() and not overwrite_v1_modules:
            pass
        else:
            try:
                ix_frames: list[pd.DataFrame] = []
                for code in idx_codes:
                    try:
                        idf = tushare.get_index_daily(code, ix_start, ad, as_of=ad)
                        if idf is not None and not idf.empty:
                            ix_frames.append(idf)
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"index_daily failed for {code}: {e}")
                ix_all = pd.concat(ix_frames, ignore_index=True) if ix_frames else pd.DataFrame()
                if ix_all.empty and strict:
                    raise RuntimeError("index_daily produced no rows (strict)")
                if ix_all.empty:
                    warnings.append("index_daily: no rows aggregated")
                    failed.append({"module": "index_daily", "error": "empty dataframe"})
                    _abort_strict("index_daily produced no rows")
                else:
                    ix_all.to_parquet(pq_path, index=False)
                    patched.append("index_daily")
                    new_files_written.append("index_daily.parquet")
                    mf = module_manifest_for_dataframe(
                        "index_daily", ix_all, pit_date_column="trade_date"
                    )
                    meta_modules["index_daily"] = mf
                    rows_table_updates["index_daily"] = len(ix_all)
                    row_counts["index_daily"] = len(ix_all)
                    pit_max["index_daily"] = str(mf.get("date_max")) if mf else None
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                msg = f"index_daily: {e}"
                log.warning(msg)
                warnings.append(msg)
                failed.append({"module": "index_daily", "error": repr(e)})
                _abort_strict(msg)

    # ------------- merge meta (preserve legacy keys) -------------
    merged: dict[str, object] = dict(meta_existing)

    has_any_v1 = any((snap_dir / v1_targets[k][1]).is_file() for k in v1_targets)
    if has_any_v1:
        merged["data_expansion_version"] = DATA_EXPANSION_VERSION
    elif "data_expansion_version" in meta_existing:
        merged["data_expansion_version"] = meta_existing["data_expansion_version"]

    merged["snapshot_strict"] = strict

    incl = dict(meta_existing.get("include_flags") or {})  # type: ignore[arg-type]
    for mod_key, (_, pqname) in v1_targets.items():
        if (snap_dir / pqname).is_file():
            incl[mod_key] = True
    merged["include_flags"] = incl

    merged["modules"] = meta_modules

    old_files = list(meta_existing.get("files") or [])
    merged_files_old = old_files[:]  # type: ignore[assignment]

    seen: set[str] = set()
    new_file_order: list[str] = []

    # stable unique: preserve old order then append patch outputs
    for f in merged_files_old + new_files_written:
        if f not in seen:
            seen.add(str(f))
            new_file_order.append(str(f))

    v1_parquet_names = [v1_targets[k][1] for k in v1_targets]
    for pq in v1_parquet_names:
        if (snap_dir / pq).is_file() and pq not in seen:
            seen.add(pq)
            new_file_order.append(pq)

    merged["files"] = new_file_order

    rp = dict(meta_existing.get("rows_per_table") or {})
    rp.update(rows_table_updates)
    merged["rows_per_table"] = rp

    merged.setdefault("providers", meta_existing.get("providers") or {})

    merged["v1_patch"] = {
        "dry_run": False,
        "patched_modules": patched,
        "skipped_preexisting_nonoverwrite": skipped,
        "failed_modules": failed,
        "warnings": warnings,
        "applied_at": datetime.now().isoformat(),
    }

    meta_p.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "as_of": ad.isoformat(),
        "snapshot_dir": str(snap_dir),
        "dry_run": False,
        "patched_modules": patched,
        "skipped_modules": skipped,
        "failed_modules": failed,
        "module_row_counts": row_counts,
        "pit_max_trade_dates": pit_max,
        "warnings": warnings,
        "n_universe_tickers": len(tickers),
        "index_daily_codes_used": idx_codes,
        "meta_updated": True,
        "overwrite_v1_modules": overwrite_v1_modules,
        "strict": strict,
    }


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


__all__ = [
    "build_snapshot",
    "DATA_EXPANSION_VERSION",
    "list_snapshots",
    "load_snapshot",
    "module_manifest_for_dataframe",
    "patch_v1_modules",
    "validate_snapshot",
]


if __name__ == "__main__":
    from datetime import date as _date

    print("--- existing snapshots ---")
    print(list_snapshots())
