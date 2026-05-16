"""quantmind.features.pipeline — 因子计算编排.

主要类
======

``FeaturePipeline``：从 snapshot 加载数据，并行计算所有因子，
应用横截面标准化（行业/市值中性化、winsorize、zscore），
持久化到 ``data/features/{universe}_{as_of}.parquet``。

使用
====

::

    from quantmind.features import FeaturePipeline
    pipe = FeaturePipeline()
    df = pipe.run_single(as_of=date(2024, 6, 30))
    # df: index=ticker, columns=factor names, values=standardized factor scores
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.core.config import get_settings
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data import load_snapshot
from quantmind.features.expansion import (
    CATEGORICAL_EXPOSURE_FIELDS,
    EXPANSION_FACTORS,
    MARKET_LEVEL_FEATURES,
    compute_all_expansion_factors,
)
from quantmind.features.fundamental import (
    FUNDAMENTAL_FACTORS,
    compute_all_fundamental_factors,
)
from quantmind.features.sentiment import (
    MARKET_LEVEL_SENTIMENT_FACTORS,
    SENTIMENT_FACTORS,
    compute_all_sentiment_factors,
)
from quantmind.features.standardize import (
    compute_market_factor_stats,
    standardize,
    timeseries_zscore,
)
from quantmind.features.technical import (
    TECHNICAL_FACTORS,
    compute_all_technical_factors,
)

log = get_logger(__name__)


def _snapshot_df_with_ticker(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """快照里的 daily_basic / stock_basic 可能只有 ts_code，对齐因子索引需 ticker."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out = df.copy()
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str)
        return out
    if "ts_code" in out.columns:
        out["ticker"] = out["ts_code"].astype(str)
        return out
    log.warning("snapshot table missing ticker/ts_code columns: {}", list(out.columns)[:12])
    return None


# 因子分类（用于元数据 + 选择性计算）
ALL_FACTOR_GROUPS = {
    "fundamental": FUNDAMENTAL_FACTORS,
    "technical": TECHNICAL_FACTORS,
    # 与 compute_all_sentiment_factors 一致：个股情绪 + 市场级北向等
    "sentiment": SENTIMENT_FACTORS + MARKET_LEVEL_SENTIMENT_FACTORS,
    "expansion": EXPANSION_FACTORS,
}


def list_all_factor_names() -> list[str]:
    """返回所有可用因子名."""
    out = []
    for fs in ALL_FACTOR_GROUPS.values():
        out.extend([name for name, _ in fs])
    return out


# ============================================================================
# FeaturePipeline
# ============================================================================


class FeaturePipeline:
    """编排因子的端到端计算与标准化."""

    def __init__(
        self,
        *,
        groups: list[str] | None = None,
        do_standardize: bool = True,
        winsorize_sigma: float = 3.0,
        do_neutralize: bool = True,
    ) -> None:
        self.groups = groups or list(ALL_FACTOR_GROUPS.keys())
        self.do_standardize = do_standardize
        self.winsorize_sigma = winsorize_sigma
        self.do_neutralize = do_neutralize

    # -------------------------------------------------------------------
    # 单时点计算
    # -------------------------------------------------------------------

    def compute_raw_from_snapshot(self, snap: dict, as_of: date) -> pd.DataFrame:
        """与 :meth:`compute_raw` 相同逻辑，但使用已加载的 snapshot（便于离线路径测试）."""
        with operation_logger("features.compute_raw", as_of=str(as_of)):
            parts: list[pd.DataFrame] = []
            if "fundamental" in self.groups:
                parts.append(compute_all_fundamental_factors(snap, as_of))
            if "technical" in self.groups:
                parts.append(compute_all_technical_factors(snap, as_of))
            if "sentiment" in self.groups:
                parts.append(compute_all_sentiment_factors(snap, as_of))
            if "expansion" in self.groups:
                parts.append(compute_all_expansion_factors(snap, as_of))

            u_df = snap.get("universe")
            if u_df is None or u_df.empty:
                raise RuntimeError("snapshot.universe is empty")
            assert isinstance(u_df, pd.DataFrame)
            master_idx = pd.Index(u_df["ticker"], name="ticker")

            full = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=master_idx)
            full = full.reindex(master_idx)
            log.info(
                f"computed {full.shape[1]} factors for {full.shape[0]} tickers "
                f"@ {as_of}"
            )
            return full

    def compute_raw(self, as_of: date) -> pd.DataFrame:
        """计算所有因子的「原始值」（未标准化），返回 (ticker × factor).

        如果 snapshot 缺失，会抛 FileNotFoundError 给上层处理。
        """
        snap = load_snapshot(as_of)
        return self.compute_raw_from_snapshot(snap, as_of)

    def standardize(
        self, df: pd.DataFrame, snapshot: dict | None = None
    ) -> pd.DataFrame:
        """对原始因子值做 winsorize + 行业/市值中性化 + zscore."""
        if not self.do_standardize:
            return df

        # 行业（stock_basic.industry）+ 市值（daily_basic）用于中性化
        industry_ser: pd.Series | None = None
        log_mv: pd.Series | None = None
        if snapshot is not None:
            db = snapshot.get("daily_basic")
            db_use = _snapshot_df_with_ticker(db)
            if db_use is not None and "total_mv" in db_use.columns:
                dbb = db_use
                if "trade_date" in dbb.columns:
                    dbb = dbb.sort_values("trade_date")
                dbb = dbb.drop_duplicates(subset=["ticker"], keep="last")
                log_mv = pd.Series(
                    np.log(dbb.set_index("ticker")["total_mv"].clip(lower=1e-9)),
                    name="log_mv",
                ).reindex(df.index)
            sb = snapshot.get("stock_basic")
            sb_use = _snapshot_df_with_ticker(sb)
            if sb_use is not None and "industry" in sb_use.columns:
                industry_ser = (
                    sb_use.drop_duplicates(subset=["ticker"])
                    .set_index("ticker")["industry"]
                    .astype("string")
                    .fillna("__NA_INDUSTRY__")
                )
                industry_ser = industry_ser.reindex(df.index, fill_value="__NA_INDUSTRY__")

        return standardize(
            df,
            industry=industry_ser,
            log_market_cap=log_mv,
            winsorize_sigma=self.winsorize_sigma,
            do_neutralize=self.do_neutralize,
            zscore_winsorize_exclude=set(MARKET_LEVEL_FEATURES),
            categorical_columns=set(CATEGORICAL_EXPOSURE_FIELDS),
        )

    # -------------------------------------------------------------------
    # 市场级因子时序标准化
    # -------------------------------------------------------------------

    # 所有需要时序 z-score 的市场级因子名（来自 expansion + sentiment，去重）
    _MARKET_LEVEL_FACTORS: list[str] = list(
        dict.fromkeys(
            list(MARKET_LEVEL_FEATURES) + [name for name, _ in MARKET_LEVEL_SENTIMENT_FACTORS]
        )
    )

    def _apply_market_timeseries_zscore(
        self, df: pd.DataFrame, *, universe: str = "csi300"
    ) -> pd.DataFrame:
        """对 MARKET_LEVEL_FACTORS 做时序 z-score（跨历史面板）.

        策略：
        1. 从 data/features/ 加载所有已存在的单截面 parquet 文件
        2. 提取市场级因子的历史时序值（每期取第一只股票的值，因为所有股票相同）
        3. 用 timeseries_zscore 对当前值标准化
        4. 原始值为 NaN 或历史不足时保留原始值

        若找不到任何历史文件，直接返回原 df（不报错）。
        """
        market_cols = [c for c in self._MARKET_LEVEL_FACTORS if c in df.columns]
        if not market_cols:
            return df

        settings = get_settings()
        features_dir = Path(settings.data.dir) / "features"
        if not features_dir.exists():
            return df

        # 收集各截面文件中市场级因子的值（每个文件对应一个 as_of 日期）
        hist: dict[str, list[float]] = {c: [] for c in market_cols}
        for fpath in sorted(features_dir.glob(f"{universe}_*.parquet")):
            try:
                snap_df = pd.read_parquet(fpath, columns=market_cols)
                if snap_df.empty:
                    continue
                for col in market_cols:
                    if col in snap_df.columns:
                        val = snap_df[col].dropna().iloc[0] if not snap_df[col].dropna().empty else float("nan")
                        hist[col].append(float(val))
            except Exception:  # noqa: BLE001
                pass

        df = df.copy()
        for col in market_cols:
            h = np.array(hist[col], dtype=float)
            if len(h) < 2:
                continue  # 不足两个历史点，跳过
            # 当前截面中取第一只股票的原始值（所有股票同值）
            raw_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else float("nan")
            z = timeseries_zscore(raw_val, h)
            df[col] = z  # 整列赋相同 z-score
            log.debug(
                f"[pipeline] market factor '{col}' timeseries z-score: "
                f"raw={raw_val:.3f} → z={z:.3f} (hist_n={len(h[~np.isnan(h)])})"
            )

        return df

    def run_single_from_snapshot(
        self, snap: dict, as_of: date, *, universe: str = "csi300"
    ) -> pd.DataFrame:
        """使用已加载 snapshot 跑全流程（raw → 横截面标准化 → 市场级时序标准化）."""
        raw = self.compute_raw_from_snapshot(snap, as_of)
        std = self.standardize(raw, snapshot=snap)
        if self.do_standardize:
            std = self._apply_market_timeseries_zscore(std, universe=universe)
        return std

    def run_single(self, as_of: date, *, universe: str = "csi300") -> pd.DataFrame:
        """全流程：raw → 横截面标准化 → 市场级时序标准化."""
        snap = load_snapshot(as_of)
        return self.run_single_from_snapshot(snap, as_of, universe=universe)

    # -------------------------------------------------------------------
    # 持久化
    # -------------------------------------------------------------------

    def save(self, df: pd.DataFrame, as_of: date, *, universe: str = "csi300") -> Path:
        settings = get_settings()
        out_dir = Path(settings.data.dir) / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{universe}_{as_of.isoformat()}.parquet"
        df.to_parquet(path)
        # 元数据
        meta_path = out_dir / f"{universe}_{as_of.isoformat()}.meta.json"
        names = list(df.columns)
        market_cols = sorted(set(names) & MARKET_LEVEL_FEATURES)
        cat_cols = sorted(set(names) & CATEGORICAL_EXPOSURE_FIELDS)
        stock_cs = sorted(
            c
            for c in names
            if c not in MARKET_LEVEL_FEATURES and c not in CATEGORICAL_EXPOSURE_FIELDS
        )
        meta = {
            "universe": universe,
            "as_of": as_of.isoformat(),
            "n_tickers": int(len(df)),
            "n_factors": int(df.shape[1]),
            "factor_names": names,
            "groups_enabled": self.groups,
            "standardized": self.do_standardize,
            "neutralized": self.do_neutralize,
            "winsorize_sigma": self.winsorize_sigma,
            "market_level_features": market_cols,
            "categorical_exposure_fields": cat_cols,
            "stock_cross_section_features": stock_cs,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        log.info(f"features saved to {path}  (meta: {meta_path})")
        return path

    def load(self, as_of: date, *, universe: str = "csi300") -> pd.DataFrame:
        settings = get_settings()
        path = Path(settings.data.dir) / "features" / f"{universe}_{as_of.isoformat()}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)


__all__ = [
    "ALL_FACTOR_GROUPS",
    "FeaturePipeline",
    "list_all_factor_names",
]
