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

import pandas as pd

from quantmind.core.config import get_settings
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data import load_snapshot
from quantmind.features.fundamental import (
    FUNDAMENTAL_FACTORS,
    compute_all_fundamental_factors,
)
from quantmind.features.sentiment import (
    SENTIMENT_FACTORS,
    compute_all_sentiment_factors,
)
from quantmind.features.standardize import standardize
from quantmind.features.technical import (
    TECHNICAL_FACTORS,
    compute_all_technical_factors,
)

log = get_logger(__name__)


# 因子分类（用于元数据 + 选择性计算）
ALL_FACTOR_GROUPS = {
    "fundamental": FUNDAMENTAL_FACTORS,
    "technical": TECHNICAL_FACTORS,
    "sentiment": SENTIMENT_FACTORS,
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

    def compute_raw(self, as_of: date) -> pd.DataFrame:
        """计算所有因子的「原始值」（未标准化），返回 (ticker × factor).

        如果 snapshot 缺失，会抛 FileNotFoundError 给上层处理。
        """
        snap = load_snapshot(as_of)
        with operation_logger("features.compute_raw", as_of=str(as_of)):
            parts: list[pd.DataFrame] = []
            if "fundamental" in self.groups:
                parts.append(compute_all_fundamental_factors(snap, as_of))
            if "technical" in self.groups:
                parts.append(compute_all_technical_factors(snap, as_of))
            if "sentiment" in self.groups:
                parts.append(compute_all_sentiment_factors(snap, as_of))

            # 用 universe 的 ticker 作为 master index
            u_df = snap.get("universe")
            if u_df is None or u_df.empty:
                raise RuntimeError("snapshot.universe is empty")
            assert isinstance(u_df, pd.DataFrame)
            master_idx = pd.Index(u_df["ticker"], name="ticker")

            # outer join 所有因子，再 reindex 到 universe
            full = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=master_idx)
            full = full.reindex(master_idx)
            log.info(
                f"computed {full.shape[1]} factors for {full.shape[0]} tickers "
                f"@ {as_of}"
            )
            return full

    def standardize(
        self, df: pd.DataFrame, snapshot: dict | None = None
    ) -> pd.DataFrame:
        """对原始因子值做 winsorize + 行业/市值中性化 + zscore."""
        if not self.do_standardize:
            return df

        # 行业 + 市值（用于中性化）
        industry: pd.Series | None = None
        log_mv: pd.Series | None = None
        if snapshot is not None:
            db = snapshot.get("daily_basic")
            if (
                db is not None
                and isinstance(db, pd.DataFrame)
                and not db.empty
                and "total_mv" in db.columns
            ):
                import numpy as np

                log_mv = pd.Series(
                    np.log(db.set_index("ticker")["total_mv"].clip(lower=1e-9)),
                    name="log_mv",
                ).reindex(df.index)
            # 行业暂用 universe.weight 区间作 proxy；后续接入 stock_basic.industry
            # 现在先不传 industry，仅做市值中性化
        return standardize(
            df,
            industry=industry,
            log_market_cap=log_mv,
            winsorize_sigma=self.winsorize_sigma,
            do_neutralize=self.do_neutralize,
        )

    def run_single(self, as_of: date) -> pd.DataFrame:
        """全流程：raw → standardized."""
        snap = load_snapshot(as_of)
        raw = self.compute_raw(as_of)
        return self.standardize(raw, snapshot=snap)

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
        meta = {
            "universe": universe,
            "as_of": as_of.isoformat(),
            "n_tickers": int(len(df)),
            "n_factors": int(df.shape[1]),
            "factor_names": list(df.columns),
            "groups_enabled": self.groups,
            "standardized": self.do_standardize,
            "neutralized": self.do_neutralize,
            "winsorize_sigma": self.winsorize_sigma,
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
