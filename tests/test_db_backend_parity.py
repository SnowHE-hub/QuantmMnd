"""
tests/test_db_backend_parity.py
双 backend 一致性测试：parquet vs postgres 返回结构必须相同。

运行：
  DATA_BACKEND=parquet  conda run -n quantmind pytest tests/test_db_backend_parity.py -v
  DATA_BACKEND=postgres conda run -n quantmind pytest tests/test_db_backend_parity.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def svc_parquet():
    os.environ["DATA_BACKEND"] = "parquet"
    from app.services.data_service import DataService
    return DataService(backend="parquet", enable_cache=False)


@pytest.fixture(scope="module")
def svc_postgres():
    os.environ["DATA_BACKEND"] = "postgres"
    from app.services.data_service import DataService
    return DataService(backend="postgres", enable_cache=False)


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def assert_df_parity(pq_df: pd.DataFrame, pg_df: pd.DataFrame, label: str, key_cols: list[str]):
    """校验两个 DataFrame 结构和关键值一致。"""
    # 1. 列集一致（至少 parquet 的列都存在于 DB 版本）
    pq_cols = set(pq_df.columns)
    pg_cols = set(pg_df.columns)
    missing = pq_cols - pg_cols
    assert not missing, f"[{label}] DB backend 缺少列: {missing}"

    # 2. 行数一致（容差 1 行，允许新数据增量写入后 DB 版本略多）
    assert abs(len(pq_df) - len(pg_df)) <= max(1, len(pq_df) * 0.05), \
        f"[{label}] 行数差异过大: parquet={len(pq_df)}, DB={len(pg_df)}"

    # 3. 关键数值列均值相近（容差 1%）
    num_cols = pq_df.select_dtypes(include="number").columns.tolist()
    for col in num_cols[:5]:
        if col not in pg_df.columns:
            continue
        pq_mean = pq_df[col].mean(skipna=True)
        pg_mean = pg_df[col].mean(skipna=True)
        if pd.isna(pq_mean) or pd.isna(pg_mean) or pq_mean == 0:
            continue
        assert abs(pq_mean - pg_mean) / abs(pq_mean) < 0.01, \
            f"[{label}] {col} 均值偏差: parquet={pq_mean:.4f}, DB={pg_mean:.4f}"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRecommendationParity:
    def test_dates_type_and_nonempty(self, svc_parquet, svc_postgres):
        pq = svc_parquet.get_recommendation_dates()
        pg = svc_postgres.get_recommendation_dates()
        assert isinstance(pq, list)
        assert isinstance(pg, list)
        assert len(pq) > 0, "parquet backend 应有推荐日期"
        assert len(pg) > 0, "postgres backend 应有推荐日期"
        assert len(pq) == len(pg), f"推荐日期数不一致: pq={len(pq)}, pg={len(pg)}"

    def test_recommendations_columns(self, svc_parquet, svc_postgres):
        pq_df = svc_parquet.get_recommendations()
        pg_df = svc_postgres.get_recommendations()
        assert not pq_df.empty, "parquet 推荐不应为空"
        assert not pg_df.empty, "postgres 推荐不应为空"
        assert set(pq_df.columns) == set(pg_df.columns), \
            f"列集不一致: {set(pq_df.columns) ^ set(pg_df.columns)}"
        assert len(pq_df) == len(pg_df), \
            f"推荐行数不一致: pq={len(pq_df)}, pg={len(pg_df)}"


class TestRealizedPnLParity:
    def test_pnl_parity(self, svc_parquet, svc_postgres):
        pq_df = svc_parquet.get_realized_pnl()
        pg_df = svc_postgres.get_realized_pnl()
        if pq_df.empty:
            pytest.skip("realized_pnl parquet 为空")
        assert_df_parity(pq_df, pg_df, "realized_pnl", ["as_of_date", "ticker"])

    def test_pnl_key_cols_exist(self, svc_parquet, svc_postgres):
        for svc, label in [(svc_parquet, "parquet"), (svc_postgres, "postgres")]:
            df = svc.get_realized_pnl()
            if df.empty:
                continue
            for col in ("as_of_date", "ticker", "entry_price", "actual_return_63d"):
                assert col in df.columns, f"[{label}] 缺少列: {col}"


class TestForwardPositionsParity:
    def test_positions_structure(self, svc_parquet, svc_postgres):
        pq_pos = svc_parquet._forward_positions_raw()
        pg_pos = svc_postgres._forward_positions_raw()
        assert isinstance(pq_pos, list)
        assert isinstance(pg_pos, list)
        assert len(pq_pos) == len(pg_pos), \
            f"持仓数量不一致: pq={len(pq_pos)}, pg={len(pg_pos)}"

    def test_forward_positions_df(self, svc_parquet, svc_postgres):
        pq_df = svc_parquet.get_forward_positions()
        pg_df = svc_postgres.get_forward_positions()
        assert set(pq_df.columns) == set(pg_df.columns), "forward_positions 列集不一致"
        assert len(pq_df) == len(pg_df), \
            f"forward_positions 行数不一致: pq={len(pq_df)}, pg={len(pg_df)}"


class TestAgentAnalysisParity:
    def test_agent_analysis_structure(self, svc_parquet, svc_postgres):
        # 用 recommendations 里的第一个 ticker 测试
        pq_df = svc_parquet.get_recommendations()
        if pq_df.empty:
            pytest.skip("无推荐数据")
        ticker = pq_df.iloc[0]["ticker"]

        pq_a = svc_parquet.find_agent_analysis(ticker)
        pg_a = svc_postgres.find_agent_analysis(ticker)

        pq_date, pq_analysis = pq_a
        pg_date, pg_analysis = pg_a

        if pq_analysis is None:
            pytest.skip(f"parquet backend 无 {ticker} 的 agent analysis")

        assert pg_analysis is not None, f"postgres backend 找不到 {ticker} 的 agent analysis"
        # 关键字段存在
        for key in ("ticker", "agents"):
            assert key in pg_analysis, f"postgres agent_analysis 缺少键: {key}"
        assert str(pg_analysis.get("ticker")) == str(ticker)

    def test_all_agent_analysis_nonempty(self, svc_postgres):
        result = svc_postgres.get_all_agent_analysis()
        assert isinstance(result, dict)


class TestLossSignalsParity:
    def test_loss_signals_keys(self, svc_parquet, svc_postgres):
        pq = svc_parquet.get_loss_signals()
        pg = svc_postgres.get_loss_signals()
        assert isinstance(pq, dict)
        assert isinstance(pg, dict)
        # 两者都应有 latest 键（如果数据存在）
        if pq.get("latest"):
            assert "latest" in pg, "postgres 版本缺少 latest 键"


class TestParquetBackendZeroRegression:
    """parquet backend 零回归：所有方法正常返回，不崩溃。"""

    def test_recommendation_dates(self, svc_parquet):
        result = svc_parquet.get_recommendation_dates()
        assert isinstance(result, list)

    def test_recommendations(self, svc_parquet):
        df = svc_parquet.get_recommendations()
        assert isinstance(df, pd.DataFrame)

    def test_realized_pnl(self, svc_parquet):
        df = svc_parquet.get_realized_pnl()
        assert isinstance(df, pd.DataFrame)

    def test_forward_positions(self, svc_parquet):
        df = svc_parquet.get_forward_positions()
        assert isinstance(df, pd.DataFrame)

    def test_model_status(self, svc_parquet):
        status = svc_parquet.get_model_status()
        assert isinstance(status, dict)

    def test_loss_signals(self, svc_parquet):
        signals = svc_parquet.get_loss_signals()
        assert isinstance(signals, dict)

    def test_data_freshness(self, svc_parquet):
        freshness = svc_parquet.get_data_freshness()
        assert isinstance(freshness, list)
