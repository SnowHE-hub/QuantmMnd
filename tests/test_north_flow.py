"""tests/test_north_flow.py — north_flow 因子单元测试。

覆盖范围：
  - build_north_flow_factor 输出格式（MultiIndex, 填0）
  - fetch_hk_hold_monthly 缓存机制（mock Tushare）
  - build_market_flow_factor 输出格式与滚动逻辑
  - fetch_market_north_flow 缓存与增量更新（mock Tushare）
  - compute_north_ic 接口一致性与边界情况
  - apply_north_flow_regime_correction 阈值与规则正确性
  - discretize_market_flow 三档离散化
  - get_latest_market_north_flow 容错（异常返回 0.0）
  - HMM predict_regime north_flow_5d 参数接入
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantmind.features.north_flow import (
    NORTH_FLOW_FACTORS,
    _FLOW_BEAR_THRESHOLD,
    _FLOW_BULL_THRESHOLD,
    apply_north_flow_regime_correction,
    build_market_flow_factor,
    build_north_flow_factor,
    compute_north_ic,
    discretize_market_flow,
    get_latest_market_north_flow,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_hk_hold_df(trade_date: str, n_stocks: int = 5) -> pd.DataFrame:
    """创建模拟 hk_hold 返回数据（A 股）."""
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n_stocks + 1)]
    return pd.DataFrame({
        "code":       range(1, n_stocks + 1),
        "trade_date": trade_date,
        "ts_code":    codes,
        "name":       [f"股票{i}" for i in range(1, n_stocks + 1)],
        "vol":        [1_000_000 * i for i in range(1, n_stocks + 1)],
        "ratio":      [1.0 + i * 0.5 for i in range(1, n_stocks + 1)],
        "exchange":   "SZ",
    })


def _make_moneyflow_df() -> pd.DataFrame:
    """创建模拟 moneyflow_hsgt 返回数据."""
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    north_money = [190000 + i * 1000 for i in range(8)]  # 递增模拟累计值
    return pd.DataFrame({
        "trade_date":  [d.strftime("%Y%m%d") for d in dates],
        "ggt_ss":      [28000.0] * 8,
        "ggt_sz":      [22000.0] * 8,
        "hgt":         [86000.0] * 8,
        "sgt":         [99000.0] * 8,
        "north_money": north_money,
        "south_money": [50000.0] * 8,
    })


def _make_returns_df(n: int = 50) -> pd.DataFrame:
    """创建模拟 stock_returns 数据."""
    rng = np.random.default_rng(42)
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    return pd.DataFrame({
        "ts_code":   codes,
        "return_3m": rng.normal(0.05, 0.15, n),
    })


# ── 1. NORTH_FLOW_FACTORS 常量 ────────────────────────────────────────────────

def test_north_flow_factors_list():
    """NORTH_FLOW_FACTORS 包含 north_net_buy_5d."""
    assert "north_net_buy_5d" in NORTH_FLOW_FACTORS


# ── 2. build_north_flow_factor 格式测试 ───────────────────────────────────────

def test_build_north_flow_factor_multiindex(tmp_path):
    """输出为 MultiIndex(ts_code, trade_date) Series，name='north_net_buy_5d'."""
    # 模拟两个月的 hk_hold 数据
    df_jan = _make_hk_hold_df("20250131", n_stocks=10)
    df_feb = _make_hk_hold_df("20250228", n_stocks=10)
    # 改变 Feb 的 ratio 使差值非零
    df_feb["ratio"] = df_feb["ratio"] + 0.3

    combined = pd.concat([df_jan, df_feb], ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])

    with patch("quantmind.features.north_flow.fetch_hk_hold_monthly", return_value=combined):
        factor = build_north_flow_factor(
            start_date="2025-01-01",
            end_date="2025-03-01",
            use_cache=False,
        )

    assert isinstance(factor, pd.Series)
    assert factor.name == "north_net_buy_5d"
    assert factor.index.names == ["ts_code", "trade_date"]


def test_build_north_flow_factor_values_correct(tmp_path):
    """因子值等于 ratio_curr - ratio_prev（月度差）."""
    df_jan = _make_hk_hold_df("20250131", n_stocks=3)
    df_feb = _make_hk_hold_df("20250228", n_stocks=3)
    df_jan["ratio"] = [2.0, 3.0, 4.0]
    df_feb["ratio"] = [2.5, 2.8, 4.6]   # expected diff: +0.5, -0.2, +0.6

    combined = pd.concat([df_jan, df_feb], ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])

    with patch("quantmind.features.north_flow.fetch_hk_hold_monthly", return_value=combined):
        factor = build_north_flow_factor(
            start_date="2025-02-01",
            end_date="2025-03-01",
            use_cache=False,
        )

    # Feb 行应有差值
    assert not factor.empty
    diffs = factor.values
    np.testing.assert_allclose(diffs, [0.5, -0.2, 0.6], atol=1e-6)


def test_build_north_flow_factor_empty_on_no_data():
    """hk_hold 返回空时，因子为空 Series."""
    empty = pd.DataFrame(columns=["ts_code", "trade_date", "vol", "ratio", "exchange"])
    with patch("quantmind.features.north_flow.fetch_hk_hold_monthly", return_value=empty):
        factor = build_north_flow_factor(use_cache=False)
    assert factor.empty
    assert factor.name == "north_net_buy_5d"


def test_build_north_flow_factor_missing_stocks_fill_zero(tmp_path):
    """未被北向持仓的股票不出现在因子中（由调用方填 0）."""
    df_jan = _make_hk_hold_df("20250131", n_stocks=5)
    df_feb = _make_hk_hold_df("20250228", n_stocks=3)  # 少 2 只
    combined = pd.concat([df_jan, df_feb], ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])

    with patch("quantmind.features.north_flow.fetch_hk_hold_monthly", return_value=combined):
        factor = build_north_flow_factor(
            start_date="2025-02-01",
            end_date="2025-03-01",
            use_cache=False,
        )

    # 只有 3 只有前后月数据的股票应出现
    assert factor.index.get_level_values("ts_code").nunique() == 3


# ── 3. fetch_hk_hold_monthly 缓存机制 ────────────────────────────────────────

def test_fetch_hk_hold_monthly_cache_hit(tmp_path):
    """有完整缓存时，不调用 Tushare API."""
    cached_df = _make_hk_hold_df("20250131")
    cached_df["trade_date"] = pd.to_datetime("2025-01-31")
    cached_path = tmp_path / "north_flow.parquet"
    cached_df.to_parquet(cached_path, index=False)

    with patch("quantmind.features.north_flow._FLOW_CACHE", cached_path), \
         patch("quantmind.features.north_flow._get_pro") as mock_pro:
        result = build_north_flow_factor(
            start_date="2025-01-01",
            end_date="2025-01-31",
            use_cache=True,
        )
        # _get_pro 不应被调用（缓存命中）
        mock_pro.assert_not_called()


def test_fetch_hk_hold_monthly_filters_hk_stocks():
    """过滤 HK 股（exchange=HK），只保留 SH/SZ A 股."""
    from quantmind.features.north_flow import _fetch_hk_hold_range

    mock_df = pd.DataFrame({
        "code":       [1, 2, 3],
        "trade_date": ["20250131"] * 3,
        "ts_code":    ["00001.HK", "000001.SZ", "600000.SH"],
        "name":       ["长和", "平安银行", "浦发银行"],
        "vol":        [1e8, 7e8, 5e8],
        "ratio":      [3.3, 3.6, 2.1],
        "exchange":   ["HK", "SZ", "SH"],
    })

    mock_pro = MagicMock()
    mock_pro.hk_hold.return_value = mock_df

    with patch("quantmind.features.north_flow._get_pro", return_value=mock_pro), \
         patch("time.sleep"):
        result = _fetch_hk_hold_range("20250101", "20250131")

    assert "00001.HK" not in result["ts_code"].values
    assert "000001.SZ" in result["ts_code"].values
    assert "600000.SH" in result["ts_code"].values
    assert len(result) == 2


# ── 4. build_market_flow_factor 格式与逻辑 ────────────────────────────────────

def test_build_market_flow_factor_series_format():
    """build_market_flow_factor 返回 Series，Index 为 DatetimeIndex."""
    mock_df = _make_moneyflow_df()
    mock_df_processed = pd.DataFrame({
        "trade_date":    pd.to_datetime(mock_df["trade_date"]),
        "north_money":   pd.to_numeric(mock_df["north_money"]),
        "daily_net_flow": [np.nan] + [1000.0] * 7,
    })

    with patch("quantmind.features.north_flow.fetch_market_north_flow",
               return_value=mock_df_processed):
        factor = build_market_flow_factor(
            start_date="2025-01-01", end_date="2025-01-31", use_cache=False
        )

    assert isinstance(factor, pd.Series)
    assert factor.name == "north_flow_market_5d"
    assert isinstance(factor.index, pd.DatetimeIndex)


def test_build_market_flow_factor_rolling_window():
    """5日滚动均值正确：全部 1000 M/day → mean = 1000."""
    mock_df = pd.DataFrame({
        "trade_date":    pd.date_range("2025-01-02", periods=10, freq="B"),
        "north_money":   [190000 + i * 1000 for i in range(10)],
        "daily_net_flow": [np.nan] + [1000.0] * 9,
    })

    with patch("quantmind.features.north_flow.fetch_market_north_flow",
               return_value=mock_df):
        factor = build_market_flow_factor(
            start_date="2025-01-01", end_date="2025-02-01",
            use_cache=False, window=5,
        )

    # 稳定后（第5天起）均值应为 1000
    stable = factor.iloc[4:]
    np.testing.assert_allclose(stable.values, 1000.0, atol=1e-6)


def test_build_market_flow_factor_empty_on_no_data():
    """moneyflow_hsgt 返回空时，因子为空 Series."""
    empty = pd.DataFrame(columns=["trade_date", "north_money", "daily_net_flow"])
    with patch("quantmind.features.north_flow.fetch_market_north_flow", return_value=empty):
        factor = build_market_flow_factor(use_cache=False)
    assert factor.empty


# ── 5. compute_north_ic 接口一致性 ────────────────────────────────────────────

def test_compute_north_ic_empty_factor():
    """空因子返回 ic=nan, valid=False."""
    result = compute_north_ic(pd.Series(dtype=float, name="north_net_buy_5d"))
    assert np.isnan(result["ic"])
    assert result["valid"] is False
    assert result["n"] == 0


def test_compute_north_ic_with_returns_df():
    """传入 returns DataFrame（不读文件）时正常计算 IC."""
    rng = np.random.default_rng(0)
    n = 60
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    dates = [pd.Timestamp("2025-01-31")] * n
    factor = pd.Series(
        rng.normal(0, 1, n),
        index=pd.MultiIndex.from_arrays([codes, dates], names=["ts_code", "trade_date"]),
        name="north_net_buy_5d",
    )
    returns = pd.DataFrame({
        "ts_code":   codes,
        "return_3m": rng.normal(0.05, 0.1, n),
    })

    result = compute_north_ic(factor, returns=returns)
    assert "ic" in result
    assert "p" in result
    assert "n" in result
    assert "valid" in result
    assert result["n"] == n
    assert -1.0 <= result["ic"] <= 1.0


def test_compute_north_ic_insufficient_samples():
    """样本量 < 30 时，n 有值但 valid=False."""
    rng = np.random.default_rng(1)
    n = 10
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    dates = [pd.Timestamp("2025-01-31")] * n
    factor = pd.Series(
        rng.normal(0, 1, n),
        index=pd.MultiIndex.from_arrays([codes, dates], names=["ts_code", "trade_date"]),
        name="north_net_buy_5d",
    )
    returns = pd.DataFrame({
        "ts_code":   codes[:5],   # 只有 5 只能 join
        "return_3m": rng.normal(0, 0.1, 5),
    })

    result = compute_north_ic(factor, returns=returns)
    assert result["valid"] is False


# ── 6. apply_north_flow_regime_correction 规则测试 ────────────────────────────

def test_correction_strong_inflow_neutral_to_bull():
    """强净流入（>5000M）且 regime=neutral → 升为 bull."""
    result = apply_north_flow_regime_correction("neutral", market_flow_5d=6000.0)
    assert result == "bull"


def test_correction_strong_inflow_bull_stays_bull():
    """强净流入且 regime=bull → 维持 bull（无重复升级）."""
    result = apply_north_flow_regime_correction("bull", market_flow_5d=8000.0)
    assert result == "bull"


def test_correction_strong_inflow_bear_no_change():
    """强净流入但 regime=bear → 维持 bear（不强制从 bear 升级）."""
    result = apply_north_flow_regime_correction("bear", market_flow_5d=6000.0)
    assert result == "bear"


def test_correction_strong_outflow_neutral_to_bear():
    """强净流出（< -5000M）且 regime=neutral → 降为 bear."""
    result = apply_north_flow_regime_correction("neutral", market_flow_5d=-6000.0)
    assert result == "bear"


def test_correction_strong_outflow_bear_stays_bear():
    """强净流出且 regime=bear → 维持 bear."""
    result = apply_north_flow_regime_correction("bear", market_flow_5d=-8000.0)
    assert result == "bear"


def test_correction_strong_outflow_bull_no_change():
    """强净流出但 regime=bull → 维持 bull（不强制从 bull 降级）."""
    result = apply_north_flow_regime_correction("bull", market_flow_5d=-6000.0)
    assert result == "bull"


def test_correction_mid_range_no_change():
    """中间区间（-5000 ~ 5000 M）不干预 HMM 判断."""
    for regime in ("bull", "neutral", "bear"):
        for flow in (0.0, 4999.9, -4999.9, 1000.0, -2000.0):
            result = apply_north_flow_regime_correction(regime, market_flow_5d=flow)
            assert result == regime, (
                f"flow={flow}, regime={regime} → 不应改变，但得到 {result}"
            )


# ── 7. discretize_market_flow 三档离散化 ──────────────────────────────────────

def test_discretize_high():
    """flow > 5000 M → 'High'."""
    assert discretize_market_flow(5001.0) == "High"
    assert discretize_market_flow(10000.0) == "High"


def test_discretize_low():
    """flow < -5000 M → 'Low'."""
    assert discretize_market_flow(-5001.0) == "Low"
    assert discretize_market_flow(-9999.0) == "Low"


def test_discretize_mid():
    """-5000 ≤ flow ≤ 5000 M → 'Mid'."""
    assert discretize_market_flow(0.0) == "Mid"
    assert discretize_market_flow(5000.0) == "Mid"
    assert discretize_market_flow(-5000.0) == "Mid"
    assert discretize_market_flow(4999.9) == "Mid"


# ── 8. get_latest_market_north_flow 容错 ──────────────────────────────────────

def test_get_latest_market_north_flow_exception_returns_zero():
    """build_market_flow_factor 抛异常时，返回 0.0（安全默认值）."""
    with patch(
        "quantmind.features.north_flow.build_market_flow_factor",
        side_effect=RuntimeError("网络超时"),
    ):
        result = get_latest_market_north_flow()
    assert result == 0.0


def test_get_latest_market_north_flow_empty_returns_zero():
    """因子为空时返回 0.0."""
    with patch(
        "quantmind.features.north_flow.build_market_flow_factor",
        return_value=pd.Series(dtype=float, name="north_flow_market_5d"),
    ):
        result = get_latest_market_north_flow()
    assert result == 0.0


def test_get_latest_market_north_flow_normal():
    """正常情况下返回最后一个值."""
    mock_series = pd.Series(
        [3000.0, 4500.0, 6200.0],
        index=pd.date_range("2025-01-01", periods=3),
        name="north_flow_market_5d",
    )
    with patch(
        "quantmind.features.north_flow.build_market_flow_factor",
        return_value=mock_series,
    ):
        result = get_latest_market_north_flow()
    assert result == pytest.approx(6200.0)


# ── 9. HMM predict_regime 接入 north_flow_5d 参数 ────────────────────────────

def test_hmm_predict_regime_north_flow_bull_correction():
    """RegimeHMM.predict_regime with north_flow_5d > 5000 升级 neutral → bull."""
    from quantmind.regime.hmm import RegimeHMM, _HMM

    # 构造一个状态序列全为 neutral 的 mock 模型
    hmm = _HMM()
    label_map = {0: "neutral", 1: "neutral", 2: "neutral"}
    dates = pd.DatetimeIndex([pd.Timestamp("2025-01-02")])
    states = np.array([0], dtype=np.int32)
    model = RegimeHMM(hmm, label_map, dates, states)

    regime = model.predict_regime(
        df=None,
        as_of=pd.Timestamp("2025-01-02"),
        north_flow_5d=6000.0,   # 强净流入 → bull
    )
    assert regime == "bull"


def test_hmm_predict_regime_no_north_flow_unchanged():
    """RegimeHMM.predict_regime without north_flow_5d 维持原 regime（向后兼容）."""
    from quantmind.regime.hmm import RegimeHMM, _HMM

    hmm = _HMM()
    label_map = {0: "neutral", 1: "neutral", 2: "neutral"}
    dates = pd.DatetimeIndex([pd.Timestamp("2025-01-02")])
    states = np.array([0], dtype=np.int32)
    model = RegimeHMM(hmm, label_map, dates, states)

    regime = model.predict_regime(
        df=None,
        as_of=pd.Timestamp("2025-01-02"),
        # north_flow_5d 省略 → None
    )
    assert regime == "neutral"   # 不受影响
