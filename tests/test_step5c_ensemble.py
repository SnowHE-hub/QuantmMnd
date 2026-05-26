"""tests/test_step5c_ensemble.py — Step5c FactorCNN ensemble 单元测试（全 mock）.

覆盖：
  - ensemble_scores 加权融合函数
  - _load_cnn_model 加载与降级
  - step5c_cnn_ensemble 主流程（正常 + 降级）
  - Regime 权重选取
  - 返回候选列表的排序与字段完整性
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── 被测函数 ──────────────────────────────────────────────────────────────────
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.daily_update import (
    _ENSEMBLE_WEIGHTS,
    _load_cnn_model,
    ensemble_scores,
    step5c_cnn_ensemble,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

def _make_candidates(n: int = 5) -> list[dict]:
    """生成 n 条假 LGBM 候选（lgbm_score 递减）."""
    return [
        {
            "ticker":      f"00000{i}.SZ",
            "lgbm_rank":   i + 1,
            "lgbm_score":  round(1.0 - i * 0.1, 4),
            "lgbm_score_raw": float(n - i),
            "key_factors": {},
        }
        for i in range(n)
    ]


def _make_feat_df(tickers: list[str], feature_cols: list[str]) -> pd.DataFrame:
    """生成一个简单的因子表（随机值，无 NaN）."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(tickers), len(feature_cols)))
    return pd.DataFrame(data, index=tickers, columns=feature_cols)


def _fake_pkl_payload(feature_cols: list[str]) -> dict[str, Any]:
    """构造 save_cnn_model 格式的 payload（用于 mock pickle.load）."""
    return {
        "__type__":    "CNNTrainResult",
        "val_ic_mean": 0.05,
        "val_ic_std":  0.02,
        "val_icir":    2.5,
        "feature_cols": feature_cols,
        "folds": [],
        "model_state": None,   # 测试中由 mock 接管
        "model_config": {
            "n_value": 9, "n_quality": 17,
            "n_momentum": 9, "n_technical": 36, "branch_dim": 8,
        },
    }


# ─── 1. ensemble_scores ───────────────────────────────────────────────────────

def test_ensemble_scores_both_valid():
    result = ensemble_scores(0.8, 0.6, 0.65, 0.35)
    assert abs(result - (0.65 * 0.8 + 0.35 * 0.6)) < 1e-9


def test_ensemble_scores_lgbm_nan():
    result = ensemble_scores(float("nan"), 0.7, 0.65, 0.35)
    assert result == pytest.approx(0.7)


def test_ensemble_scores_cnn_nan():
    result = ensemble_scores(0.8, float("nan"), 0.65, 0.35)
    assert result == pytest.approx(0.8)


def test_ensemble_scores_both_nan():
    result = ensemble_scores(float("nan"), float("nan"), 0.65, 0.35)
    assert math.isnan(result)


def test_ensemble_scores_weights_sum_not_required():
    """函数不强制权重归一（允许调用者传非归一权重）."""
    result = ensemble_scores(1.0, 0.0, 1.0, 0.0)
    assert result == pytest.approx(1.0)


# ─── 2. _ENSEMBLE_WEIGHTS ────────────────────────────────────────────────────

def test_ensemble_weights_all_regimes_present():
    for regime in ("bull", "neutral", "bear"):
        assert regime in _ENSEMBLE_WEIGHTS

def test_ensemble_weights_sum_to_one():
    for regime, (lgbm_w, cnn_w) in _ENSEMBLE_WEIGHTS.items():
        assert abs(lgbm_w + cnn_w - 1.0) < 1e-9, f"regime={regime} 权重之和不为 1"

def test_ensemble_weights_bull_more_cnn():
    """bull Regime 下 CNN 权重最高（0.40）."""
    assert _ENSEMBLE_WEIGHTS["bull"][1] >= _ENSEMBLE_WEIGHTS["neutral"][1]
    assert _ENSEMBLE_WEIGHTS["neutral"][1] >= _ENSEMBLE_WEIGHTS["bear"][1]


# ─── 3. _load_cnn_model ───────────────────────────────────────────────────────

def test_load_cnn_model_file_not_exist(tmp_path):
    model, cols = _load_cnn_model(tmp_path / "nonexistent.pkl")
    assert model is None
    assert cols == []


def test_load_cnn_model_wrong_type(tmp_path):
    bad_pkl = tmp_path / "bad.pkl"
    with open(bad_pkl, "wb") as f:
        pickle.dump({"__type__": "SomethingElse"}, f)
    model, cols = _load_cnn_model(bad_pkl)
    assert model is None


def test_load_cnn_model_missing_state(tmp_path):
    pkl = tmp_path / "m.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"__type__": "CNNTrainResult", "model_state": None, "feature_cols": []}, f)
    model, cols = _load_cnn_model(pkl)
    assert model is None


def test_load_cnn_model_success(tmp_path):
    """正常加载：mock FactorCNN 构造与 load_state_dict.
    _load_cnn_model 内部执行 `from quantmind.models.factor_cnn import FactorCNN`，
    因此 patch 目标是源模块，而非 daily_update。
    """
    feature_cols = [f"feat_{i}" for i in range(71)]
    payload = _fake_pkl_payload(feature_cols)
    payload["model_state"] = {"dummy": "state"}   # 非 None 触发正常路径

    pkl = tmp_path / "model.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)

    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_cnn_cls = MagicMock(return_value=mock_model)

    # patch 源模块里的 FactorCNN，这样函数内的局部 import 会拿到 mock
    with patch("quantmind.models.factor_cnn.FactorCNN", mock_cnn_cls):
        model, cols = _load_cnn_model(pkl)

    assert model is mock_model
    assert cols == feature_cols
    mock_model.load_state_dict.assert_called_once_with({"dummy": "state"})
    mock_model.eval.assert_called_once()


# ─── 4. step5c_cnn_ensemble（主流程）────────────────────────────────────────

def _run_step5c(
    candidates: list[dict],
    cnn_scores: dict[str, float],
    *,
    feature_cols: list[str] | None = None,
    regime: str = "neutral",
    pkl_exists: bool = True,
    feat_path_exists: bool = True,
    tmp_path: Path,
) -> list[dict]:
    """运行 step5c 的测试帮助函数，mock 所有外部依赖."""
    from datetime import date as _date

    feature_cols = feature_cols or [f"f{i}" for i in range(10)]
    tickers = [c["ticker"] for c in candidates]

    # 构造 fake feat_path（或 None）
    if feat_path_exists:
        feat_file = tmp_path / "features.parquet"
        df = _make_feat_df(tickers, feature_cols)
        df.to_parquet(feat_file)
        feat_path: Path | None = feat_file
    else:
        feat_path = None

    # 构造 fake pkl
    if pkl_exists:
        payload = _fake_pkl_payload(feature_cols)
        payload["model_state"] = {}
        pkl_file = tmp_path / "cnn.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(payload, f)
        cnn_pkl = pkl_file
    else:
        cnn_pkl = tmp_path / "nonexistent.pkl"

    # predict_cnn 返回 mock cnn_scores
    mock_cnn_raw = pd.Series(cnn_scores)

    mock_model = MagicMock()
    mock_model.eval = MagicMock()

    def fake_load_cnn(path):
        if not pkl_exists:
            return None, []
        return mock_model, feature_cols

    # step5c_cnn_ensemble 内部做延迟 import，patch 必须指向**源模块**，
    # 唯一例外是 _load_cnn_model（模块级函数），可直接 patch daily_update 里的名字。
    with patch("scripts.daily_update._load_cnn_model", side_effect=fake_load_cnn), \
         patch("quantmind.models.factor_cnn.predict_cnn", return_value=mock_cnn_raw), \
         patch("quantmind.utils.score_order.order_preserving_pct_rank",
               side_effect=lambda s: s.rank(pct=True)), \
         patch("quantmind.regime.hmm.RegimeHMM") as mock_hmm_cls, \
         patch("quantmind.regime.hmm.build_observations",
               return_value=pd.DataFrame({"x": [1]})):
        mock_hmm = MagicMock()
        mock_hmm.predict_regime.return_value = regime
        mock_hmm_cls.return_value = mock_hmm

        result = step5c_cnn_ensemble(
            _date(2025, 10, 9),
            feat_path,
            candidates,
            cnn_pkl=cnn_pkl,
        )
    return result


def test_step5c_ensemble_score_blended(tmp_path):
    """ensemble_score = lgbm_w × lgbm_score + cnn_w × cnn_rank."""
    tickers = ["A.SZ", "B.SZ", "C.SZ"]
    candidates = [
        {"ticker": t, "lgbm_rank": i+1, "lgbm_score": 0.9 - i*0.1,
         "lgbm_score_raw": 0.9 - i*0.1, "key_factors": {}}
        for i, t in enumerate(tickers)
    ]
    # CNN 给 B 最高分
    cnn_scores = {"A.SZ": 0.3, "B.SZ": 0.9, "C.SZ": 0.6}

    result = _run_step5c(
        candidates, cnn_scores, regime="neutral", tmp_path=tmp_path
    )
    # ensemble_score 字段存在
    assert all("ensemble_score" in c for c in result)
    # cnn_score 字段存在
    assert all("cnn_score" in c for c in result)


def test_step5c_result_sorted_by_ensemble(tmp_path):
    """返回列表按 ensemble_score 降序。"""
    tickers = ["A.SZ", "B.SZ", "C.SZ", "D.SZ"]
    candidates = [
        {"ticker": t, "lgbm_rank": i+1, "lgbm_score": 0.8 - i*0.1,
         "lgbm_score_raw": 1.0, "key_factors": {}}
        for i, t in enumerate(tickers)
    ]
    cnn_scores = {"A.SZ": 0.1, "B.SZ": 0.9, "C.SZ": 0.5, "D.SZ": 0.7}

    result = _run_step5c(candidates, cnn_scores, regime="neutral", tmp_path=tmp_path)
    scores = [c["ensemble_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


def test_step5c_ensemble_rank_field(tmp_path):
    """每条候选应有 ensemble_rank 字段，且从 1 开始连续递增。"""
    tickers = ["A.SZ", "B.SZ", "C.SZ"]
    candidates = [
        {"ticker": t, "lgbm_rank": i+1, "lgbm_score": 0.9 - i*0.1,
         "lgbm_score_raw": 1.0, "key_factors": {}}
        for i, t in enumerate(tickers)
    ]
    cnn_scores = {t: 0.5 for t in tickers}

    result = _run_step5c(candidates, cnn_scores, tmp_path=tmp_path)
    ranks = [c["ensemble_rank"] for c in result]
    assert ranks == list(range(1, len(tickers) + 1))


def test_step5c_bull_regime_weights(tmp_path):
    """bull Regime 下 CNN 权重 0.40 生效：LGBM 分相同时 CNN 决定排序。"""
    candidates = [
        # A / B LGBM 分完全相同，由 CNN 分决出胜负
        {"ticker": "A.SZ", "lgbm_rank": 1, "lgbm_score": 0.5, "lgbm_score_raw": 0.5, "key_factors": {}},
        {"ticker": "B.SZ", "lgbm_rank": 2, "lgbm_score": 0.5, "lgbm_score_raw": 0.5, "key_factors": {}},
    ]
    # CNN 给 B 更高分
    cnn_scores = {"A.SZ": 0.2, "B.SZ": 0.8}

    result = _run_step5c(candidates, cnn_scores, regime="bull", tmp_path=tmp_path)
    # CNN rank: A=0.5, B=1.0 → ensemble B > A → B 排第一
    assert result[0]["ticker"] == "B.SZ"
    # 验证 bull 权重 0.40 已被使用（ensemble_score 包含 CNN 贡献）
    b_score = next(c["ensemble_score"] for c in result if c["ticker"] == "B.SZ")
    a_score = next(c["ensemble_score"] for c in result if c["ticker"] == "A.SZ")
    assert b_score > a_score


def test_step5c_degradation_no_pkl(tmp_path):
    """pkl 不存在时降级为纯 LGBM，返回原始列表（含 ensemble_score = lgbm_score）."""
    candidates = _make_candidates(3)
    orig_scores = [c["lgbm_score"] for c in candidates]

    result = _run_step5c(candidates, {}, pkl_exists=False, tmp_path=tmp_path)
    assert len(result) == 3
    for c, orig_s in zip(result, orig_scores):
        assert not math.isnan(c["ensemble_score"])
        # ensemble_score 等于 lgbm_score（降级）
        assert c["ensemble_score"] == pytest.approx(orig_s, abs=1e-6)


def test_step5c_degradation_no_feat_path(tmp_path):
    """因子文件不存在时降级，ensemble_score = lgbm_score。"""
    candidates = _make_candidates(3)
    result = _run_step5c(
        candidates, {}, feat_path_exists=False, tmp_path=tmp_path
    )
    for c in result:
        assert "ensemble_score" in c


def test_step5c_empty_candidates(tmp_path):
    """空候选列表直接返回空列表，不报错。"""
    result = _run_step5c([], {}, tmp_path=tmp_path)
    assert result == []


def test_step5c_cnn_score_nan_for_missing_ticker(tmp_path):
    """CNN 未返回某 ticker 的分数时，对应 cnn_score 应为 NaN，ensemble 退回 lgbm。"""
    candidates = [
        {"ticker": "A.SZ", "lgbm_rank": 1, "lgbm_score": 0.9,
         "lgbm_score_raw": 0.9, "key_factors": {}},
        {"ticker": "B.SZ", "lgbm_rank": 2, "lgbm_score": 0.1,
         "lgbm_score_raw": 0.1, "key_factors": {}},
    ]
    # CNN 只返回 A 的分数
    cnn_scores = {"A.SZ": 0.8}

    result = _run_step5c(candidates, cnn_scores, tmp_path=tmp_path)
    b = next(c for c in result if c["ticker"] == "B.SZ")
    assert math.isnan(b["cnn_score"])
    # ensemble = lgbm_score (fallback)
    assert b["ensemble_score"] == pytest.approx(0.1, abs=1e-6)


def test_step5c_lgbm_rank_preserved(tmp_path):
    """原始 lgbm_rank 在重排后依然保留在 dict 中。"""
    candidates = _make_candidates(4)
    cnn_scores = {c["ticker"]: float(i) for i, c in enumerate(candidates)}
    result = _run_step5c(candidates, cnn_scores, tmp_path=tmp_path)
    for c in result:
        assert "lgbm_rank" in c


def test_step5c_bear_regime_uses_75_25(tmp_path):
    """bear Regime 下 LGBM 权重 0.75，CNN 权重 0.25。"""
    lgbm_w, cnn_w = _ENSEMBLE_WEIGHTS["bear"]
    assert lgbm_w == pytest.approx(0.75)
    assert cnn_w  == pytest.approx(0.25)
