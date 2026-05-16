"""Momentum LSTM / MomentumAgent 行为测试（不训练全量模型）."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import json

import numpy as np
import pandas as pd
import torch

from quantmind.agents.investment_agents.agent_registry import AgentModelRecord
from quantmind.agents.investment_agents.momentum_agent import MomentumAgent
from quantmind.models.momentum_lstm import (
    MomentumLSTM,
    build_feature_matrix_for_ticker,
    build_lstm_arrays,
)


def _write_minimal_price_ohlcv(tmp: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-02", periods=220, freq="B")
    tickers = ["AAA.SZ", "BBB.SZ"]
    px = pd.DataFrame(index=dates)
    for t in tickers:
        r = rng.normal(0, 0.01, len(dates))
        px[t] = 100 * np.cumprod(1 + r)

    adj_path = tmp / "adj.parquet"
    px.to_parquet(adj_path)

    rows = []
    prev_close = {}
    for dt in dates:
        for t in tickers:
            c = float(px.loc[dt, t])
            o = c * (1 + rng.normal(0, 0.002))
            h = c * 1.01
            lo = c * 0.99
            v = float(rng.uniform(1e5, 2e5))
            pc = prev_close.get(t, c)
            rows.append({
                "trade_date": dt,
                "ts_code": t,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": v,
                "pre_close": pc,
            })
            prev_close[t] = c
    ohl_path = tmp / "ohlc.parquet"
    pd.DataFrame(rows).to_parquet(ohl_path)
    return adj_path, ohl_path


def test_build_lstm_arrays_shapes(tmp_path):
    adj, ohl = _write_minimal_price_ohlcv(tmp_path)
    px = pd.read_parquet(adj)
    px.index = pd.to_datetime(px.index)
    X_tr, y_tr, X_va, y_va, _, _ = build_lstm_arrays(
        px,
        ohl,
        label_horizon=3,
        seq_len=20,
        train_end="2020-08-31",
        val_end="2020-11-30",
        tickers=["AAA.SZ"],
    )
    assert X_tr.ndim == 3 and X_tr.shape[1:] == (20, 5)
    assert y_tr.shape[0] == X_tr.shape[0]


def test_momentum_lstm_forward_shape():
    m = MomentumLSTM()
    x = torch.randn(8, 60, 5)
    y = m(x)
    assert tuple(y.shape) == (8, 1)
    assert torch.all(y >= 0) and torch.all(y <= 1)


def test_build_feature_matrix_shape():
    rng = np.random.default_rng(0)
    cal = pd.date_range("2020-01-02", periods=40, freq="B")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(cal))), index=cal)
    ohl = pd.DataFrame({
        "open": close.values,
        "high": close.values * 1.01,
        "low": close.values * 0.99,
        "close": close.values,
        "volume": np.full(len(cal), 1e6),
        "pre_close": close.shift(1).fillna(close.iloc[0]).values,
    }, index=cal)
    mat = build_feature_matrix_for_ticker(close, ohl, cal=cal)
    assert mat is not None
    assert mat.shape == (len(cal), 5)


def test_momentum_agent_lstm_degrades_when_bundle_invalid():
    rec = AgentModelRecord(
        agent_name="MomentumAgent",
        model_version="lstm_v3",
        model_type="dl",
        model_path=str(Path("/nonexistent/momentum_lstm_v3.pt")),
        created_at="2026-01-01",
        performance={"val_acc": 0.55},
        is_active=True,
        upgrade_notes="test",
    )
    mock_reg = MagicMock()
    mock_reg.get_active.return_value = rec

    prices = pd.Series(np.linspace(100.0, 120.0, 130))

    agent = object.__new__(MomentumAgent)
    agent.ticker = "000001.SZ"
    agent.as_of = None
    agent.context = {}
    agent._model_record = rec
    agent._ml_model = {"invalid": True}

    with patch.object(MomentumAgent, "_get_registry", return_value=mock_reg):
        with patch.object(MomentumAgent, "_load_price_series", return_value=prices):
            sig = MomentumAgent.analyze(agent)

    assert sig.agent_name == "MomentumAgent"
    assert sig.evidence.get("model_version") == "rules_v1"


def test_registry_set_active_switch(tmp_path):
    """临时 registry 文件：set_active 切换版本."""
    from quantmind.agents.investment_agents.agent_registry import AgentModelRegistry

    reg_path = tmp_path / "registry.json"
    data = {
        "MomentumAgent": [
            {
                "agent_name": "MomentumAgent",
                "model_version": "rules_v1",
                "model_type": "rules",
                "model_path": None,
                "created_at": "2024-01-01",
                "performance": {},
                "is_active": False,
                "upgrade_notes": "",
            },
            {
                "agent_name": "MomentumAgent",
                "model_version": "lstm_v3",
                "model_type": "dl",
                "model_path": "/tmp/x.pt",
                "created_at": "2026-01-02",
                "performance": {"val_acc": 0.55},
                "is_active": True,
                "upgrade_notes": "",
            },
        ]
    }
    reg_path.write_text(json.dumps(data), encoding="utf-8")
    reg = AgentModelRegistry(reg_path)
    assert reg.get_active("MomentumAgent").model_version == "lstm_v3"
    assert reg.set_active("MomentumAgent", "rules_v1")
    assert reg.get_active("MomentumAgent").model_version == "rules_v1"
