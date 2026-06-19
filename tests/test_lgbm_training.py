"""LGBM 训练链路：IC 分析 / 训练 / 预测 / 报告 回归测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # 脚本依赖 _ROOT on sys.path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


def _make_panel(
    dates: list[pd.Timestamp],
    n_stocks: int,
    feat_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    tickers = [f"T{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["as_of", "ticker"])
    n = len(idx)
    data: dict[str, np.ndarray] = {}
    for j, c in enumerate(feat_cols):
        data[c] = rng.standard_normal(n).astype(np.float64)
    latent = sum(data[c] for c in feat_cols[:3]) / max(1, min(3, len(feat_cols)))
    data["forward_return_21d"] = latent * 0.1 + 0.02 * rng.standard_normal(n)
    return pd.DataFrame(data, index=idx)


def test_analyze_factor_ic_mock_panel_ic_range() -> None:
    af = _load_script_module("analyze_factor_ic")
    rng = np.random.default_rng(0)
    dates = [pd.Timestamp("2019-03-31"), pd.Timestamp("2019-06-30"), pd.Timestamp("2019-09-30")]
    panel = _make_panel(dates, 20, [f"f{k}" for k in range(8)], rng)
    df, ic_matrix = af.run_ic_analysis(panel, "forward_return_21d", min_cross_section=10)
    assert len(df) > 0
    for _, row in df.iterrows():
        f = row["factor"]
        for d, v in ic_matrix[str(f)].items():
            if v == v:
                assert -1.0 <= v <= 1.0, (f, d, v)


@pytest.mark.integration
@pytest.mark.stale_panel_fixture
def test_train_lgbm_base_produces_model_and_metrics(tmp_path: Path) -> None:
    tr = _load_script_module("train_lgbm_model")
    rng = np.random.default_rng(1)
    feat_names = [f"f{i}" for i in range(6)]
    meta = {
        "feature_columns": [*feat_names[:3], "list_age_years", *feat_names[3:]],
        "label_column": "forward_return_21d",
    }
    meta_path = tmp_path / "split_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    d_tr = pd.date_range("2018-03-31", periods=5, freq="QE")
    d_va = pd.date_range("2019-06-30", periods=3, freq="QE")
    d_te = pd.date_range("2020-03-31", periods=2, freq="QE")

    train_p = _make_panel(list(d_tr), 24, feat_names, rng)
    val_p = _make_panel(list(d_va), 24, feat_names, rng)
    test_p = _make_panel(list(d_te), 24, feat_names, rng)
    train_pp = tmp_path / "train.parquet"
    val_pp = tmp_path / "val.parquet"
    test_pp = tmp_path / "test.parquet"
    train_p.to_parquet(train_pp)
    val_p.to_parquet(val_pp)
    test_p.to_parquet(test_pp)

    model_path = tmp_path / "model.pkl"
    out_dir = tmp_path / "out"
    argv = [
        "--train",
        str(train_pp),
        "--val",
        str(val_pp),
        "--test",
        str(test_pp),
        "--label",
        "forward_return_21d",
        "--feature-set",
        "base",
        "--split-meta",
        str(meta_path),
        "--model-output",
        str(model_path),
        "--output-dir",
        str(out_dir),
        "--n-estimators",
        "80",
        "--early-stopping-rounds",
        "15",
        "--num-leaves",
        "15",
        "--run-tag",
        "pytest_base",
    ]
    tr.run_training(tr.parse_args(argv))
    assert model_path.is_file()
    from quantmind.models.factor_model import FactorModel

    m = FactorModel.load(model_path)
    assert m.predict(np.zeros((1, 3), dtype=np.float32)).shape == (1,)

    fi = out_dir / "feature_importance_pytest_base.csv"
    assert fi.is_file()
    fdf = pd.read_csv(fi)
    assert len(fdf) == 3
    assert set(fdf["feature"]) == set(feat_names[:3])
    assert "gain" in fdf.columns and fdf["gain"].sum() >= 0

    mj = json.loads((out_dir / "metrics_pytest_base.json").read_text(encoding="utf-8"))
    assert mj["test_IC_mean"] == mj["test_IC_mean"]  # not NaN
    assert mj["test_ICIR"] == mj["test_ICIR"]


@pytest.mark.integration
@pytest.mark.stale_panel_fixture
def test_predict_rankings_csv_columns(tmp_path: Path) -> None:
    tr = _load_script_module("train_lgbm_model")
    pr = _load_script_module("predict_rankings")
    rng = np.random.default_rng(2)
    feat_names = [f"f{i}" for i in range(6)]
    meta = {
        "feature_columns": [*feat_names[:3], "list_age_years", *feat_names[3:]],
        "label_column": "forward_return_21d",
    }
    meta_path = tmp_path / "split_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    d_tr = pd.date_range("2018-03-31", periods=5, freq="QE")
    d_va = pd.date_range("2019-06-30", periods=3, freq="QE")
    d_te = pd.date_range("2020-03-31", periods=2, freq="QE")
    train_p = _make_panel(list(d_tr), 24, feat_names, rng)
    val_p = _make_panel(list(d_va), 24, feat_names, rng)
    test_p = _make_panel(list(d_te), 24, feat_names, rng)
    train_pp = tmp_path / "train.parquet"
    val_pp = tmp_path / "val.parquet"
    test_pp = tmp_path / "test.parquet"
    train_p.to_parquet(train_pp)
    val_p.to_parquet(val_pp)
    test_p.to_parquet(test_pp)

    model_path = tmp_path / "model.pkl"
    out_dir = tmp_path / "out"
    tr.run_training(
        tr.parse_args(
            [
                "--train",
                str(train_pp),
                "--val",
                str(val_pp),
                "--test",
                str(test_pp),
                "--feature-set",
                "base",
                "--split-meta",
                str(meta_path),
                "--model-output",
                str(model_path),
                "--output-dir",
                str(out_dir),
                "--n-estimators",
                "60",
                "--early-stopping-rounds",
                "10",
                "--run-tag",
                "pytest_predict",
            ]
        )
    )

    # 预测集：两期，取最新
    d_pred = [pd.Timestamp("2021-03-31"), pd.Timestamp("2021-06-30")]
    pred_p = _make_panel(d_pred, 18, feat_names, rng)
    pred_p["exposure_industry"] = np.array(["A"] * len(pred_p))
    pred_pp = tmp_path / "predict.parquet"
    pred_p.to_parquet(pred_pp)

    out_csv = tmp_path / "rank.csv"
    args = mock.Mock(
        model=model_path,
        predict=pred_pp,
        output=out_csv,
        top_n=10,
    )
    with mock.patch.object(pr, "parse_args", return_value=args):
        pr.main()

    got = pd.read_csv(out_csv)
    assert {"ticker", "score", "rank"}.issubset(got.columns)
    assert len(got) == 18
    assert got["rank"].tolist() == list(range(1, 19))


@pytest.mark.skipif(
    not (ROOT / "reports/model_training/metrics_lgbm_v1_all_features.json").is_file(),
    reason="full training metrics absent",
)
def test_feature_importance_real_run_matches_feature_names() -> None:
    fi = ROOT / "reports/model_training/feature_importance_lgbm_v1_all_features.csv"
    mj_path = ROOT / "reports/model_training/metrics_lgbm_v1_all_features.json"
    fdf = pd.read_csv(fi)
    mj = json.loads(mj_path.read_text(encoding="utf-8"))
    assert len(fdf) == mj["n_features"]
    assert list(fdf["feature"]) != []


@pytest.mark.skipif(
    not (ROOT / "data/panel/train.parquet").is_file(),
    reason="panel data absent",
)
def test_factor_ic_csv_row_count_63() -> None:
    path = ROOT / "reports/model_training/factor_ic_analysis.csv"
    if not path.is_file():
        pytest.skip("factor_ic_analysis.csv not generated yet")
    df = pd.read_csv(path)
    assert len(df) == 63
