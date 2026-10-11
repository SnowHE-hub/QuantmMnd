"""T-022 第二轮修订：行业分类体系发布前，行业与行业特征必须缺失（PIT 反例，合成数据）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_P = Path(__file__).resolve().parents[1] / "docs/work/evidence/T-022/build_sector_features_pit.py"
_spec = importlib.util.spec_from_file_location("t022_build_pit", _P)
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

PUBLIC = pd.Timestamp("2021-12-13")


def members() -> pd.DataFrame:
    # 成员 in_date 很早（2015），但 2021 体系 2021-12-13 才发布
    return pd.DataFrame({"ts_code": ["A.SZ", "B.SZ", "C.SZ"], "l1_name": ["电子", "电子", "银行"],
                         "l2_name": ["半导体", "元件", "国有大型银行Ⅱ"], "l3_name": ["x", "y", "z"],
                         "in_date": ["20150101", "20150101", "20220301"], "out_date": [None, "20220601", None]})


def test_no_industry_before_taxonomy_publication():
    dates = pd.to_datetime(["2020-06-05", "2021-12-10", "2021-12-17", "2022-03-04", "2022-07-01"])
    idx = pd.MultiIndex.from_product([dates, ["A.SZ", "B.SZ", "C.SZ"]], names=["as_of", "ticker"])
    ind = bp.pit_industry(idx, members(), PUBLIC)
    before = ind[idx.get_level_values("as_of") < PUBLIC]
    assert before.isna().all().all()                                   # 成员早于发布，但发布前一律缺失
    assert ind.loc[(pd.Timestamp("2021-12-17"), "A.SZ"), "sw_l1"] == "电子"
    assert pd.isna(ind.loc[(pd.Timestamp("2021-12-17"), "C.SZ"), "sw_l1"])   # 还未调入
    assert ind.loc[(pd.Timestamp("2022-03-04"), "C.SZ"), "sw_l1"] == "银行"
    assert pd.isna(ind.loc[(pd.Timestamp("2022-07-01"), "B.SZ"), "sw_l1"])   # 已调出


def test_features_missing_before_publication():
    dates = pd.to_datetime(["2021-06-04", "2022-01-07"])
    idx = pd.MultiIndex.from_product([dates, ["A.SZ", "B.SZ", "C.SZ"]], names=["as_of", "ticker"])
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=len(idx)) for c in
                       ("momentum_1m", "momentum_3m", "momentum_6m", "roe_ttm", "pb", "pe_ttm", "operating_profit_yoy")},
                      index=idx)
    df["sw_l1"] = bp.pit_industry(idx, members(), PUBLIC)["sw_l1"]
    feat = bp.sector_features(df)
    assert feat.loc[pd.Timestamp("2021-06-04")].isna().all().all()
    assert feat.loc[(pd.Timestamp("2022-01-07"), "A.SZ")].notna().all()
