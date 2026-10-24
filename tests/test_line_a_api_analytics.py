"""前端审查后新增的聚合接口（api_analytics）的契约与合理性测试。用真实数据根；数据不在就跳过。"""
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path("/home/lenovo/projects/quantmind/data/line_a_v2")
pytestmark = pytest.mark.skipif(not (ROOT / "runs").exists(), reason="no data root")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from quantmind.line_a.api_server import app
    return TestClient(app)


@pytest.fixture(scope="module")
def run_dates():
    return sorted(p.name for p in (ROOT / "runs").iterdir() if p.is_dir() and p.name[:2] == "20")


def test_forward_entry_is_next_trading_day_and_numbers_are_sane(client, run_dates):
    d = run_dates[0]
    r = client.get(f"/api/forward/{d}?source=f3top&top=30&horizons=1,5").json()
    m = r["meta"]
    assert m["entry_date"] > d
    from quantmind.line_a.api_analytics import market
    dates = market()["dates"]
    assert pd.Timestamp(m["entry_date"]) == dates[dates > pd.Timestamp(d)][0]
    assert len(r["rows"]) == 30
    filled = [x for x in r["rows"] if x["status"] == "filled"]
    assert filled, "至少有一只能成交"
    for x in filled:
        assert math.isfinite(x["ret_1"]) and -0.35 < x["ret_1"] < 0.35            # 单日涨跌幅上限 30%（北交所）
        assert abs(x["excess_pool_1"] - (x["ret_1"] - r["benchmarks"]["1"]["pool_ew"])) < 1e-12
    assert int(m["benchmark_names"]["pool_ew"].split("（")[1].split(" ")[0]) > 1000


def test_forward_latest_date_has_no_future(client, run_dates):
    r = client.get(f"/api/forward/{run_dates[-1]}?source=f3top&top=5").json()
    assert all(x["ret_20"] is None for x in r["rows"])


def test_bad_inputs_rejected(client, run_dates):
    assert client.get(f"/api/forward/{run_dates[0]}?tickers=rm%20-rf").status_code == 400
    assert client.get(f"/api/claim/{run_dates[-1]}/abc/A1:c1").status_code == 400
    assert client.get(f"/api/forward/{run_dates[0]}?horizons=0").status_code == 400


def test_consensus_percentiles(client, run_dates):
    d = run_dates[-1]
    top = client.get(f"/api/scores/{d}?model=ens_20d&top=1").json()[0]
    one = client.get(f"/api/scores/{d}?model=ens_20d&ticker={top['ticker']}").json()
    assert one[0]["rank"] == 1
    c = client.get(f"/api/consensus/{d}?ticker={top['ticker']}").json()
    row = c["row"]
    for k in ("xgb_pct", "cat_pct", "tabm_pct", "seqfusion_pct", "ENS_pct"):
        assert 0 <= row[k] <= 100
    assert row["ENS_pct"] >= 99                     # 全市场第一，在研究池里也应在最前（研究池是全市场的子集）
    assert row["dispersion_pp"] >= 0


def test_funnel_matches_files(client, run_dates):
    d = run_dates[-1]
    f = client.get(f"/api/funnel/{d}").json()
    by = {x["layer"]: x for x in f["layers"]}
    from quantmind.line_a.api_analytics import day_outputs
    o = day_outputs(d)
    assert by["F1"]["n"] == len(pd.read_parquet(o["f1"]))
    assert by["F2"]["n"] == int((pd.read_parquet(o["f2"])["domain"] == "research_pool").sum())
    assert by["F1"]["n"] >= by["F2"]["n"]
    if "F7" in by:
        assert sum(v for k, v in by["F7"]["detail"].items() if k not in ("每案例平均", "已核实比例")) == by["F7"]["n"]


def test_claim_trace_is_real(client):
    days = sorted(p.name for p in (ROOT / "research/daily").iterdir() if (p / "f7_claims.parquet").exists())
    d = days[-1]
    cl = pd.read_parquet(ROOT / "research/daily" / d / "f7_claims.parquet")
    comp = cl[(cl["type"] == "computed") & cl["ref"].str.startswith("A1")].head(3)
    rows = comp if len(comp) else cl.head(3)
    for _, x in rows.iterrows():
        t = client.get(f"/api/claim/{d}/{x['ticker']}/{x['ref']}")
        assert t.status_code == 200, t.text
        j = t.json()
        assert j["claim"]["text"] == x["text"]                       # 和 F7 汇总表是同一条声明
        assert j["check_at_run"]["status"] == x["check"]
        if j["claim"]["claim_type"] == "computed" and j["recompute"] and "value" in j["recompute"]:
            assert math.isfinite(j["recompute"]["value"])


def test_redteam_v2_is_selective_and_has_base_rate(client, run_dates):
    r = client.get(f"/api/redteam/{run_dates[0]}?source=f4&horizon=5&version=v2").json()
    assert r["meta"]["flagged_share"] < 0.9
    g = r["by_group"]["excess_pool"]
    assert g["base_rate"] is not None and 0 <= g["base_rate"] <= 1


def test_execution_rules(client):
    j = client.get("/api/rules/execution").json()
    assert j["lot"]["STAR"] == {"min": 200, "step": 1}
    assert j["execution"]["min_commission"] == 5.0


def test_missing_step_is_404_not_500(client, run_dates):
    from quantmind.line_a.api_analytics import day_outputs
    for d in run_dates:
        if "bypass" not in day_outputs(d):
            assert client.get(f"/api/bypass/{d}").status_code == 404
            r = client.get(f"/api/redteam/{d}?source=f4").json()
            assert "bypass" in r["missing_inputs"]
            assert client.get(f"/api/funnel/{d}").status_code == 200
            break
