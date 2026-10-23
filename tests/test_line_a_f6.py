"""T-040：旁路 v2（巨潮公告标题）、F6 选择与按角色缺口调用、运行（续跑、预算、沿用）、F7 表。不联网、不调用模型。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantmind.line_a import f6_research as f6
from quantmind.line_a.event_monitor import cninfo_ts_code, cninfo_window, scan_cninfo


def _lake(tmp: Path) -> Path:
    lake = tmp / "data/lake"
    (lake / "text/cninfo/_parts").mkdir(parents=True)
    (lake / "events").mkdir()
    cal = pd.date_range("2026-10-01", "2026-10-31", freq="D")
    pd.DataFrame({"cal_date": cal.strftime("%Y%m%d"), "is_open": (cal.weekday < 5).astype(int)}).to_parquet(
        lake / "trade_cal_sse.parquet")
    pd.DataFrame({"ts_code": ["600001.SH", "000002.SZ", "300003.SZ", "920004.BJ"]}).to_parquet(
        lake / "stock_basic_full_merged_20261020.parquet")
    return lake


def _ann(lake: Path, day: str, rows: list[tuple[str, str]]) -> None:
    pd.DataFrame({"announcementId": [f"{day}{i}" for i in range(len(rows))], "secCode": [c for c, _ in rows],
                  "announcementTitle": [t for _, t in rows]}).to_parquet(lake / f"text/cninfo/_parts/{day}.parquet")


def test_cninfo_mapping_window_and_scan(tmp_path):
    lake = _lake(tmp_path)
    assert [cninfo_ts_code(c) for c in ("600001", "2", "300003", "920004", "430001", "200002", "113001")] == \
        ["600001.SH", "000002.SZ", "300003.SZ", "920004.BJ", "430001.BJ", None, None]
    assert cninfo_window(lake, "2026-10-19") == ["20261017", "20261018", "20261019"]      # 周一：周末归到周一
    _ann(lake, "20261017", [("600001", "关于收到中国证监会立案告知书的公告")])
    _ann(lake, "20261018", [("000002", "关于回购股份的进展公告"), ("200002", "关于收到问询函的公告")])
    _ann(lake, "20261019", [("300003", "关于涉及诉讼的公告"), ("920004", "2026年第三季度报告摘要"),
                            ("000002", "关于控股股东变更的提示性公告")])
    _ann(lake, "20261016", [("600001", "关于重大资产重组的公告")])                           # 窗口外
    ev = {(e.ticker, e.event_type, e.importance) for e in scan_cninfo(lake, "2026-10-19")}
    assert ev == {("600001.SH", "investigation", "major"), ("300003.SZ", "litigation", "watch"),
                  ("000002.SZ", "control_change", "major")}                              # 回购是 routine，B 股不在证券表


def _f4() -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append({"horizon": "20d", "ticker": f"{600100 + i}.SH", "domain": "research_pool",
                     "reasons": "top150_ens20", "ens20": 30 - i, "ridge20": 30 - i, "ridge63": None})
        rows.append({"horizon": "63d", "ticker": f"{600500 + i}.SH", "domain": "research_pool",
                     "reasons": "top150_ridge63", "ens20": None, "ridge20": None, "ridge63": 30 - i})
    for i in range(5):
        rows.append({"horizon": "20d", "ticker": f"{600900 + i}.SH", "domain": "research_pool",
                     "reasons": "stratified_sample", "ens20": None, "ridge20": None, "ridge63": None})
    return pd.DataFrame(rows)


def test_select_order_caps_fresh_and_plan(tmp_path):
    lake = _lake(tmp_path)
    days = f6.open_days(lake)
    f4 = _f4()
    bypass = pd.DataFrame({"ticker": ["600101.SH", "600777.SH", "600102.SH"], "importance": ["major"] * 3,
                           "n_events": [3, 2, 1]})
    hist = pd.DataFrame({"ticker": ["600100.SH", "600102.SH"], "role": ["A1", "A2"], "as_of": ["20261019", "20261020"],
                         "path": ["x", "y"]})
    cfg = f6.CFG | {"max_cases": 10, "n_sample": 2}
    sel = f6.select("2026-10-22", f4, bypass, {"600777.SH"}, hist, days, cfg).set_index("ticker")
    assert sel.loc["600777.SH", "category"] == "holding_event"          # 持仓里的重大事件优先（即使不在 F4）
    assert sel.loc["600101.SH", "category"] == "event" and sel.loc["600102.SH", "category"] == "event"
    assert bool(sel.loc["600102.SH", "fresh"]) and "600100.SH" not in sel.index       # 新鲜的排名候选跳过
    tops = [t for t, c in zip(sel.index, sel["category"], strict=True) if c.startswith("top")]
    assert tops[:3] == ["600103.SH", "600104.SH", "600500.SH"]          # 20 日两只、63 日一只交替
    assert (sel["category"] == "sample").sum() == 2 and len(sel) == 10
    plan = f6.plan_roles(sel.reset_index(), {"600102.SH": {"A2"}}).set_index("ticker")
    assert plan.loc["600102.SH", "roles_run"] == "A2" and plan.loc["600103.SH", "roles_run"] == "A1,A2,A3,A4"


def test_triggers_from_new_data(tmp_path):
    lake = _lake(tmp_path)
    days = f6.open_days(lake)
    pd.DataFrame({"ts_code": ["600001.SH", "000002.SZ"], "ann_date": ["20261021", "20261010"],
                  "end_date": ["20260930"] * 2}).to_parquet(lake / "income.parquet")
    _ann(lake, "20261021", [("000002", "关于股东减持股份的预披露公告")])
    last = {"600001.SH": {"A1": "20261020", "A2": "20261020", "A3": "20261020", "A4": "20261020"},
            "000002.SZ": {"A1": "20261020", "A2": "20261020", "A3": "20261020"}}
    need = f6.triggers(lake, "2026-10-22", last, days)
    assert need["600001.SH"] == {"A1"}                                  # 新财报；A4 不到 5 个交易日、没有新预测
    assert need["000002.SZ"] == {"A2", "A3", "A4"}                      # 减持公告 → A2、A3；以前没跑过 A4


def _fake_runner(calls: list, cost: float = 0.4):
    def run(llm, lake, as_of, ticker, roles, out_root, reuse=None):
        calls.append((ticker, tuple(roles), tuple(sorted(reuse or {}))))
        d = Path(out_root) / as_of / ticker
        d.mkdir(parents=True, exist_ok=True)
        ups = {}
        for r in roles:
            p = d / f"{r}_x.json"
            p.write_text(json.dumps({"ticker": ticker, "as_of": as_of, "role": r, "status": "ok", "cost_rmb": 0.1,
                                     "output": {"claims": [{"local_id": "c1", "claim_type": "reported_fact",
                                                            "text": "营业收入 1 亿元", "value": 1e8, "unit": "元"}],
                                                "risk_flags": ["质押比例较高"] if r == "A3" else []},
                                     "checks": [{"local_id": "c1", "status": "verified"}], "evidence": {}}),
                         encoding="utf-8")
            ups[r] = {"status": "ok", "path": str(p), "as_of": as_of, "check_status": {"verified": 1}}
        for r, v in (reuse or {}).items():
            ups[r] = {"status": "ok", "path": v["path"], "as_of": v["as_of"], "check_status": {}}
        a6 = d / "A6_x.json"
        a6.write_text(json.dumps({"output": {"judgments": [{"claim_ref": f"{roles[0]}:c1", "verdict": "supported"}]}}),
                      encoding="utf-8")
        case = {"ticker": ticker, "as_of": as_of, "roles": roles, "cost_rmb": cost, "upstream": ups,
                "a6": {"status": "ok", "path": str(a6), "verdicts": {"supported": 1}}}
        (d / "case_x.json").write_text(json.dumps(case), encoding="utf-8")
        case["path"] = str(d / "case_x.json")
        return case
    return run


def test_run_research_budget_resume_reuse_and_f7(tmp_path):
    lake = _lake(tmp_path)
    out = tmp_path / "data/line_a_v2"
    f4 = _f4()
    bypass = pd.DataFrame(columns=["ticker", "importance", "n_events"])
    calls = []
    cfg = f6.CFG | {"max_cases": 6, "n_sample": 0, "budget_rmb": 1.0, "per_case_rmb": 0.4, "workers": 1}
    res = f6.run_research(tmp_path, out, "2026-10-21", f4, bypass, set(), cfg=cfg, llm_factory=lambda t: None,
                          case_runner=_fake_runner(calls), log=lambda m: None)
    st = res["status"]
    assert st["n_ok"] == 2 and st["stopped"] == "daily budget" and st["state"] == "partial"   # 第 3 个会超过 ¥1.0
    assert set(res["claims"]["column"]) == {"facts"} and len(res["flags"]) == 2
    calls.clear()
    res = f6.run_research(tmp_path, out, "2026-10-21", f4, bypass, set(), cfg=cfg | {"budget_rmb": 10},
                          llm_factory=lambda t: None, case_runner=_fake_runner(calls), log=lambda m: None)
    assert res["status"]["n_ok"] == 6 and len(calls) == 4                # 续跑：已完成的 2 个不重跑
    # 第二天：前一天研究过的股票是新鲜的，只有新数据触发的角色重跑，其余沿用
    pd.DataFrame({"ts_code": ["600100.SH"], "ann_date": ["20261022"], "end_date": ["20260930"]}).to_parquet(
        lake / "income.parquet")
    bp2 = pd.DataFrame({"ticker": ["600100.SH"], "importance": ["major"], "n_events": [1]})
    calls.clear()
    res = f6.run_research(tmp_path, out, "2026-10-22", f4, bp2, set(), cfg=cfg | {"budget_rmb": 10, "max_cases": 3},
                          llm_factory=lambda t: None, case_runner=_fake_runner(calls), log=lambda m: None)
    c = res["cases"].set_index("ticker")
    assert c.loc["600100.SH", "category"] == "event" and c.loc["600100.SH", "roles_run"] == "A1"
    assert ("600100.SH", ("A1",), ("A2", "A3", "A4")) in calls
    md = f6.report(res, {"600100.SH": "测试股份"}, None)
    assert "影子研究报告" in md and "测试股份" in md and "沿用的角色" in md
