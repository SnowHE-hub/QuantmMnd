"""T-031 download_cninfo：100 页回绕、正倒序两遍、板块 / 类别拆分对账、truncated → partial、续跑、时间预算。
用假接口，不联网。"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_P = Path(__file__).resolve().parents[1] / "scripts/data_pipeline/download_cninfo.py"
_spec = importlib.util.spec_from_file_location("download_cninfo", _P)
dc = importlib.util.module_from_spec(_spec)
sys.modules["download_cninfo"] = dc
_spec.loader.exec_module(dc)

NOSLEEP = lambda s: None  # noqa: E731
T0 = 1556236800000         # 2019-04-26 08:00 北京时间（毫秒）


def make_items(counts: dict, uncategorized: int = 0, overlap: int = 0, uncat_code: str | None = None) -> list[dict]:
    """counts = {(plate, category): n}。前 overlap 条额外属于第二个类别；uncategorized 条（深主板）没有类别，
    uncat_code 给定时这些条目都用这个代码（放在代码排序的中段，正倒序两遍都够不到）。"""
    items, i = [], 0
    for (plate, cat), n in counts.items():
        for _ in range(n):
            items.append({"announcementId": str(100_000 + i), "secCode": f"{i % 997:06d}",
                          "announcementTime": T0 + i * 1000, "announcementTitle": f"t{i}", "plate": plate,
                          "cats": {cat}})
            i += 1
    for it in items[:overlap]:
        it["cats"].add("category_zj_szsh")
    for _ in range(uncategorized):
        items.append({"announcementId": str(100_000 + i), "secCode": uncat_code or f"{i % 997:06d}",
                      "announcementTime": T0 + i * 1000, "announcementTitle": f"t{i}", "plate": "szmb", "cats": set()})
        i += 1
    return items


def fake_post(items: list[dict], calls: list):
    def post(data):
        calls.append((data["plate"], data["category"], data["sortType"], data["pageNum"]))
        sel = [a for a in items if (not data["plate"] or a["plate"] == data["plate"])
               and (not data["category"] or data["category"] in a["cats"])]
        if data["sortName"] == "code":
            sel = sorted(sel, key=lambda a: (a["secCode"], a["announcementId"]), reverse=data["sortType"] == "desc")
        page = data["pageNum"] if data["pageNum"] <= dc.MAX_PAGES else 1          # 第 101 页起回绕到第 1 页
        chunk = sel[(page - 1) * dc.PAGE_SIZE: page * dc.PAGE_SIZE]
        return {"totalAnnouncement": len(sel), "hasMore": True,
                "announcements": [{k: v for k, v in a.items() if k != "cats"} for a in chunk]}
    return post


def client(items, calls=None):
    return dc.Client(post=fake_post(items, [] if calls is None else calls), interval=0, sleep=NOSLEEP)


def test_small_day_single_leaf_reuses_first_page():
    calls = []
    rows, ok, total = dc.fetch_node(client(make_items({("szmb", "category_rcjy_szsh"): 75}), calls), "2019-04-26")
    assert ok and total == 75 and len(rows) == 75
    assert [c[3] for c in calls] == [1, 2, 3]                            # 第 1 页不重复请求


def test_category_split_then_sort_supplement_recovers_uncategorized():
    calls = []
    items = make_items({("szmb", "category_rcjy_szsh"): 2000, ("szmb", "category_ndbg_szsh"): 1500}, uncategorized=5)
    rows, ok, total = dc.fetch_node(client(items, calls), "2019-04-26")
    assert ok and total == 3505 and len(rows) == 3505                  # 类别合并 3500，正倒序补上 5 条
    assert "asc" in {c[2] for c in calls if c[0] == "szmb" and not c[1]}
    assert max(c[3] for c in calls) <= dc.MAX_PAGES


def test_big_day_splits_plate_then_category_and_reconciles():
    items = make_items({("szmb", "category_rcjy_szsh"): 4000, ("szmb", "category_ndbg_szsh"): 2500,
                        ("szcy", "category_rcjy_szsh"): 3500, ("shmb", "category_gszl_szsh"): 400}, overlap=50)
    rows, ok, total = dc.fetch_node(client(items), "2019-04-26")
    assert total == 10_400 and ok and len(rows) == 10_400          # szmb 6500 > 6000 → 按类别；重叠去重
    assert {r["plate"] for r in rows.values()} == {"szmb", "szcy", "shmb"}


def test_uncategorized_gap_truncated_then_partial_and_not_retried(tmp_path):
    # 总数 6105 > 6000；5 条无类别公告的代码在排序中段，正倒序各 3000 条都够不到
    items = make_items({("szmb", "category_rcjy_szsh"): 2900, ("szmb", "category_ndbg_szsh"): 2900,
                        ("szmb", "category_gszl_szsh"): 300}, uncategorized=5, uncat_code="000485")
    states = dc.run(tmp_path, ["2019-04-26"], client(items), log=lambda *a: None)
    ck_path = tmp_path / "_checkpoints/cninfo_meta.json"
    ck = json.loads(ck_path.read_text(encoding="utf-8"))["days"]["2019-04-26"]
    assert states == {dc.TRUNCATED: 1} and ck["total"] == 6105 and ck["rows"] == 6100
    assert (tmp_path / "_parts/20190426.parquet").exists()             # 已取到的数据照存
    states = dc.run(tmp_path, ["2019-04-26"], client(items), log=lambda *a: None)     # 重试结果不变 → partial
    ck = json.loads(ck_path.read_text(encoding="utf-8"))["days"]["2019-04-26"]
    assert states == {dc.PARTIAL: 1} and ck["gap"] == 5
    calls = []
    dc.run(tmp_path, ["2019-04-26"], client(items, calls), log=lambda *a: None)
    assert calls == []                                                  # 终态，不再请求


def test_category_leaf_over_6000_is_truncated():
    items = make_items({("szmb", "category_rcjy_szsh"): 7000})
    rows, ok, total = dc.fetch_node(client(items), "d", "szmb", "category_rcjy_szsh")
    assert total == 7000 and not ok and len(rows) == 6000


def test_run_resume_budget_and_frame(tmp_path):
    items = make_items({("szmb", "category_rcjy_szsh"): 40})
    days = ["2019-04-26", "2019-04-27", "2019-04-28"]
    ticks = [0, 0]                                                      # 起点、第 1 天之前；之后一律 100 秒
    dc.run(tmp_path, days, client(items), max_seconds=50, log=lambda *a: None,
           clock=lambda: ticks.pop(0) if ticks else 100)
    ck = json.loads((tmp_path / "_checkpoints/cninfo_meta.json").read_text(encoding="utf-8"))["days"]
    assert list(ck) == ["2019-04-26"]                                   # 预算到了：在两天之间停下
    calls = []
    states = dc.run(tmp_path, days, client(items, calls), log=lambda *a: None)
    assert states == {dc.DONE: 3} and len(calls) == 2 * 2               # 只补剩下两天，每天 2 页
    df = pd.read_parquet(tmp_path / "_parts/20190426.parquet")
    assert df["announcementId"].is_unique and df["publish_ts_bj"].iloc[0] == "2019-04-26T08:00:00+08:00"


def test_failed_request_retries_then_records_failed(tmp_path):
    def bad(data):
        raise ConnectionError("reset")
    c = dc.Client(post=bad, interval=0, sleep=NOSLEEP, retries=2)
    states = dc.run(tmp_path, ["2019-04-26"], c, log=lambda *a: None)
    ck = json.loads((tmp_path / "_checkpoints/cninfo_meta.json").read_text(encoding="utf-8"))["days"]["2019-04-26"]
    assert states == {dc.FAILED: 1} and "reset" in ck["error"] and ck["requests"] == 2
