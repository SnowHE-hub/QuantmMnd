"""T-019 download_events：分块、分页、截断拆分、空表语义、续跑、合并。"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts/data_pipeline/download_events.py"
_spec = importlib.util.spec_from_file_location("download_events", _P)
de = importlib.util.module_from_spec(_spec)
sys.modules["download_events"] = de  # dataclass 需要模块已注册
_spec.loader.exec_module(de)

NOSLEEP = lambda s: None  # noqa: E731


def _rows(n, start=0, day="20260105"):
    return pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(start, start + n)],
                         "ann_date": [day] * n, "v": [float(i) for i in range(start, start + n)]})


def test_chunks_modes():
    td = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))
    s = de.Spec("trade_day", "trade_date")
    assert [k for k, _ in de.chunks_for("x", s, "2026-01-01", "2026-01-05", td)] == ["20260102", "20260105"]
    m = de.chunks_for("x", de.Spec("month", "ann_date"), "2026-01-15", "2026-03-10", td)
    assert m[0] == ("202601", {"start_date": "20260115", "end_date": "20260131"})
    assert m[-1] == ("202603", {"start_date": "20260301", "end_date": "20260310"})
    w = de.chunks_for("x", de.Spec("window", "d", window_days=10), "2026-01-01", "2026-01-25", td)
    assert [p for _, p in w][-1] == {"start_date": "20260121", "end_date": "20260125"}
    q = de.chunks_for("x", de.Spec("period", "end_date"), "2025-01-01", "2025-12-31", td)
    assert [k for k, _ in q] == ["20250331", "20250630", "20250930", "20251231"]


def test_pagination_with_offset():
    data = _rows(250)

    def call(api, offset=0, **kw):
        return data.iloc[offset:offset + 100].reset_index(drop=True)
    state, df, pages = de.fetch_chunk(call, "x", {}, de.Spec("month", "ann_date"), sleep=NOSLEEP)
    assert state == de.DONE and len(df) == 250 and pages == 3


def test_offset_unsupported_is_truncated_and_split_by_date():
    # 接口忽略 offset：整月一次只给 100 行（截断）；按天拆分后每天 < 100 行
    days = {f"202601{d:02d}": _rows(40, start=d * 100, day=f"202601{d:02d}") for d in range(1, 5)}

    def call(api, start_date, end_date, offset=0):
        frames = [v for k, v in days.items() if start_date <= k <= end_date]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return df.iloc[:100].reset_index(drop=True)
    spec = de.Spec("month", "ann_date")
    st, df, _ = de.fetch_chunk(call, "x", {"start_date": "20260101", "end_date": "20260104"}, spec, sleep=NOSLEEP)
    assert st == de.TRUNCATED
    st, df, _ = de.fetch_adaptive(call, "x", {"start_date": "20260101", "end_date": "20260104"}, spec, sleep=NOSLEEP)
    assert st == de.DONE and len(df) == 160 and df["ts_code"].is_unique


def test_page_cap_with_full_last_page_is_truncated_not_done():
    # offset 正常、连续满页、超过 max_pages：旧代码返回 done（漏掉第 3 页之后的行）
    data = _rows(500)

    def call(api, offset=0, **kw):
        return data.iloc[offset:offset + 100].reset_index(drop=True)
    st, df, pages = de.fetch_chunk(call, "x", {}, de.Spec("month", "ann_date"), sleep=NOSLEEP, max_pages=2)
    assert st == de.TRUNCATED and len(df) == 200 and pages == 2


def test_page_cap_splits_date_range_without_missing_rows():
    # 整个区间 4 天 × 150 行 = 600 行，超过 2 页上限；对半拆分到单日（每天 2 页内取完）后一行不漏
    days = {f"202601{d:02d}": _rows(150, start=d * 1000, day=f"202601{d:02d}") for d in range(1, 5)}

    def call(api, start_date, end_date, offset=0):
        frames = [v for k, v in days.items() if start_date <= k <= end_date]
        df = pd.concat(frames, ignore_index=True)
        return df.iloc[offset:offset + 100].reset_index(drop=True)
    spec = de.Spec("month", "ann_date")
    st, df, _ = de.fetch_adaptive(call, "x", {"start_date": "20260101", "end_date": "20260104"}, spec,
                                  sleep=NOSLEEP, max_pages=2)
    assert st == de.DONE and len(df) == 600 and df["ts_code"].is_unique


def test_page_cap_on_single_day_stays_truncated_and_is_retried(tmp_path):
    # 单日分块无法再拆：保留 truncated（非成功），续跑会重试
    data = _rows(300, day="20260105")
    seen = []

    def call(api, trade_date, offset=0):
        seen.append((trade_date, offset))
        return data.iloc[offset:offset + 100].reset_index(drop=True)
    spec = de.Spec("trade_day", "trade_date")
    chunks = [("20260105", {"trade_date": "20260105"})]
    kw = {"sleep": NOSLEEP, "retries": 1, "log": lambda *a: None, "max_pages": 2}
    states = de.run_api("x", spec, chunks, tmp_path, call, **kw)
    assert states == {de.TRUNCATED: 1}
    seen.clear()
    de.run_api("x", spec, chunks, tmp_path, call, **kw)
    assert seen and seen[0] == ("20260105", 0)             # 没有被当成完成而跳过


def test_true_eof_short_page_and_duplicate_page():
    exact = _rows(200)                                      # 恰好两满页，第三页为空 → 真 EOF，done

    def call_exact(api, offset=0, **kw):
        return exact.iloc[offset:offset + 100].reset_index(drop=True)
    st, df, pages = de.fetch_chunk(call_exact, "x", {}, de.Spec("month", "ann_date"), sleep=NOSLEEP, max_pages=5)
    assert st == de.DONE and len(df) == 200 and pages == 3
    short = _rows(130)                                      # 短页 → done

    def call_short(api, offset=0, **kw):
        return short.iloc[offset:offset + 100].reset_index(drop=True)
    st, df, _ = de.fetch_chunk(call_short, "x", {}, de.Spec("month", "ann_date"), sleep=NOSLEEP, max_pages=5)
    assert st == de.DONE and len(df) == 130
    st, df, _ = de.fetch_chunk(lambda api, offset=0, **kw: exact.iloc[:100], "x", {}, de.Spec("month", "ann_date"),
                               sleep=NOSLEEP, max_pages=5)  # 忽略 offset 的重复页 → truncated
    assert st == de.TRUNCATED and len(df) == 100


def test_audit_page_cap_lists_and_marks(tmp_path):
    ck = {"version": 1, "chunks": {"a": {"state": "done", "rows": 20000, "pages": 200},
                                   "b": {"state": "done", "rows": 50, "pages": 1},
                                   "c": {"state": "failed", "pages": 200}}}
    de.save_ckpt(tmp_path / "_checkpoints/x.json", ck)
    res = de.audit_page_cap(tmp_path, ["x", "missing"], max_pages=200)
    assert res == {"x": {"a": {"rows": 20000, "pages": 200}}}
    de.audit_page_cap(tmp_path, ["x"], max_pages=200, mark=True)
    after = json.loads((tmp_path / "_checkpoints/x.json").read_text(encoding="utf-8"))["chunks"]
    assert after["a"]["state"] == de.TRUNCATED and after["a"]["page_cap_audit"]["previous_state"] == "done"
    assert after["b"]["state"] == "done" and after["c"]["state"] == "failed"


def test_empty_vs_must_have_and_errors(tmp_path):
    calls = {"n": 0}

    def call(api, trade_date):
        calls["n"] += 1
        if trade_date == "20260106":
            raise RuntimeError("网络错误")
        return pd.DataFrame() if trade_date == "20260105" else _rows(3, day=trade_date)
    spec = de.Spec("trade_day", "trade_date", must_have=True)
    chunks = [("20260102", {"trade_date": "20260102"}), ("20260105", {"trade_date": "20260105"}),
              ("20260106", {"trade_date": "20260106"})]
    states = de.run_api("x", spec, chunks, tmp_path, call, sleep=NOSLEEP, retries=2, log=lambda *a: None)
    ck = json.loads((tmp_path / "_checkpoints/x.json").read_text(encoding="utf-8"))["chunks"]
    assert ck["20260102"]["state"] == de.DONE
    assert ck["20260105"]["state"] == de.FAILED and "empty" in ck["20260105"]["error"]   # 交易日不应为空
    assert ck["20260106"]["state"] == de.FAILED and "网络错误" in ck["20260106"]["error"]
    assert states == {de.FAILED: 2, de.DONE: 1}
    # 非 must_have：空表是正常终态
    spec2 = de.Spec("trade_day", "trade_date")
    de.run_api("y", spec2, chunks[1:2], tmp_path, call, sleep=NOSLEEP, log=lambda *a: None)
    ck2 = json.loads((tmp_path / "_checkpoints/y.json").read_text(encoding="utf-8"))["chunks"]
    assert ck2["20260105"]["state"] == de.EMPTY


def test_resume_skips_terminal_and_retries_failed(tmp_path):
    seen = []
    fail = {"on": True}

    def call(api, trade_date):
        seen.append(trade_date)
        if trade_date == "20260105" and fail["on"]:
            raise RuntimeError("timeout")
        return _rows(2, day=trade_date)
    spec = de.Spec("trade_day", "trade_date")
    chunks = [("20260102", {"trade_date": "20260102"}), ("20260105", {"trade_date": "20260105"})]
    de.run_api("x", spec, chunks, tmp_path, call, sleep=NOSLEEP, retries=1, log=lambda *a: None)
    fail["on"] = False
    seen.clear()
    states = de.run_api("x", spec, chunks, tmp_path, call, sleep=NOSLEEP, retries=1, log=lambda *a: None)
    assert seen == ["20260105"] and states == {de.DONE: 2}


def test_rate_limit_message_waits(tmp_path):
    waits, n = [], {"i": 0}

    def call(api, trade_date):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("抱歉，您访问接口(report_rc)频率超限(1次/分钟)")
        return _rows(1, day=trade_date)
    de.run_api("x", de.Spec("trade_day", "trade_date"), [("20260102", {"trade_date": "20260102"})], tmp_path,
               call, sleep=waits.append, retries=2, log=lambda *a: None)
    assert 65 in waits
    waits.clear()
    n["i"] = 0

    def call_hourly(api, trade_date):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("抱歉，您访问接口(report_rc)频率超限(10次/小时)")
        return _rows(1, day=trade_date)
    de.run_api("z", de.Spec("trade_day", "trade_date"), [("20260102", {"trade_date": "20260102"})], tmp_path,
               call_hourly, sleep=waits.append, retries=2, log=lambda *a: None)
    assert 3700 in waits


def test_compact_coerces_and_refuses_overwrite(tmp_path):
    parts = tmp_path / "_parts/x"
    parts.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["000001.SZ"], "ann_date": ["20260105"], "v": [1.5]}).to_parquet(parts / "a.parquet")
    # 另一块里数值列整列为空（对象型 None）
    pd.DataFrame({"ts_code": ["000002.SZ", "000002.SZ"], "ann_date": ["20260106"] * 2,
                  "v": [None, None]}).to_parquet(parts / "b.parquet")
    de.save_ckpt(tmp_path / "_checkpoints/x.json", {"version": 1, "chunks": {"a": {"state": "done"}, "b": {"state": "done"}}})
    man = de.compact("x", tmp_path, "2026-10-09", de.Spec("month", "ann_date"))
    out = pd.read_parquet(tmp_path / "x_20261009.parquet")
    assert man["rows"] == 2 and man["exact_duplicate_rows_dropped"] == 1
    assert out["v"].dtype == "float64" and out["ts_code"].iloc[0] == "000001.SZ"
    assert man["date_min"] == "20260105" and man["date_max"] == "20260106"
    with pytest.raises(FileExistsError):
        de.compact("x", tmp_path, "2026-10-09", de.Spec("month", "ann_date"))
