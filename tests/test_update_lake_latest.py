"""T-015：lake 增量更新只补缺失日期、不改已有行、空表不算完成."""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.data_pipeline.update_lake_latest as ul
from quantmind.data import lake

D = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])


@pytest.fixture()
def tmp_lake(tmp_path, monkeypatch):
    monkeypatch.setattr(lake, "DATA_LAKE_DIR", tmp_path / "lake")
    monkeypatch.setattr(lake, "CHECKPOINT_DIR", tmp_path / "lake" / "_checkpoints")
    (tmp_path / "lake").mkdir()
    old = pd.DataFrame({"ts_code": ["000001.SZ"] * 2, "trade_date": D[[0, 2]],
                        "close": [10.0, 11.0], "ticker": ["000001.SZ"] * 2})
    old.to_parquet(lake.lake_path("daily"), index=False)
    return tmp_path


class FakeTP:
    def __init__(self, empty=()):
        self.empty = set(empty)
        self.calls = []

    def _call(self, api, **kw):
        self.calls.append(kw.get("trade_date"))
        if kw.get("trade_date") in self.empty:
            return pd.DataFrame()
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [kw["trade_date"]],
                             "close": [99.0], "extra_col": [1]})


def test_missing_dates_only_absent_days(tmp_lake):
    todo = ul.missing_dates(pd.DatetimeIndex(D), ul.present_dates("daily"), "2024-01-01", "2024-01-05")
    assert todo == [D[1], D[3]]


def test_update_fills_gaps_without_touching_existing_rows(tmp_lake):
    todo = [D[1], D[3]]
    rep = ul.update_by_date(FakeTP(), "daily", "daily", todo, tmp_lake / "bk", flush_every=10)
    out = pd.read_parquet(lake.lake_path("daily")).sort_values("trade_date")
    assert list(out["close"]) == [10.0, 99.0, 11.0, 99.0]         # 已有两行原样保留
    assert list(out.columns) == ["ts_code", "trade_date", "close", "ticker"]  # 多余列被丢弃
    assert rep["filled"] == ["20240103", "20240105"]
    assert (tmp_lake / "bk" / "daily.parquet").exists()             # 首次写入前已备份


def test_empty_trading_day_is_reported_and_not_written(tmp_lake):
    rep = ul.update_by_date(FakeTP(empty={"20240105"}), "daily", "daily", [D[1], D[3]],
                            tmp_lake / "bk", flush_every=10)
    assert rep["empty"] == ["20240105"]
    assert D[3] not in ul.present_dates("daily")                   # 下次运行仍会重试


def test_fundamentals_merge_keeps_old_values_and_adds_new(tmp_path):
    from scripts.data_pipeline.update_fundamentals_latest import merge_keep_old

    p = tmp_path / "income.parquet"
    pd.DataFrame({"ts_code": ["A", "A"], "ann_date": ["20260425", "20250820"],
                  "end_date": ["20260331", "20250630"], "revenue": [1.0, 2.0]}).to_parquet(p, index=False)
    new = pd.DataFrame({"ts_code": ["A", "A"], "ann_date": ["20260425", "20260828"],
                        "end_date": ["20260331", "20260630"], "revenue": [999.0, 3.0], "extra": [0, 0]})
    added, total = merge_keep_old(p, new, tmp_path / "bk")
    out = pd.read_parquet(p).set_index("end_date")["revenue"]
    assert (added, total) == (1, 3)
    assert out["20260331"] == 1.0                 # 主键已存在：保留旧值
    assert out["20260630"] == 3.0                 # 新主键：新增
    assert list(pd.read_parquet(p).columns) == ["ts_code", "ann_date", "end_date", "revenue"]
    assert (tmp_path / "bk" / "income.parquet").exists()


def test_align_to_lake_casts_types(tmp_lake):
    raw = pd.DataFrame({"ts_code": ["000002.SZ"], "trade_date": ["20240103"], "close": ["12.5"]})
    out = ul.align_to_lake("daily", raw)
    assert out["close"].dtype == "float64" and out["trade_date"].dtype.kind == "M"
    assert out.loc[0, "ticker"] == "000002.SZ"


# ---------------------------------------------------------------- 2026-10-11：回应 Codex REJECT
import json  # noqa: E402

import scripts.data_pipeline.update_fundamentals_latest as uf  # noqa: E402
import scripts.data_pipeline.verify_lake_update as vl  # noqa: E402


def _fund_old() -> pd.DataFrame:
    return pd.DataFrame({"ts_code": ["A", "A", "B"], "ann_date": ["20250820", None, "20250425"],
                         "end_date": ["20250630", "20260331", "20250331"],
                         "revenue": [1.0, float("nan"), 3.0], "note": ["x", None, "z"]})


def test_merge_preserves_null_keys_dtypes_and_every_old_value(tmp_path):
    p = tmp_path / "income.parquet"
    _fund_old().to_parquet(p, index=False)
    old = pd.read_parquet(p)
    new = pd.DataFrame({"ts_code": ["A", "A", "B", "B"], "ann_date": [None, "20260828", None, "20250425"],
                        "end_date": ["20260331", "20260630", "20260630", "20250331"],
                        "revenue": [999.0, 5.0, 6.0, 999.0], "note": ["dup", "n1", "n2", "dup"], "extra": 0})
    added, total = uf.merge_keep_old(p, new, tmp_path / "bk")
    out = pd.read_parquet(p)
    assert (added, total) == (2, 5)
    pd.testing.assert_frame_equal(out.iloc[:3].reset_index(drop=True), old, check_exact=True)   # 旧行逐值（含 null）
    assert out["ann_date"].isna().sum() == 2 and not out["ann_date"].isin(["None", "nan"]).any()
    assert list(out.columns) == list(old.columns) and (out.dtypes == old.dtypes).all()
    assert set(out.iloc[3:]["revenue"]) == {5.0, 6.0}            # 同主键（含 null 主键）的新值不覆盖旧值


def test_merge_without_new_keys_does_not_rewrite(tmp_path):
    p = tmp_path / "income.parquet"
    _fund_old().to_parquet(p, index=False)
    mtime = p.stat().st_mtime_ns
    added, _ = uf.merge_keep_old(p, _fund_old().assign(revenue=7.0), tmp_path / "bk")
    assert added == 0 and p.stat().st_mtime_ns == mtime and not (tmp_path / "bk").exists()


def test_repair_restores_old_nulls_and_keeps_new_rows(tmp_path):
    lake_dir, bk = tmp_path / "data" / "lake", tmp_path / "bk"
    lake_dir.mkdir(parents=True)
    bk.mkdir()
    _fund_old().to_parquet(bk / "income.parquet", index=False)
    old = pd.read_parquet(bk / "income.parquet")
    bad = pd.concat([old.assign(ann_date=old["ann_date"].astype(str)),                 # 旧 bug：None → 'None'
                     pd.DataFrame({"ts_code": ["C", "C"], "ann_date": ["20260828", "20261010"],
                                   "end_date": ["20260630", "20260930"], "revenue": [8.0, 9.0],
                                   "note": ["c", "late"]})], ignore_index=True)
    bad.to_parquet(lake_dir / "income.parquet", index=False)
    rep = uf.repair_from_backup(lake_dir, bk, "20261009", plan=False)["income"]
    out = pd.read_parquet(lake_dir / "income.parquet")
    pd.testing.assert_frame_equal(out.iloc[:3].reset_index(drop=True), old, check_exact=True)
    assert list(out["ts_code"].iloc[3:]) == ["C"] and rep["dropped_ann_after_end"] == 1
    assert rep["null_literals_in_current_keys"]["ann_date"] == 1
    assert (lake_dir / f"_quarantine_{pd.Timestamp.now():%Y%m%d}" / "income.parquet").exists()


OK, EMPTY, PARTIAL, EXC_ONCE, EMPTY_ONCE, EXC, LATE = (f"00000{i}.SZ" for i in range(1, 8))


class FakeFundTP:
    """假 Tushare（不联网）：OK 正常；EMPTY 四个接口始终为空；PARTIAL 的 cashflow 始终为空；
    EXC_ONCE 第一次请求抛异常后恢复；EMPTY_ONCE 第一轮全空、第二轮有数据；EXC 始终抛异常；LATE 只有截止日之后的公告。"""

    def __init__(self, tickers):
        self.tickers = tickers
        self.calls: list[tuple[str, str]] = []
        self.n: dict = {}

    def _raw_stock_basic_all(self):
        return pd.DataFrame({"ts_code": self.tickers, "list_status": "L", "list_date": "20100101",
                             "delist_date": None})

    def _call(self, api, ts_code, start_date, end_date, fields):
        self.calls.append((api, ts_code))
        k = (api, ts_code)
        self.n[k] = self.n.get(k, 0) + 1
        self.n[ts_code] = self.n.get(ts_code, 0) + 1
        if ts_code == EXC or (ts_code == EXC_ONCE and self.n[ts_code] == 1):
            raise RuntimeError("connection reset")
        if ts_code == EMPTY or (ts_code == PARTIAL and api == "cashflow") or (ts_code == EMPTY_ONCE and self.n[k] == 1):
            return pd.DataFrame(columns=fields.split(","))
        ann = "20261010" if ts_code == LATE else "20260828"
        row = dict.fromkeys(fields.split(","), 1.0) | {"ts_code": ts_code, "ann_date": ann, "end_date": "20260630"}
        return pd.DataFrame([row])


def _fund_lake(tmp_path):
    lake_dir = tmp_path / "data" / "lake"
    lake_dir.mkdir(parents=True)
    for _api, (table, fields) in uf.SPEC.items():
        row = dict.fromkeys(fields.split(","), 0.0) | {"ts_code": "600000.SH", "ann_date": None, "end_date": "20251231"}
        pd.DataFrame([row]).astype(dict.fromkeys(uf.KEY, object)).to_parquet(lake_dir / f"{table}.parquet", index=False)
    pd.read_parquet(lake_dir / "fina_indicator_full.parquet")[["ts_code", "ann_date", "end_date", "roe"]] \
        .to_parquet(lake_dir / "fina_indicator.parquet", index=False)
    return lake_dir


def _ck(lake_dir):
    return json.loads((lake_dir / "_checkpoints" / "fundamentals_update_20261009.v2.json").read_text())


def test_fundamentals_states_empty_partial_exception(tmp_path):
    lake_dir = _fund_lake(tmp_path)
    rc = uf.run(FakeFundTP([OK, EMPTY, PARTIAL, EXC_ONCE, EMPTY_ONCE]), lake_dir, since="20260101",
                end="20261009", tag="20261009", flush_every=2, empty_retries=2, retry_wait=0)
    ck = _ck(lake_dir)
    assert rc == 0 and ck["summary"]["complete"]
    assert set(ck["done"]) == {OK, EXC_ONCE, EMPTY_ONCE}             # 异常后恢复、首次空后有数据 → done
    assert set(ck["no_data"]) == {EMPTY} and len(ck["no_data"][EMPTY]["attempts"]) == 2
    assert ck["no_data"][EMPTY]["list_status"] == "L"                # 可审计：附上市状态
    assert ck["partial"][PARTIAL]["empty_apis"] == ["cashflow"]
    inc = pd.read_parquet(lake_dir / "income.parquet")
    assert set(inc["ts_code"]) == {"600000.SH", OK, PARTIAL, EXC_ONCE, EMPTY_ONCE}
    assert inc["ann_date"].isna().sum() == 1
    assert set(pd.read_parquet(lake_dir / "fina_indicator.parquet")["ts_code"]) == set(inc["ts_code"])


def test_single_empty_return_is_not_terminal():
    ck = {"done": {}, "partial": {}, "no_data": {}, "empty_attempts": {}, "failed": {}}
    zero = dict.fromkeys(uf.SPEC, 0)
    assert uf.classify(ck, "B", zero, None, empty_retries=2, info={}, now="t1") == "pending"
    assert "B" not in uf.terminal(ck)
    assert uf.classify(ck, "B", zero, None, empty_retries=2, info={}, now="t2") == "no_data"


def test_persistent_exception_is_failed_and_run_incomplete(tmp_path):
    lake_dir = _fund_lake(tmp_path)
    rc = uf.run(FakeFundTP([OK, EXC]), lake_dir, since="20260101", end="20261009", tag="20261009",
                empty_retries=2, retry_wait=0)
    ck = _ck(lake_dir)
    assert rc == 3 and not ck["summary"]["complete"] and set(ck["failed"]) == {EXC}


def test_announcements_after_end_are_not_written(tmp_path):
    lake_dir = _fund_lake(tmp_path)
    uf.run(FakeFundTP([LATE]), lake_dir, since="20260101", end="20261009", tag="20261009",
           empty_retries=1, retry_wait=0)
    assert LATE not in set(pd.read_parquet(lake_dir / "income.parquet")["ts_code"])


def test_legacy_v1_checkpoint_is_not_trusted(tmp_path):
    lake_dir = _fund_lake(tmp_path)
    (lake_dir / "_checkpoints").mkdir()
    (lake_dir / "_checkpoints" / "fundamentals_update_20261009.json").write_text(json.dumps({"done": ["000001.SZ"]}))
    fake = FakeFundTP(["000001.SZ"])
    uf.run(fake, lake_dir, since="20260101", end="20261009", tag="20261009", empty_retries=1, retry_wait=0)
    assert ("income", "000001.SZ") in fake.calls and _ck(lake_dir)["legacy_v1_done_ignored"] == 1


# ---------------------------------------------------------------- verifier 反例
CAL = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])


def _verify_args(tmp_path, series, old_days, new_days, run_empty=None):
    lake_dir, bk = tmp_path / "data" / "lake", tmp_path / "bk"
    (lake_dir / "_checkpoints").mkdir(parents=True)
    bk.mkdir()
    pd.DataFrame({"cal_date": CAL}).to_parquet(lake_dir / "trade_cal_sse.parquet", index=False)

    def frame(days):
        d = pd.DatetimeIndex(days)
        if series == "north_bound":
            return pd.DataFrame({"trade_date": d, "north_money": 1.0})
        return pd.DataFrame({"ts_code": "000001.SZ", "trade_date": d, "close": 1.0})
    frame(old_days).to_parquet(bk / f"{series}.parquet", index=False)
    frame(new_days).to_parquet(lake_dir / f"{series}.parquet", index=False)
    if run_empty is not None:
        (lake_dir / "_checkpoints" / "update_latest_20240106_000000.json").write_text(json.dumps(
            {"plan": False, "reports": [{"series": series, "empty": run_empty}]}))
    return ["--backup", str(bk), "--end", "2024-01-05", "--data-root", str(tmp_path), "--series", series]


def test_verifier_fails_on_missing_trading_day(tmp_path):
    assert vl.main(_verify_args(tmp_path, "daily", CAL[:2], CAL[[0, 1, 3]])) == 1


def test_verifier_passes_complete_daily(tmp_path):
    assert vl.main(_verify_args(tmp_path, "daily", CAL[:2], CAL)) == 0


def test_verifier_fails_on_non_trading_day(tmp_path):
    assert vl.main(_verify_args(tmp_path, "daily", CAL[:2], CAL.append(pd.DatetimeIndex(["2024-01-06"])))) == 1


def test_market_series_get_no_exemption_from_run_report(tmp_path):
    assert vl.main(_verify_args(tmp_path, "daily", CAL[:2], CAL[[0, 1, 3]], run_empty=["20240104"])) == 1


def test_north_bound_gap_needs_recorded_empty_return(tmp_path):
    assert vl.main(_verify_args(tmp_path, "north_bound", CAL[:2], CAL[[0, 1, 3]])) == 1


def test_north_bound_recorded_empty_day_is_exempt(tmp_path):
    assert vl.main(_verify_args(tmp_path, "north_bound", CAL[:2], CAL[[0, 1, 3]], run_empty=["20240104"])) == 0


def test_verifier_fails_when_old_null_key_became_literal(tmp_path):
    lake_dir, bk = tmp_path / "data" / "lake", tmp_path / "bk"
    lake_dir.mkdir(parents=True)
    bk.mkdir()
    pd.DataFrame({"cal_date": CAL}).to_parquet(lake_dir / "trade_cal_sse.parquet", index=False)
    old = _fund_old()
    old.to_parquet(bk / "income.parquet", index=False)
    old.assign(ann_date=old["ann_date"].astype(str)).to_parquet(lake_dir / "income.parquet", index=False)
    args = ["--backup", str(bk), "--end", "2026-10-09", "--data-root", str(tmp_path), "--series", "income"]
    assert vl.main(args) == 1


def test_verifier_requires_complete_fund_checkpoint(tmp_path):
    lake_dir = tmp_path / "data" / "lake"
    lake_dir.mkdir(parents=True)
    pd.DataFrame({"cal_date": CAL}).to_parquet(lake_dir / "trade_cal_sse.parquet", index=False)
    ck = tmp_path / "ck.json"
    ck.write_text(json.dumps({"summary": {"complete": False, "failed": 0, "pending": 3}}))
    assert vl.main(["--backup", str(tmp_path / "none"), "--end", "2024-01-05", "--data-root", str(tmp_path),
                    "--series", "--fund-checkpoint", str(ck)]) == 1
