"""T-035 P0：任务表、数据契约、可用时点、磁盘预算、GPU 租约。"""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quantmind.line_a import jobs as jb
from quantmind.line_a.availability import (
    TradingCalendar,
    available_at_date_only,
    available_at_minute,
    available_at_trade_bar,
)
from quantmind.line_a.contracts import (
    AvailabilityBasis,
    Claim,
    ClaimType,
    Producer,
    RecordTimes,
    SnapshotManifest,
    SupportStatus,
    snapshot_input,
)
from quantmind.line_a.gpu_lease import GpuLease, GpuLeaseError
from quantmind.line_a.resources import check_cache_budget, effective_free_bytes

TZ = ZoneInfo("Asia/Shanghai")


# ------------------------------------------------------------------ 任务表
def test_job_idempotent_first_commit_wins(tmp_path):
    store = jb.JobStore(tmp_path / "jobs.db")
    fields = {"snapshot": "s1", "role": "A1@v1", "case": "600000.SH", "model": "m1", "prompt": "p1"}
    out = tmp_path / "out" / "a.json"
    calls = []

    def produce():
        calls.append(1)
        return b'{"answer": 1}'

    r1 = store.run("F6", fields, out, produce)
    assert r1["status"] == jb.SUCCEEDED and out.read_bytes() == b'{"answer": 1}'
    r2 = store.run("F6", fields, out, lambda: b'{"answer": 2}')   # 已成功：不再执行，不替换
    assert r2["output_sha256"] == r1["output_sha256"] and calls == [1] and out.read_bytes() == b'{"answer": 1}'
    key = jb.task_key(stage="F6", **fields)
    assert store.commit_output(key, out, b'{"answer": 3}')["output_sha256"] == r1["output_sha256"]
    assert out.read_bytes() == b'{"answer": 1}'


def test_job_key_changes_with_any_field():
    base = {"snapshot": "s1", "role": "A1@v1", "case": "x", "model": "m1", "prompt": "p1"}
    keys = {jb.task_key(stage="F6", **base)} | {jb.task_key(stage="F6", **dict(base, **{k: "z"})) for k in base}
    assert len(keys) == 1 + len(base)


def test_job_running_lease_and_crash_recovery(tmp_path):
    now = {"t": 1000.0}
    store = jb.JobStore(tmp_path / "jobs.db", lease_seconds=60, clock=lambda: now["t"])
    key = jb.task_key(stage="F5", case="a")
    assert store.claim(key, "F5", {"case": "a"})
    assert not store.claim(key, "F5", {"case": "a"})        # 租约内：别人在跑
    now["t"] += 61
    assert store.claim(key, "F5", {"case": "a"})            # 租约过期：视为崩溃，重新认领
    assert store.get(key)["attempts"] == 2


def test_job_failure_retry_and_skip(tmp_path):
    store = jb.JobStore(tmp_path / "jobs.db")
    with pytest.raises(RuntimeError):
        store.run("F5", {"case": "b"}, tmp_path / "b.json", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    key = jb.task_key(stage="F5", case="b")
    assert store.get(key)["status"] == jb.RETRYABLE_FAILED
    assert store.run("F5", {"case": "b"}, tmp_path / "b.json", lambda: b"ok")["status"] == jb.SUCCEEDED
    skey = jb.task_key(stage="F6", case="c")
    store.skip(skey, "F6", {"case": "c"}, "no new announcements")
    assert store.get(skey)["status"] == jb.SKIPPED and store.counts("F6") == {jb.SKIPPED: 1}


# ------------------------------------------------------------------ 契约
def _producer():
    return Producer(role="A2", model="m@sha", prompt_schema_hash="h")


def test_claim_binding_rules():
    Claim(claim_id="c1", claim_type=ClaimType.REPORTED_FACT, subject_id="issuer:600000.SH", predicate="repurchase_status",
          value="proposed", evidence_ids=["ev-1"], support_status=SupportStatus.SUPPORTED, producer=_producer())
    with pytest.raises(ValidationError):   # 事实必须有证据
        Claim(claim_id="c2", claim_type=ClaimType.REPORTED_FACT, subject_id="s", predicate="p",
              support_status=SupportStatus.UNRESOLVED, producer=_producer())
    with pytest.raises(ValidationError):   # 计算值必须有公式和输入
        Claim(claim_id="c3", claim_type=ClaimType.COMPUTED, subject_id="s", predicate="roe", value=0.1,
              support_status=SupportStatus.UNRESOLVED, producer=_producer())
    Claim(claim_id="c4", claim_type=ClaimType.OPEN_QUESTION, subject_id="s", predicate="guarantee_status",
          support_status=SupportStatus.SEARCHED_NO_SUPPORTING_EVIDENCE, producer=_producer())


def test_record_times_forward_vs_history():
    cutoff = datetime(2026, 10, 21, 21, 0, tzinfo=TZ)
    rec = RecordTimes(published_at=datetime(2026, 10, 21, 19, 32, tzinfo=TZ),
                      first_seen_at=datetime(2026, 10, 21, 21, 5, tzinfo=TZ),
                      availability_basis=AvailabilityBasis.SOURCE)
    assert not rec.usable_forward(cutoff, decision_ts=cutoff)                 # 发布了，但系统决策时还没取得
    assert rec.usable_forward(cutoff, decision_ts=cutoff + timedelta(minutes=10))
    assert rec.usable_history(cutoff) and not rec.usable_history(cutoff, timedelta(hours=2))


def test_snapshot_manifest_hash_ignores_creation_time(tmp_path):
    f = tmp_path / "daily.parquet"
    f.write_bytes(b"abc")
    inp = [snapshot_input("daily", f, rows=1, watermark="2026-10-21")]
    m1 = SnapshotManifest(as_of="2026-10-21", data_cutoff_ts=datetime(2026, 10, 21, 21, tzinfo=TZ),
                          created_utc=datetime(2026, 10, 21, 13), inputs=inp)
    m2 = m1.model_copy(update={"created_utc": datetime(2026, 10, 22, 1)})
    assert m1.content_hash() == m2.content_hash()
    f.write_bytes(b"abd")
    m3 = m1.model_copy(update={"inputs": [snapshot_input("daily", f)]})
    assert m3.content_hash() != m1.content_hash()


# ------------------------------------------------------------------ 可用时点
def test_availability_rules():
    cal = TradingCalendar([date(2026, 10, 16), date(2026, 10, 19), date(2026, 10, 20)], cutoff=time(21, 0))
    assert available_at_date_only(date(2026, 10, 16), cal) == datetime(2026, 10, 19, 21, 0, tzinfo=TZ)  # 跨周末
    assert available_at_date_only(date(2026, 10, 18), cal) == datetime(2026, 10, 19, 21, 0, tzinfo=TZ)  # 周日公告
    assert available_at_trade_bar(date(2026, 10, 19), cal) == datetime(2026, 10, 19, 21, 0, tzinfo=TZ)
    with pytest.raises(ValueError):
        available_at_trade_bar(date(2026, 10, 18), cal)
    assert available_at_minute(datetime(2026, 10, 20, 19, 32)).tzinfo == TZ


# ------------------------------------------------------------------ 磁盘
def test_effective_free_is_min_of_vhd_and_host(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(0, 0, 830 if str(p) == "/" else 240))
    host = tmp_path / "d"
    host.mkdir()
    assert effective_free_bytes("/", str(host))["effective_free"] == 240
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "x.pdf").write_bytes(b"0" * 10)
    r = check_cache_budget(tmp_path / "cache", cap_bytes=100, min_effective_free=200, host_mount=str(host))
    assert r["ok"] and r["cache_used"] == 10
    assert not check_cache_budget(tmp_path / "cache", cap_bytes=5, min_effective_free=200, host_mount=str(host))["ok"]


# ------------------------------------------------------------------ GPU 租约
def test_gpu_lease_waits_for_free_memory_and_times_out(tmp_path):
    frees = iter([1000, 3000, 6000])
    unloaded = []
    lease = GpuLease("t", need_mib=5000, lock_path=tmp_path / "gpu.lock", free_timeout=30, poll=5,
                     free_mib=lambda: next(frees), unloaders=(lambda: unloaded.append(1) or ["qwen3.5:9b"],),
                     sleep=lambda s: None)
    with lease as held:
        assert held.info["free_mib"] == 6000 and held.info["unloaded"] == ["qwen3.5:9b"]
        assert held.holder_file.exists()
    assert not (tmp_path / "gpu.holder.json").exists()
    starved = GpuLease("t", need_mib=5000, lock_path=tmp_path / "gpu.lock", free_timeout=10, poll=5,
                       free_mib=lambda: 100, unloaders=(), sleep=lambda s: None)
    with pytest.raises(GpuLeaseError), starved:
        pass
    with GpuLease("t2", need_mib=10, lock_path=tmp_path / "gpu.lock", free_mib=lambda: 100, unloaders=(),
                  sleep=lambda s: None):
        pass   # 失败后锁已经释放


# ------------------------------------------------------------------ 停机补跑
def test_catchup_heartbeat_and_missed_days(tmp_path):
    import json as _json

    from quantmind.line_a.catchup import last_ready_snapshot, missed_trading_days, read_heartbeat
    hb = tmp_path / "heartbeat.json"
    assert read_heartbeat(hb) == {"present": False}
    hb.write_text(_json.dumps({"utc": "2026-10-21T00:00:00.0000000Z", "keepalive_pid": 1, "cron": "active"}),
                  encoding="utf-8-sig")
    r = read_heartbeat(hb, now=datetime(2026, 10, 21, 0, 10, tzinfo=ZoneInfo("UTC")))
    assert r["present"] and r["age_s"] == 600 and r["cron"] == "active"
    snaps = tmp_path / "snapshots"
    for d, ready in (("2026-10-16", True), ("2026-10-19", False)):
        (snaps / d).mkdir(parents=True)
        (snaps / d / "snapshot_x.json").write_text(_json.dumps({"as_of": d, "notes": [f"data_ready={ready}"]}))
    last = last_ready_snapshot(snaps)
    assert last == date(2026, 10, 16)
    days = [date(2026, 10, 16), date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)]
    assert missed_trading_days(days, last, date(2026, 10, 20)) == [date(2026, 10, 19), date(2026, 10, 20)]
    assert missed_trading_days(days, None, date(2026, 10, 18)) == []


# ------------------------------------------------------------------ F1 证券身份与交易状态
def test_f1_instrument_state_classes(tmp_path):
    import pandas as pd

    from quantmind.line_a.f1_instruments import instrument_state
    lake = tmp_path / "data/lake"
    (lake / "events").mkdir(parents=True)
    days = pd.bdate_range("2026-01-01", "2026-10-20")
    pd.DataFrame({"cal_date": days.strftime("%Y%m%d")}).to_parquet(lake / "trade_cal_sse.parquet")
    pd.DataFrame({"ts_code": ["600000.SH", "300001.SZ", "688001.SH", "000002.SZ", "600999.SH"],
                  "list_date": ["20100101", "20100101", "20260901", "20100101", "20100101"],
                  "delist_date": [None, None, None, None, "20260101"],
                  "name": ["浦发银行", "特锐德", "新股", "万科A", "已退市"]}).to_parquet(
        lake / "stock_basic_full_merged_20261020.parquet")
    pd.DataFrame({"ts_code": ["000002.SZ", "300001.SZ"], "name": ["*ST万科", "特锐德"],
                  "start_date": ["20260601", "20100101"], "end_date": [None, None]}).to_parquet(
        lake / "events/namechange_20261020.parquet")
    pd.DataFrame({"ts_code": ["300001.SZ"], "trade_date": ["20261020"], "suspend_type": ["S"]}).to_parquet(
        lake / "events/suspend_d_20261020.parquet")
    d = pd.Timestamp("2026-10-20")
    pd.DataFrame({"ts_code": ["600000.SH", "688001.SH", "000002.SZ"], "trade_date": [d] * 3,
                  "close": [11.0, 30.0, 1.5], "high": [11.0, 31.0, 1.6], "low": [11.0, 29.0, 1.4],
                  "pre_close": [10.0, 30.0, 1.5]}).to_parquet(lake / "daily.parquet")
    pd.DataFrame({"trade_date": ["20261020"] * 3, "ts_code": ["600000.SH", "688001.SH", "000002.SZ"],
                  "up_limit": [11.0, 36.0, 1.58], "down_limit": [9.0, 24.0, 1.43]}).to_parquet(
        lake / "events/stk_limit_20261020.parquet")
    prof = {"all": {"MAIN": True, "CHINEXT": True, "STAR": True, "BSE": True}, "main": {"MAIN": True}}
    st = instrument_state(tmp_path, "2026-10-20", prof, holdings={"600999.SH"})
    assert set(st.index) == {"600000.SH", "300001.SZ", "688001.SH", "000002.SZ", "600999.SH"}   # 已退市的持仓仍保留
    assert st.loc["600000.SH", "one_line_up"] and st.loc["600000.SH", "buy_execution_risk"]
    assert st.loc["300001.SZ", "suspended"] and st.loc["300001.SZ", "sell_execution_risk"]
    assert st.loc["000002.SZ", "is_st"] and st.loc["000002.SZ", "c_low_price"]
    assert st.loc["688001.SH", "c_listed_lt_min"] and not st.loc["688001.SH", "new_buy_main"]
    assert st.loc["688001.SH", "new_buy_all"] and st.loc["600000.SH", "new_buy_main"]


# ------------------------------------------------------------------ F2 / F4
def test_f2_domains_and_f4_candidates(tmp_path):
    import numpy as np
    import pandas as pd

    from quantmind.line_a.f2f4 import research_candidates, support_domains
    mdir = tmp_path / "data/gpu_v1/model/v1"
    mdir.mkdir(parents=True)
    tick = [f"{i:06d}.SZ" for i in range(8)]
    meta = pd.DataFrame({"as_of": pd.Timestamp("2026-10-16"), "ticker": tick,
                         "top1500": [True] * 6 + [False, False]})
    meta.to_parquet(mdir / "2026-10.meta.parquet")
    X = np.zeros((8, 4), dtype=np.float16)
    X[5, :3] = np.nan                                          # 缺失过多 → 数据不足域
    np.save(mdir / "2026-10.X.npy", X)
    dom = support_domains(tmp_path, "2026-10-16")
    assert dom["domain"].tolist() == ["research_pool"] * 5 + ["data_insufficient", "extended", "extended"]
    sc = pd.DataFrame({"ens20": [5, 4, 3, 2, 1, 0, 9, 9], "ridge20": [1, 2, 3, 4, 5, 6, 9, 9]}, index=tick, dtype=float)
    c = research_candidates(sc, dom, "2026-10-16", holdings={"000007.SZ"}, event_queue={"000004.SZ"}, top_k=2,
                            n_samples=1)
    assert "000006.SZ" not in c.index                           # 扩展域的高分不进入宽候选
    assert c.loc["000007.SZ", "reasons"] == ["holding"] and c.loc["000007.SZ", "priority"] == 0
    assert set(c.loc["000000.SZ", "reasons"]) == {"top2_ens20"} and "top2_ridge20" in c.loc["000004.SZ", "reasons"]
    assert "event_queue" in c.loc["000004.SZ", "reasons"]
    samples = [t for t, r in c["reasons"].items() if "stratified_sample" in r]
    assert len(samples) == 1 and samples[0] in {"000002.SZ"}    # 剩下的原研究池只有 000002
    c2 = research_candidates(sc, dom, "2026-10-16", holdings={"000007.SZ"}, event_queue={"000004.SZ"}, top_k=2,
                             n_samples=1)
    assert c2.index.tolist() == c.index.tolist()                # 同一天抽样可复现


# ------------------------------------------------------------------ F8 容量情景
def test_f8_capacity_lots_and_participation():
    import pandas as pd

    from quantmind.line_a.f8_books import _shares, capacity_report, restrict_to_profile
    assert _shares(1000, 15.0, "MAIN") == 0              # 不足一手
    assert _shares(10000, 15.0, "MAIN") == 600           # 主板整百股
    assert _shares(10000, 15.0, "STAR") == 666           # 科创板 200 股起、1 股递增
    assert _shares(2000, 15.0, "STAR") == 0
    days = pd.bdate_range("2026-09-01", periods=25)
    px = pd.DataFrame([{"ts_code": t, "trade_date": d, "close": c, "amount": 1000.0}   # 千元 → 100 万元 / 日
                       for t, c in (("600000.SH", 10.0), ("688001.SH", 100.0)) for d in days])
    tg = pd.DataFrame([{"as_of": days[-1], "targets": ["600000.SH", "688001.SH"], "target_weight_each": 0.5}])
    r = capacity_report(tg, px, aum=100000).iloc[0]
    assert r["n_unbuyable_one_lot"] == 0 and r["n_over_participation"] == 2      # 5 万 / 100 万 = 5% > 1%
    r2 = capacity_report(tg, px, aum=20000).iloc[0]
    assert r2["n_unbuyable_one_lot"] == 1                                      # 1 万买不起 200 股 × 100 元
    preds = pd.DataFrame({"as_of": days[-1], "ticker": ["600000.SH", "688001.SH", "300001.SZ"], "score": [1, 2, 3]})
    assert restrict_to_profile(preds, {"MAIN": True})["ticker"].tolist() == ["600000.SH"]


# ------------------------------------------------------------------ 前瞻网格相位
def test_forward_grid_keeps_frozen_phase():
    import pandas as pd

    from quantmind.line_a.f8_books import extended_grid, forward_grid
    cal = pd.bdate_range("2026-08-03", "2026-11-30")
    frozen = list(cal[cal <= "2026-09-15"][::-5][::-1])           # 以 09-15 结尾、步长 5
    ext = extended_grid(cal, frozen)
    i = cal.get_loc(pd.Timestamp("2026-09-15"))
    assert ext[len(frozen)] == cal[i + 5] and ext[len(frozen) + 1] == cal[i + 10]
    fw = forward_grid(cal, frozen)
    assert fw[0] > pd.Timestamp("2026-10-09") and all(d in ext for d in fw)
    assert all((cal.get_loc(b) - cal.get_loc(a)) == 5 for a, b in zip(fw, fw[1:], strict=False))


def test_md_table_formats():
    import pandas as pd

    from quantmind.line_a.pipeline import _md_table
    t = _md_table(pd.DataFrame({"a": [1.23456, None], "b": ["x", "y"]}))
    assert t.splitlines()[0] == "| a | b |" and "1.235" in t and "|  | y |" in t
