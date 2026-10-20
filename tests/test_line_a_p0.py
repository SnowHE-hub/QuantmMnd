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
