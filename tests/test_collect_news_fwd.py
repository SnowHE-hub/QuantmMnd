"""T-031 L4 collect_news_fwd：跨次去重、按首次看到的日期分文件、来源失败互不影响、超时。不联网。"""
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_P = Path(__file__).resolve().parents[1] / "scripts/data_pipeline/collect_news_fwd.py"
_spec = importlib.util.spec_from_file_location("collect_news_fwd", _P)
cn = importlib.util.module_from_spec(_spec)
sys.modules["collect_news_fwd"] = cn
_spec.loader.exec_module(cn)


def _items(n, start=0):
    return [{"publish_time": f"2026-10-17 09:{i:02d}:00", "title": f"t{i}", "content": f"c{i}", "url": ""}
            for i in range(start, start + n)]


def test_dedupe_across_runs_and_day_files(tmp_path):
    feed = {"items": _items(3)}
    src = {"a": lambda: feed["items"]}
    t1 = datetime(2026, 10, 17, 9, 10, tzinfo=cn.BJ)
    assert cn.collect(tmp_path, src, now=t1, log=lambda *a: None) == {"a": "fetched 3 new 3"}
    feed["items"] = _items(4, start=1)                       # 与上次重叠 2 条，新 2 条
    t2 = datetime(2026, 10, 17, 9, 20, tzinfo=cn.BJ)
    assert cn.collect(tmp_path, src, now=t2, log=lambda *a: None) == {"a": "fetched 4 new 2"}
    df = pd.read_parquet(tmp_path / "a/20261017.parquet")
    assert len(df) == 5 and df["item_id"].is_unique
    assert df["first_seen_ts_bj"].iloc[-1] == t2.isoformat() and df["first_seen_ts_bj"].iloc[0] == t1.isoformat()
    t3 = datetime(2026, 10, 18, 0, 5, tzinfo=cn.BJ)          # 跨天：旧条目不再写入新一天的文件
    cn.collect(tmp_path, src, now=t3, log=lambda *a: None)
    assert not (tmp_path / "a/20261018.parquet").exists()
    assert len(json.loads((tmp_path / "_seen/a.json").read_text())) == 5


def test_one_source_failing_or_hanging_does_not_block_other(tmp_path):
    def boom():
        raise ConnectionError("blocked")

    def hang():
        time.sleep(5)
        return _items(1)
    waits = []
    rep = cn.collect(tmp_path, {"bad": boom, "slow": hang, "ok": lambda: _items(2)},
                     now=datetime(2026, 10, 17, tzinfo=cn.BJ), timeout=1, log=lambda *a: None, sleep=waits.append)
    assert rep["bad"].startswith("error ConnectionError") and rep["slow"].startswith("error _TimeoutError")
    assert rep["ok"] == "fetched 2 new 2" and waits == [15, 15]          # 每个失败来源重试一次


def test_transient_error_is_retried(tmp_path):
    n = {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] == 1:
            raise ConnectionError("Remote end closed connection without response")
        return _items(3)
    rep = cn.collect(tmp_path, {"a": flaky}, now=datetime(2026, 10, 17, tzinfo=cn.BJ), log=lambda *a: None,
                     sleep=lambda s: None)
    assert rep == {"a": "fetched 3 new 3"} and n["i"] == 2


def test_beijing_local_time_for_library_fromtimestamp(monkeypatch):
    # 同花顺接口内部用 datetime.fromtimestamp() 按本机时区转换；强制 TZ 后应得到北京时间
    monkeypatch.setenv("TZ", "Australia/Sydney")
    time.tzset()
    try:
        cn.use_beijing_local_time()
        assert datetime.fromtimestamp(0).strftime("%Y-%m-%d %H:%M") == "1970-01-01 08:00"
    finally:
        monkeypatch.undo()
        time.tzset()
