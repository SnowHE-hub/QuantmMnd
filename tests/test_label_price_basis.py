"""T-003：forward_return 标签必须使用单一一致的价格基准.

合成场景：一只票在两个快照之间发生 2:1 拆股。
- 快照 A（拆股前生成）里的 close 以 A 的日期为基准前复权 → 拆股前价格保持 10。
- 快照 B（拆股后生成）里的 close 以 B 的日期为基准前复权 → 拆股前价格被调成 5。
- 原始价格从头到尾没有真实涨跌（pct_change 恒为 0），真实标签应为 0。
旧实现把两个快照的 close 拼成宽表，基准日落在 A、目标日落在 B，得到 −50% 的错误标签；
新实现用 pct_change 连乘的收益指数，得到 0。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantmind.features.panel import (
    build_return_index_pivot,
    compute_forward_returns,
    splice_return_index,
)
from scripts.build_full_panel import (
    _price_pivot_from_snapshots,
    _relabel_panel,
    _return_index_from_snapshots,
)

DAYS = pd.bdate_range("2024-01-02", periods=40)
SPLIT_DAY = DAYS[25]


def _snapshot_frame(days: pd.DatetimeIndex, snapshot_day: pd.Timestamp) -> pd.DataFrame:
    """'AAA' 原始价拆股前 10、拆股后 5；按 snapshot_day 为基准前复权."""
    raw = np.where(days >= SPLIT_DAY, 5.0, 10.0)
    factor = 0.5 if snapshot_day >= SPLIT_DAY else 1.0  # 基准日已拆股 → 拆股前价格 ×0.5
    qfq = np.where(days >= SPLIT_DAY, raw, raw * factor)
    aaa = pd.DataFrame({"trade_date": days, "ticker": "AAA", "close": qfq, "pct_change": 0.0})
    # 'BBB' 每日 +1%，无公司行动，两个快照完全一致
    k = np.array([DAYS.get_loc(d) for d in days])
    bbb = pd.DataFrame({"trade_date": days, "ticker": "BBB", "close": 20.0 * 1.01 ** k, "pct_change": 1.0})
    return pd.concat([aaa, bbb], ignore_index=True)


@pytest.fixture()
def snapshot_root(tmp_path: Path) -> Path:
    root = tmp_path / "snapshots"
    snaps = {
        DAYS[19]: DAYS[:20],   # 快照 A：拆股前生成
        DAYS[39]: DAYS[10:],   # 快照 B：拆股后生成，与 A 有重叠
    }
    for snap_day, days in snaps.items():
        d = root / snap_day.date().isoformat()
        d.mkdir(parents=True)
        _snapshot_frame(days, snap_day).to_parquet(d / "prices.parquet")
        (d / "meta.json").write_text(json.dumps({"files": ["prices.parquet"]}))
    return root


def test_old_level_pivot_gives_wrong_label(snapshot_root: Path):
    """反例：旧的 close 拼接在跨快照的除权日上产生 −50% 的假标签."""
    dates_kept = [date.fromisoformat(p.name) for p in sorted(snapshot_root.iterdir())]
    pivot = _price_pivot_from_snapshots(snapshot_root, dates_kept)
    fr = compute_forward_returns(pivot, DAYS[5].date(), (21,))
    assert fr.loc["AAA", "forward_return_21d"] == pytest.approx(-0.5)


def test_return_index_gives_true_label(snapshot_root: Path):
    pivot = _return_index_from_snapshots(snapshot_root)
    fr = compute_forward_returns(pivot, DAYS[5].date(), (21,))
    assert fr.loc["AAA", "forward_return_21d"] == pytest.approx(0.0)
    assert fr.loc["BBB", "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)


def test_suspension_keeps_nan_semantics():
    """目标日无价 → NaN；停牌日不贡献收益，复牌日的 pct_change 相对停牌前收盘."""
    days = pd.bdate_range("2024-01-02", periods=6)
    long = pd.DataFrame({
        "trade_date": list(days[[0, 1, 4, 5]]),
        "ticker": "CCC",
        "close": [10.0, 11.0, 12.1, 12.1],
        "pct_change": [0.0, 10.0, 10.0, 0.0],
    })
    long = pd.concat([long, pd.DataFrame({
        "trade_date": days, "ticker": "DDD", "close": 1.0, "pct_change": 0.0,
    })], ignore_index=True)
    idx = build_return_index_pivot(long)
    assert np.isnan(idx.loc[days[2], "CCC"]) and np.isnan(idx.loc[days[3], "CCC"])
    fr = compute_forward_returns(idx, days[0].date(), (2, 4))
    assert np.isnan(fr.loc["CCC", "forward_return_2d"])           # 目标日停牌
    assert fr.loc["CCC", "forward_return_4d"] == pytest.approx(0.21)


def test_missing_pct_change_is_an_error():
    long = pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-02", periods=2), "ticker": "EEE",
        "close": [1.0, 1.1], "pct_change": [0.0, np.nan],
    })
    with pytest.raises(ValueError, match="no pct_change"):
        build_return_index_pivot(long)


def test_splice_uses_returns_not_levels():
    """日线长表的 adj_close 基准与指数不同（差 100 倍），衔接后只继承日收益."""
    head_days = pd.bdate_range("2024-01-02", periods=3)
    tail_days = pd.bdate_range(head_days[-1], periods=4)
    index_pivot = pd.DataFrame({"FFF": [1.0, 1.1, 1.21]}, index=head_days)
    level = pd.DataFrame({"FFF": [121.0, 133.1, np.nan, 146.41]}, index=tail_days)
    out = splice_return_index(index_pivot, level)
    assert list(out.index) == list(head_days) + list(tail_days[1:])
    assert out.loc[tail_days[1], "FFF"] == pytest.approx(1.331)
    assert np.isnan(out.loc[tail_days[2], "FFF"])                  # 停牌
    assert out.loc[tail_days[3], "FFF"] == pytest.approx(1.4641)   # 相对停牌前收盘


def _tiny_panel() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp(DAYS[5])], ["AAA", "BBB"]], names=["as_of", "ticker"],
    )
    return pd.DataFrame({
        "pe_ttm": [0.3, -0.3],
        "exposure_industry": pd.array(["银行", pd.NA], dtype="string"),
        "forward_return_21d": [-0.5, 0.2],
        "forward_return_63d": [np.nan, np.nan],
    }, index=idx)


def test_relabel_rewrites_only_labels(snapshot_root: Path, tmp_path: Path):
    src = tmp_path / "panel.parquet"
    out = tmp_path / "panel_relabeled.parquet"
    _tiny_panel().to_parquet(src)
    panel = pd.read_parquet(src)
    rc = _relabel_panel(src, out, snapshot_root=snapshot_root,
                        raw_daily=tmp_path / "absent.parquet", through=None)
    assert rc == 0
    new = pd.read_parquet(out)
    pd.testing.assert_frame_equal(
        new.drop(columns=["forward_return_21d", "forward_return_63d"]),
        panel.drop(columns=["forward_return_21d", "forward_return_63d"]),
    )
    assert new.loc[(pd.Timestamp(DAYS[5]), "AAA"), "forward_return_21d"] == pytest.approx(0.0)
    assert np.isnan(new.loc[(pd.Timestamp(DAYS[5]), "AAA"), "forward_return_63d"])


def test_relabel_refuses_to_overwrite(snapshot_root: Path, tmp_path: Path):
    src = tmp_path / "panel.parquet"
    _tiny_panel().to_parquet(src)
    kw = {"snapshot_root": snapshot_root, "raw_daily": tmp_path / "absent.parquet", "through": None}
    assert _relabel_panel(src, src, **kw) == 1
    existing = tmp_path / "existing.parquet"
    existing.write_bytes(b"keep")
    assert _relabel_panel(src, existing, **kw) == 1
    assert existing.read_bytes() == b"keep"


def test_price_through_caps_label_availability(snapshot_root: Path, tmp_path: Path):
    src = tmp_path / "panel.parquet"
    out = tmp_path / "capped.parquet"
    _tiny_panel().to_parquet(src)
    rc = _relabel_panel(src, out, snapshot_root=snapshot_root,
                        raw_daily=tmp_path / "absent.parquet", through=pd.Timestamp(DAYS[20]))
    assert rc == 0
    new = pd.read_parquet(out)
    assert new["forward_return_21d"].isna().all()   # 目标日 DAYS[26] 超出 price-through


def test_daily_panel_rows_fill_snapshot_gaps(snapshot_root: Path, tmp_path: Path):
    """快照没有的票（GGG）由日线长表的 pct_chg 补齐；快照已有的行以快照为准."""
    from scripts.build_full_panel import _daily_panel_long

    raw = pd.DataFrame({
        "trade_date": DAYS, "ts_code": "GGG", "close": 3.0 * 1.02 ** np.arange(len(DAYS)),
        "pct_chg": 2.0, "adj_close": 999.0,
    })
    # 同日同票的冲突行：快照里 AAA 的 pct_change 是 0，日线里故意写 50
    raw = pd.concat([raw, pd.DataFrame({
        "trade_date": DAYS, "ts_code": "AAA", "close": 1.0, "pct_chg": 50.0, "adj_close": 1.0,
    })], ignore_index=True)
    raw_path = tmp_path / "daily.parquet"
    raw.to_parquet(raw_path)
    pivot = _return_index_from_snapshots(snapshot_root, extra_long=_daily_panel_long(raw_path))
    fr = compute_forward_returns(pivot, DAYS[5].date(), (21,))
    assert fr.loc["GGG", "forward_return_21d"] == pytest.approx(1.02 ** 21 - 1)
    assert fr.loc["AAA", "forward_return_21d"] == pytest.approx(0.0)


def test_keep_label_mask_preserves_sample_set(snapshot_root: Path, tmp_path: Path):
    src = tmp_path / "panel.parquet"
    out = tmp_path / "masked.parquet"
    panel = _tiny_panel()
    panel.loc[(pd.Timestamp(DAYS[5]), "BBB"), "forward_return_21d"] = np.nan
    panel.to_parquet(src)
    rc = _relabel_panel(src, out, snapshot_root=snapshot_root,
                        raw_daily=tmp_path / "absent.parquet", through=None, keep_label_mask=True)
    assert rc == 0
    new = pd.read_parquet(out)
    assert np.isnan(new.loc[(pd.Timestamp(DAYS[5]), "BBB"), "forward_return_21d"])
    assert new.loc[(pd.Timestamp(DAYS[5]), "AAA"), "forward_return_21d"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 补充修复：某票不在某快照的票池里 ≠ 停牌（2026-10-10 独立核验发现）
# ---------------------------------------------------------------------------
@pytest.fixture()
def gap_root(tmp_path: Path) -> Path:
    """快照 A 含 AAA、CCC（DAYS[0..19]）；快照 B 只含 AAA（DAYS[10..38]）；快照 C 含 AAA、CCC（DAYS[25..39]）。
    CCC 实际每日 +1% 一直在交易，但 DAYS[20..24] 没有任何快照覆盖它（真实案例：2025 年初部分快照只含约 300 只票）。"""
    root = tmp_path / "gap_snapshots"

    def frame(days, tickers):
        parts = []
        for t in tickers:
            k = np.array([DAYS.get_loc(d) for d in days])
            parts.append(pd.DataFrame({"trade_date": days, "ticker": t,
                                       "close": 10.0 * 1.01 ** k, "pct_change": 1.0}))
        return pd.concat(parts, ignore_index=True)

    for snap_day, days, tickers in ((DAYS[19], DAYS[:20], ["AAA", "CCC"]), (DAYS[38], DAYS[10:39], ["AAA"]),
                                    (DAYS[39], DAYS[25:], ["AAA", "CCC"])):
        d = root / snap_day.date().isoformat()
        d.mkdir(parents=True)
        frame(days, tickers).to_parquet(d / "prices.parquet")
        (d / "meta.json").write_text(json.dumps({"files": ["prices.parquet"]}))
    return root


def _gap_panel(tmp_path: Path) -> Path:
    idx = pd.MultiIndex.from_product([[pd.Timestamp(DAYS[5])], ["AAA", "CCC"]], names=["as_of", "ticker"])
    p = tmp_path / "gap_panel.parquet"
    pd.DataFrame({"pe_ttm": [0.0, 0.0], "forward_return_21d": [np.nan, np.nan],
                  "forward_return_63d": [np.nan, np.nan]}, index=idx).to_parquet(p)
    return p


def test_missing_coverage_is_not_treated_as_suspension(gap_root: Path, tmp_path: Path):
    """CCC 的 21 日窗口 (DAYS[5], DAYS[26]] 跨过没有数据源覆盖它的 DAYS[20..24]：标签必须是 NaN，
    而不是把缺失的 5 天当停牌、只累计 16 天收益。"""
    pivot = _return_index_from_snapshots(gap_root)
    wrong = compute_forward_returns(pivot, DAYS[5].date(), (21,)).loc["CCC", "forward_return_21d"]
    assert wrong == pytest.approx(1.01 ** 16 - 1)       # 未加覆盖检查时的错误值：缺失的 5 天被当成停牌

    out = tmp_path / "gap_out.parquet"
    rc = _relabel_panel(_gap_panel(tmp_path), out, snapshot_root=gap_root,
                        raw_daily=tmp_path / "absent.parquet", through=None)
    assert rc == 0
    new = pd.read_parquet(out)
    assert np.isnan(new.loc[(pd.Timestamp(DAYS[5]), "CCC"), "forward_return_21d"])
    assert new.loc[(pd.Timestamp(DAYS[5]), "AAA"), "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)


def test_complete_daily_source_fills_the_gap(gap_root: Path, tmp_path: Path):
    """补充一份完整的日线长表（含 pct_chg）后，CCC 的缺口被真实数据填上，标签正确."""
    k = np.arange(len(DAYS))
    raw = pd.DataFrame({"trade_date": DAYS, "ts_code": "CCC", "close": 10.0 * 1.01 ** k, "pct_chg": 1.0})
    raw_path = tmp_path / "daily.parquet"
    raw.to_parquet(raw_path)
    out = tmp_path / "gap_out2.parquet"
    rc = _relabel_panel(_gap_panel(tmp_path), out, snapshot_root=gap_root, raw_daily=raw_path, through=None)
    assert rc == 0
    new = pd.read_parquet(out)
    assert new.loc[(pd.Timestamp(DAYS[5]), "CCC"), "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)


def test_true_suspension_inside_coverage_is_kept(tmp_path: Path):
    """票在数据源覆盖范围内、只是某几天无行：这是停牌，标签照算（与旧语义一致）."""
    root = tmp_path / "susp"
    d = root / DAYS[39].date().isoformat()
    d.mkdir(parents=True)
    days = DAYS.delete([8, 9])          # DDD 在 DAYS[8]、DAYS[9] 停牌；EEE 每天交易，撑起完整日历
    k = np.array([DAYS.get_loc(x) for x in days])
    ddd = pd.DataFrame({"trade_date": days, "ticker": "DDD", "close": 10.0 * 1.01 ** k,
                        "pct_change": np.where(k == 10, (1.01 ** 3 - 1) * 100, 1.0)})
    eee = pd.DataFrame({"trade_date": DAYS, "ticker": "EEE", "close": 1.0, "pct_change": 0.0})
    pd.concat([ddd, eee]).to_parquet(d / "prices.parquet")
    (d / "meta.json").write_text(json.dumps({"files": ["prices.parquet"]}))
    # 面板里同时有 EEE：relabel 只读面板内的票，交易日历由这些票的行构成
    idx = pd.MultiIndex.from_product([[pd.Timestamp(DAYS[5])], ["DDD", "EEE"]], names=["as_of", "ticker"])
    src = tmp_path / "p.parquet"
    pd.DataFrame({"x": [0.0, 0.0], "forward_return_21d": [np.nan, np.nan],
                  "forward_return_63d": [np.nan, np.nan]}, index=idx).to_parquet(src)
    out = tmp_path / "o.parquet"
    assert _relabel_panel(src, out, snapshot_root=root, raw_daily=tmp_path / "absent.parquet", through=None) == 0
    new = pd.read_parquet(out)
    assert new.loc[(pd.Timestamp(DAYS[5]), "DDD"), "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)


# ---------------------------------------------------------------------------
# 重验修复（2026-10-11，回应 Codex REJECT）：交易日轴不得依赖面板票池；来源覆盖按实际有数据的日期
# ---------------------------------------------------------------------------
ABSENT = Path("/nonexistent/daily.parquet")


def _write_snapshot(root: Path, snap_day: pd.Timestamp, frame: pd.DataFrame) -> None:
    d = root / snap_day.date().isoformat()
    d.mkdir(parents=True)
    frame.to_parquet(d / "prices.parquet")
    (d / "meta.json").write_text(json.dumps({"files": ["prices.parquet"]}))


def _steady(ticker: str, days: pd.DatetimeIndex, pct: float = 1.0) -> pd.DataFrame:
    """每日 +pct%，close 按 DAYS 上的位置定价（停牌日不在 days 里时，复牌日的 pct_change 补足停牌期间的涨幅）."""
    k = np.array([DAYS.get_loc(x) for x in days])
    gap = np.diff(np.concatenate([[k[0] - 1], k]))
    return pd.DataFrame({"trade_date": days, "ticker": ticker, "close": 10.0 * (1 + pct / 100) ** k,
                         "pct_change": ((1 + pct / 100) ** gap - 1) * 100})


def _panel(tmp_path: Path, tickers: list[str], as_ofs=(5,)) -> Path:
    idx = pd.MultiIndex.from_product([[pd.Timestamp(DAYS[i]) for i in as_ofs], tickers], names=["as_of", "ticker"])
    p = tmp_path / f"panel_{'_'.join(tickers)}.parquet"
    pd.DataFrame({"x": 0.0, "forward_return_21d": np.nan, "forward_return_63d": np.nan}, index=idx).to_parquet(p)
    return p


def _calendar(tmp_path: Path, days: pd.DatetimeIndex = DAYS) -> Path:
    p = tmp_path / "trade_cal.parquet"
    pd.DataFrame({"cal_date": days}).to_parquet(p)
    return p


def _relabel(tmp_path, src, root, name, **kw) -> pd.DataFrame:
    out = tmp_path / f"{name}.parquet"
    assert _relabel_panel(src, out, snapshot_root=root, raw_daily=kw.pop("raw_daily", ABSENT), through=None, **kw) == 0
    return pd.read_parquet(out)


@pytest.mark.parametrize("with_calendar", [False, True])
def test_all_panel_tickers_suspended_does_not_shift_horizon(tmp_path: Path, with_calendar: bool):
    """Codex 反例：面板只有 DDD，DDD 在 DAYS[8]、DAYS[9] 停牌，快照里的 EEE 每天交易。
    目标日必须是完整交易日历上基准日后第 21 天（DAYS[26]），不能因为面板票全停牌就把这两天从日历上删掉。"""
    root = tmp_path / "s"
    _write_snapshot(root, DAYS[39], pd.concat([_steady("DDD", DAYS.delete([8, 9])), _steady("EEE", DAYS, 0.0)]))
    # 旧行为：按面板票过滤后再建日历 → 目标日后移 2 天
    only_ddd = build_return_index_pivot(_steady("DDD", DAYS.delete([8, 9])))
    wrong = compute_forward_returns(only_ddd, DAYS[5].date(), (21,)).loc["DDD", "forward_return_21d"]
    assert wrong == pytest.approx(1.01 ** 23 - 1)
    kw = {"trade_calendar": _calendar(tmp_path)} if with_calendar else {}
    new = _relabel(tmp_path, _panel(tmp_path, ["DDD"]), root, "o", **kw)
    assert new.loc[(pd.Timestamp(DAYS[5]), "DDD"), "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)


def test_target_day_suspension_gives_nan(tmp_path: Path):
    """目标日 DAYS[26] DDD 停牌（其他票交易）→ 标签 NaN（"目标日无价 → NaN"的既有语义）。"""
    root = tmp_path / "s"
    _write_snapshot(root, DAYS[39], pd.concat([_steady("DDD", DAYS.delete([26])), _steady("EEE", DAYS)]))
    new = _relabel(tmp_path, _panel(tmp_path, ["DDD"]), root, "o", trade_calendar=_calendar(tmp_path))
    assert np.isnan(new.loc[(pd.Timestamp(DAYS[5]), "DDD"), "forward_return_21d"])


def test_source_missing_a_whole_day_is_unknown_not_suspension(tmp_path: Path):
    """唯一来源整日缺 DAYS[12]（所有票都没有行），独立日历里有这一天：窗口跨过它的标签 → NaN（未知），
    窗口不含它的标签照算；补一份覆盖这一天的日线来源后，标签恢复。"""
    root = tmp_path / "s"
    full = pd.concat([_steady("AAA", DAYS), _steady("BBB", DAYS)])
    _write_snapshot(root, DAYS[39], full[full["trade_date"] != DAYS[12]])   # 丢行：次日 pct_change 仍是单日收益
    src = _panel(tmp_path, ["AAA"], as_ofs=(5, 13))
    cal = _calendar(tmp_path)
    new = _relabel(tmp_path, src, root, "o1", trade_calendar=cal)
    assert np.isnan(new.loc[(pd.Timestamp(DAYS[5]), "AAA"), "forward_return_21d"])       # 窗口含 DAYS[12]
    assert new.loc[(pd.Timestamp(DAYS[13]), "AAA"), "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)
    raw = _steady("AAA", DAYS).rename(columns={"ticker": "ts_code", "pct_change": "pct_chg"})
    raw.to_parquet(tmp_path / "daily.parquet")
    new2 = _relabel(tmp_path, src, root, "o2", trade_calendar=cal, raw_daily=tmp_path / "daily.parquet")
    assert new2.loc[(pd.Timestamp(DAYS[5]), "AAA"), "forward_return_21d"] == pytest.approx(1.01 ** 21 - 1)


def test_ticker_pool_change_between_sources(tmp_path: Path):
    """快照 1（DAYS[0..19]）含 AAA、BBB；快照 2（DAYS[15..39]）含 AAA、CCC。
    BBB 离开票池后没有来源覆盖 → 跨出去的窗口是未知；CCC 进入票池前没有来源覆盖 → 无基准价；AAA 不受影响。"""
    root = tmp_path / "s"
    _write_snapshot(root, DAYS[19], pd.concat([_steady("AAA", DAYS[:20]), _steady("BBB", DAYS[:20])]))
    _write_snapshot(root, DAYS[39], pd.concat([_steady("AAA", DAYS[15:]), _steady("CCC", DAYS[15:])]))
    new = _relabel(tmp_path, _panel(tmp_path, ["AAA", "BBB", "CCC"], as_ofs=(5, 16)), root, "o",
                   trade_calendar=_calendar(tmp_path))

    def at(i: int, t: str) -> float:
        return new.loc[(pd.Timestamp(DAYS[i]), t), "forward_return_21d"]
    assert at(5, "AAA") == pytest.approx(1.01 ** 21 - 1) and at(16, "AAA") == pytest.approx(1.01 ** 21 - 1)
    assert np.isnan(at(5, "BBB")) and np.isnan(at(16, "BBB"))
    assert np.isnan(at(5, "CCC")) and at(16, "CCC") == pytest.approx(1.01 ** 21 - 1)


def test_source_day_outside_trade_calendar_is_an_error(tmp_path: Path):
    root = tmp_path / "s"
    _write_snapshot(root, DAYS[39], _steady("AAA", DAYS))
    with pytest.raises(ValueError, match="not trading days"):
        _relabel_panel(_panel(tmp_path, ["AAA"]), tmp_path / "o.parquet", snapshot_root=root, raw_daily=ABSENT,
                       through=None, trade_calendar=_calendar(tmp_path, DAYS.delete([30])))
