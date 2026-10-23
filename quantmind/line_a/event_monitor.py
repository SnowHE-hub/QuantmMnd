"""quantmind.line_a.event_monitor — 旁路：全市场事件监测（03_final_plan_v2 第 5 节"旁路"；在昂贵计算之前运行）.

两个来源：
- Tushare 结构化事件表：取"公告日或交易日 = as_of"的新记录 → 证券身份就是 ts_code → 规则分级 → 事件队列；
- 巨潮公告元数据（v2，2026-10-23，T-040）：发布日期在 (上一个交易日, as_of] 的公告 → secCode 映射到 ts_code →
  公告分类器（v3，2026-10-24：T-049 采用 MacBERT 多标签，14 类由模型判断，其余类型与阶段仍用标题规则）
  → major / watch 进入队列，routine 不进入。
- 只提高**研究优先级**，不提高投资排名；不在支持域内的股票只进入研究观察。
- 每条事件都带证据编号 `tushare:<表>:<主键…>`，规则编号写在 rule 字段里，便于审计。
分级规则（第一版，阈值都可在这里调整并记录版本）：
- major：业绩预告首亏 / 预减 / 续亏；业绩快报净利润同比 ≤ −50%；审计意见非标准；名称新增 ST / *ST 或"退"；
  单个股东减持比例 ≥ 1%；未来 20 日解禁比例 ≥ 5%；
- watch：其余业绩预告与快报；其他减持与增持；回购；龙虎榜；大宗交易折价 ≥ 5%；股东户数；质押比例 ≥ 30%；停牌。
- 巨潮公告：重要性取各类型的默认值（立案、处罚、退市风险、风险警示、重组、控制权变更、澄清为 major），
  规则编号 C:M:<模型版本> 或 C:<T 规则号>（类型来自规则时）；模型不可用时整批降级到规则，编号仍是 C:<T 规则号>。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

RULE_VERSION = "event_monitor-v3-2026-10-24"
NEG_FORECAST = {"首亏", "预减", "续亏", "略减"}
CNINFO_SUFFIX = (("92", ".BJ"), ("6", ".SH"), ("0", ".SZ"), ("3", ".SZ"), ("4", ".BJ"), ("8", ".BJ"))


@dataclass
class Event:
    ticker: str
    source: str
    event_type: str
    importance: str
    rule: str
    evidence_id: str
    summary: str


def _latest(lake: Path, api: str) -> Path | None:
    fs = sorted((lake / "events").glob(f"{api}_2*.parquet"))
    return fs[-1] if fs else None


def _filter_value(path: Path, col: str, ymd: str):
    import pyarrow.dataset as ds
    typ = str(ds.dataset(str(path), format="parquet").schema.field(col).type)
    if typ.startswith(("timestamp", "date")):
        return pd.Timestamp(ymd)
    return int(ymd) if typ.startswith("int") else ymd


def _on(path: Path | None, col: str, ymd: str) -> pd.DataFrame:
    """只读某一天的行（按日期列的类型构造过滤条件，不把整张表读进内存）。"""
    if path is None:
        return pd.DataFrame()
    return pd.read_parquet(path, filters=[(col, "==", _filter_value(path, col, ymd))])


def _between(path: Path, col: str, lo: str, hi: str) -> pd.DataFrame:
    """读日期列在 (lo, hi] 的行。"""
    return pd.read_parquet(path, filters=[(col, ">", _filter_value(path, col, lo)),
                                          (col, "<=", _filter_value(path, col, hi))])


def scan(data_root: Path, as_of) -> list[Event]:
    lake = Path(data_root) / "data/lake"
    ymd = pd.Timestamp(as_of).strftime("%Y%m%d")
    ev: list[Event] = []
    for _, r in _on(_latest(lake, "forecast"), "ann_date", ymd).iterrows():
        neg = str(r.get("type")) in NEG_FORECAST
        ev.append(Event(r["ts_code"], "forecast", "performance_forecast", "major" if neg else "watch",
                        "E01" if neg else "E02", f"tushare:forecast:{r['ts_code']}:{r['end_date']}:{r['ann_date']}",
                        f"业绩预告 {r.get('type')}，变动 {r.get('p_change_min')}～{r.get('p_change_max')}%"))
    for _, r in _on(_latest(lake, "express"), "ann_date", ymd).iterrows():
        y = r.get("yoy_net_profit")
        bad = y is not None and y == y and float(y) <= -50
        ev.append(Event(r["ts_code"], "express", "performance_express", "major" if bad else "watch",
                        "E03" if bad else "E04", f"tushare:express:{r['ts_code']}:{r['end_date']}:{r['ann_date']}",
                        f"业绩快报，净利润同比 {y}%"))
    for _, r in _on(_latest(lake, "fina_audit"), "ann_date", ymd).iterrows():
        res = str(r.get("audit_result") or "")
        nonstd = res and "标准无保留" not in res
        ev.append(Event(r["ts_code"], "fina_audit", "audit_opinion", "major" if nonstd else "watch",
                        "E05" if nonstd else "E06", f"tushare:fina_audit:{r['ts_code']}:{r['end_date']}:{r['ann_date']}",
                        f"审计意见：{res}"))
    for _, r in _on(_latest(lake, "namechange"), "ann_date", ymd).iterrows():
        nm = str(r.get("name") or "")
        flag = "ST" in nm or "退" in nm
        ev.append(Event(r["ts_code"], "namechange", "st_change" if flag else "name_change", "major" if flag else "watch",
                        "E07" if flag else "E08", f"tushare:namechange:{r['ts_code']}:{r.get('start_date')}",
                        f"证券简称变更为 {nm}"))
    for _, r in _on(_latest(lake, "stk_holdertrade"), "ann_date", ymd).iterrows():
        de = str(r.get("in_de")) == "DE"
        big = de and r.get("change_ratio") == r.get("change_ratio") and float(r.get("change_ratio") or 0) >= 1.0
        ev.append(Event(r["ts_code"], "stk_holdertrade", "holder_reduction" if de else "holder_increase",
                        "major" if big else "watch", "E09" if big else ("E10" if de else "E11"),
                        f"tushare:stk_holdertrade:{r['ts_code']}:{r['ann_date']}:{r.get('holder_name')}",
                        f"{r.get('holder_name')} {'减持' if de else '增持'} {r.get('change_ratio')}%"))
    for _, r in _on(_latest(lake, "repurchase"), "ann_date", ymd).iterrows():
        ev.append(Event(r["ts_code"], "repurchase", "repurchase", "watch", "E12",
                        f"tushare:repurchase:{r['ts_code']}:{r['ann_date']}:{r.get('proc')}", f"回购：{r.get('proc')}"))
    sf = _latest(lake, "share_float")
    if sf is not None:
        hi = (pd.Timestamp(as_of) + pd.Timedelta(days=20)).strftime("%Y%m%d")
        df = _between(sf, "float_date", ymd, hi)
        df = df[df["ann_date"].astype(str).str.replace("-", "").str.slice(0, 8) <= ymd]
        tot = df.groupby("ts_code")["float_ratio"].sum()
        for t, v in tot.items():
            if v >= 5:
                ev.append(Event(t, "share_float", "unlock", "major", "E13", f"tushare:share_float:{t}:{ymd}:next20d",
                                f"未来 20 日解禁合计 {v:.2f}%"))
    for _, r in _on(_latest(lake, "top_list"), "trade_date", ymd).drop_duplicates("ts_code").iterrows():
        ev.append(Event(r["ts_code"], "top_list", "dragon_tiger", "watch", "E14",
                        f"tushare:top_list:{r['ts_code']}:{ymd}", f"龙虎榜：{r.get('reason')}"))
    bt = _on(_latest(lake, "block_trade"), "trade_date", ymd)
    if len(bt):
        from quantmind.line_a.f1_instruments import _rows_on
        px = _rows_on(lake / "daily.parquet", "trade_date", ymd, ["ts_code", "close"]).set_index("ts_code")["close"]
        for t, g in bt.groupby("ts_code"):
            c = px.get(t)
            if c:
                disc = float((g["price"] * g["vol"]).sum() / g["vol"].sum() / c - 1)
                if disc <= -0.05:
                    ev.append(Event(t, "block_trade", "block_trade_discount", "watch", "E15",
                                    f"tushare:block_trade:{t}:{ymd}", f"大宗交易成交均价折价 {disc:.1%}"))
    for _, r in _on(_latest(lake, "suspend_d"), "trade_date", ymd).iterrows():
        if str(r.get("suspend_type")) == "S":
            ev.append(Event(r["ts_code"], "suspend_d", "suspension", "watch", "E16",
                            f"tushare:suspend_d:{r['ts_code']}:{ymd}", "当日停牌"))
    return ev + scan_cninfo(lake, as_of)


def cninfo_ts_code(sec_code) -> str | None:
    c = str(sec_code).strip().zfill(6)
    for pre, suf in CNINFO_SUFFIX:
        if c.startswith(pre):
            return c + suf
    return None


def cninfo_window(lake: Path, as_of) -> list[str]:
    """(上一个交易日, as_of] 的自然日（YYYYMMDD）；周末和节假日的公告归到下一个交易日。"""
    d = pd.Timestamp(as_of).normalize()
    cal = pd.read_parquet(lake / "trade_cal_sse.parquet")
    if "is_open" in cal:
        cal = cal[cal["is_open"].astype(int) == 1]
    opens = pd.to_datetime(cal["cal_date"].astype(str)).sort_values()
    prev = opens[opens < d]
    start = prev.iloc[-1] + pd.Timedelta(days=1) if len(prev) else d
    return [x.strftime("%Y%m%d") for x in pd.date_range(start, d, freq="D")]


def scan_cninfo(lake: Path, as_of, universe: set[str] | None = None) -> list[Event]:
    from quantmind.line_a.evidence.ann_classifier import classify_titles
    parts = Path(lake) / "text/cninfo/_parts"
    files = [f for f in (parts / f"{ymd}.parquet" for ymd in cninfo_window(lake, as_of)) if f.exists()]
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f, columns=["announcementId", "secCode", "announcementTitle"]) for f in files])
    df = df.drop_duplicates("announcementId")
    if universe is None:
        sb = sorted(Path(lake).glob("stock_basic_full_merged_2*.parquet"))
        universe = set(pd.read_parquet(sb[-1], columns=["ts_code"])["ts_code"]) if sb else None
    df["ticker"] = df["secCode"].map(cninfo_ts_code)
    df = df[df["ticker"].notna()]
    if universe is not None:
        df = df[df["ticker"].isin(universe)]
    ev: list[Event] = []
    for aid, t, title, c in zip(df["announcementId"], df["ticker"], df["announcementTitle"],
                                classify_titles(df["announcementTitle"].astype(str).tolist()), strict=True):
        if c.importance == "routine" or c.is_summary_or_meta or c.event_type == "other":
            continue
        ev.append(Event(t, "cninfo", c.event_type, c.importance, f"C:{c.type_rule}", f"cninfo:ann:{aid}",
                        str(title)[:80]))
    return ev


def event_queue(events: list[Event]) -> pd.DataFrame:
    """按股票聚合：最高重要性、事件条数、规则与证据编号；送入 F4 的事件队列。"""
    if not events:
        return pd.DataFrame(columns=["ticker", "importance", "n_events", "rules", "evidence_ids", "summaries"])
    df = pd.DataFrame([asdict(e) for e in events])
    rank = {"major": 2, "watch": 1}
    df["r"] = df["importance"].map(rank)
    g = df.groupby("ticker")
    out = pd.DataFrame({"importance": g["r"].max().map({2: "major", 1: "watch"}), "n_events": g.size(),
                        "rules": g["rule"].agg(lambda s: ",".join(sorted(set(s)))),
                        "evidence_ids": g["evidence_id"].agg(list), "summaries": g["summary"].agg(list)})
    out["rule_version"] = RULE_VERSION
    return out.reset_index().sort_values(["importance", "n_events"], ascending=[True, False])
