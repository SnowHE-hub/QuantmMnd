"""quantmind.line_a.harness.tools_data — A1（财务）与 A2（公告事件）角色的只读数据工具（P2 两个垂直切片）.

所有工具都：
- 只读数据湖；
- 只返回 as_of 当天或之前已经可用的记录（财报按公告日 ann_date，公告按发布时间）；
- 给每条记录一个稳定的证据编号 evidence_id，格式为 `<来源>:<表>:<主键…>`。校验器用这个编号回到原始记录核对数字。

返回值都是可 JSON 序列化的字典，数值保留原单位（Tushare 财报金额单位是元）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from quantmind.line_a.evidence.ann_classifier import classify_titles
from quantmind.line_a.evidence.ann_types import subject_tag
from quantmind.line_a.evidence.event_chain import Ann, state_as_of

FIN_FIELDS = {
    "income": ["total_revenue", "revenue", "n_income", "n_income_attr_p", "operate_profit"],
    "balancesheet": ["total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "total_cur_assets", "total_cur_liab"],
    "cashflow": ["n_cashflow_act", "c_paid_invest", "c_pay_acq_const_fiolta", "free_cashflow"],
    "fina_indicator_full": ["roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "current_ratio",
                            "or_yoy", "netprofit_yoy", "dt_netprofit_yoy"],
}


# 每个字段的单位（Tushare 口径）：财报金额为元，业绩预告净利润为万元，比率与同比为 %
UNITS = {"income": "元", "balancesheet": "元", "cashflow": "元", "fina_indicator_full": "%"}
FORECAST_UNITS = {"p_change_min": "%", "p_change_max": "%", "net_profit_min": "万元", "net_profit_max": "万元"}
EXPRESS_UNITS = {"revenue": "元", "n_income": "元", "yoy_net_profit": "%", "yoy_sales": "%", "diluted_roe": "%"}


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _rows(path: Path, ticker: str, columns: list[str] | None = None) -> pd.DataFrame:
    dset = ds.dataset(str(path), format="parquet")
    cols = None if columns is None else [c for c in columns if c in dset.schema.names]
    return dset.to_table(filter=ds.field("ts_code") == ticker, columns=cols).to_pandas()


def _latest(lake: Path, api: str) -> Path | None:
    fs = sorted((lake / "events").glob(f"{api}_2*.parquet"))
    return fs[-1] if fs else None


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _disp(v, unit: str | None) -> str | None:
    """金额的可读写法，由代码换算（T-051：小模型自己换算单位常错一个数量级）。"""
    if v is None or not isinstance(v, int | float) or isinstance(v, bool):
        return None
    if unit == "元":
        return f"{v / 1e8:.2f}亿元" if abs(v) >= 1e8 else (f"{v / 1e4:.2f}万元" if abs(v) >= 1e4 else f"{v:.2f}元")
    if unit == "万元":
        return f"{v / 1e4:.2f}亿元" if abs(v) >= 1e4 else f"{v:.2f}万元"
    return None


def _with_display(rec: dict, units: dict) -> dict:
    d = {k: x for k, u in units.items() if (x := _disp(rec.get(k), u)) is not None}
    return rec | ({"_display": d} if d else {})


@dataclass
class DataTools:
    lake: Path                       # data/lake
    as_of: str                       # YYYYMMDD；所有工具的时点上限
    pdf_dir: Path | None = None      # 已取得的公告 PDF 缓存目录
    display_amounts: bool = False    # 金额另附 _display 可读写法（T-051 harness v3；判定之前默认关闭）

    # ------------------------------------------------------------------ A1 财务
    def financial_snapshot(self, ticker: str, periods: int = 8) -> dict:
        out: dict = {"ticker": ticker, "as_of": self.as_of, "tables": {}}
        for table, fields in FIN_FIELDS.items():
            df = _rows(self.lake / f"{table}.parquet", ticker, ["ts_code", "ann_date", "end_date", *fields])
            df = df[df["ann_date"].astype(str) <= self.as_of]
            # 同一报告期多次披露（更正）：保留全部版本，按公告日排序，最新的在前
            df = df.sort_values(["end_date", "ann_date"], ascending=[False, False])
            keep_periods = list(dict.fromkeys(df["end_date"].astype(str)))[:periods]
            df = df[df["end_date"].astype(str).isin(keep_periods)]
            out["tables"][table] = [
                {"evidence_id": f"tushare:{table}:{ticker}:{r['end_date']}:{r['ann_date']}",
                 "end_date": str(r["end_date"]), "ann_date": str(r["ann_date"]),
                 **{f: _clean(r[f]) for f in fields if f in r},
                 "_units": dict.fromkeys(fields, UNITS[table])} for _, r in df.iterrows()]
            if self.display_amounts:
                out["tables"][table] = [_with_display(x, x["_units"]) for x in out["tables"][table]]
        return out

    def forecasts_and_express(self, ticker: str, lookback_days: int = 400) -> dict:
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        res: dict = {"ticker": ticker, "as_of": self.as_of}
        for api, cols in (("forecast", ["ann_date", "end_date", "type", "p_change_min", "p_change_max",
                                        "net_profit_min", "net_profit_max", "summary", "change_reason"]),
                          ("express", ["ann_date", "end_date", "revenue", "n_income", "yoy_net_profit",
                                       "yoy_sales", "diluted_roe"])):
            p = _latest(self.lake, api)
            if p is None:
                res[api] = {"status": "source_absent"}
                continue
            df = _rows(p, ticker)
            a = df["ann_date"].astype(str)
            df = df[(a > lo) & (a <= self.as_of)].sort_values("ann_date", ascending=False)
            units = FORECAST_UNITS if api == "forecast" else EXPRESS_UNITS
            res[api] = [{"evidence_id": f"tushare:{api}:{ticker}:{r['end_date']}:{r['ann_date']}",
                         **{c: _clean(r[c]) for c in cols if c in r}, "_units": units} for _, r in df.iterrows()]
            if self.display_amounts:
                res[api] = [_with_display(x, units) for x in res[api]]
        return res

    def disclosure_schedule(self, ticker: str) -> dict:
        p = _latest(self.lake, "disclosure_date")
        if p is None:
            return {"status": "source_absent"}
        df = _rows(p, ticker)
        df = df[df["ann_date"].astype(str) <= self.as_of].sort_values("end_date", ascending=False).head(4)
        recs = []
        for _, r in df.iterrows():
            act = str(r.get("actual_date") or "")
            recs.append({"evidence_id": f"tushare:disclosure_date:{ticker}:{r['end_date']}",
                         "end_date": str(r["end_date"]), "pre_date": _clean(r.get("pre_date")),
                         "actual_date": act if act and act <= self.as_of else None,     # 未来的实际日期不可见
                         "modify_date": _clean(r.get("modify_date"))})
        return {"ticker": ticker, "as_of": self.as_of, "schedule": recs}

    def audit_and_pledge(self, ticker: str) -> dict:
        res: dict = {"ticker": ticker, "as_of": self.as_of}
        p = _latest(self.lake, "fina_audit")
        if p is not None:
            df = _rows(p, ticker)
            df = df[df["ann_date"].astype(str) <= self.as_of].sort_values("end_date", ascending=False).head(3)
            res["audit"] = [{"evidence_id": f"tushare:fina_audit:{ticker}:{r['end_date']}:{r['ann_date']}",
                             "end_date": str(r["end_date"]), "audit_result": _clean(r.get("audit_result")),
                             "audit_agency": _clean(r.get("audit_agency"))} for _, r in df.iterrows()]
        else:
            res["audit"] = {"status": "source_absent"}
        p = _latest(self.lake, "pledge_stat")
        if p is not None:
            df = _rows(p, ticker)
            df = df[df["end_date"].astype(str) <= self.as_of].sort_values("end_date", ascending=False).head(4)
            res["pledge"] = [{"evidence_id": f"tushare:pledge_stat:{ticker}:{r['end_date']}",
                              "end_date": str(r["end_date"]), "pledge_count": _clean(r.get("pledge_count")),
                              "pledge_ratio": _clean(r.get("pledge_ratio")),
                              "_units": {"pledge_ratio": "%", "pledge_count": "笔"}} for _, r in df.iterrows()]
        else:
            res["pledge"] = {"status": "source_absent"}
        return res

    # ------------------------------------------------------------------ A2 公告事件
    def _announcements(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        code = ticker.split(".")[0]
        parts = [p for p in sorted((self.lake / "text/cninfo/_parts").glob("*.parquet")) if lo <= p.stem <= self.as_of]
        if not parts:
            return pd.DataFrame()
        rows = [ds.dataset(str(p), format="parquet").to_table(filter=ds.field("secCode") == code).to_pandas()
                for p in parts]
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if df.empty:
            return df
        df["available_at"] = df["publish_ts_bj"].astype(str)
        df = df[df["available_at"].str.slice(0, 10).str.replace("-", "") <= self.as_of]
        self._remember(df)
        return df

    def list_announcements(self, ticker: str, lookback_days: int = 180, limit: int = 120,
                           event_types: list[str] | None = None, min_importance: str = "routine") -> dict:
        """按时点列出公告。可按事件类型和最低重要性过滤；结果超过 limit 时明确标出截断和总数。"""
        df = self._announcements(ticker, lookback_days)
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        meta = {"ticker": ticker, "as_of": self.as_of, "window": [lo, self.as_of]}
        if df.empty:
            return {**meta, "announcements": [], "total": 0, "truncated": False, "coverage_note":
                    "巨潮元数据在该区间没有记录（可能是采集尚未覆盖，不能据此推断没有公告）"}
        rank = {"routine": 0, "watch": 1, "major": 2}
        rows = []
        df = df.sort_values("available_at", ascending=False)
        for (_, r), c in zip(df.iterrows(), classify_titles(df["announcementTitle"].astype(str).tolist()), strict=True):
            if event_types and not set(c.types) & set(event_types):
                continue
            if rank.get(c.importance, 0) < rank.get(min_importance, 0):
                continue
            rows.append((r, c))
        total = len(rows)
        out = []
        for r, c in rows[:limit]:
            out.append({"evidence_id": f"cninfo:ann:{r['announcementId']}", "title": r["announcementTitle"],
                        "available_at": r["available_at"], "event_type": c.event_type, "event_types": list(c.types),
                        "stage": c.stage, "importance": c.importance, "classified_by": c.source,
                        "rule": f"{c.type_rule}/{c.stage_rule}"})
        return {**meta, "announcements": out, "total": total, "truncated": total > limit,
                "note": (f"只返回最近的 {limit} 条，共 {total} 条；更早的请缩小 lookback_days 或按 event_types 过滤"
                         if total > limit else "")}

    def event_chains(self, ticker: str, lookback_days: int = 365) -> dict:
        df = self._announcements(ticker, lookback_days)
        anns = []
        cls = classify_titles(df["announcementTitle"].astype(str).tolist()) if len(df) else []
        for (_, r), c in zip(df.iterrows(), cls, strict=True):
            if c.event_type != "other" and not c.is_summary_or_meta:
                anns.append(Ann(str(r["announcementId"]), ticker, c.event_type, c.stage, r["available_at"],
                                r["announcementTitle"], subject_tag(r["announcementTitle"], c.event_type)))
        chains = state_as_of(anns, self.as_of + "T23:59:59")
        return {"ticker": ticker, "as_of": self.as_of, "chains": [
            {"evidence_id": f"chain:{c.chain_id}", "chain_id": c.chain_id, "event_type": c.event_type,
             "subject": c.subject, "state": c.state, "open": not c.closed,
             "opened_at": c.opened_at, "updated_at": c.updated_at,
             "evidence_ids": [f"cninfo:ann:{i}" for i in c.ann_ids],
             "corrections": [f"cninfo:ann:{a}" for a, _ in c.corrections], "anomalies": c.anomalies}
            for c in chains]}

    # ------------------------------------------------------------------ 公告正文（A2 / A3）
    def _ann_row(self, ann_id: str) -> dict | None:
        """只允许读本次研究对象、as_of 之前发布、且在公告列表里出现过的公告（工具强制注入时点与证券范围）。"""
        return getattr(self, "_known", {}).get(str(ann_id))

    def _remember(self, df: pd.DataFrame) -> None:
        known = getattr(self, "_known", {})
        for _, r in df.iterrows():
            known[str(r["announcementId"])] = {"adjunctUrl": r.get("adjunctUrl"), "title": r["announcementTitle"],
                                               "available_at": r["available_at"]}
        self._known = known

    def _parsed(self, ann_id: str):
        from quantmind.line_a.evidence.fetch import fetch_pdf
        from quantmind.line_a.evidence.pdf_locate import parse_pdf
        row = self._ann_row(ann_id)
        if row is None:
            raise ValueError("unknown announcement: call list_announcements first (only the subject's announcements "
                             "published on or before as_of can be read)")
        cache = getattr(self, "_docs", {})
        if ann_id not in cache:
            pdf_dir = self.pdf_dir or (self.lake / "text/cninfo_pdf")
            f = fetch_pdf(ann_id, row["adjunctUrl"], pdf_dir)
            doc = parse_pdf(Path(f["path"]), f"doc:{ann_id}")
            cache[ann_id] = (doc, f)
            self._docs = cache
            # 旁路证据：全文与逐段记录只登记给校验器（引文逐字核对），不发给模型、不占上下文
            side = getattr(self, "side_evidence", {})
            side[f"doc:{ann_id}"] = {"evidence_id": f"doc:{ann_id}", "title": row["title"],
                                     "available_at": row["available_at"], "file_sha256": f["sha256"],
                                     "full_text": "\n".join(p.text for p in doc.paras)}
            for p in doc.paras:
                side[f"doc:{ann_id}:p{p.page}:{p.idx}"] = {"evidence_id": f"doc:{ann_id}:p{p.page}:{p.idx}",
                                                           "page": p.page, "para": p.idx, "text": p.text,
                                                           "text_sha256": p.sha256}
            self.side_evidence = side
        return cache[ann_id]

    def read_document(self, ticker: str, ann_id: str) -> dict:
        doc, f = self._parsed(str(ann_id))
        row = self._ann_row(str(ann_id))
        return {"evidence_id": f"doc:{ann_id}", "title": row["title"], "available_at": row["available_at"],
                "file_sha256": f["sha256"], "n_pages": doc.n_pages, "needs_ocr_pages": doc.needs_ocr,
                "n_tables": len(doc.tables), "parser": doc.parser,
                "paragraphs": [{"evidence_id": f"doc:{ann_id}:p{p.page}:{p.idx}", "page": p.page, "para": p.idx,
                                "text": p.text[:300]} for p in doc.paras[:12]],
                "note": "只给出前 12 段的开头；用 locate_section 按问题定位具体段落"}

    def locate_section(self, ticker: str, ann_id: str, query: str, top_k: int = 5) -> dict:
        from quantmind.line_a.evidence.retrieval import BM25
        doc, _ = self._parsed(str(ann_id))
        paras = {f"doc:{ann_id}:p{p.page}:{p.idx}": p for p in doc.paras}
        hits = BM25([(k, p.text) for k, p in paras.items()]).search(query, top_k=min(int(top_k), 8))
        return {"evidence_id": f"doc:{ann_id}", "query": query, "hits": [
            {"evidence_id": h.para_id, "page": paras[h.para_id].page, "para": paras[h.para_id].idx,
             "score": round(h.score, 3), "text": paras[h.para_id].text[:800],
             "text_sha256": paras[h.para_id].sha256} for h in hits]}
