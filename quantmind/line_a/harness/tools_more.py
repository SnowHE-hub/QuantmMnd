"""quantmind.line_a.harness.tools_more — A3（治理与资本运作）、A4（预期与资金）、A6（证据审查）的只读工具.

与 tools_data.DataTools 同一套约束：只读、按 as_of 截止、每条记录有证据编号和字段单位。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pyarrow.dataset as ds

from quantmind.line_a.harness.tools_data import DataTools, _clean, _latest, _rows, _ymd


@dataclass
class ResearchTools(DataTools):
    registry: dict = field(default_factory=dict)     # A6 用：上游角色本次实际取得的证据记录

    def _event_rows(self, api: str, ticker: str, date_col: str, lo: str, hi: str) -> pd.DataFrame:
        p = _latest(self.lake, api)
        if p is None:
            return pd.DataFrame()
        df = _rows(p, ticker)
        if df.empty:
            return df
        v = df[date_col].astype(str).str.replace("-", "").str.slice(0, 8)
        return df[(v > lo) & (v <= hi)]

    # ------------------------------------------------------------------ A3 治理与资本运作
    def holder_trade_records(self, ticker: str, lookback_days: int = 365) -> dict:
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        df = self._event_rows("stk_holdertrade", ticker, "ann_date", lo, self.as_of)
        df = df.sort_values("ann_date", ascending=False) if len(df) else df
        recs = [{"evidence_id": f"tushare:stk_holdertrade:{ticker}:{r['ann_date']}:{r.get('holder_name')}",
                 "ann_date": str(r["ann_date"]), "holder_name": _clean(r.get("holder_name")),
                 "holder_type": _clean(r.get("holder_type")), "in_de": _clean(r.get("in_de")),
                 "change_ratio": _clean(r.get("change_ratio")), "change_vol": _clean(r.get("change_vol")),
                 "after_ratio": _clean(r.get("after_ratio")), "avg_price": _clean(r.get("avg_price")),
                 "_units": {"change_ratio": "%", "after_ratio": "%", "change_vol": "股", "avg_price": "元"}}
                for _, r in df.head(60).iterrows()]
        return {"ticker": ticker, "as_of": self.as_of, "window": [lo, self.as_of], "total": int(len(df)), "records": recs}

    def share_float_schedule(self, ticker: str, ahead_days: int = 180, back_days: int = 90) -> dict:
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=back_days))
        hi = _ymd(pd.Timestamp(self.as_of) + pd.Timedelta(days=ahead_days))
        df = self._event_rows("share_float", ticker, "float_date", lo, hi)
        if len(df):
            df = df[df["ann_date"].astype(str).str.replace("-", "").str.slice(0, 8) <= self.as_of].sort_values("float_date")
        recs = [{"evidence_id": f"tushare:share_float:{ticker}:{r['float_date']}:{r.get('holder_name')}",
                 "ann_date": str(r["ann_date"]), "float_date": str(r["float_date"]),
                 "float_ratio": _clean(r.get("float_ratio")), "float_share": _clean(r.get("float_share")),
                 "holder_name": _clean(r.get("holder_name")), "share_type": _clean(r.get("share_type")),
                 "_units": {"float_ratio": "%", "float_share": "股"}} for _, r in df.head(60).iterrows()]
        return {"ticker": ticker, "as_of": self.as_of, "window": [lo, hi], "records": recs,
                "note": "只列出 as_of 当天或之前已经公告的解禁安排"}

    def repurchase_records(self, ticker: str, lookback_days: int = 365) -> dict:
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        df = self._event_rows("repurchase", ticker, "ann_date", lo, self.as_of)
        df = df.sort_values("ann_date", ascending=False) if len(df) else df
        recs = [{"evidence_id": f"tushare:repurchase:{ticker}:{r['ann_date']}:{r.get('proc')}",
                 "ann_date": str(r["ann_date"]), "proc": _clean(r.get("proc")), "exp_date": _clean(r.get("exp_date")),
                 "vol": _clean(r.get("vol")), "amount": _clean(r.get("amount")),
                 "high_limit": _clean(r.get("high_limit")), "low_limit": _clean(r.get("low_limit")),
                 "_units": {"vol": "股", "amount": "元", "high_limit": "元", "low_limit": "元"}}
                for _, r in df.head(40).iterrows()]
        return {"ticker": ticker, "as_of": self.as_of, "window": [lo, self.as_of], "records": recs}

    # ------------------------------------------------------------------ A4 预期与资金
    def analyst_revisions(self, ticker: str, lookback_days: int = 180) -> dict:
        """卖方盈利预测（report_rc）：同一机构对同一预测年度最近两次预测的变化（同口径修正）与各年度的中位数。"""
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        parts = sorted((self.lake / "events/_parts/report_rc").glob("*.parquet"))
        covered = max((p.stem.split("_")[-1] for p in parts), default="")
        note = "" if covered >= self.as_of else (
            f"report_rc 回补只到 {covered or '（无）'}，as_of 附近的预测尚未取得；不能据此推断没有分析师覆盖")
        sel = [p for p in parts if p.stem.split("_")[-1] > lo and p.stem.split("_")[0] <= self.as_of]
        if not sel:
            return {"ticker": ticker, "as_of": self.as_of, "status": "not_covered", "coverage_note": note}
        df = pd.concat([ds.dataset(str(p), format="parquet").to_table(filter=ds.field("ts_code") == ticker).to_pandas()
                        for p in sel], ignore_index=True)
        d = df["report_date"].astype(str)
        df = df[(d > lo) & (d <= self.as_of)].sort_values("report_date")
        rev = []
        for (org, q), g in df.groupby(["org_name", "quarter"]):
            if len(g) >= 2:
                a, b = g.iloc[-2], g.iloc[-1]
                rev.append({"evidence_id": f"tushare:report_rc:{ticker}:{org}:{q}:{b['report_date']}", "org_name": org,
                            "quarter": q, "prev_date": str(a["report_date"]), "last_date": str(b["report_date"]),
                            "np_prev": _clean(a.get("np")), "np_last": _clean(b.get("np")),
                            "eps_prev": _clean(a.get("eps")), "eps_last": _clean(b.get("eps")),
                            "rating_prev": _clean(a.get("rating")), "rating_last": _clean(b.get("rating")),
                            "_units": {"np_prev": "万元", "np_last": "万元", "eps_prev": "元", "eps_last": "元"}})
        cons = [{"evidence_id": f"tushare:report_rc:{ticker}:median:{q}:{self.as_of}", "quarter": q,
                 "n_reports": int(len(g)), "n_orgs": int(g["org_name"].nunique()),
                 "np_median": _clean(g["np"].median()), "_units": {"np_median": "万元"}} for q, g in df.groupby("quarter")]
        return {"ticker": ticker, "as_of": self.as_of, "window": [lo, self.as_of], "n_reports": int(len(df)),
                "revisions_same_org": rev[:30], "consensus_by_year": cons, "coverage_note": note}

    def flow_snapshot(self, ticker: str) -> dict:
        """资金数据：个股资金流向（大单与特大单净额）与融资余额的 5 / 20 日变化。"""
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=45))
        mf = self._event_rows("moneyflow", ticker, "trade_date", lo, self.as_of)
        out: dict = {"ticker": ticker, "as_of": self.as_of}
        if len(mf):
            mf = mf.sort_values("trade_date")
            big = mf["buy_lg_amount"] + mf["buy_elg_amount"] - mf["sell_lg_amount"] - mf["sell_elg_amount"]
            out["moneyflow"] = {"evidence_id": f"tushare:moneyflow:{ticker}:{mf['trade_date'].iloc[-1]}",
                                "last_date": str(mf["trade_date"].iloc[-1]),
                                "net_mf_5d": float(mf["net_mf_amount"].tail(5).sum()),
                                "net_mf_20d": float(mf["net_mf_amount"].tail(20).sum()),
                                "big_order_net_5d": float(big.tail(5).sum()), "big_order_net_20d": float(big.tail(20).sum()),
                                "_units": dict.fromkeys(("net_mf_5d", "net_mf_20d", "big_order_net_5d",
                                                         "big_order_net_20d"), "万元")}
        mg = _rows(self.lake / "margin.parquet", ticker, ["ts_code", "trade_date", "rzye"])
        if len(mg):
            mg["trade_date"] = pd.to_datetime(mg["trade_date"])
            mg = mg[mg["trade_date"] <= pd.Timestamp(self.as_of)].sort_values("trade_date")
            if len(mg) > 20:
                v = mg["rzye"].to_numpy()
                out["margin"] = {"evidence_id": f"tushare:margin:{ticker}:{mg['trade_date'].iloc[-1].date()}",
                                 "last_date": str(mg["trade_date"].iloc[-1].date()), "rzye": float(v[-1]),
                                 "chg_5d": float(v[-1] / v[-6] - 1) if v[-6] else None,
                                 "chg_20d": float(v[-1] / v[-21] - 1) if v[-21] else None, "_units": {"rzye": "元"}}
        return out

    def top_list_records(self, ticker: str, lookback_days: int = 90) -> dict:
        lo = _ymd(pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days))
        tl = self._event_rows("top_list", ticker, "trade_date", lo, self.as_of)
        ti = self._event_rows("top_inst", ticker, "trade_date", lo, self.as_of)
        inst = ti.groupby("trade_date")["net_buy"].sum() if len(ti) and "net_buy" in ti else pd.Series(dtype=float)
        tl = tl.sort_values("trade_date", ascending=False).drop_duplicates("trade_date") if len(tl) else tl
        recs = [{"evidence_id": f"tushare:top_list:{ticker}:{r['trade_date']}", "trade_date": str(r["trade_date"]),
                 "reason": _clean(r.get("reason")), "pct_change": _clean(r.get("pct_change")),
                 "net_amount": _clean(r.get("net_amount")), "inst_net_buy": _clean(inst.get(r["trade_date"])),
                 "_units": {"pct_change": "%", "net_amount": "元", "inst_net_buy": "元"}} for _, r in tl.head(30).iterrows()]
        return {"ticker": ticker, "as_of": self.as_of, "window": [lo, self.as_of], "records": recs}

    # ------------------------------------------------------------------ A6 证据审查
    def read_evidence(self, ticker: str, evidence_id: str) -> dict:
        rec = self.registry.get(evidence_id)
        if rec is None:
            return {"evidence_id": evidence_id, "status": "not_in_registry",
                    "note": "这条证据不在上游角色本次实际取得的记录里"}
        return rec

    def search_counterevidence(self, ticker: str, query: str, lookback_days: int = 365) -> dict:
        """在研究对象的公告标题（以及本次已解析的公告段落）里做 BM25，用来找更正、撤回、澄清、条件未满足等反证。"""
        from quantmind.line_a.evidence.retrieval import BM25
        df = self._announcements(ticker, lookback_days)
        items = [(f"cninfo:ann:{r['announcementId']}", r["announcementTitle"]) for _, r in df.iterrows()]
        for ann_id, (doc, _) in getattr(self, "_docs", {}).items():
            items += [(f"doc:{ann_id}:p{p.page}:{p.idx}", p.text) for p in doc.paras]
        hits = BM25(items).search(query, top_k=8) if items else []
        text = dict(items)
        return {"ticker": ticker, "as_of": self.as_of, "query": query,
                "hits": [{"evidence_id": h.para_id, "score": round(h.score, 3), "text": text[h.para_id][:400]}
                         for h in hits],
                "note": "" if items else "该区间没有公告元数据（采集可能尚未覆盖），不能据此推断没有反证"}
