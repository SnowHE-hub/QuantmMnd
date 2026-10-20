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

from quantmind.line_a.evidence.ann_types import classify_title
from quantmind.line_a.evidence.event_chain import Ann, state_as_of

FIN_FIELDS = {
    "income": ["total_revenue", "revenue", "n_income", "n_income_attr_p", "operate_profit"],
    "balancesheet": ["total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "total_cur_assets", "total_cur_liab"],
    "cashflow": ["n_cashflow_act", "c_paid_invest", "c_pay_acq_const_fiolta", "free_cashflow"],
    "fina_indicator_full": ["roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "current_ratio",
                            "or_yoy", "netprofit_yoy", "dt_netprofit_yoy"],
}


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


@dataclass
class DataTools:
    lake: Path                       # data/lake
    as_of: str                       # YYYYMMDD；所有工具的时点上限
    pdf_dir: Path | None = None      # 已取得的公告 PDF 缓存目录

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
                 **{f: _clean(r[f]) for f in fields if f in r}} for _, r in df.iterrows()]
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
            res[api] = [{"evidence_id": f"tushare:{api}:{ticker}:{r['end_date']}:{r['ann_date']}",
                         **{c: _clean(r[c]) for c in cols if c in r}} for _, r in df.iterrows()]
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
                              "pledge_ratio": _clean(r.get("pledge_ratio"))} for _, r in df.iterrows()]
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
        return df[df["available_at"].str.slice(0, 10).str.replace("-", "") <= self.as_of]

    def list_announcements(self, ticker: str, lookback_days: int = 180, limit: int = 60) -> dict:
        df = self._announcements(ticker, lookback_days)
        if df.empty:
            return {"ticker": ticker, "as_of": self.as_of, "announcements": [], "coverage_note":
                    "巨潮元数据在该区间没有记录（可能是采集尚未覆盖，不能据此推断没有公告）"}
        df = df.sort_values("available_at", ascending=False).head(limit)
        out = []
        for _, r in df.iterrows():
            c = classify_title(r["announcementTitle"])
            out.append({"evidence_id": f"cninfo:ann:{r['announcementId']}", "title": r["announcementTitle"],
                        "available_at": r["available_at"], "event_type": c.event_type, "stage": c.stage,
                        "importance": c.importance, "rule": f"{c.type_rule}/{c.stage_rule}"})
        return {"ticker": ticker, "as_of": self.as_of, "announcements": out}

    def event_chains(self, ticker: str, lookback_days: int = 365) -> dict:
        df = self._announcements(ticker, lookback_days)
        anns = []
        for _, r in df.iterrows():
            c = classify_title(r["announcementTitle"])
            if c.event_type != "other" and not c.is_summary_or_meta:
                anns.append(Ann(str(r["announcementId"]), ticker, c.event_type, c.stage, r["available_at"],
                                r["announcementTitle"]))
        chains = state_as_of(anns, self.as_of + "T23:59:59")
        return {"ticker": ticker, "as_of": self.as_of, "chains": [
            {"chain_id": c.chain_id, "event_type": c.event_type, "state": c.state, "open": not c.closed,
             "opened_at": c.opened_at, "updated_at": c.updated_at,
             "evidence_ids": [f"cninfo:ann:{i}" for i in c.ann_ids],
             "corrections": [f"cninfo:ann:{a}" for a, _ in c.corrections], "anomalies": c.anomalies}
            for c in chains]}
