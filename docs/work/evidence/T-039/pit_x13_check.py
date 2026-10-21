"""T-039：用 X13 的 80 份年报 PDF（原始公告）检验 f_ann_date 规则.

真值：X13 开发集、测试集（T-038）里 pymupdf-rules-v2 解析出的本期数。X13 里所有「错」都已按登记条件裁定为
「标准值（lake）有误」，所以 PDF 解析值可以当原始公告的数用；「没找到」「跳过」的字段不参与。
对每份年报、每个字段：PDF 值是否等于 lake 值、等于原始版（f_ann_date == ann_date）、等于后来版（f_ann_date > ann_date）。
规则判定取 pit_analyze.py --sub x13_cases 的 lake_match_<api>.parquet（lake 行只对得上后来版 = 晚到值）。

用法：PYTHONPATH=. python docs/work/evidence/T-039/pit_x13_check.py --data-root /home/lenovo/projects/quantmind
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
API_OF = {"total_revenue": "income", "revenue": "income", "operate_profit": "income", "n_income": "income",
          "n_income_attr_p": "income", "total_assets": "balancesheet", "total_liab": "balancesheet",
          "total_hldr_eqy_exc_min_int": "balancesheet", "total_cur_assets": "balancesheet",
          "total_cur_liab": "balancesheet", "n_cashflow_act": "cashflow"}
SETS = {"dev": ("x13_dev_v1", REPO / "docs/work/evidence/T-038/x13_dev_score_rules_v2.json"),
        "test": ("x13_test_v1", REPO / "docs/work/evidence/T-038/x13_test_v1/score_rules_v2.json")}


def near(a: float, b: float, tol: float) -> bool:
    return not (np.isnan(a) or np.isnan(b)) and abs(a - b) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()
    root = a.data_root.resolve()
    base = root / "data/research/pit_audit/x13_cases"
    p4 = root / "data/line_a_v2/p4"
    raw = {api: pd.concat([pd.read_parquet(f) for f in sorted((base / "raw" / api).glob("*.parquet"))], ignore_index=True)
           for api in ("income", "balancesheet", "cashflow")}
    lake = {api: pd.read_parquet(root / f"data/lake/{api}.parquet") for api in raw}
    match = {api: pd.read_parquet(base / "results" / f"lake_match_{api}.parquet") for api in raw}

    rows = []
    for name, (d, score_path) in SETS.items():
        cases = pd.read_csv(p4 / d / "cases.csv", dtype=str, encoding="utf-8-sig")
        ext = json.loads((p4 / d / "baseline_pymupdf_rules_v2" / "extract.json").read_text())
        outc = {(r["ann_id"], r["field"]): r["outcome"] for r in json.loads(score_path.read_text())["records"]}
        for c in cases.itertuples(index=False):
            end, doc_date = f"{c.year}1231", c.date
            for field, api in API_OF.items():
                o = outc.get((c.announcementId, field))
                e = ext.get(c.announcementId, {}).get(field) or {}
                rec = {"set": name, "ann_id": c.announcementId, "ts_code": c.ts_code, "year": c.year,
                       "doc_date": doc_date, "field": field, "api": api, "x13_outcome": o}
                if o not in ("correct", "wrong") or e.get("value") is None:
                    rows.append(rec)
                    continue
                pdf = float(e["value"])
                tol = max(0.006, 0.51 * float(e.get("unit") or 1) * 10 ** (-int(e.get("decimals") or 2)))
                lk = lake[api][(lake[api].ts_code == c.ts_code) & (lake[api].end_date == end)].copy()
                lk["gap"] = (pd.to_datetime(lk.ann_date) - pd.Timestamp(doc_date)).dt.days.abs()
                lk = lk[lk.gap <= 7].sort_values("gap")
                g = raw[api][(raw[api].ts_code == c.ts_code) & (raw[api].end_date == end)]
                if lk.empty:
                    rows.append({**rec, "note": "no_lake_row_within_7d"})
                    continue
                ann = lk.ann_date.iloc[0]
                g = g[g.ann_date == ann]
                orig = g[g.f_ann_date == ann][field].astype(float)
                late = g[g.f_ann_date > ann][field].astype(float)
                m = match[api]
                cls = m[(m.ts_code == c.ts_code) & (m.end_date == end) & (m.ann_date == ann)].cls
                rec.update({"pdf": pdf, "lake": float(lk[field].iloc[0]), "ann_date": ann,
                            "pdf_eq_lake": near(pdf, float(lk[field].iloc[0]), tol),
                            "orig_present": bool(len(orig)), "late_present": bool(len(late)),
                            "pdf_eq_orig": any(near(pdf, x, tol) for x in orig),
                            "pdf_eq_late": any(near(pdf, x, tol) for x in late),
                            "rule_cls": cls.iloc[0] if len(cls) else None})
                rows.append(rec)
    df = pd.DataFrame(rows)
    out = base / "results"
    df.to_parquet(out / "x13_check_fields.parquet", index=False)

    ok = df[df.pdf.notna()]
    # 年报 × 报表一级：PDF 与 lake 是否有字段不同；规则是否判为晚到值
    fs = ok.groupby(["set", "ann_id", "ts_code", "year", "api"]).agg(
        lake_wrong=("pdf_eq_lake", lambda s: bool((~s.astype(bool)).any())),
        flagged=("rule_cls", lambda s: bool((s == "match_late_only").any())),
        orig_present=("orig_present", "max"), pdf_eq_orig_all=("pdf_eq_orig", "all"),
        pdf_eq_late_all=("pdf_eq_late", "all")).reset_index()
    fy = fs.groupby(["set", "ann_id", "ts_code", "year"]).agg(lake_wrong=("lake_wrong", "max"),
                                                              flagged=("flagged", "max")).reset_index()

    def cm(t: pd.DataFrame) -> dict:
        return {"n": int(len(t)), "wrong_and_flagged": int((t.lake_wrong & t.flagged).sum()),
                "wrong_not_flagged": int((t.lake_wrong & ~t.flagged).sum()),
                "right_but_flagged": int((~t.lake_wrong & t.flagged).sum()),
                "right_not_flagged": int((~t.lake_wrong & ~t.flagged).sum())}

    summary = {
        "fields_compared": int(len(ok)), "fields_not_compared": int(df.pdf.isna().sum()),
        "excluded_or_no_lake_row": df.get("note", pd.Series(dtype=object)).value_counts().to_dict(),
        "firm_year": cm(fy), "firm_year_statement": cm(fs),
        "wrong_cases": fy[fy.lake_wrong][["set", "ts_code", "year"]].to_dict("records"),
        "flagged_cases": fs[fs.flagged][["set", "ts_code", "year", "api", "lake_wrong", "orig_present",
                                          "pdf_eq_orig_all"]].to_dict("records"),
        # PDF 值等于原始版的比例（原始版存在时）：检验「原始版 = 原始公告」这一假设
        "pdf_eq_orig_when_orig_present": float(ok[ok.orig_present].pdf_eq_orig.mean()),
    }
    (out / "x13_check_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
