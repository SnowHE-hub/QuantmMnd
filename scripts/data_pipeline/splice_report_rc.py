"""scripts/data_pipeline/splice_report_rc.py — 把购买的数据包（清洗版）和中转补的缺口拼成一张连续的 report_rc（T-038）.

规则（2026-10-22 交叉核对之后定，见 T-038）：
- 数据包清洗版用到 2026-05-21（其中 2018-10-01～2019-04-13 已是官方数据）；
- 2026-05-22 起用中转数据（数据包最后一天不全），只取中转检查点里状态为 done / empty 的股票；
- 拼好之后重新计算 version_conflict（同 KEY 多行的整组标出）；加 source 列（official / pack / relay）；
- 写到新文件 report_rc_extended.parquet，并在 _extended.json 记录每段来源的行数、日期范围和中转没拉完的股票数。
用法：python -I scripts/data_pipeline/splice_report_rc.py --data-root <root>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data_pipeline.fetch_report_rc_relay import KEY, _normalize  # noqa: E402

PACK_LAST = "20260521"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()
    lake = a.data_root / "data/lake"
    out = lake / "events_purchased/xianyu_20261022"
    clean = pd.read_parquet(out / "report_rc_clean.parquet")
    clean = clean[clean["report_date"] <= PACK_LAST].drop(columns=["version_conflict"])
    rdir = lake / "events_relay/report_rc/by_ticker"
    ck = json.loads((rdir / "_checkpoint.json").read_text())["segments"]
    seg = {k.split(":")[0]: v for k, v in ck.items() if k.split(":")[1] == "20260501"}
    ok = {t for t, v in seg.items() if v.get("state") in ("done", "empty")}
    rel = pd.concat([pd.read_parquet(p) for p in sorted(rdir.glob("part_*.parquet"))], ignore_index=True)
    rel = rel[rel["ts_code"].isin(ok) & (rel["report_date"] > PACK_LAST)]
    rel = _normalize(rel[[c for c in rel.columns if not c.endswith("__raw")]]).assign(source="relay")
    cols = list(clean.columns)
    for c in cols:
        if c not in rel.columns:
            rel[c] = pd.NA
    ext = pd.concat([clean, rel[cols]], ignore_index=True)
    ext["version_conflict"] = ext[KEY].astype(str).duplicated(keep=False)
    ext = ext.sort_values(["report_date", "ts_code", "org_name", "quarter"]).reset_index(drop=True)
    ext.to_parquet(out / "report_rc_extended.parquet", compression="zstd")
    info = {"rows": int(len(ext)), "by_source": ext["source"].value_counts().to_dict(),
            "report_date_min": ext["report_date"].min(), "report_date_max": ext["report_date"].max(),
            "pack_last": PACK_LAST, "relay_tickers_done": len(ok), "relay_tickers_not_done": len(set(seg) - ok),
            "version_conflict_rows": int(ext["version_conflict"].sum())}
    (out / "_extended.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
