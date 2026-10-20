"""scripts/data_pipeline/probe_tushare_permissions.py — 探测本账号对若干 Tushare 接口的访问权限（T-035 P0）.

每个接口只发一次很小的查询，记录：能否访问、返回行数、列名、错误类型（截断到 200 字符）。
不打印、不写入 token；token 由 `.env` 加载，走 `quantmind.data.tushare_provider` 的同一套客户端。

用法：python scripts/data_pipeline/probe_tushare_permissions.py --out docs/work/evidence/T-035/tushare_permissions.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

PROBES = [
    ("pledge_stat", {"ts_code": "000001.SZ"}),
    ("pledge_detail", {"ts_code": "000001.SZ"}),
    ("fina_audit", {"ts_code": "000001.SZ", "start_date": "20200101", "end_date": "20251231"}),
    ("income_vip", {"period": "20250930", "fields": "ts_code,ann_date,f_ann_date,end_date"}),
    ("fina_indicator_vip", {"period": "20250930", "fields": "ts_code,ann_date,end_date"}),
    ("forecast_vip", {"period": "20250930", "fields": "ts_code,ann_date,end_date,type"}),
    ("express_vip", {"period": "20250930", "fields": "ts_code,ann_date,end_date"}),
    ("disclosure_date", {"end_date": "20250930"}),
    ("top10_floatholders", {"ts_code": "000001.SZ", "period": "20250930"}),
    ("stk_managers", {"ts_code": "000001.SZ"}),
    ("dividend", {"ts_code": "000001.SZ"}),
]


def _client():
    from dotenv import load_dotenv
    load_dotenv(os.getenv("QM_ENV_FILE", "/home/lenovo/projects/quantmind/.env"), override=False)
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger
        logger.remove()
    except Exception:  # noqa: BLE001
        pass
    return tp, tp._get_tushare_pro()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tp, pro = _client()
    results = []
    for api, kw in PROBES:
        rec = {"api": api, "params": {k: v for k, v in kw.items() if k != "fields"}}
        t0 = time.time()
        try:
            tp._rate_limit()
            df = pro.query(api, **kw)
            rec.update(status="ok", rows=int(len(df)), columns=list(df.columns)[:40])
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")[:200]
            denied = any(s in msg for s in ("权限", "积分", "permission"))
            rec.update(status="denied" if denied else "error", error_type=type(e).__name__, message=msg)
        rec["seconds"] = round(time.time() - t0, 2)
        results.append(rec)
        print(f"{api}: {rec['status']} rows={rec.get('rows', '-')}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"utc": datetime.now(UTC).isoformat(), "results": results},
                                ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
