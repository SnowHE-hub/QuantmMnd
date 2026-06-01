"""scripts/backfill_rec_contract.py — 给历史推荐回填数据契约字段.

为已存在的 data/recommendations/*.json（及嵌套 {date}/top10.json）回填：
  - name / industry  ← alpha_universe.parquet
  - entry_price      ← snapshot/{as_of}/daily_basic.parquet 的 close
  - raw_pe_ttm/pb/roe← 原始基本面
并重算 market_summary（用原始 PE/ROE，修复"z-score 当原始值"的 bug）。

同时回填 data/paper_trading/forward_positions.json 的 entry_price（按 as_of+ticker）。

复用 daily_update.py 的 _enrich_top10 / _generate_market_summary，保证逻辑单一来源。

用法
====
  python scripts/backfill_rec_contract.py            # 回填全部
  python scripts/backfill_rec_contract.py --dry-run  # 只预览不写

注意：与 backfill_recommendations.py 不同——那个脚本是"从面板生成新推荐"，
本脚本只"给已有推荐补字段"。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.daily_update import _enrich_top10, _generate_market_summary  # noqa: E402

REC_DIR = _ROOT / "data" / "recommendations"
FWD_PATH = _ROOT / "data" / "paper_trading" / "forward_positions.json"


def _parse_as_of(payload: dict, fallback_stem: str) -> date | None:
    """从 payload['as_of'] 或文件名解析日期。"""
    raw = str(payload.get("as_of", "") or fallback_stem)
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            from datetime import datetime
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    return None


def backfill_recommendations(dry_run: bool = False) -> dict[str, int]:
    """回填 recommendations/*.json 与嵌套 top10.json。"""
    stats = {"files": 0, "enriched_items": 0, "skipped": 0}
    if not REC_DIR.exists():
        print(f"[Backfill] 推荐目录不存在：{REC_DIR}")
        return stats

    # 顶层 {date}.json
    json_files = sorted(REC_DIR.glob("*.json"))
    # 嵌套 {date}/top10.json
    nested = sorted(REC_DIR.glob("*/top10.json"))

    for fp in json_files + nested:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[Backfill] 跳过 {fp.name}（读取失败：{e}）")
            stats["skipped"] += 1
            continue

        top10 = payload.get("top10")
        if not isinstance(top10, list) or not top10:
            stats["skipped"] += 1
            continue

        as_of = _parse_as_of(payload, fp.parent.name if fp.name == "top10.json" else fp.stem)
        if as_of is None:
            print(f"[Backfill] 跳过 {fp}（无法解析日期）")
            stats["skipped"] += 1
            continue

        payload["top10"] = _enrich_top10(top10, as_of)
        # 顶层文件才有 market_summary；重算
        if "market_summary" in payload or fp.name != "top10.json":
            payload["market_summary"] = _generate_market_summary(payload["top10"])

        stats["files"] += 1
        stats["enriched_items"] += len(payload["top10"])

        n_price = sum(1 for it in payload["top10"] if it.get("entry_price"))
        print(f"[Backfill] {fp.relative_to(_ROOT)}  ({len(top10)} 只, "
              f"{n_price} 只有入场价)  summary='{payload.get('market_summary','')}'")

        if not dry_run:
            fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return stats


def backfill_forward_positions(dry_run: bool = False) -> int:
    """按 (as_of, ticker) 回填 forward_positions.json 的 entry_price。"""
    if not FWD_PATH.exists():
        print(f"[Backfill] forward_positions 不存在：{FWD_PATH}")
        return 0
    try:
        data = json.loads(FWD_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[Backfill] forward_positions 读取失败：{e}")
        return 0

    positions = data.get("positions", [])
    if not positions:
        return 0

    # 按 as_of 分组，复用 _enrich_top10 取每组的 entry_price
    from datetime import datetime
    by_date: dict[str, list[dict]] = {}
    for pos in positions:
        by_date.setdefault(str(pos.get("as_of", "")), []).append(pos)

    filled = 0
    for as_of_str, group in by_date.items():
        try:
            as_of = datetime.strptime(as_of_str[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        # 构造临时 top10 走 enrich 拿 entry_price
        tmp = [{"ticker": p["ticker"]} for p in group]
        enriched = _enrich_top10(tmp, as_of)
        price_map = {e["ticker"]: e.get("entry_price") for e in enriched}
        for pos in group:
            if pos.get("entry_price") is None:
                ep = price_map.get(pos["ticker"])
                if ep is not None:
                    pos["entry_price"] = ep
                    filled += 1

    print(f"[Backfill] forward_positions：回填 {filled} 个 entry_price")
    if not dry_run and filled:
        FWD_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return filled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回填历史推荐数据契约字段")
    parser.add_argument("--dry-run", action="store_true", help="只预览不写")
    parser.add_argument("--skip-forward", action="store_true", help="跳过 forward_positions 回填")
    args = parser.parse_args(argv)

    print("=" * 64)
    print(f"  推荐字段回填{'（DRY-RUN）' if args.dry_run else ''}")
    print("=" * 64)
    stats = backfill_recommendations(dry_run=args.dry_run)
    print(f"\n  回填文件数：{stats['files']}，条目数：{stats['enriched_items']}，"
          f"跳过：{stats['skipped']}")

    if not args.skip_forward:
        print("-" * 64)
        backfill_forward_positions(dry_run=args.dry_run)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
