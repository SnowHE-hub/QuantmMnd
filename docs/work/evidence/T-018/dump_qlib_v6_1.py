"""T-018 步骤 1：完整价格表 → 新的 qlib bin 目录（qlib_bakeoff env）。

= scripts/survivorship/p4a_dump_v6.py 的 v6_1 版：复用 scripts/bakeoff/p1_dump_bin 的 prepare_csvs / run_dump / g1_selfcheck，
只改输入输出路径。p1_dump_bin 在写入前会 rmtree 目标目录，所以这里**先断言目标目录不存在**（只写新目录，绝不碰 v5 / v6 的 qlib 目录）。
用法：<qlib_bakeoff python> dump_qlib_v6_1.py <repo_with_scripts> <data_root>
"""
import json
import sys
from pathlib import Path

repo, root = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
import scripts.bakeoff.p1_dump_bin as d  # noqa: E402

d.PRICES = root / "data/raw/alpha_prices_panel_v6_20261009.parquet"
d.CSV_DIR = Path("/tmp/qlib_csv_v6_1")
d.QLIB_DIR = root / "data/qlib_cn_daily_v6_1"
d.MAP_PATH = root / "data/bakeoff/ticker_qsym_map_v6_1.json"
d.DUMP = repo / "scripts/bakeoff/dump_bin.py"
for p in (d.CSV_DIR, d.QLIB_DIR, d.MAP_PATH):
    if p.exists():
        raise SystemExit(f"refusing: {p} already exists (p1_dump_bin would delete or overwrite it)")
for old in ("qlib_cn_daily", "qlib_cn_daily_v6"):
    assert d.QLIB_DIR.resolve() != (root / "data" / old).resolve()

info = d.prepare_csvs()
d.run_dump()
g1 = d.g1_selfcheck(info["ref"])
out = repo / "docs/work/evidence/T-018/qlib_v6_1_g1.json"
out.write_text(json.dumps(g1, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
cal = (d.QLIB_DIR / "calendars/day.txt").read_text().split()
print("G1:", "PASS" if g1.get("g1_pass") else "FAIL", "close_recon_maxrelerr=", g1.get("close_recon_maxrelerr"),
      "alpha158_shape=", g1.get("alpha158_shape"), "calendar", cal[0], "->", cal[-1], len(cal), flush=True)
raise SystemExit(0 if g1.get("g1_pass") else 1)
