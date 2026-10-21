"""T-036：前瞻 qlib bin 目录（与 T-018 dump_qlib_v6_1.py 同一流程，只把价格文件和输出路径改成参数；qlib_bakeoff env）.

复用 scripts/bakeoff/p1_dump_bin 的 prepare_csvs / run_dump / g1_selfcheck。p1_dump_bin 写入前会 rmtree 目标目录，
所以先断言目标目录、映射文件和临时 CSV 目录都不存在（只写新目录）。
用法：<qlib_bakeoff python> dump_qlib_fwd.py <repo_with_scripts> <data_root> <END_YYYYMMDD>
"""
import json
import sys
from pathlib import Path

repo, root, end = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(repo))
import scripts.bakeoff.p1_dump_bin as d  # noqa: E402

d.PRICES = root / f"data/raw/alpha_prices_panel_v6_{end}.parquet"
d.CSV_DIR = Path(f"/tmp/qlib_csv_fwd_{end}")
d.QLIB_DIR = root / f"data/qlib_cn_daily_fwd_{end}"
d.MAP_PATH = root / f"data/bakeoff/ticker_qsym_map_fwd_{end}.json"
d.DUMP = repo / "scripts/bakeoff/dump_bin.py"
for p in (d.CSV_DIR, d.QLIB_DIR, d.MAP_PATH):
    if p.exists():
        raise SystemExit(f"refusing: {p} already exists (p1_dump_bin would delete or overwrite it)")
for old in ("qlib_cn_daily", "qlib_cn_daily_v6", "qlib_cn_daily_v6_1"):
    assert d.QLIB_DIR.resolve() != (root / "data" / old).resolve()

info = d.prepare_csvs()
d.run_dump()
# 自检取沪深的前 3 个（2026-10-22）：价格面板里有了北交所，排序后前 3 个是 BJ 开头，它们在自检窗口（2023 年一季度）
# 没有数据，Alpha158/360 拿到空表报错，并不是数据问题
g1 = d.g1_selfcheck(info["ref"], insts=sorted(p.stem.upper() for p in d.CSV_DIR.glob("S[HZ]*.csv"))[:3])
out = repo / f"docs/work/evidence/T-036/qlib_fwd_{end}_g1.json"
out.write_text(json.dumps(g1, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
cal = (d.QLIB_DIR / "calendars/day.txt").read_text().split()
print("G1:", "PASS" if g1.get("g1_pass") else "FAIL", "close_recon_maxrelerr=", g1.get("close_recon_maxrelerr"),
      "alpha158_shape=", g1.get("alpha158_shape"), "calendar", cal[0], "->", cal[-1], len(cal), flush=True)
raise SystemExit(0 if g1.get("g1_pass") else 1)
