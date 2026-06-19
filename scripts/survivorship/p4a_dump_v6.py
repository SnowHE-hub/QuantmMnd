"""P4a：v6 全市场 qlib bin 重 dump（qlib_bakeoff env）.

复用 scripts/bakeoff/p1_dump_bin 的 prepare_csvs/run_dump/g1_selfcheck，仅重指向 v6 路径：
  prices → alpha_prices_panel_v6.parquet（5741票）
  qlib_dir → data/qlib_cn_daily_v6
  map → ticker_qsym_map_v6.json
后复权口径、ts_to_qsym、FIELDS、G1(close 还原 + handler 非空) 全部不变。
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import json
import scripts.bakeoff.p1_dump_bin as d

# 重指向 v6（不动 v5 的 qlib_cn_daily / ticker_qsym_map.json）
d.PRICES = REPO / "data" / "raw" / "alpha_prices_panel_v6.parquet"
d.CSV_DIR = Path("/tmp/qlib_csv_v6")
d.QLIB_DIR = REPO / "data" / "qlib_cn_daily_v6"
d.MAP_PATH = REPO / "data" / "bakeoff" / "ticker_qsym_map_v6.json"

if __name__ == "__main__":
    info = d.prepare_csvs()
    d.run_dump()
    g1 = d.g1_selfcheck(info["ref"])
    (REPO / "scripts" / "survivorship" / "_p4a_g1_v6.json").write_text(
        json.dumps(g1, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("P4A_G1:", "PASS" if g1.get("g1_pass") else "FAIL",
          "close_recon_maxrelerr=", g1.get("close_recon_maxrelerr"),
          "alpha158_shape=", g1.get("alpha158_shape"))
    raise SystemExit(0 if g1.get("g1_pass") else 1)
