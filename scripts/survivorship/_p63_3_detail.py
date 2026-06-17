import json
from pathlib import Path
REPO = Path("/home/lenovo/projects/quantmind")
v = json.loads((REPO / "data/bakeoff/p63_3_verdict_v6.json").read_text())
for variant, detail in v["details"].items():
    t = detail["pit_top1500"]
    print("{}: turnover={} maxDD={} n_asof={} posfrac={}".format(
        variant, t["avg_oneway_turnover"], t["max_drawdown"], t["n_asof"], t["neut_ic_posfrac"]))
    if "full_market" in detail:
        fm = detail["full_market"]
        print("  full_market: neut_ic={} net={} icir={} regime={}".format(
            fm["neut_ic"], fm["net_excess_cost"], fm["neut_icir"], fm["regime"]))
