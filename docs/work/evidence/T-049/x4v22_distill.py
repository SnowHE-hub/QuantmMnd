"""T-049 X4 v2.2：用指南 v2.1 重新标注 v2 训练 / 校准集和难例训练 / 校准补充（写到 x4_v22，不覆盖旧标签）。"""
from pathlib import Path
import pandas as pd
from quantmind.line_a.harness.llm import LLM
from quantmind.line_a.p4 import x4v2
V2 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v2")
V21 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v21")
OUT = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v22")
OUT.mkdir(parents=True, exist_ok=True)
fac = lambda: LLM(ledger=OUT / "llm_ledger.jsonl", cache_dir=OUT / "_llm_cache", budget_rmb=60.0)  # noqa: E731
for name, src in (("calib", V2), ("calib_add", V21), ("train_add", V21), ("train", V2)):
    df = pd.read_parquet(src / f"{name}.parquet")[["announcementId", "title"]]
    df["announcementId"] = df["announcementId"].astype(str)
    dst = OUT / f"distill_{name}.jsonl"
    for rnd in range(3):
        got = x4v2.distill(fac(), df, dst, log=lambda m: print(m, flush=True), system=x4v2.GUIDE_V21, workers=6,
                           llm_factory=fac)
        miss = len(set(df["announcementId"]) - set(got.index.astype(str)))
        print(name, "round", rnd, "labeled", len(got), "missing", miss, flush=True)
        if not miss:
            break
