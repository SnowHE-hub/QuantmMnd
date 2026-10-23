"""scripts/line_a/x4v21_split.py — X4 v2.1 难例池切分（T-049 登记，种子 20261027）.

- gold_hard：随机 300 条，导出只含编号和标题的文本（不含任何模型或规则的判断），供 Claude 盲标；
- 其余：90% 训练补充（train_add），10% 校准补充（calib_add），之后由 DeepSeek 按指南 v2 标注。
用法：python scripts/line_a/x4v21_split.py --dir data/line_a_v2/p4/x4_v21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEED = 20261027


def split(pool: pd.DataFrame, n_gold: int = 300) -> dict[str, pd.DataFrame]:
    pool = pool.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    gold, rest = pool.iloc[:n_gold], pool.iloc[n_gold:]
    n_cal = round(len(rest) * 0.1)
    return {"gold_hard": gold, "calib_add": rest.iloc[:n_cal], "train_add": rest.iloc[n_cal:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    pool = pd.read_parquet(a.dir / "hard_pool.parquet")
    parts = split(pool)
    for name, d in parts.items():
        d.to_parquet(a.dir / f"{name}.parquet")
        print(name, len(d))
    g = parts["gold_hard"].reset_index(drop=True)
    lines = [f"{i:03d}\t{t}" for i, t in enumerate(g["title"])]
    (a.dir / "gold_hard_titles_for_blind_labeling.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    g[["announcementId", "title"]].to_parquet(a.dir / "gold_hard_index.parquet")


if __name__ == "__main__":
    main()
