"""T-022 第二轮修订：按时点（PIT）的申万 2021 行业与行业特征——分类体系发布前一律缺失（只读输入，写新文件）.

Codex 第二轮驳回：原 build_sector_features.py 把 2021 体系往回套到发布之前（只按成员 in_date / out_date 筛），属于前视。
修订：行业标签只在 as_of ≥ TAXONOMY_PUBLIC 时可用；之前的行业与全部行业特征为缺失，不套用未来的归属。
TAXONOMY_PUBLIC = 2021-12-13：新版申万行业指数（含成分）正式推出的日期；分类标准文件本身 2021-07-31 发布，
  取更晚的日期是保守选择（2021-07-31 到 12-13 之间个股归属是否已公开可查，本卡没有一手证据）。
其余（成员时点筛选、9 个行业特征的定义）与 build_sector_features.py 相同。
输出 data/panel/sector_features_v6_1_pit.parquet（index = (as_of, ticker)），拒绝覆盖。
用法：python build_sector_features_pit.py <data_root> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

TAXONOMY_PUBLIC = pd.Timestamp("2021-12-13")
LEVELS = {"l1_name": "sw_l1", "l2_name": "sw_l2", "l3_name": "sw_l3"}


def pit_industry(index: pd.MultiIndex, sw: pd.DataFrame, taxonomy_public: pd.Timestamp = TAXONOMY_PUBLIC) -> pd.DataFrame:
    """index=(as_of, ticker)。返回 sw_l1/l2/l3：as_of 时点的成员（in_date ≤ as_of < out_date），且 as_of ≥ 体系发布日；否则缺失。"""
    sw = sw.copy()
    sw["in_date"] = pd.to_datetime(sw["in_date"], errors="coerce")
    sw["out_date"] = pd.to_datetime(sw["out_date"], errors="coerce")
    idx = index.to_frame(index=False)
    m = idx.merge(sw[["ts_code", *LEVELS, "in_date", "out_date"]], left_on="ticker", right_on="ts_code", how="left")
    ok = ((m["in_date"] <= m["as_of"]) & (m["out_date"].isna() | (m["out_date"] > m["as_of"]))
          & (m["as_of"] >= taxonomy_public))
    return (m[ok].drop_duplicates(["as_of", "ticker"]).set_index(["as_of", "ticker"])[list(LEVELS)]
            .rename(columns=LEVELS).reindex(index))


def sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """df：momentum_1m/3m/6m、roe_ttm、pb、pe_ttm、operating_profit_yoy、sw_l1。无行业的行全部缺失。"""
    key = [df.index.get_level_values("as_of"), df["sw_l1"]]
    feat = pd.DataFrame(index=df.index)
    for c, name in (("momentum_1m", "ind_mom_1m"), ("momentum_3m", "ind_mom_3m"), ("momentum_6m", "ind_mom_6m")):
        feat[name] = df.groupby(key)[c].transform("mean")
    feat["rel_mom_3m"] = df["momentum_3m"] - feat["ind_mom_3m"]
    feat["rel_mom_6m"] = df["momentum_6m"] - feat["ind_mom_6m"]
    for c in ("roe_ttm", "pb", "pe_ttm", "operating_profit_yoy"):
        feat[f"ind_dm_{c}"] = df[c] - df.groupby(key)[c].transform("median")
    feat[df["sw_l1"].isna()] = float("nan")
    return feat


def main() -> None:
    root, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    D = root / "data"
    out = D / "panel/sector_features_v6_1_pit.parquet"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    mom = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["momentum_1m", "momentum_3m", "momentum_6m"])
    fnd = pd.read_parquet(D / "panel/fundamental_factors_v6_1.parquet",
                          columns=["roe_ttm", "pb", "pe_ttm", "operating_profit_yoy"]).reindex(mom.index)
    sw = pd.read_parquet(D / "lake/events/index_member_all_20261009.parquet")
    ind = pit_industry(mom.index, sw)
    feat = pd.concat([ind, sector_features(pd.concat([mom, fnd, ind], axis=1))], axis=1)
    feat.to_parquet(out)
    a = feat.index.get_level_values("as_of")
    res = {"taxonomy_public": str(TAXONOMY_PUBLIC.date()), "rows": int(len(feat)),
           "nonnull_l1_before_public": int(feat.loc[a < TAXONOMY_PUBLIC, "sw_l1"].notna().sum()),
           "nonnull_features_before_public": int(feat.loc[a < TAXONOMY_PUBLIC].drop(columns=list(LEVELS.values())).notna().sum().sum()),
           "coverage_l1_after_public": round(float(feat.loc[a >= TAXONOMY_PUBLIC, "sw_l1"].notna().mean()), 4),
           "n_l1": int(feat["sw_l1"].nunique()), "seconds": round(time.time() - t0, 1), "output": str(out)}
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
