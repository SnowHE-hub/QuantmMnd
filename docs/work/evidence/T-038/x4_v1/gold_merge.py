"""把 Claude 的三批复核结果写进复核表，导入成最终金标准，并在金标准上给规则和本机 MacBERT 打分。"""
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from quantmind.line_a.p4 import x4_gold

X4 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v1")
SHEET = Path("/mnt/d/MCS_26S2课程资料/quantmind_labeling/x4_gold_review.csv")
PACK = Path("/mnt/d/MCS_26S2课程资料/quantmind_colab_upload/pack_x4_v1")
labs = {}
for fp in sorted(X4.glob("claude_review_*.txt")):
    for line in fp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i, lab, note = line.split("\t", 2)
            assert int(i) not in labs and lab in ("是", "否"), line
            labs[int(i)] = (lab, note)
s = pd.read_csv(SHEET, dtype=str, encoding="utf-8-sig").fillna("")
assert sorted(labs) == list(range(len(s))), (len(labs), len(s))
bak = SHEET.with_name(f"x4_gold_review.before_claude_{time.strftime('%Y%m%d_%H%M%S')}.csv")
shutil.copy2(SHEET, bak)
s["你的判断（是/否）"] = [labs[i][0] for i in range(len(s))]
s["备注"] = ["Claude 复核：" + labs[i][1] for i in range(len(s))]
s.to_csv(SHEET, index=False, encoding="utf-8-sig")

pre = pd.read_parquet(X4 / "gold_prelabel.parquet")
fin = x4_gold.import_review(pre, SHEET)
fin["gold_source"] = fin["gold_source"].replace({"user": "claude_review"})
assert fin["gold_label"].notna().all() and fin.attrs["n_unreviewed"] == 0
fin.to_parquet(X4 / "gold_final.parquet")

rv = fin[fin["needs_review"]]
agree = {}
for reason, g in rv.groupby("review_reason"):
    agree[reason] = {"n": len(g), "claude_eq_ds": int((g["user_label"] == g["ds_label"]).sum()),
                     "claude_eq_rule": int((g["user_label"] == g["rule_label"]).sum()),
                     "rule_undetermined": int(g["rule_label"].isna().sum())}
flips = rv[(rv["review_reason"] == "随机抽查") & (rv["user_label"] != rv["ds_label"])][["gold_id", "gold_for", "title", "ds_label", "user_label"]]
res = {"n_gold": len(fin), "label_pos": int((fin["gold_label"] == "是").sum()),
       "by_source": fin["gold_source"].value_counts().to_dict(), "review_agreement": agree,
       "spot_check_flips": flips.to_dict("records"),
       "by_type_pos": fin.groupby("gold_for")["gold_label"].apply(lambda x: f"{(x == '是').sum()}/{len(x)}").to_dict(),
       "by_stratum_pos": fin.groupby("stratum")["gold_label"].apply(lambda x: f"{(x == '是').sum()}/{len(x)}").to_dict(),
       "sheet_backup": str(bak)}
sc = x4_gold.score_gold(fin, PACK, X4 / "results_local")
res["score"] = {k: {"all": v["all"], "macro_f1": v["macro_f1"], "severe": v["severe"],
                    "by_type_f1": {t: round(b["f1"], 3) for t, b in v["by_type"].items()}} for k, v in sc.items()}
(X4 / "gold_score_local.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
