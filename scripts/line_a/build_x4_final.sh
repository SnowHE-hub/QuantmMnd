#!/bin/bash
# T-038：巨潮公告元数据补完之后，构建 X4 正式数据集与 Colab 题包（方案第 49 条：VT 不拿部分数据训练）。
# - 完整的判定：检查点里 2019-01-01～2026-10-16 每一天都是 done 或 empty；
# - --wait：每 10 分钟查一次，完整后再开始；不带 --wait 时不完整就退出 2；
# - 构建时拿住巨潮下载的文件锁，并等每日影子运行（qm-daily-*）结束，避免两边同时占内存；
# - 产物：data/line_a_v2/p4/x4_v1/（按年数据、标签汇总、金标准样本、build_record.json），
#   以及 D:/MCS_26S2课程资料/quantmind_colab_upload/ 下的 pack_x4_v1.zip 和运行本。
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${QM_DATA_ROOT:-/home/lenovo/projects/quantmind}
CK=$DATA/data/lake/text/cninfo/_checkpoints/cninfo_meta.json
export QM_X4_OUT=$DATA/data/line_a_v2/p4/x4_v1 QM_X4_UP=/mnt/d/MCS_26S2课程资料/quantmind_colab_upload QM_CK=$CK

complete() {
  python3 - "$CK" <<'P'
import json, sys
from datetime import date, timedelta
d = json.load(open(sys.argv[1]))["days"]
want = {(date(2019, 1, 1) + timedelta(i)).isoformat() for i in range((date(2026, 10, 16) - date(2019, 1, 1)).days + 1)}
bad = [k for k in want if d.get(k, {}).get("state") not in ("done", "empty")]
print(f"{len(want)} days, {len(bad)} incomplete", flush=True)
sys.exit(0 if not bad else 1)
P
}

if [ "${1:-}" = "--wait" ]; then
  until complete; do sleep 600; done
else
  complete || exit 2
fi
while systemctl --user list-units 'qm-daily-*' --state=active --no-legend | grep -q .; do sleep 120; done

exec 9>/tmp/qm_cninfo_meta.lock
flock 9
cd "$REPO" || exit 1
echo "=== $(date '+%F %T %z') build (repo $(git rev-parse --short HEAD))"
PYTHONPATH=. python -u - <<'P'
import glob, hashlib, json, os, time
from pathlib import Path
import pandas as pd
from quantmind.line_a.p4.x4_dataset import build_chunked, gold_sample
from quantmind.line_a.p4.x4_pack import export, write_notebook
lake = Path(os.environ["QM_DATA_ROOT"] if "QM_DATA_ROOT" in os.environ else "/home/lenovo/projects/quantmind") / "data/lake"
out, up = Path(os.environ["QM_X4_OUT"]), Path(os.environ["QM_X4_UP"])
t0 = time.time()
s = build_chunked(lake, out)
s.to_csv(out / "label_summary.csv", index=False)
print(f"built in {time.time() - t0:.0f}s", flush=True)
test = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(out / "x4_202[4-9].parquet")))])
g = gold_sample(test)
g.to_csv(out / "gold_sample_todo.csv", index=False, encoding="utf-8-sig")
del test
up.mkdir(parents=True, exist_ok=True)
zp = export(out, up)
write_notebook(up / "QuantMind_X4_Colab.ipynb")
man = json.loads((up / "pack_x4_v1" / "manifest.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": os.popen("git rev-parse HEAD").read().strip(),
       "cninfo_checkpoint_sha256": sha(os.environ["QM_CK"]), "zip": str(zp), "zip_sha256": sha(zp),
       "manifest_files": man["files"], "rows": man["rows"], "gold_rows": len(g),
       "gold_by_stratum": g.groupby("stratum").size().to_dict(),
       "years": sorted(p.stem for p in out.glob("x4_*.parquet"))}
(out / "build_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
print("RECORD", json.dumps(rec, ensure_ascii=False), flush=True)
P
echo "=== $(date '+%F %T %z') exit $?"
