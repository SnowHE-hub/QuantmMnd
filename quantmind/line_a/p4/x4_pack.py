"""quantmind.line_a.p4.x4_pack — 导出 X4 的 Colab 题包（VT 协议；本地出题、云端只执行、本地按哈希收分）.

题包内容：
- train.jsonl：训练题；
- cells.jsonl：校准 + 测试题，按 sha256(cell) 排序；
- manifest.json：每个文件的 sha256、协议版本、抽样规则与种子、行数；
- x4_cloud.py：云端运行器，自包含。

每题一行：{cell, ann_id, type, split, stratum, label, user, sha}，其中 sha = sha256(user)。
抽样（每个类型 T，只用 keep = True 的公告）：
- 正例：label_T = pos_strong（两个来源一致），或 pos_rule（只有严格标题规则；稀有类型），来源记在 label_source；
- 负例：一半是"难负例"（标题含 T 的关键词但 label_T = neg），一半随机；
- 数量：训练 ≤ 1000 + 1000，校准 ≤ 150 + 150，测试 ≤ 300 + 300；
- uncertain 不进入任何集合，单独留给人工金标准。

用法：python -m quantmind.line_a.p4.x4_pack <x4_dir> <out_dir>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

from quantmind.line_a.p4.x4_dataset import TYPES

PROTOCOL = "x4-vt-v1"
QUOTA = {"train": 1000, "calib": 150, "test": 300}
KEYWORDS = {"performance_forecast": "业绩", "performance_express": "快报", "repurchase": "回购", "holder_reduction": "减持",
            "holder_increase": "增持", "unlock": "限售", "dividend": "分配", "st_change": "风险警示", "investigation": "调查",
            "penalty": "处罚", "regulatory_letter": "函", "litigation": "诉讼", "restructuring": "重组", "control_change": "控制"}
SEED = 20261021


def message(type_key: str, title: str) -> str:
    desc = TYPES[type_key][0]
    return (f"判断下面这条 A 股上市公司公告的标题是否属于这一类事件：{desc}。\n"
            f"只根据标题判断，只回答 Yes 或 No。\n公告标题：{title}")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    k = df[df["keep"]]
    for i, t in enumerate(TYPES):
        lab = k[f"label_{t}"]
        for split, q in QUOTA.items():
            ks = k[k["split"] == split]
            ls = lab[ks.index]
            pos = ks[ls.isin(["pos_strong", "pos_rule"])]
            neg = ks[ls == "neg"]
            hard = neg[neg["title"].str.contains(KEYWORDS[t], regex=False)]
            rest = neg.drop(hard.index)
            n = min(q, len(pos))
            p = pos.sample(n, random_state=SEED + i) if len(pos) > n else pos
            nh = min(len(hard), max(1, n // 2))
            h = hard.sample(nh, random_state=SEED + 100 + i) if len(hard) > nh else hard
            r = rest.sample(min(len(rest), n - len(h) if n > len(h) else n // 2), random_state=SEED + 200 + i)
            for stratum, part, y in (("pos", p, 1), ("hard_neg", h, 0), ("rand_neg", r, 0)):
                for idx, row in part.iterrows():
                    user = message(t, row["title"])
                    cell = _sha(f"{row['announcementId']}:{t}")[:16]
                    src = ls[idx] if y == 1 else "neg"
                    rows.append({"cell": cell, "ann_id": str(row["announcementId"]), "type": t, "split": split,
                                 "stratum": stratum, "label": y, "label_source": src, "date": row["date"],
                                 "user": user, "sha": _sha(user)})
    return rows


def export(x4_dir: Path, out_dir: Path) -> Path:
    x4_dir, out_dir = Path(x4_dir), Path(out_dir)
    df = pd.concat([pd.read_parquet(p, columns=["announcementId", "date", "title", "split", "keep",
                                                *[f"label_{t}" for t in TYPES]])
                    for p in sorted(x4_dir.glob("x4_*.parquet"))], ignore_index=True)
    rows = make_rows(df)
    ann_by_split: dict[str, set] = {}
    for r in rows:
        ann_by_split.setdefault(r["split"], set()).add(r["ann_id"])
    assert not (ann_by_split.get("train", set()) & (ann_by_split.get("calib", set()) | ann_by_split.get("test", set())))
    pack = out_dir / "pack_x4_v1"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    train = [r for r in rows if r["split"] == "train"]
    cells = sorted([r for r in rows if r["split"] != "train"], key=lambda r: _sha(r["cell"]))
    for name, rs in (("train.jsonl", train), ("cells.jsonl", cells)):
        with (pack / name).open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(Path(__file__).with_name("x4_cloud.py"), pack / "x4_cloud.py")
    files = {n: {"sha256": hashlib.sha256((pack / n).read_bytes()).hexdigest()} for n in ("train.jsonl", "cells.jsonl",
                                                                                           "x4_cloud.py")}
    counts = pd.DataFrame(rows).groupby(["split", "type", "label"]).size().unstack(fill_value=0)
    man = {"protocol": PROTOCOL, "files": files, "rows": {"train": len(train), "cells": len(cells)}, "seed": SEED,
           "quota": QUOTA, "label_source": "silver: title rule agreeing with a structured table or cninfo category; "
           "uncertain excluded", "counts": {f"{s}|{t}": [int(v) for v in row] for (s, t), row in counts.iterrows()}}
    (pack / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    zp = out_dir / "pack_x4_v1.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pack.iterdir()):
            z.write(f, f"pack_x4_v1/{f.name}")
    print(json.dumps({"zip": str(zp), "rows": man["rows"]}, ensure_ascii=False))
    return zp


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))


NOTEBOOK_INTRO = """# QuantMind X4 · Colab 运行本（公告分类，VT 协议：零样本 + LoRA 微调）

**操作步骤：**

1. 打开 [drive.google.com](https://drive.google.com)，在「我的云端硬盘」里新建文件夹 **`qm_x4`**，把 **`pack_x4_v1.zip`** 拖进去（不要解压）。
2. 菜单 **代码执行程序（Runtime）→ 更改运行时类型 → 硬件加速器选 `A100 GPU`**（没有就选 `L4 GPU`）→ 保存。
3. 从上到下逐个运行单元（点左边的 ▶ 或按 `Ctrl+Enter`）。第 3 个单元会弹出 Google 授权窗口：这是你自己的云端硬盘，点允许。
4. 先跑 **④ 冒烟**，看到 `SMOKE OK` 再跑 **⑤ 正式队列**。
5. 运行期间 **保持这个标签页打开、电脑不睡眠**。断线了就重新连接，从 ① 开始全部重跑一遍：已完成的任务和已算过的题会自动跳过。
6. 随时可以运行 **⑥ 进度**；右上角能看计算单元余额。队列按优先级排好，单元不够时跑完前两项也有用。
7. 全部完成（或想先交一部分）：在云端硬盘里右键 **`qm_x4/out`** → 下载，把 zip 放到电脑的
   `D:/MCS_26S2课程资料/quantmind_colab_results/`，告诉 Claude。

题包只含公开的公告标题，不含任何账户或持仓信息。"""
JOBS = [("zs", "q35-2b"), ("ft", "q35-2b"), ("ft", "q35-4b"), ("zs", "q35-9b")]


def write_notebook(path: Path) -> Path:
    def md(s):
        return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.splitlines(keepends=True)}
    cells = [
        md(NOTEBOOK_INTRO),
        md("## ① 设置与 GPU 检查"),
        code("JOBS = " + repr(JOBS) + "\nDRIVE_DIR = \"/content/drive/MyDrive/qm_x4\"\n"
             "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        md("## ② 安装依赖（约 2–4 分钟）"),
        code("import subprocess, sys\n"
             "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"transformers>=5.0\", \"peft>=0.17\",\n"
             "                \"accelerate\", \"safetensors\", \"bitsandbytes\"], check=True)\n"
             "subprocess.run([sys.executable, \"-m\", \"pip\", \"uninstall\", \"-y\", \"-q\", \"torchao\"], check=False)\n"
             "r = subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"flash-linear-attention\"], check=False)\n"
             "print(\"flash-linear-attention:\", \"ok\" if r.returncode == 0 else \"not installed (fallback)\")\n"
             "import torch, transformers, peft\n"
             "print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__, \"| peft\", peft.__version__)\n"
             "if int(transformers.__version__.split(\".\")[0]) < 5:\n"
             "    print(\"!! transformers 还是旧版本：菜单 Runtime → Restart session，然后从 ① 重新运行\")"),
        md("## ③ 连接云端硬盘、解压题包、校验"),
        code("from google.colab import drive\ndrive.mount(\"/content/drive\")\n"
             "import zipfile, sys, importlib\nfrom pathlib import Path\n"
             "zp = Path(DRIVE_DIR) / \"pack_x4_v1.zip\"\n"
             "assert zp.exists(), f\"云端硬盘里找不到 {zp}：确认文件夹名是 qm_x4、zip 在里面\"\n"
             "if not Path(\"/content/pack_x4_v1/manifest.json\").exists():\n"
             "    zipfile.ZipFile(zp).extractall(\"/content\")\n"
             "PACK = Path(\"/content/pack_x4_v1\")\nOUT = Path(DRIVE_DIR) / \"out\"\nOUT.mkdir(parents=True, exist_ok=True)\n"
             "sys.path.insert(0, str(PACK))\nimport x4_cloud as xc\nimportlib.reload(xc)\nxc.verify_pack(PACK)\n"
             "print(\"GPU 显存 GiB:\", round(xc.gpu_gib(), 1))"),
        md("## ④ 冒烟（约 5–10 分钟）：整条链路在几道题上走一遍，结果写在 out/_smoke，不进正式结果"),
        code("xc.smoke(PACK, OUT)"),
        md("## ⑤ 正式队列（A100 上前两项约 1.5 小时，全部约 4–5 小时；可以分几次跑完）"),
        code("xc.run_queue(JOBS, PACK, OUT)"),
        md("## ⑥ 进度"),
        code("xc.status(OUT)"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "accelerator": "GPU", "colab": {"gpuType": "A100"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
