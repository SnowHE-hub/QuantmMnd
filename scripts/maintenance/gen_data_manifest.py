"""生成 docs/data_manifest.md：data/ 下所有 parquet 的权威清单（文件名/行数/sha256/大小）。
作为"该存在哪些 parquet"的权威来源（产物不入库，靠此 + run_manifest sha256 溯源）。"""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIRS = ["data/lake", "data/panel", "data/raw", "data/bakeoff", "data/bakeoff/preds",
        "data/loss_signals_v4"]


def sha256(p: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def nrows(p: Path):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return None


def main():
    lines = ["# Data Manifest（data/ parquet 权威清单）", "",
             "> 大数据产物不入库（见 .gitignore）。本清单 = \"该存在哪些 parquet\" 的权威来源；",
             "> 配合各 run_manifest_*.json 的 artifacts_sha256 溯源。生成：`scripts/maintenance/gen_data_manifest.py`。",
             "", "| 文件 | 行数 | 大小 | sha256[:16] |", "|---|---|---|---|"]
    seen = set()
    total_sz = 0
    for d in DIRS:
        for p in sorted((REPO / d).glob("*.parquet")):
            if p in seen:
                continue
            seen.add(p)
            sz = p.stat().st_size
            total_sz += sz
            rel = p.relative_to(REPO)
            n = nrows(p)
            nstr = f"{n:,}" if isinstance(n, int) else "—"
            lines.append(f"| `{rel}` | {nstr} | {sz/1e6:.1f} MB | `{sha256(p)[:16]}` |")
            print(f"  {rel}", flush=True)
    lines += ["", f"**合计：{len(seen)} 个 parquet，{total_sz/1e9:.2f} GB**", "",
              "## 产出脚本对照",
              "- `data/raw/alpha_prices_panel_v6.parquet` ← scripts/survivorship/p2_build_v6.py --prices",
              "- `data/panel/alpha_panel_weekly_v6.parquet` ← p2_build_v6.py --panel",
              "- `data/panel/fundamental_factors_v6.parquet` ← p63_2_build_factors.py",
              "- `data/lake/{daily,adj_factor,daily_basic,fina_indicator,income,balancesheet,cashflow}.parquet` ← scripts/survivorship/p1_pull.py / p63_1*_pull*.py",
              "- `data/bakeoff/preds/ridge_*_v6.parquet` ← p4c_train_v6.py / p63_3_train_v6.py",
              "- `data/bakeoff/alpha158_asof_v6.parquet` ← p4b_alpha158_v6.py (qlib_bakeoff env)",
              "- 各 `run_manifest_*.json` 内 artifacts_sha256 = 落盘时预录哈希（防静默改写）。"]
    out = REPO / "docs/data_manifest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[manifest] {out} : {len(seen)} parquet, {total_sz/1e9:.2f} GB")


if __name__ == "__main__":
    main()
