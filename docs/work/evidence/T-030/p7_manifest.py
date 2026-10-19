"""P7：给结论依赖的证据文件和预测折文件写 sha256。不读文件里的 IC。

用法（仓库根目录）：python docs/work/evidence/T-030/p7_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DATA = Path("/home/lenovo/projects/quantmind/data/gpu_v1/preds")
OUT = REPO / "docs/work/evidence/T-030/p7/data_sha256.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    evidence = REPO / "docs/work/evidence/T-030/p6"
    for path in sorted(evidence.iterdir()):
        if path.is_file() and not path.name.endswith(".tmp"):
            rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted(DATA.glob("*/*.parquet")):
        if path.name.startswith("fold_"):
            rows.append({"path": f"data/gpu_v1/preds/{path.parent.name}/{path.name}",
                         "sha256": sha256(path), "bytes": path.stat().st_size})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_files": len(rows), "files": rows}, indent=1), encoding="utf-8")
    print(f"wrote {OUT} n_files={len(rows)}")


if __name__ == "__main__":
    main()
