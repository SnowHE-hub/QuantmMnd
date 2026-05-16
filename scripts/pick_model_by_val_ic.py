#!/usr/bin/env python3
"""按验证集 Rank IC 选型：使用 val_IC_mean × direction（反映 auto_flip 后的有效信号）."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _effective_val_ic(m: dict) -> float:
    raw = m.get("val_IC_mean")
    if raw is None or (isinstance(raw, float) and raw != raw):
        return float("-inf")
    direction = int(m.get("direction") or 1)
    return float(raw) * direction


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics_a", type=Path)
    p.add_argument("metrics_b", type=Path)
    args = p.parse_args()
    ja = json.loads(Path(args.metrics_a).read_text(encoding="utf-8"))
    jb = json.loads(Path(args.metrics_b).read_text(encoding="utf-8"))
    ea, eb = _effective_val_ic(ja), _effective_val_ic(jb)

    print(
        f"A {Path(args.metrics_a).name}: val_IC_raw={ja.get('val_IC_mean')} "
        f"direction={ja.get('direction')} → effective={ea:.6f}"
    )
    print(
        f"B {Path(args.metrics_b).name}: val_IC_raw={jb.get('val_IC_mean')} "
        f"direction={jb.get('direction')} → effective={eb:.6f}"
    )

    if ea >= eb:
        chosen, jwin, path = "A", ja, args.metrics_a
        eff = ea
    else:
        chosen, jwin, path = "B", jb, args.metrics_b
        eff = eb

    mp = str(jwin.get("model_path", ""))
    best = (root / mp).resolve() if mp and not Path(mp).is_absolute() else Path(mp).resolve() if mp else Path()
    print(f"pick: {chosen}")
    print(f"Best model: {mp}  val_IC_effective={eff:.6f}")
    print(f"BEST_MODEL={best}")
    print(f"recommended_metrics_json: {Path(path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
