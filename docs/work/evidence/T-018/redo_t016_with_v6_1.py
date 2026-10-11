"""T-018：用 B 的 v6_1 连续样本外（C 方案）63d 分数重做 T-016 的评估（只换 B 的预测文件，其余规则不变）.

复用分支 t/016-eval 的 docs/work/evidence/T-016/eval_a_with_b_scores.py（需要 t/010 的 scripts/run_line_a_oos.py）：
对齐 ≤ 14 天、四个变体、主比较 = 只换打分模型。原 B 预测每年只覆盖约一半，T-016 只有 8 次半年调仓；
C 方案时间轴连续，A 的每个季末调仓日都能对齐。
用法：python redo_t016_with_v6_1.py <qm-t016e 工作区> <data_root> <b_preds 相对 data_root> <out.json>
"""
import importlib.util
import sys
from pathlib import Path

wt, root, b_preds, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, str(wt))
spec = importlib.util.spec_from_file_location("t016eval", wt / "docs/work/evidence/T-016/eval_a_with_b_scores.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.B_PREDS = b_preds
sys.argv = ["eval_a_with_b_scores.py", "--data-root", root,
            "--a-scores", str(wt / "docs/work/evidence/T-010/run_v4_1/oos_scores.csv"), "--out", out]
mod.main()
