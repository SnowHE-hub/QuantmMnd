#!/usr/bin/env python3
"""QuantMind 流水线编排器（无需 Prefect/Temporal，原生 Python 实现）.

功能：
  - 任务依赖图：指定依赖后自动顺序 / 并行执行
  - 失败重试：可配置次数和间隔
  - 进度实时打印：每个任务的耗时和状态
  - 完成通知：通过 notify.py 推送微信/手机
  - 断点续跑：已完成的任务自动跳过（基于状态文件）

快速启动（step4完成后）：
  python scripts/pipeline_orchestrator.py --phase P1

完成所有阶段（step2+step4都完成后）：
  python scripts/pipeline_orchestrator.py --phase all

查看任务状态：
  python scripts/pipeline_orchestrator.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "logs" / "pipeline_state.json"
PYTHON = ROOT / ".." / ".." / "miniforge3" / "envs" / "quantmind" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


# ── 通知工具 ─────────────────────────────────────────────────────────────────

def _notify(title: str, content: str) -> None:
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.notify import notify
        notify(title, content, silent=True)
    except Exception:
        pass


# ── 任务状态 ──────────────────────────────────────────────────────────────────

@dataclass
class TaskState:
    task_id: str
    status: str = "pending"    # pending / running / done / failed / skipped
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    exit_code: int = -1
    retries: int = 0


def _load_state() -> dict[str, TaskState]:
    if STATE_FILE.exists():
        raw = json.loads(STATE_FILE.read_text())
        return {k: TaskState(**v) for k, v in raw.items()}
    return {}


def _save_state(state: dict[str, TaskState]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({k: asdict(v) for k, v in state.items()}, indent=2, ensure_ascii=False)
    )


# ── 任务定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    description: str
    cmd: list[str]                    # 命令列表（传给 subprocess）
    depends_on: list[str] = field(default_factory=list)
    max_retries: int = 1
    retry_wait_sec: int = 30
    log_file: str = ""                # 空=直接打印到 stdout
    check_fn: Callable[[], bool] | None = None  # 前置检查（如文件是否存在）


# ── 所有任务注册 ───────────────────────────────────────────────────────────────

def build_tasks() -> dict[str, Task]:
    py = str(PYTHON)
    tasks: dict[str, Task] = {}

    # ─ Phase 1（step4完成后，不依赖step2）─────────────────────────────────────

    tasks["step3"] = Task(
        task_id="step3",
        description="Step3: 下载季末 daily_basic（PE/PB/市值）",
        cmd=[py, "-u", "scripts/data_pipeline/step3_download_daily_basic.py"],
        log_file="logs/step3.log",
        max_retries=2,
        check_fn=lambda: (ROOT / "data/alpha_universe/alpha_daily_basic_combined.parquet").exists(),
    )

    tasks["train_valuation_v3"] = Task(
        task_id="train_valuation_v3",
        description="ValuationAgent v3 训练（行业相对PE/PEG/ROE趋势）",
        cmd=[
            py, "-u", "scripts/train_valuation_agent_v3.py",
            "--panel", "data/panel/monthly_panel_v2.parquet",
            "--fina",  "data/alpha_universe/fina_indicator_combined.parquet",
            "--basic", "data/alpha_universe/alpha_daily_basic_combined.parquet",
            "--out",   "models/agents/valuation_lgbm_v3.pkl",
        ],
        depends_on=["step3"],
        log_file="logs/train_valuation_v3.log",
    )

    tasks["train_risk_v3"] = Task(
        task_id="train_risk_v3",
        description="RiskAgent v3 重训（HMM + CVaR，1374只）",
        cmd=[
            py, "-u", "scripts/train_risk_agent_v3.py",
            "--prices", "data/raw/alpha_prices_panel.parquet",
            "--out",    "models/agents/risk_hmm_v3.pkl",
        ],
        log_file="logs/risk_v3.log",
        check_fn=lambda: (ROOT / "data/raw/alpha_prices_panel.parquet").exists(),
    )

    # ─ Phase 2（step2完成后）──────────────────────────────────────────────────

    tasks["build_prices_wide"] = Task(
        task_id="build_prices_wide",
        description="构建价格宽表（PatchTST训练用）",
        cmd=[
            py, "-u", "-c",
            """
import pandas as pd, sys
from pathlib import Path
ROOT = Path('.')
df = pd.read_parquet('data/raw/alpha_prices_panel.parquet')
col = 'adj_close' if 'adj_close' in df.columns else 'close'
df['trade_date'] = pd.to_datetime(df['trade_date'])
wide = df.pivot_table(index='trade_date', columns='ts_code', values=col)
wide.to_parquet('data/alpha_universe/alpha_prices_wide.parquet')
print(f'宽表: {wide.shape}')
""",
        ],
        log_file="logs/build_prices_wide.log",
        check_fn=lambda: (
            (ROOT / "data/raw/alpha_prices_panel.parquet").exists() and
            pd_shape_ok(ROOT / "data/raw/alpha_prices_panel.parquet", min_rows=2_000_000)
        ),
    )

    tasks["build_full_panel"] = Task(
        task_id="build_full_panel",
        description="构建1374只全特征面板（63因子+前视标签）",
        cmd=[
            py, "-u", "scripts/build_full_panel.py",
            "--snapshot-dir", "data/snapshots",
            "--price-panel",  "data/raw/alpha_prices_panel.parquet",
            "--universe",     "data/alpha_universe/alpha_universe.txt",
            "--output",       "data/panel/alpha_panel_v1.parquet",
        ],
        depends_on=["build_prices_wide"],
        log_file="logs/build_full_panel.log",
        max_retries=1,
    )

    tasks["build_split"] = Task(
        task_id="build_split",
        description="切分 train/val/test panel",
        cmd=[
            py, "-u", "scripts/build_train_test_split.py",
            "--panel",     "data/panel/alpha_panel_v1.parquet",
            "--train-end", "2022-12-31",
            "--val-end",   "2023-12-31",
            "--output",    "data/panel/",
        ],
        depends_on=["build_full_panel"],
        log_file="logs/build_split.log",
    )

    # ─ Phase 3（并行训练，panel split后）──────────────────────────────────────

    tasks["train_lgbm_alpha"] = Task(
        task_id="train_lgbm_alpha",
        description="主因子模型（漏斗Layer5，1374只口径）",
        cmd=[
            py, "-u", "scripts/train_lgbm_model.py",
            "--train", "data/panel/alpha_train.parquet",
            "--val",   "data/panel/alpha_val.parquet",
            "--test",  "data/panel/alpha_test.parquet",
            "--model-output", "models/lgbm_v2_alpha1374.pkl",
            "--n-estimators", "600",
        ],
        depends_on=["build_split"],
        log_file="logs/train_lgbm_alpha.log",
    )

    tasks["train_patchtst"] = Task(
        task_id="train_patchtst",
        description="MomentumAgent PatchTST v4（1374只×7年）",
        cmd=[
            py, "-u", "scripts/train_momentum_patchtst.py",
            "--panel",  "data/alpha_universe/alpha_prices_wide.parquet",
            "--ohlcv",  "data/raw/alpha_prices_panel.parquet",
            "--out",    "models/agents/momentum_patchtst_v4.pt",
            "--epochs", "30",
            "--batch",  "512",
        ],
        depends_on=["build_split"],
        log_file="logs/train_patchtst.log",
    )

    tasks["backtest_factor"] = Task(
        task_id="backtest_factor",
        description="因子分层回测（Q1~Q5净值曲线）",
        cmd=[
            py, "-u", "scripts/run_backtest_factor.py",
            "--panel",  "data/panel/alpha_panel_v1.parquet",
            "--top-n",  "50",
            "--output", "reports/backtest_alpha1374/",
        ],
        depends_on=["build_split"],
        log_file="logs/backtest_factor.log",
    )

    # ─ Phase 4（激活注册表）──────────────────────────────────────────────────

    tasks["activate_models"] = Task(
        task_id="activate_models",
        description="激活新模型版本（lgbm_v3 / patchtst_v4 / hmm_garch_v3）",
        cmd=[
            py, "-u", "-c",
            """
import json, shutil, sys
from pathlib import Path
reg_path = Path('data/agent_models/registry.json')
reg = json.loads(reg_path.read_text())
for r in reg.get('ValuationAgent', []):
    r['is_active'] = (r['model_version'] == 'lgbm_v3')
for r in reg.get('MomentumAgent', []):
    r['is_active'] = (r['model_version'] == 'patchtst_v4')
for r in reg.get('RiskAgent', []):
    r['is_active'] = (r['model_version'] == 'hmm_garch_v3')
reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
# 替换主因子模型
src = Path('models/lgbm_v2_alpha1374.pkl')
if src.exists():
    shutil.copy(src, 'models/lgbm_v1_final.pkl')
    print('主因子模型已更新')
print('注册表已更新')
""",
        ],
        depends_on=["train_lgbm_alpha", "train_patchtst", "train_valuation_v3", "train_risk_v3"],
        log_file="logs/activate_models.log",
    )

    # ─ Phase 5（验证）─────────────────────────────────────────────────────────

    tasks["build_kb"] = Task(
        task_id="build_kb",
        description="RAG知识库重建（34期快照）",
        cmd=[py, "-u", "scripts/build_kb_all_snapshots.py"],
        log_file="logs/build_kb.log",
        max_retries=1,
    )

    tasks["funnel_test"] = Task(
        task_id="funnel_test",
        description="全市场漏斗测试（5500→15只，验证无fallback）",
        cmd=[
            py, "-u", "scripts/run_funnel_selection.py",
            "--universe", "full_a",
        ],
        depends_on=["activate_models"],
        log_file="logs/funnel_fullmarket.log",
    )

    return tasks


def pd_shape_ok(path: Path, min_rows: int) -> bool:
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return len(df) >= min_rows
    except Exception:
        return False


# ── 执行引擎 ──────────────────────────────────────────────────────────────────

def run_task(task: Task, state: dict[str, TaskState], force: bool = False) -> bool:
    ts = state.setdefault(task.task_id, TaskState(task_id=task.task_id))

    # 已完成且非强制
    if ts.status == "done" and not force:
        print(f"  [{task.task_id}] 已完成，跳过（使用 --force 重跑）")
        return True

    # 前置检查
    if task.check_fn and not task.check_fn():
        print(f"  [{task.task_id}] 前置条件不满足，跳过")
        ts.status = "skipped"
        _save_state(state)
        return False

    # 检查依赖
    for dep in task.depends_on:
        dep_state = state.get(dep)
        if not dep_state or dep_state.status != "done":
            print(f"  [{task.task_id}] 依赖 [{dep}] 未完成，跳过")
            ts.status = "skipped"
            _save_state(state)
            return False

    print(f"\n{'='*60}")
    print(f"[{task.task_id}] {task.description}")
    print(f"{'='*60}")

    log_path = ROOT / task.log_file if task.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, task.max_retries + 2):
        ts.status = "running"
        ts.started_at = datetime.now().isoformat()
        ts.retries = attempt - 1
        _save_state(state)

        t0 = time.time()
        try:
            if log_path:
                print(f"  日志: {log_path}")
                with open(log_path, "w") as log_f:
                    proc = subprocess.run(
                        task.cmd, cwd=ROOT,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    )
                    log_f.write(proc.stdout.decode("utf-8", errors="replace"))
            else:
                proc = subprocess.run(task.cmd, cwd=ROOT)

            elapsed = time.time() - t0
            ts.elapsed_sec = elapsed
            ts.finished_at = datetime.now().isoformat()
            ts.exit_code = proc.returncode

            if proc.returncode == 0:
                ts.status = "done"
                _save_state(state)
                print(f"  [{task.task_id}] 完成 ({elapsed:.0f}s)")
                return True
            else:
                print(f"  [{task.task_id}] 失败（exit={proc.returncode}）尝试 {attempt}/{task.max_retries+1}")
                if log_path:
                    tail = _tail_log(log_path, 10)
                    print(f"  最后10行日志:\n{tail}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{task.task_id}] 异常: {e}")

        if attempt <= task.max_retries:
            print(f"  等待 {task.retry_wait_sec}s 后重试...")
            time.sleep(task.retry_wait_sec)

    ts.status = "failed"
    _save_state(state)
    _notify(f"QuantMind 任务失败", f"{task.task_id}: {task.description}")
    return False


def _tail_log(path: Path, n: int = 10) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(f"    {l}" for l in lines[-n:])
    except Exception:
        return "  (日志读取失败)"


def run_phase(phase_tasks: list[str], tasks: dict[str, Task],
              state: dict[str, TaskState], parallel: bool = False) -> bool:
    """运行一组任务（可并行）."""
    if not parallel:
        for tid in phase_tasks:
            if tid not in tasks:
                continue
            ok = run_task(tasks[tid], state)
            if not ok:
                return False
        return True
    else:
        # 简化并行：用子进程（后台）
        import multiprocessing
        results = {}

        def _run(tid):
            ok = run_task(tasks[tid], state)
            results[tid] = ok

        procs = []
        for tid in phase_tasks:
            if tid not in tasks:
                continue
            p = multiprocessing.Process(target=_run, args=(tid,))
            p.start()
            procs.append((tid, p))

        for tid, p in procs:
            p.join()

        return all(state.get(tid, TaskState(tid)).status == "done" for tid, _ in procs)


def print_status(state: dict[str, TaskState], tasks: dict[str, Task]) -> None:
    print(f"\n{'='*70}")
    print(f"{'任务ID':<25} {'状态':<10} {'耗时':<10} {'更新时间'}")
    print(f"{'='*70}")
    for tid, task in tasks.items():
        ts = state.get(tid, TaskState(tid))
        elapsed = f"{ts.elapsed_sec:.0f}s" if ts.elapsed_sec > 0 else "-"
        updated = ts.finished_at[:16] if ts.finished_at else ts.started_at[:16] if ts.started_at else "-"
        status_icon = {"done": "✓", "failed": "✗", "running": "→", "pending": "·", "skipped": "○"}.get(ts.status, "?")
        print(f"  {tid:<23} {status_icon} {ts.status:<8} {elapsed:<10} {updated}")
    print()


# ── 主入口 ────────────────────────────────────────────────────────────────────

PHASE_MAP = {
    "P1": {
        "desc": "step4完成后：step3 + ValuationAgent v3 + RiskAgent v3",
        "groups": [["step3"], ["train_valuation_v3", "train_risk_v3"]],
        "parallel": [False, True],
    },
    "P2": {
        "desc": "step2完成后：构建价格宽表 → 全特征面板 → 切分",
        "groups": [["build_prices_wide"], ["build_full_panel"], ["build_split"]],
        "parallel": [False, False, False],
    },
    "P3": {
        "desc": "并行训练：LGBM + PatchTST + 因子回测",
        "groups": [["train_lgbm_alpha", "train_patchtst", "backtest_factor"]],
        "parallel": [True],
    },
    "P4": {
        "desc": "激活新模型注册表",
        "groups": [["activate_models"]],
        "parallel": [False],
    },
    "P5": {
        "desc": "端到端验证：漏斗测试 + KB重建",
        "groups": [["build_kb", "funnel_test"]],
        "parallel": [True],
    },
}


def main():
    parser = argparse.ArgumentParser(description="QuantMind 流水线编排器")
    parser.add_argument("--phase", choices=["P1", "P2", "P3", "P4", "P5", "all", "status"], default="status")
    parser.add_argument("--force", action="store_true", help="强制重跑已完成任务")
    args = parser.parse_args()

    tasks = build_tasks()
    state = _load_state()

    if args.phase == "status":
        print_status(state, tasks)
        return

    phases_to_run = list(PHASE_MAP.keys()) if args.phase == "all" else [args.phase]
    total_start = time.time()

    for phase_id in phases_to_run:
        phase = PHASE_MAP[phase_id]
        print(f"\n{'#'*60}")
        print(f"# {phase_id}: {phase['desc']}")
        print(f"{'#'*60}")

        for group, parallel in zip(phase["groups"], phase["parallel"]):
            ok = run_phase(group, tasks, state, parallel=parallel)
            if not ok and not args.force:
                print(f"\n[{phase_id}] 遇到失败，中止后续任务")
                _notify(f"QuantMind {phase_id} 中止", "有任务失败，请检查日志")
                return

    elapsed = time.time() - total_start
    print(f"\n[完成] 总耗时 {elapsed/60:.1f} 分钟")
    print_status(state, tasks)
    _notify(
        f"QuantMind {args.phase} 全部完成",
        f"总耗时 {elapsed/60:.0f} 分钟，请查看 Dashboard"
    )


if __name__ == "__main__":
    main()
