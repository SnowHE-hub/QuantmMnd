"""系统命令执行器 — 异步运行脚本，捕获实时输出."""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator

PYTHON = "/home/lenovo/miniforge3/envs/quantmind/bin/python"
ROOT   = Path(__file__).resolve().parents[4]

# ── 可用命令注册表 ────────────────────────────────────────────────────────────
COMMANDS: dict[str, dict] = {
    "simulate": {
        "label": "30日全A股模拟",
        "cmd": [PYTHON, "scripts/run_30day_sim.py", "--step", "simulate"],
        "desc": "运行三系统选股模拟（约需5-15分钟）",
        "icon": "🚀",
        "quick": False,
    },
    "simulate_evaluate": {
        "label": "绩效评估",
        "cmd": [PYTHON, "scripts/run_30day_sim.py", "--step", "evaluate"],
        "desc": "计算各持仓期收益统计",
        "icon": "📈",
        "quick": True,
    },
    "optimize": {
        "label": "优化分析",
        "cmd": [PYTHON, "scripts/optimize_30day_results.py"],
        "desc": "因子IC分析 + System2权重校准 + realized_pnl扩充",
        "icon": "📊",
        "quick": False,
    },
    "backtest": {
        "label": "NAV回测",
        "cmd": [PYTHON, "scripts/run_nav_backtest.py"],
        "desc": "运行季度NAV净值曲线回测",
        "icon": "📉",
        "quick": False,
    },
    "train_meta": {
        "label": "重训 meta-learner",
        "cmd": [PYTHON, "scripts/train_meta_learner.py",
                "--pnl", "data/feedback/realized_pnl.parquet"],
        "desc": "基于379条realized_pnl重训 meta-learner",
        "icon": "🧠",
        "quick": False,
    },
    "build_features": {
        "label": "构建特征面板",
        "cmd": [PYTHON, "scripts/build_features.py"],
        "desc": "重建v4特征面板（alpha_panel_v4）",
        "icon": "🔧",
        "quick": False,
    },
}


class ExecutionResult:
    def __init__(self):
        self.lines: list[str] = []
        self.returncode: int = -1
        self.elapsed: float = 0.0
        self.success: bool = False

    def add_line(self, line: str):
        self.lines.append(line)

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "success": self.success,
            "elapsed": round(self.elapsed, 1),
            "lines": self.lines,
            "output": self.output,
        }


async def run_command_stream(cmd_key: str) -> AsyncGenerator[str, None]:
    """异步流式运行命令，逐行 yield 输出."""
    if cmd_key not in COMMANDS:
        yield f"[错误] 未知命令: {cmd_key}\n"
        return

    cmd = COMMANDS[cmd_key]["cmd"]
    label = COMMANDS[cmd_key]["label"]
    t0 = time.time()

    yield f"[开始] {label}\n"
    yield f"[命令] {' '.join(cmd)}\n"
    yield f"[工作目录] {ROOT}\n\n"

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(ROOT),
    )

    async for line in proc.stdout:  # type: ignore
        text = line.decode("utf-8", errors="replace")
        yield text

    await proc.wait()
    elapsed = time.time() - t0
    status = "✅ 成功" if proc.returncode == 0 else f"❌ 失败(code={proc.returncode})"
    yield f"\n[完成] {status} | 耗时 {elapsed:.1f}s\n"


def run_command_sync(cmd_key: str, timeout: int = 600) -> ExecutionResult:
    """同步运行命令（Streamlit 调用用）."""
    result = ExecutionResult()
    if cmd_key not in COMMANDS:
        result.lines = [f"[错误] 未知命令: {cmd_key}"]
        return result

    cmd = COMMANDS[cmd_key]["cmd"]
    label = COMMANDS[cmd_key]["label"]
    result.add_line(f"[开始] {label}")
    result.add_line(f"[命令] {' '.join(cmd)}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        result.returncode = proc.returncode
        result.success = proc.returncode == 0
        for line in (proc.stdout + proc.stderr).splitlines():
            result.add_line(line)
    except subprocess.TimeoutExpired:
        result.add_line(f"[超时] 命令执行超过 {timeout}s")
    except Exception as e:
        result.add_line(f"[异常] {e}")

    result.elapsed = time.time() - t0
    status = "✅ 成功" if result.success else "❌ 失败"
    result.add_line(f"\n[完成] {status} | 耗时 {result.elapsed:.1f}s")
    return result
