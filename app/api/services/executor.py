"""系统命令执行器 — 异步/同步运行脚本，捕获输出."""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator

PYTHON = "/home/lenovo/miniforge3/envs/quantmind/bin/python"
ROOT   = Path(__file__).resolve().parents[4]

# ── 命令注册表（只注册实际存在的脚本）────────────────────────────────────────
COMMANDS: dict[str, dict] = {
    # ── 核心：30日模拟流程 ─────────────────────────────────────────────────
    "simulate_evaluate": {
        "label": "📈 绩效评估",
        "short": "绩效评估",
        "cmd": [PYTHON, "scripts/run_30day_sim.py", "--step", "evaluate"],
        "desc": "基于已有数据重新计算各持仓期收益统计（约1-3分钟）",
        "group": "模拟盘",
        "quick": True,
        "timeout": 300,
        "color": "#0984E3",
    },
    "simulate": {
        "label": "🚀 运行30日模拟",
        "short": "30日模拟",
        "cmd": [PYTHON, "scripts/run_30day_sim.py", "--step", "simulate"],
        "desc": "重跑三系统选股模拟（需要原始数据，约5-20分钟）",
        "group": "模拟盘",
        "quick": False,
        "timeout": 1800,
        "color": "#6C5CE7",
    },
    # ── 核心：优化分析 ─────────────────────────────────────────────────────
    "optimize": {
        "label": "📊 IC优化分析",
        "short": "IC优化",
        "cmd": [PYTHON, "scripts/optimize_30day_results.py"],
        "desc": "因子IC分析 + System2权重校准 + realized_pnl扩充（约2-5分钟）",
        "group": "分析",
        "quick": False,
        "timeout": 600,
        "color": "#00B894",
    },
    # ── NAV回测 ────────────────────────────────────────────────────────────
    "backtest": {
        "label": "📉 NAV回测",
        "short": "NAV回测",
        "cmd": [PYTHON, "scripts/run_nav_backtest.py"],
        "desc": "构建日频NAV净值曲线，对比CSI300基准（约3-8分钟）",
        "group": "回测",
        "quick": False,
        "timeout": 900,
        "color": "#E17055",
    },
    # ── 模型训练 ───────────────────────────────────────────────────────────
    "train_lgbm": {
        "label": "🤖 训练LGBM模型",
        "short": "训LGBM",
        "cmd": [PYTHON, "scripts/train_lgbm_model.py"],
        "desc": "重训 lgbm_v6_alpha 模型（需要alpha_panel，约10-30分钟）",
        "group": "模型",
        "quick": False,
        "timeout": 3600,
        "color": "#FDCB6E",
    },
    # ── 数据构建 ───────────────────────────────────────────────────────────
    "build_features": {
        "label": "🔧 重建特征面板",
        "short": "建特征",
        "cmd": [PYTHON, "scripts/build_features.py"],
        "desc": "重建v4特征面板 alpha_panel_v4（约15-60分钟）",
        "group": "数据",
        "quick": False,
        "timeout": 3600,
        "color": "#636E72",
    },
}

# 按组分类
COMMAND_GROUPS = {
    "模拟盘": ["simulate_evaluate", "simulate"],
    "分析":   ["optimize"],
    "回测":   ["backtest"],
    "模型":   ["train_lgbm"],
    "数据":   ["build_features"],
}


class ExecutionResult:
    def __init__(self):
        self.lines: list[str] = []
        self.returncode: int  = -1
        self.elapsed: float   = 0.0
        self.success: bool    = False
        self.cmd_key: str     = ""
        self.cmd_label: str   = ""

    def add_line(self, line: str):
        self.lines.append(line)

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def tail(self, n: int = 50) -> str:
        return "\n".join(self.lines[-n:])

    def to_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "success":    self.success,
            "elapsed":    round(self.elapsed, 1),
            "lines":      self.lines,
            "output":     self.output,
            "cmd_key":    self.cmd_key,
            "cmd_label":  self.cmd_label,
        }


async def run_command_stream(cmd_key: str) -> AsyncGenerator[str, None]:
    """异步流式运行命令，逐行 yield 输出（供 SSE 端点使用）."""
    if cmd_key not in COMMANDS:
        yield f"[错误] 未知命令: {cmd_key}\n"
        return

    info  = COMMANDS[cmd_key]
    cmd   = info["cmd"]
    label = info["short"]
    t0    = time.time()

    yield f"▶ 开始执行：{label}\n"
    yield f"  命令: {' '.join(cmd)}\n"
    yield f"  目录: {ROOT}\n\n"

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
        )
        async for line in proc.stdout:  # type: ignore
            yield line.decode("utf-8", errors="replace")

        await proc.wait()
        elapsed = time.time() - t0
        status  = "✅ 成功" if proc.returncode == 0 else f"❌ 失败(code={proc.returncode})"
        yield f"\n{'─'*50}\n{status}  耗时 {elapsed:.1f}s\n"
    except Exception as e:
        yield f"\n[异常] {e}\n"


def run_command_sync(cmd_key: str, timeout: int | None = None) -> ExecutionResult:
    """同步阻塞运行命令，返回完整结果（供 FastAPI 后台任务使用）."""
    result          = ExecutionResult()
    result.cmd_key  = cmd_key

    if cmd_key not in COMMANDS:
        result.lines    = [f"[错误] 未知命令: {cmd_key}"]
        result.cmd_label = cmd_key
        return result

    info             = COMMANDS[cmd_key]
    cmd              = info["cmd"]
    result.cmd_label = info["short"]
    effective_timeout = timeout or info.get("timeout", 600)

    result.add_line(f"▶ {info['label']}")
    result.add_line(f"  命令: {' '.join(cmd)}")
    result.add_line(f"  超时限制: {effective_timeout}s")
    result.add_line("")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            encoding="utf-8",
            errors="replace",
        )
        result.returncode = proc.returncode
        result.success    = proc.returncode == 0
        combined = (proc.stdout or "") + (proc.stderr or "")
        for line in combined.splitlines():
            result.add_line(line)
    except subprocess.TimeoutExpired:
        result.success = False
        result.add_line(f"[超时] 超过 {effective_timeout}s 未完成")
    except FileNotFoundError as e:
        result.success = False
        result.add_line(f"[找不到脚本] {e}")
    except Exception as e:
        result.success = False
        result.add_line(f"[异常] {e}")

    result.elapsed = time.time() - t0
    result.add_line("")
    result.add_line(f"{'─'*50}")
    result.add_line(f"{'✅ 成功' if result.success else '❌ 失败'}  耗时 {result.elapsed:.1f}s")
    return result
