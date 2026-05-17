"""QuantMind FastAPI 后端 — 命令执行 + LLM 分析 + 数据 API."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.services import executor as exec_svc
from app.api.services import analyzer as ai_svc
from app.api.services import llm_client as llm

# ── App 初始化 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QuantMind API",
    description="量化投资系统后端 API — 命令执行 + AI 分析",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic 模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    provider: str = "dashscope"   # dashscope | ollama | deepseek
    model: str | None = None
    auto_execute: bool = True     # 是否自动执行识别到的命令


class AnalyzeRequest(BaseModel):
    data: dict
    data_type: str = "general"
    provider: str = "dashscope"


class ExecuteRequest(BaseModel):
    cmd_key: str
    provider: str = "dashscope"
    analyze_result: bool = True


class StreamRequest(BaseModel):
    cmd_key: str


# ── 健康检查 ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ollama_models": llm.list_ollama_models(),
        "dashscope_key": bool(llm.DASHSCOPE_API_KEY),
        "available_commands": list(exec_svc.COMMANDS.keys()),
    }


# ── 数据 API ──────────────────────────────────────────────────────────────────
@app.get("/api/data/summary")
async def data_summary():
    """返回系统关键指标摘要."""
    try:
        from app.utils.sim_data import (
            load_sim30d_days, load_sim30d_stock_returns,
            load_ic_analysis, load_realized_pnl,
            horizon_portfolio_ts, load_nav_curves,
        )
        days = load_sim30d_days()
        sr   = load_sim30d_stock_returns()
        rpnl = load_realized_pnl()
        navs = load_nav_curves()

        perf: dict[str, Any] = {}
        for hz in ["1w", "2w", "21d", "3m"]:
            ts = horizon_portfolio_ts(days, hz)
            if not ts.empty:
                mean = float(ts["mean"].mean())
                std  = float(ts["mean"].std()) if ts["mean"].std() > 0 else 0.0001
                perf[hz] = {
                    "mean_return": round(mean * 100, 2),
                    "win_rate":    round(float((ts["mean"] > 0).mean()) * 100, 1),
                    "ir":          round(mean / std, 3),
                    "cum_return":  round(float(ts["cum_return"].iloc[-1]) * 100, 2) if not ts.empty else 0,
                }

        ic = load_ic_analysis()
        ic_top = {}
        for factor, vals in ic.get("ic_final_picks", {}).items():
            ic_3m = vals.get("ic_3m")
            if ic_3m is not None:
                ic_top[factor] = round(ic_3m, 4)

        return {
            "simulation_days": len(days),
            "date_range": {
                "start": days[0]["date"] if days else None,
                "end":   days[-1]["date"] if days else None,
            },
            "total_analyzed": len(sr),
            "final_picks_total": int(sr["in_final"].sum()) if not sr.empty else 0,
            "realized_pnl_count": len(rpnl),
            "nav_strategies": list(navs.keys()),
            "performance": perf,
            "ic_3m_top": dict(sorted(ic_top.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/ic")
async def data_ic():
    try:
        from app.utils.sim_data import load_ic_analysis
        return load_ic_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/pnl")
async def data_pnl():
    try:
        from app.utils.sim_data import load_realized_pnl
        rpnl = load_realized_pnl()
        return {
            "count": len(rpnl),
            "columns": list(rpnl.columns),
            "sample": rpnl.tail(5).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/commands")
async def data_commands():
    """返回所有可用命令列表."""
    return {k: {kk: vv for kk, vv in v.items() if kk != "cmd"}
            for k, v in exec_svc.COMMANDS.items()}


# ── 命令执行（同步，含 AI 分析）────────────────────────────────────────────────
@app.post("/api/execute")
async def execute_command(req: ExecuteRequest):
    """同步执行命令，返回输出 + AI 分析."""
    if req.cmd_key not in exec_svc.COMMANDS:
        raise HTTPException(status_code=400, detail=f"未知命令: {req.cmd_key}")

    # 在线程池中运行同步命令（避免阻塞事件循环）
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, exec_svc.run_command_sync, req.cmd_key
    )

    analysis = ""
    if req.analyze_result:
        cmd_label = exec_svc.COMMANDS[req.cmd_key]["label"]
        try:
            analysis = ai_svc.analyze_execution_result(
                result.output, cmd_label, provider=req.provider
            )
        except Exception as e:
            analysis = f"AI 分析失败：{e}"

    return {
        **result.to_dict(),
        "cmd_key": req.cmd_key,
        "cmd_label": exec_svc.COMMANDS[req.cmd_key]["label"],
        "ai_analysis": analysis,
    }


# ── 命令执行（SSE 流式输出）──────────────────────────────────────────────────
@app.get("/api/stream/{cmd_key}")
async def stream_command(cmd_key: str):
    """SSE 流式输出命令执行日志."""
    if cmd_key not in exec_svc.COMMANDS:
        raise HTTPException(status_code=400, detail=f"未知命令: {cmd_key}")

    async def event_generator():
        async for line in exec_svc.run_command_stream(cmd_key):
            # SSE 格式
            data = json.dumps({"line": line}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── AI 对话（含命令识别）─────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """AI 对话：识别意图 → 可选执行命令 → 生成回复."""
    # 1. 解析用户意图
    intent = ai_svc.parse_user_intent(
        req.message, req.history, provider=req.provider
    )

    action   = intent.get("action", "none")
    exec_res = None
    data_ctx = ""

    # 2. 若识别到命令且允许自动执行
    if action != "none" and req.auto_execute and action in exec_svc.COMMANDS:
        loop = asyncio.get_event_loop()
        exec_result = await loop.run_in_executor(
            None, exec_svc.run_command_sync, action
        )
        exec_res = exec_result.to_dict()
        data_ctx = f"命令「{exec_svc.COMMANDS[action]['label']}」执行结果摘要：\n{exec_result.output[-2000:]}"

    # 3. 获取数据上下文（如需要）
    if not data_ctx and intent.get("needs_data"):
        data_ctx = ai_svc.build_data_context()

    # 4. 生成完整回复
    reply = ai_svc.generate_full_response(
        req.message, req.history,
        data_context=data_ctx,
        provider=req.provider,
    )

    return {
        "reply":       reply,
        "action":      action,
        "intent":      intent,
        "exec_result": exec_res,
        "provider":    req.provider,
    }


# ── 数据分析 ──────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """LLM 分析指定数据，返回中文解读."""
    analysis = ai_svc.analyze_data(req.data, req.data_type, provider=req.provider)
    return {"analysis": analysis, "provider": req.provider}


@app.post("/api/analyze/chart")
async def analyze_chart(payload: dict):
    """为图表生成 AI 解读."""
    title    = payload.get("title", "图表")
    data_pts = payload.get("data", {})
    provider = payload.get("provider", "dashscope")
    commentary = ai_svc.generate_chart_commentary(title, data_pts, provider=provider)
    return {"commentary": commentary}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)
