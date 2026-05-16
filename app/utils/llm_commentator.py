"""LLM 图表/文本解读：Ollama → Dashscope → 模板；磁盘缓存 + Streamlit 缓存入口。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_dir() -> Path:
    d = _project_root() / ".streamlit_llm_cache"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _disk_cache_get(cache_key: str, ttl_sec: int = 3600) -> str | None:
    h = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:48]
    p = _cache_dir() / f"{h}.json"
    try:
        if not p.is_file():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        ts = float(raw.get("ts", 0))
        if time.time() - ts > ttl_sec:
            return None
        return str(raw.get("text", ""))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _disk_cache_set(cache_key: str, text: str) -> None:
    h = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:48]
    p = _cache_dir() / f"{h}.json"
    try:
        p.write_text(json.dumps({"ts": time.time(), "text": text}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _ollama_online(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            headers={"User-Agent": "QuantMind-Streamlit"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ollama_generate(prompt: str, max_tokens: int) -> str | None:
    body = json.dumps(
        {
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "QuantMind-Streamlit"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("response", "") or "").strip() or None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, TypeError):
        return None


def _dashscope_generate(prompt: str, max_tokens: int) -> str | None:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        return None
    payload = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"max_tokens": max_tokens, "result_format": "message"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "QuantMind-Streamlit",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out = data.get("output") or {}
        msgs = out.get("choices") or out.get("text")
        if isinstance(msgs, list) and msgs:
            first = msgs[0]
            if isinstance(first, dict):
                msg = first.get("message") or first
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"]).strip()
        if isinstance(out.get("text"), str):
            return out["text"].strip()
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


class LLMCommentator:
    """智能图表/指标解读引擎：Ollama（本地）→ Dashscope → 模板文字。"""

    def __init__(self) -> None:
        self._provider = self._detect_provider()

    @property
    def provider(self) -> str:
        return self._provider

    def _detect_provider(self) -> str:
        if _ollama_online():
            return "ollama"
        if os.environ.get("DASHSCOPE_API_KEY", "").strip():
            return "dashscope"
        return "template"

    def comment(self, prompt: str, cache_key: str, max_tokens: int = 200) -> str:
        """生成解读文字（含磁盘 TTL 缓存）。"""
        disk_key = f"{self._provider}|{cache_key}|{max_tokens}|{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
        cached = _disk_cache_get(disk_key)
        if cached:
            return cached
        text = self._generate_raw(prompt, max_tokens)
        _disk_cache_set(disk_key, text)
        return text

    def _generate_raw(self, prompt: str, max_tokens: int) -> str:
        if self._provider == "ollama":
            r = _ollama_generate(prompt, max_tokens)
            if r:
                return r
            self._provider = "dashscope" if os.environ.get("DASHSCOPE_API_KEY", "").strip() else "template"
        if self._provider == "dashscope":
            r = _dashscope_generate(prompt, max_tokens)
            if r:
                return r
            self._provider = "template"
        return self._template_fallback(prompt)

    def _template_fallback(self, prompt: str) -> str:
        head = (prompt or "").strip().replace("\n", " ")[:200]
        return (
            "（当前未检测到可用的本地 Ollama 服务，也未配置 Dashscope 环境变量 DASHSCOPE_API_KEY，"
            "以下为占位摘要。）\n\n"
            f"要点摘录：{head}"
        )

    def explain_ic(self, ic_mean: float, icir: float, periods: int) -> str:
        level = "较强" if abs(ic_mean) > 0.05 else "中等" if abs(ic_mean) > 0.03 else "偏弱"
        stable_word = "稳定" if abs(icir) > 0.5 else "一般"
        direction = "正向" if ic_mean > 0 else "反向"
        conf = "统计置信度较高" if periods >= 8 else "样本较少，需谨慎解读"
        return (
            f"**IC 解读**：信息系数均值 {ic_mean:.3f}，预测能力{level}（{direction}）。"
            f"ICIR={icir:.2f}，信号稳定性{stable_word}。"
            f"基于 {periods} 个评估期，{conf}。"
        )

    def explain_sharpe(self, sharpe: float, context: str = "多空") -> str:
        try:
            sh = float(sharpe)
        except (TypeError, ValueError):
            return "**夏普比率**：数据缺失，无法解读。"
        qual = "良好" if sh > 1.0 else "偏弱" if sh > 0 else "为负"
        return (
            f"**夏普解读（{context}）**：Sharpe={sh:.3f}，风险调整后收益{qual}。"
            "注意不同回测口径（因子分层 vs 引擎实盘近似）不可直接横向对比。"
        )

    def explain_drawdown(self, max_dd: float) -> str:
        try:
            dd = float(max_dd)
        except (TypeError, ValueError):
            return "**最大回撤**：数据缺失。"
        ad = abs(dd)
        tier = "较低" if ad < 0.10 else "中等" if ad < 0.30 else "较高"
        return f"**回撤解读**：最大回撤约 {dd:.2%}，风险暴露{tier}（绝对值越大回撤越深）。"

    def explain_funnel_layer(self, layer_name: str, before: int, after: int) -> str:
        drop = max(0, before - after)
        pct = (after / before * 100) if before else 0.0
        return (
            f"**{layer_name}**：由 {before} 只压缩至 {after} 只（留存约 {pct:.1f}%），"
            f"剔除约 {drop} 只。"
        )

    def _build_chart_prompt(self, chart_type: str, data_summary: dict[str, Any]) -> str:
        return (
            f"你是量化投资助手，请用 2~4 句中文解读下面图表摘要（客观、不过度承诺）。\n"
            f"图表类型：{chart_type}\n数据摘要（JSON）：{json.dumps(data_summary, ensure_ascii=False)}"
        )

    def analyze_chart(self, chart_type: str, data_summary: dict[str, Any], cache_key: str) -> str:
        templates: dict[str, str] = {}
        if chart_type == "quintile_bar":
            q5 = float(data_summary.get("q5", 0) or 0)
            q1 = float(data_summary.get("q1", 0) or 0)
            spread = float(data_summary.get("spread", q5 - q1))
            templates["quintile_bar"] = (
                f"分层收益柱状图显示 Q5 月均收益为 {q5:.2%}，Q1 为 {q1:.2%}，多空价差约 {spread:.2%}。"
                f"{'分层效果显著' if spread > 0.02 else '分层效果一般'}。"
            )
        elif chart_type == "ic_series":
            n = int(data_summary.get("n", 0) or 0)
            mean = float(data_summary.get("mean", 0) or 0)
            std = float(data_summary.get("std", 0) or 0)
            stab = "波动相对收敛" if std < 0.05 else "波动较大，需注意阶段性失效"
            templates["ic_series"] = (
                f"IC 相关时序显示近 {n} 期均值约 {mean:.3f}（标准差 {std:.3f}），{stab}。"
            )
        elif chart_type == "radar":
            templates["radar"] = (
                "雷达图展示五维 Agent 信号（风险已按「越低越好」方向调整展示）。"
                f"综合可读性：{'偏多' if float(data_summary.get('composite', 0) or 0) > 0.15 else '偏空' if float(data_summary.get('composite', 0) or 0) < -0.15 else '中性'}。"
            )
        elif chart_type == "funnel":
            templates["funnel"] = (
                f"漏斗图显示从全市场逐步收紧至最终候选约 {data_summary.get('final', '?')} 只，"
                f"主要过滤集中在：{data_summary.get('note', '流动性/趋势/模型打分')}。"
            )
        elif chart_type == "nav_curve":
            templates["nav_curve"] = (
                f"净值曲线对比：策略区间收益约 {float(data_summary.get('strat_ret', 0) or 0):.2%}，"
                f"基准近似收益约 {float(data_summary.get('bench_ret', 0) or 0):.2%}（数据来源本地 parquet，仅供参考）。"
            )

        if chart_type in templates:
            return templates[chart_type]

        prompt = self._build_chart_prompt(chart_type, data_summary)
        return self.comment(prompt, cache_key=f"chart_{chart_type}_{cache_key}", max_tokens=220)

    def generate_stock_report(
        self,
        ticker: str,
        agent_signals: dict[str, Any],
        strategy: dict[str, Any],
        validation: dict[str, Any],
    ) -> str:
        """单股完整报告（优先 LLM，≤500 字）。"""
        wr = validation.get("win_rate")
        ar = validation.get("avg_return")
        try:
            wr_f = float(wr) if wr is not None else 0.0
        except (TypeError, ValueError):
            wr_f = 0.0
        try:
            ar_f = float(ar) if ar is not None else 0.0
        except (TypeError, ValueError):
            ar_f = 0.0

        def _sig_block(name: str) -> str:
            block = agent_signals.get(name) or {}
            if not isinstance(block, dict):
                return f"{name}: (无结构)"
            return (
                f"{name}: signal={float(block.get('signal', 0) or 0):.2f}, "
                f"summary={block.get('summary', '')}"
            )

        prompt = f"""请为 {ticker} 生成一份简明投资分析报告（中文，300-500字），
包含：估值分析、动量特征、财务质量、市场情绪、风险提示、综合建议。
请客观分析，不要夸大，指出主要风险。

数据摘要：
- {_sig_block("ValuationAgent")}
- {_sig_block("MomentumAgent")}
- {_sig_block("QualityAgent")}
- {_sig_block("SentimentAgent")}
- {_sig_block("RiskAgent")}
- 综合评级: {strategy.get("rating", "未知")}
- 综合信号: {float(strategy.get("composite_signal", 0) or 0):+.2f}
- 历史胜率: {wr_f:.1%}
- 期望月收益: {ar_f:.2%}
"""
        return self.comment(prompt, cache_key=f"stock_report_{ticker}", max_tokens=512)


if st is not None:

    @st.cache_data(ttl=3600)
    def cached_llm_comment(prompt: str, cache_key: str, max_tokens: int = 200) -> str:
        """Streamlit 会话级缓存（TTL 3600s），内部仍写入磁盘缓存。"""
        return LLMCommentator().comment(prompt, cache_key, max_tokens)

    @st.cache_data(ttl=3600)
    def cached_generate_stock_report(
        ticker: str,
        signals_json: str,
        strategy_json: str,
        validation_json: str,
    ) -> str:
        ag = json.loads(signals_json)
        st_dict = json.loads(strategy_json)
        vd = json.loads(validation_json)
        return LLMCommentator().generate_stock_report(ticker, ag, st_dict, vd)

    @st.cache_data(ttl=3600)
    def cached_analyze_chart(chart_type: str, summary_json: str, cache_key: str) -> str:
        ds = json.loads(summary_json)
        return LLMCommentator().analyze_chart(chart_type, ds, cache_key)

else:

    def cached_llm_comment(prompt: str, cache_key: str, max_tokens: int = 200) -> str:
        return LLMCommentator().comment(prompt, cache_key, max_tokens)

    def cached_generate_stock_report(
        ticker: str,
        signals_json: str,
        strategy_json: str,
        validation_json: str,
    ) -> str:
        ag = json.loads(signals_json)
        st_dict = json.loads(strategy_json)
        vd = json.loads(validation_json)
        return LLMCommentator().generate_stock_report(ticker, ag, st_dict, vd)

    def cached_analyze_chart(chart_type: str, summary_json: str, cache_key: str) -> str:
        ds = json.loads(summary_json)
        return LLMCommentator().analyze_chart(chart_type, ds, cache_key)


def stock_report_cached(
    ticker: str,
    agent_signals: dict[str, Any],
    strategy: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    return cached_generate_stock_report(
        ticker,
        json.dumps(agent_signals, sort_keys=True, default=str),
        json.dumps(strategy, sort_keys=True, default=str),
        json.dumps(validation, sort_keys=True, default=str),
    )


def analyze_chart_cached(chart_type: str, data_summary: dict[str, Any], cache_key: str) -> str:
    return cached_analyze_chart(
        chart_type,
        json.dumps(data_summary, sort_keys=True, default=str),
        cache_key,
    )
