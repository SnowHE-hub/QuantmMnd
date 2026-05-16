"""quantmind.models.inference — DPO 微调模型本地推理.

用于加载 LoRA 权重后在本地进行 rerank 推理，替代 API 调用。

主要接口
========
- load_dpo_model(model_dir)     : 加载 LoRA 微调模型 + 分词器
- rerank_with_dpo(candidates, as_of) : 替代 Ollama 做本地推理
- benchmark(n)                  : 对比本地推理 vs Ollama 延迟
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Any

__all__ = [
    "DPOInferenceEngine",
    "load_dpo_model",
    "rerank_with_dpo",
    "benchmark",
]


# ============================================================================
# 推理引擎
# ============================================================================


class DPOInferenceEngine:
    """加载 DPO 微调 LoRA 模型并进行文本生成.

    参数
    ----
    model_dir:     包含 LoRA 权重的目录（由 dpo_trainer.train() 保存）
    base_model:    基础模型名称（若 model_dir 中无记录则需指定）
    device:        "auto" | "cuda" | "cpu"
    load_in_4bit:  是否用 4-bit 量化加载（推理时减小显存）
    """

    def __init__(
        self,
        model_dir: str | Path,
        base_model: str | None = None,
        device: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._base_model = base_model or self._read_base_model()
        self._model: Any = None
        self._tokenizer: Any = None

    def _read_base_model(self) -> str:
        """从 train_meta.json 中读取基模型名称."""
        meta_path = self.model_dir / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        # 尝试从 adapter_config.json 读取
        adapter_cfg = self.model_dir / "adapter_config.json"
        if adapter_cfg.exists():
            cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            return cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        return "Qwen/Qwen2.5-1.5B-Instruct"

    def load(self) -> "DPOInferenceEngine":
        """加载模型和分词器到内存."""
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "Missing packages. Install: pip install transformers peft bitsandbytes"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._base_model, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            self._base_model,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self._model = PeftModel.from_pretrained(base, str(self.model_dir))
        self._model.eval()
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.05,
    ) -> str:
        """生成文本.

        Args:
            prompt:         用户 prompt
            system:         系统提示（可选）
            max_new_tokens: 最大生成 token 数
            temperature:    采样温度

        Returns:
            生成的文本字符串
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 使用 apply_chat_template（Qwen2.5 支持）
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # 只取新生成的 token
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)


# ============================================================================
# 对外接口
# ============================================================================

# 全局单例（避免重复加载）
_engine_singleton: DPOInferenceEngine | None = None


def load_dpo_model(
    model_dir: str | Path,
    base_model: str | None = None,
    load_in_4bit: bool = True,
    reload: bool = False,
) -> DPOInferenceEngine:
    """加载 DPO LoRA 模型，返回 DPOInferenceEngine（全局单例）.

    Args:
        model_dir:    LoRA 权重目录
        base_model:   基础模型名（默认从 train_meta.json 读取）
        load_in_4bit: 4-bit 量化（推荐，省显存）
        reload:       True = 强制重新加载

    Returns:
        已加载的 DPOInferenceEngine
    """
    global _engine_singleton
    if reload or _engine_singleton is None:
        _engine_singleton = DPOInferenceEngine(
            model_dir=model_dir,
            base_model=base_model,
            load_in_4bit=load_in_4bit,
        ).load()
    return _engine_singleton


def rerank_with_dpo(
    candidates,
    as_of: str = "",
    top_n: int = 30,
    model_dir: str | Path = "models/dpo_qwen",
) -> list[Any]:
    """用本地 DPO 模型替代 API 调用进行 rerank.

    Args:
        candidates:  list[RerankCandidate]
        as_of:       截面日期字符串
        top_n:       目标精排数量
        model_dir:   LoRA 权重目录

    Returns:
        list[RerankResult]（接口与 LLMListwiseReranker.rerank() 相同）
    """
    from quantmind.models.llm_reranker import LLMListwiseReranker, RerankResult

    engine = load_dpo_model(model_dir)

    # 复用 LLMListwiseReranker 的 prompt 构建逻辑（但用本地推理替代 API）
    tmp_reranker = LLMListwiseReranker.__new__(LLMListwiseReranker)
    tmp_reranker.provider = "local_dpo"
    tmp_reranker.model = str(model_dir)
    tmp_reranker.batch_size = 50
    tmp_reranker._router = None

    from quantmind.models.llm_reranker import _SYSTEM_PROMPT
    prompt = tmp_reranker._build_user_prompt(candidates, top_n, as_of)

    raw_text = engine.generate(prompt=prompt, system=_SYSTEM_PROMPT, max_new_tokens=1024)

    ordered_tickers, is_fallback = tmp_reranker._parse_rankings(raw_text, candidates, top_n)
    reason_map = tmp_reranker._extract_field_map(raw_text, "reason")
    portfolio_thesis = tmp_reranker._extract_portfolio_thesis(raw_text)
    risk_warnings    = tmp_reranker._extract_risk_warnings(raw_text)

    ticker_map = {c.ticker: c for c in candidates}
    results = []
    for rank, ticker in enumerate(ordered_tickers[:top_n], start=1):
        cand = ticker_map.get(ticker)
        results.append(RerankResult(
            rank=rank,
            ticker=ticker,
            lgbm_rank=cand.lgbm_rank if cand else 999,
            reason=reason_map.get(ticker, ""),
            portfolio_thesis=portfolio_thesis,
            risk_warnings=risk_warnings,
            is_fallback=is_fallback,
        ))
    return results


def benchmark(
    n: int = 10,
    model_dir: str | Path = "models/dpo_qwen",
    ollama_model: str = "qwen2.5:7b",
    prompt_text: str | None = None,
) -> dict[str, Any]:
    """对比本地 DPO 推理 vs Ollama API 的延迟.

    Args:
        n:             测试样本数
        model_dir:     LoRA 权重目录
        ollama_model:  Ollama 模型名
        prompt_text:   测试用 prompt（默认使用内置样本）

    Returns:
        dict with keys: local_avg_s, ollama_avg_s, local_vs_ollama_ratio
    """
    if prompt_text is None:
        prompt_text = (
            "对以下候选股票进行排名，输出 JSON：\n"
            "| LGBM | 代码 | ROE% |\n| 1 | 600519.SH | 26.8% |\n| 2 | 000858.SZ | 18.2% |"
        )

    results: dict[str, Any] = {}

    # 本地 DPO 推理
    try:
        engine = load_dpo_model(model_dir)
        local_times = []
        for _ in range(n):
            t0 = time.monotonic()
            engine.generate(prompt_text, max_new_tokens=256)
            local_times.append(time.monotonic() - t0)
        results["local_avg_s"] = round(sum(local_times) / len(local_times), 3)
        results["local_success"] = True
    except Exception as e:  # noqa: BLE001
        results["local_avg_s"] = None
        results["local_success"] = False
        results["local_error"] = str(e)

    # Ollama 推理
    try:
        from quantmind.core.llm_router import LLMRouter
        router = LLMRouter()
        ollama_times = []
        for _ in range(n):
            t0 = time.monotonic()
            router.chat(
                messages=[{"role": "user", "content": prompt_text}],
                provider="ollama",
                model=ollama_model,
                max_tokens=256,
                fallback=None,
            )
            ollama_times.append(time.monotonic() - t0)
        results["ollama_avg_s"] = round(sum(ollama_times) / len(ollama_times), 3)
        results["ollama_success"] = True
    except Exception as e:  # noqa: BLE001
        results["ollama_avg_s"] = None
        results["ollama_success"] = False
        results["ollama_error"] = str(e)

    if results.get("local_avg_s") and results.get("ollama_avg_s"):
        results["local_vs_ollama_ratio"] = round(
            results["local_avg_s"] / results["ollama_avg_s"], 3
        )

    return results
