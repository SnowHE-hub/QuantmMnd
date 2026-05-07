"""quantmind.core.config — 全局配置加载.

设计：
    - 用 ``pydantic-settings`` 加载 ``.env``（仅 KV 字符串）
    - 用 ``PyYAML`` 加载 ``configs/<name>.yaml`` 并支持 ``${ENV_VAR}`` / ``${ENV_VAR:-default}`` 插值
    - 把 yaml 树 merge 进 ``Settings`` 嵌套字段
    - 单例：``get_settings()`` 第一次调用加载，之后缓存
    - 重新加载：``get_settings(reload=True)`` 或 ``load_config("xxx")``

用法::

    from quantmind.core.config import get_settings
    s = get_settings()
    print(s.deepseek_api_key)         # 来自 .env
    print(s.data.universe)            # 来自 configs/default.yaml
    print(s.llm.provider)             # ${DEFAULT_LLM_PROVIDER} 已被解析

切换配置文件::

    from quantmind.core.config import load_config
    s = load_config("universe_csi500")  # 加载 configs/universe_csi500.yaml
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件位于 quantmind/core/config.py，上溯两级即根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_NAME = "default"


# ============================================================================
# Sub-Configs（与 configs/default.yaml 的层次对应）
# ============================================================================


class DataConfig(BaseModel):
    universe: str = "csi300"
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    pit_strict: bool = True
    primary_provider: str = "akshare"
    fallback_providers: list[str] = Field(default_factory=lambda: ["tushare"])
    timezone: str = "Asia/Shanghai"
    trading_calendar: str = "SSE"


class NeutralizeConfig(BaseModel):
    industry: bool = True
    market_cap: bool = True


class FeatureConfig(BaseModel):
    lookback_days: int = 252
    rebalance_freq: str = "M"
    standardize: bool = True
    neutralize: NeutralizeConfig = Field(default_factory=NeutralizeConfig)
    winsorize: float = 3.0
    fillna_strategy: str = "industry_median"


class LGBMConfig(BaseModel):
    objective: str = "lambdarank"
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 500
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    early_stopping_rounds: int = 50


class ModelConfig(BaseModel):
    factor_model: str = "lgbm"
    use_llm_rerank: bool = True
    llm_rerank_top_n: int = 50
    final_top_k: int = 10
    holding_days: int = 20
    lgbm: LGBMConfig = Field(default_factory=LGBMConfig)


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000
    commission_bps: float = 3
    stamp_tax_bps: float = 10
    slippage_bps: float = 5
    benchmark: str = "000300.SH"
    rebalance_at: str = "open"
    enable_t_plus_1: bool = True
    max_position_pct: float = 0.05
    max_volume_pct: float = 0.05


class WalkForwardConfig(BaseModel):
    train_window_months: int = 36
    val_window_months: int = 6
    test_window_months: int = 3
    step_months: int = 3


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_s: int = 120
    retry_attempts: int = 3
    retry_min_wait: int = 1
    retry_max_wait: int = 30


class AgentSubConfig(BaseModel):
    max_iterations: int = 3
    enable_streaming: bool = True
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./.cache/agent_checkpoints"
    parallel_data_agents: int = 4
    parallel_analysis: int = 3


class KBRetrieverConfig(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 10
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


class KBConfig(BaseModel):
    vector_store: str = "chroma"
    vector_store_dir: str = "./data/kb/chroma"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever: KBRetrieverConfig = Field(default_factory=KBRetrieverConfig)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "1 day"
    retention: str = "7 days"
    format: str = ""  # 为空表示用 logger 模块默认


class CacheConfig(BaseModel):
    default_ttl_hours: int = 24
    max_size_gb: int = 10


# ============================================================================
# Top-level Settings
# ============================================================================


class Settings(BaseSettings):
    """全局配置：扁平 env vars + 嵌套 yaml configs."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- 来自 .env 的扁平 env vars ----------------
    deepseek_api_key: str = ""
    dashscope_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "qwen2.5:7b"

    tushare_token: str = ""

    data_root: str = "./data"
    cache_dir: str = "./.cache"
    log_dir: str = "./logs"
    log_level: str = "INFO"

    default_llm_provider: str = "ollama"
    default_llm_model: str = "qwen2.5:7b"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"

    max_tokens_per_call: int = 4096
    max_daily_api_cost_cny: float = 20.0

    # ---------------- 来自 configs/<name>.yaml 的嵌套 ----------------
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentSubConfig = Field(default_factory=AgentSubConfig)
    kb: KBConfig = Field(default_factory=KBConfig)
    logging_cfg: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # 元信息
    config_name: str = DEFAULT_CONFIG_NAME

    # ----------- 便捷方法 -----------
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def data_root_path(self) -> Path:
        p = Path(self.data_root)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def cache_dir_path(self) -> Path:
        p = Path(self.cache_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def log_dir_path(self) -> Path:
        p = Path(self.log_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ============================================================================
# YAML 加载与 ${VAR} 插值
# ============================================================================

# 支持 ${VAR} 和 ${VAR:-default} 两种形式
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """递归把字符串里的 ``${VAR}`` 替换为 env var 当前值."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default)

        # 全字符串就是单个变量时尝试还原类型（如 "true"/"3.14"）
        full_match = _ENV_VAR_PATTERN.fullmatch(value)
        if full_match:
            replaced = _sub(full_match)
            return _coerce_str(replaced)
        return _ENV_VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _coerce_str(s: str) -> Any:
    """把字符串还原为最自然的类型（仅在 ``${VAR}`` 整体替换时使用）."""
    low = s.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"null", "none", ""}:
        return s if s == "" else None
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """加载 yaml 并解析所有 ``${VAR}``."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_env_vars(data)


def _normalize_yaml_keys(data: dict[str, Any]) -> dict[str, Any]:
    """把 yaml 里的 key 映射到 Settings 字段名.

    特殊处理：``logging`` 是 Python 关键模块名，Settings 里改名为 ``logging_cfg``。
    """
    if "logging" in data and "logging_cfg" not in data:
        data["logging_cfg"] = data.pop("logging")
    return data


# ============================================================================
# 公共加载入口
# ============================================================================


def load_config(
    config_name: str = DEFAULT_CONFIG_NAME,
    config_dir: Path | None = None,
) -> Settings:
    """加载指定配置文件并返回 Settings 实例（不缓存，每次重新读盘）.

    流程：
        1. 先把 ``.env`` 加载进 ``os.environ``（不覆盖已有 env vars）
        2. 读 yaml 文件，解析 ``${VAR}`` 插值
        3. 用 ``Settings(**yaml_data)`` 构造（pydantic-settings 会再读一次 .env 覆盖）

    Args:
        config_name: 配置文件名（不含 .yaml），如 ``"default"``、``"universe_csi500"``
        config_dir: 配置目录，默认 ``configs/``

    Returns:
        Settings 实例

    Raises:
        FileNotFoundError: 配置文件不存在
    """
    # 步骤 1：把 .env 注入 os.environ，让 YAML 里的 ${VAR} 能解析到
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass  # 没装 python-dotenv 也不致命

    cfg_dir = config_dir or DEFAULT_CONFIG_DIR
    yaml_path = cfg_dir / f"{config_name}.yaml"

    yaml_data = _load_yaml_file(yaml_path)
    yaml_data = _normalize_yaml_keys(yaml_data)
    yaml_data["config_name"] = config_name

    # 步骤 3：构造 Settings；pydantic-settings 自己会读 .env 覆盖扁平字段
    settings = Settings(**yaml_data)

    return settings


@lru_cache(maxsize=4)
def _cached_settings(config_name: str) -> Settings:
    return load_config(config_name)


def get_settings(reload: bool = False) -> Settings:
    """获取全局 Settings（默认 default.yaml，带缓存）.

    Args:
        reload: True 时清缓存重新加载
    """
    if reload:
        _cached_settings.cache_clear()
    return _cached_settings(DEFAULT_CONFIG_NAME)


__all__ = [
    "AgentSubConfig",
    "BacktestConfig",
    "CacheConfig",
    "DataConfig",
    "FeatureConfig",
    "KBConfig",
    "LGBMConfig",
    "LLMConfig",
    "LoggingConfig",
    "ModelConfig",
    "NeutralizeConfig",
    "PROJECT_ROOT",
    "Settings",
    "WalkForwardConfig",
    "get_settings",
    "load_config",
]


if __name__ == "__main__":
    s = get_settings()
    print(f"== config_name = {s.config_name}")
    print(f"== project_root = {s.project_root()}")
    print(f"== data.universe = {s.data.universe}")
    print(f"== data.pit_strict = {s.data.pit_strict}")
    print(f"== llm.provider = {s.llm.provider}  (resolved from ${{DEFAULT_LLM_PROVIDER}})")
    print(f"== llm.model = {s.llm.model}")
    print(f"== ollama_base_url = {s.ollama_base_url}")
    print(
        f"== deepseek_api_key set? "
        f"{'yes' if s.deepseek_api_key else 'no'} (length={len(s.deepseek_api_key)})"
    )
    print(f"== max_daily_api_cost_cny = {s.max_daily_api_cost_cny}")
