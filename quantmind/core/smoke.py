"""Smoke test: 验证 Phase 0 项目骨架是否正常工作.

检查项：
    1. Python 版本 ≥ 3.11
    2. 关键依赖可 import（pandas / numpy / pydantic / yaml / dotenv / loguru）
    3. .env 文件存在且关键变量已设置
    4. configs/default.yaml 可加载
    5. （可选）Ollama 可访问
    6. GPU 可用性（仅检测，不报错）

用法::

    python -m quantmind.core.smoke
    # or
    make smoke
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# 项目根目录（以 pyproject.toml 为锚）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# 简易彩色打印（不依赖 rich，避免 smoke 时也要装一堆东西）
# ----------------------------------------------------------------------------
class _C:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    INFO = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{_C.OK}[ OK ]{_C.END} {msg}")


def _warn(msg: str) -> None:
    print(f"{_C.WARN}[WARN]{_C.END} {msg}")


def _fail(msg: str) -> None:
    print(f"{_C.FAIL}[FAIL]{_C.END} {msg}")


def _info(msg: str) -> None:
    print(f"{_C.INFO}[INFO]{_C.END} {msg}")


def _section(title: str) -> None:
    print(f"\n{_C.BOLD}── {title} ──{_C.END}")


# ----------------------------------------------------------------------------
# 各项检查
# ----------------------------------------------------------------------------
def check_python() -> bool:
    _section("Python")
    v = sys.version_info
    _info(f"Python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if v < (3, 11):
        _fail("需要 Python ≥ 3.11")
        return False
    if v >= (3, 13):
        _warn("Python 3.13 部分依赖（vectorbt / chromadb 等）兼容性差，强烈建议 3.11")
    _ok("Python 版本满足要求")
    return True


def check_imports() -> bool:
    _section("Core Imports")
    required = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("polars", "polars"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
        ("loguru", "loguru"),
        ("tenacity", "tenacity"),
        ("joblib", "joblib"),
        ("diskcache", "diskcache"),
        ("rich", "rich"),
    ]
    all_ok = True
    for mod, pkg in required:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            _ok(f"{pkg:20s} v{ver}")
        except ImportError as e:
            _fail(f"{pkg} 未安装：{e}")
            all_ok = False
    return all_ok


def check_dotenv() -> bool:
    _section(".env 与必备 API Key")
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        _fail(f".env 不存在：{env_file}")
        _info("解决：cp .env.example .env 并填入 key")
        return False
    _ok(f".env 存在：{env_file}")

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:
        _warn("python-dotenv 未安装，跳过 .env 解析")
        return True

    keys_to_check = [
        ("DEEPSEEK_API_KEY", "强烈推荐", "DeepSeek 主力 LLM"),
        ("DASHSCOPE_API_KEY", "推荐", "通义千问 fallback"),
        ("TUSHARE_TOKEN", "强烈推荐", "A股财报披露日 PIT 严格性"),
        ("OPENAI_API_KEY", "可选", "DPO 数据生成 / Judge"),
        ("ANTHROPIC_API_KEY", "可选", "Critical reasoning"),
    ]
    has_any_llm = False
    for k, level, purpose in keys_to_check:
        v = os.getenv(k, "").strip()
        if v:
            masked = v[:8] + "***" + v[-4:] if len(v) > 12 else "***"
            _ok(f"{k:25s} = {masked} ({purpose})")
            if "API_KEY" in k:
                has_any_llm = True
        else:
            (_warn if level == "可选" else _info)(f"{k:25s} 未设置（{level}：{purpose}）")

    if not has_any_llm:
        _warn("没有任何 LLM API Key — 必须依赖 Ollama 才能跑 Agent 系统")
    return True


def check_config() -> bool:
    _section("配置文件")
    cfg = PROJECT_ROOT / "configs" / "default.yaml"
    if not cfg.exists():
        _fail(f"缺少 {cfg}")
        return False
    try:
        import yaml

        with cfg.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _ok(f"configs/default.yaml 可加载，顶层 keys: {sorted(data.keys())}")

        llm_cfg = PROJECT_ROOT / "configs" / "llm_providers.yaml"
        if llm_cfg.exists():
            with llm_cfg.open(encoding="utf-8") as f:
                llm_data = yaml.safe_load(f)
            providers = list(llm_data.get("providers", {}).keys())
            _ok(f"configs/llm_providers.yaml 可加载，providers: {providers}")
    except Exception as e:
        _fail(f"配置文件解析失败：{e}")
        return False
    return True


def check_ollama() -> bool:
    _section("Ollama（可选，本地 LLM）")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import httpx

        r = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            models = [m.get("name", "?") for m in data.get("models", [])]
            _ok(f"Ollama 可访问 ({base_url})，已安装模型: {models}")
            default = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:7b")
            if default not in models:
                _warn(
                    f"OLLAMA_DEFAULT_MODEL={default} 未在已安装列表，可执行：ollama pull {default}"
                )
            return True
        _warn(f"Ollama HTTP {r.status_code}")
        return False
    except ImportError:
        _warn("httpx 未安装，跳过 ollama 检测")
        return False
    except Exception as e:
        _warn(f"Ollama 不可访问（{e}）— 不影响 API LLM 使用，但本地开发推荐启动 ollama serve")
        return False


def check_gpu() -> bool:
    _section("GPU（可选）")
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                total_mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                _ok(f"CUDA[{i}] {name} ({total_mem_gb:.1f} GB)")
            return True
        _warn("torch.cuda.is_available() = False — DPO 训练将不可用")
        return False
    except ImportError:
        _info("torch 未安装（属于 [dl] extras，可后续 install）")
        return False
    except Exception as e:
        _warn(f"GPU 检测异常：{e}")
        return False


def check_core_modules() -> bool:
    """端到端验证 core 模块都能加载并互相工作."""
    _section("Core Modules")
    try:
        from quantmind.core import (
            AgentState,
            InvestmentQuery,
            LLMRouter,
            cached,
            get_logger,
            get_settings,
        )
    except Exception as e:  # noqa: BLE001
        _fail(f"core 模块导入失败：{e}")
        return False
    _ok("quantmind.core 公共 API 全部可导入")

    try:
        s = get_settings()
        _ok(f"get_settings() OK: llm.provider={s.llm.provider}, data.universe={s.data.universe}")
    except Exception as e:  # noqa: BLE001
        _fail(f"Settings 加载失败：{e}")
        return False

    try:
        log = get_logger("smoke")
        log.debug("smoke debug message")
        _ok("logger 可用")
    except Exception as e:  # noqa: BLE001
        _fail(f"logger 失败：{e}")
        return False

    try:
        @cached(ttl_hours=1)
        def _add(a: int, b: int) -> int:
            return a + b

        assert _add(1, 2) == 3
        assert _add(1, 2) == 3  # cached
        _ok("@cached 装饰器可用")
    except Exception as e:  # noqa: BLE001
        _fail(f"cache 失败：{e}")
        return False

    try:
        from datetime import date

        q = InvestmentQuery(raw_query="test", as_of=date(2024, 1, 1))
        st = AgentState(query=q)
        assert st.iteration_count == 0
        _ok("AgentState / InvestmentQuery schema 可构造")
    except Exception as e:  # noqa: BLE001
        _fail(f"state 失败：{e}")
        return False

    try:
        router = LLMRouter()
        avail = router.available_providers()
        _ok(f"LLMRouter 初始化 OK；可用 providers: {avail}")
    except Exception as e:  # noqa: BLE001
        _fail(f"LLMRouter 失败：{e}")
        return False

    return True


def check_directories() -> bool:
    _section("目录结构")
    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/features",
        "data/snapshots",
        "data/kb",
        "docs",
        "scripts",
        "tests",
        "notebooks",
        "quantmind/core",
        "quantmind/data",
        "quantmind/features",
        "quantmind/models",
        "quantmind/agents",
        "quantmind/backtest",
        "quantmind/kb",
        "quantmind/risk",
        "quantmind/ui",
    ]
    all_ok = True
    for d in required_dirs:
        p = PROJECT_ROOT / d
        if p.is_dir():
            _ok(f"{d}/")
        else:
            _fail(f"缺少 {d}/")
            all_ok = False
    return all_ok


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    """Run all smoke checks. Returns 0 on success, 1 on critical failure."""
    print(f"{_C.BOLD}QuantMind Smoke Test{_C.END}")
    print(f"项目根：{PROJECT_ROOT}")

    results = {
        "python": check_python(),
        "imports": check_imports(),
        "dotenv": check_dotenv(),
        "config": check_config(),
        "directories": check_directories(),
        "core_modules": check_core_modules(),
    }
    # 这两个非关键，失败不影响整体 exit code
    check_ollama()
    check_gpu()

    _section("Summary")
    critical_pass = all(results.values())
    for name, ok in results.items():
        (_ok if ok else _fail)(f"{name}: {'PASS' if ok else 'FAIL'}")

    if critical_pass:
        print(f"\n{_C.OK}{_C.BOLD}✔ Phase 0 骨架就绪，可以进入 Phase 1（数据层）{_C.END}\n")
        return 0
    print(f"\n{_C.FAIL}{_C.BOLD}✘ 有关键检查失败，请修复后重跑{_C.END}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
