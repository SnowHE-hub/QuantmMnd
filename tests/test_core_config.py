"""测试 quantmind.core.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantmind.core.config import (
    Settings,
    _resolve_env_vars,
    get_settings,
    load_config,
)


class TestEnvVarResolver:
    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "bar")
        assert _resolve_env_vars("hello ${FOO}") == "hello bar"

    def test_full_match_coerces_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLAG", "true")
        assert _resolve_env_vars("${FLAG}") is True

        monkeypatch.setenv("NUM", "42")
        assert _resolve_env_vars("${NUM}") == 42

        monkeypatch.setenv("RATIO", "3.14")
        assert _resolve_env_vars("${RATIO}") == 3.14

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNDEFINED_VAR", raising=False)
        assert _resolve_env_vars("${UNDEFINED_VAR:-fallback}") == "fallback"

    def test_recursive_dict_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A", "alpha")
        result = _resolve_env_vars({"x": "${A}", "y": [1, "v=${A}", {"z": "${A}-z"}]})
        assert result == {"x": "alpha", "y": [1, "v=alpha", {"z": "alpha-z"}]}


class TestLoadConfig:
    def test_load_default(self) -> None:
        s = load_config("default")
        assert isinstance(s, Settings)
        assert s.config_name == "default"
        assert s.data.universe == "csi300"
        assert s.data.pit_strict is True
        assert s.backtest.benchmark == "000300.SH"

    def test_yaml_var_resolution(self) -> None:
        s = load_config("default")
        # default.yaml 里 llm.provider = ${DEFAULT_LLM_PROVIDER}
        # .env 里 DEFAULT_LLM_PROVIDER=ollama
        assert s.llm.provider == "ollama"
        assert s.llm.model == "qwen2.5:7b"

    def test_env_overrides_flat_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "deepseek")
        s = load_config("default")
        assert s.default_llm_provider == "deepseek"

    def test_missing_yaml_returns_defaults(self, tmp_path: Path) -> None:
        """yaml 不存在时应该用 schema 默认值，而不是抛异常."""
        s = load_config("nonexistent_xxx", config_dir=tmp_path)
        assert s.data.universe == "csi300"  # 默认

    def test_path_helpers(self) -> None:
        s = load_config("default")
        assert s.project_root().is_dir()
        assert str(s.data_root_path()).endswith("data")


class TestGetSettings:
    def test_singleton_caching(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_returns_new(self) -> None:
        s1 = get_settings()
        s2 = get_settings(reload=True)
        # 内容应一样
        assert s1.config_name == s2.config_name
