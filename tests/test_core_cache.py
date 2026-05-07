"""测试 quantmind.core.cache."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from quantmind.core import cache as cache_mod


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """每个 test 用独立 cache 目录."""
    monkeypatch.setattr(cache_mod, "_cache_instance", None)
    monkeypatch.setattr(cache_mod, "_cache_dir", tmp_path / "diskcache")


class TestCachedDecorator:
    def test_cache_hit_skips_computation(self) -> None:
        calls = {"n": 0}

        @cache_mod.cached(ttl_hours=1)
        def slow(x: int) -> int:
            calls["n"] += 1
            return x * 2

        assert slow(5) == 10
        assert slow(5) == 10
        assert calls["n"] == 1  # 第二次命中缓存

    def test_different_args_different_cache(self) -> None:
        calls = {"n": 0}

        @cache_mod.cached(ttl_hours=1)
        def f(x: int, y: int) -> int:
            calls["n"] += 1
            return x + y

        f(1, 2)
        f(1, 2)
        f(3, 4)
        assert calls["n"] == 2

    def test_ttl_zero_disables_cache(self) -> None:
        calls = {"n": 0}

        @cache_mod.cached(ttl_hours=0)
        def f() -> int:
            calls["n"] += 1
            return 1

        f()
        f()
        f()
        assert calls["n"] == 3

    def test_invalidate(self) -> None:
        calls = {"n": 0}

        @cache_mod.cached(ttl_hours=1)
        def f(x: int) -> int:
            calls["n"] += 1
            return x * 2

        f(7)
        assert cache_mod.cached_invalidate(f, 7) is True
        f(7)  # 重新算
        assert calls["n"] == 2

    def test_dataframe_via_parquet(self) -> None:
        @cache_mod.cached(ttl_hours=1, serializer="auto")
        def make_df(n: int) -> pd.DataFrame:
            return pd.DataFrame({"a": range(n), "b": [str(i) for i in range(n)]})

        df1 = make_df(5)
        df2 = make_df(5)
        pd.testing.assert_frame_equal(df1, df2)

    def test_kwargs_order_independent(self) -> None:
        calls = {"n": 0}

        @cache_mod.cached(ttl_hours=1)
        def f(*, a: int, b: int) -> int:
            calls["n"] += 1
            return a + b

        f(a=1, b=2)
        f(b=2, a=1)
        assert calls["n"] == 1

    def test_clear_cache(self) -> None:
        @cache_mod.cached(ttl_hours=1)
        def f(x: int) -> int:
            return x

        f(1)
        f(2)
        n = cache_mod.clear_cache()
        assert n >= 2

    def test_short_ttl_expires(self) -> None:
        calls = {"n": 0}

        @cache_mod.cached(ttl_hours=0.5 / 3600)  # 0.5s
        def f() -> int:
            calls["n"] += 1
            return 1

        f()
        time.sleep(0.7)
        f()  # 应过期重算
        assert calls["n"] == 2
