"""tests/test_board_router.py — BoardModelRouter 单元测试.

共 18 个测试，覆盖：
  - 板块识别规则（688→STAR, 300→GEM, 000/600/8→MAIN）
  - 专用模型不存在时降级 fallback
  - predict() 输出 index 与输入完全一致
  - 混合板块 DataFrame 分组路由后 index 顺序保持
  - 空 DataFrame 处理
  - router.status() / which_model() 功能
  - 自定义模型路径 / 覆盖配置
  - 多次 predict() 懒加载缓存（不重复加载）
  - direction=-1 质量门禁：_get_model 加载时降级 fallback（测试 15-16）
  - direction=-1 二次防御：predict() 内部兜底切换 fallback（测试 17）
  - direction gate 全板块降级集成场景（测试 18）
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantmind.models.board_router import BoardModelRouter, get_board  # noqa: E402


# ── 辅助工厂 ─────────────────────────────────────────────────────────────────

def _make_features(
    tickers: list[str],
    n_feats: int = 10,
) -> pd.DataFrame:
    """构造 (ticker × n_feats) 特征 DataFrame，values 随机。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((len(tickers), n_feats)),
        index=tickers,
        columns=[f"f{i}" for i in range(n_feats)],
    )


def _mock_model(scores: np.ndarray | None = None) -> MagicMock:
    """返回一个 predict() 固定分数的 mock FactorModel。

    显式设置 direction=1，确保通过 BoardModelRouter 的方向质量门禁。
    """
    m = MagicMock()
    m._feature_names = [f"f{i}" for i in range(10)]
    m.direction = 1                # ← 必须显式设置，MagicMock 默认属性 != 1
    if scores is not None:
        m.predict.return_value = scores
    else:
        m.predict.side_effect = lambda X: np.arange(len(X), dtype=float)
    return m


def _make_router_with_mock(
    boards: dict[str, MagicMock],
    fallback: MagicMock | None = None,
) -> BoardModelRouter:
    """构造一个所有模型都是 mock 的 router（跳过文件加载）。"""
    router = BoardModelRouter()
    router._models = dict(boards)
    if fallback is not None:
        router._fallback = fallback
    return router


# ── 测试 1：get_board — 688.SH → STAR ────────────────────────────────────────

def test_get_board_star():
    assert get_board("688001.SH") == "STAR"
    assert get_board("688981.SH") == "STAR"
    assert get_board("688050.SH") == "STAR"


# ── 测试 2：get_board — 300.SZ → GEM ─────────────────────────────────────────

def test_get_board_gem():
    assert get_board("300750.SZ") == "GEM"
    assert get_board("300001.SZ") == "GEM"
    assert get_board("300999.SZ") == "GEM"


# ── 测试 3：get_board — 其他 → MAIN ──────────────────────────────────────────

def test_get_board_main():
    assert get_board("600519.SH") == "MAIN"   # 贵州茅台
    assert get_board("000001.SZ") == "MAIN"   # 平安银行
    assert get_board("000858.SZ") == "MAIN"   # 五粮液
    assert get_board("920576.BJ") == "MAIN"   # 北交所
    assert get_board("601398.SH") == "MAIN"   # 工商银行
    assert get_board("600036.SH") == "MAIN"   # 招商银行


# ── 测试 4：predict() 输出 index 与输入完全一致 ───────────────────────────────

def test_predict_index_matches_input():
    tickers = ["600519.SH", "688001.SH", "300750.SZ", "000858.SZ"]
    feat_df = _make_features(tickers)
    router  = _make_router_with_mock({
        "MAIN": _mock_model(),
        "GEM":  _mock_model(),
        "STAR": _mock_model(),
    })
    scores = router.predict(feat_df)
    assert list(scores.index) == tickers, (
        f"输出 index {list(scores.index)} 应等于输入 {tickers}"
    )


# ── 测试 5：predict() 输出 index 顺序与输入一致（乱序也保持） ────────────────

def test_predict_preserves_input_order():
    tickers = ["300750.SZ", "000001.SZ", "688001.SH", "600519.SH", "300012.SZ"]
    feat_df = _make_features(tickers)
    router  = _make_router_with_mock({
        "MAIN": _mock_model(),
        "GEM":  _mock_model(),
        "STAR": _mock_model(),
    })
    scores = router.predict(feat_df)
    assert list(scores.index) == tickers


# ── 测试 6：分板块路由，每只股票用对应模型打分 ───────────────────────────────

def test_predict_routes_to_correct_board():
    """各板块模型返回固定值，验证路由逻辑正确。"""
    main_model = MagicMock()
    main_model.direction = 1                                  # 通过方向质量门禁
    main_model._feature_names = [f"f{i}" for i in range(10)]
    main_model.predict.return_value = np.array([1.0, 2.0])   # 2 只主板股

    gem_model = MagicMock()
    gem_model.direction = 1
    gem_model._feature_names = [f"f{i}" for i in range(10)]
    gem_model.predict.return_value = np.array([9.0])          # 1 只创业板股

    star_model = MagicMock()
    star_model.direction = 1
    star_model._feature_names = [f"f{i}" for i in range(10)]
    star_model.predict.return_value = np.array([5.0])         # 1 只科创板股

    tickers = ["600519.SH", "000001.SZ", "300750.SZ", "688001.SH"]
    feat_df = _make_features(tickers)
    router  = _make_router_with_mock({
        "MAIN": main_model,
        "GEM":  gem_model,
        "STAR": star_model,
    })
    scores = router.predict(feat_df)

    # 创业板/科创板打分独立
    assert scores["300750.SZ"] == pytest.approx(9.0)
    assert scores["688001.SH"] == pytest.approx(5.0)
    # 主板由 main_model 打分（1.0 和 2.0，顺序取决于输入顺序）
    main_scores = {scores["600519.SH"], scores["000001.SZ"]}
    assert main_scores == {1.0, 2.0}


# ── 测试 7：STAR 模型不存在时降级 fallback ───────────────────────────────────

def test_star_model_missing_falls_back():
    """STAR 模型文件不存在时，should 使用 fallback 打分。"""
    fallback_model = MagicMock()
    fallback_model._feature_names = [f"f{i}" for i in range(10)]
    fallback_model.predict.return_value = np.array([99.0])

    router = BoardModelRouter()
    # 强制 STAR 路径不存在
    router._path_map["STAR"] = Path("/nonexistent/star_model.pkl")
    router._fallback = fallback_model
    router._models = {
        "MAIN": _mock_model(),
        "GEM":  _mock_model(),
        # STAR 故意不提前加载，让路由器走 fallback
    }

    tickers = ["688001.SH"]
    feat_df = _make_features(tickers)
    scores  = router.predict(feat_df)

    # STAR 应由 fallback 打分
    assert not scores.isna().any()
    assert scores["688001.SH"] == pytest.approx(99.0)


# ── 测试 8：空 DataFrame 返回空 Series ───────────────────────────────────────

def test_predict_empty_dataframe():
    router = BoardModelRouter()
    result = router.predict(pd.DataFrame())
    assert isinstance(result, pd.Series)
    assert len(result) == 0


# ── 测试 9：GEM 模型不存在，全降级 fallback ──────────────────────────────────

def test_gem_model_missing_falls_back():
    """GEM 路径不存在时所有 GEM 股票用 fallback 打分，不抛异常。"""
    fallback_model = MagicMock()
    fallback_model._feature_names = [f"f{i}" for i in range(10)]
    fallback_model.predict.return_value = np.array([7.0, 8.0])

    router = BoardModelRouter()
    router._path_map["GEM"] = Path("/nonexistent/gem.pkl")
    router._fallback = fallback_model
    router._models   = {
        "MAIN": _mock_model(),
        "STAR": _mock_model(),
    }

    tickers = ["300001.SZ", "300750.SZ"]
    feat_df = _make_features(tickers)
    scores  = router.predict(feat_df)

    assert not scores.isna().any()


# ── 测试 10：status() 返回各板块状态 ─────────────────────────────────────────

def test_status_returns_dict():
    router = BoardModelRouter()
    s = router.status()
    assert isinstance(s, dict)
    assert set(s.keys()) == {"MAIN", "GEM", "STAR"}


# ── 测试 11：which_model() 返回正确路径描述 ───────────────────────────────────

def test_which_model_returns_string():
    router = BoardModelRouter()
    result_main = router.which_model("600519.SH")
    result_gem  = router.which_model("300750.SZ")
    result_star = router.which_model("688001.SH")
    assert isinstance(result_main, str)
    assert isinstance(result_gem,  str)
    assert isinstance(result_star, str)


# ── 测试 12：自定义模型路径 ───────────────────────────────────────────────────

def test_custom_model_paths():
    """传入自定义 model_paths 应覆盖默认路径。"""
    custom_paths = {
        "MAIN": "models/custom_main.pkl",
        "GEM":  "models/custom_gem.pkl",
        "STAR": "models/custom_star.pkl",
    }
    router = BoardModelRouter(model_paths=custom_paths)
    for board, relpath in custom_paths.items():
        expected = Path(ROOT) / relpath
        assert router._path_map[board] == expected, (
            f"{board} 路径期望 {expected}，得 {router._path_map[board]}"
        )


# ── 测试 13：重复 predict() 不重复加载模型（懒加载缓存） ─────────────────────

def test_lazy_loading_cache():
    """同一 board 的模型只应被加载一次（缓存命中）。"""
    call_count = {"n": 0}
    mock_m = _mock_model()

    router = BoardModelRouter()
    router._path_map["GEM"] = Path("/fake/gem.pkl")
    router._fallback = _mock_model()

    # 注入一个计数版的 _get_model
    original_get = router._get_model.__func__

    def counting_get(self, board):
        if board == "GEM":
            call_count["n"] += 1
            return mock_m
        return original_get(self, board)

    import types as _types
    router._get_model = _types.MethodType(counting_get, router)

    tickers  = ["300001.SZ", "300750.SZ", "300012.SZ"]
    feat_df  = _make_features(tickers)
    router.predict(feat_df)
    router.predict(feat_df)   # 第二次调用

    # GEM board 被调用 2 次（每次 predict 各一次），但底层文件只加载 0 次
    assert call_count["n"] == 2   # _get_model 被调用，但缓存判断在其内部


# ── 测试 14：所有板块混合大截面，无 NaN ──────────────────────────────────────

def test_full_mixed_cross_section_no_nan():
    """混合板块截面（含主板/GEM/STAR）打分后不应出现 NaN。"""
    tickers = [
        "600519.SH", "601398.SH", "000001.SZ", "000858.SZ",  # MAIN
        "300750.SZ", "300012.SZ", "300001.SZ",                # GEM
        "688001.SH", "688599.SH",                             # STAR
    ]
    feat_df = _make_features(tickers, n_feats=5)

    # 给每个 board 注入不同维度的 mock 模型
    def make_mock(n_feat=5):
        m = MagicMock()
        m.direction = 1           # 通过方向质量门禁
        m._feature_names = [f"f{i}" for i in range(n_feat)]
        m.predict.side_effect = lambda X: np.ones(len(X)) * len(X)
        return m

    router = _make_router_with_mock({
        "MAIN": make_mock(),
        "GEM":  make_mock(),
        "STAR": make_mock(),
    })
    scores = router.predict(feat_df)

    assert len(scores) == len(tickers)
    assert not scores.isna().any(), f"出现 NaN: {scores[scores.isna()]}"
    assert list(scores.index) == tickers


# ── 测试 15：_get_model() direction=-1 → 质量门禁降级 fallback ────────────────

def test_get_model_direction_minus_one_degrades_to_fallback():
    """_get_model 加载到 direction=-1 的模型时，应返回 fallback 而非该模型。

    FactorModel 在 _get_model 内是局部导入的，所以要 patch
    'quantmind.models.factor_model.FactorModel'。
    """
    bad_model = MagicMock()
    bad_model.direction = -1                        # ← 触发门禁
    bad_model._feature_names = [f"f{i}" for i in range(10)]
    bad_model.predict.return_value = np.array([77.0])

    fallback_model = _mock_model(np.array([55.0]))  # direction=1（已在 _mock_model 设置）

    router = BoardModelRouter()
    router._fallback = fallback_model

    # 模拟文件存在 + FactorModel.load 返回 direction=-1 的模型
    fake_path = Path("/fake/gem_bad.pkl")
    router._path_map["GEM"] = fake_path

    # FactorModel 在 board_router 中通过局部 import 引用，
    # 需要 patch 其原始所在模块 quantmind.models.factor_model
    with patch("quantmind.models.factor_model.FactorModel") as MockFM:
        MockFM.load.return_value = bad_model
        with patch.object(Path, "is_file", return_value=True):
            result = router._get_model("GEM")

    # direction=-1 → 门禁触发 → 应该返回 fallback，不是 bad_model
    assert result is fallback_model, (
        "direction=-1 模型应被门禁拦截，_get_model 应返回 fallback"
    )
    bad_model.predict.assert_not_called()           # bad_model.predict 不应被调用


# ── 测试 16：_get_model() direction=+1 → 正常使用板块专用模型 ────────────────

def test_get_model_direction_plus_one_uses_board_model():
    """_get_model 加载到 direction=+1 的模型时，应正常缓存并返回该模型。"""
    good_model = MagicMock()
    good_model.direction = 1                        # ← 通过门禁
    good_model._feature_names = [f"f{i}" for i in range(10)]
    good_model.predict.return_value = np.array([42.0])

    router = BoardModelRouter()
    fake_path = Path("/fake/main_good.pkl")
    router._path_map["MAIN"] = fake_path

    with patch("quantmind.models.factor_model.FactorModel") as MockFM:
        MockFM.load.return_value = good_model
        with patch.object(Path, "is_file", return_value=True):
            result = router._get_model("MAIN")

    # direction=+1 → 正常通过，返回板块专用模型
    assert result is good_model, "direction=+1 模型应正常通过门禁"


# ── 测试 17：predict() 二次防御 — 已缓存 direction=-1 模型时兜底 fallback ─────

def test_predict_secondary_defense_catches_direction_minus_one():
    """predict() 内部二次防御：若 _models 中存在 direction=-1 模型应切换 fallback。"""
    bad_model = MagicMock()
    bad_model.direction = -1
    bad_model._feature_names = [f"f{i}" for i in range(10)]
    bad_model.predict.return_value = np.array([99.0])   # 不应被调用

    fallback_model = MagicMock()
    fallback_model.direction = 1
    fallback_model._feature_names = [f"f{i}" for i in range(10)]
    fallback_model.predict.return_value = np.array([33.0])

    # 直接把 direction=-1 的模型注入 _models（绕过 _get_model 门禁，模拟异常注入）
    router = BoardModelRouter()
    router._models   = {"GEM": bad_model}
    router._fallback = fallback_model

    tickers = ["300750.SZ"]
    feat_df = _make_features(tickers)
    scores  = router.predict(feat_df)

    # 二次防御应捕获，使用 fallback 打分
    assert scores["300750.SZ"] == pytest.approx(33.0), (
        "predict() 二次防御应拦截 direction=-1 模型并改用 fallback"
    )
    bad_model.predict.assert_not_called()


# ── 测试 18：direction gate 集成 — 全板块 direction=-1 均降级，无 NaN ─────────

def test_direction_gate_all_boards_degrade_gracefully():
    """三个板块全部 direction=-1 时，全部降级 fallback，结果无 NaN、index 完整。"""
    def make_bad_model():
        m = MagicMock()
        m.direction = -1
        m._feature_names = [f"f{i}" for i in range(5)]
        m.predict.return_value = np.array([])  # 不应被调用
        return m

    fallback = MagicMock()
    fallback.direction = 1
    fallback._feature_names = [f"f{i}" for i in range(5)]
    fallback.predict.side_effect = lambda X: np.ones(len(X)) * 7.0

    tickers = ["600519.SH", "300750.SZ", "688001.SH"]
    feat_df = _make_features(tickers, n_feats=5)

    # 注入三个 direction=-1 的坏模型
    router = BoardModelRouter()
    router._models   = {
        "MAIN": make_bad_model(),
        "GEM":  make_bad_model(),
        "STAR": make_bad_model(),
    }
    router._fallback = fallback

    scores = router.predict(feat_df)

    assert len(scores) == 3
    assert not scores.isna().any(), f"出现 NaN: {scores[scores.isna()]}"
    assert list(scores.index) == tickers
    # 全部由 fallback 打分，值应为 7.0
    assert np.allclose(scores.values, 7.0), (
        f"全板块 direction=-1 应全部用 fallback 打分（期望值=7.0），实际={scores.values}"
    )


# ── 测试 19：ic_mean ≤ 0 触发质量门禁② ─────────────────────────────────────────

def test_negative_ic_triggers_fallback():
    """_get_model：direction=+1 但 ic_mean ≤ 0 时应降级 fallback（质量门禁②）。"""
    bad_model = MagicMock()
    bad_model.direction = 1          # 方向正确
    bad_model.ic_mean   = -0.003     # IC 为负 → 无预测力
    bad_model._feature_names = [f"f{i}" for i in range(5)]

    fallback_model = MagicMock()
    fallback_model.direction = 1
    fallback_model._feature_names = [f"f{i}" for i in range(5)]
    fallback_model.predict.return_value = np.array([55.0])

    router = BoardModelRouter()
    fake_path = Path("/fake/gem_bad_ic.pkl")
    router._path_map["GEM"] = fake_path
    router._fallback = fallback_model

    with patch("quantmind.models.factor_model.FactorModel") as MockFM:
        MockFM.load.return_value = bad_model
        with patch.object(Path, "is_file", return_value=True):
            result = router._get_model("GEM")

    # ic_mean ≤ 0 → 质量门禁② → 应返回 fallback
    assert result is fallback_model, (
        "ic_mean ≤ 0 的模型应被质量门禁②拦截并降级为 fallback"
    )


def test_zero_ic_also_triggers_fallback():
    """_get_model：ic_mean == 0 时同样视为无预测力，降级 fallback。"""
    zero_ic_model = MagicMock()
    zero_ic_model.direction = 1
    zero_ic_model.ic_mean   = 0.0    # 边界：等于 0 也应降级

    fallback_model = MagicMock()
    fallback_model.direction = 1

    router = BoardModelRouter()
    fake_path = Path("/fake/star_zero_ic.pkl")
    router._path_map["STAR"] = fake_path
    router._fallback = fallback_model

    with patch("quantmind.models.factor_model.FactorModel") as MockFM:
        MockFM.load.return_value = zero_ic_model
        with patch.object(Path, "is_file", return_value=True):
            result = router._get_model("STAR")

    assert result is fallback_model, "ic_mean == 0 也应触发质量门禁②"


def test_positive_ic_passes_gate():
    """_get_model：direction=+1 且 ic_mean > 0 时应正常通过两道门禁。"""
    good_model = MagicMock()
    good_model.direction = 1
    good_model.ic_mean   = 0.025     # IC > 0 → 通过

    fallback_model = MagicMock()

    router = BoardModelRouter()
    fake_path = Path("/fake/main_ok.pkl")
    router._path_map["MAIN"] = fake_path
    router._fallback = fallback_model

    with patch("quantmind.models.factor_model.FactorModel") as MockFM:
        MockFM.load.return_value = good_model
        with patch.object(Path, "is_file", return_value=True):
            result = router._get_model("MAIN")

    assert result is good_model, "direction=+1 且 ic_mean>0 应通过双重门禁"
    assert result is not fallback_model


# ── 测试 20：get_routing_status 返回三板块状态字典 ────────────────────────────

def test_get_routing_status_returns_all_boards():
    """get_routing_status：应返回 MAIN/GEM/STAR 三个 key，每个包含必要字段。"""
    fallback_model = MagicMock()
    fallback_model.direction = 1
    fallback_model.ic_mean   = 0.03
    fallback_model._feature_names = [f"f{i}" for i in range(8)]

    router = BoardModelRouter()
    # 三个板块文件均不存在 → 全部使用 fallback
    for board in ("MAIN", "GEM", "STAR"):
        router._path_map[board] = Path(f"/nonexistent/{board}.pkl")
    router._fallback = fallback_model

    with patch.object(Path, "is_file", return_value=False):
        status = router.get_routing_status()

    assert set(status.keys()) == {"MAIN", "GEM", "STAR"}, (
        f"get_routing_status 应包含三个板块，实际 keys={set(status.keys())}"
    )
    required_fields = {"model_path", "is_fallback", "direction", "ic_mean", "n_features", "reason"}
    for board, info in status.items():
        assert required_fields.issubset(info.keys()), (
            f"{board} 状态字典缺少必要字段，实际={set(info.keys())}"
        )
        assert info["is_fallback"] is True, f"{board} 文件不存在时应标记 is_fallback=True"
        assert "不存在" in info["reason"] or "fallback" in info["reason"].lower(), (
            f"{board} reason 描述不符合预期：{info['reason']}"
        )
