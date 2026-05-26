"""quantmind/models/board_router.py — 按板块路由 LGBM 模型打分.

根据股票代码（ts_code）将截面特征 DataFrame 分组，
路由到对应板块（主板/创业板/科创板）的专用 LGBM 模型进行打分，
结果合并后保持与输入完全相同的 index。

若某板块专用模型不存在，自动降级到 lgbm_v6_alpha 混训模型，
不影响现有流程。

Example
-------
>>> from quantmind.models.board_router import BoardModelRouter
>>> router = BoardModelRouter()
>>> scores = router.predict(features_df)  # pd.Series, same index as features_df
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


# ── 板块识别 ──────────────────────────────────────────────────────────────────

def get_board(ticker: str) -> str:
    """按 A 股代码规则判断板块。

    规则
    ----
    - 688xxx.SH → STAR  （科创板）
    - 300xxx.SZ → GEM   （创业板）
    - 其余（含 .BJ 北交所）→ MAIN （主板，统一降级处理）

    Parameters
    ----------
    ticker : str
        例如 ``"688001.SH"``、``"300750.SZ"``、``"600519.SH"``

    Returns
    -------
    str  ``"STAR"`` | ``"GEM"`` | ``"MAIN"``
    """
    code = ticker.split(".")[0]
    if ticker.endswith(".SH") and code.startswith("688"):
        return "STAR"
    if ticker.endswith(".SZ") and code.startswith("300"):
        return "GEM"
    return "MAIN"


# ── 路由器 ────────────────────────────────────────────────────────────────────

class BoardModelRouter:
    """按板块路由 LGBM 模型，对混合截面 DataFrame 分组打分.

    Parameters
    ----------
    model_paths : dict | None
        {board: path} 覆盖默认路径。
        board ∈ {"MAIN", "GEM", "STAR"}
    fallback_path : str | Path | None
        混训模型路径（降级用），默认 models/lgbm_v6_alpha.pkl。

    Attributes
    ----------
    models : dict[str, FactorModel]
        已加载的模型缓存（懒加载）。
    """

    # 默认板块模型路径（相对于 _ROOT）
    DEFAULT_PATHS: dict[str, str] = {
        "MAIN": "models/lgbm_v6_main.pkl",
        "GEM":  "models/lgbm_v6_gem.pkl",
        "STAR": "models/lgbm_v6_star.pkl",
    }
    FALLBACK_PATH = "models/lgbm_v6_alpha.pkl"

    def __init__(
        self,
        model_paths: dict[str, str | Path] | None = None,
        fallback_path: str | Path | None = None,
    ) -> None:
        self._path_map: dict[str, Path] = {
            board: _ROOT / path
            for board, path in (model_paths or self.DEFAULT_PATHS).items()
        }
        self._fallback_path: Path = _ROOT / (fallback_path or self.FALLBACK_PATH)
        self._models: dict[str, Any] = {}  # board → loaded model（懒加载）
        self._fallback: Any = None

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """对混合板块的特征 DataFrame 分组路由打分.

        对每只股票，按板块选择对应模型预测，再合并为一个 Series。
        输出 index 与输入完全一致（顺序和内容均相同）。

        Parameters
        ----------
        features_df : pd.DataFrame
            行索引为 ticker（ts_code），列为因子特征。

        Returns
        -------
        pd.Series
            index 与 features_df.index 完全一致，values 为预测分数（越高越好）。
        """
        if features_df.empty:
            return pd.Series(dtype=float)

        scores = pd.Series(np.nan, index=features_df.index, dtype=float)

        # 按板块分组
        board_groups: dict[str, list] = {}
        for ticker in features_df.index:
            board = get_board(str(ticker))
            board_groups.setdefault(board, []).append(ticker)

        for board, tickers in board_groups.items():
            model = self._get_model(board)
            feat_names = getattr(model, "_feature_names", None)

            sub = features_df.loc[tickers]

            if feat_names:
                # 只取模型实际需要的特征列
                missing = [c for c in feat_names if c not in sub.columns]
                if missing:
                    log.warning(
                        "[BoardRouter] %s 模型缺少 %d 列特征（首5: %s），用 0 填充",
                        board, len(missing), missing[:5],
                    )
                    for m in missing:
                        sub = sub.copy()
                        sub[m] = 0.0
                X = sub[feat_names].fillna(0.0).to_numpy(dtype=np.float32)
            else:
                X = sub.fillna(0.0).to_numpy(dtype=np.float32)

            try:
                pred = model.predict(X)
                scores.loc[tickers] = pred
                log.debug("[BoardRouter] %s: %d 只股票打分完成", board, len(tickers))
            except Exception as exc:
                log.warning(
                    "[BoardRouter] %s 模型打分失败（%s），降级 fallback", board, exc
                )
                fb = self._get_fallback()
                if fb is not None:
                    fb_feat = getattr(fb, "_feature_names", None)
                    X_fb = sub[fb_feat].fillna(0.0).to_numpy(dtype=np.float32) \
                        if fb_feat else X
                    scores.loc[tickers] = fb.predict(X_fb)

        return scores

    def which_model(self, ticker: str) -> str:
        """返回该股票将使用的模型路径（用于调试/日志）."""
        board = get_board(ticker)
        path  = self._path_map.get(board)
        if path and path.is_file():
            return str(path)
        return str(self._fallback_path)

    # ── 私有：模型懒加载 ──────────────────────────────────────────────────────

    def _get_model(self, board: str) -> Any:
        """懒加载指定板块模型；若不存在则返回 fallback."""
        if board not in self._models:
            path = self._path_map.get(board)
            if path and path.is_file():
                try:
                    from quantmind.models.factor_model import FactorModel
                    self._models[board] = FactorModel.load(path)
                    log.info("[BoardRouter] 已加载 %s 模型: %s", board, path.name)
                except Exception as exc:
                    log.warning(
                        "[BoardRouter] 加载 %s 模型失败（%s），将使用 fallback", board, exc
                    )
                    self._models[board] = self._get_fallback()
            else:
                log.info(
                    "[BoardRouter] %s 专用模型不存在（%s），使用 fallback",
                    board, path
                )
                self._models[board] = self._get_fallback()
        return self._models[board]

    def _get_fallback(self) -> Any:
        """懒加载混训 fallback 模型."""
        if self._fallback is None:
            if self._fallback_path.is_file():
                from quantmind.models.factor_model import FactorModel
                self._fallback = FactorModel.load(self._fallback_path)
                log.info("[BoardRouter] Fallback 模型已加载: %s", self._fallback_path.name)
            else:
                raise FileNotFoundError(
                    f"BoardModelRouter fallback 模型不存在: {self._fallback_path}"
                )
        return self._fallback

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def status(self) -> dict[str, str]:
        """返回各板块模型路径状态（存在/不存在/降级）."""
        result = {}
        for board, path in self._path_map.items():
            if path.is_file():
                result[board] = f"✅ {path.name}"
            else:
                result[board] = f"❌ 不存在 → fallback ({self._fallback_path.name})"
        return result

    def __repr__(self) -> str:
        status_lines = ", ".join(
            f"{b}={'✅' if p.is_file() else '❌'}"
            for b, p in self._path_map.items()
        )
        return f"BoardModelRouter({status_lines})"
