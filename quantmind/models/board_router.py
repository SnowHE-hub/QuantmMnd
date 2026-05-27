"""quantmind/models/board_router.py — 按板块路由 LGBM 模型打分.

根据股票代码（ts_code）将截面特征 DataFrame 分组，
路由到对应板块（主板/创业板/科创板）的专用 LGBM 模型进行打分，
结果合并后保持与输入完全相同的 index。

板块模型质量门禁
-----------------
只有 ``direction == +1`` 的模型才会被使用。若某板块专用模型
direction=-1（即 auto_flip 触发、raw IC 为负），路由器自动降级到
lgbm_v6_alpha 混训 fallback，并打印 WARNING 日志。这保证了打分
方向永远是"分数越高 → 预期收益越高"的正向含义，不会因为
LGBMRankerModel.predict() 内部乘以 direction 后虽然方向正确但
模型本身质量过低而污染组合。

若某板块专用模型不存在，同样自动降级到 lgbm_v6_alpha 混训模型，
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

    质量门禁
    --------
    只有 ``direction == +1`` 的板块模型才会被实际使用；
    ``direction == -1`` 的模型会在加载时被降级为 fallback，
    并记录 WARNING 日志。

    Parameters
    ----------
    model_paths : dict | None
        {board: path} 覆盖默认路径。
        board ∈ {"MAIN", "GEM", "STAR"}
    fallback_path : str | Path | None
        混训模型路径（降级用），默认 models/lgbm_v6_alpha.pkl。

    Attributes
    ----------
    _models : dict[str, FactorModel]
        已加载且通过方向校验的模型缓存（懒加载）。
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

        所有经过 _get_model() 的模型已确保 direction==+1；
        若发现异常（direction≠+1），此处作二次防御并切换 fallback。

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

            # ── 二次防御：确认 direction == +1 ───────────────────────────────
            direction = getattr(model, "direction", 1)
            if direction != 1:
                log.error(
                    "[BoardRouter] ⚠ 断言失败：%s 模型 direction=%d（应=+1），"
                    "_get_model 应已在加载时降级。强制切换 fallback。",
                    board, direction,
                )
                model = self._get_fallback()

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
                    sub = sub.copy()
                    for m in missing:
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
            # 还要检查方向
            try:
                from quantmind.models.factor_model import FactorModel
                m = FactorModel.load(path)
                if getattr(m, "direction", 1) == 1:
                    return str(path)
            except Exception:
                pass
        return str(self._fallback_path)

    # ── 私有：模型懒加载 ──────────────────────────────────────────────────────

    def _should_use_board_model(
        self,
        board_model: Any,
        fallback_model: Any,
        board: str,
    ) -> bool:
        """IC 相对比较门禁：专用模型 IC 须高于混训 × 0.9 才被使用。

        规则
        ----
        - 专用模型 ic_mean ≤ 0 → 无预测力，返回 False。
        - fallback ic_mean > 0 且专用模型 ic_mean < fallback × 0.9 →
          专用模型性能显著落后于混训，降级。容忍 10% 波动，避免随机噪声触发切换。
        - 专用模型 ic_mean 缺失（None/0 属性不存在）→ 跳过本门禁，返回 True
          （由调用方的双重门禁兜底）。

        Parameters
        ----------
        board_model :
            候选专用模型。
        fallback_model :
            混训 fallback 模型。
        board : str
            板块名称（用于日志）。

        Returns
        -------
        bool
            True → 使用专用模型；False → 降级 fallback。
        """
        board_ic    = getattr(board_model,    "ic_mean", None)
        fallback_ic = getattr(fallback_model, "ic_mean", None)

        # 只处理数值型 ic_mean；非数值（None / MagicMock / str）→ 跳过本门禁
        if not isinstance(board_ic, (int, float)):
            return True
        if not isinstance(fallback_ic, (int, float)):
            fallback_ic = None

        # 专用 IC ≤ 0 → 无预测力
        if board_ic <= 0:
            log.warning(
                "[BoardRouter] %s 专用模型 ic_mean=%.4f ≤ 0 → IC 比较门禁③：降级 fallback",
                board, board_ic,
            )
            return False

        # 若混训 IC 有效，要求专用 IC 不低于混训 × 0.9
        if fallback_ic is not None and fallback_ic > 0:
            threshold = fallback_ic * 0.9
            if board_ic < threshold:
                log.warning(
                    "[BoardRouter] %s 专用 IC=%.4f < 混训 IC=%.4f × 0.9=%.4f"
                    " → IC 比较门禁③：降级到混训模型",
                    board, board_ic, fallback_ic, threshold,
                )
                return False

        return True

    def _get_model(self, board: str) -> Any:
        """懒加载指定板块模型；不达质量门禁时返回 fallback.

        质量门禁（三重校验）
        --------------------
        加载成功后按以下顺序校验：

        1. ``direction == +1``
           direction=-1 → raw IC 为负，auto_flip 已触发但模型质量可疑；
           记录 WARNING 并降级 fallback。

        2. ``ic_mean > 0``（若属性存在）
           ic_mean <= 0 → 有效 IC 非正，模型无预测力；
           记录 WARNING 并降级 fallback。

        3. ``_should_use_board_model()`` IC 相对比较
           专用模型 IC 显著低于混训（< fallback × 0.9）→ 降级 fallback。

        通过三重门禁的模型才进入缓存并被实际使用。

        Returns
        -------
        Any
            通过质量门禁的模型实例（板块专用 或 fallback）
        """
        if board not in self._models:
            path = self._path_map.get(board)
            if path and path.is_file():
                try:
                    from quantmind.models.factor_model import FactorModel
                    candidate = FactorModel.load(path)
                    direction = getattr(candidate, "direction", 1)
                    ic_mean   = getattr(candidate, "ic_mean", None)

                    # 门禁 1：direction 必须为 +1
                    if direction != 1:
                        log.warning(
                            "[BoardRouter] %s 专用模型 direction=%d（raw IC 为负，"
                            "auto_flip 已触发）→ 质量门禁①：降级 fallback。"
                            "建议重新训练该板块模型或检查标签方向。",
                            board, direction,
                        )
                        self._models[board] = self._get_fallback()
                    # 门禁 2：若有 ic_mean，必须 > 0
                    elif ic_mean is not None and ic_mean <= 0:
                        log.warning(
                            "[BoardRouter] %s 专用模型 ic_mean=%.4f ≤ 0 → 质量门禁②：降级 fallback。"
                            "建议检查特征质量或增加训练数据量。",
                            board, ic_mean,
                        )
                        self._models[board] = self._get_fallback()
                    # 门禁 3：专用模型 IC 须高于混训 × 0.9
                    elif not self._should_use_board_model(
                        candidate, self._get_fallback(), board
                    ):
                        self._models[board] = self._get_fallback()
                    else:
                        self._models[board] = candidate
                        log.info(
                            "[BoardRouter] 已加载 %s 模型: %s (direction=%d, ic_mean=%s)",
                            board, path.name, direction,
                            f"{ic_mean:.4f}" if ic_mean is not None else "N/A",
                        )
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
                fb_dir = getattr(self._fallback, "direction", "N/A")
                log.info(
                    "[BoardRouter] Fallback 模型已加载: %s (direction=%s)",
                    self._fallback_path.name, fb_dir,
                )
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
            if not path.is_file():
                result[board] = f"❌ 不存在 → fallback ({self._fallback_path.name})"
                continue
            # 检查 direction
            try:
                from quantmind.models.factor_model import FactorModel
                m = FactorModel.load(path)
                direction = getattr(m, "direction", 1)
                if direction != 1:
                    result[board] = (
                        f"⚠ {path.name} direction={direction} "
                        f"→ 质量门禁降级 fallback ({self._fallback_path.name})"
                    )
                else:
                    result[board] = f"✅ {path.name} (direction=+1)"
            except Exception as exc:
                result[board] = f"❌ 加载失败({exc}) → fallback"
        return result

    def get_routing_status(self) -> dict[str, dict]:
        """返回各板块当前路由状态（懒加载后的实际使用情况）.

        供 Streamlit 控制台页面和 CLI 展示路由决策，格式：

        .. code-block:: python

            {
              "MAIN": {
                "model_path": "models/lgbm_v6_main.pkl",
                "is_fallback": True,
                "direction": -1,
                "ic_mean": "N/A",
                "n_features": 38,
                "reason": "direction=-1，质量门禁①降级",
              },
              ...
            }

        Returns
        -------
        dict[str, dict]
            每个 board → 详细状态字典
        """
        result: dict[str, dict] = {}
        for board in ("MAIN", "GEM", "STAR"):
            path  = self._path_map.get(board)
            model = self._get_model(board)   # 触发懒加载（含门禁校验）
            fb    = self._get_fallback()

            is_fallback = model is fb

            # 从实际加载的模型读属性（may be fallback）
            direction  = getattr(model, "direction", "N/A")
            ic_mean    = getattr(model, "ic_mean",   "N/A")
            feat_names = getattr(model, "_feature_names", None) or []
            n_features = len(feat_names)

            # 判断降级原因
            if not is_fallback:
                reason = "✅ 专用模型通过质量门禁"
            elif path and path.is_file():
                # 文件存在但被降级 → 读原始模型的 direction/ic_mean
                try:
                    from quantmind.models.factor_model import FactorModel
                    raw = FactorModel.load(path)
                    raw_dir = getattr(raw, "direction", "N/A")
                    raw_ic  = getattr(raw, "ic_mean",   None)
                    fb_model = self._get_fallback()
                    if raw_dir != 1:
                        reason = f"⚠ direction={raw_dir}，质量门禁①降级"
                    elif raw_ic is not None and raw_ic <= 0:
                        reason = f"⚠ ic_mean={raw_ic:.4f}≤0，质量门禁②降级"
                    elif not self._should_use_board_model(raw, fb_model, board):
                        fb_ic = getattr(fb_model, "ic_mean", None)
                        reason = (
                            f"⚠ 专用IC={raw_ic:.4f} < 混训IC={fb_ic:.4f}×0.9，"
                            f"IC比较门禁③降级"
                        ) if raw_ic is not None and fb_ic is not None else "⚠ IC比较门禁③降级"
                    else:
                        reason = "⚠ 加载异常，降级 fallback"
                except Exception:
                    reason = "⚠ 模型加载失败，降级 fallback"
            else:
                reason = "❌ 专用模型文件不存在，使用 fallback"

            result[board] = {
                "model_path": str(path) if path else "",
                "is_fallback": is_fallback,
                "direction":   direction,
                "ic_mean":     ic_mean,
                "n_features":  n_features,
                "reason":      reason,
            }
        return result

    def __repr__(self) -> str:
        status_lines = ", ".join(
            f"{b}={'✅' if p.is_file() else '❌'}"
            for b, p in self._path_map.items()
        )
        return f"BoardModelRouter({status_lines})"
