"""scripts/daily_update.py — 每日数据更新 + 推荐流水线.

按顺序执行 Step1–Step8；Step9（RAG 报告）在 Step8 之后可选执行。
任一步失败记录日志后继续（Step9 失败不视为流水线失败）。

用法
====
::

    # 今日更新
    python scripts/daily_update.py

    # 指定日期（补跑历史）
    python scripts/daily_update.py --date 2025-05-08

    # 强制重跑（忽略缓存）
    python scripts/daily_update.py --force

    # 只跑到某一步（调试用）
    python scripts/daily_update.py --stop-after step3

    # 跳过 LLM（仅做 LGBM 排名，不调 Ollama）
    python scripts/daily_update.py --no-llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

# ── 常量 ─────────────────────────────────────────────────────────────────────

STEPS = ["step1", "step2", "step3", "step4", "step5", "step5b", "step6", "step7", "step7a", "step8"]

_STEP_NAMES = {
    "step1": "确定交易日",
    "step2": "检查缓存",
    "step3": "下载快照",
    "step4": "构建因子",
    "step5": "LGBM 粗排",
    "step5b": "HRP/Kelly 仓位优化",
    "step6": "LLM Reranker",
    "step7": "保存推荐 JSON",
    "step7a": "6-Agent 投资分析",
    "step8": "生成 HTML 报告",
    "step11": "每日预警推送",
}

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QuantMind 每日更新流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--date", default=None,
                   help="指定日期 YYYY-MM-DD（默认今日最近交易日）")
    p.add_argument("--force", action="store_true",
                   help="强制重跑，忽略已存在的快照/因子缓存")
    p.add_argument("--stop-after", choices=STEPS, default=None,
                   help="执行到指定 step 后停止（调试用）")
    p.add_argument("--universe", default="alpha",
                   help="股票池（默认 alpha，即 1374 只 alpha 宇宙；也可用 csi300）")
    p.add_argument("--lgbm-top", type=int, default=50,
                   help="LGBM 粗排保留数量（默认 50）")
    p.add_argument("--llm-top", type=int, default=10,
                   help="LLM 精排最终推荐数量（默认 10）")
    p.add_argument("--no-llm", action="store_true",
                   help="跳过 LLM Reranker（step6），仅使用 LGBM 排名")
    p.add_argument("--provider", default="dashscope",
                   help="LLM provider（默认 dashscope）")
    p.add_argument("--model", default="qwen-plus",
                   help="LLM 模型名（默认 qwen-plus）")
    p.add_argument(
        "--lgbm-model",
        type=Path,
        default=Path("models/lgbm_v3_top18.pkl"),
        help="因子排序 FactorModel pickle（默认 models/lgbm_v3_top18.pkl，即 18因子 alpha v3 模型）",
    )
    p.add_argument("--collection-name", default="default",
                   help="Chroma collection（Step 9 RAG 与 KB 一致）")
    p.add_argument("--chroma-dir", default=".cache/chromadb",
                   help="Chroma 持久化目录")
    p.add_argument("--funnel-skip-layers", type=int, nargs="*", default=[],
                   help="漏斗选股跳过的层（例如 --funnel-skip-layers 4，仅 universe=full_a 时有效）")
    p.add_argument("--max-tickers", type=int, default=None,
                   help="下载快照时限制股票数（调试加速）")
    p.add_argument("--no-agent", action="store_true",
                   help="跳过 step7a 6-Agent 分析（加快 daily_update，适合调试）")
    p.add_argument("--no-alert", action="store_true",
                   help="跳过 step11 每日预警推送（企业微信/邮件）")
    p.add_argument("--no-cnn", action="store_true",
                   help="跳过 step5c FactorCNN ensemble，仅使用 LGBM 排序")
    p.add_argument("--agent-top", type=int, default=10,
                   help="step7a 中对多少只 top 股票做 Agent 分析（默认 10）")
    p.add_argument("--agent-provider", default="none",
                   help="step7a Agent LLM provider（默认 none，仅 ML+rules 模式）")
    p.add_argument("--agent-model", default="qwen-plus",
                   help="step7a Agent LLM 模型（agent-provider=none 时忽略）")
    p.add_argument("--auto-regime", action="store_true",
                   help="根据市场状态自动切换 large/small 子模型（需要 lgbm_ensemble_large/small.pkl）")
    p.add_argument(
        "--position-sizing",
        choices=["equal", "hrp", "kelly", "blend"],
        default="equal",
        help="step5b 仓位优化方式：equal 等权 | hrp HRP | kelly 分数Kelly | blend 混合（默认 equal）",
    )
    p.add_argument("--kelly-fraction", type=float, default=0.5,
                   help="Kelly 分数（0.25~0.5 推荐，默认 0.5）")
    return p.parse_args()


_CHAMPION_MODEL_PATH = _ROOT / "models" / "lgbm_v6_alpha.pkl"
_CHAMPION_META_PATH  = _ROOT / "models" / "lgbm_v6_alpha.meta.json"
_EXPECTED_FEATURES   = 38
_EXPECTED_DIRECTION  = 1


def _sanity_check_champion_model() -> None:
    """启动时校验冠军模型完整性，任一项不通过立即 raise RuntimeError."""
    import pickle

    errors: list[str] = []

    if not _CHAMPION_MODEL_PATH.exists():
        raise RuntimeError(f"[SanityCheck] 冠军模型文件不存在: {_CHAMPION_MODEL_PATH}")

    try:
        with open(_CHAMPION_MODEL_PATH, "rb") as _f:
            _m = pickle.load(_f)
        direction = getattr(_m, "direction", None)
        n_features = len(getattr(_m, "_feature_names", None) or [])
        if direction != _EXPECTED_DIRECTION:
            errors.append(f"direction={direction}（期望 {_EXPECTED_DIRECTION}）")
        if n_features != _EXPECTED_FEATURES:
            errors.append(f"特征数量={n_features}（期望 {_EXPECTED_FEATURES}）")
    except Exception as _e:
        raise RuntimeError(f"[SanityCheck] 无法加载冠军模型: {_e}") from _e

    if _CHAMPION_META_PATH.exists():
        try:
            _meta = json.loads(_CHAMPION_META_PATH.read_text(encoding="utf-8"))
            ic_mean = _meta.get("ic_mean")
            if ic_mean is None or float(ic_mean) <= 0:
                errors.append(f"meta.ic_mean={ic_mean}（期望 >0）")
        except Exception as _e:
            errors.append(f"meta.json 解析失败: {_e}")
    else:
        errors.append("meta.json 不存在，无法验证 ic_mean")

    if errors:
        msg = "冠军模型完整性校验失败：" + "；".join(errors)
        logger.error(f"[SanityCheck] ❌ {msg}")
        raise RuntimeError(msg)

    logger.info(
        f"[SanityCheck] ✅ 冠军模型校验通过 "
        f"(direction={_EXPECTED_DIRECTION}, features={n_features}, ic_mean={ic_mean:.4f})"
    )


def _snapshot_has_usable_cache(as_of: date) -> bool:
    """本地是否有可用的快照价格文件（下载失败时降级继续）."""
    from quantmind.core.config import get_settings

    snap_dir = Path(get_settings().data.dir).resolve() / "snapshots" / as_of.isoformat()
    return (snap_dir / "prices.parquet").is_file()


def _detect_market_regime() -> tuple[str, str]:
    """从 risk_hmm_v3 的 market_hmm bundle 读取最近市场状态。

    Returns:
        (regime_name, model_hint): regime_name = 'bull_low_vol'/'normal'/'bear_crisis'
                                   model_hint  = 'large'/'small'/'default'
    """
    try:
        import pickle
        bundle_path = _ROOT / "models" / "agents" / "risk_hmm_v3.pkl"
        if not bundle_path.is_file():
            return "unknown", "default"
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        mh = bundle.get("market_hmm") or {}
        current = int(mh.get("current_regime", 1))
        labels = mh.get("regime_labels") or {0: "bull_low_vol", 1: "normal", 2: "bear_crisis"}
        regime_name = labels.get(current, "normal")
        probs = mh.get("recent_30d_probs") or {}
        # 小盘/危机 → small-cap model；牛市低波 → large-cap model
        bear_prob = float(probs.get(2, 0.0)) + float(probs.get("bear_crisis", 0.0))
        bull_prob = float(probs.get(0, 0.0)) + float(probs.get("bull_low_vol", 0.0))
        if bear_prob > 0.5 or regime_name == "bear_crisis":
            model_hint = "small"
        elif bull_prob > 0.7 or regime_name == "bull_low_vol":
            model_hint = "large"
        else:
            model_hint = "default"
        return regime_name, model_hint
    except Exception as e:
        logger.warning(f"[Regime] 市场状态检测失败（{e}），使用默认模型")
        return "unknown", "default"


def _resolve_regime_model(args, regime_hint: str) -> Path:
    """根据 regime_hint 选择最优子模型路径。"""
    model_map = {
        "large": _ROOT / "models" / "lgbm_ensemble_large.pkl",
        "small": _ROOT / "models" / "lgbm_ensemble_small.pkl",
        "default": _ROOT / args.lgbm_model,
    }
    candidate = model_map.get(regime_hint, _ROOT / args.lgbm_model)
    if candidate.is_file():
        return candidate
    # fallback
    fallback = _ROOT / args.lgbm_model if not args.lgbm_model.is_absolute() else args.lgbm_model
    logger.warning(f"[Regime] 子模型 {candidate.name} 不存在，回退到 {fallback.name}")
    return fallback


def _today_calendar_skip_non_trading(args: argparse.Namespace) -> bool:
    """未指定 --date 时：若今日非 SSE 交易日则提示并结束（exit 0）."""
    if args.date is not None:
        return False
    today = date.today()
    try:
        from quantmind.data.sse_calendar import list_sse_trade_dates

        trade_days = list_sse_trade_dates(today - timedelta(days=15), today)
        if trade_days and today not in trade_days:
            logger.info("今日非交易日")
            return True
    except Exception as e:
        logger.warning(f"[日历] 无法判断今日是否交易日（{e}），继续执行")
    return False


# ── Step 实现 ─────────────────────────────────────────────────────────────────

def step1_resolve_date(args) -> date:
    """确定 as_of 日期（交易日历）."""
    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = date.today()

    # 尝试用 SSE 日历找最近交易日（需要 Tushare Token）
    try:
        from quantmind.data.sse_calendar import list_sse_trade_dates
        start = target - timedelta(days=10)
        trade_days = list_sse_trade_dates(start, target)
        if trade_days:
            as_of = max(d for d in trade_days if d <= target)
            logger.info(f"[Step1] 交易日历查询成功，as_of={as_of}")
            return as_of
    except Exception as e:
        logger.warning(f"[Step1] 交易日历查询失败（{e}），使用目标日期本身")

    # fallback：简单排除周末
    d = target
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    logger.info(f"[Step1] as_of={d}（fallback：排除周末）")
    return d


def step2_check_cache(as_of: date, force: bool) -> bool:
    """检查 snapshot 是否已存在.

    Returns:
        True = 缓存已存在且不 force → 跳过下载
        False = 需要下载
    """
    from quantmind.core.config import get_settings
    settings = get_settings()
    snap_dir = Path(settings.data.dir) / "snapshots" / as_of.isoformat()
    meta = snap_dir / "meta.json"
    if meta.exists() and not force:
        logger.info(f"[Step2] 快照已存在：{snap_dir}，跳过下载（--force 可强制重跑）")
        return True
    logger.info(f"[Step2] 快照不存在或 --force，将重新下载：{snap_dir}")
    return False


def step3_download_snapshot(as_of: date, args) -> bool:
    """下载今日数据快照."""
    from quantmind.data import build_snapshot
    from quantmind.data import get_universe
    from quantmind.data.universe import _ALPHA_UNIVERSE_NAMES
    logger.info(f"[Step3] 开始下载快照 as_of={as_of} universe={args.universe}")
    try:
        universe = get_universe(args.universe, as_of=as_of)
        if args.max_tickers:
            universe = universe[: args.max_tickers]
            logger.info(f"[Step3] 调试模式：只下载前 {args.max_tickers} 只股票")

        is_alpha = args.universe.strip().lower() in _ALPHA_UNIVERSE_NAMES
        if is_alpha:
            # alpha 宇宙：用 tickers_override=replace 强制替换成分股
            build_snapshot(
                as_of=as_of,
                tickers_override=universe,
                tickers_override_policy="replace",
                overwrite=args.force,
            )
        else:
            build_snapshot(
                as_of=as_of,
                universe_name=args.universe,
                tickers_override=universe if args.max_tickers else None,
                tickers_override_policy="filter",
                overwrite=args.force,
            )
        logger.info(f"[Step3] ✅ 快照下载完成（{len(universe)} 只股票）")
        return True
    except Exception as e:
        logger.error(f"[Step3] ❌ 快照下载失败：{e}")
        return False


def step4_build_features(as_of: date, args) -> tuple[bool, Path | None]:
    """构建今日因子截面."""
    from quantmind.core.config import get_settings
    from quantmind.features import FeaturePipeline

    settings = get_settings()
    feat_path = Path(settings.data.dir) / "features" / f"{args.universe}_{as_of.isoformat()}.parquet"

    if feat_path.exists() and not args.force:
        logger.info(f"[Step4] 因子已存在：{feat_path}，跳过重建")
        return True, feat_path

    logger.info(f"[Step4] 构建因子截面 as_of={as_of}")
    try:
        pipe = FeaturePipeline()
        df = pipe.run_single(as_of, universe=args.universe)
        saved = pipe.save(df, as_of, universe=args.universe)
        logger.info(f"[Step4] ✅ 因子构建完成，shape={df.shape}，保存至 {saved}")
        return True, saved
    except Exception as e:
        logger.error(f"[Step4] ❌ 因子构建失败：{e}")
        return False, None


def step5_lgbm_rank(
    as_of: date, feat_path: Path | None, args
) -> tuple[bool, list[dict]]:
    """运行 LightGBM 粗排，返回 Top-N 候选列表."""
    import numpy as np
    import pandas as pd

    from quantmind.models.factor_model import FactorModel
    from quantmind.utils.score_order import order_preserving_pct_rank

    # Regime-aware 模型选择（--auto-regime 开关）
    if getattr(args, "auto_regime", False):
        regime_name, regime_hint = _detect_market_regime()
        model_path = _resolve_regime_model(args, regime_hint)
        logger.info(f"[Step5] 🔀 Regime={regime_name} → 子模型={model_path.name}")
    else:
        model_path = (
            args.lgbm_model if args.lgbm_model.is_absolute() else _ROOT / args.lgbm_model
        )
    if not model_path.is_file():
        logger.error(f"[Step5] ❌ 模型文件不存在：{model_path}")
        return False, []

    # 加载因子数据
    if feat_path and feat_path.exists():
        df = pd.read_parquet(feat_path)
    else:
        # 尝试从已存在的合并面板里截取
        logger.warning("[Step5] 单日因子文件不存在，尝试从合并面板截取")
        panel_files = sorted((_ROOT / "data" / "features").glob("*Q*.parquet"))
        if not panel_files:
            logger.error("[Step5] ❌ 找不到任何因子文件")
            return False, []
        panel = pd.read_parquet(panel_files[-1])
        dates = sorted(panel.index.get_level_values(0).unique())
        closest = min(dates, key=lambda d: abs((pd.Timestamp(d) - pd.Timestamp(as_of)).days))
        df = panel.xs(closest, level=0)
        logger.info(f"[Step5] 使用合并面板，最近截面日期={closest}")

    logger.info(f"[Step5] 加载 FactorModel（粗排），路径={model_path}")
    try:
        model = FactorModel.load(model_path)
        feat_names = getattr(model, "_feature_names", None) or model.feature_names
        if not feat_names:
            logger.error("[Step5] ❌ 模型无特征名")
            return False, []
        missing = [c for c in feat_names if c not in df.columns]
        if missing:
            logger.error(
                f"[Step5] ❌ 因子表缺特征（前 5 个）：{missing[:5]} … 共 {len(missing)} 个"
            )
            return False, []
        X = df[list(feat_names)].to_numpy(dtype=np.float32, copy=True)
        scores_raw = model.predict(X)
        ser_raw = pd.Series(scores_raw, index=df.index, dtype=np.float64)
        rank_sel = order_preserving_pct_rank(ser_raw)
        score_norm = rank_sel.to_numpy(dtype=np.float64)
        norm_by_ticker = dict(zip(df.index.astype(str), score_norm))
        n_raw_u = int(len(np.unique(scores_raw.astype(np.float64))))
        n_norm_u = int(len(np.unique(score_norm[~np.isnan(score_norm)])))
        logger.info(
            f"[Step5] 原始分数 unique={n_raw_u}，order_pct 唯一值={n_norm_u}（全截面）"
        )
        top_n = min(args.lgbm_top, int(rank_sel.notna().sum()))
        top_order = rank_sel.dropna().sort_values(ascending=False).head(top_n)
        candidates = []
        for rank, ticker in enumerate(top_order.index, start=1):
            row = df.loc[ticker]
            raw = float(ser_raw.loc[ticker])
            norm = float(norm_by_ticker.get(str(ticker), np.nan))
            candidates.append({
                "lgbm_rank": rank,
                "ticker": str(ticker),
                "lgbm_score": norm,
                "lgbm_score_raw": raw,
                "key_factors": {
                    col: float(row[col])
                    for col in ["pe_ttm", "pb", "roe_ttm", "momentum_6m", "accruals"]
                    if col in row.index and not np.isnan(float(row[col]))
                },
            })

        logger.info(f"[Step5] ✅ LGBM粗排完成，Top-{top_n} 候选已生成")
        return True, candidates
    except Exception as e:
        logger.error(f"[Step5] ❌ LGBM 粗排失败：{e}")
        return False, []


# ── Step 5c helpers ───────────────────────────────────────────────────────────

_CNN_MODEL_PATH = _ROOT / "models" / "factor_cnn_v2_augmented.pkl"

# Regime → (lgbm_weight, cnn_weight)，与 strategy_config_v2.json 保持一致
_ENSEMBLE_WEIGHTS: dict[str, tuple[float, float]] = {
    "bull":    (0.60, 0.40),
    "neutral": (0.65, 0.35),
    "bear":    (0.75, 0.25),
}


def _load_cnn_model(pkl_path: Path):
    """从 save_cnn_model 生成的 pkl 中重建 FactorCNN 并返回 (model, feature_cols).

    文件不存在 → 返回 (None, []) 并记录 WARNING。
    """
    import pickle

    from quantmind.models.factor_cnn import FactorCNN

    if not pkl_path.is_file():
        logger.warning(
            f"[Step5c] ⚠ FactorCNN pkl 不存在：{pkl_path}，降级为纯 LGBM 排序"
        )
        return None, []

    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)

    if payload.get("__type__") != "CNNTrainResult":
        logger.warning("[Step5c] ⚠ pkl 类型不符（期望 CNNTrainResult），降级为纯 LGBM")
        return None, []

    model_state  = payload.get("model_state")
    model_config = payload.get("model_config") or {}
    feature_cols = payload.get("feature_cols") or []

    if model_state is None or not feature_cols:
        logger.warning("[Step5c] ⚠ pkl 缺少 model_state 或 feature_cols，降级为纯 LGBM")
        return None, []

    model = FactorCNN(**model_config)
    model.load_state_dict(model_state)
    model.eval()
    logger.info(
        f"[Step5c] FactorCNN 加载成功（val_ic={payload.get('val_ic_mean', float('nan')):.4f}，"
        f"特征数={len(feature_cols)}）"
    )
    return model, list(feature_cols)


def ensemble_scores(
    lgbm_score: float,
    cnn_score: float,
    lgbm_weight: float,
    cnn_weight: float,
) -> float:
    """加权融合两个已归一化到 [0,1] 的排名分数.

    任一分数为 NaN → 返回另一个（全 NaN 则返回 NaN）。
    """
    import math

    lgbm_nan = math.isnan(lgbm_score)
    cnn_nan  = math.isnan(cnn_score)
    if lgbm_nan and cnn_nan:
        return float("nan")
    if lgbm_nan:
        return float(cnn_score)
    if cnn_nan:
        return float(lgbm_score)
    return lgbm_weight * lgbm_score + cnn_weight * cnn_score


def step5c_cnn_ensemble(
    as_of: date,
    feat_path: "Path | None",
    candidates: list[dict],
    *,
    cnn_pkl: "Path | None" = None,
) -> list[dict]:
    """Step 5c: FactorCNN rank-based ensemble（6:4）.

    Parameters
    ----------
    as_of      : 当前交易日
    feat_path  : step4 输出的因子 parquet 路径（全宇宙）
    candidates : step5 输出的 Top-N 候选列表，每条含 ticker / lgbm_score
    cnn_pkl    : CNN 模型 pkl 路径；None = 使用默认路径

    Returns
    -------
    按 ensemble_score 降序重排的候选列表；每条新增：
      cnn_score       : CNN rank 百分位 [0,1]
      ensemble_score  : lgbm_w × lgbm_score + cnn_w × cnn_score
    """
    import numpy as np
    import pandas as pd

    from quantmind.models.factor_cnn import predict_cnn
    from quantmind.utils.score_order import order_preserving_pct_rank

    if not candidates:
        return candidates

    pkl_path = cnn_pkl or _CNN_MODEL_PATH
    model, feature_cols = _load_cnn_model(pkl_path)
    if model is None:
        # 降级：保留原始 lgbm_score 作为 ensemble_score
        for c in candidates:
            c.setdefault("cnn_score", float("nan"))
            c["ensemble_score"] = c.get("lgbm_score", float("nan"))
        return candidates

    # ── 读取全宇宙因子表 ───────────────────────────────────────────────────────
    df: pd.DataFrame | None = None
    if feat_path and Path(feat_path).exists():
        df = pd.read_parquet(feat_path)
    else:
        logger.warning("[Step5c] ⚠ 因子文件不存在，CNN 无法推理，降级纯 LGBM")
        for c in candidates:
            c.setdefault("cnn_score", float("nan"))
            c["ensemble_score"] = c.get("lgbm_score", float("nan"))
        return candidates

    # 只保留 CNN 需要的特征列（缺失列填 NaN）
    available = [col for col in feature_cols if col in df.columns]
    missing   = [col for col in feature_cols if col not in df.columns]
    if missing:
        logger.warning(
            f"[Step5c] 因子表缺少 {len(missing)} 列（前 5：{missing[:5]}），用 NaN 填充"
        )
        for col in missing:
            df[col] = np.nan

    # ── CNN 推理（全宇宙）─────────────────────────────────────────────────────
    try:
        cnn_raw: pd.Series = predict_cnn(model, df, feature_cols, device="cpu")
    except Exception as exc:
        logger.warning(f"[Step5c] ⚠ predict_cnn 失败（{exc}），降级纯 LGBM")
        for c in candidates:
            c.setdefault("cnn_score", float("nan"))
            c["ensemble_score"] = c.get("lgbm_score", float("nan"))
        return candidates

    # ── 排名归一化（与 LGBM order_pct_rank 对齐）──────────────────────────────
    cnn_rank: pd.Series = order_preserving_pct_rank(cnn_raw)   # [0,1]，全宇宙

    # ── Regime 权重 ───────────────────────────────────────────────────────────
    regime = "neutral"
    try:
        from quantmind.regime.hmm import RegimeHMM, build_observations
        hmm = RegimeHMM()
        obs_path = str(_ROOT / "data" / "raw" / "index_daily_panel.parquet")
        obs = build_observations(obs_path)
        if obs is not None and len(obs) > 0:
            regime = hmm.predict_regime(obs, pd.Timestamp(as_of))
    except Exception as exc:
        logger.debug(f"[Step5c] HMM Regime 获取失败，使用 neutral: {exc}")

    lgbm_w, cnn_w = _ENSEMBLE_WEIGHTS.get(str(regime), _ENSEMBLE_WEIGHTS["neutral"])
    logger.info(
        f"[Step5c] Regime={regime}，融合权重 LGBM:{lgbm_w:.2f} / CNN:{cnn_w:.2f}"
    )

    # ── 计算 ensemble_score 并更新候选列表 ────────────────────────────────────
    for c in candidates:
        ticker = c["ticker"]
        lgbm_s = float(c.get("lgbm_score", float("nan")))
        # cnn_rank 的 index 可能是 ts_code string 或 ticker
        cnn_s  = float(cnn_rank.get(ticker, float("nan")))
        c["cnn_score"]      = round(cnn_s,  6) if not np.isnan(cnn_s)  else float("nan")
        c["ensemble_score"] = round(
            ensemble_scores(lgbm_s, cnn_s, lgbm_w, cnn_w), 6
        )

    # 按 ensemble_score 降序重排
    candidates.sort(
        key=lambda c: c.get("ensemble_score") or float("-inf"),
        reverse=True,
    )
    # 更新 lgbm_rank → ensemble_rank
    for i, c in enumerate(candidates, start=1):
        c["lgbm_rank"] = c.get("lgbm_rank", i)   # 保留原始 LGBM rank
        c["ensemble_rank"] = i

    n_cnn_valid = sum(1 for c in candidates if not np.isnan(c.get("cnn_score", float("nan"))))
    logger.info(
        f"[Step5c] ✅ CNN ensemble 完成，有效 CNN 分数={n_cnn_valid}/{len(candidates)}，"
        f"Top-1={candidates[0]['ticker'] if candidates else 'N/A'}"
    )
    return candidates


def step5b_position_sizing(
    as_of: date,
    candidates: list[dict],
    args,
) -> tuple[bool, list[dict]]:
    """HRP / Kelly / 等权 仓位优化 — 为 Top-N 候选计算建议权重。

    流程
    ----
    1. 用历史价格计算 HRP / Kelly / 混合原始权重
    2. 若启用 Barra 约束（``args.barra_constrained`` 或配置 barra_* 字段），
       在原始权重基础上叠加行业 / 风格 / 换手率约束（cvxpy 求解）
    3. 写入每个 candidate dict 的 'weight' 字段，保存至
       reports/daily/<date>/position_weights.json
    """
    import json

    method = getattr(args, "position_sizing", "equal")
    if method == "equal" or not candidates:
        logger.info(f"[Step5b] 等权模式，跳过 HRP/Kelly 优化")
        for c in candidates:
            c.setdefault("weight", round(1.0 / len(candidates), 6))
        return True, candidates

    import pandas as pd
    from quantmind.portfolio.position_sizing import hrp_weights, kelly_weights, blend_weights

    tickers = [c["ticker"] for c in candidates]
    price_path = _ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
    try:
        import pyarrow.parquet as pq
        schema_names = pq.ParquetFile(str(price_path)).schema_arrow.names
        price_col = "adj_close" if "adj_close" in schema_names else "close"
        prices_long = pd.read_parquet(price_path, columns=["trade_date", "ts_code", price_col])
        prices_long = prices_long[prices_long["ts_code"].isin(tickers)]
        wide = prices_long.pivot_table(
            index="trade_date", columns="ts_code", values=price_col, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        cutoff = pd.Timestamp(as_of)
        hist = wide.loc[wide.index < cutoff].tail(252)
        rets = hist.pct_change().dropna(how="all")

        if method in ("hrp", "blend"):
            w_hrp = hrp_weights(rets, min_periods=63)
        if method in ("kelly", "blend"):
            mu = rets.mean() * 252
            cov_m = rets.cov() * 252
            w_kelly = kelly_weights(mu, cov_m.values, fraction=getattr(args, "kelly_fraction", 0.5))
            w_kelly.index = mu.index

        if method == "hrp":
            w_final = w_hrp
        elif method == "kelly":
            w_final = w_kelly
        else:
            w_final = blend_weights(w_hrp, w_kelly, alpha=0.5)

        # ── Barra 约束层（可选）─────────────────────────────────────────────
        barra_enabled = getattr(args, "barra_constrained", False)
        if not barra_enabled:
            # 也支持从 strategy_config_v2.json 读取开关
            _cfg_path = _ROOT / "data" / "paper_trading" / "strategy_config_v2.json"
            if _cfg_path.exists():
                try:
                    import json as _json
                    _cfg = _json.loads(_cfg_path.read_text(encoding="utf-8"))
                    barra_enabled = _cfg.get("barra_constrained", False)
                except Exception:
                    pass

        if barra_enabled:
            try:
                from quantmind.portfolio.constrained_optimizer import ConstrainedPortfolioOptimizer
                from quantmind.risk.barra import STYLE_MAP

                # 读取配置（若无则使用默认值）
                _cfg_path = _ROOT / "data" / "paper_trading" / "strategy_config_v2.json"
                barra_cfg: dict = {}
                if _cfg_path.exists():
                    try:
                        import json as _json
                        _cfg = _json.loads(_cfg_path.read_text(encoding="utf-8"))
                        barra_cfg = _cfg.get("barra_constraints", {})
                    except Exception:
                        pass

                ind_limit  = float(barra_cfg.get("industry_limit",  0.3))
                sty_limit  = float(barra_cfg.get("style_limit",     0.5))
                to_limit   = float(barra_cfg.get("turnover_limit",  0.6))
                risk_avers = float(barra_cfg.get("risk_aversion",   1.0))

                # 读取 alpha panel 最近一期截面
                panel_path = _ROOT / "data" / "panel" / "alpha_panel_v4.parquet"
                if panel_path.exists():
                    panel = pd.read_parquet(panel_path)
                    panel_dates = panel.index.get_level_values("as_of").unique()
                    valid_dates = panel_dates[panel_dates <= pd.Timestamp(as_of)]
                    if len(valid_dates) > 0:
                        latest_panel = valid_dates.max()
                        panel_xs = panel.xs(latest_panel, level="as_of")

                        # 构建因子暴露矩阵
                        factor_exp = ConstrainedPortfolioOptimizer.build_factor_exposures(
                            panel_xs.reindex(tickers)
                        )

                        # alpha scores = 候选排名得分
                        score_map = {c["ticker"]: float(c.get("predicted_score", 0.5))
                                     for c in candidates}
                        alpha_scores = pd.Series(score_map).reindex(tickers).fillna(0.0)

                        # 协方差矩阵（年化）
                        cov_mat = rets.reindex(columns=tickers).cov() * 252

                        # 上期权重（换手约束）
                        prev_w_json = _ROOT / "reports" / "daily"
                        prev_weights = None
                        # 尝试读取最新 position_weights.json
                        try:
                            prev_dates = sorted(prev_w_json.glob("*/position_weights.json"))
                            if prev_dates:
                                prev_data = json.loads(prev_dates[-1].read_text())
                                prev_weights = pd.Series(
                                    {r["ticker"]: r["weight"]
                                     for r in prev_data.get("weights", [])}
                                ).reindex(tickers).fillna(0.0)
                        except Exception:
                            pass

                        opt = ConstrainedPortfolioOptimizer(
                            industry_limit=ind_limit,
                            style_limit=sty_limit,
                            turnover_limit=to_limit,
                            risk_aversion=risk_avers,
                        )
                        w_constrained = opt.optimize(
                            alpha_scores, factor_exp, cov_mat, prev_weights
                        )

                        # 检验：若约束权重有效，替换原始权重
                        if w_constrained is not None and abs(w_constrained.sum() - 1.0) < 0.01:
                            logger.info(
                                "[Step5b] Barra 约束优化成功 "
                                "max_w=%.3f  HHI=%.4f",
                                float(w_constrained.max()),
                                float((w_constrained ** 2).sum()),
                            )
                            w_final = w_constrained

                            # 记录因子暴露到报告
                            exp_report = opt.compute_active_exposure(w_final, factor_exp)
                            out_dir = _ROOT / "reports" / "daily" / str(as_of)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            exp_report.reset_index().to_csv(
                                out_dir / "barra_exposure.csv", index=False
                            )
            except Exception as barra_exc:
                logger.warning("[Step5b] Barra 约束层失败（%s），使用原始权重", barra_exc)
        # ── end Barra 约束层 ──────────────────────────────────────────────────

        w_dict = w_final.to_dict()
        for c in candidates:
            c["weight"] = round(float(w_dict.get(c["ticker"], 1.0 / len(candidates))), 6)

        # 保存权重 JSON
        out_dir = _ROOT / "reports" / "daily" / str(as_of)
        out_dir.mkdir(parents=True, exist_ok=True)
        weight_records = [{"ticker": c["ticker"], "weight": c["weight"]} for c in candidates]
        (out_dir / "position_weights.json").write_text(
            json.dumps({"date": str(as_of), "method": method, "weights": weight_records,
                        "barra_constrained": barra_enabled},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[Step5b] ✅ {method} 仓位优化完成，已保存 position_weights.json")
        return True, candidates
    except Exception as e:
        logger.warning(f"[Step5b] ⚠ 仓位优化失败（{e}），回退等权")
        for c in candidates:
            c.setdefault("weight", round(1.0 / len(candidates), 6))
        return True, candidates


def step6_llm_rerank(
    as_of: date, candidates: list[dict], args
) -> tuple[bool, list[dict]]:
    """运行 LLM Listwise Reranker，精排到 Top-K."""
    if args.no_llm:
        logger.info("[Step6] --no-llm 已设置，跳过 LLM 重排，使用 LGBM 排名前 N")
        top = candidates[: args.llm_top]
        for i, c in enumerate(top, start=1):
            c["llm_rank"] = i
            c["rank"] = i
            c["reason"] = "（跳过 LLM，仅 LGBM 排名）"
        return True, top

    from quantmind.models.llm_reranker import LLMListwiseReranker, RerankCandidate
    import numpy as np

    logger.info(f"[Step6] 调用 LLM Reranker ({args.provider}/{args.model})，"
                f"候选数={len(candidates)}，目标 Top-{args.llm_top}")
    try:
        rc_list = [
            RerankCandidate(
                ticker=c["ticker"],
                lgbm_score=c["lgbm_score"],
                lgbm_rank=c["lgbm_rank"],
                pe_ttm=c["key_factors"].get("pe_ttm", float("nan")),
                pb=c["key_factors"].get("pb", float("nan")),
                roe_ttm=c["key_factors"].get("roe_ttm", float("nan")),
                accruals=c["key_factors"].get("accruals", float("nan")),
                distance_to_52w_high=c["key_factors"].get("distance_to_52w_high", float("nan")),
                momentum_6m=c["key_factors"].get("momentum_6m", float("nan")),
                volatility_3m=c["key_factors"].get("volatility_3m", float("nan")),
            )
            for c in candidates
        ]
        reranker = LLMListwiseReranker(
            provider=args.provider,
            model=args.model,
            batch_size=min(50, len(rc_list)),
        )
        results = reranker.rerank(rc_list, top_n=args.llm_top, as_of_date=str(as_of))

        top10 = []
        for r in results:
            orig = next((c for c in candidates if c["ticker"] == r.ticker), {})
            top10.append({
                "rank": r.rank,
                "ticker": r.ticker,
                "lgbm_score": orig.get("lgbm_score", float("nan")),
                "lgbm_rank": r.lgbm_rank,
                "llm_rank": r.rank,
                "reason": r.reason or "",
                "key_factors": orig.get("key_factors", {}),
                "is_fallback": r.is_fallback,
            })

        logger.info(f"[Step6] ✅ LLM 重排完成，Top-{len(top10)} 已生成"
                    f"{'（降级模式）' if any(r.is_fallback for r in results) else ''}")
        return True, top10
    except Exception as e:
        logger.error(f"[Step6] ❌ LLM 重排失败：{e}，降级为 LGBM 排名")
        fallback = candidates[: args.llm_top]
        for i, c in enumerate(fallback, 1):
            c.update({"llm_rank": i, "rank": i, "reason": f"LLM失败，使用LGBM排名（{e}）"})
        return False, fallback


def step7_save_json(
    as_of: date,
    top10: list[dict],
    runtime: float,
    step_results: dict[str, bool],
) -> tuple[bool, Path | None]:
    """保存推荐结果 JSON."""
    out_dir = _ROOT / "data" / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{as_of.isoformat()}.json"

    payload: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "top10": top10,
        "market_summary": _generate_market_summary(top10),
        "runtime_seconds": round(runtime, 1),
        "step_status": step_results,
    }
    try:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[Step7] ✅ 推荐 JSON 已保存：{out_path}")
        nested_dir = out_dir / as_of.isoformat()
        nested_dir.mkdir(parents=True, exist_ok=True)
        top10_only = nested_dir / "top10.json"
        top10_only.write_text(
            json.dumps(
                {"as_of": as_of.isoformat(), "top10": top10},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"[Step7] ✅ Top10 JSON：{top10_only}")
        return True, out_path
    except Exception as e:
        logger.error(f"[Step7] ❌ JSON 保存失败：{e}")
        return False, None


def _generate_market_summary(top10: list[dict]) -> str:
    """根据 Top-10 生成简短市场摘要."""
    if not top10:
        return "暂无推荐数据"
    # 统计平均 PE、ROE
    pes = [c["key_factors"].get("pe_ttm") for c in top10 if c.get("key_factors", {}).get("pe_ttm")]
    roes = [c["key_factors"].get("roe_ttm") for c in top10 if c.get("key_factors", {}).get("roe_ttm")]
    pe_str = f"平均 PE {sum(pes)/len(pes):.1f}x" if pes else ""
    roe_str = f"平均 ROE {sum(roes)/len(roes)*100:.1f}%" if roes else ""
    parts = [s for s in [pe_str, roe_str] if s]
    summary = f"今日推荐 {len(top10)} 只股票"
    if parts:
        summary += f"，{', '.join(parts)}"
    return summary


def step7a_agent_analysis(
    as_of: date,
    top10: list[dict],
    args,
) -> bool:
    """Step 7a: 对 Top-N 候选运行 6-Agent 投资分析.

    产出：
      reports/investment_pipeline/<date>/strategies.json
      reports/investment_pipeline/<date>/strategies/<ticker>_strategy.json
      reports/investment_pipeline/<date>/final_recommendations.md
      reports/investment_pipeline/<date>/validations.json

    失败时不影响后续 step8，仅记录 warning。
    """
    from scripts.run_investment_pipeline import (
        _load_price_df,
        generate_final_report,
        run_six_agents,
        save_strategy,
        _strategy_to_dict,
    )
    from scripts.validate_strategies import batch_validate
    from quantmind.agents.investment_agents.strategy_agent import (
        InvestmentStrategy,
        StrategyAgent,
    )

    date_str = as_of.isoformat()
    n = min(args.agent_top, len(top10))
    tickers = [c["ticker"] for c in top10[:n]]
    out_dir = _ROOT / "reports" / "investment_pipeline" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = out_dir / "strategies"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[Step7a] 开始 6-Agent 分析，共 {len(tickers)} 只股票"
        f"（provider={args.agent_provider}）"
    )

    # 加载价格宽表（供 validate_strategies 用）
    try:
        price_df = _load_price_df()
    except Exception as e:
        logger.warning(f"[Step7a] 价格面板加载失败（{e}），跳过回测验证")
        price_df = None

    all_strategies: list[InvestmentStrategy] = []
    errors: list[str] = []

    for i, ticker in enumerate(tickers, start=1):
        logger.info(f"[Step7a] [{i}/{len(tickers)}] 分析 {ticker} ...")
        t0 = time.monotonic()
        try:
            context: dict = {
                "ticker": ticker,
                "as_of": date_str,
                "news": [],
                "reports": [],
                "snapshot": {},
            }

            signals = run_six_agents(
                ticker=ticker,
                as_of=date_str,
                context=context,
                provider=args.agent_provider,
                model=args.agent_model,
            )

            strategy_agent = StrategyAgent(
                ticker=ticker,
                as_of=date_str,
                context=context,
                agent_signals=signals,
                provider=args.agent_provider,
                model=args.agent_model,
            )
            strategy = strategy_agent.analyze_with_llm()
            all_strategies.append(strategy)
            save_strategy(strategy, ticker_dir)

            elapsed = time.monotonic() - t0
            logger.info(
                f"[Step7a]   {ticker} ✅ rating={strategy.rating} "
                f"signal={strategy.composite_signal:+.2f} ({elapsed:.1f}s)"
            )
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            logger.warning(f"[Step7a]   {ticker} ⚠️ 分析失败，跳过: {e}")

    if not all_strategies:
        logger.error("[Step7a] ❌ 全部 Agent 分析失败，无输出")
        return False

    # 保存汇总 strategies.json
    strategies_data = [_strategy_to_dict(s) for s in all_strategies]
    strategies_path = out_dir / "strategies.json"
    strategies_path.write_text(
        json.dumps(strategies_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[Step7a] ✅ strategies.json 已写入：{strategies_path}")

    # 回测验证（可选）
    acceptable, watchlist, avoid = [], [], []
    try:
        if price_df is not None and not price_df.empty:
            panel_path = _ROOT / "data" / "panel" / "alpha_panel_v3.parquet"
            panel_df = None
            if panel_path.exists():
                import pandas as _pd
                panel_df = _pd.read_parquet(panel_path)
            results = batch_validate(strategies_data, price_df, panel_df)
            for r in results:
                status = getattr(r, "validation_status", "") or ""
                if status == "ACCEPTABLE":
                    acceptable.append(r)
                elif status == "WATCHLIST":
                    watchlist.append(r)
                else:
                    avoid.append(r)
            # 保存 validations.json
            val_path = out_dir / "validations.json"
            val_path.write_text(
                json.dumps(
                    [
                        {
                            "ticker": getattr(r, "ticker", ""),
                            "validation_status": getattr(r, "validation_status", ""),
                            "action": getattr(r, "action", ""),
                            "stop_loss_triggered": getattr(r, "stop_loss_triggered", False),
                            "win_rate": getattr(r, "win_rate", 0.0),
                            "summary": getattr(r, "summary", ""),
                        }
                        for r in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(f"[Step7a] ✅ validations.json 已写入")
    except Exception as e:
        logger.warning(f"[Step7a] ⚠️ 回测验证异常，跳过: {e}")

    # 生成 final_recommendations.md
    try:
        md_path = generate_final_report(
            acceptable=acceptable,
            watchlist=watchlist,
            avoid=avoid,
            all_strategies=all_strategies,
            output_dir=out_dir,
            date_str=date_str,
        )
        logger.info(f"[Step7a] ✅ final_recommendations.md 已写入：{md_path}")
    except Exception as e:
        logger.warning(f"[Step7a] ⚠️ 报告生成异常，跳过: {e}")

    ok_count = len(all_strategies)
    fail_count = len(errors)
    logger.info(
        f"[Step7a] 完成：{ok_count} 只成功，{fail_count} 只失败"
        + (f"（失败：{errors[:3]}）" if errors else "")
    )
    return ok_count > 0


def step8_generate_html(
    as_of: date, top10: list[dict], json_path: Path | None, runtime: float
) -> tuple[bool, Path | None]:
    """生成今日推荐 HTML 报告."""
    out_dir = _ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_{as_of.isoformat()}.html"

    try:
        html = _build_html_report(as_of, top10, runtime, json_path)
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"[Step8] ✅ HTML 报告已生成：{out_path}")
        return True, out_path
    except Exception as e:
        logger.error(f"[Step8] ❌ HTML 生成失败：{e}")
        return False, None


def _build_html_report(
    as_of: date,
    top10: list[dict],
    runtime: float,
    json_path: Path | None,
) -> str:
    """构建推荐报告 HTML."""
    rows_html = ""
    for item in top10:
        kf = item.get("key_factors", {})
        pe = f"{kf['pe_ttm']:.1f}x" if kf.get("pe_ttm") else "N/A"
        pb = f"{kf['pb']:.2f}x" if kf.get("pb") else "N/A"
        roe = f"{kf['roe_ttm']*100:.1f}%" if kf.get("roe_ttm") else "N/A"
        lgbm_rank = item.get("lgbm_rank", "-")
        llm_rank = item.get("llm_rank", item.get("rank", "-"))
        diff = (item.get("lgbm_rank", 0) - item.get("rank", 0))
        arrow = (f"<span style='color:#27ae60'>↑{diff}</span>" if diff > 0
                 else (f"<span style='color:#e74c3c'>↓{abs(diff)}</span>" if diff < 0 else "→"))
        reason = item.get("reason", "")[:80]
        rows_html += f"""
        <tr>
          <td style='text-align:center;font-weight:bold;color:#8e44ad'>{llm_rank}</td>
          <td style='font-family:monospace'>{item['ticker']}</td>
          <td style='text-align:center'>{lgbm_rank}</td>
          <td style='text-align:center'>{arrow}</td>
          <td style='text-align:center'>{pe}</td>
          <td style='text-align:center'>{pb}</td>
          <td style='text-align:center'>{roe}</td>
          <td style='color:#555;font-size:0.88em'>{reason}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>QuantMind 每日推荐 {as_of}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;
        margin:0;padding:24px;background:#f8f9fa;color:#333}}
  h1{{color:#1a252f;border-bottom:4px solid #8e44ad;padding-bottom:10px}}
  .meta{{background:#fff;border-radius:8px;padding:14px 20px;
         box-shadow:0 2px 8px rgba(0,0,0,.08);margin:16px 0;
         border-left:5px solid #8e44ad}}
  table{{width:100%;border-collapse:collapse;background:#fff;
         border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
  th{{background:#8e44ad;color:#fff;padding:10px 12px;text-align:center}}
  td{{padding:10px 12px;border-bottom:1px solid #eee}}
  tr:hover{{background:#f9f4ff}}
  .footer{{margin-top:20px;color:#888;font-size:0.85em}}
</style>
</head>
<body>
<h1>📈 QuantMind 每日推荐 — {as_of}</h1>
<div class="meta">
  <strong>生成时间：</strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
  <strong>运行耗时：</strong>{runtime:.1f}s &nbsp;|&nbsp;
  <strong>推荐数量：</strong>{len(top10)} 只
</div>
<table>
  <tr>
    <th>LLM 排名</th><th>股票代码</th><th>LGBM 排名</th><th>变化</th>
    <th>PE(TTM)</th><th>PB</th><th>ROE(TTM)</th><th>推荐理由</th>
  </tr>
  {rows_html}
</table>
<p class="footer">
  数据截止日：{as_of} &nbsp;|&nbsp;
  模型：LightGBM LambdaRank + LLM Listwise Rerank &nbsp;|&nbsp;
  本报告仅供研究参考，不构成投资建议
</p>
</body>
</html>"""


def step9_rag_reports(
    as_of: date, top_candidates: list[dict], args: argparse.Namespace
) -> bool:
    """对 Top-N 股票生成 RAG 投资分析报告（可选；失败不抬高进程退出码）."""
    from argparse import Namespace

    from quantmind.agents.rag_report_agent import RAGReportAgent

    from scripts.generate_investment_report import retrieve_context

    if args.no_llm:
        logger.info("[Step9] --no-llm 已设置，跳过 RAG 报告生成")
        return True

    if not top_candidates:
        logger.info("[Step9] 无推荐股票，跳过 RAG 报告")
        return True

    reports_dir = _ROOT / "data" / "recommendations" / as_of.isoformat() / "rag_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    agent = RAGReportAgent()
    batch = top_candidates[: args.llm_top]
    success = 0

    for cand in batch:
        ticker = cand.get("ticker") or cand.get("ts_code") or ""
        ticker = str(ticker).strip()
        if not ticker:
            continue
        ctx_ns = Namespace(
            ticker=ticker,
            as_of=as_of.isoformat(),
            top_k_news=5,
            top_k_reports=3,
            top_k_per_doc_type=3,
            collection_name=args.collection_name,
            chroma_dir=str(Path(args.chroma_dir).expanduser()),
            json_output=None,
            output=None,
        )
        try:
            context = retrieve_context(
                ctx_ns,
                silent=True,
                on_empty_kb="empty",
            )
            result = agent.generate_report(
                context=context,
                use_llm=True,
                llm_provider=args.provider,
                llm_model=args.model,
            )
            report_path = reports_dir / f"report_{ticker}.md"
            report_path.write_text(result["report_markdown"], encoding="utf-8")
            success += 1
            logger.info(
                f"[Step9] ✅ {ticker} 报告生成完成（llm_used={result.get('llm_used')}）"
            )
        except Exception as e:
            logger.warning(f"[Step9] ⚠️ {ticker} 报告失败，跳过: {e}")

    denom = len(batch)
    logger.info(f"[Step9] 完成: {success}/{denom} 只股票")
    return success > 0


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    from quantmind.core.logger import setup_logger

    args = parse_args()
    t_start = time.monotonic()

    # 配置日志（同时写文件）
    setup_logger()
    log_dir = _ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _date_str = args.date or date.today().isoformat()
    log_file = log_dir / f"daily_update_{_date_str}.log"
    logger.add(str(log_file), rotation="10 MB", retention="30 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.info(f"{'='*60}")
    logger.info(f"QuantMind 每日更新流水线 — 目标日期={_date_str}")
    logger.info(f"{'='*60}")

    # ── 冠军模型完整性校验（优先于一切业务逻辑）────────────────────────────
    try:
        _sanity_check_champion_model()
    except RuntimeError as _e:
        logger.error(f"[SanityCheck] 流水线中止：{_e}")
        return 1

    if _today_calendar_skip_non_trading(args):
        return 0

    step_results: dict[str, bool] = {s: False for s in STEPS}
    feat_path: Path | None = None
    candidates: list[dict] = []
    top10: list[dict] = []
    json_path: Path | None = None
    as_of: date = date.today()

    def _done(step: str) -> bool:
        """如果 --stop-after 指定该步则提前退出."""
        return args.stop_after == step

    # ── Step 1 ────────────────────────────────────────────────────────────────
    try:
        as_of = step1_resolve_date(args)
        step_results["step1"] = True
    except Exception as e:
        logger.error(f"[Step1] ❌ {e}")
    if _done("step1"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    try:
        cached = step2_check_cache(as_of, args.force)
        step_results["step2"] = True
    except Exception as e:
        logger.error(f"[Step2] ❌ {e}")
        cached = False
    if _done("step2"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    if not cached:
        ok = False
        try:
            ok = step3_download_snapshot(as_of, args)
        except Exception as e:
            logger.error(f"[Step3] ❌ 未捕获异常：{e}")
        if not ok and _snapshot_has_usable_cache(as_of):
            logger.warning(
                "[Step3] 快照下载失败，检测到本地 prices.parquet，降级继续（视为缓存可用）"
            )
            ok = True
        step_results["step3"] = ok
    else:
        step_results["step3"] = True  # 缓存命中视为成功
        logger.info("[Step3] 快照已存在，跳过下载")
    if _done("step3"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    try:
        ok, feat_path = step4_build_features(as_of, args)
        step_results["step4"] = ok
    except Exception as e:
        logger.error(f"[Step4] ❌ 未捕获异常：{e}")
    if _done("step4"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 5 ────────────────────────────────────────────────────────────────
    try:
        ok, candidates = step5_lgbm_rank(as_of, feat_path, args)
        step_results["step5"] = ok
    except Exception as e:
        logger.error(f"[Step5] ❌ 未捕获异常：{e}")
    if _done("step5"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 5c: FactorCNN ensemble 融合（rank-based 6:4）────────────────────
    if not getattr(args, "no_cnn", False):
        try:
            candidates = step5c_cnn_ensemble(as_of, feat_path, candidates)
        except Exception as e:
            logger.warning(f"[Step5c] ⚠ CNN ensemble 异常（{e}），维持 LGBM 顺序")

    # ── Step 5b ───────────────────────────────────────────────────────────────
    try:
        ok, candidates = step5b_position_sizing(as_of, candidates, args)
        step_results["step5b"] = ok
    except Exception as e:
        logger.warning(f"[Step5b] ⚠ 仓位优化异常（{e}），继续")
        step_results["step5b"] = False
    if _done("step5b"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 6 ────────────────────────────────────────────────────────────────
    try:
        ok, top10 = step6_llm_rerank(as_of, candidates, args)
        step_results["step6"] = ok
    except Exception as e:
        logger.error(f"[Step6] ❌ 未捕获异常：{e}")
        top10 = candidates[: args.llm_top]
    if _done("step6"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 7 ────────────────────────────────────────────────────────────────
    runtime_so_far = time.monotonic() - t_start
    try:
        ok, json_path = step7_save_json(as_of, top10, runtime_so_far, step_results)
        step_results["step7"] = ok
    except Exception as e:
        logger.error(f"[Step7] ❌ 未捕获异常：{e}")
    if _done("step7"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 7a（6-Agent 分析，可选；失败不影响后续）────────────────────────
    if not getattr(args, "no_agent", False):
        try:
            ok_7a = step7a_agent_analysis(as_of, top10, args)
            step_results["step7a"] = ok_7a
        except Exception as e:
            logger.warning(f"[Step7a] ⚠️ 未捕获异常，跳过: {e}")
            step_results["step7a"] = False
    else:
        logger.info("[Step7a] --no-agent 已设置，跳过 6-Agent 分析")
        step_results["step7a"] = True

    if _done("step7a"):
        return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)

    # ── Step 8 ────────────────────────────────────────────────────────────────
    try:
        ok, _ = step8_generate_html(as_of, top10, json_path, runtime_so_far)
        step_results["step8"] = ok
    except Exception as e:
        logger.error(f"[Step8] ❌ 未捕获异常：{e}")

    # ── Step 9（RAG 投资报告，可选；不影响退出码）────────────────────────────
    if args.stop_after is None:
        try:
            step_results["step9"] = step9_rag_reports(as_of, top10, args)
        except Exception as e:
            logger.warning(f"[Step9] ⚠️ 批量 RAG 异常，跳过: {e}")
            step_results["step9"] = False

    # ── Step 10（漏斗统计，可选；仅 universe=full_a 时触发）──────────────────
    if args.stop_after is None and getattr(args, "universe", "csi300") == "full_a":
        try:
            _step10_funnel_stats(as_of, args)
            step_results["step10"] = True
        except Exception as e:
            logger.warning(f"[Step10] ⚠️ 漏斗统计异常，跳过: {e}")
            step_results["step10"] = False

    # ── Step 11（每日预警推送：企业微信 + 邮件，可选；不影响退出码）─────────
    if args.stop_after is None and not getattr(args, "no_alert", False):
        try:
            _step11_daily_alert(as_of, top10)
            step_results["step11"] = True
        except Exception as e:
            logger.warning(f"[Step11] ⚠️ 预警推送异常，跳过: {e}")
            step_results["step11"] = False

    return _finalize(step_results, t_start, log_file, stop_after=args.stop_after)


def _step11_daily_alert(as_of, top10: list[dict]) -> None:
    """Step 11: 每日预警推送（企业微信 Webhook + SMTP 邮件）."""
    from quantmind.notify.daily_alert import DailyAlertSender
    from quantmind.features.north_flow import get_latest_market_north_flow

    logger.info("[Step11] 准备每日预警推送...")

    # 构建 picks DataFrame
    picks_rows = []
    for item in top10:
        picks_rows.append({
            "ticker":          item.get("ticker", ""),
            "name":            item.get("name", ""),
            "composite_score": item.get("composite_score", float("nan")),
            "rating":          item.get("rating", ""),
            "industry":        item.get("industry", ""),
        })
    picks_df = pd.DataFrame(picks_rows) if picks_rows else pd.DataFrame()

    # 获取 Regime（HMM 或 neutral 兜底）
    try:
        from quantmind.regime import RegimeHMM, build_observations
        import pathlib
        obs_path = str(pathlib.Path(_ROOT) / "data" / "raw" / "index_daily_panel.parquet")
        hmm  = RegimeHMM()
        obs  = build_observations(obs_path)
        regime = hmm.predict_regime(obs, pd.Timestamp(as_of)) if obs is not None else "neutral"
    except Exception as exc:
        logger.warning(f"[Step11] HMM Regime 获取失败，使用 neutral: {exc}")
        regime = "neutral"

    # 北向资金辅助信号
    try:
        market_flow = get_latest_market_north_flow()
    except Exception:
        market_flow = 0.0

    sender = DailyAlertSender()
    ok = sender.send_daily_picks(picks_df, regime=regime, market_flow=market_flow)
    if ok:
        logger.info("[Step11] ✅ 每日预警推送完成")
    else:
        logger.warning("[Step11] ⚠️ 所有推送通道均失败（可能未配置 WECOM_WEBHOOK_URL / SMTP_*）")


def _step10_funnel_stats(as_of, args) -> None:
    """Step 10: 全A股漏斗统计（universe=full_a 时触发）."""
    logger.info("[Step10] 运行全A股漏斗统计...")
    from quantmind.selection.funnel_selector import FunnelSelector

    skip_layers = getattr(args, "funnel_skip_layers", [])
    selector = FunnelSelector(
        as_of=as_of.isoformat(),
        universe="full_a",
        provider=getattr(args, "provider", "none"),
    )
    candidates, stats = selector.run(skip_layers=skip_layers, top_n=args.lgbm_top)
    out = selector.to_output_json(candidates, stats)

    import json
    from pathlib import Path
    from quantmind.core.config import get_settings
    out_path = (
        Path(get_settings().data.dir).resolve()
        / "recommendations"
        / as_of.isoformat()
        / "funnel_candidates.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"[Step10] ✅ 漏斗统计完成: Layer1={stats.layer1_in}→Layer6={stats.layer6_out}，"
        f"输出={out_path}"
    )


def _finalize(
    step_results: dict[str, bool],
    t_start: float,
    log_file: Path,
    *,
    stop_after: str | None = None,
) -> int:
    """汇报各步骤结果并返回退出码."""
    total = time.monotonic() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"{'QuantMind 每日更新流水线 — 汇报':^58}")
    logger.info(f"{'='*60}")
    limit_idx = STEPS.index(stop_after) if stop_after in STEPS else len(STEPS) - 1
    relevant = STEPS[: limit_idx + 1]
    all_ok = all(step_results.get(s, False) for s in relevant)

    for i, step in enumerate(STEPS):
        ok = step_results.get(step, False)
        name = _STEP_NAMES.get(step, step)
        if i > limit_idx:
            logger.info(f"  ⏭  {step}  {name}（--stop-after，跳过）")
            continue
        icon = "✅" if ok else "❌"
        logger.info(f"  {icon}  {step}  {name}")

    if stop_after is None and "step9" in step_results:
        s9 = step_results["step9"]
        icon = "✅" if s9 else "⚠️"
        logger.info(f"  {icon}  step9  RAG 投资报告（可选，不影响退出码）")
    if stop_after is None and "step10" in step_results:
        s10 = step_results["step10"]
        icon = "✅" if s10 else "⚠️"
        logger.info(f"  {icon}  step10 漏斗统计（universe=full_a，不影响退出码）")
    if stop_after is None and "step11" in step_results:
        s11 = step_results["step11"]
        icon = "✅" if s11 else "⚠️"
        logger.info(f"  {icon}  step11 每日预警推送（可选，不影响退出码）")

    logger.info(f"{'─'*60}")
    logger.info(f"  总耗时：{total:.1f}s  |  日志：{log_file}")
    logger.info(f"{'='*60}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
