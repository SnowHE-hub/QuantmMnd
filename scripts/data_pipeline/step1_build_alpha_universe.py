#!/usr/bin/env python3
"""Step 1: 构建阿尔法核心池（Alpha Universe）~1000 只股票.

分层抽样策略：
  CSI300 (300) + CSI500 (500) = CSI800 基础池
  行业补位 (~100): 确保申万31个行业各有代表
  高波动样本 (~50): 极端行情训练样本
  合计目标：900~1000 只

用途：三大子系统（选股漏斗、6 Agent 研究、LightGBM 因子模型）的数据基础。

运行：python scripts/data_pipeline/step1_build_alpha_universe.py
输出：data/alpha_universe/alpha_universe.parquet
       data/alpha_universe/alpha_universe.txt  (每行一个 ts_code)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ── 复用既有 .env 机制读取凭证 + 屏蔽 provider 日志（防 token 外泄）──────────────
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass
try:
    from quantmind.utils.silence_provider_logging import silence_provider_logging
    silence_provider_logging()
except Exception:
    pass
OUT_DIR = ROOT / "data" / "alpha_universe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── API 初始化（使用高频 API，15000 积分）────────────────────────────────────
import tushare as ts

def _init_pro(high_rate: bool = False):
    """初始化 Tushare。
    Step1 只用官方 API（index_weight/stock_basic 需要官方地址）。
    high_rate=True 仅供价格下载类脚本使用。
    """
    if high_rate:
        token = os.environ.get("TUSHARE_TOKEN_HI", "").strip()
        if not token:
            raise SystemExit("请先设置 TUSHARE_TOKEN_HI 环境变量")
        ts.set_token(token)
        pro = ts.pro_api(timeout=120)
        endpoint = os.environ.get("TUSHARE_HI_URL", "http://tsy.xiaodefa.cn")
        pro._DataApi__http_url = endpoint
        print(f"[API] 高频模式 → {endpoint}")
    else:
        # Step1 固定用自己的 2000 积分官方 token
        token = os.environ.get(
            "TUSHARE_TOKEN",
            "",
        ).strip()
        ts.set_token(token)
        pro = ts.pro_api(timeout=120)
        print("[API] 官方模式（2000积分，index_weight/stock_basic）")
    return pro


def _retry(fn, label: str, attempts: int = 4, pause: float = 1.5):
    for i in range(attempts):
        try:
            result = fn()
            if result is not None and not result.empty:
                return result
        except Exception as e:
            print(f"  [{label}] 第{i+1}次失败: {e}")
            time.sleep(pause * (i + 1))
    return pd.DataFrame()


def get_index_components(pro, index_code: str) -> set[str]:
    """获取指数最新成分股。"""
    df = _retry(
        lambda: pro.index_weight(
            index_code=index_code,
            start_date="20241201",
            end_date="20241231",
        ),
        label=f"index_weight {index_code}",
    )
    if df.empty:
        # fallback: 用最新一期成分
        df = _retry(
            lambda: pro.index_weight(index_code=index_code),
            label=f"index_weight fallback {index_code}",
        )
    return set(df["con_code"].dropna().tolist()) if not df.empty else set()


def get_all_stocks(pro) -> pd.DataFrame:
    """获取全 A 股基础信息（用于行业补位）。"""
    df = _retry(
        lambda: pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name,industry,market,list_date",
        ),
        label="stock_basic",
    )
    if df.empty:
        print("  [WARNING] 无法获取 stock_basic")
        return pd.DataFrame()
    # 过滤：去除北交所（8开头）、ST
    df = df[~df["ts_code"].str.startswith("8")]
    return df


def main():
    pro = _init_pro(high_rate=False)
    print("[Step1] 构建 Alpha Universe...")

    # ── 1. CSI800 = CSI300 + CSI500 ─────────────────────────────────────────
    print("  拉取 CSI300 成分...")
    csi300 = get_index_components(pro, "000300.SH")
    time.sleep(0.5)

    print("  拉取 CSI500 成分...")
    csi500 = get_index_components(pro, "000905.SH")
    time.sleep(0.5)

    csi800 = csi300 | csi500
    print(f"  CSI800: {len(csi300)} + {len(csi500)} = {len(csi800)} 只")

    # ── 2. 全市场股票（行业补位 + 高波动样本）────────────────────────────────
    all_df = get_all_stocks(pro)
    time.sleep(0.5)

    universe: set[str] = set(csi800)

    if not all_df.empty:
        # 2a. 行业补位：申万一级行业，每行业各取 5 只（按名称排序，稳定可重复）
        print("  行业补位...")
        industry_supplement: set[str] = set()
        for ind, grp in all_df.groupby("industry"):
            not_in_csi = grp[~grp["ts_code"].isin(csi800)]["ts_code"].tolist()
            # 排序保证每次结果一致
            industry_supplement.update(sorted(not_in_csi)[:5])
        universe.update(industry_supplement)
        print(f"  行业补位后: {len(universe)} 只（+{len(industry_supplement)} 只）")

        # 2b. 高波动样本：从非 CSI800 中取 50 只（科创板 688、创业板 300 优先）
        star = all_df[
            (~all_df["ts_code"].isin(universe))
            & all_df["ts_code"].str.startswith("688")
        ]["ts_code"].tolist()
        gem = all_df[
            (~all_df["ts_code"].isin(universe))
            & all_df["ts_code"].str.startswith("300")
        ]["ts_code"].tolist()
        volatile_pool = sorted(star)[:25] + sorted(gem)[:25]
        universe.update(volatile_pool[:50])
        print(f"  高波动补入后: {len(universe)} 只（科创{min(25,len(star))}+创业板{min(25,len(gem))}）")

    # ── 3. 保存结果 ──────────────────────────────────────────────────────────
    df_out = all_df[all_df["ts_code"].isin(universe)].copy() if not all_df.empty else pd.DataFrame({"ts_code": sorted(universe)})
    if "ts_code" not in df_out.columns:
        df_out = pd.DataFrame({"ts_code": sorted(universe)})

    # 补充 CSI300/CSI500 标记
    df_out["in_csi300"] = df_out["ts_code"].isin(csi300)
    df_out["in_csi500"] = df_out["ts_code"].isin(csi500)
    df_out["in_csi800"] = df_out["ts_code"].isin(csi800)
    df_out = df_out.drop_duplicates("ts_code").sort_values("ts_code").reset_index(drop=True)

    out_pq = OUT_DIR / "alpha_universe.parquet"
    out_txt = OUT_DIR / "alpha_universe.txt"
    df_out.to_parquet(out_pq, index=False)
    out_txt.write_text("\n".join(df_out["ts_code"].tolist()), encoding="utf-8")

    print(f"\n[Done] Alpha Universe: {len(df_out)} 只")
    print(f"  CSI300: {df_out['in_csi300'].sum()} | CSI500: {df_out['in_csi500'].sum()}")
    if "industry" in df_out.columns:
        print(f"  行业数: {df_out['industry'].nunique()}")
    print(f"  输出: {out_pq}")
    print(f"  列表: {out_txt}")


if __name__ == "__main__":
    main()
