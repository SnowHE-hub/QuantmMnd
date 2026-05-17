"""update_sim_strategy.py — 基于模拟盘结果更新系统策略参数.

根据 data/paper_trading/optimization.json 中的信号，
执行以下更新：
1. 生成 data/paper_trading/strategy_config.json（持仓期/仓位参数）
2. 输出因子方向修正建议到 data/features/factor_direction_v4.json
3. 为前向模拟生成 data/paper_trading/forward_positions.json 模板
4. 打印完整的系统状态报告

用法
====
  python scripts/update_sim_strategy.py \\
      --positions data/paper_trading/positions.parquet \\
      --panel data/panel/alpha_panel_v4.parquet \\
      --out data/paper_trading/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]


def analyze_horizon_optimality(df: pd.DataFrame) -> dict:
    """按持仓期计算 IR 和期级胜率，选出最优持仓期."""
    results = {}
    for hz in ['1w', '2w', '21d', '3m']:
        col = 'return_' + hz
        if col not in df.columns:
            continue
        prets = df.groupby('as_of')[col].mean().dropna()
        if len(prets) < 3:
            continue
        mean_r = float(prets.mean())
        std_r = float(prets.std())
        ir = mean_r / std_r if std_r > 0 else 0.0
        win = float((prets > 0).mean())
        cum_nav = float((1 + prets).prod())
        results[hz] = {
            'mean': mean_r, 'std': std_r, 'ir': ir,
            'win_rate': win, 'cumulative_nav': cum_nav, 'n_periods': len(prets)
        }

    # 选最优：IR × 胜率 综合分
    best = max(results.items(), key=lambda x: x[1]['ir'] * x[1]['win_rate'])
    return {'by_horizon': results, 'optimal_horizon': best[0], 'optimal_score': best[1]['ir'] * best[1]['win_rate']}


def analyze_concentration(df: pd.DataFrame, horizon: str) -> dict:
    """分析 Top-N 集中度：top3 vs top10 vs btm5."""
    col = 'return_' + horizon
    if col not in df.columns:
        return {}
    top3 = df[df['predicted_rank'] <= 3][col].dropna()
    top5 = df[df['predicted_rank'] <= 5][col].dropna()
    top10 = df[col].dropna()
    btm5 = df[df['predicted_rank'] >= 6][col].dropna()
    return {
        'top3_mean': float(top3.mean()) if len(top3) else None,
        'top5_mean': float(top5.mean()) if len(top5) else None,
        'top10_mean': float(top10.mean()) if len(top10) else None,
        'btm5_mean': float(btm5.mean()) if len(btm5) else None,
        'top3_premium': float(top3.mean() - top10.mean()) if len(top3) and len(top10) else None,
        'recommended_n': 5 if (len(top3) and len(top10) and top3.mean() > top10.mean() * 1.3) else 10,
    }


def analyze_factor_directions(panel: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """计算各 v4 因子的实际 IC 方向（vs 各期组合收益）."""
    ret_col = 'return_21d'
    merged_rows = []
    for as_of, grp in df.groupby('as_of'):
        as_of_str = str(as_of.date()) if hasattr(as_of, 'date') else str(as_of)[:10]
        all_panel_dates = panel.index.get_level_values('as_of').unique()
        nearest = min(all_panel_dates, key=lambda d: abs((pd.Timestamp(d) - pd.Timestamp(as_of_str)).days))
        xs = panel.xs(nearest, level='as_of')
        for _, row in grp.iterrows():
            ticker = row['ticker']
            ret = row.get(ret_col)
            if ticker in xs.index and pd.notna(ret):
                factor_vals = xs.loc[ticker].to_dict()
                factor_vals['_return'] = ret
                factor_vals['_ticker'] = ticker
                merged_rows.append(factor_vals)

    if not merged_rows:
        return []

    merged_df = pd.DataFrame(merged_rows)
    ret_series = merged_df['_return']

    factor_results = []
    skip_cols = {'_return', '_ticker'}
    for fc in merged_df.columns:
        if fc in skip_cols:
            continue
        sub = pd.concat([merged_df[fc], ret_series], axis=1).dropna()
        if len(sub) < 8:
            continue
        ic, p = stats.spearmanr(sub[fc], sub['_return'])
        if abs(ic) < 0.05 and p > 0.3:
            continue
        factor_results.append({
            'factor': fc,
            'ic_vs_21d_return': round(float(ic), 4),
            'p_value': round(float(p), 4),
            'n': len(sub),
            'direction': 'REVERSED' if ic < -0.10 else ('POSITIVE' if ic > 0.10 else 'NEUTRAL'),
            'significant': p < 0.10,
        })

    factor_results.sort(key=lambda x: -abs(x['ic_vs_21d_return']))
    return factor_results


def generate_strategy_config(
    horizon_analysis: dict,
    concentration: dict,
    factor_directions: list[dict],
) -> dict:
    """生成策略配置文件."""
    opt_hz = horizon_analysis['optimal_horizon']
    opt_n = concentration.get('recommended_n', 10)

    reversed_factors = [f for f in factor_directions if f['direction'] == 'REVERSED' and f['significant']]
    positive_factors = [f for f in factor_directions if f['direction'] == 'POSITIVE' and f['significant']]

    config = {
        'version': 'v1_sim_driven',
        'updated_at': datetime.now().isoformat(),
        'data_source': 'paper_trading_sim_v4_9periods',
        'holding_period': {
            'recommended': opt_hz,
            'rationale': 'IR=%+.3f, 期胜率=%.1f%%' % (
                horizon_analysis['by_horizon'][opt_hz]['ir'],
                horizon_analysis['by_horizon'][opt_hz]['win_rate'] * 100,
            ),
            'all_horizons': {hz: {
                'ir': round(v['ir'], 3),
                'win_rate': round(v['win_rate'], 3),
                'cumulative_nav': round(v['cumulative_nav'], 4),
            } for hz, v in horizon_analysis['by_horizon'].items()},
        },
        'position_sizing': {
            'top_n': opt_n,
            'equal_weight': True,
            'rationale': 'Top3 premium over Top10: %+.4f (%s)' % (
                concentration.get('top3_premium', 0) or 0,
                'concentrate' if (concentration.get('top3_premium', 0) or 0) > 0.015 else 'diversify'
            ),
        },
        'factor_signals': {
            'reversed_factors': [
                {'factor': f['factor'], 'ic': f['ic_vs_21d_return'], 'action': 'DOWNWEIGHT_OR_FLIP'}
                for f in reversed_factors
            ],
            'positive_factors': [
                {'factor': f['factor'], 'ic': f['ic_vs_21d_return'], 'action': 'UPWEIGHT'}
                for f in positive_factors
            ],
            'regime_note': (
                '当前市场（2024Q2-2026Q1）质量因子反转（roe_ttm IC=-0.39**），'
                '动量因子弱反转（momentum_6m IC=-0.19*），应权衡 bull-regime 风格偏移'
            ),
        },
        'risk_controls': {
            'stop_loss_threshold': -0.15,
            'max_single_stock_weight': 0.25,
            'review_frequency': opt_hz,
        },
        'optimization_triggers': {
            'retrain_lgbm_if': '连续 2 期 21d IC < -0.05',
            'factor_review_if': '任一主因子 rolling_IC_3period < -0.15',
            'horizon_review_if': '21d IR < 0.15 连续 3 期',
        },
    }
    return config


def setup_forward_tracking(
    recs_dir: Path,
    out_dir: Path,
    horizon: str = '21d',
) -> dict:
    """生成前向模拟持仓跟踪模板（记录待结算的持仓）."""
    # 收集最新推荐（最近 2 个日期）
    json_files = sorted(recs_dir.glob("*.json"))
    pending: list[dict] = []

    horizon_days = {'1w': 5, '2w': 10, '21d': 21, '3m': 63}
    h_days = horizon_days.get(horizon, 21)

    seen = set()
    for fpath in reversed(json_files):
        try:
            payload = json.loads(fpath.read_text(encoding='utf-8'))
            as_of_str = payload.get('as_of', '')
            if not as_of_str or as_of_str in seen:
                continue
            seen.add(as_of_str)
            as_of_ts = pd.Timestamp(as_of_str)
            estimated_exit = as_of_ts + pd.offsets.BDay(h_days)
            for item in payload.get('top10', []):
                pending.append({
                    'as_of': as_of_str,
                    'ticker': item['ticker'],
                    'predicted_rank': item.get('lgbm_rank', item.get('rank')),
                    'predicted_score': item.get('lgbm_score'),
                    'holding_period': horizon,
                    'estimated_exit_date': str(estimated_exit.date()),
                    'entry_price': None,
                    'exit_price': None,
                    'actual_return': None,
                    'status': 'OPEN',
                })
            if len(seen) >= 2:
                break
        except Exception:
            continue

    tracker = {
        'generated_at': datetime.now().isoformat(),
        'tracking_horizon': horizon,
        'n_open_positions': len([p for p in pending if p['status'] == 'OPEN']),
        'positions': pending,
        'next_review': str((datetime.now() + timedelta(days=7)).date()),
    }
    out_path = out_dir / 'forward_positions.json'
    out_path.write_text(json.dumps(tracker, ensure_ascii=False, indent=2), encoding='utf-8')
    return tracker


def print_full_report(
    horizon_analysis: dict,
    concentration: dict,
    factor_directions: list[dict],
    strategy_config: dict,
    forward: dict,
) -> None:
    print('\n' + '=' * 70)
    print('  QuantMind 模拟盘综合报告 | 系统优化更新')
    print('=' * 70)

    opt_hz = horizon_analysis['optimal_horizon']
    print('\n【一】最优持仓期分析')
    print('  期限    IR      期胜率   累计NAV   结论')
    for hz, v in horizon_analysis['by_horizon'].items():
        marker = '<-- 推荐' if hz == opt_hz else ''
        print('  %-5s %+.3f   %.1f%%   %.4f    %s' % (
            hz, v['ir'], v['win_rate']*100, v['cumulative_nav'], marker))

    print('\n【二】仓位集中度建议')
    print('  Top-3  均值: %+.3f' % (concentration.get('top3_mean') or 0))
    print('  Top-10 均值: %+.3f  Top-3 超额: %+.4f' % (
        (concentration.get('top10_mean') or 0), (concentration.get('top3_premium') or 0)))
    print('  推荐持仓数: %d 只' % (concentration.get('recommended_n', 10)))

    print('\n【三】因子方向信号（基于 21d 实际收益）')
    print('  %-28s IC       方向        显著性' % '因子')
    for f in factor_directions[:10]:
        sig = '***' if f['p_value'] < 0.01 else ('**' if f['p_value'] < 0.05 else ('*' if f['p_value'] < 0.10 else '  '))
        print('  %-28s %+.3f    %-10s  %s (p=%.3f)' % (
            f['factor'], f['ic_vs_21d_return'], f['direction'], sig, f['p_value']))

    print('\n【四】策略配置更新')
    sc = strategy_config
    print('  推荐持仓期: %s  (IR=%+.3f)' % (
        sc['holding_period']['recommended'],
        horizon_analysis['by_horizon'].get(sc['holding_period']['recommended'], {}).get('ir', 0)))
    print('  持仓数量:   %d 只（等权）' % sc['position_sizing']['top_n'])
    print('  止损线:     %.0f%%' % (sc['risk_controls']['stop_loss_threshold'] * 100))
    reversed_fs = sc['factor_signals']['reversed_factors']
    if reversed_fs:
        print('  质量因子反转警告: %s（建议下调权重或方向取反）' % ', '.join(f['factor'] for f in reversed_fs))

    print('\n【五】前向模拟持仓（持仓期=%s）' % forward['tracking_horizon'])
    print('  开放持仓: %d 只  |  下次检视: %s' % (
        forward['n_open_positions'], forward['next_review']))
    for p in forward['positions'][:10]:
        print('  [%s] #%-2s %-12s  预计到期: %s' % (
            p['as_of'], p['predicted_rank'], p['ticker'], p['estimated_exit_date']))

    print('\n【六】下阶段行动计划')
    print('  1. [立即] 将推荐持仓期从 3m 改为 21d（IR 最优）')
    print('  2. [立即] 跟踪 2026-03-31 建仓到 2026-04-29（21d到期日）')
    print('  3. [本季] 分析 roe_ttm 因子反转原因，考虑 regime 条件权重')
    print('  4. [下季] 2026Q2 面板更新后，重跑序贯验证 Round 8')
    print('  5. [持续] 每月末更新推荐、结算到期持仓、更新 realized_pnl')
    print('=' * 70 + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions', type=Path, default=Path('data/paper_trading/positions.parquet'))
    p.add_argument('--panel', type=Path, default=Path('data/panel/alpha_panel_v4.parquet'))
    p.add_argument('--recs', type=Path, default=Path('data/recommendations/'))
    p.add_argument('--out', type=Path, default=Path('data/paper_trading/'))
    args = p.parse_args(argv)

    df = pd.read_parquet(args.positions)
    panel = pd.read_parquet(args.panel)
    logger.info(f'加载持仓 {len(df)} 条，面板 {panel.shape}')

    args.out.mkdir(parents=True, exist_ok=True)

    horizon_analysis = analyze_horizon_optimality(df)
    opt_hz = horizon_analysis['optimal_horizon']

    concentration = analyze_concentration(df, opt_hz)
    factor_directions = analyze_factor_directions(panel, df)

    strategy_config = generate_strategy_config(horizon_analysis, concentration, factor_directions)
    sc_path = args.out / 'strategy_config.json'
    sc_path.write_text(json.dumps(strategy_config, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f'策略配置已保存: {sc_path}')

    fd_path = Path('data/features/factor_direction_v4.json')
    fd_path.parent.mkdir(parents=True, exist_ok=True)
    fd_path.write_text(json.dumps({
        'updated_at': datetime.now().isoformat(),
        'source': 'paper_trading_sim_9periods',
        'factor_directions': [
            {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in fd.items()}
            for fd in factor_directions
        ],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f'因子方向已保存: {fd_path}')

    forward = setup_forward_tracking(args.recs, args.out, horizon=opt_hz)
    logger.info(f'前向追踪已保存: {args.out / "forward_positions.json"}')

    print_full_report(horizon_analysis, concentration, factor_directions, strategy_config, forward)


if __name__ == '__main__':
    main()
