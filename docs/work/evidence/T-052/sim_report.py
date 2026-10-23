"""T-052 汇报：主配置的结果表、网格分布、净值图（三档资金分面）。读 sim_grid.py 的输出。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_data as sd  # noqa: E402

OUT = sd.ROOT / "data/line_a_v2/t052_sim"
BLUE, ORANGE, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
AUM_NAME = {100000: "10 万", 1000000: "100 万", 10000000: "1000 万"}


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.1f}%"


def main() -> None:
    g = pd.read_parquet(OUT / "grid_summary.parquet")
    g["buffer"] = g["buffer"].fillna(0.0)
    # 事后补充的描述性指标（登记里没有）：普跌行情里现金本身就有"超额"，按平均仓位扣掉基准的贡献
    g["excess_exposure_adj"] = g["total_return"] - (1 - g["avg_cash_share"]) * g["bench_pool_ew_return"]
    prim = json.loads((OUT / "primary.json").read_text(encoding="utf-8"))
    lines = ["## 主配置（事先指定，每档资金一个）", "",
             "| 窗口 | 资金 | 持股 | 权限 | 账户收益 | 前 1500 等权 | 超额 | 超额 95% 区间 | 按仓位调整的超额* | 中证 1000 | 最大回撤 | 费用占本金 | 换手（倍） | 平均现金 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(prim.items(), key=lambda kv: (kv[1]["config"]["window"], kv[1]["config"]["aum"],
                                                     kv[1]["config"]["boards"])):
        s, c = v["summary"], v["config"]
        ci = v["vs_pool_ew"]["ci95"]
        lines.append(f"| {c['window']} | {AUM_NAME[int(c['aum'])]} | {c['n']} | {c['boards']} | {pct(s['total_return'])} | "
                     f"{pct(s['bench_pool_ew_return'])} | {pct(v['vs_pool_ew']['excess_compound'])} | "
                     f"[{pct(ci[0])}, {pct(ci[1])}] | "
                     f"{pct(s['total_return'] - (1 - s['avg_cash_share']) * s['bench_pool_ew_return'])} | "
                     f"{pct(s.get('ret_中证1000'))} | {pct(s['max_drawdown'])} | "
                     f"{s['cost_share_of_aum'] * 100:.2f}% | {s['turnover_x']:.1f} | {s['avg_cash_share'] * 100:.1f}% |")
    lines += ["", "## 网格分布（全部组合；按信号 × 资金 × 窗口，账户收益的中位数与四分位）", "",
              "| 窗口 | 信号 | 资金 | 组合数 | 中位数 | 25% | 75% | 最好 | 最差 | 中位超额（对前 1500 等权） | 中位按仓位调整的超额* | 中位平均现金 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gg = g[g["boards"] != "main_only"]
    for (w, sig, aum), x in gg.groupby(["window", "signal", "aum"]):
        r = x["total_return"]
        lines.append(f"| {w} | {sig} | {AUM_NAME[int(aum)]} | {len(x)} | {pct(r.median())} | {pct(r.quantile(.25))} | "
                     f"{pct(r.quantile(.75))} | {pct(r.max())} | {pct(r.min())} | {pct(x['excess_vs_pool'].median())} | "
                     f"{pct(x['excess_exposure_adj'].median())} | {x['avg_cash_share'].median() * 100:.0f}% |")
    lines += ["", "## 单个维度的影响（W6，按仓位调整的超额*，其余维度取全部组合的中位数）", ""]
    for dim in ("n", "step", "buffer", "exec"):
        t = gg[gg["window"] == "W6"].groupby([dim, "aum"])["excess_exposure_adj"].median().unstack()
        lines.append(f"**{dim}**：" + "；".join(f"{i}: " + " / ".join(pct(t.loc[i, a]) for a in t.columns)
                                               for i in t.index) + "（10 万 / 100 万 / 1000 万）")
        lines.append("")
    lines += ["", "\\* 按仓位调整的超额 = 账户收益 − (1 − 平均现金占比) × 前 1500 等权收益。事后补充的描述性指标，不在登记里；"
              "用来扣掉普跌行情里现金本身带来的“超额”。"]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # 净值图：主配置，三档资金分面，W6
    from matplotlib import font_manager
    for f in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"):     # WSL 里没有中文字体，借用 Windows 的
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURF)
    idx = sd.indexes(pd.Timestamp("2026-04-20"), pd.Timestamp("2026-10-23"))
    for ax, aum in zip(axes, (100000, 1000000, 10000000), strict=True):
        k = next(k for k, v in prim.items() if v["config"]["window"] == "W6" and int(v["config"]["aum"]) == aum
                 and v["config"]["boards"] != "main_only")
        nav = pd.read_parquet(OUT / "runs" / k / "nav.parquet")
        a = nav["equity"] / aum
        b = nav["bench_pool_ew"] / nav["bench_pool_ew"].iloc[0]
        z = idx["中证1000"].reindex(nav.index).ffill()
        z = z / z.iloc[0]
        ax.set_facecolor(SURF)
        ax.plot(b.index, b, color=GRAY, lw=1.5, label="前 1500 等权")
        ax.plot(z.index, z, color=ORANGE, lw=1.5, label="中证 1000")
        ax.plot(a.index, a, color=BLUE, lw=2.0, label="模拟账户")
        for s, col in ((a, BLUE), (z, ORANGE), (b, GRAY)):
            ax.annotate(f"{(s.iloc[-1] - 1) * 100:+.1f}%", (s.index[-1], s.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", color=INK2, fontsize=8, va="center")
        c = prim[k]["config"]
        ax.set_title(f"{AUM_NAME[aum]}元 · ENS20 前 {c['n']} · 每 4 周调仓", color=INK, fontsize=10, loc="left")
        ax.axhline(1.0, color="#d8d7d2", lw=0.8)
        ax.grid(axis="y", color="#ecebe7", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    axes[0].set_ylabel("净值（起点 = 1）", color=INK2, fontsize=9)
    fig.suptitle("T-052 模拟盘：2026-04-29 至 10-23（含整数手、最低佣金、涨跌停、参与率、红利税）", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "primary_nav_W6.png", dpi=150, facecolor=SURF)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
