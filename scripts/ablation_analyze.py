"""scripts/ablation_analyze.py — T-007 匿名化消融：完整性、lift、预注册分析（设计 v3，19 期探索性估计）.

输入（同一冻结目录）：requests.jsonl、private_mapping.jsonl、truth.csv、time_audit.csv、manifest.json、responses.jsonl。

lift（每个请求）：LLM 给出的 top_n 按其顺序在前，其余候选按 LGBM 排名接在后面；在 Top-30 内
  lift = Spearman(−名次, 真实 63d) − Spearman(−LGBM 排名, 真实 63d)
  各组用同一条规则。两组 lift 之差 = 两组完整排序与真实收益的 Spearman 之差（同一个 LGBM 基线被减掉）。

预注册分析（v3，执行前冻结；回应 G1 评审）：
  合法响应：JSON 符合 schema（编号在候选内、不重复、恰好 top_n 个）、finish_reason=stop、思考内容为空。
  配对单位：(截面, 重复)。四个主组都合法才算一个合法单位；某截面至少 MIN_REPEATS 个合法重复才纳入，
    组 lift 在该截面的合法重复上求平均（四组用同一组重复）。纳入截面少于 MIN_PERIODS → 只报告工程失败，不做统计判读。
  确认性比较族（4 个，Bonferroni）：
    A−B                  七因子可见时的显式标识增量（主量）
    B−Dstar              匿名条件下，七因子值在 LGBM 信息之上的增量
    Cstar−Dstar          七因子不可见时的显式标识增量
    (A−B)−(Cstar−Dstar)  标识效应是否随七因子信息而变化
  每个量分别记录（两者可以同时成立）：
    方向   98.75% 双侧区间整体 > 0 → "正"；整体 < 0 → "负"；否则"方向未定"
    等价   TOST：97.5% 区间整体在 (−δ, +δ) 内 → "在 ±δ 内"，δ = 0.05（事先设定的、针对本排序指标的容忍界限，
          尚未与净投资收益建立对应关系）
  区间：对每期的配对差直接重采样；移动块自助法（不循环；块起点在 [0, T−L] 上均匀抽；拼 ceil(T/L) 块后截断到 T），
    百分位区间，10,000 次。主块长 2；敏感性 3、4、6；独立自助法作对照。所有量在同一块长下使用同一组重采样索引。
    主块长与敏感性块长的方向 / 等价结论不一致 → 标"对依赖假设敏感"。
  表述：只用"显式标识提供额外增益 / 降低表现 / 在 ±δ 内 / 未定"等可由实验直接支持的说法；
    **不得**写"因子推理""以记忆为主""部分记忆"，也不把 A−B 称为记忆贡献或其下界。负的 A−B 是有解释价值的结果，不是异常。
  描述性（不进比较族）：各组 lift 与绝对 IC；Top-30 内 LGBM 的 IC；随机 Top-10 + LGBM 补尾的零分布
    （理论均值 −(1−λ)·ρ_LGBM，λ = (N−k)(N−k−1)/(N(N−1))，并模拟核对）；A 与 B 的 Top-10 重合率与排序一致性；
    相对同一 Top-30 等权收益的差；相对事先固定的简单因子规则的 lift 差；按时间审计暴露分类的分组均值。
  探索性：A/B 有日期 vs 无日期的配对差（重复 0）；身份探测正确率（PROBE）。
  判读只描述统计结果，**不得**直接推出投资动作；动作由 C1 另行决定。

  --mock：生成随机合法输出，只用来检查流水线（"管线烟测"，不是功效分析）
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "forward_return_63d"
MAIN = ("A", "B", "Cstar", "Dstar")
EXPLORATORY = ("A_nodate", "B_nodate")
PROBE = "PROBE"
CONTRASTS = ("A-B", "B-Dstar", "Cstar-Dstar", "(A-B)-(Cstar-Dstar)")
DELTA = 0.05
ALPHA = 0.05
FAMILY = len(CONTRASTS)
DIR_LEVEL = 1 - ALPHA / FAMILY            # 0.9875
TOST_LEVEL = 1 - 2 * ALPHA / FAMILY       # 0.975
N_BOOT = 10_000
BLOCK_MAIN = 2
BLOCK_SENS = (3, 4, 6)
MIN_REPEATS = 3
MIN_PERIODS = 17
SEED = 0
# 事先固定的简单因子规则（符号来自生产 system prompt 的评估维度；等权，缺失按 0）
SIMPLE_RULE = {"roe_ttm": 1, "accruals": -1, "pe_ttm": -1, "pb": -1, "distance_to_52w_high": 1, "momentum_6m": 1,
               "volatility_3m": -1}
NO_ACTION = "判读只描述统计结果，不得直接推出投资动作；动作由 C1 另行决定。"
STATEMENTS = {
    "A-B": ("七因子可见时，显式标识提供额外增益", "七因子可见时，显式标识降低表现", "七因子可见时，显式标识效应在 ±0.05 以内"),
    "B-Dstar": ("匿名条件下，七因子值在 LGBM 信息之上带来增益", "匿名条件下，七因子值降低表现", "匿名条件下，七因子值的增量在 ±0.05 以内"),
    "Cstar-Dstar": ("七因子不可见时，显式标识提供额外增益", "七因子不可见时，显式标识降低表现", "七因子不可见时，显式标识效应在 ±0.05 以内"),
    "(A-B)-(Cstar-Dstar)": ("显式标识效应在七因子可见时更大", "显式标识效应在七因子可见时更小", "两种信息条件下的显式标识效应之差在 ±0.05 以内"),
}


def spearman(a, b) -> float:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return float("nan")
    return float(s["a"].rank().corr(s["b"].rank()))


def order_ic(order: list[str], truth: dict[str, float]) -> float:
    """排序（第 1 名在前）与真实收益的 Spearman：按收益从高到低的排序 = +1。"""
    return spearman([-i for i in range(len(order))], [truth.get(t, np.nan) for t in order])


# ── 解析与完整性 ────────────────────────────────────────────────────────────
def parse_rankings(content, ids: list[str], top_n: int) -> tuple[list[str] | None, str]:
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid_json"
    if not isinstance(obj, dict):
        return None, "not_object"
    if not isinstance(obj.get("portfolio_thesis"), str) or not isinstance(obj.get("risk_warnings"), list):
        return None, "missing_required_fields"
    ranks = obj.get("rankings")
    if not isinstance(ranks, list) or not all(isinstance(r, dict) for r in ranks):
        return None, "no_rankings"
    if any(not isinstance(r.get("id"), str) or not isinstance(r.get("reason"), str) for r in ranks):
        return None, "ranking_item_fields"
    order = [r["id"] for r in ranks]
    if any(t not in ids for t in order):
        return None, "id_outside_candidates"
    if len(order) != top_n or len(set(order)) != top_n:
        return None, "wrong_count_or_duplicates"
    return order, "ok"


def parse_probe(content, ids: list[str]) -> tuple[dict[str, str] | None, str]:
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid_json"
    g = obj.get("guesses") if isinstance(obj, dict) else None
    if not isinstance(g, list) or not all(isinstance(x, dict) for x in g):
        return None, "no_guesses"
    out = {x.get("id"): str(x.get("code")) for x in g}
    if set(out) != set(ids) or len(g) != len(ids):
        return None, "ids_mismatch"
    return out, "ok"


def integrity(reqs: list[dict], resps: list[dict]) -> tuple[dict[str, dict], dict]:
    ok = [r for r in resps if r.get("status", "ok") == "ok"]
    runs = Counter(r.get("run_id") for r in ok)
    models = Counter(r.get("model") for r in ok)
    errors = Counter(r["request_id"] for r in resps if r.get("status") == "error")
    rep = {"runs": dict(runs), "models": dict(models), "duplicates": [], "missing": [], "foreign": [],
           "transport_error_records": int(sum(errors.values())), "complete_main": False}
    if len(runs) > 1 or len(models) > 1:
        rep["reason"] = "响应来自多个 run_id 或模型"
        return {}, rep
    want = {r["request_id"] for r in reqs}
    cnt = Counter(r["request_id"] for r in ok)
    rep["duplicates"] = sorted(k for k, n in cnt.items() if n > 1)
    rep["foreign"] = sorted(set(cnt) - want)
    rep["missing"] = sorted(want - set(cnt))
    main_ids = {r["request_id"] for r in reqs if r["group"] in MAIN}
    rep["missing_main"] = sorted(main_ids - set(cnt))
    rep["complete_main"] = not (rep["duplicates"] or rep["foreign"] or rep["missing_main"])
    return ({r["request_id"]: r for r in ok} if not rep["duplicates"] else {}), rep


def validity(resp: dict | None, ids: list[str], top_n: int) -> tuple[list[str] | None, str]:
    if resp is None:
        return None, "missing"
    if resp.get("finish_reason") not in (None, "stop"):
        return None, f"finish_{resp.get('finish_reason')}"
    if resp.get("reasoning_content"):
        return None, "reasoning_not_disabled"
    return parse_rankings(resp.get("content"), ids, top_n)


# ── lift 与零分布 ───────────────────────────────────────────────────────────
def full_order(top_tickers: list[str], lgbm_rank: dict[str, int]) -> list[str]:
    rest = [t for t in sorted(lgbm_rank, key=lgbm_rank.get) if t not in top_tickers]
    return list(top_tickers) + rest


def lifts(cands: list[dict], order_ids: list[str], truth: dict[str, float]) -> dict:
    """cands：私有映射中该 (as_of, repeat) 的行（id, ticker, lgbm_rank）."""
    id2t = {c["id"]: c["ticker"] for c in cands}
    lgbm_rank = {c["ticker"]: c["lgbm_rank"] for c in cands}
    llm_top = [id2t[i] for i in order_ids]
    full = full_order(llm_top, lgbm_rank)
    lgbm_full = sorted(lgbm_rank, key=lgbm_rank.get)
    llm30, lgbm30 = order_ic(full, truth), order_ic(lgbm_full, truth)
    k = len(llm_top)
    rets = [truth.get(t, np.nan) for t in lgbm_full]
    return {"lift_top30": llm30 - lgbm30, "llm_ic_top30": llm30, "lgbm_ic_top30": lgbm30,
            "llm_topn_ret": float(np.nanmean([truth.get(t, np.nan) for t in llm_top])),
            "lgbm_topn_ret": float(np.nanmean([truth.get(t, np.nan) for t in lgbm_full[:k]])),
            "top30_ew_ret": float(np.nanmean(rets)), "top_tickers": llm_top, "full_order": full}


def null_lambda(n: int, k: int) -> float:
    return (n - k) * (n - k - 1) / (n * (n - 1))


def random_null_lift(lgbm_rank: dict[str, int], truth: dict[str, float], k: int, n_sim: int = 2000,
                     seed: int = 0) -> float:
    """随机有序抽 k 只放前面、其余按 LGBM 补尾时的 lift 模拟均值（向量化；与 order_ic 同一 Spearman 口径）."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    tick = sorted(lgbm_rank, key=lgbm_rank.get)
    n = len(tick)
    base = order_ic(tick, truth)
    ret = np.array([truth.get(t, np.nan) for t in tick], dtype=float)
    keep = ~np.isnan(ret)
    r_ret = rankdata(ret[keep])
    perm = np.argsort(rng.random((n_sim, n)), axis=1)[:, :k]          # 每行：随机有序的 k 个 LGBM 下标
    pos = np.tile(np.arange(n, dtype=float) + k, (n_sim, 1))         # 未入选：按 LGBM 次序排在 k 之后
    pos[np.arange(n_sim)[:, None], perm] = np.arange(k)[None, :]
    p = pos[:, keep]
    r_pos = p.argsort(axis=1).argsort(axis=1) + 1.0                   # 名次无并列
    a = -(r_pos - r_pos.mean(axis=1, keepdims=True))
    b = r_ret - r_ret.mean()
    ic = (a @ b) / (np.sqrt((a * a).sum(axis=1)) * np.sqrt((b * b).sum()))
    return float(np.mean(ic) - base)


def simple_rule_order(fac: pd.DataFrame, k: int) -> list[str]:
    s = sum(sign * fac[f].fillna(0.0) for f, sign in SIMPLE_RULE.items()) / len(SIMPLE_RULE)
    return list(s.sort_values(ascending=False, kind="mergesort").index[:k])


# ── 区间 ────────────────────────────────────────────────────────────────────
def block_index(T: int, block: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """移动块自助法的重采样索引 (n, T)；block<=1 为独立自助法。不循环，最后一块截断。"""
    rng = np.random.default_rng(seed)
    if block <= 1:
        return rng.integers(0, T, size=(n, T))
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, k))
    return (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]


def interval(dist: np.ndarray, level: float) -> tuple[float, float]:
    lo = (1 - level) / 2 * 100
    return float(np.percentile(dist, lo)), float(np.percentile(dist, 100 - lo))


def assess(x: np.ndarray, idx: np.ndarray) -> dict:
    dist = np.asarray(x, dtype=float)[idx].mean(axis=1)
    d_lo, d_hi = interval(dist, DIR_LEVEL)
    t_lo, t_hi = interval(dist, TOST_LEVEL)
    direction = "pos" if d_lo > 0 else ("neg" if d_hi < 0 else "none")
    equivalent = bool(t_lo > -DELTA and t_hi < DELTA)
    return {"mean": float(np.mean(x)), "direction": direction, "within_delta": equivalent,
            "dir_ci": [d_lo, d_hi], "tost_ci": [t_lo, t_hi], "ci95": list(interval(dist, 0.95))}


def statement(name: str, a: dict) -> list[str]:
    pos, neg, eq = STATEMENTS[name]
    out = [pos] if a["direction"] == "pos" else ([neg] if a["direction"] == "neg" else [])
    if a["within_delta"]:
        out.append(eq)
    return out or [f"{name}：方向未定，也未落在 ±0.05 以内（未定）"]


# ── 主分析 ──────────────────────────────────────────────────────────────────
def analyze(reqs: list[dict], mapping: list[dict], truth_df: pd.DataFrame, resps: list[dict], top_n: int,
            audit: pd.DataFrame | None = None) -> dict:
    by_id, rep = integrity(reqs, resps)
    truth = {}
    for r in truth_df.itertuples(index=False):
        truth.setdefault(r.as_of, {})[r.ticker] = getattr(r, LABEL)
    fac = {d: g.set_index("ticker") for d, g in truth_df.groupby("as_of")}
    cand: dict[tuple, list] = {}
    for m in mapping:
        cand.setdefault((m["as_of"], m["repeat"]), []).append(m)
    result = {"n_requests": len(reqs), "integrity": rep, "note": NO_ACTION,
              "preregistered": {"delta": DELTA, "alpha": ALPHA, "family": list(CONTRASTS), "dir_level": DIR_LEVEL,
                                "tost_level": TOST_LEVEL, "block_main": BLOCK_MAIN, "block_sensitivity": BLOCK_SENS,
                                "n_boot": N_BOOT, "min_repeats": MIN_REPEATS, "min_periods": MIN_PERIODS,
                                "simple_rule": SIMPLE_RULE}}
    rows, fails, probes = [], [], []
    for req in reqs:
        key = (req["as_of"], req["repeat"])
        cs = cand[key]
        ids = [c["id"] for c in cs]
        resp = by_id.get(req["request_id"])
        if req["group"] == PROBE:
            g, st = parse_probe(resp.get("content"), ids) if resp else (None, "missing")
            if g is not None:
                id2t = {c["id"]: c["ticker"][:6] for c in cs}
                probes.append({"as_of": req["as_of"], "accuracy": float(np.mean([g[i] == id2t[i] for i in ids])),
                               "share_unknown": float(np.mean([g[i] == "unknown" for i in ids]))})
            else:
                fails.append({"as_of": req["as_of"], "group": PROBE, "repeat": req["repeat"], "status": st})
            continue
        order, st = validity(resp, ids, top_n)
        if order is None:
            fails.append({"as_of": req["as_of"], "group": req["group"], "repeat": req["repeat"], "status": st})
            continue
        rows.append({"as_of": req["as_of"], "group": req["group"], "repeat": req["repeat"],
                     **lifts(cs, order, truth[req["as_of"]])})
    result["invalid_or_missing"] = fails
    result["json_valid_rate_main"] = (sum(1 for r in rows if r["group"] in MAIN)
                                      / max(1, sum(1 for r in reqs if r["group"] in MAIN)))
    df = pd.DataFrame(rows)
    if df.empty:
        result["verdict"] = ["不判读：没有合法响应"]
        return result

    # 配对单位与纳入规则
    main = df[df.group.isin(MAIN)]
    unit_ok = main.groupby(["as_of", "repeat"])["group"].nunique() == len(MAIN)
    good_units = unit_ok[unit_ok].index
    per_period_rep = pd.Series([a for a, _ in good_units]).value_counts()
    included = sorted(per_period_rep[per_period_rep >= MIN_REPEATS].index)
    result["n_periods_total"] = int(main.as_of.nunique())
    result["n_periods_included"] = len(included)
    result["valid_repeats_per_period"] = per_period_rep.sort_index().to_dict()
    if len(included) < MIN_PERIODS:
        result["verdict"] = [f"不判读：合法截面 {len(included)} < {MIN_PERIODS}（工程失败，见 invalid_or_missing）"]
        return result
    m = main.set_index(["as_of", "repeat"]).loc[[u for u in good_units if u[0] in included]].reset_index()
    wide = m.pivot_table(index="as_of", columns="group", values="lift_top30", aggfunc="mean").sort_index()[list(MAIN)]
    contrasts = pd.DataFrame({"A-B": wide["A"] - wide["B"], "B-Dstar": wide["B"] - wide["Dstar"],
                              "Cstar-Dstar": wide["Cstar"] - wide["Dstar"]})
    contrasts["(A-B)-(Cstar-Dstar)"] = contrasts["A-B"] - contrasts["Cstar-Dstar"]
    T = len(contrasts)
    idx = {b: block_index(T, b) for b in (1, BLOCK_MAIN, *BLOCK_SENS)}
    out = {}
    for c in CONTRASTS:
        x = contrasts[c].to_numpy()
        res = {f"block_{b}": assess(x, idx[b]) for b in (BLOCK_MAIN, *BLOCK_SENS)}
        res["iid"] = assess(x, idx[1])
        mainres = res[f"block_{BLOCK_MAIN}"]
        sens = {(res[f"block_{b}"]["direction"], res[f"block_{b}"]["within_delta"]) for b in (BLOCK_MAIN, *BLOCK_SENS)}
        res["sensitive_to_block_length"] = len(sens) > 1
        res["statements"] = statement(c, mainres) + (["（对依赖假设敏感：不同块长结论不一致）"] if len(sens) > 1 else [])
        out[c] = res
    result["contrasts"] = out
    result["per_period"] = pd.concat([wide, contrasts], axis=1).round(6).reset_index().to_dict("records")

    # 描述性
    groups_desc = {}
    for g in MAIN:
        a = assess(wide[g].to_numpy(), idx[BLOCK_MAIN])
        sub = m[m.group == g]
        groups_desc[g] = {"lift_mean": a["mean"], "lift_ci95": a["ci95"],
                          "llm_ic_top30": float(sub.llm_ic_top30.mean()), "lgbm_ic_top30": float(sub.lgbm_ic_top30.mean()),
                          "topn_ret_minus_top30_ew": float((sub.llm_topn_ret - sub.top30_ew_ret).mean())}
    result["groups_descriptive"] = groups_desc
    k = top_n
    nulls, rule = [], []
    for d in included:
        cs = cand[(d, 0)]
        lr = {c["ticker"]: c["lgbm_rank"] for c in cs}
        rho = order_ic(sorted(lr, key=lr.get), truth[d])
        lam = null_lambda(len(lr), k)
        nulls.append({"as_of": d, "rho_lgbm_top30": rho, "theory_null_lift": -(1 - lam) * rho,
                      "sim_null_lift": random_null_lift(lr, truth[d], k)})
        r_top = simple_rule_order(fac[d].loc[list(lr)], k)
        rule.append({"as_of": d, "rule_lift": order_ic(full_order(r_top, lr), truth[d]) - rho})
    nulls, rule = pd.DataFrame(nulls).set_index("as_of"), pd.DataFrame(rule).set_index("as_of")
    result["random_construction_null"] = {"lambda": null_lambda(30, k), "per_period": nulls.round(6).reset_index().to_dict("records"),
                                          "mean_theory": float(nulls.theory_null_lift.mean()),
                                          "mean_sim": float(nulls.sim_null_lift.mean())}
    result["vs_simple_rule"] = {g: float((wide[g] - rule["rule_lift"]).mean()) for g in MAIN} | {
        "rule_lift_mean": float(rule["rule_lift"].mean())}
    # A 与 B 的 Top-10 重合与排序一致
    ov = []
    for (d, _r), sub in m[m.group.isin(["A", "B"])].groupby(["as_of", "repeat"]):
        ta, tb = sub[sub.group == "A"].iloc[0], sub[sub.group == "B"].iloc[0]
        pos_b = {t: i for i, t in enumerate(tb.full_order)}
        ov.append({"as_of": d, "top_overlap": len(set(ta.top_tickers) & set(tb.top_tickers)) / k,
                   "order_spearman": spearman(list(range(len(ta.full_order))), [pos_b[t] for t in ta.full_order])})
    ov = pd.DataFrame(ov)
    result["A_vs_B_agreement"] = {"top_overlap_mean": float(ov.top_overlap.mean()),
                                  "order_spearman_mean": float(ov.order_spearman.mean())}
    # 时间审计分组（描述）
    if audit is not None:
        au = audit.set_index("as_of")
        result["A-B_by_exposure"] = {col: contrasts["A-B"].groupby(au.loc[contrasts.index, col]).agg(["mean", "count"])
                                     .round(6).to_dict("index") for col in ("exposure_conservative", "exposure_if_probe_estimate")}
    # 探索性：日期
    ex = df[(df.repeat == 0) & df.group.isin(["A", "B", *EXPLORATORY])]
    ew = ex.pivot_table(index="as_of", columns="group", values="lift_top30").dropna()
    if len(ew) >= 5 and set(EXPLORATORY) <= set(ew.columns):
        diff = ((ew["A"] - ew["B"]) - (ew["A_nodate"] - ew["B_nodate"])).to_numpy()
        a = assess(diff, block_index(len(diff), BLOCK_MAIN))
        result["exploratory_date_sensitivity"] = {"n": int(len(diff)), "mean": a["mean"], "ci95": a["ci95"],
                                                  "label": "标识效应对日期可见性的敏感性（探索性，重复 0）"}
    if probes:
        p = pd.DataFrame(probes)
        result["identity_probe"] = {"n_periods": int(len(p)), "accuracy_mean": float(p.accuracy.mean()),
                                    "accuracy_max": float(p.accuracy.max()), "share_unknown_mean": float(p.share_unknown.mean()),
                                    "note": "成功说明存在重识别能力；失败不能证明排序任务中没有重识别"}
    result["verdict"] = [s for c in CONTRASTS for s in out[c]["statements"]] + ["（19 期探索性估计）"]
    return result


def mock_responses(reqs: list[dict], top_n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for r in reqs:
        props = r["payload"]["response_format"]["json_schema"]["schema"]["properties"]
        if "guesses" in props:
            ids = props["guesses"]["items"]["properties"]["id"]["enum"]
            content = json.dumps({"guesses": [{"id": i, "code": "unknown"} for i in ids]})
        else:
            ids = props["rankings"]["items"]["properties"]["id"]["enum"]
            pick = list(rng.permutation(ids)[:top_n])
            content = json.dumps({"rankings": [{"id": t, "reason": "mock"} for t in pick],
                                  "portfolio_thesis": "mock", "risk_warnings": []})
        out.append({"run_id": "mock", "model": "mock", "status": "ok", "request_id": r["request_id"],
                    "as_of": r["as_of"], "group": r["group"], "finish_reason": "stop", "content": content})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="结果 json，默认 <dir>/analysis.json")
    a = ap.parse_args(argv)

    def jl(name: str) -> list[dict]:
        return [json.loads(x) for x in (a.dir / name).read_text(encoding="utf-8").splitlines() if x]

    reqs, mapping = jl("requests.jsonl"), jl("private_mapping.jsonl")
    top_n = json.loads((a.dir / "manifest.json").read_text(encoding="utf-8"))["top_n"]
    resps = mock_responses(reqs, top_n) if a.mock else jl("responses.jsonl")
    audit = pd.read_csv(a.dir / "time_audit.csv") if (a.dir / "time_audit.csv").exists() else None
    result = analyze(reqs, mapping, pd.read_csv(a.dir / "truth.csv"), resps, top_n, audit)
    result["mock"] = a.mock
    out = a.out or (a.dir / ("analysis_mock.json" if a.mock else "analysis.json"))
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("n_periods_included", "json_valid_rate_main", "verdict")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
