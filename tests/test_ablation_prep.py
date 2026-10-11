"""T-007 执行准备（设计 v3）：请求构造、匿名化检查、信息条件、时间审计、预注册分析（不调用模型）."""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from scripts.ablation_analyze import (
    STATEMENTS,
    analyze,
    assess,
    block_index,
    integrity,
    lifts,
    mock_responses,
    null_lambda,
    order_ic,
    parse_probe,
    parse_rankings,
    random_null_lift,
)
from scripts.ablation_build_prompts import (
    ANON_GROUPS,
    FACTORS,
    MAIN_GROUPS,
    N_REPEATS,
    NA,
    IdentityDictionary,
    anonymization_violations,
    build_cross_section,
    main,
    schema_enum,
    sha,
    time_audit,
)

TICKERS = [f"{600500 + i:06d}.SH" for i in range(30)]          # 含 600519（贵州茅台）
NAMES = {t: f"公司{i:02d}号" for i, t in enumerate(TICKERS)} | {"600519.SH": "贵州茅台"}


def _stock_basic() -> pd.DataFrame:
    rows = [{"ts_code": t, "name": NAMES[t], "fullname": NAMES[t] + "股份有限公司", "enname": f"Company {t[:6]} Ltd."}
            for t in TICKERS]
    rows.append({"ts_code": "000001.SZ", "name": "平安银行", "fullname": "平安银行股份有限公司",
                 "enname": "Ping An Bank Co., Ltd."})
    rows.append({"ts_code": "000004.SZ", "name": "*ST国华", "fullname": "深圳国华网安", "enname": None})
    return pd.DataFrame(rows)


IDENT = IdentityDictionary(_stock_basic())


def _top(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.standard_normal((30, len(FACTORS))), columns=FACTORS, index=TICKERS)
    df["score"] = np.linspace(1, 0, 30)
    return df


@pytest.fixture(scope="module")
def built():
    return build_cross_section(pd.Timestamp("2023-03-31"), _top(), np.random.default_rng(7), top_n=10)


def _req(reqs, group, repeat=0):
    return next(r for r in reqs if r["group"] == group and r["repeat"] == repeat)


def _table(payload) -> list[list[str]]:
    user = payload["messages"][1]["content"]
    rows = [ln for ln in user.splitlines() if re.match(r"^\| S\d{2} ", ln)]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in rows]


# ── 构造 ──────────────────────────────────────────────────────────────────
def test_request_counts(built):
    reqs, mapping = built
    by = pd.Series([r["group"] for r in reqs]).value_counts().to_dict()
    assert all(by[g] == N_REPEATS for g in MAIN_GROUPS)
    assert by["A_nodate"] == by["B_nodate"] == by["PROBE"] == 1
    assert len(mapping) == N_REPEATS * 30


@pytest.mark.parametrize("group", ANON_GROUPS)
def test_anonymous_groups_pass_full_payload_check(built, group):
    reqs, mapping = built
    r = _req(reqs, group)
    ids = sorted({m["id"] for m in mapping if m["repeat"] == 0})
    assert anonymization_violations(r["payload"], IDENT, ids) == []


@pytest.mark.parametrize("inject,expect", [("600519", "code:600519"), ("贵州茅台", "name:贵州茅台"),
                                           ("国华", "name:国华"), ("ping an bank co., ltd.", "name:ping an bank co., ltd.")])
def test_identity_leaks_anywhere_in_payload_are_caught(built, inject, expect):
    reqs, mapping = built
    p = copy.deepcopy(_req(reqs, "B")["payload"])
    p["messages"][0]["content"] += inject
    ids = sorted({m["id"] for m in mapping if m["repeat"] == 0})
    assert expect in anonymization_violations(p, IDENT, ids)


def test_factor_values_are_not_mistaken_for_codes():
    assert IDENT.violations("PE 12.600519 动量 0.600519") == []


def test_identified_groups_show_real_codes(built):
    reqs, _ = built
    for g in ("A", "Cstar"):
        codes = [row[1] for row in _table(_req(reqs, g)["payload"])]
        assert codes == TICKERS


def test_schema_is_identical_across_groups_and_sorted_by_id(built):
    reqs, _ = built
    for r in range(N_REPEATS):
        fmts = [json.dumps(_req(reqs, g, r)["payload"]["response_format"], sort_keys=True) for g in MAIN_GROUPS]
        assert len(set(fmts)) == 1
        enum = schema_enum(_req(reqs, "A", r)["payload"])
        assert enum == [f"S{k:02d}" for k in range(1, 31)]             # 按编号排序，不携带排名


def test_ids_shared_within_repeat_random_across_repeats(built):
    reqs, mapping = built
    for r in range(N_REPEATS):
        seqs = {tuple(row[0] for row in _table(_req(reqs, g, r)["payload"])) for g in MAIN_GROUPS}
        assert len(seqs) == 1                                            # 同一重复内四组同一编号
    perms = {tuple(m["id"] for m in mapping if m["repeat"] == r) for r in range(N_REPEATS)}
    assert len(perms) == N_REPEATS                                       # 各重复不同
    assert all(p != tuple(f"S{k:02d}" for k in range(1, 31)) for p in perms)   # 编号 ≠ 排名


def test_row_order_and_lgbm_rank_identical_across_groups(built):
    reqs, _ = built
    ranks = {tuple(row[2] for row in _table(_req(reqs, g)["payload"])) for g in MAIN_GROUPS}
    assert ranks == {tuple(str(k) for k in range(1, 31))}


def test_groups_carry_the_designed_information(built):
    reqs, _ = built
    tab = {g: _table(_req(reqs, g)["payload"]) for g in MAIN_GROUPS}
    assert all(row[1] != NA for row in tab["A"]) and all(row[3] != NA for row in tab["A"])
    assert all(row[1] == NA for row in tab["B"]) and all(row[3] != NA for row in tab["B"])
    assert all(row[1] != NA for row in tab["Cstar"]) and all(set(row[3:]) == {NA} for row in tab["Cstar"])
    assert all(row[1] == NA for row in tab["Dstar"]) and all(set(row[3:]) == {NA} for row in tab["Dstar"])
    assert len({_req(reqs, g)["payload"]["messages"][0]["content"] for g in MAIN_GROUPS}) == 1   # 同一 system prompt


def test_undated_variants_hide_only_the_date(built):
    reqs, _ = built
    for g, base in (("A_nodate", "A"), ("B_nodate", "B")):
        u_nd = _req(reqs, g)["payload"]["messages"][1]["content"]
        u = _req(reqs, base)["payload"]["messages"][1]["content"]
        assert f"截止日期（as_of）：{NA}" in u_nd and "2023-03-31" in u
        assert u_nd.replace(f"截止日期（as_of）：{NA}", "截止日期（as_of）：2023-03-31") == u


def test_private_mapping_stays_out_of_payload(built):
    reqs, _ = built
    for r in reqs:
        assert "ticker" not in json.dumps(r["payload"], ensure_ascii=False)
        assert set(r) == {"request_id", "as_of", "group", "repeat", "payload"}


def test_request_id_binds_payload(built):
    reqs, _ = built
    for r in reqs:
        assert r["request_id"] == sha(json.dumps(r["payload"], ensure_ascii=False, sort_keys=True))


def test_time_audit_classifies_by_return_end():
    cal = pd.bdate_range("2023-01-02", "2027-06-30")
    au = time_audit(["2024-06-28", "2024-10-15", "2025-06-30", "2026-09-30"], cal).set_index("as_of")
    assert au.loc["2024-06-28", "exposure_conservative"] == "潜在完整暴露"   # 收益结束早于 2025-01-01
    assert au.loc["2024-10-15", "exposure_conservative"] == "边界不确定"     # 收益窗口跨过下界
    assert au.loc["2025-06-30", "exposure_conservative"] == "边界不确定"
    assert au.loc["2025-06-30", "exposure_if_probe_estimate"] == "截止后"
    assert au.loc["2026-09-30", "exposure_conservative"] == "截止后"         # 晚于上游发布日


def test_main_freezes_inputs_and_lock(tmp_path):
    days = pd.to_datetime(["2023-03-31", "2023-06-30"])
    idx = pd.MultiIndex.from_product([days, TICKERS], names=["as_of", "ticker"])
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.standard_normal((len(idx), len(FACTORS) + 1)), index=idx,
                         columns=FACTORS + ["forward_return_63d"])
    panel.to_parquet(tmp_path / "panel.parquet")
    pd.DataFrame([{"as_of": d.date(), "ticker": t, "score": float(i)} for d in days
                  for i, t in enumerate(TICKERS)]).to_csv(tmp_path / "scores.csv", index=False)
    _stock_basic().to_parquet(tmp_path / "sb.parquet")
    pd.DataFrame({"cal_date": pd.bdate_range("2023-01-02", "2024-12-31").strftime("%Y%m%d")}).to_parquet(
        tmp_path / "cal.parquet")
    (tmp_path / "run_pilot.json").write_text(json.dumps({"model_sha256": "abc", "model_file": "x.gguf", "server_props": {
        "build_info": "b1-9a9394a", "chat_template": "{{ t }}"}}), encoding="utf-8")
    out = tmp_path / "frozen"
    args = ["--scores", str(tmp_path / "scores.csv"), "--panel", str(tmp_path / "panel.parquet"), "--stock-basic",
            str(tmp_path / "sb.parquet"), "--trade-calendar", str(tmp_path / "cal.parquet"), "--lock",
            str(tmp_path / "run_pilot.json"), "--out", str(out)]
    assert main(args) == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["design_version"] == "t007-design-v3" and man["n_cross_sections"] == 2
    assert man["lock"]["gguf_sha256"] == "abc" and man["lock"]["llama_cpp_build_info"] == "b1-9a9394a"
    assert man["n_requests"] == 2 * (4 * N_REPEATS + 3)
    assert len(pd.read_csv(out / "time_audit.csv")) == 2
    assert main(args) == 1                                               # 拒绝覆盖


# ── 分析 ──────────────────────────────────────────────────────────────────
def test_parse_rankings_checks_every_required_field():
    ids = ["S01", "S02", "S03"]
    good = {"rankings": [{"id": "S01", "reason": "r"}, {"id": "S02", "reason": "r"}], "portfolio_thesis": "t",
            "risk_warnings": []}
    assert parse_rankings(json.dumps(good), ids, 2) == (["S01", "S02"], "ok")
    bad = copy.deepcopy(good)
    bad["rankings"][1]["id"] = "S01"
    assert parse_rankings(json.dumps(bad), ids, 2)[1] == "wrong_count_or_duplicates"
    bad = copy.deepcopy(good)
    bad["rankings"][0]["id"] = "S09"
    assert parse_rankings(json.dumps(bad), ids, 2)[1] == "id_outside_candidates"
    bad = copy.deepcopy(good)
    del bad["portfolio_thesis"]
    assert parse_rankings(json.dumps(bad), ids, 2)[1] == "missing_required_fields"


def test_probe_parsing():
    ids = ["S01", "S02"]
    assert parse_probe(json.dumps({"guesses": [{"id": "S01", "code": "600519"}, {"id": "S02", "code": "unknown"}]}),
                       ids)[1] == "ok"
    assert parse_probe(json.dumps({"guesses": [{"id": "S01", "code": "600519"}]}), ids)[1] == "ids_mismatch"


def test_integrity_rejects_mixed_runs_duplicates_and_missing():
    reqs = [{"request_id": "a", "group": "A"}, {"request_id": "b", "group": "B"}]
    _, rep = integrity(reqs, [{"run_id": "r1", "model": "m", "request_id": "a"},
                              {"run_id": "r2", "model": "m", "request_id": "b"}])
    assert not rep["complete_main"] and "多个" in rep["reason"]
    _, rep = integrity(reqs, [{"run_id": "r", "model": "m", "request_id": "a"}])
    assert rep["missing_main"] == ["b"] and not rep["complete_main"]


def test_rank_direction():
    truth = {f"T{i}": float(30 - i) for i in range(30)}                 # T0 收益最高
    assert order_ic([f"T{i}" for i in range(30)], truth) == pytest.approx(1.0)
    assert order_ic([f"T{i}" for i in reversed(range(30))], truth) == pytest.approx(-1.0)


def test_baseline_identity_gives_zero_lift():
    cands = [{"id": f"S{k:02d}", "ticker": f"T{k}", "lgbm_rank": k + 1} for k in range(30)]
    truth = dict(zip([c["ticker"] for c in cands], np.random.default_rng(3).standard_normal(30), strict=True))
    lgbm_top10 = [c["id"] for c in cands[:10]]
    assert lifts(cands, lgbm_top10, truth)["lift_top30"] == 0.0


def test_random_construction_null_matches_theory():
    rng = np.random.default_rng(5)
    lr = {f"T{k}": k + 1 for k in range(30)}
    truth = {t: -r + rng.normal(0, 4.0) for t, r in lr.items()}         # 与 LGBM 排名正相关
    rho = order_ic(sorted(lr, key=lr.get), truth)
    sim = random_null_lift(lr, truth, 10, n_sim=20_000, seed=1)
    assert null_lambda(30, 10) == pytest.approx(380 / 870)
    assert sim == pytest.approx(-(1 - null_lambda(30, 10)) * rho, abs=0.01)


def test_direction_and_equivalence_are_recorded_separately():
    x = -0.01 + np.random.default_rng(0).normal(0, 0.002, 19)           # 精确的小负效应
    a = assess(x, block_index(19, 2))
    assert a["direction"] == "neg" and a["within_delta"] is True
    w = np.random.default_rng(1).normal(0, 0.3, 19)                     # 宽区间：既无方向也不等价
    a = assess(w, block_index(19, 2))
    assert a["direction"] == "none" and a["within_delta"] is False


def test_block_index_is_contiguous_and_truncated():
    idx = block_index(19, 4, n=50, seed=2)
    assert idx.shape == (50, 19) and idx.max() <= 18
    assert all(idx[0, j + 1] == idx[0, j] + 1 for j in range(0, 3))    # 第一块连续


def test_statements_attribute_no_mechanism():
    words = ("记忆", "推理", "memory", "reasoning")
    assert not any(w in s for triple in STATEMENTS.values() for s in triple for w in words)


def _study(n_periods: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    reqs, mapping, truth = [], [], []
    for p in range(n_periods):
        day = pd.Timestamp("2021-12-31") + pd.offsets.QuarterEnd(p)
        top = _top(seed + p)
        r, m = build_cross_section(day, top, rng, top_n=10)
        reqs += r
        mapping += m
        truth += [{"as_of": str(day.date()), "ticker": t, "forward_return_63d": float(rng.normal()),
                   **{f: float(top.loc[t, f]) for f in FACTORS}} for t in TICKERS]
    return reqs, mapping, pd.DataFrame(truth)


def test_mock_pipeline_runs_end_to_end():
    reqs, mapping, truth = _study(19)
    res = analyze(reqs, mapping, truth, mock_responses(reqs, 10), 10)
    assert res["n_periods_included"] == 19 and set(res["contrasts"]) == set(STATEMENTS)
    for c in res["contrasts"].values():
        assert "sensitive_to_block_length" in c and {"block_2", "block_3", "block_4", "block_6", "iid"} <= set(c)
    assert res["random_construction_null"]["lambda"] == pytest.approx(380 / 870)
    assert "exploratory_date_sensitivity" in res and res["identity_probe"]["share_unknown_mean"] == 1.0
    assert res["verdict"][-1] == "（19 期探索性估计）"


def test_pairing_rule_drops_periods_and_gate_stops_inference():
    reqs, mapping, truth = _study(18)
    resps = mock_responses(reqs, 10)
    first = reqs[0]["as_of"]
    bad_ids = {r["request_id"] for r in reqs if r["as_of"] == first and r["group"] == "B" and r["repeat"] in (0, 1, 2)}
    for x in resps:
        if x["request_id"] in bad_ids:
            x["content"] = "not json"
    res = analyze(reqs, mapping, truth, resps, 10)
    assert res["valid_repeats_per_period"][first] == 2 and res["n_periods_included"] == 17
    second = sorted({r["as_of"] for r in reqs})[1]                     # 再坏一个截面 → 16 < 17：不判读
    bad2 = {r["request_id"] for r in reqs if r["as_of"] == second and r["group"] == "Dstar"}
    for x in resps:
        if x["request_id"] in bad2:
            x["finish_reason"] = "length"
    res = analyze(reqs, mapping, truth, resps, 10)
    assert res["n_periods_included"] == 16 and res["verdict"][0].startswith("不判读")
