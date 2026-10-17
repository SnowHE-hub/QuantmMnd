"""T-030：日频 GPU 研究工具的离线单元测试（合成数据，不读项目数据、不需要 GPU）."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.gpu.batching import day_batches
from quantmind.research.gpu.ensemble import rank_average
from quantmind.research.gpu.folds import check_folds, make_folds
from quantmind.research.gpu.metrics import (
    daily_rank_ic,
    moving_block_bootstrap,
    rank_ic_by_date,
    two_sided_ci,
)
from quantmind.research.gpu.runtime import UnitLedger, load_checkpoint, save_checkpoint
from quantmind.research.gpu.transforms import (
    TrainZScore,
    day_bounds,
    gauss_rank_block,
    gauss_rank_by_day,
)
from quantmind.research.gpu.trees import N_GRADES, grades

torch = pytest.importorskip("torch")
from quantmind.research.gpu.losses import daily_ic_loss, listnet_loss  # noqa: E402


@pytest.fixture(scope="module")
def cal():
    return pd.bdate_range("2019-01-02", "2026-10-09")


@pytest.fixture(scope="module")
def grid(cal):
    return list(cal[5::5])


@pytest.mark.parametrize("h", ["20d", "63d"])
def test_folds_label_maturity_and_isolation(cal, grid, h):
    folds = make_folds(cal, grid, h)
    out = check_folds(folds, cal)
    assert out["n_folds"] == len(folds) > 0
    pos = {d: i for i, d in enumerate(cal)}
    for f in folds:
        ci = pos[f.cutoff]
        assert max(pos[d] for d in f.train_days) + f.H == ci - 1          # label_end 严格早于切点
        assert min(pos[d] for d in f.train_days) == ci - 756
        assert max(pos[d] for d in f.inner_train_days) + f.H < pos[f.inner_val_days[0]]
        assert min(pos[d] for d in f.pred_days) > ci + f.E
    tests = [d for f in folds for d in f.test_asof]
    assert len(tests) == len(set(tests))


def test_folds_check_rejects_leaky_fold(cal, grid):
    folds = make_folds(cal, grid, "20d")
    pos = {d: i for i, d in enumerate(cal)}
    f = folds[3]
    f.train_days = f.train_days + [cal[pos[f.cutoff] - f.H]]               # label_end == cutoff
    with pytest.raises(AssertionError):
        check_folds(folds, cal)


def test_gauss_rank_block_nan_and_monotone():
    x = np.array([3.0, np.nan, 1.0, 2.0, np.inf])
    g = gauss_rank_block(x)
    assert np.isnan(g[1]) and np.isnan(g[4])
    assert g[2] < g[3] < g[0]
    assert abs(g[3]) < 1e-12                                              # 中位数 → 0


def test_gauss_rank_by_day_uses_only_same_day():
    day = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([[1.0], [2.0], [3.0], [100.0], [200.0], [300.0]])
    g = gauss_rank_by_day(x, day)
    np.testing.assert_allclose(g[:3], g[3:])
    with pytest.raises(ValueError):
        day_bounds(np.array([1, 0]))


def test_train_zscore_fit_on_train_only():
    z = TrainZScore().fit(np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(z.transform(np.array([[2.0], [100.0]])), [[0.0], [5.0]])


def test_day_batches_never_split_a_day():
    day = np.repeat(np.arange(10), [3, 1, 4, 2, 5, 3, 3, 1, 2, 6])
    bs = day_batches(day, 3, np.random.default_rng(0))
    allrows = np.sort(np.concatenate(bs))
    np.testing.assert_array_equal(allrows, np.arange(len(day)))
    for b in bs:
        for d in np.unique(day[b]):
            assert (day[b] == d).sum() == (day == d).sum()


def test_daily_ic_loss_equals_mean_pearson():
    rng = np.random.default_rng(1)
    day = np.repeat([0, 1, 2], 50)
    p, y = rng.standard_normal(150), rng.standard_normal(150)
    want = np.mean([np.corrcoef(p[day == d], y[day == d])[0, 1] for d in range(3)])
    got = daily_ic_loss(torch.tensor(p), torch.tensor(y), torch.tensor(day))
    assert abs(-float(got) - want) < 1e-5


def test_daily_ic_loss_members_equal_mean_of_columns():
    rng = np.random.default_rng(2)
    day = torch.tensor(np.repeat([0, 1, 2], 40))
    p, y = torch.tensor(rng.standard_normal((120, 4))), torch.tensor(rng.standard_normal(120))
    want = np.mean([float(daily_ic_loss(p[:, j], y, day)) for j in range(4)])
    assert abs(float(daily_ic_loss(p, y, day)) - want) < 1e-6


def test_daily_ic_loss_ignores_cross_day_level():
    day = torch.tensor([0, 0, 0, 1, 1, 1])
    y = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    p = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])                    # 第二天整体平移不影响
    assert abs(float(daily_ic_loss(p, y, day)) + 1.0) < 1e-4


def test_listnet_prefers_correct_order():
    day = torch.tensor([0, 0, 0, 1, 1, 1])
    y = torch.tensor([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    good, bad = y * 3, -y * 3
    assert float(listnet_loss(good, y, day)) < float(listnet_loss(bad, y, day))


def test_rank_ic_by_date_and_bootstrap_deterministic():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2024-01-01", periods=4), [f"t{i}" for i in range(40)]],
                                     names=["as_of", "ticker"])
    y = pd.Series(np.arange(160) % 40, index=idx, dtype=float)
    ic = rank_ic_by_date(y, y)
    assert len(ic) == 4 and np.allclose(ic, 1.0)
    x = np.random.default_rng(0).standard_normal(100)
    b1, b2 = moving_block_bootstrap(x, 13, 500), moving_block_bootstrap(x, 13, 500)
    np.testing.assert_array_equal(b1, b2)
    lo, hi = two_sided_ci(b1, 0.975)
    assert lo < x.mean() < hi


def test_checkpoint_resume_and_ledger(tmp_path):
    m = torch.nn.Linear(4, 1)
    opt = torch.optim.AdamW(m.parameters(), 1e-2)
    for _ in range(3):
        opt.zero_grad()
        m(torch.ones(2, 4)).sum().backward()
        opt.step()
    save_checkpoint(tmp_path / "c.pt", {"model": m.state_dict(), "opt": opt.state_dict(), "step": 3})
    st = load_checkpoint(tmp_path / "c.pt")
    m2 = torch.nn.Linear(4, 1)
    m2.load_state_dict(st["model"])
    assert st["step"] == 3 and torch.equal(m.weight, m2.weight)
    assert load_checkpoint(tmp_path / "missing.pt") is None
    led = UnitLedger(tmp_path / "units")
    assert not led.done("a")
    led.mark("a", {"ic": 0.1})
    assert led.done("a") and led.get("a")["ic"] == 0.1 and len(led.all()) == 1


def test_daily_rank_ic_matches_spearman_and_respects_mask():
    rng = np.random.default_rng(2)
    day = np.repeat([0, 1, 2], 60)
    p, y = rng.standard_normal(180), rng.standard_normal(180)
    want = [pd.Series(p[day == d]).corr(pd.Series(y[day == d]), method="spearman") for d in range(3)]
    np.testing.assert_allclose(daily_rank_ic(p, y, day), want, atol=1e-12)
    mask = np.ones(180, dtype=bool)
    mask[day == 1] = False
    assert len(daily_rank_ic(p, y, day, mask)) == 2


def test_rank_average_is_scale_free_and_same_day():
    day = np.repeat([0, 1], 5)
    a = np.arange(10.0)
    np.testing.assert_allclose(rank_average([a, a * 1000 - 7], day), gauss_rank_by_day(a, day))


def test_grades_cover_range():
    g = grades(np.array([-5.0, 0.0, 5.0, np.nan]))
    assert g[0] == 0 and g[2] == N_GRADES - 1 and g[1] == N_GRADES // 2 and np.isnan(g[3])


@pytest.mark.parametrize("arch", ["mlp", "seqfusion", "gru", "alstm", "master"])
def test_nets_output_shape(arch):
    from quantmind.research.gpu.nets import ARCHS
    n, L, C = 12, 8, 5
    kw = {"mlp": {"d_in": 7}, "seqfusion": {"c_in": C, "d_snap_in": 6}, "gru": {"c_in": C}, "alstm": {"c_in": C},
          "master": {"c_in": C, "d_mkt": 3}}[arch]
    m = ARCHS[arch](**kw)
    out = m(tab=torch.randn(n, 7), seq=torch.randn(n, L, C), snap=torch.randn(n, 6), mkt=torch.randn(n, 3),
            day=torch.tensor([0] * 6 + [1] * 6))
    assert out.shape == (n,)


def _margin_snapshot():
    """构造含重复日、NaN、短历史、未入池票、as_of 之后行的融资快照."""
    as_of = pd.Timestamp("2020-03-10")
    uni = pd.DataFrame({"ticker": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]})
    rows = [
        ("000001.SZ", "2020-02-03", 1.0), ("000001.SZ", "2020-02-04", np.nan),
        ("000001.SZ", "2020-03-10", 2.0), ("000001.SZ", "2020-03-11", 99.0),   # as_of 之后，应剔除
        ("000002.SZ", "2020-03-01", 3.0), ("000002.SZ", "2020-03-01", 4.0),   # 同日两行
        ("000003.SZ", "2020-03-09", 5.0),                                     # 短历史
        ("600000.SH", "2020-03-10", 7.0),                                     # 不在 universe
    ]
    m = pd.DataFrame(rows, columns=["ticker", "trade_date", "rzmre"])
    return {"universe": uni, "margin": m, "daily_basic": pd.DataFrame({"ticker": uni["ticker"]})}, as_of.date()


def test_fast_margin_amount_matches_original():
    from quantmind.features.expansion import margin_buy_amount_20d as orig
    from quantmind.research.gpu.fast_factors import margin_buy_amount_20d as fast
    snap, as_of = _margin_snapshot()
    a, b = orig(snap, as_of), fast(snap, as_of)
    pd.testing.assert_series_equal(a, b, check_names=True)


def test_fast_margin_intensity_nan_without_mv():
    from quantmind.features.expansion import margin_buy_intensity as orig
    from quantmind.research.gpu.fast_factors import margin_buy_intensity_no_mv as fast
    snap, as_of = _margin_snapshot()
    pd.testing.assert_series_equal(orig(snap, as_of), fast(snap, as_of), check_names=True)


def test_fast_factors_install_rewires_table():
    from quantmind.features import expansion as ex
    from quantmind.research.gpu import fast_factors as ff
    orig = list(ex.EXPANSION_FACTORS)
    try:
        ff.install()
        got = dict(ex.EXPANSION_FACTORS)
        assert got["margin_buy_amount_20d"] is ff.margin_buy_amount_20d
        assert got["margin_buy_intensity"] is ff.margin_buy_intensity_no_mv
    finally:
        ex.EXPANSION_FACTORS[:] = orig


def test_master_inter_attention_stays_within_day():
    from quantmind.research.gpu.nets import MASTERNet
    torch.manual_seed(0)
    m = MASTERNet(c_in=4, d_mkt=2).eval()
    seq, mkt = torch.randn(10, 6, 4), torch.randn(10, 2)
    day = torch.tensor([0] * 5 + [1] * 5)
    base = m(seq=seq, mkt=mkt, day=day)
    seq2 = seq.clone()
    seq2[5:] += 3.0                                                      # 只改第二天的股票
    out = m(seq=seq2, mkt=mkt, day=day)
    torch.testing.assert_close(out[:5], base[:5])


def test_fast_margin_buy_amount_bitwise_equal_to_production():
    from quantmind.features import expansion as ex
    from quantmind.research.gpu import fast_factors as ff
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2026-01-01", periods=40)
    rows = []
    for i, t in enumerate(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]):
        n = [40, 7, 40, 0][i]
        for d in days[:n]:
            rows.append({"ticker": t, "trade_date": d, "rzmre": rng.lognormal(10, 2) if rng.random() > 0.1 else np.nan})
    rows += [dict(rows[3]), {"ticker": "000001.SZ", "trade_date": days[-1] + pd.Timedelta(days=3), "rzmre": 1e12}]
    m = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
    snap = {"margin": m, "universe": pd.DataFrame({"ticker": ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH",
                                                              "300001.SZ"]}),
            "daily_basic": pd.DataFrame({"ticker": ["000001.SZ"]})}
    as_of = days[-1].date()
    pd.testing.assert_series_equal(ff.margin_buy_amount_20d(snap, as_of), ex.margin_buy_amount_20d(snap, as_of),
                                   check_exact=True)
    pd.testing.assert_series_equal(ff.margin_buy_intensity_no_mv(snap, as_of).rename(None),
                                   ex.margin_buy_intensity(snap, as_of).rename(None), check_exact=True)


class _FakeMD:
    def __init__(self, mk):
        self.mk = mk

    def market_rows(self, block):
        return self.mk


def _tree_block(n=1200, f=7, seed=0):
    from quantmind.research.gpu.data import Block
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float16)
    X[rng.random((n, f)) < 0.05] = np.nan
    day = np.repeat(np.arange(n // 60), 60)
    meta = pd.DataFrame({"day": day})
    y = (X[:, 0].astype(np.float64) - X[:, 1] + rng.standard_normal(n) * 0.5)
    return Block(X, meta, pd.DatetimeIndex([])), _FakeMD(rng.standard_normal((n, 3)).astype(np.float32)), np.nan_to_num(y)


def test_tree_input_chunks_and_fortran_equal_dense(monkeypatch):
    import quantmind.research.gpu.trees as trees
    monkeypatch.setattr(trees, "CHUNK", 97)
    b, md, _ = _tree_block()
    rows = np.arange(5, 1100, 3)
    extra = np.arange(len(b.meta), dtype=np.float32)[:, None]
    src = trees.TreeInput(md, b, rows, extra=extra)
    ref = np.concatenate([b.X[rows].astype(np.float32), md.mk[rows], extra[rows]], axis=1)
    np.testing.assert_array_equal(np.concatenate(list(src.chunks())), ref)
    F = src.fortran()
    assert F.flags.f_contiguous and F.dtype == np.float32
    np.testing.assert_array_equal(F, ref)


def test_xgb_iter_dmatrix_trains_like_dense(monkeypatch):
    xgb = pytest.importorskip("xgboost")
    import quantmind.research.gpu.trees as trees
    monkeypatch.setattr(trees, "CHUNK", 250)
    b, md, y = _tree_block()
    src = trees.TreeInput(md, b)
    dense = np.concatenate(list(src.chunks()))
    prm = {"device": "cpu", "tree_method": "hist", "max_depth": 3, "seed": 0, "nthread": 1}
    d_it = trees.xgb_dmatrix(src, y, device="cpu")
    assert (d_it.num_row(), d_it.num_col()) == dense.shape
    p_it = xgb.train(prm, d_it, 20).inplace_predict(dense)
    p_de = xgb.train(prm, xgb.QuantileDMatrix(dense, label=y, max_bin=256), 20).inplace_predict(dense)
    assert np.corrcoef(p_it, p_de)[0, 1] > 0.999


def test_lgb_sequence_trains_like_dense(monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    import quantmind.research.gpu.trees as trees
    monkeypatch.setattr(trees, "CHUNK", 250)
    b, md, y = _tree_block()
    src = trees.TreeInput(md, b)
    dense = np.concatenate(list(src.chunks()))
    prm = {"objective": "regression", "num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 20}
    m = trees.LGB().fit(prm, src, y, b.meta["day"].to_numpy(), 20, seed=0)
    ref = lgb.train({**prm, "verbosity": -1, "seed": 0, "num_threads": trees.LGB_THREADS, "max_bin": 255, "deterministic": True},
                    lgb.Dataset(dense, label=y), 20)
    np.testing.assert_allclose(trees.LGB().predict(m, src, 20), ref.predict(dense), rtol=1e-6, atol=1e-9)


def test_split_queries_contiguous_bounded_and_within_day():
    from quantmind.research.gpu.trees import split_queries
    day = np.repeat(np.arange(4), [2500, 999, 1001, 10])
    order, gid = split_queries(day, 1000, seed=3)
    assert sorted(order.tolist()) == list(range(len(day)))
    assert np.all(np.diff(day[order]) >= 0)
    change = np.flatnonzero(np.diff(gid)) + 1
    starts = np.r_[0, change]
    assert len(np.unique(gid)) == len(starts)                 # 同组相邻
    sizes = np.diff(np.r_[starts, len(gid)])
    assert sizes.max() <= 1000
    assert all(len(np.unique(day[order][gid == g])) == 1 for g in np.unique(gid))
    assert len(np.unique(gid[day[order] == 0])) == 3 and len(np.unique(gid[day[order] == 2])) == 2


def test_dl_tab_inputs_chunked_equal_full_concat(monkeypatch):
    from quantmind.research.gpu import dl
    from quantmind.research.gpu.data import Block
    monkeypatch.setattr(dl, "CHUNK", 37)
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-02", periods=5)
    n, groups = 300, {"F1": [0, 1], "F2": [2], "F3": [3, 4], "F4": [5], "F5": [6]}
    X = rng.standard_normal((n, 7)).astype(np.float16)
    X[rng.random((n, 7)) < 0.3] = np.nan
    X[:20, 3:5] = np.nan
    day = np.sort(rng.integers(0, 5, n))
    meta = pd.DataFrame({"as_of": days[day], "day": day, "y20": rng.standard_normal(n), "y63": np.nan,
                         "top1500": True})
    b = Block(X, meta, days)

    class MD:
        market = pd.DataFrame(rng.standard_normal((5, 2)), index=days)
        stock_features = [f"c{i}" for i in range(7)]
        market_features = [0, 1]

        def group_columns(self, gs):
            return [c for g in gs for c in groups[g]]

    norm = dl.Norm(mkt=dl.TrainZScore().fit(MD.market.to_numpy()))
    inp = dl.Inputs(MD(), b, norm, "mlp", "20d", None, device="cpu")
    flags = np.stack([np.isnan(X[:, groups[g]]).all(axis=1) for g in dl.GROUPS], axis=1).astype(np.float16)
    ref = torch.cat([torch.from_numpy(np.nan_to_num(X, nan=0.0)), torch.from_numpy(flags), inp.mkt], dim=1)
    assert torch.equal(inp.tab, ref)
    assert inp.tab[:20, 7 + 2].eq(1).all()                    # F3 全缺 → 标志 = 1


def test_official_input_rejects_unauthorized_lookahead_column():
    from quantmind.research.gpu.data import assert_official_feature_names

    assert_official_feature_names(["sh_resid_rev_10", "mkt_ret"])
    with pytest.raises(ValueError, match="unauthorized lookahead column"):
        assert_official_feature_names(["CANARY_y63_train"])
    with pytest.raises(ValueError, match="unauthorized lookahead column"):
        assert_official_feature_names(["ret_1d", "fwd_return"])


def test_incremental_raw_segments_sum_to_full_prediction():
    pytest.importorskip("lightgbm")
    import quantmind.research.gpu.trees as trees
    b, md, y = _tree_block()
    src = trees.TreeInput(md, b)
    prm = {"objective": "regression", "num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 20}
    fam = trees.LGB()
    m = fam.fit(prm, src, y, b.meta["day"].to_numpy(), 30, seed=0)
    acc = np.zeros(len(src))
    for r in (10, 20, 30):
        acc += fam.predict(m, src, r, start=r - 10, raw=True)
        np.testing.assert_allclose(acc - fam.predict(m, src, r, raw=True), 0.0, atol=1e-9)
