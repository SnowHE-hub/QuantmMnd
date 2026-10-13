"""树模型族（LightGBM CPU / XGBoost GPU / CatBoost GPU）的统一接口.

特征 = 股票级当日 gauss-rank（float16 → 按块转 float32，NaN 保留）+ 市场级原始值（树对单调变换不敏感，不标准化）。
目标：回归 → 当日 gauss-rank 标签；排序 → 当日分位等级 0..N_GRADES−1，query = 交易日。
内层：在内层训练日上训练到 max_rounds，每 eval_every 轮在内层验证日（前 1500 池）上算日均 RankIC，选 (配置, 轮数)。
"""
from __future__ import annotations

import os

import numpy as np
from scipy.special import ndtr

from quantmind.research.gpu.data import Block, ModelData
from quantmind.research.gpu.metrics import daily_rank_ic
from quantmind.research.gpu.transforms import day_bounds

N_GRADES = 20
CHUNK = 200_000
LGB_THREADS = int(os.environ.get("QM_LGB_THREADS", "8"))   # 与 GPU 任务并行时调小，避免 OpenMP 超额订阅


def grades(y_rank: np.ndarray) -> np.ndarray:
    """当日 gauss-rank → 分位等级（0..N_GRADES−1）；NaN 保留 NaN（调用方剔除）。"""
    g = np.floor(ndtr(y_rank) * N_GRADES)
    return np.clip(g, 0, N_GRADES - 1)


class TreeInput:
    """block 的选定行 → float32 [股票级 | 市场级 | extra]，按块产出.

    一折全训练窗 ≈ 3.7M 行 × 276 列，整块 float32 ≈ 4 GB；XGBoost 走 DataIter、LightGBM 走 Sequence，
    都只按块转换；CatBoost 用一份 Fortran 序数组（Pool 直接共享，不再复制）。
    """

    def __init__(self, md: ModelData, block: Block, rows: np.ndarray | None = None, extra: np.ndarray | None = None):
        self.X = block.X
        self.mk = md.market_rows(block)
        self.extra = extra
        self.rows = np.arange(len(block.meta)) if rows is None else rows
        self.n_stock = block.X.shape[1]
        self.n_cols = self.n_stock + self.mk.shape[1] + (0 if extra is None else extra.shape[1])

    def __len__(self) -> int:
        return len(self.rows)

    def take(self, order: np.ndarray) -> TreeInput:
        """同一份底层数据、行按 order 重排（CatBoost 要求同组样本相邻）。"""
        out = object.__new__(TreeInput)
        out.__dict__.update(self.__dict__)
        out.rows = self.rows[order]
        return out

    def chunk(self, s: int, e: int) -> np.ndarray:
        r = self.rows[s:e]
        out = np.empty((len(r), self.n_cols), dtype=np.float32)
        out[:, :self.n_stock] = self.X[r]
        out[:, self.n_stock:self.n_stock + self.mk.shape[1]] = self.mk[r]
        if self.extra is not None:
            out[:, self.n_stock + self.mk.shape[1]:] = self.extra[r]
        return out

    def chunks(self):
        for s in range(0, len(self), CHUNK):
            yield self.chunk(s, s + CHUNK)

    def fortran(self) -> np.ndarray:
        out = np.empty((len(self), self.n_cols), dtype=np.float32, order="F")
        for s in range(0, len(self), CHUNK):
            c = self.chunk(s, s + CHUNK)
            out[s:s + len(c)] = c
        return out


def dense_features(md: ModelData, block: Block, rows: np.ndarray | None = None) -> np.ndarray:
    """小块（预测日、测试 as_of）用的整块 float32；训练窗用 TreeInput。"""
    src = TreeInput(md, block, rows)
    return np.concatenate(list(src.chunks())) if len(src) else np.empty((0, src.n_cols), dtype=np.float32)


def xgb_dmatrix(src: TreeInput, y: np.ndarray, qid: np.ndarray | None = None, max_bin: int = 256,
                device: str = "cuda"):
    """按块喂给 QuantileDMatrix。device=cuda 时每块先转成 CuPy 数组，ELLPACK 直接建在显存里
    （numpy 块会在主机上留一份分箱索引：实测 3M 行多占 ≈ 2.0 GB，CuPy 块 ≈ 0.6 GB）。"""
    import xgboost as xgb
    to_dev = np.asarray
    if device == "cuda":
        import cupy
        to_dev = cupy.asarray

    class _It(xgb.DataIter):
        def __init__(self):
            self.i = 0
            super().__init__()

        def next(self, input_data) -> bool:
            s = self.i * CHUNK
            if s >= len(src):
                return False
            e = s + CHUNK
            kw = {"data": to_dev(src.chunk(s, e)), "label": to_dev(y[s:e])}
            if qid is not None:
                kw["qid"] = to_dev(qid[s:e])
            input_data(**kw)
            self.i += 1
            return True

        def reset(self) -> None:
            self.i = 0

    return xgb.QuantileDMatrix(_It(), max_bin=max_bin)


def train_target(block: Block, horizon: str, objective: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (可训练行下标, 目标)；标签缺失的行剔除。"""
    yr = block.label_rank(horizon)
    rows = np.flatnonzero(np.isfinite(yr))
    y = yr[rows] if objective == "regression" else grades(yr[rows])
    return rows, y


class XGB:
    name = "xgb"

    def fit(self, params: dict, src: TreeInput, y: np.ndarray, day: np.ndarray, rounds: int, seed: int):
        import xgboost as xgb
        p = dict(params)
        obj = p.pop("objective")
        base = {"device": "cuda", "tree_method": "hist", "seed": seed, "verbosity": 0, "max_bin": 256}
        if obj == "regression":
            base["objective"] = "reg:squarederror"
            d = xgb_dmatrix(src, y)
        else:
            base.update(objective="rank:pairwise", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8)
            d = xgb_dmatrix(src, y, qid=day)
        model = xgb.train({**base, **p}, d, num_boost_round=rounds)
        del d
        return model

    def predict(self, model, src: TreeInput, rounds: int, start: int = 0, raw: bool = False) -> np.ndarray:
        import cupy
        model.set_param({"device": "cuda"})
        kw = {"iteration_range": (start, rounds), "predict_type": "margin" if raw else "value"}
        return np.concatenate([cupy.asnumpy(model.inplace_predict(cupy.asarray(c), **kw)) for c in src.chunks()]
                              ).astype(np.float64)


def split_queries(day: np.ndarray, max_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """每个交易日随机均分成 ceil(n / max_size) 个子查询（CatBoost GPU 单个查询最多 1023 个样本）.

    返回 (行重排 order, 重排后的 group_id)；同一子查询的行相邻、日期顺序不变。
    """
    rng = np.random.default_rng(seed)
    b = day_bounds(day)
    sub = np.empty(len(day), dtype=np.int64)
    for s, e in zip(b[:-1], b[1:], strict=True):
        n_g = -(-(e - s) // max_size)
        sub[s:e] = rng.permutation(e - s) % n_g
    order = np.lexsort((np.arange(len(day)), sub, day))
    gid = day[order].astype(np.int64) * 1024 + sub[order]
    return order, gid


class CAT:
    name = "cat"
    MAX_QUERY = 1000
    GPU_RAM_PART = 0.55                      # 默认 0.95 会预占 ≈ 7.1 GiB；给日更 Step7a 的 ollama 留显存

    def fit(self, params: dict, src: TreeInput, y: np.ndarray, day: np.ndarray, rounds: int, seed: int):
        from catboost import CatBoost, Pool
        p = dict(params)
        obj = p.pop("objective")
        loss = {"regression": "RMSE", "yetirank": "YetiRank", "queryrmse": "QueryRMSE"}[obj]
        if obj == "regression":
            X, gid = src.fortran(), None
        else:
            order, gid = split_queries(day, self.MAX_QUERY, seed)
            X, y = src.take(order).fortran(), y[order]
        pool = Pool(X, label=y, group_id=gid)
        # metric_period：YetiRank 的训练集 NDCG 只能在 CPU 上算，默认每 5 轮算一次
        model = CatBoost({"loss_function": loss, "task_type": "GPU", "iterations": rounds, "random_seed": seed,
                          "verbose": 0, "border_count": 128, "allow_writing_files": False,
                          "gpu_ram_part": self.GPU_RAM_PART, "metric_period": 100_000, **p})
        model.fit(pool)
        del pool, X
        return model

    def predict(self, model, src: TreeInput, rounds: int, start: int = 0, raw: bool = False) -> np.ndarray:
        kw = {"ntree_start": start, "ntree_end": rounds, "thread_count": 8}
        if raw:
            kw["prediction_type"] = "RawFormulaVal"
        return np.concatenate([np.asarray(model.predict(c, **kw), dtype=np.float64) for c in src.chunks()])


class LGB:
    name = "lgbm"

    def fit(self, params: dict, src: TreeInput, y: np.ndarray, day: np.ndarray, rounds: int, seed: int):
        import lightgbm as lgb

        class _Seq(lgb.Sequence):
            batch_size = CHUNK

            def __len__(self) -> int:
                return len(src)

            def __getitem__(self, i):
                if isinstance(i, slice):
                    return src.chunk(i.start or 0, len(src) if i.stop is None else i.stop)
                return src.chunk(i, i + 1)[0].astype(np.float64)   # 分箱抽样要求 double

        p = dict(params)
        obj = p.pop("objective")
        base = {"verbosity": -1, "seed": seed, "num_threads": LGB_THREADS, "max_bin": 255, "deterministic": True}
        if obj == "regression":
            base["objective"] = "regression"
            ds = lgb.Dataset(_Seq(), label=y, free_raw_data=True)
        else:
            base.update(objective="lambdarank", lambdarank_truncation_level=1000)
            _, counts = np.unique(day, return_counts=True)
            ds = lgb.Dataset(_Seq(), label=y, group=counts, free_raw_data=True)
        return lgb.train({**base, **p}, ds, num_boost_round=rounds)

    def predict(self, model, src: TreeInput, rounds: int, start: int = 0, raw: bool = False) -> np.ndarray:
        kw = {"start_iteration": start, "num_iteration": rounds - start, "num_threads": LGB_THREADS, "raw_score": raw}
        return np.concatenate([model.predict(c, **kw) for c in src.chunks()])


FAMILIES = {"xgb": XGB, "cat": CAT, "lgbm": LGB}


def inner_curve(fam, params: dict, md: ModelData, tr: Block, va: Block, horizon: str, max_rounds: int,
                eval_every: int, seed: int = 0) -> list[dict]:
    """在内层训练日上训练，返回每 eval_every 轮的内层验证日均 RankIC（前 1500 池）.

    分数按段累加原始分（第 r 轮的分数 = Σ 各段 [r − eval_every, r) 的树输出）：与一次预测 r 轮只差一个
    全体相同的常数（各段都含 base_score），日内排序不变；总预测量 = 一次 max_rounds 预测。
    """
    rows, y = train_target(tr, horizon, params["objective"])
    day = tr.meta["day"].to_numpy()[rows]
    model = fam.fit(params, TreeInput(md, tr, rows), y, day, max_rounds, seed)
    sv = TreeInput(md, va)
    yv = va.meta["y20" if horizon == "20d" else "y63"].to_numpy(dtype=np.float64)
    dv, mv = va.meta["day"].to_numpy(), va.meta["top1500"].to_numpy()
    curve, acc = [], np.zeros(len(sv))
    for r in range(eval_every, max_rounds + 1, eval_every):
        acc += fam.predict(model, sv, r, start=r - eval_every, raw=True)
        ic = daily_rank_ic(acc, yv, dv, mv)
        curve.append({"rounds": r, "ic_mean": float(np.mean(ic)), "n_days": int(len(ic))})
    return curve
