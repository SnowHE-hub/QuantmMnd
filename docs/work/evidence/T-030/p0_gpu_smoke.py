"""T-030 P0：sm_120 上的 GPU 闭环（前向 / 反向 / 保存 / 加载 / 预测，加载后预测须与保存前逐值相同）.

torch（含 BF16 autocast）、TabM、XGBoost（device=cuda：回归、rank:pairwise、ExtMemQuantileDMatrix 外存迭代器）、CatBoost（task_type=GPU）。
合成数据，不读任何项目数据。用法：python p0_gpu_smoke.py <out.json>
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
res: dict = {}
rng = np.random.default_rng(0)
N, F = 20_000, 64
X = rng.standard_normal((N, F)).astype(np.float32)
y = (X[:, :4].sum(1) + 0.5 * rng.standard_normal(N)).astype(np.float32)
qid = np.repeat(np.arange(N // 500), 500)
tmp = Path(tempfile.mkdtemp())


def run(name, fn):
    t0 = time.time()
    try:
        r = fn()
        r["ok"] = True
    except Exception as e:  # noqa: BLE001
        r = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-1500:]}
    r["seconds"] = round(time.time() - t0, 2)
    res[name] = r
    print(name, json.dumps(r)[:300], flush=True)


def t_torch():
    import torch
    dev = "cuda"
    m = torch.nn.Sequential(torch.nn.Linear(F, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev)
    for _ in range(50):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((m(xt).squeeze(-1).float() - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p0 = m(xt).squeeze(-1).cpu().numpy()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, tmp / "m.pt")
    m2 = torch.nn.Sequential(torch.nn.Linear(F, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)).to(dev)
    m2.load_state_dict(torch.load(tmp / "m.pt")["model"])
    m2.eval()
    with torch.no_grad():
        p1 = m2(xt).squeeze(-1).cpu().numpy()
    return {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "final_loss": float(loss), "reload_max_abs_diff": float(np.abs(p0 - p1).max()),
            "corr_pred_y": float(np.corrcoef(p0, y)[0, 1]), "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def t_tabm():
    import torch
    from tabm import TabM
    dev = "cuda"
    m = TabM.make(n_num_features=F, cat_cardinalities=[], d_out=1).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev)
    for _ in range(30):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = m(xt).squeeze(-1).float()          # (N, k)
            loss = ((p - yt[:, None]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p0 = m(xt).squeeze(-1).float().mean(1).cpu().numpy()
    torch.save(m.state_dict(), tmp / "tabm.pt")
    m2 = TabM.make(n_num_features=F, cat_cardinalities=[], d_out=1).to(dev)
    m2.load_state_dict(torch.load(tmp / "tabm.pt"))
    m2.eval()
    with torch.no_grad():
        p1 = m2(xt).squeeze(-1).float().mean(1).cpu().numpy()
    return {"k": int(p.shape[1]), "reload_max_abs_diff": float(np.abs(p0 - p1).max()),
            "corr_pred_y": float(np.corrcoef(p0, y)[0, 1])}


def t_xgb():
    import xgboost as xgb
    r = {"version": xgb.__version__}
    d = xgb.QuantileDMatrix(X, y)
    b = xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 6, "eta": 0.1}, d, 100)
    p0 = b.inplace_predict(X)
    b.save_model(tmp / "x.ubj")
    b2 = xgb.Booster()
    b2.load_model(tmp / "x.ubj")
    p1 = b2.inplace_predict(X)
    r["reg_reload_max_abs_diff"] = float(np.abs(p0 - p1).max())
    r["reg_corr"] = float(np.corrcoef(p0, y)[0, 1])
    lab = np.zeros(N, dtype=np.float32)
    for q in np.unique(qid):
        m = qid == q
        lab[m] = np.floor(np.argsort(np.argsort(y[m])) / m.sum() * 5)
    dr = xgb.DMatrix(X, lab, qid=qid)
    br = xgb.train({"device": "cuda", "tree_method": "hist", "objective": "rank:pairwise", "max_depth": 6, "eta": 0.1}, dr, 50)
    r["rank_corr"] = float(np.corrcoef(br.predict(dr), y)[0, 1])

    class It(xgb.DataIter):
        def __init__(self, k):
            self.k, self.i = k, 0
            super().__init__(cache_prefix=str(tmp / "cache"))

        def next(self, input_data):
            if self.i == self.k:
                return False
            s = slice(self.i * N // self.k, (self.i + 1) * N // self.k)
            input_data(data=X[s], label=y[s])
            self.i += 1
            return True

        def reset(self):
            self.i = 0

    try:
        de = xgb.ExtMemQuantileDMatrix(It(4), max_bin=256)
        be = xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 6, "eta": 0.1}, de, 50)
        r["extmem_corr"] = float(np.corrcoef(be.inplace_predict(X), y)[0, 1])
        r["extmem_ok"] = True
    except Exception as e:  # noqa: BLE001
        r["extmem_ok"] = False
        r["extmem_error"] = repr(e)[:400]
    return r


def t_cat():
    import catboost
    from catboost import CatBoostRanker, CatBoostRegressor, Pool
    r = {"version": catboost.__version__}
    m = CatBoostRegressor(iterations=200, depth=6, task_type="GPU", devices="0", verbose=0, random_seed=0)
    m.fit(X, y)
    p0 = m.predict(X)
    m.save_model(tmp / "c.cbm")
    m2 = CatBoostRegressor()
    m2.load_model(tmp / "c.cbm")
    p1 = m2.predict(X)
    r["reg_reload_max_abs_diff"] = float(np.abs(p0 - p1).max())
    r["reg_corr"] = float(np.corrcoef(p0, y)[0, 1])
    rk = CatBoostRanker(iterations=100, depth=6, loss_function="YetiRank", task_type="GPU", devices="0", verbose=0, random_seed=0)
    rk.fit(Pool(X, y, group_id=qid))
    r["yetirank_corr"] = float(np.corrcoef(rk.predict(X), y)[0, 1])
    qr = CatBoostRanker(iterations=100, depth=6, loss_function="QueryRMSE", task_type="GPU", devices="0", verbose=0, random_seed=0)
    qr.fit(Pool(X, y, group_id=qid))
    r["queryrmse_corr"] = float(np.corrcoef(qr.predict(X), y)[0, 1])
    return r


TESTS = {"torch": t_torch, "tabm": t_tabm, "xgboost": t_xgb, "catboost": t_cat}
if len(sys.argv) > 2:                      # 子进程：只跑一个库（同一进程里先建 torch CUDA 上下文会让 XGBoost 报显存为 0）
    run(sys.argv[2], TESTS[sys.argv[2]])
    out.write_text(json.dumps(res[sys.argv[2]]), encoding="utf-8")
else:
    import subprocess
    for name in TESTS:
        part = out.with_suffix(f".{name}.part")
        subprocess.run([sys.executable, __file__, str(part), name], check=False)
        res[name] = json.loads(part.read_text()) if part.exists() else {"ok": False, "error": "subprocess died"}
        part.unlink(missing_ok=True)
    res["note"] = "每个库在单独的子进程里测试"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
