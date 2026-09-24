#!/usr/bin/env bash
# scripts/bakeoff/p0_env_setup.sh — P0：建 qlib_bakeoff env + R2 兼容性探针。
# 不动 quantmind env。日志落 scripts/bakeoff/p0_env_setup.log，结构化结果落 p0_probe_result.json。
set -uo pipefail
cd /home/lenovo/projects/quantmind
LOG=scripts/bakeoff/p0_env_setup.log
RES=scripts/bakeoff/p0_probe_result.json
ENV=qlib_bakeoff
exec > >(tee "$LOG") 2>&1
echo "==== P0 env setup start $(date -Is) ===="

source /home/lenovo/miniforge3/etc/profile.d/conda.sh

if conda env list | grep -qE "^${ENV}\s"; then
  echo "[env] $ENV already exists, reuse"
else
  echo "[env] creating $ENV (python 3.11)"
  mamba create -y -n "$ENV" python=3.11
fi

PY="/home/lenovo/miniforge3/envs/${ENV}/bin/python"
PIP="$PY -m pip"
echo "[py] $($PY --version)"

echo "==== install pyqlib (lets it resolve numpy/pandas; classic qlib wants numpy<2) ===="
# pandas 2.2.3：qlib 0.9.7 用 'ME' 频率别名（需 pandas>=2.2）；numpy 1.26 保持（qlib 要 <2）
$PIP install --upgrade pip setuptools wheel cython numpy==1.26.4 "pandas==2.2.3"
# pyqlib builds Cython ext at install; gcc present
$PIP install pyqlib 2>&1 | tail -40 || echo "[warn] pyqlib install returned nonzero"

echo "==== install torch cu128 (sm_120 Blackwell) ===="
$PIP install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -20 || echo "[warn] torch install nonzero"

echo "==== R2 compatibility probe ===="
$PY - <<'PY'
import json, traceback
res = {"stage": "probe"}
def ver(m):
    import importlib
    try:
        mod = importlib.import_module(m); return getattr(mod, "__version__", "?")
    except Exception as e:
        return f"MISSING({type(e).__name__})"
res["numpy"] = ver("numpy"); res["pandas"] = ver("pandas")
res["torch"] = ver("torch"); res["qlib"] = ver("qlib")
# torch cuda
try:
    import torch
    res["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        res["device"] = torch.cuda.get_device_name(0)
        res["capability"] = list(torch.cuda.get_device_capability(0))
        res["torch_cuda"] = torch.version.cuda
except Exception as e:
    res["torch_err"] = repr(e)
# qlib import + init
try:
    import qlib
    res["qlib_import"] = True
except Exception as e:
    res["qlib_import"] = False
    res["qlib_import_err"] = repr(e)[:300]
# tiny GRU train step on GPU under AMP (validates sm_120 kernels)
try:
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = nn.GRU(6, 16, num_layers=2, batch_first=True).to(dev)
    head = nn.Linear(16, 1).to(dev)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), 1e-3)
    x = torch.randn(64, 60, 6, device=dev)
    y = torch.randn(64, 1, device=dev)
    use_amp = (dev == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast(device_type="cuda" if dev=="cuda" else "cpu", enabled=use_amp):
            out, _ = m(x); pred = head(out[:, -1, :]); loss = ((pred - y) ** 2).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    res["gru_gpu_trainstep"] = True
    res["gru_final_loss"] = float(loss.detach().cpu())
    res["gru_device"] = dev
except Exception as e:
    res["gru_gpu_trainstep"] = False
    res["gru_err"] = repr(e)[:300]
    res["gru_tb"] = traceback.format_exc()[-600:]
# verdict
res["R2_path"] = ("real_qlib" if res.get("qlib_import") and res.get("gru_gpu_trainstep")
                  else "vendor_fallback")
print("PROBE_RESULT_JSON_BEGIN")
print(json.dumps(res, ensure_ascii=False, indent=2))
print("PROBE_RESULT_JSON_END")
import pathlib; pathlib.Path("scripts/bakeoff/p0_probe_result.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
PY
echo "==== P0 env setup done $(date -Is) ===="
