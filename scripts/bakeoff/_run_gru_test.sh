#!/usr/bin/env bash
# 2-fold GRU 内存修复验证（kernels=2 + fold间 gc）。setsid 完全脱离，免被父会话杀。
cd /home/lenovo/projects/quantmind
export QM_MAX_FOLDS=2
exec /home/lenovo/miniforge3/envs/qlib_bakeoff/bin/python -u \
  scripts/bakeoff/p3c_sequence.py --model gru --period 12d --refit quarterly --seed 0
