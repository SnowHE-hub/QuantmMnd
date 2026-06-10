# Progress Log — 短周期特异性因子（Step 2）

> 注：本仓库 progress 之前记录的是「数据湖建设」任务（已 ✅ 完成，存档见 git 历史 / data_lake_completion_report.md）。
> 自 2026-06-06 起本文件承接「短周期特异性因子」任务。

## 2026-06-06

### 承接上下文
- Step 1（诊断）完成：`docs/plans/wf_v2_diagnostics.md`。结论：12d v2 信号中性化后 IC +0.0139→+0.0017（retention 12%），≈88% 是行业/规模 tilt；方向 16 fold 翻 8 次（neutral regime 飘）。
- 本步（Step 2）：先做计划再实现。

### 调研（P0）✓
- 读 `alpha_prices_panel.parquet`：原始 OHLCV + adj_close + adj_factor；vol=手、amount=千元 → vwap=10×amount/vol（元）。
- 读 `weekly_panel.py`：as_of 周频 350 个，`merge_increment` 契约（MultiIndex 唯一键、列不碰撞、行数不变）。
- `daily_basic.parquet` circ_mv 做规模，PIT 按 trade_date==as_of。
- 中性化筛选逻辑可直接复用 `scripts/_diag_wf_v2.py`。

### 计划（P1）✓
- 写 `docs/plans/short_horizon_factor_plan.md`：A1 日内反转（剔隔夜）+ 隔夜分量 / A2 残差化反转·波动 / A3 量条件反转 / A4 日线代理已实现矩（标注弱代理）/ B WQ101+GTJA191 子集；含中性化 IC 筛选方法 + 结论模板。

### 当前状态
⏸ **停在评审门**：计划已呈交，等用户批准后才进入 P2 实现。

### 待办（获批后）
- P2 实现 short_horizon_factors.py（算子 + 因子 + 日频 PIT + 周频取值 + 增量 join）
- P3 中性化筛选表（raw/neut IC、neut ICIR、翻转率、最大相关）
- P4 结论与建议（并入重训 / 升级数据）
- P5（可选）wf_gate 复跑对比 v2

---

## 2026-06-06（晚）— Step 3：模型 Bake-off 设计

### 环境调研（关键发现）
- 本会话 Bash 工具默认 = **Git Bash/Windows**（python=WindowsApps torch-cpu、qlib 无）；GPU 计算须 `wsl -e bash -lic` 走 WSL。
- WSL conda quantmind：torch **2.11.0+cu128**、cuda_available=True、**GPU=RTX 5060 Laptop 8GB（sm_120）**、driver 595.79。lightgbm 4.6、statsmodels 0.14.6。
- ⚠ 任务书写 "16GB Ti"，**实为 8GB 笔记本卡** → 显存减半，已据此设计 batch/模型。
- **qlib 未安装** + env 是 numpy2/pandas2.3 → 设计采用**双 env 隔离**（新建 qlib_bakeoff，只训练+导出预测；评估留在 quantmind）。

### 设计文档（P1'）✓
- 写 `docs/plans/model_bakeoff_plan.md`：双 env 架构 + 桥（预测 parquet）/ 特征双制式（tab=35+短周期+Alpha158；seq=Alpha360 60日 CSZScoreNorm）/ 双周期（12d H12E20、63d H63E63）/ refit≙cutoff 间距（季度+月度）/ 各模型 8GB 配置 / fold 驱动训练（用我们 PurgedWalkForwardSplit，非 qlib rolling）/ 评估管线（raw+neut IC+purge inflation+含成本净超额，G2 复现 v2）/ leaderboard / 分阶段网格（批 A pilot→B 季度全网格→C 月度仅 baseline+优胜者）/ 风险 / 评审点 R1–R4。

### 当前状态（设计期）
⏸ 设计文档已呈交。

---

## 2026-06-06（执行）— Bake-off 获批，进 P0 + R3

### 评审通过，决策记录写入 model_bakeoff_plan.md
- R1 八留五模型/8GB；R2 双 env 隔离首选+探针；R3 short_horizon 先跑/并行；R4 月度=baseline+全 contender；R5 L=60/seed=5；R6 排序(neut IC,净超额)。
- 3 修订：63d=量价基线非产品验证 / 选择乐观偏差解读规则 / 可选图模型(§13)。
- 评审门翻为 ✅ 已批准；获批范围内不再停，除非 G1/G2/R2 探针失败。

### P0（进行中）建 qlib_bakeoff env + R2 兼容性探针
- WSL: conda 26.1 + mamba + gcc 13.3（cython 可编）；GPU sm_120 / torch 2.11+cu128。
- 写 `scripts/bakeoff/p0_env_setup.sh`：mamba create qlib_bakeoff(py3.11) → numpy1.26/pandas<2.2 + pyqlib → torch cu128 → 探针(pyqlib import + GRU GPU AMP 训练步)。
- **后台运行中**（nohup pid 53765），日志 `scripts/bakeoff/p0_env_setup.log`，结果 `p0_probe_result.json`。

### R3（并行）short_horizon 因子筛选 — 进行中
- 定稿口径：~31 候选（A 类 19 + B 代表 12）、市场项 cap 加权截面均值、保留线 |neut ICIR|≳0.2 & 翻转<35% & corr<0.7、选因子用 in-sample(pre-2022) neut IC、OOS(2022+) 无偏确认。
- 写 `quantmind/features/short_horizon_factors.py`（wide 算子 + 残差日收益 + 31 因子 + sample_at_asof）+ `scripts/screen_short_horizon.py`。smoke(120d) 通过。
- 全量筛选**后台运行中**（Windows python，输出 block-buffered，查 artifacts：short_horizon_screen.csv / short_horizon_factors.parquet）。

### P0 完成 ✅ — R2 = real_qlib
- pyqlib 0.9.7 / numpy1.26 / pandas2.1.4 / torch 2.11+cu128；探针：qlib import OK、GRU 在 **sm_120 GPU** AMP 训练步 OK → **R2 走真 pyqlib，不退 vendor**。quantmind env 未动。

### 验收意见处理（2026-06-06）
- ①B 不缩水：B 补回 28（+算子 ts_argmax/signedpower/ts_cov + 16 因子），总 47=A19+B28；登记排除项（IndNeutralize/decay 按 scope 排除，非缺字段）。
- ②成本 holding-period-aware：evaluate_bakeoff 改非重叠持有期再平衡 + 真实 membership 换手；G2 仍过；v2 含成本净超额 −8.4%（旧全换手 −12.3%）。记入 plan §6.3 + task_plan。

### R3 完成 ✅ — 16 survivors（贪心去冗余后）
- 47 候选 screen（in-sample 选 / OOS 无偏确认），`short_horizon_final.csv` + survivors.json。
- **结论：日线量价 YES——有中性化后稳定为正(或稳定负、用符号)且与现有35低冗余的因子。**
- 核心存活（in-sample neut ICIR | OOS neut ICIR）：wq13_cov_close_vol(0.87|0.58)、wq44_corr_high_rkvol(0.77|0.58)、wq3_rkopen_rkvol(0.68|0.49)、wq15/wq6(量价相关族)、resid_vol_10(−0.65|−0.93)、rskew_21(−0.65|−0.72)、resid_rev_10(0.41|0.29)。
- 主题：**量价 rank 相关/协方差族**（35 因子里没有）+ 残差化波动/反转 + 已实现偏度弱代理。强信号 OOS 同号确认；弱的几个(wq5/wq42/wq2)OOS 近零。
- 注：intraday_rev_21 等虽 OOS neut IC 强(+0.040)，但 maxcorr_base>0.7（与现有 reversal 重复）→ 正确硬毙。
- survivors 经 merge_increment 进 tabular（P3 tabular 用 35 + 16 survivors + Alpha158）。

### ⭐ P3 关键早读（最高价值单点）：survivors 在模型层兑现
**LGBM-tabular(35 + 16 survivors) 12d vs v2(35)**：
- 中性化 IC **+0.0017 → +0.017（~10×）**；中性化 ICIR **0.029 → 0.33**；neut posfrac 0.57→0.65；
  最大回撤 21%→9%；含成本净超额 −8.4%→−3.4%；翻转 8→7/16。
- → R3 因子研究**在模型层确实兑现**：量价 rank 相关/协方差 + 残差化族带来真正交 alpha。
- 还未加 Alpha158（full 在跑）；仍幸存者乐观上界、1 配置；neut IC 0.017 仍 < gate 0.03 线但轨迹强正。

### P3 基建（已验证）
- dump_bin → Alpha158/360 handler（G1 全过）；Alpha158 抽到 as_of（452347×158）。
- tabular harness `p3b_tabular.py`（lgbm=v2 口径 LGBMPredictor / ridge 基线；featset v2_35/plus_surv/full）。
- sequence harness `p3c_sequence.py`（qlib GRU/LSTM/Transformer on Alpha360，GPU）—— GRU smoke 通过（val IC 0.033）。修 mlflow3.13 filestore：`MLFLOW_ALLOW_FILE_STORE=true` + R.start。
- leaderboard `p3d_leaderboard.py`（含 v2 对照行，排序 neut_ic/net_excess）。
- 在跑：tabular ladder(lgbm full/ridge full/lgbm v2_35) + GRU pilot。

### ⭐⭐ 首张 leaderboard（batch-A pilot tabular，12d 季度）—— 管线已验证
| 模型 | 特征 | neut IC | neut ICIR | 含成本净超额 | maxDD | 胜率 |
|---|---|---|---|---|---|---|
| **Ridge** | full(35+16+158) | **+0.034** | **0.64** | **+1.9%** | 4.8% | 52% |
| LGBM | +surv(35+16) | +0.017 | 0.33 | −3.4% | 9% | 42% |
| LGBM | full(35+16+158) | +0.015 | 0.29 | −7.3% | 19% | 31% |
| LGBM(v2) | 35 | +0.0017 | 0.03 | −8.4% | 21% | 35% |
**发现**：(1) survivors 兑现：35→35+16 把 LGBM neut IC 拉 10×。(2) **Ridge(full) 最强**：neut IC 0.034 > gate 0.03 线、**净超额转正 +1.9%**、ICIR 0.64——诚实线性基线在富特征上碾压 LGBM。(3) **LGBM full < LGBM plus_surv**：158 Alpha158 噪声列让树过拟合，L2 线性反而吃得下高维。
**caveat**：幸存者乐观上界 + 单配置 + 跨模型搜索（§8.1 选择偏差）+ 成本是 proxy；**pilot=管线验证非最终判定**；深度模型(GRU/LSTM/Transformer)在跑，真读数等 batch B 5seed。
- 自检待补：per-fold purge-ablation（split 级已在 v2 验 inflation>0，全模型共享同一 split）。

### ⭐⭐⭐ Ridge(full) 硬化（CPU，不重训，p3e_harden.py）—— 不是 v2 的鬼魂
4 项硬化全过：
- **纪律(check4)**：alpha=10 固定先验(无选择→无 OOS 窥)、fit 仅 train、标准化 μ/σ 仅 train、方向 decide_direction 仅 val（H-A，model-agnostic）。干净。
- **(1) regime 条件 IC（决定性）**：bear +0.034 / bull +0.041 / **neutral +0.028（posfrac 0.75）**。对比 v2 neutral **−0.007**（震荡市方向飘）。→ Ridge **三档全正、震荡市也正** = 真特异 alpha，非 regime tilt 换脸。
- **(2) 逐 fold**：**16/16 fold 全正**，mean +0.036 std 0.026，去 top-1 → +0.034、去 top-2 → +0.031。宽基础，非 1-2 fold 顶起。
- **(3) 早/晚段**：early(22-24) +0.038、**late(25-26) +0.026（posfrac 0.74）**。out-of-selection 仍立。
对比 LGBM(35+16)：bear 仅 +0.005(posfrac0.47)、12/16 fold 正——远不如 Ridge 稳健。
**结论**：Ridge(full) 是宽基础、regime 稳健、时段稳健的特异 alpha。仍有 caveat：幸存者上界 + 4选1冠军(check2/3 为 out-of-selection 缓解) + proxy 成本 + 完整 held-out 留 batch B。
**深度模型关键假设**：seq(Alpha360) 中性化 IC 能否 > Ridge(full) +0.034 —— 待 GRU/LSTM/Transformer 跑完。

### 深度模型 pilot 排障（多次"失败"全非建模问题）
- 崩因排查：①原始链 fold1 OOM(WSL 12GB cap + qlib 12 kernel 内存翻倍) ②inline memmon 引号污染启动 shell ③Bash 工具 120s 超时腰斩同步跑 ④WSL2 实例 idle 时拆掉、杀 setsid/nohup 子进程。全部**非 OOM/非建模**。
- **三处真修复**：
  1. **cache-once**：Alpha360 handler 全范围建一次、按 fold 重切（CSZScoreNorm 逐日 fit-free→复用无泄漏）。fold1 仅 +40s（不再重载）→ 消除内存累积 + 提速 ~8×（每模型 ~12min 而非 ~112min）。
  2. **Fillna 修复**：自定义 processors 漏了 ProcessInf+Fillna → inf/0方差日特征→NaN→**预测全NaN→preds 空**。补 ProcessInf→CSZScoreNorm→Fillna 后 score 正常、preds(10335,3)。
  3. **可靠启动**：harness-tracked 后台任务保活 wsl.exe（230s 测试通过）→ 不再被 WSL 拆。
- 现状：全链 GRU/LSTM/Transformer(16 fold each, 12d 季度 s0) harness 后台跑中(bokebjq5l)。**新风险**：全范围(2019-2026)一次性 build ~7GB，逼近 12GB cap，需盯内存；若 OOM 退 kernels=1 或流动性子集。

### 深度 pilot 内存设计（用户实测辅助定档）
- 实测：全范围 cache-once OOM(7.3年×1374~14GB)；per-fold×1374 ~10GB(逼近12GB,弃)；per-fold×700 WSL峰值 **8.4GB**、主机90%(太顶)。
- **定档：per-fold + 流动性子集 N=500**（按日均 amount 中位数选 top500）→ WSL峰值~6GB、主机~85%；fold间 gc 已验证回落(6.3→3.4GB)不累积；kernels=2。
- VRAM 非瓶颈(batch800≈0.6GB<8GB)。preds 已确认非空(Fillna 修复)。
- 公平性：深度用 500 子集，评估端把 tabular 预测筛到同 500 票做 head-to-head（不重训），存 data/bakeoff/deep_universe.json。
- 全链 GRU/LSTM/Transformer(16fold,12d季度s0,N=500) harness 后台跑中(bhm4qao4r)，每模型落盘断点。

### 深度 pilot 最终方案：流动性分桶（用户选 C，全 1374 覆盖）
- 1374 按日均 amount 中位数降序切 **3 桶各 ~458**（桶0最活跃…桶2最不活跃）→ 每桶单折 WSL 峰值 ~6.5GB（主机~72%，安全）。
- 3 模型 × 3 桶 = 9 训练（各 16 fold，~5-6h）。p3c 加 QM_N_BUCKETS/QM_BUCKET；输出 `<model>_alpha360_12d_quarterly_s0_b{b}of3.parquet`；桶成员存 data/bakeoff/deep_buckets/。
- 合并评估 `p3f_bucket_eval.py`：seq 3 桶【桶内 as_of rank→[0,1]】拼成全 1374 算 neut IC（消尺度差）+ 分桶报；tabular（全1374）全+分桶。对决：seq full neut IC vs Ridge(full) +0.034。
- 9 桶链 harness 后台跑中(bsgrbo0oj)，memmon 盯 RAM。

### ⚠ 进程重叠 bug + 修复（2026-06-07 晚）
- 现象：memmon 峰值冲到 **11847MB**（差点爆 12GB），用户监控到内存吃紧。
- 根因：`pkill -f p3c_sequence` 只杀 python，**外层 for-loop bash 没死**→ 继续 spawn 下个模型；旧 500 链残留 loop + 新 9-桶链**并发**两个 p3c → 内存翻倍。
- 修复：① `wsl --shutdown` 彻底清场（env/数据在磁盘，不丢）② 链改成脚本文件 `_run_chain9.sh`（可 `pkill -f _run_chain9.sh` 连 bash loop 一起杀）③ 用 `bash -c`（非 -lic）去 tty 乱码。
- 重启后单链确认：1 个 main p3c(PID6694) + 2 loky worker(kernels=2 继承 cmdline，易误数)，WSL used **~6GB**、free 7.4GB → 单跑安全。
- 教训：停链必须连 for-loop bash 一起杀；监控看 `free` 实际内存而非进程数（会被 self-match/worker 干扰）。

### 自驱动桶接力（用户批准全跑 9 桶 + 预批权限 + 过夜无人值守）
- 根因：harness 后台任务 **~2 小时寿命上限**（非 OOM，峰值 8GB）→ 单任务塞 9 桶必被腰斩。
- 方案：**一桶一任务**（`_run_one.sh <model> <bucket>`，~84min<2h，完整落盘）；`_next_bucket.sh` 幂等判定下一个；每次唤醒/通知执行 driver（RUNNING→等；NEXT→启动+查内存+机会性 p3f；ALL_DONE→最终 leaderboard+harden+batchB）。
- 顺序：gru 0✅/1(跑)/2 → lstm 0/1/2 → transformer 0/1/2。每桶查 WSL used<9.5GB。
- **首信号（桶0 最活跃458）**：GRU(seq) neut IC **+0.0057** vs Ridge(full) **+0.0319** / LGBM(35+16) +0.0183 / v2 +0.0051 → **序列模型在原始 Alpha360 上未赢手工因子**（GRU/1seed/1桶，待 LSTM/Transformer+全桶确认）。

### ⚠ 过夜接力停滞 + 修复（06-08 早）
- 现象：一夜只推进 1 个桶（gru b1）。gru b1 **计算 01:12 就跑完落盘**（90min，63934行），但 harness 任务直到 **09:37** 才报完成——**挂了 8.5 小时**。
- 根因：qlib 的 **loky fork worker（kernels=2）在主 python 退出后残留**（"leaked semlock"），挂住 wsl.exe → 任务一直"running" → **同时阻塞了完成通知和定时唤醒** → 接力无法前进。
- 修复：`_run_one.sh` 末尾 `pkill -9 -f "p3c_sequence.py --model $M"` 杀残留 worker + `exit 0` 立即退出 → 任务计算完即报完成 → 接力恢复。
- 状态（09:37）：gru b0✅ b1✅，**gru b2 重启跑中**（含修复）。剩 gru2+lstm3+transformer3 = 7 桶 × 90min ≈ 10.5h。

### ✅ 健壮架构：脱离式编排器（替代会挂的 harness 接力）
- 教训：harness 后台任务会 ① 2h 寿命上限 ② loky 残留挂死、挂死时**堵住 agent 的唤醒/通知** → 一卡一夜。
- 新方案：`_orchestrator.sh` 单进程顺序跑完所有未完成桶（可断点续跑、loky 每桶清理、每模型出 p3f），`setsid nohup` **完全脱离**运行（非 harness 任务，无 2h 上限、不堵唤醒）。已验证存活过 20s 危险期。
- agent 只做**秒级轮询** `_orch_check.sh`（ALIVE/DEAD/ALL_DONE + 内存 + 已完成桶数）：DEAD 且未完→重启编排器（自动续跑）；ALL_DONE→出终版 leaderboard。
- ⚠ pkill 自匹配教训：inline `pkill -f _run_one.sh` 会杀自己的 shell（命令串含该字符串）→ 杀进程要用脚本文件或避开自匹配模式。
- 09:47 编排器启动，跳过 gru b0/b1，跑 gru b2 中。剩 7 桶 ~10.5h。

### ⭐ GRU 全1374 结果（3桶合并，06-08 11:30）— 比"完败"精确
| 模型 | 全1374 neut IC | ICIR | b0活跃 | b1中 | b2不活跃 | 含成本净超额 |
|---|---|---|---|---|---|---|
| Ridge(full) | 0.0340 | 0.64 | 0.032 | 0.032 | 0.034 | +1.9% |
| **GRU(seq,3桶)** | **0.0281** | **0.66** | 0.005 | 0.020 | **0.053** | **−2.9%** |
| LGBM(35+16) | 0.0170 | 0.33 | 0.018 | 0.020 | 0.003 | −3.4% |
| LGBM(v2 35) | 0.0017 | 0.03 | | | | −8.4% |
- **GRU neut IC +0.0281 ≈ Ridge +0.0340**（ICIR 略高），远超之前只看活跃桶的 +0.0057 → 序列模型抓到真信号、甩开 LGBM。
- **信号全在 illiquid（b2 +0.053 反超 Ridge）**：序列模型在低流动性票上挖到手工因子漏掉的时序模式；活跃票上无优势（b0 +0.005）。→ 印证全桶覆盖的必要性。
- **但含成本 −2.9%**：超额集中在最不可交易的票上，滑点吃光 → 不可收割。Ridge 仍是可交易赢家。
- 初步结论："序列模型能挖到手工因子漏掉的 alpha——能，但在 illiquid 角落、成本不可收割"。待 LSTM/Transformer 确认。
- bug 修复：p3f rename score_col 与既有 score 列碰撞致 2D → 改为 `p["score"]=preds[score_col].values`。p3f 评估须用 quantmind/Windows python（有 diskcache），非 qlib_bakeoff env。

### ⭐ LSTM 全1374（06-08 18:20）— 确认 GRU 发现（2/3 seq 一致）
| 模型 | 全1374 neut IC | ICIR | b0活跃 | b1中 | b2不活跃 | 净超额 |
|---|---|---|---|---|---|---|
| Ridge(full) | 0.0340 | 0.64 | 0.032 | 0.032 | 0.034 | +1.9% |
| **LSTM(seq)** | **0.0316** | 0.63 | 0.004 | 0.015 | **0.075** | −1.8% |
| GRU(seq) | 0.0281 | 0.66 | 0.005 | 0.020 | 0.053 | −2.9% |
| LGBM(35+16) | 0.0170 | 0.33 | | | | −3.4% |
- LSTM neut IC +0.0316 更逼近 Ridge +0.0340；illiquid 桶 b2=+0.075（>GRU 0.053，>Ridge 0.034 2倍+）。两 seq 同一模式→稳健真信号。
- 含成本 −1.8%（优于 GRU −2.9% 但仍负）。结论：seq 在 illiquid 挖到 Ridge 拿不到的时序 alpha，当前成本下不可收割；Ridge 仍唯一净正。
- follow-up：illiquid seq alpha(~0.07) 若用更长持有期/耐心执行压成本，可能可挖。
- PC 重启续跑：6/9(gru+lstm)落盘无损，编排器跳过已完成、续跑 transformer。

### ✅✅ Batch-A 全完（9/9，2026-06-09 04:14）— 终版 leaderboard
排序（全1374 neut IC）：Ridge(full) **0.0340**(+1.9%净) > LSTM 0.0316(−1.8%) > GRU 0.0281(−2.9%) > Transformer 0.0235(−5.1%) > LGBM(35+16) 0.0170 > LGBM_full 0.0146 > v2 0.0017。
- **没有序列模型超过 Ridge(full)**；但 3 seq 全碾压 LGBM。**Ridge 唯一净正**。
- 3 seq 一致：活跃桶弱/负(b0 ~0)、不活跃桶强(b2 0.05-0.075 全 > Ridge 0.034)。
- 结论：序列模型确从原始量价挖到手工因子漏掉的时序结构，但**只在 illiquid、净不可收割**。Ridge(full) 仍是可交易赢家。
- ⚠ 收尾修正（用户定案）：① 序列模型"全1374 neut IC"是 **3 桶拼接**（未评审偏离，加†标，不与 Ridge 单模型混比；偏离若有影响是抬高 seq）。② 桶2 的 0.05-0.075 **最不可信**：illiquid=小盘/ST/易退市=幸存者偏差最重 × 成本最高的双重角落 → **不是 follow-up 线索**。
- 详见 model_bakeoff_plan.md §14。**阶段关闭**；下一件事=幸存者修复；修好后第一步=真实 universe 重跑 Ridge(full) 看 +0.034/+1.9% 是否还在。63d/batch B/illiquid 全部等幸存者修好。
- 运维：脱离编排器架构成功跑完全部（含 1 次 PC 重启断点续跑、loky 清理、内存全程 <8GB）。

### 2026-06-09 — Bake-off 关闭 + 幸存者修复计划
- batch-A 定案 Ridge(full) 赢；§14 收尾修正：序列模型全1374=3桶拼接(†标) + illiquid 最不可信(幸存者+成本双重角落，非线索)。
- 调研幸存者根因：stock_basic 1388 全 L、delist 全空（纯幸存者）+ 只 1137 票 2019前（是 SH+SZ 子样本，非全市场）。
- 写 `docs/plans/survivorship_repair_plan.md`：scope 决策(A全市场PIT/B样本+退市) + 拉 stock_basic L/D/P + 退市票在市期价量(复用 backfill) + PIT 重建(list≤as_of<delist) + v6 面板(不动 v5/评估零改) + 量化 + 验收。
- ⏸ **停在评审门**：等用户拍板 scope。修好后第一步=v6 重跑 Ridge(full) 看 +0.034/+1.9% 是否还在（决定有没有产品）。63d/batch B/illiquid 全部推迟到幸存者修好后。

### P1 完成 ✅ — dump_bin + G1 PASS
- 下载官方 dump_bin.py(542行)→ scripts/bakeoff/；写 `scripts/bakeoff/p1_dump_bin.py`：alpha_prices_panel → 后复权 OHLC/vwap(raw×adj_factor)+vol+factor+amount → qlib bin（data/qlib_cn_daily）；symbol 000001.SZ→SZ000001。
- G1 硬门：$close 还原一致 + Alpha158/360 handler 非空 + PIT。**后台运行中**（qlib env，pid 82969，日志 p1_dump.log，结果 p1_g1_result.json）。

### P2 评估 harness（提前做，独立于 GPU）✓ + **G2 通过**
- 写 `scripts/bakeoff/evaluate_bakeoff.py`：score_predictions（raw/中性化 IC + 含成本净超额 + DD/换手）+ g2_selfcheck。
- **G2 自检通过**：复现 v2 raw +0.0139 / neut +0.0017（n=142）→ 中性化口径与 _diag_wf_v2 一致，可接深度模型预测。
- 注：含成本净超额 = -12.3%（v2 的 -2.65% 是税前/pre-cost）；全 rotation 成本假设保守，统一施加于所有模型→相对排名公平；后续可改持有期感知换手。
