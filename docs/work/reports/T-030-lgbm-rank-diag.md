# T-030 · LightGBM lambdarank 内层 RankIC 为负

诊断日期 2026-10-17。只解释内层选择指标。不改已写出的预测，不改 `configs.py`，不改训练代码。这些数字不是预登记的主比较。

汇总文件：`docs/work/evidence/T-030/lgbm_rank_diag.json`。原始行在 `~/t030_scratch/logs/queue_lgbm/02_run_trees_py_quantmind_lgbm_20d.log`。

## 现象

LightGBM 网格是目标 {回归, 排序} × 叶子 {63, 255}，所以 c0、c1 是回归，c2、c3 是 lambdarank（`quantmind/research/gpu/configs.py`）。

20 日已完成内层选择的 12 折（折 0–11）[实测，上列日志与 json]：

- 两个排序配置共 24 个 best inner RankIC，均值 **−0.0306**，其中 18 个为负。
- c2（`num_leaves=63` 的 lambdarank）在 12 折里有 **10 折为负**。
- 同一批折的回归配置全部为正，13 个已完成回归内层（含正在进行的第 13 折的 c0、c1）均值 **0.1344**，0 个为负。
- 这 12 折选出的配置全是 c0 或 c1，也就是回归。写出的 LightGBM 预测用的是回归模型。

63 日 9 折的排序内层均值是 **+0.0062**（18 个里 5 个为负），回归均值 0.1482、没有负值。系统性为负的是 20 日，63 日只是接近 0。

同一套 0..19 分位等级，另外两族的排序目标在 20 日上为正 [实测，同一 json]：

- XGBoost `rank:pairwise`：19 折、76 个内层，均值 0.1178，1 个为负。
- CatBoost YetiRank：19 折、38 个内层，均值 0.1106，3 个为负。

XGBoost 63 日当前日志是 2026-10-15 续跑之后留下的，只含 3 折的内层行，不拿来代表全部 9 折。

## 组顺序

组边界和行顺序是对齐的。

[代码] `quantmind/research/gpu/data.py`：块内 `day` 用排序后的交易日做 `get_indexer`，取值 0..n−1；行按 `as_of` 排列。`train_target` 只丢掉标签缺失的行，剩下的行仍按日排好。

[代码] `quantmind/research/gpu/trees.py` 的 `LGB.fit`：`_, counts = np.unique(day, return_counts=True)`，再把 `counts` 交给 LightGBM 的 `group`。`np.unique` 按取值排序计数。对已经按日非降的 `day`，这个计数就是各日的游程长度，组的先后和行的先后一致。

[实测] `lgbm_rank_diag.json` 里的 `group_alignment_toy`：游程 `[3,1,4,2,2]` 删掉中间一行之后，`np.unique` 的计数 `[3,1,3,2,2]` 与剩余游程相同。

XGBoost 用逐行 `qid=day`，CatBoost 用 `split_queries` 把同一天排到一起。三族看到的是同一套等级、同一种按日分组。LightGBM 这一支没有把组的顺序写反。

## label_gain

[代码] `LGB.fit` 在排序分支只设置 `objective=lambdarank` 和 `lambdarank_truncation_level=1000`，没有设置 `label_gain`。等级来自 `grades()`：`floor(Φ(gauss-rank) × 20)`，再截到 0..19（`N_GRADES = 20`）。高收益对应高等级，方向和 XGBoost、CatBoost 相同。

[文档] LightGBM 4.6 的 `label_gain` 默认是 `0,1,3,7,15,...,2^30−1`，也就是第 i 档的增益为 `2^i − 1`。本环境是 lightgbm 4.6.0。不传这个参数时，保存的模型里该项为空，由库按最大标签补全。

[实测] 20 档时各档增益：0、1、31（第 5 档）、1023（第 10 档）、32767（第 15 档）、262143（第 18 档）、524287（第 19 档）。第 19 档相对第 10 档大约是 512 倍，相对第 1 档大约是 5×10^5 倍。lambdarank 的配对权重正比于增益差。内层用来选配置的指标是前 1500 池的等权 RankIC，每一对股票权重相同。

代码还把截断层从库默认的 30 改成了 1000。一天大约五千只股票时，当前排名 1000 名之外的位置不产生梯度。损失几乎只看最高几档能不能进当前列表的前部。

[实测] 一组构造数据（12 天、每天 500 只、信号很强、后 4 天评估，不是真实面板）：默认增益的 Spearman 均值 0.886，线性增益 `0..19` 是 0.893，默认增益且截断 30 是 0.830。强信号下默认增益没有把相关翻成负数。所以 20 日日志里的负号不能解释成标签方向写反，也不能解释成组边界写反。

## 结论

20 日 LightGBM 排序配置的内层 RankIC 均值为 −0.031，c2 在 12 折里有 10 折为负。同一等级在 XGBoost 排序和 CatBoost YetiRank 上为正。组顺序与行顺序一致。没设 `label_gain` 时，20 个等级用的是 `2^i−1`，再叠加截断层 1000，优化目标和等权 RankIC 对不上。回归配置在这些折上全为正，内层选择因此落在回归；已经写出的折就是这样来的，保持不动。网格保持不动。
