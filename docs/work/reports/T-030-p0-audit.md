# T-030 · P0 审计：GPU 闭环、日频面板、一致性、Ridge C 复现、资源实测

- 分支 `t/030-gpu-models`（worktree `~/projects/qm-t030`）；环境 `qm-gpu`（从 `quantmind` 克隆，加装 catboost / tabm / cupy-cuda12x；生产 env 未动）
- 证据目录：`docs/work/evidence/T-030/p0/`（JSON 与构建日志副本）；数据产物 `data/gpu_v1/`（只新增，未覆盖任何已有文件）
- 本文件只含 P0；**不含任何外层样本外结果**（P1 预登记在本文件之后提交）。

## 1. 结论

| 项 | 结果 | 证据 |
|---|---|---|
| GPU 栈（torch / XGBoost / CatBoost / 6 个网络） | 全部通过 `[实测]` | `p0/gpu_smoke.json`、`p0/gpu_retest.json` |
| 外层折与基线 `.dirs.json` 一致 | 20d 19 折 224 个测试 as_of；63d 9 折 200 个 `[实测]` | `p0/check_folds.json` |
| §3.2 日频构建器 vs v6_1 周频文件（7 组） | `all_ok = true` `[实测]` | `p0/consistency_v2.json` |
| Ridge C 逐值复现（20d、63d） | 行集合相同；最大绝对差 7.6e-12 / 3.6e-12；方向相同 `[实测]` | `p0/ridge_repro_{20d,63d}.json` |
| 内存 / 显存 / 耗时 | 每族最大单元峰值 RSS ≤ 5.1 GB、显存 ≤ 6.5 GB（见第 6 节） `[实测]` | `p0/resource_probe_v3.json`、`p0/resource_probe_v2.json` |

P0 没有出现需要停下交给用户的不一致。过程中有三处工程改动与两次事故，见第 7、8 节。

## 2. GPU 环境 `[实测]`（`p0/gpu_retest.json`）

- torch 2.11.0+cu128，CUDA 12.8，RTX 5060 Laptop，计算能力 12.0，bf16 可用，显存 8151 MiB。
- XGBoost 3.2.0：`device=cuda` 训练后模型内 device = `cuda:0`；回归与 rank:pairwise 都出有限预测。
- CatBoost 1.2.10：`task_type=GPU` 的 RMSE 与 YetiRank 都通过；YetiRank 的 NDCG 指标在 GPU 上没有实现，CatBoost 只把**指标**放到 CPU 算，训练仍在 GPU。
- MLP / TabM / 序列融合 / GRU / ALSTM / MASTER：参数与梯度都在 cuda，bf16 autocast，3 步损失有限且下降。
- 早先（10-13 00:50）dmesg 记录过一次 CatBoost GPU 段错误；之后的 GPU 复测与第 6 节真实数据上的 4 个 CatBoost 单元都没有复现 `[实测]`。没有改走 CPU。
- XGBoost 的 `ExtMemQuantileDMatrix` 不接受 CPU 数据（`p0/gpu_smoke.json` 的 `extmem_error`），本卡不用它。

## 3. 日频面板构建 `[实测]`（`p0/build_logs/build_timing_summary.json`）

| 组 | 产物 | 墙钟 | 峰值 RSS |
|---|---|---:|---:|
| base（v6_1 35 列 + 价格 / 标签） | `data/gpu_v1/panel/base/`（94 个月） | 4:08:11 | 3.87 GB |
| fnd（基本面 27） | `panel/fnd/` | 3:23:18 | 3.10 GB |
| a158（Alpha158） | `panel/a158/` | 0:30:36 | 3.27 GB |
| seq（F6，16 通道） | `seq/seq_raw_f32.npy` [5813, 1882, 16] | 0:21:07 | 2.39 GB |
| sh（短周期 16） | `panel/sh/` | 0:55:46 | 7.34 GB |
| sector（PIT 行业 9） | `panel/sector/` | 0:00:43 | 0.26 GB |
| events（事件 11） | `panel/event/` | 0:16:56 | 7.91 GB |

训练矩阵 `data/gpu_v1/model/v1/`：每月 `X.npy`（float16，当日截面 gauss-rank，NaN 保留）+ `meta.parquet`；
股票级 250 列（F1 45、F2 27、F3 158、F4 9、F5 11）、市场级 26 列（`features.json`）`[实测]`。

**提速改动（不改结果）**：base 构建的性能剖析显示，一个 2026 年交易日 343 s 中有 224 s 花在
`expansion.margin_buy_amount_20d`（逐只股票在整张融资融券表上做对象比较）`[实测]`（`p0/build_logs/prof_base.log`）。
`quantmind/research/gpu/fast_factors.py` 用按股票分组的等价实现替换它（只在构建进程内替换，生产代码未改）；
`tests/test_t030_gpu_research.py::test_fast_margin_buy_amount_bitwise_equal_to_production` 断言与生产实现逐位相同 `[代码]`，
第 4 节 base 组 38 列最大误差 0 也包含这两列 `[实测]`。

## 4. §3.2 一致性：日频构建器在 374 个周频截面上 vs v6_1 文件 `[实测]`（`p0/consistency_v2.json`）

| 组 | 参照文件 | 参照行 | 日频在周频日上的行 | 参照行在日频中缺失 | 列 | 最大标度误差 | 通过 |
|---|---|---:|---:|---:|---:|---:|---|
| base | `panel/alpha_panel_weekly_v6_1_base.parquet` | 1,683,972 | 1,683,972 | 0 | 38 | 0 | ✓ |
| labels | `panel/labels_extra_v6_1.parquet` | 1,683,972 | 1,683,972 | 0 | 2 | 0 | ✓ |
| sh | `panel/short_horizon_factors_v6_1.parquet` | 1,683,972 | 1,683,972（见下） | 0 | 16 | 3.1e-10 | ✓ |
| fnd | `panel/fundamental_factors_v6_1.parquet` | 1,683,972 | 1,936,807 | 288（全为空值行） | 27 | 0 | ✓ |
| a158 | `bakeoff/alpha158_asof_v6_1.parquet` | 1,814,158 | 1,814,158 | 0 | 158 | 0 | ✓ |
| sector | `panel/sector_features_v6_1_pit.parquet` | 1,683,972 | 1,683,972 | 0 | 12 | 0 | ✓ |
| event | `panel/event_factors_v6_1.parquet` | 1,683,972 | 1,683,972 | 0 | 11 | 0 | ✓ |

- **sh 的行规则**：日频 sh 构建器对所有有价格的股票都算（周频日上 2,174,062 行），v6_1 的 sh 文件只含面板行。
  第一次检查按"行集合完全相等"判 sh 失败（`p0/consistency_v1_sh_rowset_fail.json`），但当时数值已全部一致
  （参照行 0 缺失、最大误差 3.1e-10、无失败列）。第二次检查先把日频 sh 限制到日频 base 面板的行再比，行集合相同，通过。
  训练矩阵本来就只取面板行（`build_model_ready.py` 按 base 的索引 reindex），所以这是检查口径问题，不影响产物 `[代码]`。
- **fnd 的 288 行**：参照文件里这 288 行全部是空值（`ref_rows_missing_nonnull = 0`），日频构建器不产出空行；值比较不受影响。
- 最大标度误差 = |差| / max(1, |参照值|)；sh 的 3.1e-10 来自浮点运算顺序。

## 5. Ridge C 复现（N4 负对照）`[实测]`

日频构建器产物 → 拼回 v6_1 周频面板（225 个特征）→ T-018 的 Ridge 代码（α = 10，滚动 756，n_val = 2）：

| 期限 | 基线文件 | 行数 | 行索引相同 | 最大绝对差 | 方向与基线相同 | 耗时 |
|---|---|---:|---|---:|---|---:|
| 63d | `bakeoff/preds_v6_1/ridge_63d_v6_1_C_infnan.parquet` | 1,000,377 | ✓ | 3.56e-12 | ✓（9 折，第 4 折 −1） | 828 s |
| 20d | `bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet` | 1,108,869 | ✓ | 7.57e-12 | ✓（19 折全 +1） | 934 s |

63d 有 162 个 ±inf 按基线的 `infnan` 规则换成 NaN（`ridge_repro_63d.json` 的 `n_inf_replaced`）。

## 6. 资源实测（只用 20d 最后一折的训练窗；不预测任何外层日期、不算任何 IC）

最大训练窗：折 18（切点 2026-07-01）736 个交易日、3,721,506 行（每日约 5,056 只）、250 + 26 列；
float16 矩阵 1,775 MB，装载 27.6 s、峰值 RSS 2.57 GB `[实测]`（`p0/resource_probe_v2.json` 的 `load`）。

| 单元 | 行 | 每轮 / 每 epoch | 峰值 RSS | 峰值显存 | 来源 |
|---|---:|---:|---:|---:|---|
| XGBoost 回归 深度 8 | 3.71M | 0.134 s / 轮（另有约 40 s 建矩阵） | 4.73 GB | 3.0 GB | v3 `xgb`、`xgb_300` |
| XGBoost rank:pairwise 深度 8 | 3.71M | 0.19 s / 轮 | 4.74 GB | 3.2 GB | v3 `xgb_rank`、`xgb_rank_300` |
| CatBoost RMSE 深度 8，全部交易日 | 3.71M | 1.03 s / 轮 | **8.31 GB** | 4.2 GB | v2 `cat_rmse` |
| CatBoost RMSE 深度 8，隔日 | 1.85M | 0.29 s / 轮 | 4.52 GB | 4.2 GB | v3 `cat_rmse_s2` |
| CatBoost YetiRank 深度 8，隔日 | 1.85M | 0.59 s / 轮 | 4.61 GB | 4.2 GB | v3 `cat_yetirank_s2` |
| LightGBM CPU 255 叶，全部交易日 | 3.71M | 2.74 s / 轮（8 线程） | 5.78 GB | — | v2 `lgbm` |
| LightGBM CPU 255 叶，隔日 | 1.85M | 1.36 s / 轮（8 线程） | 3.21 GB | — | v3 `lgbm_s2` |
| TabM k=32 d=512 ×3 | 3.72M | 58 s / epoch（隔日） | 4.04 GB | torch 5.7 GB | v3 `tabm` |
| MLP d=512 ×3 | 3.72M | 6.6 s / epoch | 4.06 GB | torch 2.4 GB | v3 `mlp` |
| 序列融合 L=128 | 3.72M | 46 s / epoch | 5.07 GB | torch 5.2 GB | v3 `seqfusion` |
| GRU L=64 | 3.72M | 16 s / epoch | 5.04 GB | torch 3.6 GB | v2 `gru` |
| ALSTM L=64 | 3.72M | 22 s / epoch | 5.13 GB | torch 4.0 GB | v2 `alstm` |
| MASTER d=128 | 3.72M | 44 s / epoch | 4.92 GB | torch 4.6 GB | v2 `master` |

- 深度模型的 epoch 都是 `day_stride = 2`（每个 epoch 用一半交易日）。"torch" = `torch.cuda.max_memory_allocated`；
  nvidia-smi 峰值（含 CUDA 上下文与缓存）≤ 6.8 GB，进程内上限 6500 MiB（`runtime.cap_torch_memory`）。
- XGBoost 每轮耗时 = (300 轮 − 100 轮的拟合时间) / 200 `[实测]`；"约 40 s 建矩阵" 是 100 轮拟合时间减 100 × 每轮耗时 `[推断]`。
- CatBoost 全量行 8.3 GB 峰值在日更运行时会把可用内存压到 1.5 GB 以下，所以 CatBoost 与 LightGBM 采用隔日训练（预登记写明）。
- 机器：12 GB 内存（WSL 可见 11.7 GB）、8 GB swap、12 个逻辑 CPU；日更 `daily_update.py` 同机运行（约 0.2–0.3 GB）。

## 7. 为把峰值内存压到 ≈ 5 GB 做的工程改动（数值不变，各有单元测试）

| 改动 | 前 → 后（实测） | 测试 |
|---|---|---|
| `ModelData.load` 预分配输出，不再逐月拼接 | 装载峰值 4.29 → 2.57 GB（前者为第一次探针的 `load` 单元，第二个主进程出现之前完成） | 装载结果同形同值（第 6 节同一折） |
| 树输入 `trees.TreeInput`：按 20 万行块产出 float32；XGBoost 用 `DataIter` + CuPy 块，LightGBM 用 `lgb.Sequence`，CatBoost 用一份 Fortran 序数组（Pool 不再复制） | XGBoost 单元 10.3 GB 时被手动停止（`ps` 观测，未落盘）→ 4.73 GB | `test_tree_input_chunks_and_fortran_equal_dense`、`test_xgb_iter_dmatrix_trains_like_dense`、`test_lgb_sequence_trains_like_dense` |
| 深度表格输入按块写进显存张量 | TabM 单元 8.94 → 4.04 GB | `test_dl_tab_inputs_chunked_equal_full_concat` |
| CatBoost `gpu_ram_part = 0.55`（默认 0.95 预占） | 显存 7.1 GB（第一次探针 `cat_rmse`，该轮其余数字不引用）→ 4.2 GB | — |
| CatBoost YetiRank：每个交易日随机均分成 ≤ 1000 只的子查询（GPU 单个查询上限 1023，一个交易日约 5,000 只） | 原样报错 → 通过 | `test_split_queries_contiguous_bounded_and_within_day` |
| 内层验证按段累加原始分（与一次预测只差全体常数，日内排序不变） | 每个配置的验证预测量从 Σr 降到 max_rounds | `test_incremental_raw_segments_sum_to_full_prediction` |
| `runtime.start_ram_guard`：可用内存 < 1.5 GB 时研究进程自行退出（退出码 75），台账续跑 | — | — |

## 8. 事故记录

1. **资源探针第一次运行**：XGBoost 单元（旧的整块 float32 实现）RSS 涨到 10.3 GB，可用内存只剩约 200 MB，
   当时日更 `daily_update.py` 正在运行。我立刻停掉探针单元；日更进程未受影响（之后仍在运行）。
   停的时候只杀了外层包装进程，探针主进程（PID 77503）还活着，接着跑后续单元，与第二次启动的探针同时运行，
   两边互相挤占内存、多个单元被内存守卫终止。这一轮的耗时与内存结果作废，保留为
   `p0/resource_probe_attempt1_two_drivers_contaminated.json`；只引用两处与挤占无关的数：第二个主进程出现之前完成的 `load` 单元，
   以及 CatBoost 按 `gpu_ram_part` 默认值预占的显存（7.1 GB）。之后的每次启动都核对实际 Python 进程号。
2. LightGBM 的合成数据冒烟曾在 CPU 超额订阅（负载 17）时卡住，按 PID 停止；第 6 节的真实数据单元在正常负载下通过。

## 9. 交给 P1 的事实

- 一折训练窗约 2.6–3.7M 行；ENS 四个成员（XGBoost、CatBoost、TabM、序列融合）都能在 ≤ 5.1 GB 内存、≤ 6.5 GB 显存内训练 `[实测]`。
- 全部计划的估计墙钟约 100–110 小时（GPU 队列串行；LightGBM 在 CPU 上并行），估算见预登记 §4.1 `[推断]`。
