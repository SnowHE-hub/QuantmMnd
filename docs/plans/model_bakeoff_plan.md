# 多模型 × 双周期 回测 Bake-off 设计文档

> 状态：**设计期，获批前不进入实现。** 写完停下评审。
> 承接：`wf_v2_diagnostics.md`（12d v2 中性化后只剩 12%）+ `short_horizon_factor_plan.md`（短周期因子，待筛）。
> 一句话目标：在**统一的 purged-WF + 中性化 IC + 含成本**框架下，公平比较 **Linear/LightGBM/GRU/LSTM/Transformer** 在 **12d / 63d** 的表现，建立"现有数据 + 最佳模型"的**经验上界**，并顺带验证 63d 产品、给出 refit 频率建议。
> 生成：2026-06-06

⚠️ **headline = 中性化 IC + 含成本净超额**，不是 raw-IC 排行榜。全程幸存者乐观上界，客户前须先补退市票。

---

## ✅ 决策记录（2026-06-06 评审通过，进入执行）
GPU 确认 = **RTX 5060 Laptop 8GB GDDR7（Blackwell, 128-bit, sm_120）**。获批范围内不再停下评审，除非 **G1/G2 自检失败** 或 **R2 兼容性退路触发**。

| 决策 | 裁定 |
|---|---|
| **R1** | 8GB 确认；**五模型全留**（小 batch + AMP）。 |
| **R2** | 双 env 隔离为**首选**（真 pyqlib = 经验证实现）。P0 先做**兼容性探针**（qlib_bakeoff env 实测 pyqlib import + 一个深度模型在 cu128/sm_120/numpy 下训练）；**仅探针失败才退 vendor 路线**（搬 handler+nn 代码、不装 qlib 包）。无论哪条，**评估永远留 quantmind env**。 |
| **R3** | short_horizon 因子筛选**先跑/并行**（单因子 IC、不训练模型，很快），在 P0/P1 期间完成；survivors 经 `merge_increment` 进 tabular，等 P3 tabular 开训时齐。**不在已知不完整的 tabular 上烧 GPU**。sequence(Alpha360) 不依赖它，照常走。筛选用 `short_horizon_factor_plan.md` + 定稿三点：保留 ~47、市场项 = **cap 加权截面均值**、保留线 **\|neut ICIR\|≳0.2 且翻转<35% 且 corr<0.7**、**选因子用 in-sample/train neut IC、OOS 留作无偏确认**。 |
| **R4** | 月度 refit 只测 **baseline（Linear/LGBM）+ 季度的所有 contender**（不止冠军，边缘的也带上），不跑全 5×seed 网格。 |
| **R5** | 序列长 **L=60**；深度模型 **seed=5**（非 3），leaderboard 报均值±std；Alpha158+Alpha360 默认 handler。 |
| **R6** | 排序 headline = **(中性化 IC, 含成本净超额) 降序**；raw IC 仅参考列；**v2(35因子+LGBM) 作对照行**。 |

**三处质量修订**（已并入 §6/§8/§13）：
1. **63d 这一跑 = "量价对 63d 的基线"，不是完整 63d 产品验证**。63d 产品 alpha 源是**基本面（质量/价值/成长）**；本跑只用 Alpha158/360 量价。真正的 63d 产品验证 = 加基本面因子（daily_basic 已回补 + 财报 PIT）那一版，作为紧接 follow-up；本跑保留为**基线 + harness 练手**。
2. **选择乐观偏差**（§8 解读规则）：leaderboard 冠军行 OOS neut IC 被**跨模型搜索抬高**。要回答的是"**有没有任意 模型×制式跨过门槛**"，不是"冠军数字=真实未来"。出现跨线赢家 → 留**最终 holdout 段**干净确认，或明确**当搜索上界读**。
3. **（可选）图模型**：P3 后若有余力，加一个 Qlib A股 benchmark 最强档图模型（GATs/HIST/IGMTF）作 sequence 额外对手；非必须，不加不影响核心结论。见 §13。

---

## 0. 核心原则（贯穿全文）
1. **模型二阶、特征一阶**：本 bake-off 的价值不在"调出一个 SOTA"，而在回答四问——(a) 序列模型在**原始量价序列**上能否抓到 tabular 手工因子漏掉的时序模式；(b) 建立**现有数据 + 最佳模型的经验上界**；(c) 验证 **63d 策略产品**；(d) 用 **refit 频率**维度回答"定期更新该多久一次"。
2. **每个模型喂发挥其长处的特征制式**，不强求统一一份特征。
3. **统一评估口径**：所有模型导出逐 (date,ticker) 预测，**送进同一套** `PurgedWalkForwardSplit` + `_diag_wf_v2` 中性化 + `wf_costs`。**v2（35因子+LGBM）保留为对照行。**
4. **诚实性硬条件**（沿用 WF gate 的 H-A/H-C）：超参与早停**只用 val**；方向**只用 train/val**（绝不碰 OOS）；保留 purge 消融反证。

---

## 1. 环境实况与两点必须先评审的偏差

实测（详见 `findings.md`）：

| 项 | 任务书 | 实测 | 设计应对 |
|---|---|---|---|
| GPU | RTX 5060 **Ti 16GB** | **RTX 5060 Laptop 8GB**（sm_120 Blackwell, driver 595.79） | **按 8GB 设计**：小 batch + AMP；Transformer 收敛在 8GB 内 |
| torch | CUDA | 2.11.0+cu128，cuda_available=True | ✅ 可 GPU 训练 |
| qlib | 用 Qlib | **未安装**，且 env 是 numpy2/pandas2.3 | **新建隔离 env** 装 pyqlib，避免降级 quantmind |
| 计算 shell | WSL | 本会话默认 Bash=Windows/Git Bash | 训练命令一律 `wsl -e bash -lic`，用 `/home/lenovo/miniforge3/envs/<env>/bin/python` |

> 🚩 **评审点 R1**：显存只有 8GB（非 16GB）。这影响 Transformer/序列模型规模与总训练时长。是否接受 8GB 配置（见 §5），或缩减网格（见 §10 分阶段）？

---

## 2. 双 env 架构 + qlib↔评估的桥

```
┌─ env: quantmind (现有, numpy2/pandas2.3) ────────────────────────┐
│  数据 / 特征 / 标签 / PurgedWalkForwardSplit / neutralize /        │
│  wf_costs / leaderboard 评分。 ← 主交付都在这里。                  │
└──────────────────────────────────────────────────────────────────┘
            ▲ 预测 parquet (date,ticker,pred,model,fmt,period,refit,seed)
            │（唯一桥接物，解耦依赖冲突）
┌─ env: qlib_bakeoff (新建, pyqlib + 兼容 numpy/pandas + cu128 torch) ┐
│  dump_bin → Alpha158/Alpha360 handler → Linear/LGBM/GRU/LSTM/       │
│  Transformer 训练（GPU）→ 逐 fold 导出 OOS 预测。                   │
└──────────────────────────────────────────────────────────────────┘
```

- **为什么隔离**：经典 pyqlib 常 pin numpy<2 / pandas<2；直接装进 quantmind 会降级整套量化栈（含已跑通的 WF/诊断脚本）。隔离后 quantmind 一行不改。
- **桥接契约**：qlib 侧只产出 `预测 parquet`（列：`datetime, instrument(→ticker), score, model, fmt, period, refit, seed`）。评估侧只消费它 + 我们自己的标签/行业/circ_mv。**评估绝不进 qlib**。
- 🚩 **评审点 R2**：同意双 env 隔离（而非把 qlib 装进 quantmind）？若 pyqlib 与 cu128/sm_120 torch 不兼容，退路 = **只 vendor qlib 的 handler+模型代码**（Alpha158/360 + GRU/LSTM/Transformer/Linear 的 nn 实现），在 quantmind env 内直接用 torch 跑（不装 qlib 包）。R2 请二选一或允许实现期按兼容性自动择路。

---

## 3. 特征双制式

### 3.1 Tabular（给 Linear / LightGBM）
`现有 35 因子` + `short_horizon 存活因子（若 Step 2 已筛出，否则缺省略并标注 pending）` + `Qlib Alpha158`。
- 35 因子直接取自 `alpha_panel_weekly_v5`。
- Alpha158 由 qlib handler 在 bin 数据上算，导出到 (date,ticker) 后**周频 as_of 取值**，与 35 因子按 (as_of,ticker) 对齐。
- 类别特征 `exposure_industry` 以 LGBM 原生 categorical 传入（沿用 v2 修复）；Linear 用行业哑变量或丢弃（见 §5）。
- **R3 已批准**：short_horizon 因子筛选**在 P0/P1 期间先跑/并行**（单因子 IC，不训练，很快），survivors 经 `merge_increment` 进 tabular，等 P3 tabular 开训时齐 → **不在不完整 tabular 上烧 GPU**。sequence(Alpha360) 不依赖 short_horizon，照常走。筛选口径见决策记录 R3（cap 加权市场项、\|neut ICIR\|≳0.2 且翻转<35% 且 corr<0.7、选因子用 train neut IC、OOS 留无偏确认）。

### 3.2 Sequence（给 GRU / LSTM / Transformer）
**Qlib Alpha360 风格**：每股过去 **~60 个交易日**的标准化原始 OHLCV 序列。
- 6 通道（open/high/low/close/volume/vwap 或 qlib 标准 6 列），每日一帧，序列长 L=60 → 输入 (L=60, C=6)。
- **逐日截面标准化 CSZScoreNorm**：每个交易日对每个特征做横截面 z-score（与 qlib Alpha360 一致），消除量纲/规模主导。
- PIT：序列窗口 [as_of−59, as_of]，只回看；标签 = 我们面板的 forward_return_{h}d。

---

## 4. 数据 → Qlib（dump_bin）与 Handler

1. **dump_bin**：`alpha_prices_panel` 长表 → qlib bin。字段 `$open/$high/$low/$close/$volume` + `$factor`（复权因子，qlib 约定）。calendar = `load_trading_calendar()`；instruments 文件用 universe，每只票 `start = list_date`（PIT 上市），`end = 末日`（delist 全空 = 幸存者，meta 标注）。
2. **vwap**：qlib `$vwap` 若用，按 `10×amount/vol`（§findings 单位自查）预计算注入，或用 qlib 表达式由 amount/vol 构造（单位对齐）。
3. **Handler**：`Alpha158`（tabular 补充）、`Alpha360`（序列）。两者均 causal。
4. **标签对齐**：不使用 qlib 默认 `Ref($close,-h)` 标签；改用我们 `compute_forward_returns`（adj_close, T+1 口径）产出的 `forward_return_{h}d`，作为 DatasetH 的 label 列注入。**自检**：抽 1 只票，qlib 注入标签 vs 面板标签逐日一致。
5. **自检门 G1**（数据转换）：① 抽 3 只票 5 个日，qlib `$close×$factor` 还原 vs `adj_close` 一致；② Alpha158/360 某列 vs 手算一致（相关 >0.99）；③ PIT 反证：篡改未来值不改变历史输出。

---

## 5. 各模型训练设计（发挥长处；超参/早停只用 val）

| 模型 | 制式 | 关键设计 | 8GB 配置（初值，val 上微调） |
|---|---|---|---|
| **Linear (Ridge)** | tab | **诚实基线**：深度模型打不过它=没学到东西。标准化 + L2；行业可哑变量。 | α∈{1,10,100} val 选；CPU 即可 |
| **LightGBM** | tab | **lambdarank** + early stopping on val + **categorical(行业)** + NaN 原生处理。沿用 v2 LGBMRankerModel 口径。 | n_estimators 400、num_leaves 31、lr 0.05；GPU 可选（hist） |
| **GRU** | seq | 序列 L=60、逐日截面标准化、dropout、**early stopping on val IC**、AMP fp16。 | hidden 64、layers 2、dropout 0.3、batch 800、lr 1e-3、AMP |
| **LSTM** | seq | 同 GRU 框架。 | hidden 64、layers 2、dropout 0.3、batch 800、AMP |
| **Transformer** | seq | 注意力头/层适中（8GB 内）、位置编码、early stopping。 | d_model 64、nhead 4、layers 2、ff 128、dropout 0.3、batch 512、AMP |

**统一规则（所有模型）**：
- **超参与早停只用 val**（绝不用 OOS）。早停指标：深度模型 = **val 截面 IC**（不是 loss），与最终评估口径一致。
- **方向（H-A）**：只用 train/val 决定（复用 `decide_direction`），`auto_flip=False`。
- **多 seed**：深度模型（GRU/LSTM/Transformer）= **5 seed**（R5），leaderboard 报 **均值 ± std**；Linear/LGBM 单 seed（近确定性）。
- **训练日志**：记录每 fold 训练时长（GPU 秒）、best_epoch/iter、val IC 曲线。

---

## 6. 双周期 + 统一评估管线

### 6.1 周期
| 周期 | H（horizon） | E（embargo） | holding_td | 标签 | 用途 |
|---|---|---|---|---|---|
| **12d** | 12 | 20 | 12 | forward_return_12d | 与 v2 同口径，短周期 |
| **63d** | 63 | ≥63（取 63） | 63 | forward_return_63d | **量价对 63d 的基线**（非完整产品验证，见下注） |

> **63d 基线说明（质量修订 1）**：本跑的 63d 只用 **Alpha158/360 量价**，是"量价能否打 63d"的基线 + harness 练手。**完整 63d 产品验证**的 alpha 源是**基本面（质量/价值/成长）**——daily_basic（已回补）+ 财报 PIT 因子那一版，作为**紧接的 follow-up**，不在本 bake-off 范围。leaderboard 的 63d 行须带此标注，避免被读成"63d 产品已验证"。

### 6.2 Fold = refit 频率驱动（核心机制）
- **refit 频率 ≙ cutoff 间距**：
  - **季度 refit** = 复用 `make_quarterly_cutoffs`（2022Q1→2025Q4，~16 cutoff）。
  - **月度 refit** = 新增月度 cutoff（每月初最近交易日，~48 cutoff）。
- 每个 cutoff：`PurgedWalkForwardSplit(rolling_lookback=756, n_val=2)` 切 train（purged）/val/test（到下个 cutoff 的 OOS 块）。
- 每个 (模型×制式×周期×refit×seed×fold)：qlib env 训练 → 导出该 fold OOS 预测 parquet。

### 6.3 评估（quantmind env，对每条预测序列）
对每个 (模型×制式×周期×refit) 聚合其 OOS 预测后：
1. **raw IC**：逐 OOS 截面 `Spearman(pred×dir, forward_return_{h}d)` 均值。
2. **中性化 IC / ICIR**：逐截面 pred 对 `C(industry)+z(log circ_mv)` 取残差重算 IC（`_diag_wf_v2.neutralize_date` 同口径）。
3. **purge 消融 inflation**：对该模型跑 `purge_ablation`（H/E 开 vs 关），inflation 应 >0（无泄漏自检）。
4. **含成本净超额 + 最大回撤 + 换手**：top-quintile 组合，过 `wf_costs`（滑点分层 + 时变印花 + 板块涨跌停，holding_td=周期），算年化净超额、回撤、换手。
   - ✅ **验收意见②（batch B 前置 must-fix）已落地**：成本改为 **holding-period-aware 真实换手**——把周频 as_of 子采样到**非重叠持有期**（step=⌈H/5⌉ 个 as_of）再平衡，逐期从 top-quintile membership 算单边换手 `0.5·Σ|wₜ−wₜ₋₁|`，成本只计真实换手部分（买卖两边滑点/佣金/过户 + 卖印花×换手）。保留 12d/63d 与模型间换手差异。leaderboard 增 `avg_oneway_turnover` / `annualized_turnover` 列。v2 对照行用同一成本口径重算（净超额 −8.4%，非旧 −12.3% 全换手）。
5. **训练时长**：累计 GPU 秒。

### 6.4 自检门 G2（评估管线）：在 v2 预测上复跑本管线，必须复现 `wf_v2_diagnostics`（neut IC +0.0017、raw +0.0139），证明 harness 与 v2 一致后才接深度模型预测。 ✅ 已过。

### 4.6 G1 完整通过记录（2026-06-07，`scripts/bakeoff/_g1_v2.py` → `p1_g1_result.json`）
**G1_FULL_PASS = True**，三项补验（之前只验 close 还原 + shape）：
1. **标签前向对齐**：面板已存 `forward_return_12d/63d`（= compute_forward_returns 产物）vs qlib `Ref($close,-h)/$close-1` **逐值精确一致**（max abs diff <1e-6）；off-by-one 守卫：与 −11/−13 明显不等 → horizon 不差一期，符号/方向/口径正确。
2. **feature↔label join PIT 反证**：t=2022-06-15，篡改未来价(>t ×2) → 特征 MA5 与 close(t) **不变**、label_12d −0.0050→+0.9900 **变**。证 (a) 特征不依赖未来 (b) label 输入区间用 >t (c) 特征只用 ≤t。
3. **handler 列手算**：Alpha158 `MA5` vs 手算 corr **1.0**(n=2484)；Alpha360 `CLOSE5` vs 手算 corr **1.0**。均 >0.99。
→ close 还原 5.8e-8 + 三项通过 = **G1 真 PASS**，放行 P3。

---

## 7. 统一对比（核心交付：Leaderboard）

每个 (模型 × 制式 × 周期 × refit) 一行：

| 模型 | 制式 | 周期 | refit | raw IC | **中性化 IC** | 中性化 ICIR | purge inflation | **含成本净超额** | 最大回撤 | 换手 | 训练时长 | seed std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LGBM(v2 对照) | tab35 | 12d | 季度 | +0.0139 | +0.0017 | 0.03 | >0 | −2.65% | 27% | … | … | — |
| Linear | tab | 12d | 季度 | | | | | | | | | |
| LightGBM | tab | 12d | 季度 | | | | | | | | | |
| GRU | seq | 12d | 季度 | | | | | | | | | ± |
| LSTM | seq | 12d | 季度 | | | | | | | | | ± |
| Transformer | seq | 12d | 季度 | | | | | | | | | ± |
| …（×63d，×月度 refit） | | | | | | | | | | | | |

**排序键 = (中性化 IC, 含成本净超额) 降序。** raw IC 仅作参考列。

---

## 8. 结论模板（交付时回答）
1. **中性化口径下哪个 模型×制式 最好？** 序列模型（Alpha360 原始量价）是否**真优于** tabular 手工因子（35+Alpha158）？——若深度模型中性化 IC 不超过 Linear/LGBM，则"现有日线没有手工因子漏掉的时序 alpha"。
2. **12d vs 63d 哪个更可交易**（中性化后 + 含成本）？注意 63d 行是**量价基线**（质量修订 1），非产品验证。
3. **经验上界 vs Gate 门槛**（中性 IC>0.03 / 净超额>5% / …）：跨得过吗？**跨不过 → 日线通道见顶，下一步上分钟数据**（届时本 harness 直接复用、测分钟衍生因子）。
4. **refit 频率建议**：季度 vs 月度，哪档 OOS 更稳 → "上线多久重训一次"。
5. **幸存者乐观上界**重申；客户前须先补退市票。

### 8.1 解读规则 —— 选择乐观偏差（质量修订 2，硬性）
leaderboard 是**跨 模型×制式×周期×refit×seed 的搜索**，冠军行的 OOS 中性化 IC 因此被**搜索抬高**（最大值的期望 > 单次期望）。报告必须：
- 把问题钉死为"**有没有任意 模型×制式 跨过门槛**"，而**不是**"冠军数字 = 真实未来收益"。
- 出现跨线赢家时：① 优先留**最终 holdout 段**（如 OOS 末段不参与任何模型/超参选择）做**一次干净确认**；② 若不留 holdout，则冠军 neut IC **明确当"搜索上界"读**，与幸存者上界叠加双重保守。
- 任何 seed 选择、早停、方向判定都不得用到 OOS（H-A 已保证）；leaderboard 报 seed 均值±std，**不报 seed 内最优**（避免二次择优）。

---

## 9. 分阶段执行（每阶段自检通过再下一阶段）
| 阶段 | 内容 | 自检门 |
|---|---|---|
| **P0** | 建 `qlib_bakeoff` env + 装 pyqlib（兼容性验证）+ `torch.cuda.is_available()` 复确认 | qlib import OK、GPU 可见、与 quantmind env 不互相降级 |
| **P1** | dump_bin 数据转换 + Alpha158/360 handler | **G1**（复权还原 / handler 对齐 / PIT 反证） |
| **P2** | 评估 harness（fold 驱动训练循环 + 导出 + 评分）+ v2 复现 | **G2**（v2 数值复现） |
| **P3** | 模型套件（Linear/LGBM/GRU/LSTM/Transformer）单 fold 跑通 | 各模型 1 fold OOS 预测非空、val IC 收敛 |
| **P4** | **双周期 × refit 全网格**（分批，见 §10） | 每批落盘断点续传 + 训练时长记录 |
| **P5** | 统一评估 → leaderboard | 排序稳定、purge inflation 全 >0 |
| **P6** | 结论文档 | 回答 §8 五问 |

---

## 10. 计算预算与分阶段网格（应对 8GB 笔记本 GPU）
深度训练总量 ≈ 模型(3) × 周期(2) × refit(季度16 + 月度48=64 fold) × seed(3) ≈ **1100+ 次**，叠加 Linear/LGBM。**多日 GPU**。为可控，建议**分批**：

- **批 A（pilot，最便宜）**：12d × 季度 × **1 seed** × 全 5 模型 → 打通 harness + 拿首张 leaderboard（半天内）。
- **批 B（季度全网格）**：12d & 63d × 季度 × 3 seed（深度）× 全模型。
- **批 C（月度 refit，最贵）**：**baseline（Linear/LGBM）+ 批 B 的所有 contender**（不止冠军，边缘的也带上；R4），而非全 5 模型全 seed —— 省算力且足够回答 refit 频率。contender 定义：批 B 中 neut IC 在门槛附近或进入前列、值得看 refit 敏感性的模型×制式。

> R4 已批准（见决策记录）；R5 深度 seed=5。批 B 深度训练量 = 3 模型 × 2 周期 × ~16 fold × 5 seed = ~480 次 + Linear/LGBM。批 C 月度按 contender 数定。

---

## 11. 风险与缓解
| 风险 | 缓解 |
|---|---|
| pyqlib × numpy2/pandas2.3 冲突 | 隔离 env；冲突则退 R2 路线 B（vendor handler+nn 代码，不装 qlib 包） |
| Blackwell sm_120 下 qlib/torch 算子缺核 | torch 已 cu128（cap 12.0 OK）；qlib 自身少量 numba/cython 算子在 P0 验证 |
| 8GB OOM | 小 batch + AMP + 梯度检查点；Transformer 降 d_model/layers |
| 标签/口径不一致致不公平 | 统一用面板 forward_return；G2 复现 v2 |
| qlib 内部 train/valid 泄漏 | **不用 qlib 的 rolling/backtest**；fold 由我们 PurgedWalkForwardSplit 驱动 |
| 深度模型方向用到 OOS | H-A：方向只 train/val；早停只 val |
| 幸存者偏差 | 全程标注乐观上界；结论须含"客户前补退市" |

---

## 12. 交付物清单（实现后）
- `qlib_bakeoff` env + `scripts/bakeoff/`（dump_bin、handler、fold 训练循环、导出）。
- `quantmind/backtest/` **不改**；新增 `scripts/bakeoff/evaluate_bakeoff.py`（复用 wf_split/gate/costs + _diag_wf_v2 中性化）。
- `data/bakeoff/preds/*.parquet`（逐模型预测）+ `leaderboard.csv`。
- `docs/plans/model_bakeoff_results.md`（leaderboard + §8 结论）。

---

## 14. Batch-A 结果与结论（2026-06-09，12d×季度×1seed，全 1374 流动性分桶覆盖）

完整 leaderboard（排序：全 1374 中性化 IC 降序；seq = 3 桶合并、桶内 rank 归一化；桶0=最活跃…桶2=最不活跃）：

| 模型 | 制式 | **全1374 neut IC** | ICIR | 桶0活跃 | 桶1中 | 桶2不活跃 | **含成本净超额** | maxDD |
|---|---|---|---|---|---|---|---|---|
| **Ridge** | 35+16+158 | **0.0340** | 0.64 | 0.032 | 0.032 | 0.034 | **+1.9%** | 0.048 |
| LSTM | Alpha360 | 0.0316 | 0.63 | 0.004 | 0.015 | **0.075** | −1.8% | 0.090 |
| GRU | Alpha360 | 0.0281 | 0.66 | 0.005 | 0.020 | 0.053 | −2.9% | 0.099 |
| Transformer | Alpha360 | 0.0235 | 0.34 | −0.004 | 0.012 | 0.060 | −5.1% | 0.142 |
| LGBM | 35+16 | 0.0170 | 0.33 | 0.018 | 0.020 | 0.003 | −3.4% | 0.092 |
| LGBM | 35+16+158 | 0.0146 | 0.29 | 0.012 | 0.005 | 0.022 | −7.3% | 0.191 |
| LGBM(v2) | 35 | 0.0017 | 0.03 | 0.005 | 0.000 | 0.007 | −8.4% | 0.210 |

> **† 未评审偏离（必读，防误读）**：受 12GB 内存限制，序列模型（GRU/LSTM/Transformer）**未在全 1374 上单模型训练**，而是改为**按流动性切 3 桶各训一个模型**（此改动**未经评审**）。其"全1374 neut IC"是 **3 个桶模型预测桶内 rank 归一化后拼接**的结果，**与 Ridge 的单模型全 universe 数字不是同一类**，不可直接并列比较。该偏离若有影响是**抬高**了序列模型（桶内截面更同质、噪声更低）→ 即便如此序列模型仍未净赢，结论方向不变、反而更稳。

### 核心结论
1. **没有任何序列模型在全 1374 上超过 Ridge(full) +0.0340**（LSTM 0.0316†最接近=93%，GRU 0.0281†，Transformer 0.0235†）。3 个序列模型都**碾压全部 LGBM**（≤0.017）。
2. **可交易净收益：Ridge 是唯一为正（+1.9%）**；3 个序列模型全为负（LSTM −1.8% / GRU −2.9% / Transformer −5.1%）。
3. **3 个序列模型一致：活跃票弱/负（桶0 ≈0），不活跃票"高"（桶2 0.05-0.075）。3 架构一致 = 该模式真实存在。**
4. **但桶2 那个 0.05-0.075 是最不可信、最挖不出钱的数字，不是 follow-up 线索（关键，防误读）**：
   - **幸存者偏差最重**：桶2=小盘/ST/低流动性票，正是 2019-2026 间最容易退市/暂停的群体；当前 v5 universe 缺退市票（源头幸存者偏差），桶2 的 IC 被幸存者**系统性抬高**——这块"alpha"很可能大半是"只看到活下来的小盘票"的假象。
   - **成本最高**：桶2 滑点档最贵，净超额已为负。
   - 双重角落（幸存者最重 × 成本最高）→ 该数字**最不可信 + 最不可交易**。在幸存者修复前，**不得**把它当作"序列模型有戏"的依据或 follow-up 方向。
5. **Ridge(full) 是赢家**：宽、跨流动性均匀、可交易净正——"诚实线性基线 + 富特征"在日频量价上胜过深度序列模型。
6. ⚠ 全程幸存者乐观上界 + 深度 1 seed（3 架构一致缓解噪声）；任何结论客户前须先补退市票复核。

### 下一步（用户定案，2026-06-09）
- **阶段关闭**：batch-A 结论定案 = Ridge(full) 赢。
- **唯一下一件事 = 幸存者修复**（见 `docs/plans/survivorship_repair_plan.md`）。**63d / batch B / illiquid 深度 follow-up 全部等幸存者修好后再议**——因当前任何数字（尤其 illiquid）都受幸存者偏差污染。
- **修好后第一步只做一件**：真实 universe 上重跑 **Ridge(full)**，看 **+0.034 / +1.9%** 还在不在 —— 这一个数字决定有没有产品。之后 63d/batch B 才在真实 universe 上做才可信。

## 13.（可选）图模型对手（质量修订 3）
P3 全 5 模型跑通后**若有余力**，加一个 Qlib A股 benchmark 最强档**图模型**（**GATs / HIST / IGMTF**）作 sequence 制式的额外对手（同 Alpha360 输入 + 股票关系图）。**非必须**，不加不影响 §8 核心结论；显存吃紧（8GB）时优先保证 5 模型主网格。若加：同样导出预测、走同一评估管线、leaderboard 增行。

---

## ✅ 评审门 —— 已批准，进入执行
设计 + 决策记录（R1–R6 + 3 修订）已评审通过。**获批范围内按 §9 逐阶段推进、不再停下评审**，仅在以下情况暂停：
- **G1** 失败（不进 P2）｜ **G2** 失败（不接深度模型预测）｜ **R2 兼容性探针失败**（触发 vendor 退路，知会后继续）。
执行顺序：P0（建 env + 探针）→ R3 并行筛 short_horizon → P1（dump_bin + handler，G1）→ P2（harness + v2 复现，G2）→ P3（模型套件单 fold）→ P4 分批（A pilot → B 季度全网格 → C 月度 baseline+contender）→ P5 leaderboard → P6 结论。每阶段落盘断点 + 自检通过再下一阶段。
