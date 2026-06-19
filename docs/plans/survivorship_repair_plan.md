# 幸存者偏差修复计划（Survivorship Repair）

> 状态：**计划期，写完停下评审，获批前不实现。**
> 承接：bake-off 阶段关闭（Ridge(full) +0.034/+1.9% 赢，但全在幸存者乐观上界上）。
> 目标：把"只含活下来的票"的 universe，修成**真 PIT universe（每个 as_of 含当时在市、含后来退市的票，按退市日正确剔除）**，新建 v6，**不动 v5、评估侧零改动**。
> 修好后第一步只做一件：真实 universe 上重跑 **Ridge(full)**，看 **+0.034/+1.9%** 还在不在——这一个数字决定有没有产品。
> 生成：2026-06-09

---

## 0. 问题（为什么必须先修这个）
- 实测：`stock_basic` 1388 票**全 list_status='L'、delist_date 全空** → universe 只含**现在还活着**的票，2019-2026 间退市/暂停的票**全部缺失**。
- 后果：所有 IC/净超额都是**幸存者乐观上界**。尤其 bake-off 里序列模型在 **illiquid 桶**的 0.05-0.075，正落在"小盘/ST/最易退市"群体——**那块"alpha"很可能大半是只看到活下来的小盘票的假象**。
- 因此：**在修好幸存者前，任何数字（含 63d、illiquid follow-up）都不可信**。这是上轮已 scope 的事，不是新问题。

## 1. ⚠ Scope 决策（评审第一问，必须先定）
当前 1388 不仅缺退市票，还只是全 A 股的**子样本**（仅 1137 票 2019 前上市，远低于真实 ~3500+ 存续）。两种口径：

| 口径 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **A. 全市场 PIT（推荐）** | 拉全 SH+SZ A 股（L+D+P），按 list/delist 日建真 PIT universe | 唯一无偏、定义清晰；彻底消除幸存者 + 样本选择偏差 | 数据量大（~4800+ 票×价量），Tushare 拉取久 |
| B. 样本+其退市票 | 保持现 1374 样本口径，只补"对应"退市票 | 数据量小 | 样本来源不明、含未知选择偏差；"对应退市票"难定义 → 仍不干净 |

**建议 A**：既然要修偏差，就修彻底；样本口径本身是另一个偏差源。是否纳入 BSE（北交所）一并定（现 universe 无 BSE，建议先不纳入，保持 SH+SZ，单独标注）。
**评审请拍板 scope = A 还是 B，是否含 BSE。** 下面按 A 写。

## 2. 数据拉取（实现期；新 Tushare 请求在本步在 scope 内）
复用 `scripts/backfill_tushare.py` 基础设施（provider 退避/超时、断点续传、loguru token 防泄漏）。

1. **全量 stock_basic**：`stock_basic(list_status='L')` + `('D')` + `('P')` 三调合并 → 全 universe，含 `list_date` / `delist_date` / `list_status`。落 `data/lake/stock_basic_full.parquet`（不覆盖现 stock_basic）。
2. **退市/暂停 + 缺失存续票的在市期价量**：对 (D∪P∪未在现 panel 的 L) 票，逐票拉 `daily`(OHLCV/amount) + `adj_factor`，区间 = `max(2019-01-01, list_date)` → `min(delist_date or 2026-05-11)`。断点续传（按 ts_code）。
   - 退市票停牌段缺口、adj_factor 缺失 → 按已有 prices 口径处理（缺口留 NaN，不前填）。
   - 落 `data/raw/alpha_prices_panel_v6.parquet`（= 现 1374 + 新增退市/缺失票，**新文件**）。

## 3. PIT universe 重建
- 对每个**周频 as_of**（沿用 v5 网格，350 个 as_of，相位不变）：纳入满足 `list_date ≤ as_of AND (delist_date 为空 OR as_of < delist_date)` 的全部票（含后来退市但当时在市的）。
- 退市票在其 `delist_date` 之后**正确剔除**（之后不再出现在 universe，标签/因子自然 NaN）。
- 复用 `weekly_panel.pit_universe`，但**真正生效 delist 过滤**（现版 delist 全空形同虚设）。

## 4. v6 面板构建（复用现 builder，不动 v5）
- 用 `weekly_panel.build_weekly_panel` 的同一套 35 因子 + 三档标签逻辑，**仅换 universe + prices 源**为 v6（全市场 PIT + v6 价量）。
- 输出 `data/panel/alpha_panel_weekly_v6.parquet`（**新文件**；`_save_panel` 已有"绝不覆盖 v5/v4"守卫，扩展到 v6 同样保护）。
- short_horizon 16 因子 + Alpha158 等增量沿用 `merge_increment`（在 v6 网格上重算）。
- **评估侧零改动**：WF/中性化/成本/p3f 全部不变，只是喂 v6 预测/标签。

## 5. 量化影响（交付物）
- 新旧 universe 票数：1374 → N（全市场）；其中 **退市(D) 数、暂停(P) 数、新增存续(L) 数**。
- **退市占比**：2019-2026 间退市票 / 全 universe；分板块（主板/创业板/科创板）。
- **加回行数**：v6 总行数 vs v5 452,439；逐 as_of universe size 曲线（早期 vs 近期，看退市票随时间累积剔除的形状）。
- **幸存者 leakage 估计**：v6 vs v5 在相同 as_of 上的 universe 差异，定性给"v5 漏了多少最终退市的票"。

## 6. 验收 / 自检（实现期硬门）
- **PIT 正确性**：① 无 `list_date > as_of` 的票（无前视）；② 退市票在 `delist_date` 后不出现；③ 抽 3 只退市票，末个在市 as_of 的价/标签存在、之后为空。
- **退市票价量真实性**：抽 3 只退市票，其末日价 vs Tushare 退市公告价一致；adj_factor 链连续。
- **v5 不动**：构建前后 `alpha_panel_weekly_v5.parquet` 字节哈希不变。
- **评估管线零改动**：v6 跑通现有 WF/p3f，无代码改动。

## 7. 风险与缓解
| 风险 | 缓解 |
|---|---|
| Tushare 速率/积分限制（全市场拉取久） | 断点续传 + 退避；分批；夜间跑 |
| 退市票 `daily`/`adj_factor` 数据缺失或不全 | 缺口留 NaN 不前填；记录覆盖率；覆盖太低的票标注 |
| scope=A 数据量大、耗时 | 接受（一次性）；或评审选 B 折中（但不干净） |
| BSE 是否纳入 | 默认不纳入，单独标注；评审定 |
| loguru token 泄漏（历史踩过两次） | 复用现有 `logger.remove()` 防护；拉取前隔离测 leak=0 |

## 8. 修好后的下一步（本计划不含，仅声明顺序）
1. **真实 universe(v6) 上重跑 Ridge(full)** → 看 neut IC **+0.034 / 净 +1.9%** 是否还在。**这一个数字决定有没有产品。**
2. 若 Ridge 在 v6 上仍立 → 63d 产品验证 / batch B 才在 v6 上做（才可信）。
3. illiquid 深度 follow-up：**仅当** v6 上 illiquid 桶的信号在补回退市票后仍存在才议（大概率会大幅缩水）。

---

## ✅ 决策记录（2026-06-09 评审通过，进入实现）
1. **Scope = A：全市场 SH+SZ A 股 PIT（list_status L+D+P 全拉）**。
   - 理由：现 1388 票是**用今日快照（现任指数成分 + 补充）回套的样本**，指数成分本身就是幸存者筛选；B 方案修不干净。
   - **BSE 北交所不纳入**：2021 底开板（历史短）、流动性差、非产品目标池。
2. **数据量接受**：~4800 票全历史，复用 backfill checkpoint/限频，**质量第一不赶时间**。
3. **命名确认**：`data/raw/alpha_prices_panel_v6.parquet` / `data/panel/alpha_panel_weekly_v6.parquet`。

## 9. 方法论要求 A（修复的灵魂，做错=白修）：退市标签规则
- **规则**：对 `as_of` 在市、但 `horizon` 内退市的票，`forward_return` 必须**算到最后可交易日（含退市整理期）**，把**真实亏损写进标签**，**不得 NaN 丢弃**。
  - 即：若 `delist_date ≤ (as_of + horizon 个交易日)`，则 `forward_return = 最后可交易日收盘 / as_of 收盘 − 1`（通常是大额亏损）。
  - 现 `compute_forward_returns` 行为：`horizon` 内数据不足 → 整列 NaN 丢弃 → **退市票的亏损被系统性删掉 = 幸存者 leak 的核心**。
- **实现**：`compute_forward_returns` **新增 delist-aware 路径**（传入 ticker→delist_date 映射；退市票用末个可交易价作 exit）。**不动 v5 现有行为**（v5 走旧路径）。
- **专项验收**：抽 ≥3 只真实退市票，核对其退市前最后几个 `as_of` 的标签 = 到末交易日的真实收益（不是 NaN、不是 0）。

## 10. 方法论要求 B（评估层）：PIT 流动性过滤
- v6 面板建**全市场**；评估侧支持按每个 `as_of` 的**当时 adv20/circ_mv 排名取 top-N** 过滤（**只用 ≤as_of 信息**，PIT 安全），模拟产品可交易池。
- **Ridge 复核须报两个口径**：① 全市场；② **PIT 流动性 top-1500**（与旧池规模可比）。

## 11. 实现顺序（P1-P2 自主推进；P3 证据 + P4 结果带回评审）
- **P1 拉数**：全市场 `stock_basic(L+D+P，含 list_date/delist_date)` → 全票价量/adj/daily_basic 回补进 lake（复用 backfill，token 走 `.env`，loguru 静默规则沿用）。落 checkpoint，报：总票数、退市票数、行数、日期覆盖。
- **P2 v6 universe + 面板**：PIT 纳入 `list_date ≤ as_of < delist_date`；退市标签按要求 A；复用 weekly_panel builder 新路径产 v6；**v5 字节不动**。
- **P3 验收（独立 code-review session）**：PIT 正确性（含未来篡改反证）、退市票价真实性抽查（对照 Tushare 原始）、退市标签专项（要求 A）、新旧 universe 量化对比（票数/退市占比/加回行数/**标签分布差异——退市票应拉低全样本均值**）。
- **P4 判定跑**：v6 上重跑 **Ridge(full) 12d 季度**（同 batch-A 配置），报两个口径（全市场 / PIT top-1500）的 neut IC + 含成本净超额，与 v5 的 **+0.034/+1.9%** 并排。**只复核 Ridge，不跑其他模型。**

## 12. P2 设计决策（2026-06-10 评审补充）
4. **新股 seasoning 过滤**：v6 universe 纳入规则改为
   **`list_date + 120 交易日 ≤ as_of < delist_date`**（PIT 安全，只用 ≤as_of 信息）。
   - 理由：全市场口径引入数千只 2019-2026 新股；A 股新股初期制度性异象（上市初高波动/换手/
     破发反转等）会污染 v5↔v6 对比，把**新股效应**和**幸存者效应**搅混。seasoning 是标准做法。
   - 120 交易日为默认；verify 报告"被 seasoning 排除的 (as_of,ticker) 行数"。
5. **ST 票不特殊处理**：保留在面板（它们正是退市风险载体，删掉=另一种幸存者偏差）。
   产品可交易性由**评估层 PIT 流动性 top-1500 镜头**处理，不在面板层删 ST。记录在案。

### P1 补验结果（before P2）
- **circ_mv 退市票覆盖 = PASS**：抽 6 只退市票，circ_mv **覆盖到末交易日**（与 daily 末日一致）；
  到 delist_date 的 40-52d "gap" = **停牌整理期**（无交易→正确无数据）。
  → 在市(可交易)as_of 上退市票**有 circ_mv**，不会被中性化静默剔除，亏损标签正常进 IC。
  **无后门偏差，无需 fallback 规则。** 停牌期 as_of 因因子 NaN 自然排除（正确）。
- adj_factor 退市票覆盖：待 P1b(daily+adj_factor) 拉完后抽 ≥5 只核对（行数与 daily 一致、末日有值）。
- 全市场覆盖摘要（总行数/票数/退市票"末交易日 vs delist_date"分布）：P1b 完成后报。

## ⚠ 硬性中止条件
拉数遇 **Tushare 积分/接口不覆盖退市票历史** 的情况 → **立即停下报告**，**不得用替代源静默凑数**。

> P1-P2 范围内自主推进；P3 验收证据 + P4 判定结果带回评审。
