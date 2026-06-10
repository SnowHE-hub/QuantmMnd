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

## ⏸ 评审门
**待确认（评审请拍板）：**
1. **Scope = A（全市场 PIT，推荐）还是 B（样本+退市票）？是否纳入 BSE？**
2. 数据量/耗时是否接受（A 约 4800+ 票全拉）？
3. v6 命名 / 落盘路径（`alpha_prices_panel_v6` / `alpha_panel_weekly_v6`）是否 OK？

获批后才进入 §2 实现（拉数据）。**本计划只写到这，停下评审。**
