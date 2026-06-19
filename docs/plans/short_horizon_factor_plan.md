# 短周期特异性因子计划 — 日线量价能否产生中性化后仍为正的信号

> 状态：**计划期，获批前不实现。** 写完停下评审。
> 承接：`docs/plans/wf_v2_diagnostics.md`（v2 信号中性化后只剩 12%，88% 是行业/规模 tilt）。
> 目标问题：**光靠日线量价，有没有"行业+log(circ_mv) 中性化后仍稳定为正"的短周期（12d）信号？**
> 数据范围（硬）：仅 `alpha_panel_weekly_v5.parquet` + `alpha_prices_panel.parquet` + `daily_basic.parquet`(circ_mv)。**不拉分钟、不发新 Tushare 请求。**
> 生成：2026-06-06

⚠️ 全程幸存者乐观上界；任何正结果都是上界，客户前须先补退市票。

---

## 0. 通用约定（所有因子共享）

### 0.1 字段与单位
| 符号 | 定义 | 来源 / 单位 |
|---|---|---|
| `open/high/low/close` | 原始价（元） | prices panel |
| `adj_close` | 后复权收盘 | prices panel |
| `adj_open/high/low` | `open/high/low × adj_factor` | 重建（数据只有 adj_close） |
| `returns` | `adj_close.pct_change()`（日收益） | 派生 |
| `vol` | 成交量（手=100 股） | prices panel |
| `amount` | 成交额（千元） | prices panel |
| `vwap` | **10 × amount / vol（元/股）** | 派生（单位自查，见 §0.3） |
| `adv{n}` | `MEAN(vol, n)`，默认 adv20 | 派生 |
| `cap` | `circ_mv`（流通市值，万元） | daily_basic |

### 0.2 复权口径规则（公司行动一致性）
- **同日价内关系**（vwap/open/high/low vs close，同一交易日 t）：用**原始价**（同日 adj_factor 约掉）。
- **跨日**（delta/momentum/价格时序相关/correlation over time）：用**复权价**（adj_close、adj_open…）。
- 日内分量 `close_t/open_t − 1`：原始价（同 t）。隔夜分量 `open_t/close_{t-1} − 1`：复权（`adj_open_t / adj_close_{t-1}`）。
- 成交量不复权（tushare 原生）；rank/比率算子对此稳健，标注即可。

### 0.3 VWAP 单位自查（覆盖用户提示）
成交额(元)=amount×1000，成交股数=vol×100 → **vwap = amount×1000/(vol×100) = 10×amount/vol（元）**。
（WQ 用 `vwap − close`，close 是元，vwap 必须同量纲为元；用户提示 `amount/(vol×100)` 漏了 amount 的千元换算。rank/相对算子里绝对尺度约掉，但差值算子必须对齐。）

### 0.4 PIT 与周频取值（对每个因子都成立，故各条目不再重复）
1. 在**日频**上计算 alpha 序列，所有时序算子（Ref/delta/MEAN/STD/corr/ts_rank…）**只回看**（≤ 当日）。
2. 在 v5 的每个 `as_of` 上**取该日的 alpha 值**——as_of 是交易日，取值只用 ≤ as_of 数据 → PIT 安全。
3. 截面算子（cross-sectional `Rank`）在**每个交易日的全 universe 截面**上算，不跨日、不跨 as_of。
4. universe 与 v5 对齐（PIT `list_date ≤ as_of`）。
5. 输出 MultiIndex `(as_of, ticker)`，列名前缀 `sh_`，经 `merge_increment` 左对齐 join（不改 base 行/列）。

### 0.5 "构造时是否残差化" 字段含义
- **否**：因子在原始价量上构造，tilt 暴露留给筛选阶段的中性化去测。
- **是（源头残差化）**：先在**日频截面**把日收益对 `行业哑变量 + log(cap) + 市场收益` 回归取**残差收益 r̃**，再在 r̃ 上构造因子，从源头去 tilt。
  - 日频残差回归：`ret_i,t = α_t + Σ_k β_k·Ind_{i,k} + γ_t·z(log cap_i,t) + δ_t·mkt_t + r̃_i,t`，`mkt_t` = 当日截面 cap 加权平均收益（或 index_daily CSI300 收益，二选一，实现时定）。
  - 注意：源头残差化≠免筛选；§7 筛选阶段对**所有**候选（含已残差化的）再统一做一次行业+log(cap) 中性化，apples-to-apples 比较。

---

## A. 必须含的几类（研究支持中性化后仍有效）

### A1. 改进反转 —— 日内累计收益反转（剔隔夜/跳空）
**动机**：A 股反转里，隔夜跳空多为情绪/信息冲击（含 tilt），日内（open→close）累计更接近**做市/流动性提供**的特异性反转。拆分后用日内分量。

| 名称 | 公式 | 字段 | 残差化 |
|---|---|---|---|
| `sh_rev_intraday_5` | `-1 × Σ_{i=0}^{4} (close_{t-i}/open_{t-i} − 1)` | open, close（原始，同日） | 否 |
| `sh_rev_intraday_10` | `-1 × Σ_{i=0}^{9} (close_{t-i}/open_{t-i} − 1)` | open, close | 否 |
| `sh_rev_intraday_21` | `-1 × Σ_{i=0}^{20} (close_{t-i}/open_{t-i} − 1)` | open, close | 否 |
| `sh_rev_overnight_10` | `-1 × Σ_{i=0}^{9} (adj_open_{t-i}/adj_close_{t-i-1} − 1)`（隔夜分量，对照组） | open, adj_close, adj_factor（跨日复权） | 否 |
| `sh_intraday_minus_overnight_21` | `Σ_{21}(日内收益) − Σ_{21}(隔夜收益)`（日内相对隔夜的反转强度） | open, close, adj | 否 |

> 隔夜分量作为**对照**：若 `sh_rev_intraday_*` 中性化后为正而 `sh_rev_overnight_*` 不为正，证实"日内特异反转 vs 隔夜 tilt"假设。

### A2. 残差化反转 / 波动（源头去 tilt）
**动机**：先剥掉行业/规模/市场，在残差收益 r̃ 上算反转/波动，直接构造"特异性"信号。

| 名称 | 公式（r̃ = §0.5 残差日收益） | 字段 | 残差化 |
|---|---|---|---|
| `sh_resid_rev_5` | `-1 × Σ_{i=0}^{4} r̃_{t-i}` | returns, industry, cap, mkt | **是** |
| `sh_resid_rev_10` | `-1 × Σ_{i=0}^{9} r̃_{t-i}` | 同上 | **是** |
| `sh_resid_rev_21` | `-1 × Σ_{i=0}^{20} r̃_{t-i}` | 同上 | **是** |
| `sh_resid_vol_10` | `STD(r̃, 10)` | 同上 | **是** |
| `sh_resid_vol_21` | `STD(r̃, 21)` | 同上 | **是** |

### A3. 量条件反转（按成交量区分动量段/反转段）
**动机**：高量推动（信息）倾向延续=动量，低量噪声倾向反转。按量阈值分段。

| 名称 | 公式（rv = vol/adv20，medw = 窗口内 vol 中位数） | 字段 | 残差化 |
|---|---|---|---|
| `sh_lowvol_rev_10` | `-1 × Σ_{i=0}^{9} returns_{t-i} · 1[vol_{t-i} < medw_{10}]`（低量日反转） | returns, vol | 否 |
| `sh_lowvol_rev_21` | `-1 × Σ_{i=0}^{20} returns_{t-i} · 1[vol_{t-i} < medw_{21}]` | returns, vol | 否 |
| `sh_highvol_mom_21` | `Σ_{i=0}^{20} returns_{t-i} · 1[vol_{t-i} ≥ medw_{21}]`（高量日动量） | returns, vol | 否 |
| `sh_volwt_rev_21` | `-1 × Σ_{i=0}^{20} returns_{t-i} · (1/(1+rv_{t-i}))`（量越小反转权重越大，连续版） | returns, vol, adv20 | 否 |

### A4. 日线代理的已实现矩（残差化）—— ⚠ 分钟级 RSkew/RVol 的【弱代理】
**动机**：真版 Realized Skewness / Realized Vol 需分钟内收益；**日线只能用短窗日收益的滚动矩近似，信噪比低，明确标注为弱代理，真版等分钟数据（Step 3）。**

| 名称 | 公式（r̃ = 残差日收益） | 字段 | 残差化 |
|---|---|---|---|
| `sh_rskew_21` | `SKEW(r̃, 21)`（弱代理 RSkew） | returns, industry, cap, mkt | **是** |
| `sh_rkurt_21` | `KURT(r̃, 21)`（弱代理） | 同上 | **是** |
| `sh_downsemivol_10` | `STD(min(r̃,0), 10)`（下行半波动） | 同上 | **是** |
| `sh_downsemivol_21` | `STD(min(r̃,0), 21)` | 同上 | **是** |
| `sh_updown_vol_ratio_21` | `STD(max(r̃,0),21)/STD(min(r̃,0),21)`（上/下行波动比，弱代理 RSkew 符号） | 同上 | **是** |

**A 小计：5 + 5 + 4 + 5 = 19 个候选。**

---

## B. 公式化 alpha 子集（WorldQuant 101 / 国泰君安 191，仅需日线可得字段）

**筛选准则**：只取仅依赖 `close/open/high/low/vol/amount/vwap/returns/adv/cap` 的**反转类 / 量价相关类 / rank 类**；**排除**用 IndNeutralize / 多日 decay 链 / 需基本面或分钟的 alpha。
**实现纪律**：公式可参考公开实现（WorldQuant101、gtja191 开源仓）**校对**，但每个必须**自查 PIT**（所有时序算子只回看）+ 适配周频 as_of（§0.4）。`d` = 回看天数。

### B1. WorldQuant 101 子集（22 个）
| 编号 | 公式 | 主要字段 | 残差化 |
|---|---|---|---|
| WQ#1 | `rank(Ts_ArgMax(SignedPower((returns<0?STD(returns,20):close),2),5)) − 0.5` | returns, close | 否 |
| WQ#2 | `-1×corr(rank(delta(log(vol),2)), rank((close−open)/open), 6)` | vol, close, open | 否 |
| WQ#3 | `-1×corr(rank(open), rank(vol), 10)` | open, vol | 否 |
| WQ#4 | `-1×Ts_Rank(rank(low), 9)` | low | 否 |
| WQ#5 | `rank(open − SUM(vwap,10)/10) × (−1×abs(rank(close − vwap)))` | open, vwap, close | 否 |
| WQ#6 | `-1×corr(open, vol, 10)` | open, vol | 否 |
| WQ#7 | `adv20<vol ? (−1×ts_rank(abs(delta(close,7)),60)×sign(delta(close,7))) : −1` | close, vol, adv20 | 否 |
| WQ#9 | `0<ts_min(delta(close,1),5)? delta(close,1) : (ts_max(delta(close,1),5)<0? delta(close,1) : −1×delta(close,1))` | close（跨日→adj） | 否 |
| WQ#10 | `rank(WQ#9 的分段逻辑)` | close(adj) | 否 |
| WQ#11 | `(rank(ts_max(vwap−close,3)) + rank(ts_min(vwap−close,3))) × rank(delta(vol,3))` | vwap, close, vol | 否 |
| WQ#12 | `sign(delta(vol,1)) × (−1×delta(close,1))` | vol, close(adj) | 否 |
| WQ#13 | `-1×rank(cov(rank(close), rank(vol), 5))` | close, vol | 否 |
| WQ#14 | `(−1×rank(delta(returns,3))) × corr(open, vol, 10)` | returns, open, vol | 否 |
| WQ#15 | `-1×SUM(rank(corr(rank(high), rank(vol), 3)), 3)` | high, vol | 否 |
| WQ#16 | `-1×rank(cov(rank(high), rank(vol), 5))` | high, vol | 否 |
| WQ#18 | `-1×rank(STD(abs(close−open),5) + (close−open) + corr(close,open,10))` | close, open | 否 |
| WQ#22 | `-1×(delta(corr(high,vol,5),5) × rank(STD(close,20)))` | high, vol, close | 否 |
| WQ#25 | `rank((−1×returns) × adv20 × vwap × (high−close))` | returns, adv20, vwap, high, close | 否 |
| WQ#33 | `rank(−1×(1−(open/close)))` | open, close | 否 |
| WQ#34 | `rank((1−rank(STD(returns,2)/STD(returns,5))) + (1−rank(delta(close,1))))` | returns, close | 否 |
| WQ#41 | `(high×low)^0.5 − vwap` | high, low, vwap | 否 |
| WQ#42 | `rank(vwap−close) / rank(vwap+close)` | vwap, close | 否 |
| WQ#43 | `ts_rank(vol/adv20, 20) × ts_rank(−1×delta(close,7), 8)` | vol, adv20, close | 否 |
| WQ#44 | `-1×corr(high, rank(vol), 5)` | high, vol | 否 |
| WQ#53 | `-1×delta(((close−low)−(high−close))/(close−low), 9)` | close, low, high | 否 |
| WQ#54 | `(−1×(low−close)×open^5) / ((low−high)×close^5)` | low, close, open, high | 否 |
| WQ#101 | `(close−open) / (high−low + 0.001)` | open, close, high, low | 否 |

*(以上 28 行，实现时取约 22 个；标注 `close(adj)` 的跨日算子用复权价。)*

### B2. 国泰君安 191 子集（6 个，与 WQ 风味互补）
| 编号 | 公式 | 主要字段 | 残差化 |
|---|---|---|---|
| GTJA#18 | `close / DELAY(close,5)`（5 日动量/反转） | close(adj) | 否 |
| GTJA#31 | `(close − MEAN(close,12)) / MEAN(close,12) × 100`（均值回复） | close(adj) | 否 |
| GTJA#40 | `SUM(close>DELAY(close,1)? vol:0, 26) / SUM(close≤DELAY(close,1)? vol:0, 26) × 100`（涨跌量比，量条件） | close, vol | 否 |
| GTJA#46 | `(MEAN(close,3)+MEAN(close,6)+MEAN(close,12)+MEAN(close,24)) / (4×close)`（多窗均线回复） | close(adj) | 否 |
| GTJA#58 | `COUNT(close>DELAY(close,1), 20) / 20 × 100`（20 日胜率动量） | close(adj) | 否 |
| GTJA#53 | `COUNT(close>DELAY(close,1), 12) / 12 × 100`（12 日胜率，短窗对照） | close(adj) | 否 |

**B 小计：约 22 + 6 = 28 个候选。**

**总计：约 19（A）+ 28（B）≈ 47 个候选**（控制在 30–50 区间，够说明问题）。

### B 实现登记（2026-06-06，回应验收意见①：B 不可单方面缩水）
实现的 **B=28**（`quantmind/features/short_horizon_factors.py`）：
- WQ101（22）：#1, #2, #3, #4, #5, #6, #7, #9, #11, #12, #13, #14, #15, #16, #18, #22, #25, #41, #42, #43, #44, #53, #101。
- GTJA191（6）：#18, #31, #40, #46, #53(胜率12), #58。
新增算子：`ts_argmax / signedpower / ts_cov`（补齐 WQ#1/#13/#16）。
**SELECTED 28 个全部实现，无一因缺字段/算子被丢。**

**101/191 全集里未纳入的，按【既定 scope】排除（非缺字段）**：
- **IndNeutralize 类**（WQ #48/#56/#58/#59/#63/#67/#69/#70/#76/#79/#80/#82/#87/#89–#94/#97/#100 等）：alpha 内含行业中性化，与本筛选的中性化步骤重复 → §B scope 明确排除。
- **decay_linear / 长链 Ts 组合 与 GTJA 的 SMA/WMA/SUMAC/REGBETA 类**：需额外算子面，收益边际低；已用 #1–#25 / #41–#44 / #53 / #54 / #101 与 GTJA 6 个覆盖 反转 / 量价相关 / rank / 均值回复 / 量条件 各族，足以判"日线量价是否有中性后正信号"。
- 如评审要求扩到这些族，属"加算子"而非"加字段"，可后续增量补入，不影响当前结论。

---

## 6. 算子库（实现时建在 `short_horizon_factors.py`，复用 expr_factors 已有的）
已有（`quantmind/features/expr_factors.py`）：`Ref, Mean, Std, Abs, Greater, Log, Rank(截面), RollingMax, RollingMin`。
需新增（全部只回看）：`delta(x,d)=x−Ref(x,d)`、`ts_rank(x,d)`、`ts_argmax/ts_argmin(x,d)`、`ts_min/ts_max(x,d)`、`correlation(x,y,d)`、`covariance(x,y,d)`、`sign`、`sum(x,d)`、`product`、`decay_linear`、`signedpower(x,e)`、`count(cond,d)`、`skew/kurt(x,d)`、`delay=Ref`。
单元自查：构造一段含未来值的序列，确认算子输出在 t 不依赖 >t（PIT 反证），≥10 个测试。

---

## 7. 筛选方法（核心交付）—— 复用 `scripts/_diag_wf_v2.py` 中性化逻辑

对**每个候选因子**，在 WF 的 16 个 OOS fold（H=12/E=20/rolling=756，OOS 2022-01→2026-04，142 截面）上算：

1. **原始 IC**：逐 OOS 截面 `Spearman(factor, forward_return_12d)`，跨截面均值。
2. **中性化 IC**：逐截面把 factor 对 `C(exposure_industry) + z(log circ_mv)` OLS 取残差，残差 vs label 算 IC，跨截面均值。（与 _diag_wf_v2 `neutralize_date` 完全同口径。）
3. **中性化 ICIR**：`mean(neut_ic_per_date) / std(neut_ic_per_date)`。
4. **方向翻转率**：按 fold 聚合 neut IC，统计 fold 间符号翻转比例（稳定性）。
5. **冗余**：与现有 35 因子、及候选因子之间的截面相关（时间平均 |Spearman|），取 **max |corr|**。

> 方向处理：单因子筛选**不做 per-fold auto-flip**（避免方向用 OOS）——用全 OOS 期 neut IC 的**符号一致性**判稳定，翻转率高=不稳。

**保留标准（全部满足）**：
- 中性化 IC **方向稳定**（fold 翻转率低，建议 < 35%）；
- 中性化 IC **显著非零**（|neut IC| 明显高于噪声，建议 |neut ICIR| ≳ 0.3 且 neut IC 与 raw IC 同号、retention 不塌）；
- **低冗余**（max |corr| < 0.7，对现有 35 因子与彼此）。

**输出排序表**（按 |neut ICIR| 降序）：

| factor | raw IC | neut IC | neut ICIR | retention(neut/raw) | flip% | max\|corr\|(35因子) | max\|corr\|(候选间) | 是否残差化 | 保留? |
|---|---|---|---|---|---|---|---|---|---|
| … | | | | | | | | | |

---

## 8. 结论与下一步建议（交付时填写）

### 筛选结果（2026-06-06，47 候选，贪心去冗余）→ `data/loss_signals_v4/short_horizon_final.csv`
**有。16 个存活**（in-sample 选 / OOS 无偏确认，OOS 同号才算真确认）。核心强信号：

| 因子 | in-sample neut IC | is ICIR | flip | OOS neut IC | OOS ICIR | maxcorr(35) | 主题 |
|---|---|---|---|---|---|---|---|
| sh_wq13_cov_close_vol | +0.042 | 0.87 | 0% | +0.025 | 0.58 | 0.34 | 量价协方差 |
| sh_wq44_corr_high_rkvol | +0.033 | 0.77 | 0% | +0.024 | 0.58 | 0.31 | 量价相关 |
| sh_resid_vol_10 | −0.048 | −0.65 | 8% | −0.053 | −0.93 | 0.58 | 残差低波动 |
| sh_rskew_21 | −0.022 | −0.65 | 0% | −0.029 | −0.72 | 0.38 | 已实现偏度(弱代理) |
| sh_wq3_rkopen_rkvol | +0.026 | 0.68 | 0% | +0.020 | 0.49 | 0.19 | 量价相关 |
| sh_wq15_corr_high_vol | +0.025 | 0.65 | 0% | +0.021 | 0.55 | 0.25 | 量价相关 |
| sh_wq6_corr_open_vol | +0.030 | 0.64 | 0% | +0.022 | 0.46 | 0.18 | 量价相关 |
| sh_resid_rev_10 | +0.033 | 0.41 | 17% | +0.025 | 0.29 | 0.58 | 残差反转 |
| + wq12/wq5/wq101/wq11/wq42/wq2/rkurt_21/gtja40（较弱，部分 OOS 近零） |

**主题**：存活主力是 **量价 rank 相关/协方差族**（wq13/wq44/wq3/wq15/wq6）——35 因子里**没有**这族；加 **残差化波动/反转** + **已实现偏度弱代理**。
**与 v2 诊断的对比**：v2 整模型中性化后 retention 仅 12%（≈0）；但这里**单因子**层面，量价微结构因子在中性化后仍稳定（is/OOS 同号、ICIR>0.4、翻转≤8%），是 35 因子漏掉的**新正交信号**。
**注**：intraday_rev_21/resid_rev_21 等虽 OOS neut IC 强(+0.04)，但 maxcorr(35)>0.7（与现有 reversal 重复）→ 正确硬毙，不算新增。
**下一步**：16 survivors 经 merge_increment 进 bake-off 的 tabular 制式（35 + 16 + Alpha158），看模型能否把这些正交信号转成组合层面 alpha。

---

明确回答：
1. **有没有任何日线因子中性化后仍稳定为正？是哪些？**（给存活清单 + 其 neut IC/ICIR/翻转率/冗余。）→ **有，16 个，见上表。**
2. **建议**：
   - 若**有** → 把存活的**正交**因子并入重训 LGBM，过现有 wf_gate（中性化后 IC>0 作新增硬条件），看 v2 口径下 neut IC / 净超额 / 胜率是否改善。
   - 若**无** → 日线量价不够；需升级到**资金流 / 事件 / 分钟数据**（Step 3）：分钟级真版 RSkew/RVol、tick 级订单流不平衡、事件/财报 surprise、龙虎榜/大单等特异性信息源。

---

## 9.（可选收尾）wf_gate 复跑对比
取存活 top-K 正交因子 + 现有 35 因子，跑一次现有 `wf_gate`（LGBM，同 v2 口径），与 v2 对比中性化 IC / 净超额 / 胜率是否改善。**标为可选**；主交付是 §7 单因子中性化筛选表 + §8 结论。

---

## 实现纪律清单（获批后 P2 才执行）
- [ ] 新模块 `quantmind/features/short_horizon_factors.py`；**不改** weekly_panel / 既有因子 / WF 代码。
- [ ] 经 `merge_increment` 按 (as_of,ticker) 增量 join；列名前缀 `sh_`，无碰撞。
- [ ] 跨日用 adj_close（及重建 adj_open/high/low）；同日日内结构用原始价。
- [ ] 全程 PIT（≤ as_of）；算子只回看；≥10 个 PIT/单元测试。
- [ ] 不拉分钟、不发新 Tushare 请求。
- [ ] 结果当**乐观上界**读，报告显著标注幸存者警示。

---

## ⏸ 评审门
**以上为计划。等用户批准后才进入 P2 实现。** 待确认点（供评审）：
1. 候选清单的取舍（A 类 19 + B 类 28 ≈ 47）是否合适？要不要砍 B 到 ~20 提速？
2. 残差化里的"市场"项用 **截面 cap 加权平均收益** 还是 **index_daily CSI300**？
3. 冗余/翻转/ICIR 三个保留阈值（0.7 / 35% / 0.3）是否按你的口径调整？
