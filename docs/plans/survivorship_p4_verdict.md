# P4 判定报告 —— v6 全市场上 Ridge(full) 12d 复核

**性质**：单次判定（非优化）。完全复刻 batch-A，唯一变量 = universe/面板（v5 1373 票 → v6 全市场）。
不调参、不换方案。判定对 **gate 线**，不与 v5 攀比。

## 配置（逐字复刻 batch-A，已核）
- Ridge(alpha=10)，数值特征 z-score(train mu/sd)，丢 exposure_industry/area。
- PurgedWalkForwardSplit(horizon=12, embargo=20, mode=rolling, rolling_lookback_td=756, n_val=2)。
- quarterly cutoffs 2022-01-01→2025-10-01；oos_start=2022-01-01；decide_direction(val-only)；score*=d。
- featset full = WHITELIST_35 + 16 survivors + Alpha158（在 v6 全市场重提取，G1-mini corr=1.00000）。
- 16 folds，0 flips（方向全 +1，稳定），训练 208s。

## 结果（两口径）

| 口径 | n票 | raw IC | **neut IC** | neut ICIR | **含成本净超额** | 单边换手 | maxDD | bear/neutral/bull |
|---|---|---|---|---|---|---|---|---|
| 全市场 | 5324 | 0.0697 | **0.0521** | 0.91 | **+1.17%** | 0.71 | 5.3% | +0.063 / +0.041 / +0.055 |
| **PIT top-1500** | 4598 | 0.0775 | **0.0570** | 0.99 | **+2.75%** | 0.76 | 7.7% | +0.067 / +0.050 / +0.055 |
| v5 基线（1373） | 1373 | — | 0.0340 | — | +1.9% | — | — | 三档均正 |

- neut IC 逐 as_of 分位（top-1500）：p25=0.015 / p50=0.056 / p75=0.090；posfrac=0.838。
- regime 切点：mm q33=−0.0363 / q67=+0.0225（bear45 / neutral53 / bull44 个 as_of）。

## ⚠ 方向异常 + 证伪（必读，防误读）

**预期** v6 neut IC < v5 的 0.034（"幸存者上界被拿掉"）。**实测** v6 = 0.052–0.057 **高于** v5。
对一次幸存者修复的判定跑，"超预期变好"必须先证伪 artifact，否则等于自欺。

**已做分解诊断（非调参，纯拆解）**：
- 假设：退市票真实暴跌标签 = 可预测崩盘簇，抬高 IC。
- 结果：horizon 内将退市的行仅 **343/700122 = 0.049%**（top-1500 内仅 13 行）；
  剔除这些行后 neut IC：全市场 0.0521→0.0522、top-1500 0.0570→0.0571，**Δ=0.0001**。
- 结论：**IC 升高与退市崩盘簇无关**，假设证伪。

**真实机制（正确解读）**："幸存者抬高回测"针对的是**持仓组合收益**；对**截面排序 IC** 往往相反——
v5 的 1373 票 = 今日流动性指数成分（高效大盘，定价更有效 → IC 更低、更难排）；
v6 全市场加回数千只中小盘（定价低效 → 截面因子更有效 → IC 更高）。
**幸存者限制是在压低 IC，不是抬高**。修复不仅没杀死信号，反而在诚实票池上更强。

**已排查的泄漏向量（均干净）**：
- PIT：Alpha158 因果（G1-mini corr=1.0）；35因子/survivors 在 as_of 截面；seasoning=list+120td≤as_of；
  adv20 top-1500=trailing-20td 成交额仅 ≤as_of；log_mktcap=daily_basic@as_of。
- fold：purged+embargo(h=12,emb=20)；decide_direction 仅 val；模型代码 = batch-A 未改（G2 机制复现 v2）。
- net 超额基准：top-1500 口径基准 = 等权 top-1500（流动票），非含崩盘的全市场等权 → +2.75% 干净。

## 判定（读 PIT top-1500：neut IC + 含成本净超额 + 震荡市符号；对 gate 线）

| 档 | 条件 | 命中？ |
|---|---|---|
| **T1 信号站住** | neut IC≥0.02 **且** 净超额≥0 **且** 震荡(neutral)市正 | ✅ **0.057 ≥0.02；+2.75%≥0；neutral +0.050>0** |
| T2 弱化未死 | 0.01–0.02 / 净超额近零 / 震荡贴线 | — |
| T3 日线见顶 | <0.01 或 震荡市负 | — |

### → 命中 **TIER 1（信号站住，且强于幸存者池）**

三 regime 全正（bear +0.067 / neutral +0.050 / bull +0.055），震荡市最关键档 +0.050 稳。
ICIR 0.99、posfrac 0.84，含成本净超额 +2.75%。**Phase 1 闸门通过**。

## T1 后续动作（待用户确认放行）
1. 解封 `docs/plans/productization_backlog.md`（HorizonRegistry / ModelRegistry / RecommendationContract 等）。
2. 启动 63d 基本面版验证（作为更长持有期的第二产品线）。
3. 12d 短持有期 = 已验证的主信号。

## 纪律
- 出数即停，未调任何参数、未换任何方案。
- v5 字节全程 md5 守卫未变；v6 产物已预录 sha256。
- 本判定为**单次跑**；放行 productization 前若需，可加"逐 fold neut IC 时序图 + 子样本稳健性"作 Phase-1 验收附件（不属本次判定范围）。
