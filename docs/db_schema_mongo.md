# QuantMind MongoDB Collection 设计

> **状态**：纯规划文档，2026-06-02 起草。MongoDB 尚未安装，本文档只设计 schema 和索引。
> 数据库名：`quantmind`

---

## 为什么用 MongoDB？

这些数据源的共同特征：
1. **字段不固定**：随模型版本演化（v1 没有 `cnn_score`，v2 有；agent_signals 字段随 Agent 数量变化）
2. **深层嵌套**：`top10[].key_factors`、`agent_signals.ValuationAgent.signal` 天然是 BSON document
3. **读取模式**：前端直接 `json.loads(file)` → 改成 `collection.find_one({"_id": date})`，改动量最小
4. **写入频率低**：每日一次，无并发冲突

---

## Collection 一览

| Collection | 文档数量 | 主要查询 | _id 格式 |
|-----------|---------|---------|---------|
| `recommendations` | ~15 → 每日+1 | 按日期查最新推荐 | `"2026-06-01"` |
| `positions` | ~20 → 持续变化 | 按 status 过滤 | ObjectId |
| `agent_analysis` | ~10 stock/日 | 按 {date, ticker} | `"2026-06-01_000001.SZ"` |
| `strategy_config` | 1（版本化） | 读最新 | `"v2"` |
| `loss_signals` | ~1/日 | 读最新，按日期查历史 | `"2026-06-01"` |
| `sim_daily` | ~60 → 持续增长 | 按 run_id + date | `"sim30d_20251009"` |
| `watchlist_scores` | ~1/日 | 按日期查 | `"2026-06-01"` |

---

## Collection 1: `recommendations`

**来源**：`data/recommendations/{date}/top10.json`

**Document 结构**：
```json
{
  "_id": "2026-06-01",
  "as_of": "2026-06-01",
  "generated_at": "2026-06-01T17:15:07",
  "model_version": "v2_30day_sim",
  "top10": [
    {
      "rank": 1,
      "ticker": "000712.SZ",
      "name": "锦龙股份",
      "industry": "证券",
      "lgbm_rank": 7,
      "lgbm_score": 0.9956,
      "cnn_score": 0.9782,
      "ensemble_score": 0.9895,
      "entry_price": 11.05,
      "raw_pe_ttm": 27.24,
      "raw_pb": 3.71,
      "raw_roe": -0.59,
      "key_factors": {
        "pe_ttm": -0.135,
        "pb": 0.208,
        "roe_ttm": -0.016,
        "momentum_6m": 0.147,
        "accruals": -0.429
      },
      "agent_signals": {
        "ValuationAgent": {"signal": 0.72, "confidence": 0.85, "summary": "估值合理"},
        "MomentumAgent":  {"signal": 0.61, "confidence": 0.70, "summary": ""},
        "QualityAgent":   {"signal": 0.80, "confidence": 0.90, "summary": ""},
        "TechnicalAgent": {"signal": 0.55, "confidence": 0.65, "summary": ""},
        "SentimentAgent": {"signal": 0.67, "confidence": 0.75, "summary": ""},
        "MacroAgent":     {"signal": 0.59, "confidence": 0.60, "summary": ""}
      },
      "reason": "（跳过 LLM，仅 LGBM 排名）"
    }
  ]
}
```

**索引**：
```javascript
// _id 默认索引（按日期查最新）
db.recommendations.createIndex({ "as_of": 1 })
db.recommendations.createIndex({ "top10.ticker": 1 })   // 查某股历史被推荐记录
```

**注意**：`key_factors` 和 `agent_signals` 字段的 key 随模型迭代可变，MongoDB 的 schema-less 天然支持，无需 migration。

---

## Collection 2: `positions`

**来源**：`data/paper_trading/forward_positions.json`（positions 数组展开，每条一个 document）

**Document 结构**：
```json
{
  "_id": {"$oid": "auto"},
  "as_of": "2026-05-27",
  "ticker": "300139.SZ",
  "predicted_rank": 34,
  "predicted_score": 0.9760,
  "holding_period": "3m",
  "entry_price": 46.69,
  "exit_price": null,
  "estimated_exit_date": "2026-08-24",
  "actual_return": null,
  "status": "OPEN",
  "model_version": "v2_30day_sim",
  "created_at": "2026-06-01T17:15:07"
}
```

**索引**：
```javascript
db.positions.createIndex({ "status": 1 })                     // 过滤 OPEN 持仓
db.positions.createIndex({ "as_of": 1, "ticker": 1 }, { unique: true })
db.positions.createIndex({ "ticker": 1 })                     // 查某股所有持仓记录
db.positions.createIndex({ "estimated_exit_date": 1 })        // 到期提醒
```

---

## Collection 3: `agent_analysis`

**来源**：`reports/investment_pipeline/{date}/strategies.json`（数组展开，每股一 document）

**Document 结构**：
```json
{
  "_id": "2026-06-01_000712.SZ",
  "date": "2026-06-01",
  "ticker": "000712.SZ",
  "name": "锦龙股份",
  "rating": "BUY",
  "composite_signal": 0.72,
  "confidence": 0.81,
  "target_price_1m": 12.5,
  "target_price_3m": 14.0,
  "stop_loss_price": 9.8,
  "position_size": 0.08,
  "holding_horizon": "3m",
  "investment_thesis": "...",
  "key_risks": ["监管风险", "利率上行"],
  "key_catalysts": ["业绩改善"],
  "agent_signals": {
    "ValuationAgent":  {"signal": 0.72, "confidence": 0.85, "summary": "PE 处于历史低位"},
    "MomentumAgent":   {"signal": 0.61, "confidence": 0.70, "summary": ""},
    "QualityAgent":    {"signal": 0.80, "confidence": 0.90, "summary": "ROE 稳定"},
    "TechnicalAgent":  {"signal": 0.55, "confidence": 0.65, "summary": ""},
    "SentimentAgent":  {"signal": 0.67, "confidence": 0.75, "summary": ""},
    "MacroAgent":      {"signal": 0.59, "confidence": 0.60, "summary": ""}
  }
}
```

**索引**：
```javascript
db.agent_analysis.createIndex({ "date": 1, "ticker": 1 })    // DataService.get_agent_analysis() 的主要查询
db.agent_analysis.createIndex({ "ticker": 1, "date": -1 })   // 按股查历史分析
db.agent_analysis.createIndex({ "rating": 1, "date": -1 })   // 按评级过滤
```

---

## Collection 4: `strategy_config`

**来源**：`data/paper_trading/strategy_config_v2.json`

**Document 结构**：
```json
{
  "_id": "v2",
  "version": "v2_30day_sim_optimized",
  "updated_at": "2026-06-01T13:29:52",
  "source": "30day_sim_2025Q4_全A股",
  "holding_period": {
    "recommended": "3m",
    "rationale": "IR=+1.799, 期胜率=96.7%, 均值=+21.56%",
    "short_term_warning": "1w/2w/21d 均为负收益"
  },
  "system1_updates": {
    "lgbm_ic_3m": 0.0339,
    "recommendation": "..."
  },
  "system2_updates": {
    "em_factor_weight": 0.2,
    "weights_calibrated": {
      "value": 0.15,
      "momentum": 0.1385,
      "quality": 0.503,
      "technical": 0.2084
    },
    "reversed_dim_factors": [
      {"factor": "value_score", "ic_3m": -0.062, "action": "DOWNWEIGHT"}
    ]
  }
}
```

**说明**：`_id = "v2"` 固定，每次更新用 `replace_one({"_id": "v2"}, doc)` 覆盖。保留版本历史可改为 `_id = timestamp`。

**索引**：无需额外索引（单文档，按 `_id` 查）

---

## Collection 5: `loss_signals`

**来源**：`data/loss_signals_v4/latest.json`（每日快照）

**Document 结构**：
```json
{
  "_id": "2026-06-01",
  "run_ts": "2026-06-01T17:05:04",
  "signal_1_ranking_loss": {
    "value": 0.5172,
    "recent_ic": -0.0344,
    "trend": "IMPROVING",
    "n_recent_rounds": 3,
    "alert": true
  },
  "signal_2_factor_decay": {
    "value": 0.000838,
    "compared_rounds": [0, 7],
    "n_factors_analyzed": 38,
    "strongly_decaying": [],
    "mildly_decaying": ["accruals", "reversal_1w"],
    "improving": ["turnover_3m_avg", "rsi_14", "momentum_1m"]
  },
  "action_plan": [
    {"priority": 1, "action": "增加 quality 维度权重", "rationale": "..."}
  ]
}
```

**索引**：
```javascript
db.loss_signals.createIndex({ "run_ts": -1 })     // 查最新报告
db.loss_signals.createIndex({ "signal_1_ranking_loss.alert": 1 })  // 过滤高风险日期
```

---

## Collection 6: `sim_daily`

**来源**：`data/sim30d/daily/YYYYMMDD.json` + `data/iteration/*/daily/YYYYMMDD.json`

**Document 结构**：
```json
{
  "_id": "sim30d_20251009",
  "run_id": "sim30d",
  "date": "20251009",
  "day_index": 1,
  "system1_candidates": [
    {
      "rank": 1,
      "ticker": "300917.SZ",
      "name": "特发服务",
      "industry": "房产服务",
      "lgbm_score": 0.9507,
      "lgbm_score_raw": 0.0186
    }
  ],
  "selected_positions": [],
  "portfolio_metrics": {
    "nav": 1.0,
    "daily_return": 0.0
  }
}
```

**索引**：
```javascript
db.sim_daily.createIndex({ "run_id": 1, "date": 1 })
db.sim_daily.createIndex({ "system1_candidates.ticker": 1 })  // 查某股在哪些日期入围
```

---

## Collection 7: `watchlist_scores`

**来源**：`data/watchlist/scores/{date}.json`

**Document 结构**：
```json
{
  "_id": "2026-06-01",
  "date": "2026-06-01",
  "scores": [
    {
      "ticker": "000001.SZ",
      "composite_score": 0.82,
      "value_score": 0.75,
      "momentum_score": 0.88
    }
  ]
}
```

**索引**：
```javascript
db.watchlist_scores.createIndex({ "date": -1 })
db.watchlist_scores.createIndex({ "scores.ticker": 1 })
```

---

## 不迁移到 MongoDB 的 JSON 文件

| 文件 | 原因 |
|------|------|
| `data/cache/shared/*.json`（meta.json） | 缓存辅助文件，过期即删，不需持久化 |
| `data/features/top_factors_*.json` | 小型配置 JSON（~KB级），保留为文件即可 |
| `data/panel/split_meta.json` | 训练集切分元数据，随模型重建，不需跨会话持久化 |
| `data/meta_learner/*.json` | 模型元数据，与 pkl 文件配套，保留文件更简单 |
| `data/iteration/*/nav/*.json` | 迁移到 PostgreSQL `nav_curve` 表 |
| `data/snapshots/*/meta.json` | 轻量元信息，保留为文件 |

---

## 迁移顺序建议

按照数据重要性和迁移难度排序：

1. **Phase 1**（高价值，低风险）：`recommendations` + `positions` + `agent_analysis`
   - DataService 中 `get_recommendations()` / `get_agent_analysis()` / `_forward_positions_raw()` 直接受益
2. **Phase 2**（中等优先级）：`loss_signals` + `sim_daily` + `watchlist_scores`
3. **Phase 3**（可选）：`strategy_config`（当前 1 个文件，文件方式也可接受）
