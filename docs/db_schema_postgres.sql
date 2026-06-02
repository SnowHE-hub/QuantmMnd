-- QuantMind PostgreSQL Schema
-- 设计原则：
--   * 无 TimescaleDB（本机未安装）→ 用 BRIN 索引替代 hypertable（时序扫描等效）
--   * 时序大表均有 (ts_code/ticker, trade_date/as_of) 复合主键
--   * alpha_prices_panel（2.27M 行）和 alpha_panel（~10M 行）是最大的表，
--     按 trade_date/as_of 做 BRIN 索引；按 ts_code 做 B-Tree 索引
--   * 所有日期字段用 DATE 类型（不用 TIMESTAMP），与 pandas 行为一致
--   * 浮点字段用 DOUBLE PRECISION（对应 pandas float64），避免精度损失
-- ============================================================

-- ── 前置：建数据库（psql 外执行，需超级用户）──────────────────
-- CREATE USER quantmind WITH PASSWORD 'your_password';
-- CREATE DATABASE quantmind OWNER quantmind;
-- \c quantmind

-- ============================================================
-- 1. 参考表：alpha_universe（股票池，1374 行，不变频）
-- ============================================================
CREATE TABLE IF NOT EXISTS alpha_universe (
    ts_code         VARCHAR(12)  NOT NULL,
    name            VARCHAR(40),
    industry        VARCHAR(40),
    market          VARCHAR(10),
    list_date       DATE,
    in_csi300       BOOLEAN,
    in_csi500       BOOLEAN,
    in_csi800       BOOLEAN,
    PRIMARY KEY (ts_code)
);

COMMENT ON TABLE alpha_universe IS '股票池定义（CSI300/500/800 成分标记），1374 行';

-- ============================================================
-- 2. 全A股日行情（alpha_prices_panel）
--    原始：2,273,529 行 × 13 列，最大表
--    索引策略：复合主键 (ts_code, trade_date)，BRIN on trade_date
-- ============================================================
CREATE TABLE IF NOT EXISTS price_daily (
    ts_code         VARCHAR(12)   NOT NULL,
    trade_date      DATE          NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    pre_close       DOUBLE PRECISION,
    change          DOUBLE PRECISION,
    pct_chg         DOUBLE PRECISION,
    vol             DOUBLE PRECISION,          -- 成交量（手）
    amount          DOUBLE PRECISION,          -- 成交额（千元）
    adj_factor      DOUBLE PRECISION,          -- 复权因子
    adj_close       DOUBLE PRECISION,          -- 后复权收盘价
    PRIMARY KEY (ts_code, trade_date)
);

-- BRIN 索引（时序范围扫描效率接近 TimescaleDB hypertable，B-Tree 的 1/100 大小）
CREATE INDEX IF NOT EXISTS idx_price_daily_date  ON price_daily USING BRIN (trade_date);
CREATE INDEX IF NOT EXISTS idx_price_daily_code  ON price_daily (ts_code);

COMMENT ON TABLE price_daily IS '全A股日行情（原 alpha_prices_panel.parquet），2.27M 行';

-- ============================================================
-- 3. Alpha universe 子集日行情（daily_prices_panel）
--    原始：728,370 行 × 11 列
--    说明：与 price_daily 字段集类似但列略少，
--          迁移时可合并到 price_daily（用 ts_code 区分）
--          或单独保留以保持向后兼容
-- ============================================================
-- 注：如果决定合并到 price_daily，此表可省略。
-- 单独保留方案（字段较少，adj_factor/pre_close/change 可能为 NULL）：
-- 由于 price_daily 已包含全A股数据，daily_prices_panel 是其子集，
-- 建议迁移时直接使用 price_daily，DataService 加 WHERE ts_code IN (...) 过滤。
-- 此处注释掉以避免数据重复存储。
-- CREATE TABLE IF NOT EXISTS price_daily_alpha_subset ...（见 price_daily）

-- ============================================================
-- 4. 指数日行情（index_daily_panel）
--    原始：6,064 行 × 8 列
-- ============================================================
CREATE TABLE IF NOT EXISTS index_daily (
    ts_code         VARCHAR(12)   NOT NULL,    -- 指数代码如 399300.SZ
    trade_date      DATE          NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    pct_chg         DOUBLE PRECISION,
    vol             DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily USING BRIN (trade_date);

COMMENT ON TABLE index_daily IS '指数日行情（CSI300/500/800 等），6064 行';

-- ============================================================
-- 5. 日度行情基本面（alpha_daily_basic）
--    原始：37,494 行 × 10 列
--    包含估值指标（PE/PB/市值），每日快照
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_basic (
    ts_code         VARCHAR(12)   NOT NULL,
    trade_date      DATE          NOT NULL,
    pe_ttm          DOUBLE PRECISION,
    pb              DOUBLE PRECISION,
    ps_ttm          DOUBLE PRECISION,
    dv_ratio        DOUBLE PRECISION,          -- 股息率
    total_mv        DOUBLE PRECISION,          -- 总市值（万元）
    circ_mv         DOUBLE PRECISION,          -- 流通市值（万元）
    -- 可扩展字段（原 parquet 中可能含更多列，用 NULL 占位）
    turnover_rate   DOUBLE PRECISION,
    volume_ratio    DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic USING BRIN (trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_basic_code ON daily_basic (ts_code);

COMMENT ON TABLE daily_basic IS '日度估值/市值指标（原 alpha_daily_basic_combined.parquet），37494 行';

-- ============================================================
-- 6. 财务指标（fina_indicator）
--    原始：62,931 行 × 14 列
--    ann_date = 公告日，end_date = 报告期截止日
-- ============================================================
CREATE TABLE IF NOT EXISTS fina_indicator (
    ts_code             VARCHAR(12)   NOT NULL,
    ann_date            DATE          NOT NULL,    -- 公告日期
    end_date            DATE          NOT NULL,    -- 报告期
    roe                 DOUBLE PRECISION,
    roa                 DOUBLE PRECISION,
    grossprofit_margin  DOUBLE PRECISION,
    netprofit_margin    DOUBLE PRECISION,
    assets_turn         DOUBLE PRECISION,          -- 资产周转率
    current_ratio       DOUBLE PRECISION,
    quick_ratio         DOUBLE PRECISION,
    debt_to_assets      DOUBLE PRECISION,
    op_yoy              DOUBLE PRECISION,          -- 营业利润同比
    ebt_yoy             DOUBLE PRECISION,          -- 利润总额同比
    netprofit_yoy       DOUBLE PRECISION,          -- 净利润同比
    PRIMARY KEY (ts_code, ann_date, end_date)
);

CREATE INDEX IF NOT EXISTS idx_fina_code     ON fina_indicator (ts_code);
CREATE INDEX IF NOT EXISTS idx_fina_end_date ON fina_indicator (end_date);

COMMENT ON TABLE fina_indicator IS '季报/年报财务指标（原 fina_indicator_combined.parquet），62931 行';

-- ============================================================
-- 7. Alpha 因子截面面板（alpha_panel）
--    来源：data/features/alpha_*.parquet + csi300_full_panel.parquet
--    规模：合并后约 10,000-20,000 行 × 70+ 列（每季/每日一个截面）
--    主键：(as_of, ticker)
-- ============================================================
CREATE TABLE IF NOT EXISTS alpha_panel (
    as_of               DATE          NOT NULL,
    ticker              VARCHAR(12)   NOT NULL,
    -- 估值因子
    pe_ttm              DOUBLE PRECISION,
    pb                  DOUBLE PRECISION,
    ps_ttm              DOUBLE PRECISION,
    book_to_market      DOUBLE PRECISION,
    earnings_yield      DOUBLE PRECISION,
    dividend_yield_ttm  DOUBLE PRECISION,
    log_market_cap      DOUBLE PRECISION,
    log_circ_market_cap DOUBLE PRECISION,
    -- 成长因子
    roe_ttm             DOUBLE PRECISION,
    roa_ttm             DOUBLE PRECISION,
    revenue_yoy         DOUBLE PRECISION,
    net_profit_yoy      DOUBLE PRECISION,
    gross_margin_ttm    DOUBLE PRECISION,
    -- 质量因子
    accruals            DOUBLE PRECISION,
    ocf_to_revenue_ttm  DOUBLE PRECISION,
    debt_to_equity      DOUBLE PRECISION,
    current_ratio       DOUBLE PRECISION,
    -- 动量因子
    momentum_1m         DOUBLE PRECISION,
    momentum_3m         DOUBLE PRECISION,
    momentum_6m         DOUBLE PRECISION,
    reversal_1w         DOUBLE PRECISION,
    -- 技术因子
    rsi_14              DOUBLE PRECISION,
    bollinger_position  DOUBLE PRECISION,
    volatility_3m       DOUBLE PRECISION,
    volume_spike_5_30   DOUBLE PRECISION,
    -- 流动性因子
    amihud_illiquidity  DOUBLE PRECISION,
    turnover_3m_avg     DOUBLE PRECISION,
    turnover_rate_quantile DOUBLE PRECISION,
    -- 北向资金
    north_hold_amount   DOUBLE PRECISION,
    north_bound_30d_net_inflow DOUBLE PRECISION,
    -- 情绪/技术
    price_to_52w_high   DOUBLE PRECISION,
    price_to_52w_low    DOUBLE PRECISION,
    amplitude_quantile  DOUBLE PRECISION,
    -- 综合评分（DataService 用到的计算列）
    composite_score     DOUBLE PRECISION,
    value_score         DOUBLE PRECISION,
    momentum_score      DOUBLE PRECISION,
    quality_score       DOUBLE PRECISION,
    technical_score     DOUBLE PRECISION,
    -- 元信息
    name                VARCHAR(40),
    industry            VARCHAR(40),
    PRIMARY KEY (as_of, ticker)
);

CREATE INDEX IF NOT EXISTS idx_alpha_panel_ticker ON alpha_panel (ticker);
CREATE INDEX IF NOT EXISTS idx_alpha_panel_date   ON alpha_panel USING BRIN (as_of);

COMMENT ON TABLE alpha_panel IS '因子截面面板（历史快照 + 每日更新），(as_of, ticker) 主键';

-- ============================================================
-- 8. 宏观 Regime 特征（regime_features）
--    原始：25 行 × 8 列（季频快照）
-- ============================================================
CREATE TABLE IF NOT EXISTS regime_features (
    as_of                   DATE    PRIMARY KEY,
    csi300_63d_return       DOUBLE PRECISION,
    csi500_csi300_63d       DOUBLE PRECISION,
    chiext_csi300_63d       DOUBLE PRECISION,    -- 中概/A股溢价
    csi300_20d_vol          DOUBLE PRECISION,
    small_large_63d_spread  DOUBLE PRECISION,
    breadth_20d             DOUBLE PRECISION,
    regime_label            VARCHAR(20),         -- bull/bear/transition 等
    regime_prob             DOUBLE PRECISION     -- HMM 后验概率
);

COMMENT ON TABLE regime_features IS 'Regime 检测特征快照（25 条，季频）';

-- ============================================================
-- 9. 已实现 PnL（realized_pnl）
--    原始：80 行 × 16 列
-- ============================================================
CREATE TABLE IF NOT EXISTS realized_pnl (
    id              SERIAL        PRIMARY KEY,
    as_of_date      DATE          NOT NULL,    -- 建仓日期（推荐日期）
    ticker          VARCHAR(12)   NOT NULL,
    entry_date      DATE,
    exit_date       DATE,
    holding_days    INTEGER,
    predicted_rank  INTEGER,
    predicted_score DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    actual_return   DOUBLE PRECISION,
    alpha_vs_csi300 DOUBLE PRECISION,
    alpha_vs_csi500 DOUBLE PRECISION,
    lgbm_score      DOUBLE PRECISION,
    ensemble_score  DOUBLE PRECISION,
    model_version   VARCHAR(20),
    UNIQUE (as_of_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_pnl_date   ON realized_pnl (as_of_date);
CREATE INDEX IF NOT EXISTS idx_pnl_ticker ON realized_pnl (ticker);

COMMENT ON TABLE realized_pnl IS '已结算持仓 PnL 记录（原 realized_pnl.parquet），80 行 → 持续增长';

-- ============================================================
-- 10. NAV 曲线（nav_curve）
--     来源：data/sim30d/nav/*.json + data/iteration/*/nav/*.json
--     规模：~60 天 × 2 run = 120 行（持续增长）
-- ============================================================
CREATE TABLE IF NOT EXISTS nav_curve (
    run_id          VARCHAR(40)   NOT NULL,    -- e.g. "sim30d", "round_001_20260601_1329"
    trade_date      DATE          NOT NULL,
    nav             DOUBLE PRECISION,          -- 净值（初始 1.0）
    daily_return    DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    benchmark_return  DOUBLE PRECISION,        -- CSI300 同期
    alpha           DOUBLE PRECISION,
    n_positions     INTEGER,
    PRIMARY KEY (run_id, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_curve USING BRIN (trade_date);

COMMENT ON TABLE nav_curve IS 'NAV 曲线（sim30d + iteration runs），按 run_id 区分回测轮次';

-- ============================================================
-- 11. 快照基本面（snapshot_daily_basic）
--     来源：data/snapshots/{date}/daily_basic.parquet
--     规模：~35 日期 × ~1374 行 = ~48K 行
--     说明：与 daily_basic 部分重叠，但 snapshot 包含更多字段
-- ============================================================
CREATE TABLE IF NOT EXISTS snapshot_daily_basic (
    as_of           DATE          NOT NULL,    -- snapshot 日期（季末/月末）
    ts_code         VARCHAR(12)   NOT NULL,
    close           DOUBLE PRECISION,
    pe_ttm          DOUBLE PRECISION,
    pb              DOUBLE PRECISION,
    ps_ttm          DOUBLE PRECISION,
    total_mv        DOUBLE PRECISION,
    circ_mv         DOUBLE PRECISION,
    turnover_rate   DOUBLE PRECISION,
    volume_ratio    DOUBLE PRECISION,
    PRIMARY KEY (as_of, ts_code)
);

CREATE INDEX IF NOT EXISTS idx_snap_basic_code ON snapshot_daily_basic (ts_code);

COMMENT ON TABLE snapshot_daily_basic IS '季度/月度快照中的 daily_basic，~35 期 × 1374 股';

-- ============================================================
-- 12. 快照财务报表（snapshot_financials）
--     来源：snapshots/{date}/financials_income.parquet 等
-- ============================================================
CREATE TABLE IF NOT EXISTS snapshot_financials (
    as_of           DATE          NOT NULL,
    ts_code         VARCHAR(12)   NOT NULL,
    report_type     VARCHAR(20)   NOT NULL,    -- 'income' | 'balance_sheet' | 'cashflow'
    end_date        DATE,
    revenue         DOUBLE PRECISION,
    net_profit      DOUBLE PRECISION,
    total_assets    DOUBLE PRECISION,
    total_liab      DOUBLE PRECISION,
    equity          DOUBLE PRECISION,
    operate_cash    DOUBLE PRECISION,
    PRIMARY KEY (as_of, ts_code, report_type)
);

COMMENT ON TABLE snapshot_financials IS '快照财务三表（income/balance_sheet/cashflow 合并）';

-- ============================================================
-- 索引汇总说明
-- ============================================================
-- 时序大表索引策略：
--   BRIN（Block Range INdex）适合顺序写入的时序数据
--   - 比 B-Tree 小 100-1000x
--   - 范围扫描（WHERE trade_date BETWEEN ...）效率与 TimescaleDB hypertable 接近
--   - 适合 append-only 的 parquet 替代场景
--
-- 关键查询模式：
--   (1) 按日期范围拉价格：WHERE ts_code='000001.SZ' AND trade_date BETWEEN ...
--       → 使用 (ts_code, trade_date) 复合主键（B-Tree，精确）
--   (2) 截面查询某天所有股票：WHERE trade_date='2026-06-01'
--       → 使用 BRIN on trade_date（低代价范围过滤）
--   (3) 因子截面查询：WHERE as_of='2026-06-01' ORDER BY composite_score DESC
--       → alpha_panel 的 (as_of, ticker) 主键
