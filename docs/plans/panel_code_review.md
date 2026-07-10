# 周频面板 + 数据湖 — 独立 Code Review（合并闸门）

> 评审人：本 session（**未参与编写**这些代码）。视角：怀疑者。
> 原则：**只审不改**，不触碰任何业务代码/数据/模型。
> 评审对象：`quantmind/data/lake.py`、`quantmind/features/weekly_panel.py`、
> `scripts/verify_weekly_panel.py`、`scripts/build_data_lake.py`、`scripts/backfill_tushare.py`。
> 自验证现状：`verify_weekly_panel.py` 自报 31/31 PASS + `tests/test_lake.py`/`tests/test_weekly_panel.py`。
> **本评审的核心任务 = 查"自验证本身是否可靠"+"自测没覆盖的缺口"。**

## 方法论说明

- 项目 `CLAUDE.md` 声明配置了 CodeGraph MCP（`codegraph_*`），但**本 session 这些 MCP 工具未加载**
  （`ToolSearch select:codegraph_callers,codegraph_impact` 返回 "No matching deferred tools"）。
  因此调用面用 `Grep` 枚举核查——调用面很小，已完整列出（见下）。
- 调用面（grep 全仓 `*.py`）：
  - `read_lake_window`：`weekly_panel.assemble_snapshot`（index_daily/north_bound/margin 三处）+
    `verify_weekly_panel`（E/D/F 段）+ `tests/test_lake.py`。**无生产代码在 `weekly_panel` 之外调用它读外部序列。**
  - `write_lake`：`scripts/build_data_lake.build_series` + `scripts/backfill_tushare.flush` + `tests/test_lake.py`。
  - `compute_forward_returns`：`panel.build_panel`、`scripts/build_full_panel`、`weekly_panel.attach_labels`（三处口径需各自校验，本评审只覆盖 weekly_panel 这条）。

---

## 🔴 高风险（放行前应处理 / 至少明确接受）

### H1. PIT 自验证存在**结构性盲区**：篡改测试只覆盖行情，外部三序列（index/north/margin）的 PIT 用"同一函数回算"自证 → 循环验证，无法发现泄漏

- **判定：缺陷（验证不可靠）**
- **位置：**
  - 篡改测试 `scripts/verify_weekly_panel.py:226-242`（E2）
  - 边界探针 `scripts/verify_weekly_panel.py:217-224`（E1）
  - 数值回算 `scripts/verify_weekly_panel.py:180-206`（D2 beta / D3 north / D4 margin）
- **风险等级：高**
- **事实：**
  1. **E2 篡改只动 `prices`**（`tampered.loc[fut_mask, c] *= 999`，列仅 close/high/low/open/pre_close/volume），
     再 `compute_factors_for_asof` 比对因子不变。但 `compute_factors_for_asof` 里 index_daily/north_bound/margin
     是经 `read_lake_window` 从**磁盘湖表**读的（`weekly_panel.py:243-245`），**篡改根本没碰这三个文件**。
     因此 E2 只证明了"行情窗口 PIT 正确"，对 index/north/margin 的 PIT **一字未测**。
  2. **E1 仅断言 `max(trade_date) <= as_of`**，且只在**单个** `audit_asof`（=最后一个 as_of）上跑。
     由于湖表数据天然止于近期、且 `as_of` 取末点，这个断言近乎恒真——它**不注入未来行**，
     无法证明"如果湖里有 as_of 之后的行，会被过滤掉"。
  3. **D2/D3/D4 是循环自证**：它们用 `read_lake_window(...)` 取回数据再"手算"，
     而面板里的值**也是**用同一个 `read_lake_window` 算的。若 `read_lake_window` 对 index/north/margin
     存在 off-by-one（例如把 `as_of+1` 漏进来），D2/D3/D4 的"手算"会同样吸入那一行 → 仍然相等 → **假 PASS**。
     只有 D2 的**行情侧**用 `close_piv.loc[:audit_asof]` 独立 PIT 了（这条可信），index 侧没有独立基准。
- **代码层面缓解（降低实际风险，但不改变"验证不可靠"的结论）：**
  `read_lake_window` 的过滤是**单一掩码** `mask = (trade_date >= start_ts) & (trade_date <= as_of_ts)`
  （`lake.py:150`），无任何分支/默认参数能把 `> as_of` 放进来；上界用 `as_of` 本身而非"日历内最近交易日"，
  故代码本身**可信**。所以 H1 的风险主要是"**护栏缺失**"——一旦未来有人改 `read_lake_window`
  （加缓存、改边界、加 `>=` 容差），现有自验证抓不到回归。
- **建议（不改业务代码，仅补测试，放行后即可做）：**
  1. 把 E2 升级为"**篡改湖表**"测试：构造一份在 `as_of` 之后追加伪造行（×999）的临时湖表
     （monkeypatch `DATA_LAKE_DIR` 或传临时 parquet），断言 index/north/margin 三个因子（beta_252d /
     north_bound_30d_net_inflow / margin_balance）值不变。这才是真正覆盖"全部外部序列"的篡改测试。
  2. D2/D3/D4 的"手算基准"改为**不经 `read_lake_window`** 的独立路径（直接 `pd.read_parquet(lake_path(...))`
     后自己按 `trade_date <= as_of` 切），打破循环。
  3. E1 改为跨**多个** as_of（至少首/中/末）断言边界。

---

## 🟠 中风险

### M1. `write_lake` 整表覆写非原子 → 单次 flush 崩溃会损毁**整张**湖表（与"最多丢在途一月"承诺矛盾）

- **判定：缺陷**
- **位置：** `quantmind/data/lake.py:185-199`（`merged.to_parquet(p, index=False)`）；
  按月 flush 调用方 `scripts/backfill_tushare.py:147-159`；注释承诺见 `backfill_tushare.py:13`（"崩溃最多丢在途一月"）。
- **风险等级：中**
- **事实：** `write_lake` 每次都 `concat(old, new)` 后把**完整合并表**写回**同一路径** `p`。
  `backfill` 每完成一个自然月 flush 一次。若进程在 `to_parquet`（`lake.py:198`）**写到一半**崩溃/断电，
  目标 parquet 被部分写入 → **整文件损坏**，不仅丢在途月，**之前所有已落盘月一并丢失**。
  注释/文档的"崩溃最多丢在途一月"对 checkpoint 成立，但对**数据文件本身不成立**。
- **并发：** 同样是读-改-写、无锁、无原子 rename。当前**无并发写者**（backfill 单进程顺序；
  build_data_lake 单进程；weekly_panel 的 `ProcessPoolExecutor` 只读不写），故现状安全。
  但 docstring 暗示"幂等/安全"易被误读为可并行——**不可并行写同一 series**。
- **建议：** `to_parquet` 写临时文件 + `os.replace()` 原子换名（POSIX/NTFS 上 rename 原子）；
  docstring 明确"单写者、非并发安全"。

### M2. 长窗 NaN 掩码是"**全市场逐 as_of**"粒度，非"**逐 ticker**"——晚期截面里短历史新股的 `iloc[-63:].std()/.mean()` 类因子产出"**薄估计**"而非约定的 NaN（违反面板"数据不足→NaN"契约）

- **判定：缺陷（已核实因子实现，确认可达）**
- **位置：** 掩码 `quantmind/features/weekly_panel.py:272-277` + `_MIN_TRADING_DAYS` `:74-83`；
  因子实现 `quantmind/features/technical.py:113-153`（`downside_volatility_3m`/`max_drawdown_3m`/`amihud_illiquidity`）、`:104-110`（`volatility_1y`）。
- **风险等级：中**
- **已核实的事实：**
  - `available_td = snap["prices"]["trade_date"].dt.normalize().nunique()`（`weekly_panel.py:273`）是
    **整个窗口内所有 ticker 汇总**的去重交易日数（≈280），**不是单只股票的可用历史长度**。
    `if available_td < req: out[col] = np.nan`（`:274-277`）只在**数据集很早期**（全市场窗口本身 < req）整列置 NaN——
    对早期 as_of 正确（验证项①仅测"最早 as_of beta_252d 全 NaN"，`verify:62-71`）。
  - 这些因子**不是 `rolling(min_periods=...)`**，而是 `_daily_returns(close).iloc[-63:].std()` /
    `.iloc[-63:].mean()` / `cummax` 这种"**取末 63 行后做 NaN-skipping 聚合**"（`technical.py:118-120 / 128-131 / 150-153`）。
    `pivot_prices` 是 date×ticker 宽表，短历史股的列在上市前为 NaN；`.std()/.mean()` **默认跳过 NaN**。
  - **后果（可达）：** 一只在某**中/晚期** `as_of` 前仅 ~5–20 个交易日上市的新股，被 `pit_universe`
    （`list_date <= as_of`）正常纳入；`available_td`（全市场）=280 ≥ req → **不掩码**；
    因子对它的列 `.iloc[-63:]` 里只有 ~5–20 个有效值，`.std()/.mean()` 直接给出**基于极少样本的"薄估计"**，
    **既不是 NaN，也不会被全市场掩码拦下**。例如 `downside_volatility_3m` 用 5 天负收益算年化下行波动 → 统计上不可靠。
  - **这违反面板自身契约**："数据不足的早期 as_of 由因子层自然产出 NaN（非垃圾值）"（`weekly_panel.py` 模块 docstring 硬条件①）。
    契约只在"全市场早期"成立；对"晚期 as_of × 单票短历史"**不成立**——它给的是薄估计而非 NaN。
  - 注：`momentum_*`（`close[t]/close[t-21]` 类）短序列天然产 NaN（`close[t-21]` 缺失），不受影响；
    受影响的是 `iloc[-N:]+std/mean/cummax` 这一族（downside_vol / max_dd_3m / amihud / volatility_1y 等）。
- **verify 缺口：** C/D 段抽样都取 `dropna().index[0]`（数据充足的票），从不构造"晚期 as_of + 短历史新股"，
  这条**永远测不到**。
- **建议：**
  1. 把掩码改成**逐 (as_of, ticker)**：按每只票在窗口内的**非 NaN bar 数**判定，对 `< req` 的票该列置 NaN，
     而非用全市场 `available_td`。
  2. 补一条 verify：晚期 `as_of` 下取一只 `list_age_years < 0.3` 的票，断言其 `_MIN_TRADING_DAYS`
     中 window > 其个股历史的因子全为 NaN。
  3. 量化影响：统计每个 as_of 截面里 `list_age` 短于 63 交易日的票数占比，评估薄估计对 alpha 判定的实际污染面。

### M3. 特征侧纳入 `trade_date == as_of` 的盘后发布序列（north/margin）——同日可得性是"约定"，非"事实"

- **判定：存疑（按约定可接受，但应在 meta 写明）**
- **位置：** `lake.py:150`（上界 `<= as_of_ts` 含等号）；标签侧 `compute_forward_returns` 以 `as_of` 当日 adj_close 为基准向后看（`panel.py:183-195`）。
- **风险等级：中偏低**
- **事实：** `margin_detail`、北向 `north_money` 均为**当日盘后**发布。特征把 `trade_date == as_of` 的行计入
  （`tail(30).sum()` 含 as_of 当日，`sentiment.py:35`），等价于假设"as_of 收盘决策时已拿到 as_of 当日盘后数据"。
  标签基准也用 as_of 当日收盘 → 特征与标签**在 as_of 当日这一点重叠是设计内的**（标签向后第 h 日，
  `future_idx = index > base_idx`，`panel.py:191`，严格不含 as_of 当日之前的未来）——**标签/特征时间边界本身不泄漏**（可信）。
  唯一可议的是"同日盘后数据用于同日收盘决策"这个 T 日可得性假设。
- **建议：** 在 `weekly_panel` meta 增一行说明"外部日频含 as_of 当日盘后值，采用 T 收盘决策约定"；
  若要严格，可对 north/margin 改用 `< as_of`（lag 1）。当前不构成 look-ahead 泄漏，仅是可得性口径。

---

## 🟡 低风险 / Nit

### L1. `merge_increment` 防护依赖 `assert`（`python -O` 下失效）+ 增量 index 含重复键会先笛卡尔展开
- **判定：存疑** — `weekly_panel.py:494-503`。列冲突用 `raise ValueError`（可信，G2 已测）；
  但"行数不变"用 `assert len(out) == len(base)`（`:502`），`-O` 下不执行。
  且若 `increment` 的 MultiIndex 含**重复键**，`base.join` 会先笛卡尔放大再被 assert 抓（抛 AssertionError 而非友好报错）。
  另：未校验两边 index `names`/层序一致，名字错配可能静默错对齐。
  **建议：** 把 assert 改成显式 `raise`；join 前校验 `increment.index.is_unique` 与 names。**风险：低。**

### L2. `coverage()` 形参 `max_gap_calendar_days` 从未使用
- **判定：缺陷（死参数/误导）** — `lake.py:207`。函数签名带 `max_gap_calendar_days=11` 但函数体内无引用。
  调用方易误以为能控阈值。**建议：** 删除该参数或在体内使用。**风险：低。**

### L3. `data/lake/_checkpoints/*.json` 未被 `.gitignore` 覆盖
- **判定：存疑** — `.gitignore` 只忽略 `*.parquet`（`:71`）与 `data/panel/*` 等，`data/lake/` 目录未整体忽略，
  且 `git status` 显示 `?? data/lake/` 为未跟踪。checkpoint JSON 存 `failures`（异常 `repr(e)[:160]`，`backfill_tushare.py:181`）。
  正常 Tushare 异常不含 token，但 JSON 一旦被 `git add` 会进版本库。
  **建议：** `.gitignore` 增 `data/lake/`（或至少 `data/lake/_checkpoints/`）。**风险：低。**

### L4. `backfill._load_token_into_env` 细节
- **判定：可信但可加固** — `backfill_tushare.py:66-81`。`open(path)` 未用 `with`（句柄靠 GC 关，低）；
  仅按**全角** `：` 切分（`:69-72`），若 `api_key.txt` 用 ASCII `:` 会"找不到 token" SystemExit（健壮性，非安全）。
  token 读后置 `None` 丢引用、不回显、不落盘——**安全部分可信**。**风险：低。**

---

## ✅ 经核查可信（重点项逐条结论）

| # | 核查点 | 结论 | 依据 |
|---|--------|------|------|
| 1 | `read_lake_window` 所有路径严格 `trade_date <= as_of`，无分支/默认参/边界放未来进 | **可信（代码层）** | `lake.py:150` 单一掩码；上界用 `as_of` 本身（`:149` 注释"杜绝未来泄漏"）；空日历早退 `:144-145`。**但自验证对此不可靠 → 见 H1** |
| 1b | 标签（as_of 后第 N 日 adj_close）vs 特征（as_of 及之前）时间边界不重叠 | **可信** | 标签 `future_idx = index > base_idx`（`panel.py:191`）严格向后；base=as_of 当日（`:183-185`）；特征窗口 `<= as_of`（`weekly_panel.py:231-233`）。仅 as_of 当日这一基准点为设计内共用，非泄漏 |
| 2 | universe 按 `list_date <= as_of` PIT 入池，无新股前视；边界 `==` 正确 | **可信** | `pit_universe` `listed = list_date.notna() & (list_date <= as_of)`（`weekly_panel.py:201`）；`list_date==as_of` 当日入池（IPO 首日有行情，合理）；verify I1（`:293-302`）独立用 `sb_all` 校验未上市票不在早期截面（非循环，**这条自验证可靠**） |
| 3 | north_bound 的 daily_basic shim 只提供结构（ticker 列表），不漏伪造数值 | **可信** | `north_bound_30d_net_inflow` 仅用 `db.set_index("ticker").index` 取 ticker 索引（`sentiment.py:45-46`），数值 `val` 全部来自 `snapshot["north_bound"]["north_money"]`（`:30-36`）；shim 只有 `ticker` 一列（`weekly_panel.py:242`），**无数值列可漏** |
| 5 | 因子 raw close / 标签 adj_close 口径无混用 | **可信** | 因子子表 `_FACTOR_PRICE_COLS` 含 `"close"`=raw（`weekly_panel.py:118-121`）；标签透视 `build_adj_close_pivot` 用 `adj_close`（`:138-145`）；`attach_labels` 把 **adj_pivot** 传给 `compute_forward_returns`（`:293`）。pivot 传参正确，未见复用时传错 |
| 7 | `write_lake` 幂等：`(ts_code,trade_date)` 去重 `keep="last"` | **可信（重跑）** | `concat([old,new])` 后 `drop_duplicates(subset=key, keep="last")`（`lake.py:189-197`），new 在后→新值覆盖；`tests/test_lake.py:75-119` 覆盖幂等/keep-last/north_bound 单键去重/缺键报错。**并发/原子性见 M1** |
| 8 | 脚本无 token 明文 | **可信** | 全仓 grep 32+位 hex / `token=` 字面量：仅 `step2_download_prices.py:16` 的 `export TUSHARE_TOKEN="xxxx"` 占位符（评审范围外，且为占位）；`api_key.txt` 已 `.gitignore`（`:7`） |
| 8b | backfill `logger.remove()` 落位是否正确（防 `token[:8]` 入 loguru 文件 sink） | **可信（但脆弱）** | token 由 loguru `log.info(... token[:8] ...)` 打印（`tushare_provider.py:83/85`），且 `_get_tushare_pro` **懒加载**（`:57`，首个 `_call` 才触发）。`backfill` 在 import top 与 **import tushare_provider 之后、任何 `_call` 之前** 各 `logger.remove()` 一次（`backfill_tushare.py:39-45` / `225-229`），顺序正确 → 运行期 token 不入屏/不入 `logs/*.log`。**脆弱点：** 依赖"懒加载 + 手动二次 remove"；若日后 provider 改为 import 时即初始化、或新增 sink，token 会重新泄漏到文件 sink。建议改用集中式 `quantmind/utils/silence_provider_logging.py` 包装（仓内已存在该工具） |

### 自验证脚本本身（`verify_weekly_panel.py`）可靠性评估

- **可靠的检查**（独立基准，非循环）：B1/B2/B3 网格相位与标签边界（用 calendar 位置独立推导，`:116-140`）；
  C 标签手算（用 `adj` 透视独立 `searchsorted`，`:143-162`，与 `compute_forward_returns` 算法一致且不经它）；
  D1 momentum（`close_piv` 独立 PIT，`:169-178`）；I1 universe PIT（独立 `sb_all`，`:293-302`）；
  A 结构 / G 增量接口。
- **不可靠/循环的检查**（见 H1）：E1（边界近恒真、单 as_of、不注入未来）、E2（只篡改行情）、
  D2/D3/D4（用同一 `read_lake_window` 回算 index/north/margin → 假 PASS 风险）。
- **完全未覆盖的缺口：** ① 外部三序列的真·篡改 PIT 测试；② 晚期 as_of × 短历史新股的 rolling 因子垃圾值（M2）；
  ③ `write_lake` 崩溃中断的数据完整性（M1）；④ 多 as_of 的 PIT 边界扫描。
- **结论：31/31 PASS 不足以放行"无 PIT 泄漏"的强结论**——通过项里最关键的"外部序列 PIT"是循环自证。
  建议放行前**至少补 H1-建议(1)（篡改湖表测试）**，其余（M1/M2）可作为放行后紧后续项并在 meta 标注。

---

## 放行建议（汇总）

| 优先级 | 项 | 动作 |
|--------|-----|------|
| 阻断（建议放行前做） | H1 | 补"篡改湖表"PIT 测试（index/north/margin），打破 D2/D3/D4 循环自证 |
| 紧后续 | M2 | 核 `technical.py` rolling 因子 `min_periods`；若 <window，掩码改逐 (as_of,ticker) |
| 紧后续 | M1 | `write_lake` 改原子写（temp+rename）；docstring 标"单写者" |
| 可放行后 | M3/L1-L4 | meta 补同日数据口径；assert→raise；删死参数；gitignore `data/lake/` |

> 本评审未修改任何业务代码/数据/模型，仅产出本文件。
