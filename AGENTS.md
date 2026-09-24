# AGENTS.md — QuantMind 协作规则（Claude Code / Cursor / Codex 共同遵守）

> 这是三个工具共享的**唯一**规则文件。Codex 与 Cursor 原生读取本文件；
> Claude Code 通过 `.claude/CLAUDE.md` 顶部的 `@AGENTS.md` 引用。**不要在别处复制规则。**

## 1. 角色（2026-09-24 按 Claude Pro 额度修订）

Claude Pro 是几个订阅里额度最紧的，所以**高频的总控交给 Codex，Claude 只用在最需要严谨的验收上**。

| 代号 | 工具 · 模型 | 职责 | 禁止 |
|---|---|---|---|
| C1 | **Codex CLI · GPT-6 Sol**（WSL，仓库根目录） | 总控：写任务卡、派发、维护账本、阶段汇总 | 写生产代码 |
| C2 | **Claude Code CLI**（WSL）· Sonnet 5；量化结论与阶段闸门用 Opus 5.5 · 每任务新会话 | 验收：跑验收命令、从数据重算、给 ACCEPT/REJECT | 改生产代码 |
| C3 | Claude Code 桌面版（Windows 侧）· Sonnet 5 | 实验室：独占 GPU（llama-server / Ollama 在 Windows 侧） | 架构决策 |
| K1 | Cursor · Grok 4.7 | 主力实现 | 越出任务卡 allowed_paths |
| K2 | Cursor · Composer 2.5 | 批量、测试、样板、小修 | 跨模块改动 |
| K3 | Cursor · Fable 5.1（凭票，$20/月） | 高难实现 | 探索代码库、写样板 |
| G1 | ChatGPT · GPT-6 Astra Pro（无仓库访问） | 统计判断的第二意见；Claude 验收结论的复核 | 下验收结论 |
| G2 | Codex CLI · GPT-6 Sol / Luna（**与 C1 不同的会话**，独立 worktree） | 并行实现 | 与 K1 同时改同一文件 |

### 1.1 跨家族规则：谁写的，就由另一家的模型来查

| 产物 | 作者家族 | 验收 / 评审 |
|---|---|---|
| K1 / K2 写的代码 | xAI / Cursor | C2（Claude） |
| G2 写的代码 | OpenAI | C2（Claude） |
| **K3（Fable）写的代码** | Anthropic | **Codex 验收**（不是 Claude）+ 客观测试 |
| C1 写的设计与任务卡 | OpenAI | **C2 · Opus 5.5 评审** |
| C2 对量化结论的验收 | Anthropic | 阶段闸门时 G1 复核 |

### 1.2 Claude 额度守则
- 默认用 Sonnet 5；**只有**标签、IC、净超额、消融实验这类"数字错了会导致错误决策"的验收，以及阶段闸门，才用 Opus 5.5
- 每个 C2 会话只验一张卡，验完即关
- Claude 额度用完时，C2 由 Codex 顶替（K3 的卡除外，那些本来就归 Codex 验）

## 2. 证据分级（所有文档、提交说明、验收报告都必须遵守）

- `[代码]` 可指到文件与行号 · `[实测]` 本机实际跑出的数字（附结果文件路径）
- `[文档]` 项目文档或注释里的主张（未必等于代码事实） · `[推断]` 推理 · `[未知]` 回答不了的
- **任何数字必须指到产出它的文件。** 找不到出处的写"来源不明"。
- **不得把 `[推断]` 写成 `[代码]` 或 `[实测]`。**

## 3. 任务纪律

1. 只处理 `docs/work/tasks/T-xxx.md` 里状态为 `assigned` 且分配给你的卡。
2. 一张卡一个分支 `t/xxx`；提交信息以 `T-xxx:` 开头。
3. **只能修改卡片 `allowed_paths` 列出的文件。** 需要改别的文件时停下来，在卡片"实现说明"里写明原因，交回 C1。
4. 实现者只能把状态改为 `implemented`。**只有 C2 能写 `accepted` / `rejected`。**
5. 提交前必须跑：`pytest tests/ -q -p no:cacheprovider`（**不要加 `-m`**，使用 pyproject 的默认 marker 过滤）。
6. 不得覆盖 `data/` 下已有的产物文件；新产物用新文件名（如 `alpha_panel_v4_1.parquet`）。

## 4. PIT（时点正确性）不变量——违反即驳回

1. 任何带 `as_of` 语义的计算，只能使用 `≤ as_of` 可知的数据；财务数据按 `f_ann_date` 过滤。
2. **禁止**在任何带 `as_of` 语义的分析中调用实时数据接口、联网搜索或未经 PIT 包装的 MCP 工具。
3. 标签（forward return）必须用**单一一致的价格基准**计算（见 T-003）。
4. 喂给 LLM 的因子值必须标明量纲：原始值，或明确标注"截面标准分"；**z-score 不得加 `%`**。
5. 任何涉及股票代码的 LLM 调用，其回测结论必须注明模型知识截止日（Bonsai 估计 2025-01～03）。

## 5. 本地 GPU（RTX 5060 Laptop，8151 MiB）

1. **只有 C3 能启动本地模型服务。** 其他角色需要本地推理时，在任务卡里申请。
2. 按模型批处理，不要来回切换（换一次模型 8–16 秒）。
3. llama-server 固定参数：`-ngl 99 -c 4096 -np 1 -ctk q4_0 -ctv q4_0 -fa on --jinja`。
4. 结构化输出必须带 `response_format`（JSON schema）并设 `chat_template_kwargs: {"enable_thinking": false}`。
5. 不开任何 `--spec-type`（实测无效或有害）。
6. 可同时驻留：Bonsai 27B + Laya；不可同时驻留：任意两个大模型。

## 6. 秘密信息

`.env`、`api_key.txt` 永不提交、永不打印到日志或报告。发现疑似凭证进入 diff 时立即停止并报告 C1。

## 7. 常用命令（在 WSL 的 ~/projects/quantmind 下）

```bash
pytest tests/ -q -p no:cacheprovider                          # 默认测试套件
python tools/label_audit.py --panel data/panel/<file>.parquet # 标签对照真值审计
git worktree add ../qm-codex-t0xx t/0xx                       # Codex 独立工作区
```
