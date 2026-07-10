# QuantMind — Claude Code 辅助工具链配置（TOOLING_SETUP.md）

> 本文档记录三个 Claude Code 辅助工具的安装、配置与使用约定。
> **范围限定**：纯工具链配置，未修改任何 QuantMind 业务代码 / 数据 / 模型 / parquet / 测试。
> 安装日期：2026-06-04 · 验证环境：Windows 11 上的 Claude Code（`C:\Users\lenovo\.local\bin\claude.exe`），操作 WSL 内项目 `/home/lenovo/projects/quantmind`。

---

## 0. 环境拓扑（重要）

- **Claude Code 本体运行在 Windows 侧**，配置根目录 `C:\Users\lenovo\.claude\`（skills / plugins / settings.json 均在此）。
- 项目源码在 **WSL**（`/home/lenovo/projects/quantmind`），Claude Code 通过 UNC 路径 `\\wsl.localhost\ubuntu\...` 访问。
- 因此三个工具均安装在 **Windows 侧**；WSL home 下并无 `~/.claude/`。
- Windows node v24.15.0 / npx 11.12.1（claude-mem 使用）；WSL node v20.20.2（codegraph 使用）。

---

## 1. 三个工具一览

| 工具 | 版本 | 类型 | 安装方式 | 来源 |
|------|------|------|----------|------|
| **claude-mem** | 13.4.0 | 插件（hooks + worker，全局） | `npx claude-mem install` | [github.com/thedotmack/claude-mem](https://github.com/thedotmack/claude-mem) · [docs](https://docs.claude-mem.ai/installation) |
| **planning-with-files** | 2.43.0 | 插件 / skill（user 作用域） | `claude plugin marketplace add OthmanAdi/planning-with-files` + `claude plugin install planning-with-files@planning-with-files` | [github.com/OthmanAdi/planning-with-files](https://github.com/OthmanAdi/planning-with-files) |
| **code-review** | (Anthropic 官方) | 插件 / 命令（user 作用域） | `claude plugin install code-review@claude-plugins-official` | [anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official) · [docs](https://code.claude.com/docs/en/code-review) |

### 验证结果（status / health 命令）

```text
# claude-mem doctor  → All required checks passed.
  ✓ Bun runtime        v1.3.14
  ✓ uv (vector search) uv 0.11.19
  ✓ Plugin installed   C:\Users\lenovo\.claude\plugins\marketplaces\thedotmack
  ✓ Marketplace deps   node_modules present
  ✓ Worker daemon      healthy at http://127.0.0.1:37777

# claude plugin list  → 2 个插件 enabled
  > code-review@claude-plugins-official        Status: ✓ enabled
  > planning-with-files@planning-with-files     v2.43.0   Status: ✓ enabled
```

> 安装期一次性依赖修复（均为良性）：
> - claude-mem 的 marketplace 依赖出现 tree-sitter peer-dep ERESOLVE，已用 `--legacy-peer-deps` 回退解决。
> - Bun 运行时在非 TTY 安装时被跳过，已手动 `npm install -g bun`（v1.3.14）补齐，doctor 复测全绿。

---

## 2. claude-mem 配置

### 2.1 本地存储（SQLite，无云依赖）

- **数据根目录**：`C:\Users\lenovo\.claude-mem\`（即文档中的 `~/.claude-mem`）
- **数据库**：`C:\Users\lenovo\.claude-mem\claude-mem.db`（SQLite）
- **设置文件**：`C:\Users\lenovo\.claude-mem\settings.json`
- 其他：`backups/`、`corpora/`（向量检索语料）、`logs/`、`worker.pid`、`supervisor.json`
- Web 查看器：`http://localhost:37777`（worker 运行时可用）

所有数据仅留在本机，不出网。

### 2.2 敏感信息排除 —— `<private>` 标签机制

claude-mem 内置 `<private>` 标签过滤（hook 层，落库前剥离，无需额外配置）：
- `UserPromptSubmit` hook 从用户 prompt 中剥离 `<private>…</private>`，再写入 `user_prompts` 表；
- `PostToolUse` hook 在生成 observation 前，从序列化的 `tool_input` / `tool_response` JSON 中剥离 `<private>` 内容；
- 会话进行中 Claude 仍能看到完整内容，**仅在持久化到数据库/检索索引时被剔除**。

**本项目须用 `<private>` 包裹的内容**（凡是粘进 prompt 的真实密钥）：
- Tushare token
- PostgreSQL `quantmind` 用户密码 / 连接串（如 `postgresql://quantmind:<password>@host/db`）
- GitHub SSH key / token
- 未来的客户数据、JWT secret、各类 API key

> 注意：claude-mem **不会主动扫描磁盘文件内容**，从磁盘加载的环境变量本身不入库；风险来自**直接粘贴进 prompt** 的明文 —— 这类内容务必用 `<private>` 包裹。

### 2.3 工具输出排除列表 —— `CLAUDE_MEM_SKIP_TOOLS`

claude-mem 用 `CLAUDE_MEM_SKIP_TOOLS`（逗号分隔）控制哪些工具的输出**不生成 observation**。

- **配置项名**：`CLAUDE_MEM_SKIP_TOOLS`
- **文件**：`C:\Users\lenovo\.claude-mem\settings.json`
- **默认值**：`ListMcpResourcesTool,SlashCommand,Skill,TodoWrite,AskUserQuestion`
- **本项目当前值**：
  ```json
  "CLAUDE_MEM_SKIP_TOOLS": "ListMcpResourcesTool,SlashCommand,Skill,TodoWrite,AskUserQuestion,Read,Bash,PowerShell"
  ```
- **追加项理由**：
  | 追加工具 | 风险 |
  |----------|------|
  | `Read` | 可能读取 parquet 大面板（大数据量）与 `.env`/配置中的密钥 |
  | `Bash` | 输出常含 `psql` 连接串、Tushare token 回显、`git remote -v` 中的 token |
  | `PowerShell` | 同上（Windows 侧 shell 同样会回显凭据 / dump 大量数据） |
- **取舍说明**：排除 `Read/Bash/PowerShell` 会让记忆少捕获"读了什么文件、跑了什么命令"的细节，换取更强的凭据/大数据防泄漏。若日后希望更丰富的过程记忆，可按需从该列表移除某项；但移除后须更严格地依赖 `<private>` 标签。

`settings.json` 完整当前内容：
```json
{
  "CLAUDE_MEM_RUNTIME": "worker",
  "CLAUDE_MEM_SKIP_TOOLS": "ListMcpResourcesTool,SlashCommand,Skill,TodoWrite,AskUserQuestion,Read,Bash,PowerShell"
}
```

### 2.4 作用域偏差说明

任务期望"安装到本项目作用域"，但 **claude-mem 官方仅支持全局安装**（global-only），不提供 project-scope 选项。因此它是全局插件，对所有项目生效。本项目的安全收敛通过 `<private>` 标签 + `CLAUDE_MEM_SKIP_TOOLS` 两道机制实现，而非作用域隔离。

### 2.5 运维命令

```powershell
npx claude-mem status     # 查看 worker 状态（PID / 端口 / 启动时间）
npx claude-mem doctor     # 健康自检（Bun / uv / 插件 / 依赖 / worker）
npx claude-mem start      # 启动 worker（本机当前 PID 11956，端口 37777）
npx claude-mem restart    # 重启 worker（改 settings.json 后需重启生效）
npx claude-mem bug-report # 生成诊断报告
```

> 备注：当前 Claude Code 环境设了 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`，禁用的是 Claude Code **内置**的 auto-memory（即 `MEMORY.md` 体系），与 claude-mem 互不冲突、可共存。

---

## 3. planning-with-files 计划目录约定

### 3.1 工具的原生行为（无法配置输出目录）

planning-with-files 通过 hooks 自动检测计划文件，**检测位置固定**为：
- 仓库根目录的 `task_plan.md` / `progress.md` / `findings.md`（root 作用域），或
- `.planning/<plan_id>/` 下的同名文件（scoped 作用域）。

官方**未提供**把输出目录改成 `docs/plans/` 的配置项。其 hooks 只扫描上述两处。

### 3.2 本项目约定（写死）

1. **所有人工撰写 / 需要长期留存、可分享的计划文档** → 统一放在 **`docs/plans/`**（本批工具的安装计划 `docs/plans/tooling_setup_plan.md` 即遵循此约定）。
2. **planning-with-files 的临时工作文件**（`task_plan.md` / `progress.md` / `findings.md`）→ 使用其 **`.planning/<plan_id>/` scoped 模式**，以保持仓库根目录干净；视为草稿/工作内存。
3. 当某个 `.planning/<id>/task_plan.md` 计划定稿、需要进入交接/评审时，**将定稿副本整理进 `docs/plans/`**。
4. 建议把 `.planning/` 视为本地工作目录（如需可在 `.gitignore` 中忽略；本次未改动 `.gitignore`）。

> 一句话：`docs/plans/` = 正式计划的归档地；`.planning/` = 工具的临时工作区。

---

## 4. code-review 使用约定（写死）

- **必须在独立的新 session 中运行**：用一个**全新的 Claude Code session** 去审查**另一个 session** 写的代码，**绝不在写代码的同一 session 里自审**（避免"自己批改自己作业"的确认偏差）。
- **用途定位**：作为**回测 / 模型改动合并前的独立验收闸门**（merge-gate）。E2/E3 NAV 回测、因子新增、仓位优化等改动，合并前先过一遍 `/code-review`。
- **配合 codegraph**：审查时用 `codegraph_impact` 核对改动的影响面，用 `codegraph_callers/callees` 确认调用关系，避免遗漏受影响模块。
- **触发**：在目标 session 内运行 `/code-review`（可加 `--comment` 发为 PR 行内评论，`--fix` 直接应用修复，`ultra` 为云端多 agent 深审）。

---

## 5. Skills 协同规划（与 codegraph 及整体项目的配合）

QuantMind 现有 codegraph（MCP）、trading-skills、scientific-figure-skill、db-migration，加上本批三个工具，按"职责分层"配合：

| 层 | 工具 | 角色 | 何时用 |
|----|------|------|--------|
| **结构真相源** | **codegraph**（MCP，`codegraph_*`） | 符号 / 调用 / 影响图的 AST 级事实基准 | "X 在哪定义""改 Z 会断什么""谁调用 Y" —— 写代码、审查、影响分析时第一选择 |
| **任务过程落盘** | **planning-with-files** | 多步任务的计划/发现/进度写盘，跨 session 续作（含 /clear 恢复） | 5+ 工具调用的复杂任务（如 E3 成本修正、标签从 63d→12d 重构） |
| **被动历史记忆** | **claude-mem** | 自动捕获过程观察并注入历史上下文（**非真相**） | 跨 session 回忆"上次做到哪""为什么这么改" |
| **合并前闸门** | **code-review** | 独立 session 的代码审查 | 回测/模型/因子改动 merge 前 |
| **领域方法论** | **trading-skills**：backtest-expert / macro-regime-detector / position-sizer | 回测鲁棒性、Regime 切换、仓位/Kelly 调参 | 设计新回测逻辑、调 Regime 权重、调仓位上限时 |
| **成果可视化** | **scientific-figure-skill** | NAV 曲线、IC 序列、Barra 归因图 | 输出研究报告图表 |
| **数据迁移** | **db-migration** | parquet/json → PG/Mongo 迁移与一致性校验 | DB 迁移任务 |

### 推荐协作流（以"一次回测/模型改动"为例）

1. **planning-with-files** 起一个 `.planning/<id>/` 计划，拆解步骤、记录发现/进度（定稿后归档到 `docs/plans/`）。
2. **codegraph** 定位要改的符号、用 `codegraph_impact` 评估影响面。
3. 必要时调 **trading-skills**（回测设计 / Regime / 仓位）方法论。
4. 写代码、跑回测；**scientific-figure-skill** 产出 NAV/IC 图。
5. **另起一个新 session** 跑 **code-review** 做合并前验收（结合 codegraph 影响面复核）。
6. 全程 **claude-mem** 被动记录过程；凡涉密内容用 `<private>` 包裹。

### 优先级与防冲突

- **事实判断以 codegraph + `docs/` 交接文档为准**，claude-mem 注入的历史观察仅作参考、不作真相。
- codegraph 索引在改文件后约 500ms 防抖；同一回合改完别立刻查，必要时手动 `wsl -e bash -ic 'qm-sync'`。
- claude-mem 的 `CLAUDE_MEM_SKIP_TOOLS` 已排除 `Skill` / `SlashCommand`，不会把调用其它 skill 的动作存成噪音记忆。

---

## 6. 记忆卫生提醒（原文）

> **claude-mem 捕获的是过程观察，不是事实真相。本项目有过被更正的误判（如早期 'PG 空表' 结论已推翻；标签即将从 63 日改为 12 日）。若 claude-mem 注入了过时结论，以 docs/ 下的交接文档和当前 spec 为准。**

---

## 7. 本次改动清单（自证未碰业务）

仅新增 / 改动以下内容，全部为工具链：
- 新增 `docs/plans/`（含 `docs/plans/tooling_setup_plan.md`）
- 新增 `docs/TOOLING_SETUP.md`（本文件）
- 工具自身配置（**均在仓库之外**，不进 git）：
  - `C:\Users\lenovo\.claude-mem\`（claude-mem 全局数据 + `settings.json`）
  - `C:\Users\lenovo\.claude\plugins\`（planning-with-files / code-review 插件 + marketplace 注册）
  - 全局工具：Bun v1.3.14（`npm install -g bun`）

未触碰 `quantmind/`、`scripts/`、`tests/`、`models/`、`data/`、`app/`、`backtesting/` 等任何业务目录与 parquet/json 数据。
（注：`git status` 中既有的 `M data/...`、`M scripts/...` 为本任务**开始前已存在**的改动，与本任务无关。）
