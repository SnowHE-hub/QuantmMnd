# Claude Code 辅助工具链安装计划（tooling_setup_plan.md）

> 本文档为**纯工具链配置**计划。严禁修改 QuantMind 任何业务代码 / 数据 / 模型 / parquet / 测试。
> 仅允许：安装工具、写工具配置文件、在 `docs/` 下新建文档。
> 生成时间：2026-06-04

## 0. 环境勘察结论

| 项目 | 结论 |
|------|------|
| Claude Code 运行位置 | **Windows 原生**（`C:\Users\lenovo\.local\bin\claude.exe`） |
| Windows Node / npx | node v24.15.0 / npx 11.12.1（claude-mem 用此版本） |
| WSL Node / npx | node v20.20.2 / npx 10.8.2（codegraph 用此版本） |
| Claude 配置根目录 | `C:\Users\lenovo\.claude\`（skills / plugins / settings.json / projects 均在此） |
| 已装 skills | `claude-trading-skills`、`db-migration`、`scientific-figure-skill` |
| 已配 marketplace | `claude-plugins-official`（anthropics/claude-plugins-official），其中含 `code-review` 插件 |
| 已装 plugins | **无**（`claude plugin list` = empty）；`/code-review` 当前由内置命令提供 |
| 项目 docs/ | 存在；`docs/plans/` 本次新建 |
| 非交互插件 CLI | `claude plugin marketplace add`、`claude plugin install <p>@<m>`、`claude plugin list` 可用 |

**关键事实**：项目文件在 WSL（`/home/lenovo/projects/quantmind`），但 Claude Code 本体与 skills/plugins 在 **Windows 侧**。因此三个工具都必须装到 Windows 的 `C:\Users\lenovo\.claude\`，而不是 WSL home（WSL `~/.claude/` 并不存在）。

## 1. 三个工具的安装方式（以官方文档为准）

### 1.1 claude-mem（持久记忆插件）
- **来源**：<https://github.com/thedotmack/claude-mem> · 文档 <https://docs.claude-mem.ai/installation>
- **安装命令**：`npx claude-mem install`（在 Windows 侧运行，目标 IDE 默认 Claude Code）
  - 官方明确：`npm install -g claude-mem` 只装 SDK，不注册 hooks/worker —— **不可用**该方式。
- **作用域**：官方文档说明 claude-mem **仅支持全局安装**（global-only），不提供 project-scope 选项。任务要求的"本项目作用域"无法由 claude-mem 原生支持 —— 将在 `docs/TOOLING_SETUP.md` 中记录此偏差，并通过 `<private>` 标签 + `CLAUDE_MEM_SKIP_TOOLS` 做安全收敛。
- **安装行为**：运行时自检（缺失则自动装 Bun / uv）→ 检测 IDE 并写入 hooks → 启动后台 worker。
- **验证命令**：`npx claude-mem status`、`npx claude-mem doctor`。

### 1.2 planning-with-files（计划落盘 skill）
- **来源**：<https://github.com/OthmanAdi/planning-with-files> · 安装文档 <https://github.com/OthmanAdi/planning-with-files/blob/master/docs/installation.md>
- **安装命令**（非交互 CLI）：
  ```
  claude plugin marketplace add OthmanAdi/planning-with-files
  claude plugin install planning-with-files@planning-with-files
  ```
- **产物**：维护 `task_plan.md` / `findings.md` / `progress.md` 三个 markdown 跨 session 持久化。
- **输出目录**：官方文档未提供"默认输出目录"配置项 → 采用**项目约定**：所有计划文件统一放在 `docs/plans/`（见 TOOLING_SETUP.md）。
- **验证命令**：`claude plugin list`。

### 1.3 code-review（代码审查 skill）
- **来源**：Anthropic 官方插件 `code-review`（marketplace `claude-plugins-official`，仓库 <https://github.com/anthropics/claude-plugins-official>）· 文档 <https://code.claude.com/docs/en/code-review>
- **现状**：`/code-review` 已由 Claude Code **内置命令**提供；本次额外安装官方 marketplace 插件以固定来源。
- **安装命令**：`claude plugin install code-review@claude-plugins-official`
- **配置**：无需特殊配置；仅在 TOOLING_SETUP.md 写死使用约定（独立 session 审查、合并前闸门）。
- **验证命令**：`claude plugin list`。

## 2. 将要写入的配置（占位符，不含真实密钥）

### claude-mem `~/.claude-mem/settings.json`（Windows = `C:\Users\lenovo\.claude-mem\settings.json`）
- `CLAUDE_MEM_DATA_DIR` = `~/.claude-mem`（本地 SQLite，路径将实测打印）
- `CLAUDE_MEM_SKIP_TOOLS`（**工具输出排除列表** —— 即任务所指的"工具输出排除项"）：
  在默认 `ListMcpResourcesTool,SlashCommand,Skill,TodoWrite,AskUserQuestion` 基础上，
  追加易 dump 凭据/大数据量的工具：`Read,Bash,PowerShell`。
  - 理由：`Read` 可能读取 parquet 大面板与 `.env`/配置中的密钥；`Bash`/`PowerShell` 输出常含
    `psql` 连接串、Tushare token 回显、`git remote` 中的 token。
- 敏感信息排除依赖 `<private>` 标签（hook 层在落库前剥离），覆盖：Tushare token、PostgreSQL `quantmind` 用户密码/连接串、GitHub SSH key/token、未来客户数据 / JWT secret / API key。

### planning-with-files
- 约定（非配置项）：计划文件输出目录 = `docs/plans/`。

### code-review
- 约定（非配置项）：仅在独立新 session 中运行，审查另一 session 写的代码；作为回测/模型改动合并前的独立验收闸门。

## 3. Skills 协同规划（codegraph / trading-skills / 本批工具）

详见 `docs/TOOLING_SETUP.md` 的"协同"章节。核心分工：
- **codegraph（MCP）**：结构事实源（符号/调用/影响），写代码与审查时的真相基准。
- **planning-with-files**：把多步任务的计划/发现/进度落盘到 `docs/plans/`，跨 session 续作。
- **claude-mem**：过程观察的被动记忆（非真相），注入历史上下文；以 docs/ 交接文档为准纠偏。
- **code-review**：独立 session 的合并前闸门，结合 codegraph 做影响面核对。
- **trading-skills（backtest-expert / macro-regime-detector / position-sizer）**：领域方法论，设计回测/Regime/仓位时调用。

## 4. 验收
- 三个工具 status/list 通过。
- `docs/plans/tooling_setup_plan.md` 与 `docs/TOOLING_SETUP.md` 均生成。
- `git status` 仅显示 docs/ 新增 + 工具自身配置（不在仓库内），无任何业务代码/数据改动。
