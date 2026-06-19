# Secrets Handling 守则

> 项目级安全守则。来源：2026Q2 token 轮换 + GitHub PAT 经聊天泄漏事故的直接教训。
> **任何接触 secret 的操作前先读本文。**

## 0. 总则
Secret = 任何凭据：API token（Tushare/DeepSeek/DashScope）、GitHub PAT、SSH key、DB 密码、
代理凭据。一旦进入**聊天 / 日志 / git 历史 / 截图**任一，即视为**已泄漏**，必须轮换。

## 1. 绝不通过聊天/IM 传递 secret（事故直接教训）
- **禁止**把 secret 明文粘到对话（含与 AI 助手的对话）——会进会话记录/日志，等于公开。
  - 2026-06 事故：新 Tushare token 经聊天传递 + GitHub PAT 因脱敏不全在助手输出中明文显示。
- **正确做法**：secret 由人**直接写入凭据文件**（.env / 单文件），告诉助手"已更新"，助手只**读取验证**，
  从不要求/接收 secret 明文。
- 助手侧：任何打印/比对一律用 `sha256(token)[:8]` 标识，**绝不输出 secret 字符**；
  脱敏正则必须覆盖非 hex 前缀（`ghp_`/`gho_`/`sk-`/`xox`…），否则整行打码。

## 2. Secret 分类存放（不混在一个文件）
- **应用配置级**（Tushare/LLM token，应用运行时读）→ `.env`（KEY=VALUE，gitignored）。
- **服务级单文件**（如需与 .env 分离的服务 token）→ 单独 `*.token` 文件（gitignored）。
- **账户级**（GitHub PAT、SSH、云账号）→ **OS keychain / 密码管理器**，**绝不**落项目文件。
  - ⚠ 现状整改：`api_key.txt` 把 DeepSeek/Tushare/DashScope/**GitHub PAT** 混在一文件 = 反面教材。
    GitHub PAT 应移出到 keychain；api_key.txt 仅留服务 token 或废弃，统一迁 .env。

## 3. gitignore 永久规则
`.env`、`.env.*`(除 `.env.example`)、`api_key.txt`、`*.token`、`*_secret*`、`*.pem`、`id_rsa*`
**永久 gitignore**。`.env.example` 只放占位/字段名，**不放真值**。
- 提交前自检：`git diff --cached -G'[0-9a-fA-F]{30,}|ghp_|sk-'`（疑似 secret 则停）。

## 4. 日志/报错模板严禁含 secret 字段（F-08 + 1c 教训）
- 任何 `log/print/raise/f-string` **不得**包含 token/key/密码字段（即使只取前缀 `[:8]`）。
  - 1c 事故：`tushare_provider`/`prefetch_bulk` 曾 `log.info(token[:8])` → 删除。
- 数据拉取脚本入口统一 `loguru.logger.remove()`（清 sink）+ `silence_provider_logging()` 兜底。
- 推荐 pre-commit hook：扫 diff 命中 secret 模式即拒绝提交。

## 5. 轮换与历史改写记录（docs/security/ 约定）
- token 轮换 → `docs/security/token_rotation_<period>.md`（**无 secret 明文**，只记
  时间戳 / api / status / 记录数 / sha256[:8] / 旧 token 作废实证）。
- git 历史改写（filter-repo + force-push）→ `docs/security/git_history_rewrite_<period>.md`
  （改写前后 HEAD SHA / 命令 / remote / 验证结果 / 协作者通知）。
- 泄漏事故 → 同目录记一条 incident，含根因 + 整改。

## 6. 泄漏响应顺序（出事就按这个走）
1. **先轮换**（让泄漏的 secret 失效）——这是真正的修复，**优先于**清历史。
2. 删代码侧根因（日志/硬编码行）。
3. 评估暴露面（public repo？已 push？已被爬？）。
4. 清 git 历史（filter-repo）——secret 已死后属**卫生**，非急务。
5. 记录 incident + 更新本守则。

## 7. 协作者 secret 分发（若引入协作者）
- **不通过 IM/邮件/聊天**发 secret。
- 通过**密码管理器共享条目** 或 **1-on-1 当面/语音**，每人**独立** token（便于单独吊销审计）。
- 协作者离开 → 立即吊销其 token + 轮换共享 secret。

---
**当前待办（本守则触发的整改）**：① 轮换聊天泄漏的 GitHub PAT；② GitHub PAT 移出 api_key.txt 进 keychain；
③（可选）清 public repo 历史中的 2 个已死 Tushare token（见 git_history_rewrite 计划，非急务）。
