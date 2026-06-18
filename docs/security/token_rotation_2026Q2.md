# Tushare Token 轮换记录 — 2026 Q2

> 安全记录，**不含任何 token 字符**；token 一律以 sha256[:8] 标识。

## 事件
- **动作**：Tushare 主 token（TUSHARE_TOKEN / Token A）轮换。
- **时间**：2026-06-18（CST）。
- **触发**：F-08 已收口 API，但旧 token 未轮换 + git 历史含旧 token（见 git_history_rewrite_2026Q2.md）。

## 验证证据（API 实证，无 token 明文）
| 项 | api | 结果 | 记录数 |
|---|---|---|---|
| **新 token**（sha256[:8]=`606181d8`，长度56） | `pro.stock_basic(list_status=L)` | **通过 ✅** | **5529** |
| **旧 token ×14**（历史中 distinct） | `pro.stock_basic(...)` | **全部鉴权失败 ✅**（"您的token不对，请确认。"） | — |

→ **新 token 可用、全部旧 token 已死**。`scripts/survivorship/_token_rotation_verify.py` 可复跑（PASS）。

## 凭据文件更新
- `api_key.txt`（`TUSHARE_TOKEN：` 行）、`.env`（`TUSHARE_TOKEN=` 行）已写入新 token。两文件均 **gitignored**（未入库）。
- `TUSHARE_TOKEN_HI`（高频代理 Token B）：**亦已死**，未替换（代理服务当前不用，全部拉数已完成）；
  如将来需高频代理，再单独 provision。

## ⚠ 本次操作中的次生事件（需另行处置）
- **GitHub PAT 泄漏**：核查 api_key.txt 时，脱敏正则仅匹配 hex 串，导致 `GitHub_token`（`ghp_…`，含非 hex 字符）
  在会话输出中**明文显示** → 已进会话记录。**该 GitHub PAT 须立即轮换**。
- **新 Tushare token 经聊天明文传递** → 也已进会话记录；建议本轮全部完成后**再轮换一次**或知悉半暴露状态。
- 教训：脱敏不能只匹配 hex；`ghp_`/`sk-`/`gh[ps]_` 等前缀 token 含字母，需更宽正则或整行打码。

## 代码侧根因（已修，commit 76535a3）
- `tushare_provider.py:83/85`、`prefetch_bulk_with_hi.py:107/114` 的 `token[:8]` 打印行已删。
