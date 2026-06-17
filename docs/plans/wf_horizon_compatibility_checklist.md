# WF Horizon 兼容性自检清单（任何新 horizon 开局必查）

> 来源：P63-3 首跑踩坑——P4 季度 refit 模板平移到 63d 时 fold 结构退化。
> 根因：**embargo ≈ cutoff 间距 → 内部 fold 的 test 窗口坍缩为空**。
> 规则：**任何新 horizon / 新 refit 节奏开跑前，先过本清单。**

## 核心不变式
PurgedWalkForwardSplit 中 fold_k 的 test 窗口 = `(C_k + E, C_{k+1}]`。
要让内部 fold 有非空 test，必须：

> **cutoff 间距(交易日) > embargo E**，且最好 `≥ E + 期望每fold test宽度`。

## 自检步骤（实跑前）
1. **间距 vs (H+E)**：列每对相邻 cutoff 的交易日间距；逐行确认 `gap > E`。
2. **fold 自检表**：打印每 fold 的 `ntrain / nval / ntest`。
   - **内部 ntest=0 的 fold 数必须 = 0**（否则 = 退化，停下报告，换节奏）。
   - 尾部 ntest=0（标签耗尽 `as_of + H > 数据末日`）= 正常"自动收尾"，允许。
3. **标签末日**：确认最后一个 cutoff 的 test 窗口起点 `C_last + E` ≤ 63d/H 标签有效末日；
   否则该 fold 全 NaN，自动收尾掉，记为有效 fold 数减一。

## 各 horizon 的安全 refit 节奏（A 股周频 as_of, 5td/格）
| horizon H | embargo E | 季度(~63td) | 半年(~120td) | 年(~245td) |
|---|---|---|---|---|
| 12d | 20 | ✅ gap63>20 | ✅ | ✅ |
| 21d | 21 | ✅ gap63>21 | ✅ | ✅ |
| **63d** | **63** | ❌ gap63≈E→坍缩 | ✅ gap120>63 | ✅ |

- **12d/21d**：季度 refit OK（P4 用此）。
- **63d**：**最少半年 refit**（季度会坍缩）。P63-3 用半年（4月底/10月底，财报披露后）。
- 经验法则：`refit 间距 ≥ 2×embargo` 才稳。

## 工具
`scripts/survivorship/_p63_3_foldcheck.py` = 自检脚本模板（打间距表 + fold 表 + PASS/FAIL 门）。
