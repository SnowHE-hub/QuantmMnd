# T-004 · v4 标签缺陷影响评估（草稿，待 G1 评审和 C2 复现）

- 日期：2026-10-22。执行：Antigravity 会话，按 T-004 卡的 4 个问题逐个查。
- 证据等级：`[代码]` 读代码得到；`[实测]` 跑出来的数；`[推断]` 由上面两类推出，没有直接验证。
- 所有重放都是只读：代码和面板从 `~/projects/quantmind` 读（不写字节码），输出只写 `data/research/t004/`。没有碰 `models/`、`reports/`、已有面板。
- 脚本：`docs/work/evidence/T-004/t004_replay.py`、`t004_v6_replay.py`、`t004_meta_replay.py`。
- 对比用面板：`alpha_panel_v4`（旧，有缺陷）、`alpha_panel_v4_1`（T-003 修正，行和因子列与 v4 完全相同，只有两列标签不同）、`alpha_panel_v4_2`（又延长到 2026-10，标签更全，只做参考）。

## 先说结论
1. **README / HANDOVER 里的 LGBM v6 ICIR = 0.380，基本是标签污染加事后选方向造成的。** 用同样的代码和参数复现出 0.3803，只把评估标签换成 v4_1 就降到 0.097，训练和评估都换成 v4_1 是 0.004（有效 IC 均值 0.0002）。
2. **卡里的假设「meta-learner 的 2024-Q3 AUC 0.24、2024-Q4 AUC 0.50 由标签污染造成」不成立，被证伪。** `hit` 不是 `actual_return_63d > panel_return_63d`，而是 `actual_rank <= predicted_rank`，和 v4 标签没有关系。用 v4 和 v4_1 各跑一遍，AUC 逐位相同。
3. 顺序验证（`run_sequential_validation.py`）的 8 轮里，换成 v4_1 后 IC 全变了，champion 从 Round 4 变成 Round 6。被晋升成 `lgbm_v6_alpha` 的正是 Round 4，它的早停验证集和方向判断用的恰好是两个被污染的截面。
4. 板块模型（main / gem / star）在 v4 上逐位复现；换成 v4_1 后 main、gem 的 IC 约减半，star 的数字大幅变化但样本太小不可信（§2.2）。
5. 另外发现两件卡外的事：被污染的截面不止 2 个；HANDOVER 里 meta-learner 的 CV AUC 0.6025 现在复现不出来（得到 0.50）。

## 1. 被污染的到底是哪些截面
`[实测]` `data/research/t004/label_diff_by_date.csv`（v4 与 v4_1 逐截面比较，63d 标签共 28 个截面、24,168 行）：

| 截面 | 行数 | 差 > 0.5% 的占比 | 差 > 10% 的占比 | Spearman(v4, v4_1) |
|---|---:|---:|---:|---:|
| 2024-09-30 | 306 | 97.7% | 87.6% | 0.21 |
| 2024-12-31 | 300 | 97.7% | 86.7% | −0.01 |
| 2023-03-31 | 1,349 | 27.4% | 2.4% | 0.94 |
| 2022-03-31 | 1,293 | 24.6% | 2.9% | 0.95 |
| 2021-03-31 | 1,238 | 24.3% | 3.6% | 0.96 |
| 2020-03-31 | 1,185 | 23.5% | 1.4% | 0.98 |
| 2023-06-30 / 2022-09-30 / 2021-06-30 / 2020-06-30 | 1,194–1,355 | 10%–13% | 0.4%–1.0% | 0.988–0.991 |
| 其余截面 | | < 1%，多数为 0 | | ≥ 0.996 |

- 全部 63d 标签里，差 > 10% 的占 2.9%，与 T-003 的 2.91% 一致。
- 21d 标签：只有 2024-12-31（97.7% 差 > 0.5%）和 2023-03-31（27.2%）明显。
- 除两个严重截面外，每年一季度和年中截面也有 10%–27% 的标签差 > 0.5%，但幅度小（差 > 10% 的只有 1%–4%）。T-003 只点名了 2024-09-30 和 2024-12-31，实际受影响的截面更多，轻重不同。

## 2. 问题 1：哪些模型或结论用到了被污染标签
`[代码]` 逐个读 `models/*.meta.json` 的 `panel_path` 和 `label_col`：
- **直接用 v4 面板训练：只有 `lgbm_v6_alpha`**（63d，2019-03 到 2026-06，20 折）。
- `lgbm_v3_*`、`lgbm_v5_*`、`lgbm_regime_*` 用的是 v2 / v3 面板，这些面板的标签没有被 T-003 审计，**我没有检查它们是否有同类问题**，所以不能说它们没事。
- 板块模型 `lgbm_v6_main / gem / star`（`board_model_results.json`，ICIR 0.384 / 0.155 / 0.007）由 `train_board_models.py` 在 v4 面板上训练，**同样受影响，已重算，见 §2.2**。
- 其他读 v4 面板的脚本和模块：`champion_challenger.py`、`run_sequential_validation.py`、`train_meta_learner.py`、`track_realized_pnl.py`、`backfill_recommendations.py`、`daily_scorer.py`、`factor_cnn.py`、`dynamic_weights.py`（其中 IC 数字来自「alpha_panel_v4 28 季」）。读标签的只有训练、验证和 `track_realized_pnl` 的交叉核对；`daily_scorer`、`backfill_recommendations` 从注释和 grep 命中看只读因子列（`[推断]`，没有逐行读代码）；因子列 v4 与 v4_1 逐位相同（`[实测]`，73 列、29,406 行），所以它们若只读因子列就不受影响。
- T-010 已经用 v4_1 做过一次同类对比，同样的 42 特征在 v4_1 标签上 RankIC −0.009 / ICIR −0.17（`auto_flip=False`），方向与本文一致。

### 2.1 LGBM v6 的 ICIR 0.380 复算
`[实测]` `t004_v6_replay.py`，参数取自 `models/lgbm_v6_alpha.meta.json`（42 特征、63d、400 树 / 31 叶 / lr 0.05 / 早停 50、walk-forward `min_train=10`、`n_val=2`）。用 v4 复现出和 meta 里完全一致的数：20 折、IC 均值 −0.0371、ICIR −0.3803、胜率 0.3333。

| 训练 / 验证标签 | 评估标签 | IC 均值 | ICIR | 说明 |
|---|---|---:|---:|---|
| v4 | v4 | −0.0371 → 反向后 +0.0371 | −0.380 → **+0.380** | 已发布的数 |
| v4 | v4_1 | −0.0089 | −0.097 | 只修评估标签 |
| v4_1 | v4 | −0.0104 | −0.157 | 只修训练标签 |
| v4_1 | v4_1 | −0.0002 | −0.004 | 全部修正 |
| v4_2 | v4_2 | −0.0051 | −0.081 | 延长、标签更全，参考 |

- 所有组合的原始 ICIR 都是负的。0.380 这个正数来自 `auto_flip`：原始 IC 均值为负，就把方向取反，得到「有效 ICIR」。方向是看完测试期 IC 之后才定的。`[代码]`
- 按测试日看（`v6_replay_2x2.json`），评估标签的修正集中在 2024-09-30（−0.025 → +0.131）和 2024-12-31（−0.250 → +0.068）两折。取反之后，后一折在已发布的数里是一个很大的正贡献。`[实测]`
- 训练标签的影响同样大（v4 训练比 v4_1 训练，几乎每折的 IC 都不同，例如 2023-09-30 是 +0.143 对 −0.065）。除两个严重截面外，前面提到的各年一季度截面也在训练窗里。`[实测]`
- 因此 0.380 不能当作模型有预测力的证据。`[推断]`
- 另外，0.380 是 20 折 walk-forward 评估的汇总，而线上的 `lgbm_v6_alpha.pkl` 是顺序验证 Round 4 的单次训练（`model_lineage.json`）。它的 meta 把两者拼在一起：`effective_ic_ir` 0.3803 来自 walk-forward，`ic_mean` 0.030841 来自晋升时的评估（序贯验证里 Round 4 的 IC 是 0.033779，两者也不相等，原因没有查）。`[实测]`

### 2.2 板块模型（main / gem / star）
`[实测]` 直接运行原脚本 `scripts/train_board_models.py`（只改 `--panel` 和 `--out-dir`，模型和结果都写到 `data/research/t004/board_{v4,v4_1}/`，没有写 `models/`）。混训基准仍是线上的 `lgbm_v6_alpha.pkl`。
**在 v4 上重跑与已存的 `models/board_model_results.json` 三个板块全部逐位一致**（IC、ICIR、胜率、折数）。

| 板块（股票数 / 折数） | 指标 | v4（已发布） | v4_1 |
|---|---|---:|---:|
| main（1,186 / 22） | 专用模型 IC | 0.0384 | 0.0207 |
| | ICIR | 0.384 | 0.299 |
| | 胜率 | 0.40 | 0.60 |
| | 混训基准 IC | 0.0653 | 0.0646 |
| gem（135 / 22） | 专用模型 IC | 0.0242 | 0.0114 |
| | ICIR | 0.155 | 0.082 |
| | 胜率 | 0.35 | 0.40 |
| | 混训基准 IC | 0.0754 | 0.0406 |
| star（97 / 18） | 专用模型 IC | 0.0020 | 0.0961 |
| | ICIR | 0.007 | 0.413 |
| | 胜率 | 0.50 | 0.31 |
| | 混训基准 IC | −0.0358 | −0.0417 |

- main、gem 在修正标签后专用模型的 IC 约减半，ICIR 下降（0.384 → 0.299，0.155 → 0.082）；两者在两种标签下都低于混训基准的 IC，「专用模型更好」的说法本来就没有被支持。
- star 反过来：IC 和 ICIR 变大（0.007 → 0.413）。但 star 只有 97 只股票、18 折，胜率只有 31%，一两个截面就能决定均值，我不把它当作有效改进，只当作「数字不稳」的又一个例子。`[推断]`
- 脚本的「方向对齐」逻辑把板块模型的 direction 强制设成混训模型的方向（+1），且 `production_ready` 在三个板块两种标签下都是 True，门禁没有起到区分作用。`[代码 + 实测]`
- 与 §2.1 一样，v4 上的结果可复现，说明差异来自标签而不是脚本或环境。

## 3. 问题 2：meta-learner 的假设
卡里的假设：`hit = actual_return_63d > panel_return_63d`，所以 2024-Q3 AUC 0.24、2024-Q4 AUC 0.50 可能由标签污染造成。

`[代码]` `scripts/track_realized_pnl.py` 里：`hit = actual_rank <= predicted_rank`（在推荐宇宙内按实际收益排名，和预测排名比）。`panel_return_63d` 和 `return_diff` 只是核对用的附加列，不参与 `hit`。
`[实测]` `t004_meta_replay.py`：
- `realized_pnl.parquet` 80 行里，`hit == (actual_rank <= predicted_rank)` 对全部行成立。
- 6 个代理分（由因子列算出）和 `hit` 在 v4 与 v4_1 下逐位相同；两种实现（主 checkout n=100、worktree 去重 n=70）下 LOQO 每季 AUC 都与面板无关。
- 受影响的只有 `panel_return_63d` 这一列：2024-09-30 的 5 行里 3 行差 > 0.5%，2024-12-31 的 10 行全部差 > 0.5%（最大差 1.235），其余季度无差异。

**判定：假设被证伪。** 2024-Q3、Q4 的低 AUC 不是标签污染造成的，原因要另找（样本只有每季 10 只，单季 AUC 噪声很大）。

附带发现：按当前代码和数据复现，LOQO 平均 AUC 是 0.504（n=100）和 0.505（n=70），逐季为 0.44 / 0.28 / 0.33 / 0.29 / 0.58 / 0.84 / 0.76，**和 HANDOVER 里的 0.6025（0.56 / 0.24 / 0.50 / 0.42 / 0.71 / 0.84 / 0.95）对不上**。原因我没有查清，可能是当时的 `realized_pnl` 或 `hit` 口径不同，属 `[推断]`。这件事与标签污染无关，建议另开卡查。

## 4. 问题 3：顺序验证哪些 round 受影响
`[实测]` `t004_replay.py`，特征和参数同已存的 `reports/sequential_val_v4`。**在 v4 上重跑与已存的 8 轮 IC 逐位一致**，说明环境和脚本可信。

| Round | 测试期 | v4 IC（决策） | v4_1 IC（决策） | v4_2 IC（决策） | 受哪里影响 |
|---|---|---|---|---|---|
| 0 | 2024Q1–Q4 | −0.083（晋升） | −0.002（晋升） | +0.009（晋升） | 测试含两个严重截面；训练含各年一季度的轻度污染 |
| 1 | 2024-06 | −0.028（晋升） | +0.047（晋升） | +0.047（晋升） | 训练含轻度污染 |
| 2 | 2024-09 | −0.109（淘汰） | +0.047（**晋升**） | +0.058（晋升） | 测试截面严重污染 |
| 3 | 2024-12 | −0.009（晋升） | +0.023（淘汰） | +0.001（淘汰） | 测试截面严重污染；验证集含 2024-09 |
| 4 | 2025-03 | +0.034（**晋升**） | −0.174（淘汰） | +0.163（晋升） | 早停验证集 = 两个严重截面 |
| 5 | 2025-06 | −0.051（淘汰） | −0.002（淘汰） | −0.096（淘汰） | 训练含两个严重截面；验证集含 2024-12-31 |
| 6 | 2025-09 | +0.017（淘汰） | +0.096（**晋升**） | −0.025（淘汰） | 同上 |
| 7 | 2025-12 | −0.070（淘汰） | +0.064（淘汰） | +0.074（淘汰） | 同上 |
| champion | | Round 4，0.034 | **Round 6，0.096** | Round 4，0.163 | |

- 8 轮都受到影响，没有一轮的 IC 保持不变。晋升决策在 Round 2、3、4、6 变了。
- `train_challenger` 用训练期最后 2 个截面做早停验证集，并用这个验证集判断 `direction`。Round 4 的验证集正好是 2024-09-30 和 2024-12-31。`[代码]`
- **`lgbm_v6_alpha` 就是 v4 上的 Round 4 晋升出来的**（`model_lineage.json`：`v4_round4_2024q4`）。所以线上主模型的早停和方向，是在被污染的标签上定的。`[实测 + 代码]`
- v4_1 与 v4_2 之间也差很多（例如 Round 4：−0.174 对 +0.163）。两者差别只在 v4_2 的标签覆盖更多票和更长价格，说明单轮、一个截面的 IC 本身很不稳定。因此上表里「v4_1 的 champion 是 Round 6」不能当作新的结论，只说明旧结论不成立。`[推断]`

## 5. 问题 4：README / HANDOVER 哪些数字需要加注
建议加注，不改原文：

| 位置 | 内容 | 建议 |
|---|---|---|
| README.md:259；HANDOVER.md:19、146、524、560 | LGBM v6 ICIR = 0.380 | 注明：基于 v4 标签；v4_1 上全部修正后约 0.004；0.380 含 auto_flip 事后取反。见本报告 §2.1 |
| HANDOVER.md §五.五 | meta-learner CV AUC = 0.6025、逐季 0.24 / 0.50 | 注明：与 v4 标签无关；当前代码复现为 0.50，数字待查 |
| HANDOVER.md 序贯验证 Round 4 champion | 「champion = Round 4」 | 注明：v4 污染下的结果，v4_1 下不成立 |
| `board_model_results.json` 的 ICIR | 0.384 / 0.155 / 0.007 | 注明：v4 标签；v4_1 上为 0.299 / 0.082 / 0.413（star 样本太小，不可信），见 §2.2 |
| `dynamic_weights.py` 注释里的「alpha_panel_v4 28 季 IC」 | 因子 IC | 注明：用了 v4 标签，未重算 |

## 6. 没做的事和限制
- `dynamic_weights` 注释里的「alpha_panel_v4 28 季」因子 IC 没有用 v4_1 重算。
- v2 / v3 面板的标签没有审计。
- HANDOVER 里 meta-learner AUC 0.6025 的来历没查清。
- 我没有重训、替换或晋升任何模型，`models/` 没动。
- 这只是草稿：卡里要求 G1 评审、C2 随机抽 2 条结论复现。

## 复现命令（在 `~/projects/qm-t035` 下）
```bash
~/c3_scratch/capped.sh t004replay 4G 'conda activate quantmind; PYTHONDONTWRITEBYTECODE=1 python -u docs/work/evidence/T-004/t004_replay.py'
~/c3_scratch/capped.sh t004v6 4G 'conda activate quantmind; PYTHONDONTWRITEBYTECODE=1 python -u docs/work/evidence/T-004/t004_v6_replay.py'
~/c3_scratch/capped.sh t004meta 3G 'conda activate quantmind; PYTHONDONTWRITEBYTECODE=1 python -u docs/work/evidence/T-004/t004_meta_replay.py'
```
```bash
# 板块模型：原脚本，只改 --panel 和 --out-dir（对 v4 和 v4_1 各跑一次）
~/c3_scratch/capped.sh t004board 4G 'conda activate quantmind; export PYTHONDONTWRITEBYTECODE=1; for n in v4 v4_1; do python -u /home/lenovo/projects/quantmind/scripts/train_board_models.py --panel /home/lenovo/projects/quantmind/data/panel/alpha_panel_$n.parquet --out-dir /home/lenovo/projects/quantmind/data/research/t004/board_$n; done'
```
每个约 10–60 秒。输出在 `data/research/t004/`（共约 0.8 MB），sha256 前 16 位：

| 文件 | sha256（前 16 位） |
|---|---|
| label_diff_by_date.csv | 52eedc4ec07df6e3 |
| v6_replay_2x2.json | ca1d8ca24f6fa712 |
| meta_learner_replay.json | c70132b4ff2b8bdb |
| seqval_v4/round_summary.json | 950e794de2749a98 |
| seqval_v4_1/round_summary.json | 8d45054cce3a8e1d |
| seqval_v4_2/round_summary.json | 819bdf9426c93bcd |
| board_v4/board_model_results.json | 683dfffead225573 |
| board_v4_1/board_model_results.json | eecac653e9a7ab31 |
