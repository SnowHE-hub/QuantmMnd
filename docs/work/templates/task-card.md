# T-xxx · <一句话标题>
- 状态: created | assigned | implemented | accepted | rejected
- 实现者: <K1/K2/K3/G2/C3> · <模型>    备选: <...>
- 预算: <Cursor 对话数 / Fable 票号与上限 $>
- 分支: t/xxx
- 依赖: <T-yyy 或 无>

## 目标（一句话）

## 背景与证据（带证据等级，数字指到文件）

## allowed_paths（只能改这些）

## 不许做

## 验收命令（C2 逐条执行，全部通过才 ACCEPT；必须客观、可执行）
1.
2.
3. `git diff --stat main...t/xxx` 只触及 allowed_paths
4. `pytest tests/ -q -p no:cacheprovider` 无新增失败

## 实现说明（实现者填写：做了什么、没做什么、已知风险）

## 验收结论（C2 填写，链接 acceptance/T-xxx.md）
