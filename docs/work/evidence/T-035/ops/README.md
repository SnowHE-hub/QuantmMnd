# D:\qm_ops — QuantMind 运行保障（T-035 P0，2026-10-21 用户批准）

| 文件 | 作用 |
|---|---|
| `wsl_supervisor.ps1` | 登录后常驻：挂一个 `wsl.exe` 会话保活；每 5 分钟检查会话和 cron，挂了就重启；写心跳文件 |
| `qm_wsl_supervisor.vbs` | 启动项：隐藏窗口运行上面的脚本。副本放在"启动"文件夹（`shell:startup`） |
| `logs/wsl_supervisor_<日期>.log` | 运行日志 |
| `logs/heartbeat.json` | 最近一次心跳（UTC、保活进程号、cron 状态） |

另外，`%UserProfile%\.wslconfig` 增加了：

```ini
[general]
instanceIdleTimeout=-1
```

这是微软文档里关闭发行版空闲自动关机的设置。原文件备份在 `.wslconfig.bak_20261021`。

**撤销**：
1. 删除启动文件夹里的 `qm_wsl_supervisor.vbs`；
2. 结束 `wsl_supervisor.ps1` 进程；
3. 用备份恢复 `.wslconfig`；
4. 执行 `wsl --shutdown`。

**局限**：关机、睡眠、注销之后都会停。重新登录后脚本会自动启动，由编排器补跑数据和研究；过期的交易意向不执行。
