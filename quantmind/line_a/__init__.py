"""quantmind.line_a — 主线 A v2 的运行基础（T-035 P0）.

只读的研究与影子系统：不写模型注册表，不发布正式荐股，不执行真实交易。

模块：
- atomic        原子写文件、sha256
- jobs          持久任务表（SQLite）：幂等任务键、单写者、原子提交、崩溃后续跑
- contracts     数据契约：双时间字段、声明（claim）、快照清单
- availability  按来源计算数据的可用时点（只有日期的数据推到下一个截止点）
- gpu_lease     GPU 租约：同一时刻只有一个进程持有显卡（Windows 与 WSL 两边的模型服务）
- resources     磁盘预算（WSL 虚拟盘与所在物理盘取较小者）
"""
