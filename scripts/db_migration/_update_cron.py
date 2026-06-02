"""
把 crontab 里的 4 个写入脚本加上 WRITE_MODE=dual 前缀。
用法: python scripts/db_migration/_update_cron.py [--dry-run]
"""
import subprocess
import sys

DRY_RUN = "--dry-run" in sys.argv

# 需要注入双写模式的脚本关键词（精确匹配命令部分）
TARGETS = [
    "scripts/daily_update.py",
    "scripts/track_realized_pnl.py",
    "scripts/dispatch_loss_signals.py",
    "scripts/update_sim_strategy.py",
]

result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
old_crontab = result.stdout

new_lines = []
changed = 0
for line in old_crontab.splitlines():
    stripped = line.strip()
    # 跳过注释和空行
    if stripped.startswith("#") or not stripped:
        new_lines.append(line)
        continue
    # 检查是否是我们要改的写入脚本行
    needs_inject = any(target in line for target in TARGETS)
    if needs_inject and "WRITE_MODE=dual" not in line:
        # 在 python/python3 命令前注入 WRITE_MODE=dual
        # 处理两种格式:
        #   ... && python scripts/xxx
        #   ... && export X= && python scripts/xxx
        new_line = line
        for py in ["&& /home/lenovo/miniforge3/envs/quantmind/bin/python3 scripts/",
                   "&& /home/lenovo/miniforge3/envs/quantmind/bin/python scripts/"]:
            if py in new_line:
                new_line = new_line.replace(py, py.replace("&& /", "&& WRITE_MODE=dual /"))
                break
        new_lines.append(new_line)
        changed += 1
        print(f"[PATCH] {line.split()[-1] if line.split() else line[:60]}")
        print(f"  → WRITE_MODE=dual 已注入")
    else:
        new_lines.append(line)

new_crontab = "\n".join(new_lines) + "\n"

if DRY_RUN:
    print("\n─── 预览（--dry-run，不实际写入）───")
    print(new_crontab)
    print(f"共 {changed} 条已修改")
    sys.exit(0)

# 写入新 crontab
proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True)
if proc.returncode == 0:
    print(f"\n✓ crontab 已更新，{changed} 条加入 WRITE_MODE=dual")
else:
    print(f"✗ crontab 更新失败")
    sys.exit(1)

# 验证
print("\n─── 当前 crontab ───")
subprocess.run(["crontab", "-l"])
