"""快速语法检查，验证所有改动文件 AST 可解析。"""
import ast
import sys

FILES = [
    "app/pages/7_系统控制台.py",
    "app/pages/13_系统健康.py",
    "app/ops/db_health.py",
    "app/db/writers.py",
]

errs = 0
for f in FILES:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f"  ✓ {f}")
    except SyntaxError as e:
        print(f"  ✗ {f}: {e}")
        errs += 1

sys.exit(errs)
