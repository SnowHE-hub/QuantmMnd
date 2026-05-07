"""QuantMind CLI 入口.

支持 `python -m quantmind ...` 与 console_scripts 安装的 `quantmind` 命令。
当前是占位实现，Phase 0 完成后会扩展为完整 CLI（download / backtest / agent ...）。
"""

from __future__ import annotations

import sys


def main() -> int:
    """CLI 主入口."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(_HELP)
        return 0

    if args[0] == "version":
        from quantmind import __version__

        print(f"quantmind {__version__}")
        return 0

    if args[0] == "smoke":
        from quantmind.core.smoke import main as smoke_main

        return smoke_main()

    print(f"Unknown command: {args[0]}\n")
    print(_HELP)
    return 1


_HELP = """\
QuantMind CLI（Phase 0 占位版）

Usage:
    quantmind <command> [args...]

Commands:
    version         打印版本号
    smoke           跑环境/配置/LLM 烟雾测试
    -h, --help      显示帮助

后续 Phase 会扩展：
    quantmind download   下载数据
    quantmind features   构建因子
    quantmind backtest   跑回测
    quantmind agent      跑 Agent 研究
    quantmind ui         启动 Streamlit
"""


if __name__ == "__main__":
    raise SystemExit(main())
