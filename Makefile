# ============================================================================
# QuantMind Makefile
# 用法： make <target>
# 假设你已激活 conda 环境 quantmind
# ============================================================================

.PHONY: help install install-dev install-all test lint format typecheck clean \
        run-ui build-features run-backtest run-agent download-data smoke

# 默认显示帮助
help:
	@echo "QuantMind 常用命令："
	@echo ""
	@echo "  环境与依赖："
	@echo "    install         只装核心依赖（最小可运行）"
	@echo "    install-dev     装核心 + dev 工具"
	@echo "    install-all     装全部依赖（推荐开发机）"
	@echo ""
	@echo "  代码质量："
	@echo "    lint            ruff 检查"
	@echo "    format          ruff 格式化"
	@echo "    typecheck       mypy 类型检查"
	@echo "    test            pytest 全套（不含 slow / gpu）"
	@echo "    test-all        pytest 包含 slow / gpu"
	@echo "    test-pit        只跑 PIT 正确性测试（核心）"
	@echo ""
	@echo "  运行："
	@echo "    smoke           Smoke test：环境 + 配置 + LLM 调用"
	@echo "    run-ui          启动 Streamlit UI"
	@echo "    download-data   下载 csi300 历史数据"
	@echo "    build-features  构建因子特征"
	@echo "    run-backtest    跑量化策略回测"
	@echo "    run-agent       跑 Agent 单股票分析"
	@echo ""
	@echo "  清理："
	@echo "    clean           清理临时文件"

# ----------------------------------------------------------------------------
# 安装
# ----------------------------------------------------------------------------
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

install-all:
	uv pip install -e ".[all]"

install-data:
	uv pip install -e ".[data,dev]"

install-ml:
	uv pip install -e ".[data,ml,dev]"

install-llm:
	uv pip install -e ".[data,llm,dev]"

# ----------------------------------------------------------------------------
# 代码质量
# ----------------------------------------------------------------------------
lint:
	ruff check quantmind tests scripts

format:
	ruff format quantmind tests scripts
	ruff check --fix quantmind tests scripts

typecheck:
	mypy quantmind

test:
	pytest -m "not slow and not gpu"

test-all:
	pytest

test-pit:
	pytest -m pit -v

test-cov:
	pytest --cov=quantmind --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ----------------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------------
smoke:
	python -m quantmind.core.smoke

run-ui:
	streamlit run quantmind/ui/streamlit_app.py

download-data:
	python scripts/download_data.py --universe csi300 --start 2018-01-01 --end 2024-12-31

build-features:
	python scripts/build_features.py --universe csi300

run-backtest:
	python scripts/run_backtest.py --strategy lgbm_factor --start 2022-01-01 --end 2024-12-31

run-agent:
	python scripts/run_agent_research.py --ticker 300750.SZ

# ----------------------------------------------------------------------------
# 清理
# ----------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "Cleaned."

clean-cache:
	rm -rf .cache/*
	@echo "Cache cleared."

clean-data:
	@echo "WARNING: This will delete all downloaded data. Press Ctrl+C to cancel..."
	@sleep 5
	rm -rf data/raw/* data/processed/* data/features/* data/snapshots/*
	@echo "Data cleared."
