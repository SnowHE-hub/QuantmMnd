# 测试 markers 约定

> 默认套件（`pytest`）排除下列 markers（pyproject.toml `addopts -m`）；按需单独跑。

| marker | 含义 | 单独跑 |
|---|---|---|
| `requires_network` | 实连 akshare/tushare（离线/限频失败）。7 个（test_pit_correctness 5 + test_lazy_data_engine 2） | `pytest -m requires_network` |
| `requires_db` | 需 PG/Mongo 实例。test_db_backend_parity（module，与 integration 并列） | `pytest -m requires_db`（先起 DB） |
| `requires_optional_deps` | 需 torch/cvxpy/mlflow/langchain。test_factor_cnn / test_momentum_lstm（module） | `pytest -m requires_optional_deps` |
| `stale_panel_fixture` | 依赖陈旧 panel 数据 / stashed factor_model refactor。**债务登记**（见 test_failure_triage_2026Q2.md），解封 stash@{0} 时连同修复 | `pytest -m stale_panel_fixture` |

默认绿：`pytest` → 0 fail / 0 error（实测 1121 passed / 67 deselected / 1 skipped）。
全量（含上述）：`pytest -m ""`（注意 network/db/optional 环境依赖会失败）。
