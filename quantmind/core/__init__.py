"""quantmind.core: 项目基础设施.

模块（部分将在后续 Task 0.2 实现）：
    smoke      - Phase 0 烟雾测试（已实现）
    config     - pydantic-settings 配置加载（Task 0.2）
    logger     - loguru 日志（Task 0.2）
    cache      - joblib + diskcache（Task 0.2）
    llm_router - 多 LLM provider 统一路由（Task 0.2）
    state      - LangGraph 全局 State Pydantic Schema（Task 0.2）
"""

__all__ = [
    "smoke",
]
