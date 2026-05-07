"""quantmind.features — 因子库（基本面 / 量价 / 情绪）+ 标准化流水线.

主要 API
========

::

    from quantmind.features import (
        FeaturePipeline,
        compute_all_fundamental_factors,
        compute_all_technical_factors,
        compute_all_sentiment_factors,
        standardize,
        information_coefficient,
        list_all_factor_names,
    )
"""

from quantmind.features.fundamental import (
    FUNDAMENTAL_FACTORS,
    compute_all_fundamental_factors,
)
from quantmind.features.pipeline import (
    ALL_FACTOR_GROUPS,
    FeaturePipeline,
    list_all_factor_names,
)
from quantmind.features.sentiment import (
    SENTIMENT_FACTORS,
    compute_all_sentiment_factors,
)
from quantmind.features.standardize import (
    cross_section_rank,
    cross_section_zscore,
    fillna_cross_section,
    information_coefficient,
    neutralize,
    standardize,
    winsorize,
)
from quantmind.features.technical import (
    TECHNICAL_FACTORS,
    compute_all_technical_factors,
)
from quantmind.features.utils import (
    cagr,
    latest_report_per_ticker,
    pivot_prices,
    safe_divide,
    ytd_to_ttm,
)

__all__ = [
    "ALL_FACTOR_GROUPS",
    "FUNDAMENTAL_FACTORS",
    "FeaturePipeline",
    "SENTIMENT_FACTORS",
    "TECHNICAL_FACTORS",
    "cagr",
    "compute_all_fundamental_factors",
    "compute_all_sentiment_factors",
    "compute_all_technical_factors",
    "cross_section_rank",
    "cross_section_zscore",
    "fillna_cross_section",
    "information_coefficient",
    "latest_report_per_ticker",
    "list_all_factor_names",
    "neutralize",
    "pivot_prices",
    "safe_divide",
    "standardize",
    "winsorize",
    "ytd_to_ttm",
]
