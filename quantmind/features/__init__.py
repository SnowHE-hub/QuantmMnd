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

from quantmind.features.expansion import (
    CATEGORICAL_EXPOSURE_FIELDS,
    EXPANSION_FACTORS,
    MARKET_LEVEL_FEATURES,
    compute_all_expansion_factors,
)
from quantmind.features.fundamental import (
    FUNDAMENTAL_FACTORS,
    compute_all_fundamental_factors,
)
from quantmind.features.panel import (
    build_features_for_date,
    build_panel,
    compute_forward_returns,
    fetch_forward_price_pivot,
    month_end_dates,
    quarter_end_dates,
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
from quantmind.features.analyst_revision import (
    ANALYST_REVISION_FACTORS,
    build_analyst_revision_factor,
    compute_analyst_ic,
    fetch_analyst_reports,
    run_analyst_pipeline,
)
from quantmind.features.text_sentiment import (
    TEXT_SENTIMENT_FACTORS,
    build_ann_sentiment_factor,
    compute_ic as text_sentiment_ic,
    run_full_pipeline as run_text_sentiment_pipeline,
    score_announcements,
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
    # analyst revision
    "ANALYST_REVISION_FACTORS",
    "build_analyst_revision_factor",
    "compute_analyst_ic",
    "fetch_analyst_reports",
    "run_analyst_pipeline",
    # text sentiment
    "TEXT_SENTIMENT_FACTORS",
    "build_ann_sentiment_factor",
    "run_text_sentiment_pipeline",
    "score_announcements",
    "text_sentiment_ic",
    "CATEGORICAL_EXPOSURE_FIELDS",
    "EXPANSION_FACTORS",
    "FUNDAMENTAL_FACTORS",
    "FeaturePipeline",
    "MARKET_LEVEL_FEATURES",
    "SENTIMENT_FACTORS",
    "TECHNICAL_FACTORS",
    "build_features_for_date",
    "build_panel",
    "cagr",
    "compute_all_expansion_factors",
    "compute_all_fundamental_factors",
    "compute_all_sentiment_factors",
    "compute_all_technical_factors",
    "compute_forward_returns",
    "cross_section_rank",
    "cross_section_zscore",
    "fetch_forward_price_pivot",
    "fillna_cross_section",
    "information_coefficient",
    "latest_report_per_ticker",
    "list_all_factor_names",
    "month_end_dates",
    "neutralize",
    "pivot_prices",
    "quarter_end_dates",
    "safe_divide",
    "standardize",
    "winsorize",
    "ytd_to_ttm",
]
