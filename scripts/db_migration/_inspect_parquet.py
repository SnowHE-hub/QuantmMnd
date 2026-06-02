"""检查关键 parquet 文件的列结构，用于确定 DDL。"""
import pandas as pd

files = {
    "daily_prices_panel": "data/raw/daily_prices_panel.parquet",
    "alpha_prices_panel": "data/raw/alpha_prices_panel.parquet",
    "alpha_daily_basic": "data/alpha_universe/alpha_daily_basic_combined.parquet",
    "fina_indicator": "data/alpha_universe/fina_indicator_combined.parquet",
    "alpha_universe": "data/alpha_universe/alpha_universe.parquet",
    "realized_pnl": "data/feedback/realized_pnl.parquet",
    "regime_features": "data/features/regime_features.parquet",
    "csi300_full_panel": "data/features/csi300_full_panel.parquet",
}

for name, path in files.items():
    try:
        df = pd.read_parquet(path)
        print(f"\n=== {name} ({df.shape}) ===")
        print(df.dtypes.to_string())
        print("index:", df.index.names)
    except Exception as e:
        print(f"\n=== {name}: ERROR {e} ===")
