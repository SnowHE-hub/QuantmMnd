# Data Manifest（data/ parquet 权威清单）

> 大数据产物不入库（见 .gitignore）。本清单 = "该存在哪些 parquet" 的权威来源；
> 配合各 run_manifest_*.json 的 artifacts_sha256 溯源。生成：`scripts/maintenance/gen_data_manifest.py`。

| 文件 | 行数 | 大小 | sha256[:16] |
|---|---|---|---|
| `data/lake/adj_factor.parquet` | 8,522,667 | 9.9 MB | `ed141fbc54ddd6a5` |
| `data/lake/balancesheet.parquet` | 151,031 | 7.5 MB | `aef2097b76c7e478` |
| `data/lake/cashflow.parquet` | 153,337 | 5.1 MB | `b50856443e2623e4` |
| `data/lake/daily.parquet` | 8,345,172 | 235.1 MB | `6cb81d54ebed9431` |
| `data/lake/daily_basic.parquet` | 8,288,000 | 406.3 MB | `6c479e33f37c6d91` |
| `data/lake/fina_indicator.parquet` | 148,451 | 11.9 MB | `9b9140627a3cf0e1` |
| `data/lake/fina_indicator_full.parquet` | 148,451 | 23.7 MB | `54d7602490677689` |
| `data/lake/hk_hold.parquet` | 1,866,789 | 12.6 MB | `b2264f8738d279eb` |
| `data/lake/income.parquet` | 153,484 | 7.4 MB | `9d9154e779833df8` |
| `data/lake/index_daily.parquet` | 7,048 | 0.5 MB | `6dbd712658c47f27` |
| `data/lake/margin.parquet` | 1,652,984 | 70.9 MB | `c39b1aa92d3943d8` |
| `data/lake/north_bound.parquet` | 1,739 | 0.1 MB | `9c9a8908199a2d69` |
| `data/lake/stock_basic.parquet` | 1,388 | 0.1 MB | `8a43abb44287c95e` |
| `data/lake/stock_basic_full.parquet` | 5,529 | 0.4 MB | `bdb31767e7a2c0eb` |
| `data/panel/alpha_panel_regime_large.parquet` | 17,862 | 10.2 MB | `637d741933ef2cf3` |
| `data/panel/alpha_panel_regime_small.parquet` | 9,618 | 6.5 MB | `f3a6611f828b925b` |
| `data/panel/alpha_panel_v1.parquet` | 6,569 | 2.9 MB | `67862376860f4510` |
| `data/panel/alpha_panel_v2.parquet` | 27,480 | 15.8 MB | `75fa3cba85995f1f` |
| `data/panel/alpha_panel_v2_regime.parquet` | 27,480 | 16.8 MB | `fefb61e2993baa11` |
| `data/panel/alpha_panel_v3.parquet` | 27,480 | 17.9 MB | `19452660f537aa40` |
| `data/panel/alpha_panel_v3_regime.parquet` | 27,480 | 19.3 MB | `9132139817a6f634` |
| `data/panel/alpha_panel_v4.parquet` | 29,406 | 18.1 MB | `8ba08c5e09d01143` |
| `data/panel/alpha_panel_v4_regime.parquet` | 29,406 | 19.4 MB | `1647d50d59cd590f` |
| `data/panel/alpha_panel_weekly_v5.parquet` | 452,439 | 99.0 MB | `f0526740d5971e3b` |
| `data/panel/alpha_panel_weekly_v6.parquet` | 1,561,617 | 511.8 MB | `6878b8ad7872784c` |
| `data/panel/fundamental_factors_v6.parquet` | 1,561,617 | 150.5 MB | `5fb66d2699502c2d` |
| `data/panel/holdout.parquet` | 1,500 | 0.8 MB | `656c34959ffb16b3` |
| `data/panel/monthly_panel.parquet` | 26,700 | 4.7 MB | `8f068c3dca06ab3b` |
| `data/panel/monthly_panel_v2.parquet` | 26,700 | 4.7 MB | `df58ee0ced85ea39` |
| `data/panel/monthly_test.parquet` | 3,600 | 0.9 MB | `09cca8cdadaa6b33` |
| `data/panel/monthly_test_v2.parquet` | 3,600 | 0.9 MB | `2f5a8a1a9acadd43` |
| `data/panel/monthly_train.parquet` | 14,400 | 2.4 MB | `27bdbe7fa437d133` |
| `data/panel/monthly_train_v2.parquet` | 14,400 | 2.4 MB | `ba4886b8f6b8bec0` |
| `data/panel/monthly_val.parquet` | 3,600 | 1.0 MB | `6fc9de3d32e516ae` |
| `data/panel/monthly_val_v2.parquet` | 3,600 | 1.0 MB | `9ac85c302c15da1e` |
| `data/panel/predict.parquet` | 301 | 0.2 MB | `61a355b0bf3cb829` |
| `data/panel/short_horizon_factors.parquet` | 452,439 | 157.3 MB | `52440c1cb55635ec` |
| `data/panel/short_horizon_factors_v6.parquet` | 1,561,617 | 201.9 MB | `7208e6d677b4f8bd` |
| `data/panel/test.parquet` | 898 | 0.5 MB | `67ab7b1d9fa43fd3` |
| `data/panel/train.parquet` | 4,770 | 2.4 MB | `aa6d8e2ba4e98a2d` |
| `data/panel/val.parquet` | 1,196 | 0.7 MB | `5dd5cbab791af864` |
| `data/panel/weekly_asof_grid.parquet` | 350 | 0.0 MB | `7c27b0ddbee7b224` |
| `data/raw/alpha_prices_panel.parquet` | 2,273,529 | 80.9 MB | `8cae4358e7719657` |
| `data/raw/alpha_prices_panel_v6.parquet` | 8,345,172 | 321.8 MB | `4240d87de8ae44b8` |
| `data/raw/daily_prices_panel.parquet` | 728,370 | 28.8 MB | `ac3f0c44ed839f6c` |
| `data/raw/index_daily_panel.parquet` | 6,064 | 0.4 MB | `6b388e06c5e7f56b` |
| `data/bakeoff/alpha158_asof.parquet` | 452,347 | 288.5 MB | `c846636486e06a65` |
| `data/bakeoff/alpha158_asof_v6.parquet` | 1,683,466 | 970.8 MB | `e4ff8d10b02c4e5d` |
| `data/bakeoff/preds/gru_alpha360_12d_quarterly_s0_b0of3.parquet` | 64,321 | 0.4 MB | `fc1767f0dc34984a` |
| `data/bakeoff/preds/gru_alpha360_12d_quarterly_s0_b1of3.parquet` | 63,934 | 0.4 MB | `db0d5df820a006ed` |
| `data/bakeoff/preds/gru_alpha360_12d_quarterly_s0_b2of3.parquet` | 63,975 | 0.4 MB | `30366039f01fdb42` |
| `data/bakeoff/preds/lgbm_full_12d_quarterly.parquet` | 192,231 | 1.0 MB | `b5e90db859b57e70` |
| `data/bakeoff/preds/lgbm_plus_surv_12d_quarterly.parquet` | 192,231 | 0.8 MB | `1e26f61449cfa989` |
| `data/bakeoff/preds/lgbm_v2_35_12d_quarterly.parquet` | 192,231 | 1.0 MB | `61ccccab6979a97c` |
| `data/bakeoff/preds/lstm_alpha360_12d_quarterly_s0_b0of3.parquet` | 64,321 | 0.4 MB | `ae1cbc559289a000` |
| `data/bakeoff/preds/lstm_alpha360_12d_quarterly_s0_b1of3.parquet` | 63,934 | 0.4 MB | `a28d0d2ca3247fae` |
| `data/bakeoff/preds/lstm_alpha360_12d_quarterly_s0_b2of3.parquet` | 63,975 | 0.4 MB | `58045078dc8bcd06` |
| `data/bakeoff/preds/ridge_fnd_only_63d_quarterly_v6.parquet` | 452,413 | 4.7 MB | `ed3501d461e1730c` |
| `data/bakeoff/preds/ridge_full_12d_quarterly.parquet` | 192,231 | 2.0 MB | `2fc7bd9ffe262b91` |
| `data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet` | 700,122 | 7.0 MB | `774eecf97465c489` |
| `data/bakeoff/preds/ridge_full_fnd_63d_quarterly_v6.parquet` | 452,413 | 4.7 MB | `73a762a3ff44d9ec` |
| `data/bakeoff/preds/ridge_full_nofnd_63d_quarterly_v6.parquet` | 452,413 | 4.7 MB | `0fafad789d9652a1` |
| `data/bakeoff/preds/transformer_alpha360_12d_quarterly_s0_b0of3.parquet` | 64,321 | 0.4 MB | `3f87160cef5b2834` |
| `data/bakeoff/preds/transformer_alpha360_12d_quarterly_s0_b1of3.parquet` | 63,934 | 0.4 MB | `f1289ef547ef6ae5` |
| `data/bakeoff/preds/transformer_alpha360_12d_quarterly_s0_b2of3.parquet` | 63,975 | 0.4 MB | `bf23c6b5f05006b3` |
| `data/loss_signals_v4/wf_v2_oos_predictions.parquet` | 191,744 | 4.5 MB | `88667195c90597db` |

**合计：66 个 parquet，3.79 GB**

## 产出脚本对照
- `data/raw/alpha_prices_panel_v6.parquet` ← scripts/survivorship/p2_build_v6.py --prices
- `data/panel/alpha_panel_weekly_v6.parquet` ← p2_build_v6.py --panel
- `data/panel/fundamental_factors_v6.parquet` ← p63_2_build_factors.py
- `data/lake/{daily,adj_factor,daily_basic,fina_indicator,income,balancesheet,cashflow}.parquet` ← scripts/survivorship/p1_pull.py / p63_1*_pull*.py
- `data/bakeoff/preds/ridge_*_v6.parquet` ← p4c_train_v6.py / p63_3_train_v6.py
- `data/bakeoff/alpha158_asof_v6.parquet` ← p4b_alpha158_v6.py (qlib_bakeoff env)
- 各 `run_manifest_*.json` 内 artifacts_sha256 = 落盘时预录哈希（防静默改写）。