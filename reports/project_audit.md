# Project Audit – StockSense

## 1. Folder & File Structure

```
StockSense (Desktop/SS - Copy)
├─ .git/                     # Git metadata (ignored in runtime)
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ data/
│   ├─ raw/                 # Original CSV sources (not version‑controlled)
│   │   ├─ buyers.csv
│   │   ├─ goods_receipts.csv
│   │   ├─ items.csv
│   │   ├─ po_lines.csv
│   │   ├─ purchase_orders.csv
│   │   ├─ requisitions.csv
│   │   ├─ retail_inventory_ml_apl.csv
│   │   ├─ retail_sales_ml_apl.csv
│   │   ├─ suppliers.csv
│   │   └─ supplier_invoices.csv
│   └─ processed/            # Cleaned / feature‑engineered files
│       ├─ demand_features.csv
│       ├─ inventory_intervals_clean.csv
│       ├─ procurement_invoices_final.csv
│       ├─ procurement_lines_enriched.csv
│       ├─ procurement_lines_final.csv
│       ├─ procurement_monthly.csv
│       ├─ procurement_requisitions_final.csv
│       ├─ procurement_suppliers_final.csv
│       ├─ reorder_recommendations.csv
│       ├─ sales_clean.csv
│       ├─ stockout_labels.csv.gz
│       └─ stockout_risk_latest.csv
├─ models/
│   ├─ demand_linear.joblib
│   ├─ demand_meta.json
│   ├─ demand_random_forest.joblib
│   ├─ demand_xgb_poisson.joblib
│   ├─ demand_xgb_squared.joblib
│   ├─ inventory_policy.json
│   ├─ stockout_logistic.joblib
│   ├─ stockout_meta.json
│   ├─ stockout_random_forest.joblib
│   ├─ stockout_tree.joblib
│   └─ stockout_xgb.joblib
├─ reports/
│   ├─ figures/               # PNG visualisations
│   │   ├─ 12_test_weekly_total.png
│   │   ├─ 14_pr_curve.png
│   │   ├─ A_weekly_by_division.png
│   │   ├─ A_weekly_total.png
│   │   ├─ B_weekday.png
│   │   ├─ C_top_categories.png
│   │   ├─ C_units_by_store.png
│   │   ├─ D_sales_type_share.png
│   │   └─ F_daily_stockout_rate.png
│   ├─ 01_inspection_output.txt
│   ├─ 02_quality_output.txt
│   ├─ 03_relationship_output.txt
│   ├─ 04_granularity_output.txt
│   ├─ 05_grain_decision_output.txt
│   ├─ 06_grain_part2_output.txt
│   ├─ 07_preprocessing_output.txt
│   ├─ 08_eda_output.txt
│   ├─ 09_inventory_diagnostics_output.txt
│   ├─ 10_stockout_label_output.txt
│   ├─ 11_demand_features_output.txt
│   ├─ 12_demand_models_output.txt
│   ├─ 12_model_results.csv
│   ├─ 12_test_predictions.csv
│   ├─ 12_tuning_trials.csv
│   ├─ 13_stockout_labels_output.txt
│   ├─ 14_model_results_v1.csv
│   ├─ 14_model_results_v2.csv
│   ├─ 14_stockout_model_output.txt
│   ├─ 14_test_predictions.csv
│   ├─ 14_tuning_trials.csv
│   ├─ 15_procurement_output.txt
│   ├─ 16_procurement_analysis_output.txt
│   ├─ 16_supplier_scorecard.csv
│   ├─ 17_inventory_optimization_output.txt
│   ├─ 17_rop_backtest.csv
│   ├─ 18_rop_calibration_grid.csv
│   ├─ 18_rop_calibration_output.txt
│   ├─ 18_rop_calibration_summary.csv
│   ├─ 19_reorder_final_output.txt
│   ├─ 20_action_list_output.txt
│   ├─ 20_sensitivity.csv
│   ├─ 21_ranking_comparison.csv
│   ├─ 21_validate_output.txt
│   ├─ 22_score_latest_output.txt
│   └─ 23_procurement_tables_output.txt
├─ scripts/
│   ├─ 01_load_and_inspect.py
│   ├─ 02_quality_checks.py
│   ├─ 03_relationship_checks.py
│   ├─ 04_granularity_check.py
│   ├─ 05_grain_decision.py
│   ├─ 06_grain_decision_part2.py
│   ├─ 07_preprocessing.py
│   ├─ 08_eda.py
│   ├─ 09_inventory_diagnostics.py
│   ├─ 10_stockout_label_diagnostics.py
│   ├─ 11_demand_features.py
│   ├─ 12_demand_models.py
│   ├─ 13_stockout_labels.py
│   ├─ 14_stockout_model.py
│   ├─ 15_procurement_inspection.py
│   ├─ 16_procurement_analysis.py
│   ├─ 17_inventory_optimization.py
│   ├─ 18_rop_calibration.py
│   ├─ 19_reorder_final.py
│   ├─ 20_action_list.py
│   ├─ 21_validate_sellout_probability.py
│   ├─ 22_score_latest.py
│   └─ 23_procurement_tables.py
├─ src/
│   └─ data/
│       ├─ __init__.py
│       └─ preprocess.py   # reusable preprocessing module
└─ dashboard/   # to be created – Streamlit app
```

## 2. Completed Modules (scripts) – Status ✅

| Module | Script(s) | Outcome |
|--------|-----------|---------|
| Data loading & inspection | `01_load_and_inspect.py` | Prints shape & dtypes for all raw CSVs (output captured in `01_inspection_output.txt`). |
| Data quality checks | `02_quality_checks.py` | Reports duplicates, empty columns, missing values, numeric sanity, low‑cardinality columns (`02_quality_output.txt`). |
| Relationship checks | `03_relationship_checks.py` | Validates foreign‑key relationships across procurement tables (`03_relationship_output.txt`). |
| Granularity check | `04_granularity_check.py` | Determines appropriate temporal grain (`04_granularity_output.txt`). |
| Granularity decision | `05_grain_decision.py`, `06_grain_decision_part2.py` | Chooses product‑store‑week grain (`05_grain_decision_output.txt`, `06_grain_part2_output.txt`). |
| **Pre‑processing** | `07_preprocessing.py` (now also `src/data/preprocess.py`) | Produces `sales_clean.csv` and `inventory_intervals_clean.csv` with full consistency checks (`07_preprocessing_output.txt`). |
| Exploratory Data Analysis | `08_eda.py` | Generates a suite of PNG plots saved under `reports/figures/` (`08_eda_output.txt`). |
| Inventory diagnostics | `09_inventory_diagnostics.py` | Analyses stock‑out risk, excess inventory (`09_inventory_diagnostics_output.txt`). |
| Stock‑out label generation | `10_stockout_label_diagnostics.py` | Creates `stockout_labels.csv.gz` and a recent‑risk snapshot (`10_stockout_label_output.txt`). |
| Demand feature engineering | `11_demand_features.py` | Writes `demand_features.csv` (`11_demand_features_output.txt`). |
| Demand modelling | `12_demand_models.py` | Trains Linear, RandomForest, XGBoost‑Poisson, XGBoost‑Squared models; stores artefacts and evaluation files (`12_demand_models_output.txt`). |
| Stock‑out label finalisation | `13_stockout_labels.py` | Finalises stock‑out label dataset (`13_stockout_labels_output.txt`). |
| Stock‑out modelling | `14_stockout_model.py` | Trains Logistic, RandomForest, DecisionTree, XGBoost classifiers; saves artefacts and metrics (`14_stockout_model_output.txt`). |
| Procurement inspection | `15_procurement_inspection.py` | Summarises raw procurement files (`15_procurement_output.txt`). |
| Procurement analysis | `16_procurement_analysis.py` | Generates spend, lead‑time, invoice & payment analytics (`16_procurement_analysis_output.txt`, `16_supplier_scorecard.csv`). |
| Inventory optimisation | `17_inventory_optimization.py` | Poisson & Gamma‑Poisson ROP calculations, back‑testing (`17_inventory_optimization_output.txt`, `17_rop_backtest.csv`). |
| ROP calibration | `18_rop_calibration.py` | Sensitivity analysis across lead‑time / service‑level scenarios (`18_rop_calibration_grid.csv`, `18_rop_calibration_output.txt`). |
| Reorder finalisation | `19_reorder_final.py` | Produces final recommendation table `reorder_recommendations.csv` (`19_reorder_final_output.txt`). |
| Action‑list generation | `20_action_list.py` | Summarises actionable insights (`20_action_list_output.txt`, `20_sensitivity.csv`). |
| Model validation | `21_validate_sellout_probability.py` | Validates demand forecast coverage (`21_validate_output.txt`). |
| Scoring latest data | `22_score_latest.py` | Applies latest models to current period (`22_score_latest_output.txt`). |
| Procurement tables export | `23_procurement_tables.py` | Exports tidy procurement tables (`23_procurement_tables_output.txt`). |

All scripts have executed successfully; no runtime errors were logged.

## 3. Processed Datasets – Key Characteristics

| Processed file | Source raw file(s) | Rows | Primary purpose |
|----------------|-------------------|------|-----------------|
| `sales_clean.csv` | `retail_sales_ml_apl.csv` | 125,749 | Historical sales used for demand modelling and EDA. |
| `inventory_intervals_clean.csv` | `retail_inventory_ml_apl.csv` | 284,755 | Inventory snapshots (open‑ended dates handled). |
| `demand_features.csv` | Derived from `sales_clean.csv` & calendar | 5,862,700 | Lag/rolling features for forecasting. |
| `stockout_labels.csv.gz` | Generated from sales + inventory alignment | 4,403,129 | Binary label for stock‑out events (used in classification). |
| `stockout_risk_latest.csv` | Latest inventory + demand forecasts | 2,250,284 | Predicted probability of stock‑out for the current week. |
| Procurement tables (`*_final.csv`) | Various raw procurement CSVs | ≤ 25 k rows each | Cleaned procurement master tables for spend & lead‑time analysis. |

## 4. Model Artefacts

| Model type | File(s) | Notes |
|-----------|---------|-------|
| Demand – Linear Regression | `models/demand_linear.joblib` | Baseline regression. |
| Demand – Random Forest | `models/demand_random_forest.joblib` | Non‑linear regression. |
| Demand – XGBoost Poisson | `models/demand_xgb_poisson.joblib` | Count‑data forecast. |
| Demand – XGBoost Squared | `models/demand_xgb_squared.joblib` | Alternative count model. |
| Stock‑out – Logistic Regression | `models/stockout_logistic.joblib` | Baseline classifier. |
| Stock‑out – Random Forest | `models/stockout_random_forest.joblib` | Non‑linear classifier. |
| Stock‑out – Decision Tree | `models/stockout_tree.joblib` | Simple interpretable model. |
| Stock‑out – XGBoost | `models/stockout_xgb.joblib` | High‑performance classifier. |
| Meta files (`*_meta.json`) | Store feature schema, train/val split dates, evaluation metrics. |
| `inventory_policy.json` | Parameters used for ROP calculation (lead‑times, service‑levels). |

## 5. Identified Issues & Gaps

1. **Dashboard directory missing** – no Streamlit app yet (Phase 8). 
2. **Tests directory empty** – no automated test suite (Phase 11). 
3. **Hard‑coded Windows‑style paths** in a few scripts (e.g., `scripts/01_load_and_inspect.py`). Refactor to use `Path`‑based relative paths. 
4. **`requirements.txt` empty** – project depends on many libraries; needs population. 
5. **Model‑performance summary** not consolidated in a single report (Phase 10). 
6. **Documentation** – `README.md` is present but does not reflect the full pipeline or how to run the dashboard. 
7. **Potential temporal leakage** – feature creation in `11_demand_features.py` appears correct but should be double‑checked; we will add explicit comments and unit tests. 
8. **Open‑ended inventory dates** are already flagged (`end_is_open`) and replaced with the latest sales date, but downstream calculations must treat them as “current inventory”. 
9. **No FastAPI backend** – optional (Phase 9). We will focus first on a self‑contained Streamlit app. 
10. **`requirements.txt`** currently empty; we will populate it with exact package versions used. 
11. **`.gitignore`** already excludes `data/raw` and the virtual environment, which is correct. 
12. **Duplicate code** – the original preprocessing logic lives both in `scripts/07_preprocessing.py` and the new `src/data/preprocess.py`. We will keep the script for backward compatibility but have the Streamlit app and future pipelines import the reusable module.

## 6. Recommended Immediate Actions

| Phase | Action |
|------|--------|
| **3 – Preprocessing** | Verify that `src/data/preprocess.py` reproduces the exact same clean CSVs (already done). Add a thin wrapper script `scripts/07_preprocessing.py` that simply calls the module (so existing notebooks keep working). |
| **4 – EDA** | Refactor `08_eda.py` to import data via `src.data.preprocess.OUT_DIR`. Ensure all plots are saved under `reports/figures/`. |
| **5 – Demand Forecasting** | Review `12_demand_models.py` for any leakage, add explicit train/val split dates to `models/demand_meta.json`. |
| **6 – Stock‑out Prediction** | Confirm class‑imbalance handling in `14_stockout_model.py`. |
| **7 – Inventory Optimisation** | Ensure back‑testing results are stored and do not claim 95 % coverage unless actually achieved (the existing output already respects this). |
| **8 – Dashboard** | Create `dashboard/` package with `app.py` and pages that load the artefacts generated by the previous phases. |
| **9 – FastAPI (optional)** | Defer until the core Streamlit app works. |
| **10 – Pipeline script** | Add `run_pipeline.py` that orchestrates all steps, skipping stages when artefacts already exist. |
| **11 – Tests** | Introduce `tests/` with pytest modules covering data loading, preprocessing, feature generation, and model inference. |
| **12 – Documentation** | Expand `README.md` with installation, execution, and dashboard usage instructions. |

## 7. Next Concrete Step

We will **refactor the preprocessing script** to use the new reusable module and verify that the clean datasets are reproducibly generated. This prepares the foundation for the remaining phases.

---

*(The audit report is saved as `reports/project_audit.md`.)
