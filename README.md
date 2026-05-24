# GAEZ Suitability ML

This repository contains the code used for my MSc thesis on approximating GAEZ v4 rainfed crop suitability scores with machine-learning models. The project focuses on wheat and maize, using climate variables from NASA POWER and soil variables from SoilGrids to predict GAEZ suitability scores under spatial block cross-validation.

The main goal is not to replace GAEZ, but to test whether a smaller set of openly available predictors can approximate its suitability outputs in a reproducible way. The workflow includes data preparation, Random Forest and XGBoost modelling, hyperparameter search, residual analysis, crop-level subgroup analysis, statistical comparison, and XGBoost feature contribution analysis.

Raw datasets are not included in this repository because they are large and should be accessed through their original sources: the GAEZ v4 data portal, NASA POWER, and SoilGrids. The scripts assume that the processed model-ready files are stored locally in the expected `data/processed/` structure.

## Main scripts

The main workflow is implemented through the scripts in the `scripts/` folder:

- `prepare_model_data.py` prepares the final model-ready dataset.
- `run_mean_baseline.py` runs the naive mean predictor baseline.
- `train_baseline_rf.py` and `train_xgboost.py` train the main model families.
- `run_rf_random_search.py` and `run_xgb_random_search.py` run the constrained random searches.
- `export_rf_cv_predictions_rsbest.py` and `export_xgb_cv_predictions_rs04.py` export out-of-sample spatial CV predictions.
- `plot_rf_cv_diagnostics_rsbest.py`, `plot_xgb_cv_diagnostics_rs04.py`, and `compare_rf_xgb_residuals.py` create residual diagnostics.
- `analyze_crop_subgroups_rs04.py` computes wheat and maize subgroup metrics.
- `analyze_shap_xgb_rs04.py` and `analyze_rf_feature_importance_rs03.py` computes XGBoost and Random Forest feature contributions for final models.
- `test_rf_xgb_significance.py` runs the fold-level RF versus XGBoost comparison.

## Environment

The analysis was run in Python 3.11. Main packages include pandas, NumPy, SciPy, scikit-learn, XGBoost, GeoPandas, Rasterio, matplotlib, and Plotly. Package versions are listed in `requirements.txt`.

## Reproducibility note

Random seeds are fixed where sampling, model training, or random search is used. Spatial validation is based on 0.5° and 2.0° block sizes, with the 2.0° setting used as the stricter test of geographic generalisation. Large data files, model files, and generated outputs are not tracked in GitHub.
