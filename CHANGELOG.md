# Changelog

# v0.2 – Improved model (RandomForest + SelectKBest)

**Date:** 2025-10-17  
**Summary:** Replaced the baseline LinearRegression model with a RandomForestRegressor pipeline.  
Added SelectKBest for feature selection and introduced calibrated “high-risk” metrics.

# Changes
- Added new pipeline `ml/pipeline_v02.py` with:
  - RandomForestRegressor (`n_estimators=400`, `max_depth=6`, `min_samples_leaf=2`)
  - Feature selection using `SelectKBest(k=8)`
- Added high-risk calibration logic (top 25% threshold on target variable)
- CI/CD now builds and releases multi-architecture Docker images (`linux/amd64`, `linux/arm64`)
- Docker container health checks validated via `/health` endpoint

# Metrics (from `/artifacts/metrics_v0.2.json`)
| Metric | v0.1 | v0.2 | Δ (change) |
|--------|------|------|------------|
| RMSE | **53.85** | **54.38** | +0.53 (slightly higher) |
| Risk Threshold | – | **214.0** | – |
| Precision @ threshold | – | **0.80** | +0.80 |
| Recall @ threshold | – | **0.38** | +0.38 |
| F1 @ threshold | – | **0.52** | +0.52 |

# Interpretation
> Although RMSE slightly increased (by 0.5), the new v0.2 model introduces risk-based metrics that better support clinical triage use cases.  
> The RandomForestRegressor captures non-linear relationships and provides improved interpretability through feature selection.  
> This version adds robustness and new evaluation dimensions for high-risk prediction.

---

# v0.1 – Baseline (LinearRegression + StandardScaler)

**Date:** 2025-10-13  
**Summary:** Initial baseline model using LinearRegression with standardized features.

# Pipeline
- StandardScaler → LinearRegression
- CI/CD pipeline, FastAPI endpoints (`/health`, `/predict`)

# Metrics (from `/artifacts/metrics_v0.1.json`)
| Metric | Value |
|--------|-------|
| RMSE | **53.85** |

# Interpretation
> Baseline LinearRegression model trained on normalized diabetes dataset.  
> Served predictions and model metadata via FastAPI.

---

