# Changelog

## v0.2 — Feature selection + RandomForest (improved RMSE)
- Pipeline: `SelectKBest(f_regression, k=8)` -> `RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=2)`
- Rationale: trees capture non-linearities; KBest removes noisy features
- Metrics (held-out split, seed=42):
  - **RMSE**: v0.1 = 53.85 → **v0.2 = 53.78** (↓ 0.07)
  - **High-risk flag** (top 25% y; threshold = `<value>` from training):
    - precision = `<p>`, recall = `<r>`, F1 = `<f1>`


## v0.1 – Baseline
- StandardScaler + LinearRegression
- Initial working API and Docker image