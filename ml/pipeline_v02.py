# ml/pipeline_v02.py
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor


def make_v02() -> Pipeline:
    """
    v0.2 pipeline:
      - SelectKBest with f_regression (keep the most predictive features)
      - RandomForestRegressor (tree-based, non-linear)

    Deterministic via fixed hyperparameters & seeds (seed provided by trainer).
    """
    return Pipeline(
        steps=[
            ("select", SelectKBest(score_func=f_regression, k=8)),
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=2,
                random_state=42,   # model-level seed; train() also sets np seed
                n_jobs=-1
            )),
        ]
    )
