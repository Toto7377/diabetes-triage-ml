# ml/pipeline_v02.py
from __future__ import annotations

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_v02() -> Pipeline:
    """
    v0.2 pipeline: StandardScaler + RidgeCV
    Uses a small grid of alphas and returns a scikit-learn Pipeline.
    """
    alphas = (0.1, 0.3, 1.0, 3.0, 10.0)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=alphas)),
        ]
    )
