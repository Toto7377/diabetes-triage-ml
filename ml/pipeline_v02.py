from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

def make_v02() -> Pipeline:
    return Pipeline(steps=[
        ("select", SelectKBest(score_func=f_regression, k=8)),
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
    ])

make = make_v02
