from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_v01() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

make = make_v01
