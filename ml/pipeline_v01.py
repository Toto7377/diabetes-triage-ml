from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
