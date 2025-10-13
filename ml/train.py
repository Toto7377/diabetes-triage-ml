import json, time, sys
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.metrics import mean_squared_error
from ml.data import load_data, split
from ml.pipeline_v01 import make_pipeline as make_v01

SEED = 42
np.random.seed(SEED)

def train(version="v0.1"):
    X, y = load_data()
    Xtr, Xte, ytr, yte = split(X, y)
    pipe = make_v01()
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    rmse = mean_squared_error(yte, preds, squared=False)

    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True)
    model_path = artifacts / f"model_{version}.joblib"
    dump(pipe, model_path)

    metrics = {"version": version, "rmse": float(rmse), "timestamp": int(time.time()), "seed": SEED}
    with open(artifacts / f"metrics_{version}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return str(model_path), metrics

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v0.1"
    train(version)
