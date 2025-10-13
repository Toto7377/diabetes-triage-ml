# ml/train.py
import json, time, sys
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from ml.data import load_data, split
from ml.pipeline_v01 import make_v01
from ml.pipeline_v02 import make_v02

SEED = 42
np.random.seed(SEED)

def train(version="v0.1"):
    X, y = load_data()
    Xtr, Xte, ytr, yte = split(X, y)

    # choose pipeline
    if version == "v0.1":
        pipe = make_v01()
    elif version == "v0.2":
        pipe = make_v02()
    else:
        raise ValueError(f"Unknown version: {version}")

    # fit & predict
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    rmse = mean_squared_error(yte, preds, squared=False)

    # save model
    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True)
    model_path = artifacts / f"model_{version}.joblib"
    dump(pipe, model_path)

    # base metrics
    metrics = {
        "version": version,
        "rmse": float(rmse),
        "timestamp": time.time(),
        "seed": SEED,
    }

    # v0.2 high-risk calibration (top 25% threshold on y)
    if version == "v0.2":
        thresh = float(np.percentile(ytr, 75))  # calibrated on training labels
        y_true_flag = (yte >= thresh).astype(int)
        y_pred_flag = (preds >= thresh).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_flag, y_pred_flag, average="binary", zero_division=0
        )
        metrics.update({
            "risk_threshold": thresh,
            "precision_at_threshold": float(prec),
            "recall_at_threshold": float(rec),
            "f1_at_threshold": float(f1),
        })

    # write metrics
    with open(artifacts / f"metrics_{version}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return str(model_path), metrics


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v0.1"
    train(version)
