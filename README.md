# Diabetes Triage ML Service

Predicts short-term diabetes progression (higher = worse) from the scikit-learn Diabetes dataset.  
Built as a portable FastAPI service with CI/CD via GitHub Actions and Docker images on GHCR.

## Quick start (local)

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt
python -m ml.train v0.1   # trains and writes artifacts/metrics_v0.1.json, artifacts/model_v0.1.joblib
uvicorn app.app:app --host 127.0.0.1 --port 8000