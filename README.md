[![CI](https://img.shields.io/github/actions/workflow/status/toto7377/diabetes-triage-ml/ci.yml?branch=main&logo=github&label=ci)](https://github.com/toto7377/diabetes-triage-ml/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/actions/workflow/status/toto7377/diabetes-triage-ml/release.yml?branch=v0.2&logo=github&label=release)](https://github.com/toto7377/diabetes-triage-ml/actions/workflows/release.yml)

# Diabetes Triage ML Service

Predicts short-term diabetes progression (higher = worse) from the scikit-learn Diabetes dataset.  
Built as a portable FastAPI service with CI/CD via GitHub Actions and Docker images on GHCR.

# Create a virtual environment
python -m venv .venv

# Powershell
.\.venv\Scripts\Activate.ps1

# Git Bash
source .venv/Scripts/activate # or: source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Train both versions (saves models + metrics to /artifacts)
python -m ml.train v0.1   
python -m ml.train v0.2

# Start the API
uvicorn app.app:app --host 0.0.0.0 --port 8000

# Run via Docker
docker run -d -p 8000:8000 ghcr.io/toto7377/diabetes-triage-ml:v0.2

# Docker compose (mapped to host port 8111)
docker compose up -d
# → http://127.0.0.1:8111/health
