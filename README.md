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

#start the API
uvicorn app.app:app --host 0.0.0.0 --port 8000