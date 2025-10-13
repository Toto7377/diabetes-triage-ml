# ---------- base image ----------
FROM python:3.11-slim

# Faster Python, cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_VERSION=v0.1

WORKDIR /app

# ---------- install deps ----------
COPY requirements.txt .
# scikit-learn wheels usually work on slim, but install build tools just in case;
# then remove them to keep the final image small.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ---------- copy code ----------
COPY app/ app/
COPY ml/ ml/

# (Optional) include tests/ only if you want to run them inside the image
# COPY tests/ tests/

# ---------- train & bake artifacts ----------
# This will create artifacts/model_v0.1.joblib and artifacts/metrics_v0.1.json
RUN python -m ml.train ${APP_VERSION}

# ---------- runtime config ----------
EXPOSE 8000

# Container healthcheck (hits /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python - <<'PY'\nimport sys,urllib.request as r\n\
try:\n resp=r.urlopen('http://127.0.0.1:8000/health',timeout=2)\n sys.exit(0 if resp.status==200 else 1)\n\
except Exception:\n sys.exit(1)\nPY

# Start the API
CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
