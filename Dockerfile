# --------- base image ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

WORKDIR /app

# install deps
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y build-essential && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# copy code + artifacts
COPY app/ app/
COPY ml/  ml/
COPY artifacts/ artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
 CMD python -c "import sys,urllib.request as r; resp=r.urlopen('http://127.0.0.1:8000/health',timeout=2); sys.exit(0 if resp.status==200 else 1)" || exit 1

CMD ["python","-m","uvicorn","app.app:app","--host","0.0.0.0","--port","8000"]
