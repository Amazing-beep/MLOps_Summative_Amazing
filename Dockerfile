FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RUNTIME_DIR=/app/runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN mkdir -p /app/runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health')"
# One worker keeps the in-memory model cache consistent; threads handle concurrency.
CMD ["sh","-c","gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 180"]
