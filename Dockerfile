FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/candidate_logistic_c1.joblib \
    MODEL_VERSION=logistic_c1-candidate

WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY models/candidate_logistic_c1.joblib ./models/candidate_logistic_c1.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
