# syntax=docker/dockerfile:1
# Backend сервис — multi-stage сборка

FROM python:3.11-slim as builder

WORKDIR /build

# системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim as runtime

WORKDIR /app

# установка wheel из builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# dvc для версионирования данных (если нужно в runtime)
RUN pip install --no-cache-dir dvc[s3]

# копируем код приложения
COPY scoring/ ./scoring/
COPY models/ ./models/

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models/onnx
ENV SCALER_PATH=/app/models/pytorch/scaler.pkl

EXPOSE 8000

CMD ["uvicorn", "scoring.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
