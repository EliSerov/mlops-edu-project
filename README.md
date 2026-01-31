# Credit Scoring MLOps

Промышленный деплой модели кредитного скоринга с полным MLOps циклом.

## Стек технологий

| Этап | Технологии |
|------|------------|
| **1. Модель** | PyTorch, ONNX, INT8 quantization |
| **2. Инфраструктура** | Terraform, Selectel MKS (Kubernetes) |
| **3. Контейнеризация** | Docker (multi-stage), Kustomize |
| **4. CI/CD** | GitHub Actions, Trivy |
| **5. Мониторинг** | Prometheus, Grafana, Loki |
| **6. Drift** | Evidently AI |
| **7. Retraining** | Apache Airflow |

---

## Быстрый старт

```bash
# клонирование
git clone https://github.com/EliSerov/mlops-edu-project.git
cd mlops-edu-project

# виртуальное окружение
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Проверка работы

### 1. Обучение модели

```bash
python -m scoring.model.train \
  --train data/processed/train.csv \
  --test data/processed/test.csv \
  --output models/pytorch
```

Результат:
```
Epoch 10/50, Loss: 0.4521
Epoch 20/50, Loss: 0.4312
...
Metrics: {'roc_auc': 0.7706, 'precision': 0.67, 'recall': 0.35, 'f1': 0.46}
```

### 2. ONNX экспорт и валидация

```bash
python -m scoring.model.onnx_export \
  --model-dir models/pytorch \
  --output models/onnx/model.onnx \
  --validate
```

Результат:
```
Экспортировано в models/onnx/model.onnx
Валидация: max_diff=7.45e-08, passed=True
```

### 3. INT8 квантизация

```bash
python -m scoring.model.quantize \
  --input models/onnx/model.onnx \
  --output models/onnx/model_int8.onnx \
  --compare
```

Результат:
```
Оригинал: 16.6 KB
Квантизованная: 8.3 KB
Сжатие: 2.01x
ROC-AUC оригинал: 0.7706
ROC-AUC квантизованная: 0.7706
Разница: 0.000010
```

### 4. Бенчмарк производительности

```bash
python -m scoring.model.benchmark \
  --pytorch-dir models/pytorch \
  --onnx models/onnx/model.onnx \
  --onnx-quant models/onnx/model_int8.onnx \
  --test-data data/processed/test.csv \
  --scaler models/pytorch/scaler.pkl
```

Результат (Apple Silicon M-series):
```
Batch size: 1
  PyTorch CPU: 0.012ms, 86263 rps
  ONNX CPU:    0.006ms, 163901 rps
  ONNX INT8:   0.004ms, 234396 rps
  ONNX CoreML: 0.015ms, 64714 rps
  PyTorch MPS: 0.266ms, 3753 rps
```

Подробные результаты: [reports/README.md](reports/README.md)

### 5. Запуск API

```bash
uvicorn scoring.api.main:app --host 0.0.0.0 --port 8000
```

#### Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

#### Предсказание (хороший клиент)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 10000, "BILL_AMT2": 9000, "BILL_AMT3": 8000,
    "BILL_AMT4": 7000, "BILL_AMT5": 6000, "BILL_AMT6": 5000,
    "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
    "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000
  }'
```

```json
{"probability": 0.197, "prediction": 0, "model_version": "v1"}
```

#### Предсказание (рисковый клиент)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 10000, "SEX": 2, "EDUCATION": 3, "MARRIAGE": 2, "AGE": 25,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 3, "PAY_4": 3, "PAY_5": 4, "PAY_6": 4,
    "BILL_AMT1": 9000, "BILL_AMT2": 9500, "BILL_AMT3": 10000,
    "BILL_AMT4": 10500, "BILL_AMT5": 11000, "BILL_AMT6": 11500,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 100,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
  }'
```

```json
{"probability": 0.611, "prediction": 1, "model_version": "v1"}
```

#### Prometheus метрики

```bash
curl http://localhost:8000/metrics | grep prediction
```

```
prediction_requests_total{status="success"} 2.0
prediction_latency_seconds_bucket{le="0.01"} 2.0
```

### 6. Unit тесты

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health PASSED
tests/test_data.py::test_schema_valid PASSED
tests/test_data.py::test_schema_invalid_age PASSED
tests/test_data.py::test_add_features PASSED
tests/test_data.py::test_feature_cols_exist PASSED
=================== 5 passed, 1 skipped ===================
```

### 7. Drift мониторинг

```bash
python -m ops.drift.monitor \
  --reference data/processed/train.csv \
  --current data/processed/test.csv \
  --output reports/drift
```

```
Отчёт сохранён: reports/drift/drift_report_20260131_033602.html
Обнаружен дрифт датасета: False
Доля дрифта: 0.0%
```

HTML отчёт Evidently: `reports/drift/drift_report_*.html`

### 8. Docker

```bash
# backend
docker build -t credit-scoring/backend .

# frontend
docker build -t credit-scoring/frontend ./frontend

# или через compose
docker-compose up -d
```

---

## Структура проекта

```
├── scoring/                 # ML код
│   ├── api/                 # FastAPI сервис
│   ├── data/                # Подготовка данных, схема
│   └── model/               # Сеть, обучение, ONNX, бенчмарки
├── infra/                   # Terraform (Selectel MKS)
│   └── modules/             # VPC, K8s, Storage, Monitoring
├── deploy/                  # Kubernetes манифесты
│   ├── base/                # Kustomize base
│   ├── overlays/            # staging, prod
│   └── monitoring/          # Prometheus, Loki, alerts
├── ops/                     # Операции
│   ├── airflow/             # DAG переобучения
│   └── drift/               # Evidently мониторинг
├── tests/                   # Unit тесты, нагрузочные
├── reports/                 # Бенчмарки, drift отчёты
└── .github/workflows/       # CI/CD
```

---

## CI/CD Pipeline

GitHub Actions: `.github/workflows/ci.yml`

```
lint-test → validate-data → build → security-scan → deploy
                                                      ↓
                                              (rollback on fail)
```

- **lint-test**: ruff, black, pytest
- **build**: Docker multi-stage, push to ghcr.io
- **security-scan**: Trivy (CRITICAL, HIGH)
- **deploy**: Kustomize → Kubernetes

---

## Выводы по бенчмаркам

Для модели размером 16KB:

| Backend | Latency | Рекомендация |
|---------|---------|--------------|
| **ONNX INT8** | 0.004ms | Latency-critical API |
| ONNX FP32 | 0.006ms | Batch processing |
| PyTorch CPU | 0.012ms | Разработка |
| CoreML (NPU) | 0.015ms | Слишком маленькая модель |
| MPS (GPU) | 0.266ms | Оверхед передачи данных |

GPU/NPU не даёт выигрыша для маленьких моделей из-за оверхеда.
