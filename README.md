# Credit Scoring Service

Сервис для расчёта PD (probability of default) по заявкам на кредитные карты.

## Быстрый старт

```bash
# установка
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# подготовка данных
python -m scoring.data.prepare data/raw/UCI_Credit_Card.csv data/processed

# обучение
python -m scoring.model.train --train data/processed/train.csv --test data/processed/test.csv --output models/pytorch

# экспорт в onnx
python -m scoring.model.onnx_export --model-dir models/pytorch --output models/onnx/model.onnx --validate

# квантизация
python -m scoring.model.quantize --input models/onnx/model.onnx

# запуск api
uvicorn scoring.api.main:app --reload
```

## Структура проекта

```
scoring/           код модели и api
  data/            схема данных, препроцессинг
  model/           сеть, обучение, onnx экспорт, бенчмарки
  api/             fastapi сервис
infra/             terraform (selectel)
deploy/            k8s манифесты (kustomize)
  base/            базовые ресурсы
  overlays/        конфиги staging/prod
  monitoring/      helm values для prometheus/loki
ops/               операционные скрипты
  drift/           мониторинг дрифта (evidently)
  airflow/         dag переобучения
```

## DVC Pipeline

```bash
dvc repro           # запуск пайплайна
dvc dag             # граф зависимостей
```

Стадии: prepare → train → export_onnx → quantize → benchmark

## Docker

```bash
docker compose up -d                    # backend + frontend
docker compose --profile dev up -d      # + mlflow
```

## Инфраструктура

Terraform для Selectel MKS (managed kubernetes).

```bash
cd infra
terraform init -backend-config=backend.hcl
terraform plan -var-file=envs/staging.tfvars
terraform apply -var-file=envs/staging.tfvars
```

Модули:
- `vpc` — проект, сеть, security groups
- `k8s` — managed кластер, node groups
- `storage` — s3 бакет для dvc/моделей
- `monitoring` — плейсхолдер (стек ставится через helm)

## Деплой

Через kustomize overlays:

```bash
kubectl apply -k deploy/overlays/staging
kubectl apply -k deploy/overlays/prod
```

## Мониторинг

Prometheus + Grafana + Loki через helm:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -f deploy/monitoring/prometheus-values.yaml

helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack -f deploy/monitoring/loki-values.yaml
```

Алерты в `deploy/monitoring/alerts.yaml`.
Runbook: `ops/runbook.md`.

## Мониторинг дрифта

Evidently запускается как CronJob ежедневно.
Ручной запуск:

```bash
python -m ops.drift.monitor --reference data/processed/train.csv --current data/processed/test.csv
```

## Переобучение

Airflow DAG в `ops/airflow/dags/retrain.py`:
- Запускается еженедельно или при обнаружении дрифта
- Валидирует новую модель против продакшена
- Автоматически промоутит если валидация пройдена

Локальный airflow:
```bash
cd ops/airflow && docker compose up -d
# UI на localhost:8080, admin/admin
```

## Модель

- Архитектура: 3-слойный MLP (64→32→1)
- Вход: 29 фичей (23 исходных + 6 сгенерированных)
- Выход: P(default)
- Формат: ONNX с INT8 квантизацией

Бенчмарк на M1 Mac:
| Формат | Batch=1 | Batch=64 |
|--------|---------|----------|
| PyTorch | ~0.5ms | ~2ms |
| ONNX FP32 | ~0.3ms | ~1.5ms |
| ONNX INT8 | ~0.2ms | ~1.2ms |

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, ...}'

# ответ: {"probability": 0.23, "prediction": 0}
```

Эндпоинты:
- `POST /predict` — получить вероятность дефолта
- `GET /health` — проверка здоровья
- `GET /metrics` — метрики prometheus

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
1. lint + тесты
2. валидация схемы данных
3. сборка docker образов
4. security scan (trivy)
5. деплой в staging (develop branch)
6. деплой в prod (main branch)
7. автоматический rollback при ошибке

## Заметки

- GPU node group отключена по умолчанию (дорого)
- В проде используется INT8 модель, FP32 как fallback
- Порог дрифта: 30% фичей задрифтовало → триггер переобучения
- Валидация модели: отклонить если AUC упал >2% vs продакшен
