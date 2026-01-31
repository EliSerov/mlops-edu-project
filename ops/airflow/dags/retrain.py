"""
Airflow DAG для пайплайна переобучения модели.
Триггеры: по расписанию (еженедельно) или при обнаружении дрифта.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ml-alerts@company.io"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift_trigger(**context):
    """Проверка нужно ли запускать переобучение на основе метрик дрифта."""
    import json
    from pathlib import Path
    
    # получаем последний отчёт о дрифте
    reports_dir = Path("/opt/airflow/data/drift")
    reports = sorted(reports_dir.glob("drift_report_*.json"))
    
    if not reports:
        return "skip_retrain"
    
    with open(reports[-1]) as f:
        drift = json.load(f)
    
    # переобучаем если дрифт обнаружен или это плановый запуск
    trigger_reason = context["dag_run"].conf.get("trigger", "scheduled")
    
    if trigger_reason == "scheduled":
        return "prepare_data"
    
    if drift.get("dataset_drift") or drift.get("drift_share", 0) > 0.3:
        return "prepare_data"
    
    return "skip_retrain"


def validate_new_model(**context):
    """Сравнение новой модели с продакшен моделью."""
    import json
    
    # получаем метрики из шага обучения
    ti = context["ti"]
    train_metrics = ti.xcom_pull(task_ids="train_model", key="metrics")
    
    if not train_metrics:
        # fallback: читаем из файла
        with open("/opt/airflow/data/models/new/metrics.json") as f:
            train_metrics = json.load(f)
    
    # получаем продакшен метрики
    prod_metrics_path = "/opt/airflow/data/models/production/metrics.json"
    try:
        with open(prod_metrics_path) as f:
            prod_metrics = json.load(f)
    except FileNotFoundError:
        # первый запуск, нет продакшен модели
        return True
    
    # валидация: новая модель не должна быть хуже чем на X%
    auc_threshold = 0.02  # максимум 2% падения
    
    new_auc = train_metrics.get("roc_auc", 0)
    prod_auc = prod_metrics.get("roc_auc", 0)
    
    if new_auc < prod_auc - auc_threshold:
        raise ValueError(
            f"AUC новой модели ({new_auc:.4f}) значительно хуже чем "
            f"продакшен ({prod_auc:.4f})"
        )
    
    return True


with DAG(
    "credit_scoring_retrain",
    default_args=default_args,
    description="Переобучение PD модели по дрифту или расписанию",
    schedule_interval="0 2 * * 0",  # еженедельно воскресенье 2am
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "credit-scoring"],
) as dag:
    
    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift_trigger,
    )
    
    skip = BashOperator(
        task_id="skip_retrain",
        bash_command="echo 'Дрифт не обнаружен, пропускаем переобучение'",
    )
    
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="""
            cd /opt/airflow/repo && \
            python -m scoring.data.prepare \
                data/raw/UCI_Credit_Card.csv \
                data/processed
        """,
    )
    
    train_model = KubernetesPodOperator(
        task_id="train_model",
        namespace="airflow",
        image="credit-scoring/backend:latest",
        cmds=["python", "-m", "scoring.model.train"],
        arguments=[
            "--train", "/data/processed/train.csv",
            "--test", "/data/processed/test.csv",
            "--output", "/data/models/new",
            "--epochs", "50",
        ],
        volumes=[{
            "name": "data",
            "persistentVolumeClaim": {"claimName": "ml-data"},
        }],
        volume_mounts=[{
            "name": "data",
            "mountPath": "/data",
        }],
        get_logs=True,
        is_delete_operator_pod=True,
    )
    
    export_onnx = KubernetesPodOperator(
        task_id="export_onnx",
        namespace="airflow",
        image="credit-scoring/backend:latest",
        cmds=["python", "-m", "scoring.model.onnx_export"],
        arguments=[
            "--model-dir", "/data/models/new",
            "--output", "/data/models/new/model.onnx",
            "--validate",
        ],
        volumes=[{
            "name": "data",
            "persistentVolumeClaim": {"claimName": "ml-data"},
        }],
        volume_mounts=[{
            "name": "data",
            "mountPath": "/data",
        }],
        get_logs=True,
        is_delete_operator_pod=True,
    )
    
    quantize = KubernetesPodOperator(
        task_id="quantize",
        namespace="airflow",
        image="credit-scoring/backend:latest",
        cmds=["python", "-m", "scoring.model.quantize"],
        arguments=[
            "--input", "/data/models/new/model.onnx",
            "--output", "/data/models/new/model_int8.onnx",
            "--compare",
        ],
        volumes=[{
            "name": "data",
            "persistentVolumeClaim": {"claimName": "ml-data"},
        }],
        volume_mounts=[{
            "name": "data",
            "mountPath": "/data",
        }],
        get_logs=True,
        is_delete_operator_pod=True,
    )
    
    validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_new_model,
    )
    
    promote = BashOperator(
        task_id="promote_model",
        bash_command="""
            # бэкап текущей продакшен модели
            cp -r /opt/airflow/data/models/production \
                  /opt/airflow/data/models/backup_$(date +%Y%m%d) 2>/dev/null || true
            
            # промоут новой модели
            rm -rf /opt/airflow/data/models/production
            cp -r /opt/airflow/data/models/new /opt/airflow/data/models/production
            
            echo "Модель промоутнута в продакшен"
        """,
    )
    
    deploy = BashOperator(
        task_id="trigger_deploy",
        bash_command="""
            # триггер k8s deployment rollout
            kubectl -n credit-scoring rollout restart deployment/backend
            kubectl -n credit-scoring rollout status deployment/backend --timeout=180s
        """,
    )
    
    notify = BashOperator(
        task_id="notify",
        bash_command="""
            echo "Переобучение завершено. Новая модель задеплоена."
            # в реальной настройке: отправка уведомления в slack/email
        """,
        trigger_rule="all_done",
    )
    
    # Граф DAG
    check_drift >> [skip, prepare_data]
    prepare_data >> train_model >> export_onnx >> quantize >> validate >> promote >> deploy >> notify
    skip >> notify
