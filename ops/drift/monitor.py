"""
Мониторинг дрифта данных с использованием Evidently.
Запускается как CronJob или триггерится из Airflow.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric, DataDriftTable, DatasetDriftMetric
from evidently.report import Report


TARGET = "default.payment.next.month"

FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def load_reference_data(path: str) -> pd.DataFrame:
    """Загрузка тренировочных данных как референса."""
    return pd.read_csv(path)


def load_production_data(path: str = None, api_logs: str = None) -> pd.DataFrame:
    """
    Загрузка продакшен данных из логов или CSV.
    В реальной настройке это запрос к логам предсказаний из БД/S3.
    """
    if path:
        return pd.read_csv(path)
    
    # заглушка: симуляция продакшен данных из тестового сета с дрифтом
    # в реальности это приходит из логов запросов предсказаний
    return None


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str = None,
) -> dict:
    """Вычисление метрик дрифта данных."""
    
    column_mapping = ColumnMapping(
        target=TARGET if TARGET in reference.columns else None,
        numerical_features=[c for c in FEATURE_COLS if c in reference.columns],
        categorical_features=["SEX", "EDUCATION", "MARRIAGE"],
    )
    
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    
    # извлечение summary
    result = report.as_dict()
    
    drift_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "reference_size": len(reference),
        "current_size": len(current),
        "dataset_drift": result["metrics"][0]["result"]["dataset_drift"],
        "drift_share": result["metrics"][0]["result"]["drift_share"],
        "drifted_columns": [],
    }
    
    # получаем дрифт по колонкам
    drift_table = result["metrics"][1]["result"]["drift_by_columns"]
    for col, info in drift_table.items():
        if info.get("drift_detected"):
            drift_summary["drifted_columns"].append({
                "column": col,
                "drift_score": info.get("drift_score"),
                "stattest": info.get("stattest_name"),
            })
    
    if output_path:
        # сохраняем HTML отчёт
        report.save_html(output_path)
        
        # сохраняем summary JSON
        json_path = Path(output_path).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(drift_summary, f, indent=2)
    
    return drift_summary


def check_model_performance_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    predictions_col: str = "prediction",
) -> dict:
    """
    Проверка деградации производительности модели.
    Требует ground truth лейблы в текущих данных.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    
    if TARGET not in current.columns:
        return {"status": "no_labels", "message": "Нет ground truth в текущих данных"}
    
    ref_auc = roc_auc_score(reference[TARGET], reference.get(predictions_col, 0.5))
    cur_auc = roc_auc_score(current[TARGET], current.get(predictions_col, 0.5))
    
    return {
        "reference_auc": ref_auc,
        "current_auc": cur_auc,
        "auc_drop": ref_auc - cur_auc,
        "alert": (ref_auc - cur_auc) > 0.05,  # порог 5% падения
    }


def run_drift_check(
    reference_path: str,
    current_path: str,
    output_dir: str = "reports/drift",
):
    """Полный пайплайн проверки дрифта."""
    reference = load_reference_data(reference_path)
    current = pd.read_csv(current_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.html"
    
    summary = compute_drift_report(reference, current, str(report_path))
    
    print(f"Обнаружен дрифт датасета: {summary['dataset_drift']}")
    print(f"Доля дрифта: {summary['drift_share']:.1%}")
    
    if summary["drifted_columns"]:
        print(f"Задрифтовавшие колонки: {[c['column'] for c in summary['drifted_columns']]}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/processed/test.csv")
    parser.add_argument("--output", default="reports/drift")
    args = parser.parse_args()
    
    run_drift_check(args.reference, args.current, args.output)
