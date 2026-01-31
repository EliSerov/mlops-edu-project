"""
Мониторинг дрифта данных с использованием Evidently.
Запускается как CronJob или триггерится из Airflow.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


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
    """Вычисление метрик дрифта данных (Evidently 0.7+ API)."""

    # фильтруем только числовые колонки которые есть в обоих датасетах
    common_cols = [c for c in FEATURE_COLS if c in reference.columns and c in current.columns]
    ref_filtered = reference[common_cols].copy()
    cur_filtered = current[common_cols].copy()

    # создаём Report с preset
    report = Report([DataDriftPreset()])
    my_report = report.run(ref_filtered, cur_filtered)

    # извлекаем результат
    result = my_report.dict()

    # парсим результаты (структура зависит от версии)
    drift_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "reference_size": len(reference),
        "current_size": len(current),
        "dataset_drift": False,
        "drift_share": 0.0,
        "drifted_columns": [],
    }

    # пытаемся извлечь метрики из результата
    try:
        for metric in result.get("metrics", []):
            metric_result = metric.get("result", {})
            if "drift_share" in metric_result:
                drift_summary["drift_share"] = metric_result["drift_share"]
                drift_summary["dataset_drift"] = metric_result.get("dataset_drift", False)
            if "drift_by_columns" in metric_result:
                for col, info in metric_result["drift_by_columns"].items():
                    if info.get("drift_detected"):
                        drift_summary["drifted_columns"].append({
                            "column": col,
                            "drift_score": info.get("drift_score"),
                        })
    except Exception:
        pass  # структура может отличаться, пропускаем

    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # сохраняем HTML отчёт
        html_path = output_dir / f"drift_report_{timestamp}.html"
        my_report.save_html(str(html_path))

        # сохраняем summary JSON
        json_path = output_dir / f"drift_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(drift_summary, f, indent=2)

        print(f"Отчёт сохранён: {html_path}")

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

    summary = compute_drift_report(reference, current, output_dir)

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
