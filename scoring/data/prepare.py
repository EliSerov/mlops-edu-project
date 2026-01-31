"""
Подготовка данных: загрузка CSV, валидация, генерация фичей, разбиение.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import validate


def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    """Загрузка сырого UCI credit card CSV."""
    df = pd.read_csv(path)
    # в некоторых версиях другое название колонки
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default.payment.next.month"})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — ничего сложного, просто агрегаты."""
    df = df.copy()

    # агрегаты истории платежей
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["pay_mean"] = df[pay_cols].mean(axis=1)
    df["pay_max"] = df[pay_cols].max(axis=1)
    df["pay_delayed_count"] = (df[pay_cols] > 0).sum(axis=1)

    # соотношение платежей к счетам (прокси утилизации)
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    total_bill = df[bill_cols].sum(axis=1).clip(lower=1)
    total_paid = df[amt_cols].sum(axis=1)
    df["payment_ratio"] = total_paid / total_bill

    # утилизация
    df["utilization"] = df["BILL_AMT1"] / df["LIMIT_BAL"].clip(lower=1)

    # возрастные бакеты (простые)
    df["age_bin"] = pd.cut(df["AGE"], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
    df["age_bin"] = df["age_bin"].astype(int)

    return df


TARGET = "default.payment.next.month"
DROP_COLS = ["ID"]

FEATURE_COLS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "pay_mean",
    "pay_max",
    "pay_delayed_count",
    "payment_ratio",
    "utilization",
    "age_bin",
]


def prepare_dataset(
    raw_path: Union[str, Path], output_dir: Union[str, Path], test_size: float = 0.2
):
    """Полный пайплайн подготовки: загрузка -> валидация -> фичи -> разбиение -> сохранение."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    df = validate(df)
    df = add_features(df)

    # удаляем ID если есть
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Default rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    return train_df, test_df


if __name__ == "__main__":
    import sys

    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/UCI_Credit_Card.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    prepare_dataset(raw_path, output_dir)
