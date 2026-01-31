import pandas as pd
import pytest

from scoring.data.prepare import FEATURE_COLS, add_features
from scoring.data.schema import validate


@pytest.fixture
def sample_data():
    """Минимальный валидный сэмпл данных."""
    return pd.DataFrame(
        {
            "ID": [1],
            "LIMIT_BAL": [50000.0],
            "SEX": [1],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "AGE": [30],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [-1],
            "PAY_4": [-1],
            "PAY_5": [-2],
            "PAY_6": [-2],
            "BILL_AMT1": [10000.0],
            "BILL_AMT2": [9000.0],
            "BILL_AMT3": [8000.0],
            "BILL_AMT4": [7000.0],
            "BILL_AMT5": [6000.0],
            "BILL_AMT6": [5000.0],
            "PAY_AMT1": [1000.0],
            "PAY_AMT2": [1000.0],
            "PAY_AMT3": [1000.0],
            "PAY_AMT4": [1000.0],
            "PAY_AMT5": [1000.0],
            "PAY_AMT6": [1000.0],
            "default.payment.next.month": [0],
        }
    )


def test_schema_valid(sample_data):
    validated = validate(sample_data)
    assert len(validated) == 1


def test_schema_invalid_age():
    bad = pd.DataFrame(
        {
            "AGE": [5],  # слишком молодой
            "LIMIT_BAL": [50000.0],
            "SEX": [1],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [0],
            "PAY_4": [0],
            "PAY_5": [0],
            "PAY_6": [0],
            "BILL_AMT1": [0.0],
            "BILL_AMT2": [0.0],
            "BILL_AMT3": [0.0],
            "BILL_AMT4": [0.0],
            "BILL_AMT5": [0.0],
            "BILL_AMT6": [0.0],
            "PAY_AMT1": [0.0],
            "PAY_AMT2": [0.0],
            "PAY_AMT3": [0.0],
            "PAY_AMT4": [0.0],
            "PAY_AMT5": [0.0],
            "PAY_AMT6": [0.0],
            "default.payment.next.month": [0],
        }
    )
    with pytest.raises(Exception):
        validate(bad)


def test_add_features(sample_data):
    df = add_features(sample_data)

    assert "pay_mean" in df.columns
    assert "pay_max" in df.columns
    assert "pay_delayed_count" in df.columns
    assert "payment_ratio" in df.columns
    assert "utilization" in df.columns
    assert "age_bin" in df.columns

    # проверка расчёта
    assert df["pay_max"].iloc[0] == 0
    assert df["pay_delayed_count"].iloc[0] == 0


def test_feature_cols_exist(sample_data):
    df = add_features(sample_data)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Отсутствует фича: {col}"
