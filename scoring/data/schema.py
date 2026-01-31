"""
Pandera схема для датасета кредитных карт.
Основано на структуре UCI Credit Card dataset.
"""

import pandera as pa
from pandera import Check, Column

# PAY_* колонки: -2=нет потребления, -1=оплачено полностью, 0=револьверный, 1+=месяцев просрочки
PAY_RANGE = (-2, 9)

credit_schema = pa.DataFrameSchema(
    {
        "LIMIT_BAL": Column(float, Check.ge(0), nullable=False),
        "SEX": Column(int, Check.isin([1, 2]), nullable=False),
        "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False),
        "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3]), nullable=False),
        "AGE": Column(int, Check.in_range(18, 100), nullable=False),
        "PAY_0": Column(int, Check.in_range(*PAY_RANGE), nullable=False),
        "PAY_2": Column(int, Check.in_range(*PAY_RANGE), nullable=False),
        "PAY_3": Column(int, Check.in_range(*PAY_RANGE), nullable=False),
        "PAY_4": Column(int, Check.in_range(*PAY_RANGE), nullable=False),
        "PAY_5": Column(int, Check.in_range(*PAY_RANGE), nullable=False),
        "PAY_6": Column(int, Check.in_range(*PAY_RANGE), nullable=False),
        "BILL_AMT1": Column(float, nullable=False),
        "BILL_AMT2": Column(float, nullable=False),
        "BILL_AMT3": Column(float, nullable=False),
        "BILL_AMT4": Column(float, nullable=False),
        "BILL_AMT5": Column(float, nullable=False),
        "BILL_AMT6": Column(float, nullable=False),
        "PAY_AMT1": Column(float, Check.ge(0), nullable=False),
        "PAY_AMT2": Column(float, Check.ge(0), nullable=False),
        "PAY_AMT3": Column(float, Check.ge(0), nullable=False),
        "PAY_AMT4": Column(float, Check.ge(0), nullable=False),
        "PAY_AMT5": Column(float, Check.ge(0), nullable=False),
        "PAY_AMT6": Column(float, Check.ge(0), nullable=False),
        "default.payment.next.month": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,  # разрешаем дополнительные колонки типа ID
    coerce=True,
)


def validate(df):
    """Валидация датафрейма по схеме, возвращает валидированный df или кидает исключение."""
    return credit_schema.validate(df)
