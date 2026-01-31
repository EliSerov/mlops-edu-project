import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # нужно замокать загрузку модели для тестов
    import os

    os.environ["MODEL_DIR"] = "models/onnx"
    os.environ["SCALER_PATH"] = "models/pytorch/scaler.pkl"

    from scoring.api.main import app

    return TestClient(app)


@pytest.mark.skip(reason="требуется обученная модель")
def test_predict(client):
    response = client.post(
        "/predict",
        json={
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 30,
            "PAY_0": 0,
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 10000,
            "BILL_AMT2": 10000,
            "BILL_AMT3": 10000,
            "BILL_AMT4": 10000,
            "BILL_AMT5": 10000,
            "BILL_AMT6": 10000,
            "PAY_AMT1": 1000,
            "PAY_AMT2": 1000,
            "PAY_AMT3": 1000,
            "PAY_AMT4": 1000,
            "PAY_AMT5": 1000,
            "PAY_AMT6": 1000,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "prediction" in data
    assert 0 <= data["probability"] <= 1
    assert data["prediction"] in [0, 1]


def test_health(client):
    # должно работать даже без модели
    response = client.get("/health")
    assert response.status_code == 200
