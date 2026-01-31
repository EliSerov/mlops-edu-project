"""
Нагрузочное тестирование с Locust.
Запуск: locust -f tests/locustfile.py --host http://localhost:8000
"""
from locust import HttpUser, task, between
import random


class ScoringUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        # прогрев
        self.client.get("/health")
    
    @task(10)
    def predict(self):
        # рандомизируем входные данные
        payload = {
            "LIMIT_BAL": random.randint(10000, 500000),
            "SEX": random.choice([1, 2]),
            "EDUCATION": random.choice([1, 2, 3, 4]),
            "MARRIAGE": random.choice([1, 2, 3]),
            "AGE": random.randint(21, 65),
            "PAY_0": random.randint(-2, 2),
            "PAY_2": random.randint(-2, 2),
            "PAY_3": random.randint(-2, 2),
            "PAY_4": random.randint(-2, 2),
            "PAY_5": random.randint(-2, 2),
            "PAY_6": random.randint(-2, 2),
            "BILL_AMT1": random.randint(0, 100000),
            "BILL_AMT2": random.randint(0, 100000),
            "BILL_AMT3": random.randint(0, 100000),
            "BILL_AMT4": random.randint(0, 100000),
            "BILL_AMT5": random.randint(0, 100000),
            "BILL_AMT6": random.randint(0, 100000),
            "PAY_AMT1": random.randint(0, 50000),
            "PAY_AMT2": random.randint(0, 50000),
            "PAY_AMT3": random.randint(0, 50000),
            "PAY_AMT4": random.randint(0, 50000),
            "PAY_AMT5": random.randint(0, 50000),
            "PAY_AMT6": random.randint(0, 50000),
        }
        self.client.post("/predict", json=payload)
    
    @task(1)
    def health(self):
        self.client.get("/health")
    
    @task(1)
    def metrics(self):
        self.client.get("/metrics")
