"""
FastAPI сервис для кредитного скоринга.
Использует ONNX runtime для продакшен инференса.
"""
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# метрики
REQUEST_COUNT = Counter(
    "prediction_requests_total", 
    "Всего запросов предсказаний",
    ["status"]
)
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Латенси предсказаний",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_DIST = Histogram(
    "prediction_probability",
    "Распределение предсказанных вероятностей",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


class ModelState:
    session: ort.InferenceSession = None
    scaler = None
    feature_names: list[str] = None


state = ModelState()


def load_model():
    model_dir = Path(os.getenv("MODEL_DIR", "models/onnx"))
    onnx_path = model_dir / os.getenv("ONNX_MODEL", "model_int8.onnx")
    
    # fallback на не-квантизованную если int8 не найдена
    if not onnx_path.exists():
        onnx_path = model_dir / "model.onnx"
    
    scaler_path = Path(os.getenv("SCALER_PATH", "models/pytorch/scaler.pkl"))
    
    state.session = ort.InferenceSession(
        str(onnx_path), 
        providers=["CPUExecutionProvider"]
    )
    
    with open(scaler_path, "rb") as f:
        state.scaler = pickle.load(f)
    
    # названия фичей
    state.feature_names = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "pay_mean", "pay_max", "pay_delayed_count", "payment_ratio", "utilization", "age_bin"
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Credit Scoring API",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    limit_bal: float = Field(..., alias="LIMIT_BAL")
    sex: int = Field(..., alias="SEX")
    education: int = Field(..., alias="EDUCATION")
    marriage: int = Field(..., alias="MARRIAGE")
    age: int = Field(..., alias="AGE")
    pay_0: int = Field(..., alias="PAY_0")
    pay_2: int = Field(..., alias="PAY_2")
    pay_3: int = Field(..., alias="PAY_3")
    pay_4: int = Field(..., alias="PAY_4")
    pay_5: int = Field(..., alias="PAY_5")
    pay_6: int = Field(..., alias="PAY_6")
    bill_amt1: float = Field(..., alias="BILL_AMT1")
    bill_amt2: float = Field(..., alias="BILL_AMT2")
    bill_amt3: float = Field(..., alias="BILL_AMT3")
    bill_amt4: float = Field(..., alias="BILL_AMT4")
    bill_amt5: float = Field(..., alias="BILL_AMT5")
    bill_amt6: float = Field(..., alias="BILL_AMT6")
    pay_amt1: float = Field(..., alias="PAY_AMT1")
    pay_amt2: float = Field(..., alias="PAY_AMT2")
    pay_amt3: float = Field(..., alias="PAY_AMT3")
    pay_amt4: float = Field(..., alias="PAY_AMT4")
    pay_amt5: float = Field(..., alias="PAY_AMT5")
    pay_amt6: float = Field(..., alias="PAY_AMT6")
    
    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    model_version: str = "v1"


def compute_features(req: PredictRequest) -> np.ndarray:
    """Собрать вектор фичей включая сгенерированные."""
    pay_cols = [req.pay_0, req.pay_2, req.pay_3, req.pay_4, req.pay_5, req.pay_6]
    bill_cols = [req.bill_amt1, req.bill_amt2, req.bill_amt3, req.bill_amt4, req.bill_amt5, req.bill_amt6]
    amt_cols = [req.pay_amt1, req.pay_amt2, req.pay_amt3, req.pay_amt4, req.pay_amt5, req.pay_amt6]
    
    pay_mean = np.mean(pay_cols)
    pay_max = max(pay_cols)
    pay_delayed_count = sum(1 for p in pay_cols if p > 0)
    
    total_bill = max(sum(bill_cols), 1)
    total_paid = sum(amt_cols)
    payment_ratio = total_paid / total_bill
    
    utilization = req.bill_amt1 / max(req.limit_bal, 1)
    
    # возрастной бакет
    if req.age <= 25:
        age_bin = 0
    elif req.age <= 35:
        age_bin = 1
    elif req.age <= 45:
        age_bin = 2
    elif req.age <= 55:
        age_bin = 3
    else:
        age_bin = 4
    
    features = [
        req.limit_bal, req.sex, req.education, req.marriage, req.age,
        req.pay_0, req.pay_2, req.pay_3, req.pay_4, req.pay_5, req.pay_6,
        req.bill_amt1, req.bill_amt2, req.bill_amt3, req.bill_amt4, req.bill_amt5, req.bill_amt6,
        req.pay_amt1, req.pay_amt2, req.pay_amt3, req.pay_amt4, req.pay_amt5, req.pay_amt6,
        pay_mean, pay_max, pay_delayed_count, payment_ratio, utilization, age_bin,
    ]
    
    return np.array(features, dtype=np.float32).reshape(1, -1)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.perf_counter()
    
    try:
        features = compute_features(request)
        scaled = state.scaler.transform(features).astype(np.float32)
        
        probs = state.session.run(None, {"features": scaled})[0]
        prob = float(probs[0][0])
        pred = 1 if prob >= 0.5 else 0
        
        REQUEST_COUNT.labels(status="success").inc()
        PREDICTION_DIST.observe(prob)
        
        return PredictResponse(probability=prob, prediction=pred)
    
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        LATENCY.observe(time.perf_counter() - start)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.session is not None}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {"service": "credit-scoring", "version": "0.1.0"}
