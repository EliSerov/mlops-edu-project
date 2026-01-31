"""
Бенчмарк производительности инференса для разных форматов моделей и устройств.
"""
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from pathlib import Path
import pickle
import json
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    name: str
    device: str
    batch_size: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    warmup_runs: int
    test_runs: int


def benchmark_onnx(
    model_path: str | Path,
    input_data: np.ndarray,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
    device: str = "cpu",
) -> BenchmarkResult:
    """Бенчмарк инференса ONNX модели."""
    
    providers = ["CPUExecutionProvider"]
    if device == "gpu" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    session = ort.InferenceSession(str(model_path), providers=providers)
    actual_device = session.get_providers()[0].replace("ExecutionProvider", "").lower()
    
    # готовим батчированный вход
    n_samples = len(input_data)
    
    # прогрев
    for _ in range(warmup):
        idx = np.random.choice(n_samples, batch_size, replace=True)
        batch = input_data[idx]
        session.run(None, {"features": batch})
    
    # бенчмарк
    latencies = []
    for _ in range(runs):
        idx = np.random.choice(n_samples, batch_size, replace=True)
        batch = input_data[idx]
        
        start = time.perf_counter()
        session.run(None, {"features": batch})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    
    latencies = np.array(latencies)
    
    return BenchmarkResult(
        name=Path(model_path).stem,
        device=actual_device,
        batch_size=batch_size,
        avg_latency_ms=latencies.mean(),
        p50_latency_ms=np.percentile(latencies, 50),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        throughput_rps=1000 / latencies.mean() * batch_size,
        warmup_runs=warmup,
        test_runs=runs,
    )


def benchmark_pytorch(
    model_dir: str | Path,
    input_data: np.ndarray,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
    device: str = "cpu",
) -> BenchmarkResult:
    """Бенчмарк инференса PyTorch модели."""
    from .network import create_model
    
    model_dir = Path(model_dir)
    
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)
    
    # поддержка разных устройств
    if device == "cpu":
        torch_device = torch.device("cpu")
    elif device == "mps":
        torch_device = torch.device("mps")  # Apple GPU
    else:
        torch_device = torch.device("cuda")
    
    model = create_model(meta["input_dim"], meta.get("config"))
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.to(torch_device)
    model.eval()
    
    input_tensor = torch.FloatTensor(input_data).to(torch_device)
    n_samples = len(input_data)
    
    # прогрев
    with torch.no_grad():
        for _ in range(warmup):
            idx = torch.randint(0, n_samples, (batch_size,))
            batch = input_tensor[idx]
            model(batch)
            # синхронизация для GPU/MPS
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
    
    # бенчмарк
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            idx = torch.randint(0, n_samples, (batch_size,))
            batch = input_tensor[idx]
            
            start = time.perf_counter()
            model(batch)
            # синхронизация для точного замера
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
    
    latencies = np.array(latencies)
    
    return BenchmarkResult(
        name="pytorch",
        device=device,
        batch_size=batch_size,
        avg_latency_ms=latencies.mean(),
        p50_latency_ms=np.percentile(latencies, 50),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        throughput_rps=1000 / latencies.mean() * batch_size,
        warmup_runs=warmup,
        test_runs=runs,
    )


def benchmark_onnx_coreml(
    model_path: str | Path,
    input_data: np.ndarray,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
) -> BenchmarkResult:
    """Бенчмарк ONNX с CoreML провайдером (Apple Neural Engine / ANE)."""
    
    # CoreML может использовать CPU, GPU или Neural Engine автоматически
    session = ort.InferenceSession(
        str(model_path), 
        providers=["CoreMLExecutionProvider"]
    )
    
    n_samples = len(input_data)
    
    # прогрев
    for _ in range(warmup):
        idx = np.random.choice(n_samples, batch_size, replace=True)
        batch = input_data[idx]
        session.run(None, {"features": batch})
    
    # бенчмарк
    latencies = []
    for _ in range(runs):
        idx = np.random.choice(n_samples, batch_size, replace=True)
        batch = input_data[idx]
        
        start = time.perf_counter()
        session.run(None, {"features": batch})
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    
    latencies = np.array(latencies)
    
    return BenchmarkResult(
        name="onnx_coreml",
        device="ane",  # Apple Neural Engine
        batch_size=batch_size,
        avg_latency_ms=latencies.mean(),
        p50_latency_ms=np.percentile(latencies, 50),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        throughput_rps=1000 / latencies.mean() * batch_size,
        warmup_runs=warmup,
        test_runs=runs,
    )


def run_benchmark_suite(
    pytorch_dir: str | Path,
    onnx_path: str | Path,
    onnx_quant_path: str | Path = None,
    test_data_path: str | Path = None,
    scaler_path: str | Path = None,
    batch_sizes: list[int] = None,
) -> pd.DataFrame:
    """Запуск полного набора бенчмарков, возвращает результаты в DataFrame."""
    
    TARGET = "default.payment.next.month"
    batch_sizes = batch_sizes or [1, 16, 64, 256]
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    test_df = pd.read_csv(test_data_path)
    X = test_df.drop(columns=[TARGET]).values
    X_scaled = scaler.transform(X).astype(np.float32)
    
    # проверяем наличие квантизованной модели
    has_quant = onnx_quant_path and Path(onnx_quant_path).exists()
    if not has_quant:
        print("INT8 модель не найдена, пропускаем бенчмарк квантизации")
    
    # проверяем CoreML (Apple Silicon NPU)
    has_coreml = "CoreMLExecutionProvider" in ort.get_available_providers()
    if has_coreml:
        print("CoreML доступен (Apple Neural Engine)")
    
    results = []
    
    for bs in batch_sizes:
        print(f"\nBatch size: {bs}")
        
        # PyTorch CPU
        r = benchmark_pytorch(pytorch_dir, X_scaled, batch_size=bs, device="cpu")
        results.append(r)
        print(f"  PyTorch CPU: {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
        
        # ONNX CPU
        r = benchmark_onnx(onnx_path, X_scaled, batch_size=bs, device="cpu")
        results.append(r)
        print(f"  ONNX CPU:    {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
        
        # ONNX CoreML / Apple Neural Engine
        if has_coreml:
            try:
                r = benchmark_onnx_coreml(onnx_path, X_scaled, batch_size=bs)
                results.append(r)
                print(f"  ONNX CoreML: {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
            except Exception as e:
                print(f"  ONNX CoreML: ошибка - {e}")
        
        # ONNX INT8 (если есть)
        if has_quant:
            r = benchmark_onnx(onnx_quant_path, X_scaled, batch_size=bs, device="cpu")
            results.append(r)
            print(f"  ONNX INT8:   {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
        
        # PyTorch MPS (Apple GPU)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                r = benchmark_pytorch(pytorch_dir, X_scaled, batch_size=bs, device="mps")
                results.append(r)
                print(f"  PyTorch MPS: {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
            except Exception as e:
                print(f"  PyTorch MPS: ошибка - {e}")
        
        # CUDA GPU (если доступен)
        if torch.cuda.is_available():
            r = benchmark_pytorch(pytorch_dir, X_scaled, batch_size=bs, device="cuda")
            results.append(r)
            print(f"  PyTorch GPU: {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
        
        if "CUDAExecutionProvider" in ort.get_available_providers():
            r = benchmark_onnx(onnx_path, X_scaled, batch_size=bs, device="gpu")
            results.append(r)
            print(f"  ONNX GPU:    {r.avg_latency_ms:.3f}ms, {r.throughput_rps:.0f} rps")
    
    df = pd.DataFrame([vars(r) for r in results])
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-dir", default="models/pytorch")
    parser.add_argument("--onnx", default="models/onnx/model.onnx")
    parser.add_argument("--onnx-quant", default="models/onnx/model_int8.onnx")
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument("--scaler", default="models/pytorch/scaler.pkl")
    parser.add_argument("--output", default="reports/benchmark.csv")
    args = parser.parse_args()
    
    df = run_benchmark_suite(
        args.pytorch_dir,
        args.onnx,
        args.onnx_quant,
        args.test_data,
        args.scaler,
    )
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nСохранено в {args.output}")
