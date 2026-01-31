"""
Квантизация ONNX — INT8 динамическая квантизация.
Уменьшает размер модели и может ускорить инференс на CPU.
"""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import os


def quantize_model(input_path: str | Path, output_path: str | Path = None) -> Path:
    """
    Применение INT8 динамической квантизации к ONNX модели.
    Динамическая квантизация: веса квантизованы, активации вычисляются в runtime.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_int8.onnx"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    
    # сравнение размеров
    orig_size = os.path.getsize(input_path) / 1024
    quant_size = os.path.getsize(output_path) / 1024
    
    print(f"Оригинал: {orig_size:.1f} KB")
    print(f"Квантизованная: {quant_size:.1f} KB")
    print(f"Сжатие: {orig_size/quant_size:.2f}x")
    
    return output_path


def compare_accuracy(
    original_path: str | Path,
    quantized_path: str | Path,
    test_data_path: str | Path,
    scaler_path: str | Path,
) -> dict:
    """Сравнение точности между оригинальной и квантизованной моделями."""
    import onnxruntime as ort
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import roc_auc_score
    
    TARGET = "default.payment.next.month"
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    test_df = pd.read_csv(test_data_path)
    X = test_df.drop(columns=[TARGET]).values
    y = test_df[TARGET].values
    X_scaled = scaler.transform(X).astype(np.float32)
    
    # оригинальная
    sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    probs_orig = sess_orig.run(None, {"features": X_scaled})[0].flatten()
    auc_orig = roc_auc_score(y, probs_orig)
    
    # квантизованная
    sess_quant = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    probs_quant = sess_quant.run(None, {"features": X_scaled})[0].flatten()
    auc_quant = roc_auc_score(y, probs_quant)
    
    result = {
        "auc_original": auc_orig,
        "auc_quantized": auc_quant,
        "auc_diff": auc_orig - auc_quant,
        "prob_max_diff": float(np.abs(probs_orig - probs_quant).max()),
    }
    
    print(f"ROC-AUC оригинал: {auc_orig:.4f}")
    print(f"ROC-AUC квантизованная: {auc_quant:.4f}")
    print(f"Разница: {result['auc_diff']:.6f}")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/onnx/model.onnx")
    parser.add_argument("--output", default="models/onnx/model_int8.onnx")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument("--scaler", default="models/pytorch/scaler.pkl")
    args = parser.parse_args()
    
    quantize_model(args.input, args.output)
    
    if args.compare:
        compare_accuracy(args.input, args.output, args.test_data, args.scaler)
