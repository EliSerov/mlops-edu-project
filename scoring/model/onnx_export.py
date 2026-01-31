"""
Утилиты экспорта в ONNX для PD модели.
"""
from __future__ import annotations
from typing import Union
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import pickle

from .network import create_model


def export_to_onnx(model_dir: Union[str, Path], onnx_path: Union[str, Path], use_legacy: bool = True):
    """
    Экспорт PyTorch модели в формат ONNX.
    
    use_legacy=True использует старый TorchScript экспортер (более совместим с quantization).
    use_legacy=False использует новый Dynamo экспортер (лучше оптимизация, но проблемы с ort quantize).
    """
    model_dir = Path(model_dir)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)
    
    model = create_model(meta["input_dim"], meta.get("config"))
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    
    dummy_input = torch.randn(1, meta["input_dim"])
    
    if use_legacy:
        # Legacy TorchScript exporter — совместим с onnxruntime quantization
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["probability"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "probability": {0: "batch_size"},
            },
            dynamo=False,  # форсируем legacy exporter
        )
    else:
        # Новый Dynamo exporter — лучше оптимизация, но проблемы с quantize
        temp_path = onnx_path.parent / "temp_model.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            str(temp_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["probability"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "probability": {0: "batch_size"},
            },
        )
        
        # объединяем внешние данные в один файл
        onnx_model = onnx.load(str(temp_path), load_external_data=True)
        onnx.save(onnx_model, str(onnx_path), save_as_external_data=False)
        
        temp_path.unlink(missing_ok=True)
        temp_data = onnx_path.parent / "temp_model.onnx.data"
        temp_data.unlink(missing_ok=True)
    
    # проверка
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"Экспортировано в {onnx_path}")
    return onnx_path


def validate_onnx(
    model_dir: Union[str, Path], 
    onnx_path: Union[str, Path], 
    test_data_path: Union[str, Path],
    tolerance: float = 1e-5
) -> dict:
    """
    Сравнение выходов PyTorch и ONNX для проверки корректности конвертации.
    Возвращает dict с max diff и статусом pass/fail.
    """
    import pandas as pd
    from .train import TARGET
    
    model_dir = Path(model_dir)
    
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)
    
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # загружаем pytorch модель
    model = create_model(meta["input_dim"], meta.get("config"))
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    
    # загружаем onnx
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    # тестовые данные
    test_df = pd.read_csv(test_data_path)
    X = test_df.drop(columns=[TARGET]).values[:100]  # выборка для валидации
    X_scaled = scaler.transform(X).astype(np.float32)
    
    # pytorch инференс
    with torch.no_grad():
        pt_out = model(torch.FloatTensor(X_scaled)).numpy()
    
    # onnx инференс
    onnx_out = session.run(None, {"features": X_scaled})[0]
    
    max_diff = np.abs(pt_out - onnx_out).max()
    mean_diff = np.abs(pt_out - onnx_out).mean()
    
    result = {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "tolerance": tolerance,
        "passed": max_diff < tolerance,
        "samples_checked": len(X_scaled),
    }
    
    print(f"Валидация: max_diff={max_diff:.2e}, passed={result['passed']}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/pytorch")
    parser.add_argument("--output", default="models/onnx/model.onnx")
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    
    export_to_onnx(args.model_dir, args.output)
    
    if args.validate:
        validate_onnx(args.model_dir, args.output, args.test_data)
