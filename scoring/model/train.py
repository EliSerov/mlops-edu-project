"""
Скрипт обучения PD модели.
Сохраняет PyTorch чекпоинт.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .network import create_model

TARGET = "default.payment.next.month"


class Trainer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

    def fit(self, train_path: str, epochs: int = 50, batch_size: int = 256, lr: float = 0.001):
        train_df = pd.read_csv(train_path)

        X = train_df.drop(columns=[TARGET]).values
        y = train_df[TARGET].values
        self.feature_names = [c for c in train_df.columns if c != TARGET]

        X_scaled = self.scaler.fit_transform(X)

        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y).unsqueeze(1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = create_model(X.shape[1], self.config)
        self.model.to(self.device)

        # BCELoss работает с sigmoid выходом модели
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        self.model.eval()
        return self

    def evaluate(self, test_path: str) -> dict:
        test_df = pd.read_csv(test_path)
        X = test_df.drop(columns=[TARGET]).values
        y = test_df[TARGET].values

        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy().flatten()

        preds = (probs >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y, probs),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
        }
        return metrics

    def save(self, output_dir: Union[str, Path], metrics: dict = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_dir / "model.pt")

        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "feature_names": self.feature_names,
            "input_dim": len(self.feature_names),
            "config": self.config,
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        if metrics:
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f)

    @classmethod
    def load(cls, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)

        with open(model_dir / "meta.json") as f:
            meta = json.load(f)

        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        trainer = cls(meta.get("config"))
        trainer.scaler = scaler
        trainer.feature_names = meta["feature_names"]

        trainer.model = create_model(meta["input_dim"], meta.get("config"))
        trainer.model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
        trainer.model.eval()

        return trainer


def train_simple(
    train_path: str,
    test_path: str,
    output_dir: str,
    epochs: int = 50,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    lr: float = 0.001,
):
    """Обучение без MLflow (для локального запуска)."""
    config = {"hidden_dims": hidden_dims or [64, 32], "dropout": dropout}

    trainer = Trainer(config)
    trainer.fit(train_path, epochs=epochs, lr=lr)

    metrics = trainer.evaluate(test_path)
    trainer.save(output_dir, metrics)

    print(f"Metrics: {metrics}")
    return trainer, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output", default="models/pytorch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_simple(args.train, args.test, args.output, epochs=args.epochs, lr=args.lr)
