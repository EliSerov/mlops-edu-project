"""
Простая feed-forward сеть для предсказания PD.
PyTorch реализация — проще экспорт в ONNX чем sklearn.
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn


class PDNetwork(nn.Module):
    """
    3-слойный MLP для предсказания дефолта.
    Ничего сложного — ReLU, dropout, sigmoid на выходе.
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))


def create_model(input_dim: int, config: dict = None) -> PDNetwork:
    config = config or {}
    return PDNetwork(
        input_dim=input_dim,
        hidden_dims=config.get("hidden_dims", [64, 32]),
        dropout=config.get("dropout", 0.3),
    )
