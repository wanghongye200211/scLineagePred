from __future__ import annotations

import os
import pickle
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..classification.models import BiLSTMModel, RNNModel, TransformerModel
except ImportError:
    from scLineagePred.classification.models import BiLSTMModel, RNNModel, TransformerModel

from .config import Config


def load_ensemble(cfg: Config, input_dim: int, n_classes: int, device: str, setting: str):
    seed = int(cfg.base_seed)
    models = {
        "BiLSTM": BiLSTMModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, n_classes).to(device),
        "RNN": RNNModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, n_classes).to(device),
        "Trans": TransformerModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead, n_classes).to(device),
    }

    for name, model in models.items():
        path = os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] missing model: {path}")
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()

    stacker_path = os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl")
    if not os.path.exists(stacker_path):
        raise FileNotFoundError(f"[ERROR] missing stacking LR: {stacker_path}")
    with open(stacker_path, "rb") as handle:
        stacker = pickle.load(handle)
    return models, stacker


@torch.no_grad()
def predict_proba_stack(models: Dict[str, torch.nn.Module], stacker, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    _, time_count, _ = X.shape
    parts = {"BiLSTM": [], "RNN": [], "Trans": []}

    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        xb = torch.from_numpy(X[start:stop]).to(device)
        lengths = torch.full((stop - start,), time_count, dtype=torch.long, device=device)

        for name in ["BiLSTM", "RNN", "Trans"]:
            logits = models[name](xb, lengths)
            parts[name].append(F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))

    features = np.concatenate(
        [
            np.concatenate(parts["BiLSTM"], axis=0),
            np.concatenate(parts["RNN"], axis=0),
            np.concatenate(parts["Trans"], axis=0),
        ],
        axis=1,
    )
    return stacker.predict_proba(features).astype(np.float32)
