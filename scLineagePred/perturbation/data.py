from __future__ import annotations

import os
import re
from typing import List

import h5py
import numpy as np
import pandas as pd

from .config import Config


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_device(cfg: Config) -> str:
    if cfg.device == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return cfg.device


def sanitize_name(text: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_")
    return clean or "NA"


def stratified_subsample_indices(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    all_idx = np.arange(len(y), dtype=np.int64)
    if len(all_idx) <= max_n:
        return all_idx

    keep = []
    for class_idx in range(int(y.max()) + 1):
        class_values = all_idx[y == class_idx]
        if len(class_values) == 0:
            continue
        quota = int(round(max_n * (len(class_values) / float(len(all_idx)))))
        quota = max(1, min(quota, len(class_values)))
        keep.append(rng.choice(class_values, size=quota, replace=False))

    keep = np.unique(np.concatenate(keep)) if keep else rng.choice(all_idx, size=max_n, replace=False)
    if len(keep) > max_n:
        keep = rng.choice(keep, size=max_n, replace=False)
    return np.sort(keep.astype(np.int64))


def read_time_labels_from_h5(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as handle:
        if "time_labels" in handle:
            raw_labels = handle["time_labels"][:]
            return [
                item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                for item in raw_labels
            ]
        if "time_values" in handle:
            return [str(int(value)) for value in np.asarray(handle["time_values"][:]).tolist()]
        time_count = int(handle["X"].shape[1])
    return [f"t{i}" for i in range(time_count)]


def infer_reverse_from_samples_order(df: pd.DataFrame, time_count: int) -> bool:
    if "samples_order" not in df.columns:
        return False
    for raw_value in df["samples_order"].dropna().astype(str).head(200):
        tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
        values = []
        for token in tokens:
            if not token.lstrip("-").isdigit():
                values = []
                break
            values.append(int(token))
        if len(values) < 2:
            continue
        if time_count > 0 and len(values) != time_count:
            pass
        return values[0] > values[-1]
    return False


def ood_rate_1d(ref: np.ndarray, new: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    new = np.asarray(new, dtype=np.float32).reshape(-1)
    min_value = float(ref.min())
    max_value = float(ref.max())
    return float(((new < min_value) | (new > max_value)).mean())


def load_sequences(cfg: Config):
    if not os.path.isfile(cfg.index_csv):
        raise FileNotFoundError(f"[ERROR] missing index csv: {cfg.index_csv}")
    if not os.path.isfile(cfg.time_series_h5):
        raise FileNotFoundError(f"[ERROR] missing sequences h5: {cfg.time_series_h5}")

    df = pd.read_csv(cfg.index_csv)
    with h5py.File(cfg.time_series_h5, "r") as handle:
        X_all = np.asarray(handle["X"], dtype=np.float32)
        label_h5 = None
        if "label_str" in handle:
            raw = handle["label_str"][:]
            label_h5 = np.array(
                [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in raw],
                dtype=object,
            )

    time_labels = read_time_labels_from_h5(cfg.time_series_h5)
    if len(time_labels) != X_all.shape[1]:
        if len(time_labels) > X_all.shape[1]:
            time_labels = time_labels[: X_all.shape[1]]
        else:
            time_labels = time_labels + [f"t{i}" for i in range(len(time_labels), X_all.shape[1])]

    if infer_reverse_from_samples_order(df, X_all.shape[1]):
        X_all = X_all[:, ::-1, :].copy()
        print("[INFO] Detected reverse samples_order; flipped X time axis to chronological order.")

    if cfg.csv_label_col in df.columns:
        labels_all = df[cfg.csv_label_col].astype(str).to_numpy()
    elif label_h5 is not None:
        labels_all = label_h5
    else:
        raise KeyError(f"[ERROR] label column missing in index csv ({cfg.csv_label_col}) and h5(label_str)")

    if len(labels_all) != len(X_all):
        raise ValueError(f"[ERROR] Label length mismatch: labels={len(labels_all)} vs X={len(X_all)}")

    if cfg.target_labels:
        class_names = list(cfg.target_labels)
        keep = np.isin(labels_all, np.array(class_names, dtype=object))
        if not keep.any():
            raise ValueError(f"[ERROR] No sequences match target_labels={class_names}")
        X = X_all[keep]
        labels = labels_all[keep]
    else:
        X = X_all
        labels = labels_all
        class_names = [str(label) for label in pd.unique(labels)]

    label_to_y = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_to_y[label] for label in labels], dtype=np.int64)

    if cfg.max_sequences_scan is not None:
        keep_idx = stratified_subsample_indices(y, int(cfg.max_sequences_scan), seed=cfg.base_seed)
        X = X[keep_idx]
        y = y[keep_idx]
        labels = labels[keep_idx]

    label_counts = {name: int((labels == name).sum()) for name in class_names}
    print(f"[INFO] Loaded sequences: N={len(X)} T={X.shape[1]} D={X.shape[2]}")
    print(f"[INFO] Label counts: {label_counts}")
    print(f"[INFO] time_labels={time_labels}")
    return X, y.astype(np.int64), class_names, time_labels


def build_endpoint_strategies(time_labels: List[str]):
    if len(time_labels) < 2:
        raise ValueError(f"Need at least 2 timepoints, got {len(time_labels)}")
    prev_idx = len(time_labels) - 2
    last_idx = len(time_labels) - 1

    prev_label = str(time_labels[prev_idx])
    last_label = str(time_labels[last_idx])
    return [
        {
            "scenario_id": "obs_to_prev_pred_last",
            "scenario_label": f"ObsTo_{prev_label}_Pred_{last_label}",
            "setting": f"UpTo_{prev_label}",
            "keep_len": prev_idx + 1,
            "perturb_t_indices": list(range(prev_idx + 1)),
        },
        {
            "scenario_id": "obs_to_last_pred_last",
            "scenario_label": f"ObsTo_{last_label}_Pred_{last_label}",
            "setting": f"All_{last_label}",
            "keep_len": last_idx + 1,
            "perturb_t_indices": list(range(last_idx + 1)),
        },
    ]


def build_strategies(cfg: Config, time_labels: List[str]):
    if str(cfg.scenario_mode).strip().lower() == "last_two":
        return build_endpoint_strategies(time_labels)
    raise ValueError(f"Unsupported scenario_mode={cfg.scenario_mode!r}")
