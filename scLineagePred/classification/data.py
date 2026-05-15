from __future__ import annotations

import os
import random
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import Config


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def pick_device(cfg: Config) -> torch.device:
    if cfg.device_prefer == "cpu":
        return torch.device("cpu")
    if cfg.device_prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.device_prefer == "mps":
        supported = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if supported else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_time_labels_from_h5(h5_path: str, time_count: int) -> List[str]:
    try:
        with h5py.File(h5_path, "r") as handle:
            if "time_labels" in handle:
                raw_labels = handle["time_labels"][:]
                labels = []
                for value in raw_labels:
                    if isinstance(value, (bytes, np.bytes_)):
                        labels.append(value.decode("utf-8"))
                    else:
                        labels.append(str(value))
                if len(labels) == time_count:
                    return labels
            if "timepoints" in handle:
                values = handle["timepoints"][:]
                return [str(float(value)) for value in np.asarray(values).tolist()]
            if "time_values" in handle:
                values = handle["time_values"][:]
                return [str(float(value)) for value in np.asarray(values).tolist()]
    except Exception:
        pass
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


def load_data(cfg: Config):
    print(f"[INFO] Loading H5: {cfg.time_series_h5}")
    with h5py.File(cfg.time_series_h5, "r") as handle:
        X_all = np.array(handle["X"], dtype=np.float32)

        y_h5 = None
        if "y" in handle:
            y_h5 = np.asarray(handle["y"], dtype=np.int64)

        label_h5 = None
        if "label_str" in handle:
            raw_labels = handle["label_str"][:]
            label_h5 = np.array(
                [
                    item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                    for item in raw_labels
                ],
                dtype=object,
            )

    labels = None
    clones = None
    reverse_time_axis = False
    if os.path.isfile(cfg.index_csv):
        df = pd.read_csv(cfg.index_csv)
        if cfg.label_col in df.columns:
            labels = df[cfg.label_col].astype(str).values
        if cfg.clone_col in df.columns:
            clones = df[cfg.clone_col].astype(str).values
        reverse_time_axis = infer_reverse_from_samples_order(df, X_all.shape[1])

    if labels is None:
        if label_h5 is None:
            raise KeyError(f"Missing labels in both CSV ({cfg.label_col}) and H5 (label_str)")
        labels = label_h5

    if clones is None:
        clones = np.array([f"clone_{i}" for i in range(len(labels))], dtype=object)

    if len(labels) != len(X_all):
        raise ValueError(f"Label length mismatch: len(labels)={len(labels)} vs N={len(X_all)}")

    if reverse_time_axis:
        X_all = X_all[:, ::-1, :].copy()
        print("[INFO] Detected reverse samples_order; flipped X time axis to chronological order.")

    selected_labels = list(cfg.target_labels) or list(pd.Index(labels).drop_duplicates())
    keep = np.isin(labels, np.array(selected_labels, dtype=object))
    X = X_all[keep]
    labels_kept = labels[keep]
    clones_kept = clones[keep]

    class_names = list(selected_labels)
    if len(class_names) < 2:
        raise ValueError(f"Need at least 2 target labels for classification, got: {class_names}")
    label_to_y = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_to_y[label] for label in labels_kept], dtype=np.int64)

    print(f"[INFO] Total sequences: {len(X_all)}")
    print(f"[INFO] Kept sequences: {len(X)}")
    for class_name in class_names:
        print(f"  - {class_name}: {(labels_kept == class_name).sum()}")

    time_labels = read_time_labels_from_h5(cfg.time_series_h5, X.shape[1])
    print(f"[INFO] T={X.shape[1]}, time_labels={time_labels}")

    if y_h5 is not None and len(y_h5) == len(X_all):
        y_h5_kept = y_h5[keep]
        mismatch = int((y_h5_kept != y).sum())
        if mismatch > 0:
            print(f"[WARN] y from H5 differs from label mapping in {mismatch} rows; using mapped labels from strings.")

    return X, y, clones_kept, class_names, time_labels


def stratified_split(y: np.ndarray, seed: int, test_frac: float, val_frac: float):
    all_idx = np.arange(len(y))
    try:
        train_idx, test_idx = train_test_split(all_idx, test_size=test_frac, random_state=seed, stratify=y)
        rel_val = val_frac / max(1e-9, 1.0 - test_frac)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=rel_val,
            random_state=seed,
            stratify=y[train_idx],
        )
    except ValueError:
        print("[WARN] Stratified split failed; fallback to random split.")
        train_idx, test_idx = train_test_split(all_idx, test_size=test_frac, random_state=seed, stratify=None)
        rel_val = val_frac / max(1e-9, 1.0 - test_frac)
        train_idx, val_idx = train_test_split(train_idx, test_size=rel_val, random_state=seed, stratify=None)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def build_time_settings(time_labels):
    settings = {}
    order = []
    x_labels = []

    for idx in range(len(time_labels) - 1):
        end_label = time_labels[idx]
        setting_name = f"UpTo_{end_label}"
        settings[setting_name] = idx + 1
        order.append(setting_name)
        x_labels.append(str(end_label))

    final_name = f"All_{time_labels[-1]}"
    settings[final_name] = len(time_labels)
    order.append(final_name)
    x_labels.append(str(time_labels[-1]))

    return settings, order, x_labels


class SeqDataset(Dataset):
    def __init__(self, X, y, indices, keep_len=None):
        self.X = X
        self.y = y
        self.indices = indices
        self.keep_len = keep_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        sample_idx = int(self.indices[item])
        x = self.X[sample_idx]
        if self.keep_len is None:
            cut = x
            length = x.shape[0]
        else:
            length = int(self.keep_len)
            cut = x[:length]
        return (
            torch.from_numpy(np.asarray(cut, dtype=np.float32)),
            torch.tensor(int(self.y[sample_idx]), dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


def collate_pad(batch):
    xs, ys, lengths = zip(*batch)
    lengths = torch.stack(lengths, dim=0)
    ys = torch.stack(ys, dim=0)

    max_len = int(lengths.max().item())
    feature_dim = xs[0].shape[1]
    padded = torch.zeros((len(xs), max_len, feature_dim), dtype=torch.float32)
    for idx, x in enumerate(xs):
        padded[idx, : x.shape[0]] = x
    return padded, ys, lengths
