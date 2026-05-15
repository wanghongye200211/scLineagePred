from __future__ import annotations

import os
import random
import re

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            print("[Device] Using CUDA")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device] Using MPS")
            return "mps"
        print("[Device] Using CPU")
        return "cpu"
    return device


def safe_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def read_time_labels_from_h5(h5_path: str, time_count: int):
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
                return [str(float(value)) for value in np.asarray(handle["timepoints"][:]).tolist()]
            if "time_values" in handle:
                return [str(int(value)) for value in np.asarray(handle["time_values"][:]).tolist()]
    except Exception:
        pass
    return [f"t{i}" for i in range(time_count)]


def sanitize_name(value: str) -> str:
    text = str(value).replace(" ", "_")
    text = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def infer_reverse_from_samples_order(df_idx: pd.DataFrame, time_count: int) -> bool:
    if "samples_order" not in df_idx.columns:
        return False

    for raw_value in df_idx["samples_order"].dropna().astype(str).head(200):
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


def maybe_flip_time_labels(timepoints):
    try:
        values = np.array([float(value) for value in timepoints], dtype=np.float64)
    except Exception:
        return timepoints, False
    if len(values) >= 2 and np.all(np.diff(values) < 0):
        return list(timepoints[::-1]), True
    return timepoints, False


def load_h5_sequences(path: str):
    with h5py.File(path, "r") as handle:
        if "X" in handle:
            X = handle["X"][:]
        elif "data" in handle:
            X = handle["data"][:]
        else:
            raise KeyError("H5 missing dataset 'X' (or 'data').")
        indices = handle["indices"][:] if "indices" in handle else None
        mask = handle["mask"][:] if "mask" in handle else None
        label_str = handle["label_str"][:] if "label_str" in handle else None

    X = np.asarray(X, dtype=np.float32)
    if indices is not None:
        indices = np.asarray(indices, dtype=np.int64)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.int8)
    return X, indices, mask, label_str


def read_gene_list(ae_result_dir: str):
    gene_path = os.path.join(ae_result_dir, "genes.txt")
    if not os.path.exists(gene_path):
        raise FileNotFoundError(f"Missing genes.txt: {gene_path}")
    with open(gene_path, "r", encoding="utf-8") as handle:
        genes = [line.strip() for line in handle if line.strip()]
    return np.array(genes, dtype=object)


def get_expr_matrix(adata_obj, source: str):
    if source == "X":
        X = adata_obj.X
    elif source == "raw":
        if adata_obj.raw is None:
            raise ValueError("adata.raw is None but adata_expr_source='raw'")
        X = adata_obj.raw.X
    elif source.startswith("layer:"):
        key = source.split("layer:", 1)[1]
        if key not in adata_obj.layers:
            raise KeyError(f"adata.layers['{key}'] not found")
        X = adata_obj.layers[key]
    else:
        raise ValueError(f"Unknown adata_expr_source: {source}")
    return safe_dense(X).astype(np.float32)


def compute_clone_means(expr, clone_ids):
    clone_ids = np.asarray(clone_ids)
    uniq, inverse = np.unique(clone_ids, return_inverse=True)
    clone_count, gene_count = len(uniq), expr.shape[1]
    sums = np.zeros((clone_count, gene_count), dtype=np.float64)
    np.add.at(sums, inverse, expr.astype(np.float64))
    counts = np.bincount(inverse).astype(np.float64)
    return (sums / np.maximum(counts, 1.0)[:, None]).astype(np.float32), uniq


def corr_cols(A, B, eps=1e-12):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    centered_a = A - A.mean(axis=0, keepdims=True)
    centered_b = B - B.mean(axis=0, keepdims=True)
    cov = (centered_a * centered_b).sum(axis=0)
    denom = np.sqrt((centered_a * centered_a).sum(axis=0) * (centered_b * centered_b).sum(axis=0)) + eps
    corr = cov / denom
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


class RegDataset(Dataset):
    def __init__(self, X_in, mask_in, tgt_cell_idx, labels_tgt, clone_ids, X_expr):
        self.X = torch.from_numpy(X_in.astype(np.float32))
        self.kpm = None if mask_in is None else torch.from_numpy(mask_in.astype(np.int8) == 0)
        self.tgt = tgt_cell_idx.astype(np.int64)
        self.lbl = labels_tgt.astype(object)
        self.clone = clone_ids.astype(object)
        self.X_expr = X_expr

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        y = torch.from_numpy(self.X_expr[self.tgt[index]].astype(np.float32))
        if self.kpm is None:
            return self.X[index], None, y, self.clone[index], self.lbl[index], self.tgt[index]
        return self.X[index], self.kpm[index].to(torch.bool), y, self.clone[index], self.lbl[index], self.tgt[index]


def build_loaders(X_in, mask_in, tgt_cell_idx, labels_tgt, clone_ids, X_expr, idx_tr, idx_va, idx_te, batch_size):
    eps = 1e-6

    if mask_in is None:
        mu = X_in[idx_tr].mean(axis=(0, 1))
        sd = X_in[idx_tr].std(axis=(0, 1)) + eps

        def normalize(X):
            return (X - mu) / sd

        def build_dataloader(indices, shuffle):
            dataset = RegDataset(normalize(X_in[indices]), None, tgt_cell_idx[indices], labels_tgt[indices], clone_ids[indices], X_expr)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    else:
        mask_train = mask_in[idx_tr].astype(np.float32)
        observed = float(mask_train.sum())
        if observed <= 0:
            raise ValueError("No observed inputs in training set after mask filter.")

        mu = (X_in[idx_tr] * mask_train[:, :, None]).sum(axis=(0, 1)) / (observed + eps)
        var = ((X_in[idx_tr] - mu) * mask_train[:, :, None]) ** 2
        sd = np.sqrt(var.sum(axis=(0, 1)) / (observed + eps)) + eps

        def normalize(X, mask_local):
            X_norm = (X - mu) / sd
            return X_norm * mask_local[:, :, None]

        def build_dataloader(indices, shuffle):
            mask_local = mask_in[indices].astype(np.float32)
            dataset = RegDataset(
                normalize(X_in[indices], mask_local),
                mask_in[indices],
                tgt_cell_idx[indices],
                labels_tgt[indices],
                clone_ids[indices],
                X_expr,
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    return (
        build_dataloader(idx_tr, True),
        build_dataloader(idx_va, False),
        build_dataloader(idx_te, False),
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def build_tasks_from_timepoints(timepoints, mode: str):
    if len(timepoints) < 2:
        raise ValueError(f"Need at least 2 timepoints, got T={len(timepoints)}")
    target_pos = len(timepoints) - 1

    def task_name(input_positions):
        target = sanitize_name(timepoints[target_pos])
        inputs = [sanitize_name(timepoints[pos]) for pos in input_positions]
        if len(inputs) > 4:
            return f"Reg_{target}_from_{inputs[0]}_to_{inputs[-1]}"
        return f"Reg_{target}_from_" + "_".join(inputs)

    if mode == "all_prev_only":
        input_positions = list(range(len(timepoints) - 1))
        return [(task_name(input_positions), input_positions, target_pos)]
    if mode == "each_prefix":
        tasks = []
        for idx in range(len(timepoints) - 1):
            input_positions = list(range(idx + 1))
            tasks.append((task_name(input_positions), input_positions, target_pos))
        return tasks
    raise ValueError(f"Unknown tasks_mode: {mode}")
