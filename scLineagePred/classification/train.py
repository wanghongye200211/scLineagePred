# -*- coding: utf-8 -*-
"""
Unified sequence classification training for scLineagePred.

- Inputs: DeepLineage/trajectory sequences in H5 format plus optional index CSV
- Supports binary and multi-class endpoint prediction
- Base models: BiLSTM / RNN / Transformer
- Stacking: LogisticRegression
- Outputs: per-setting ROC, macro ROC, performance trend, and probability-space PCA plots
"""

import argparse
import os
import random
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================
# Config
# =========================
@dataclass
class Config:
    version: str = "unified-v1"

    # paths
    time_series_h5: str = ""
    index_csv: str = ""
    out_dir: str = "./outputs/classification"
    model_dir: str = ""

    # columns
    label_col: str = "label_str"
    clone_col: str = "clone_id"

    # classes (stable order). Empty means "use all labels found in data".
    target_labels: Tuple[str, ...] = ()

    # training
    base_seed: int = 2026
    batch_size: int = 512
    epochs: int = 600
    patience: int = 30
    min_delta: float = 1e-4
    lr: float = 1e-3

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    nhead: int = 4

    # stability
    grad_clip_norm: float = 1.0
    use_scheduler: bool = True
    lr_patience: int = 6
    lr_factor: float = 0.5
    min_lr: float = 1e-5

    # label smoothing
    label_smoothing: float = 0.0

    # dataloader
    num_workers: int = 0

    # stacking
    stack_max_iter: int = 4000
    stack_C: float = 0.8

    # split
    test_frac: float = 0.10
    val_frac: float = 0.10

    # device
    device_prefer: str = "auto"  # auto/cpu/mps/cuda

    # 3D Pred-only plot settings
    max_points_3d: int = 8000
    alpha_3d: float = 0.75
    size_3d: float = 24.0


# =========================
# Utilities
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_all_seeds(seed: int):
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
        return torch.device("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_plot(fig, path_no_ext: str):
    fig.savefig(f"{path_no_ext}.pdf", format="pdf", bbox_inches=None, pad_inches=0.0, dpi=300)
    fig.savefig(f"{path_no_ext}.png", format="png", bbox_inches=None, pad_inches=0.0, dpi=300)
    print(f"   [Plot] Saved: {path_no_ext}.png")


def read_time_labels_from_h5(h5_path: str, T: int) -> List[str]:
    try:
        with h5py.File(h5_path, "r") as f:
            if "time_labels" in f:
                tl = f["time_labels"][:]
                out = []
                for x in tl:
                    if isinstance(x, (bytes, np.bytes_)):
                        out.append(x.decode("utf-8"))
                    else:
                        out.append(str(x))
                if len(out) == T:
                    return out
            if "timepoints" in f:
                tp = f["timepoints"][:]
                return [str(float(x)) for x in np.asarray(tp).tolist()]
            if "time_values" in f:
                tv = f["time_values"][:]
                return [str(float(x)) for x in np.asarray(tv).tolist()]
    except Exception:
        pass
    return [f"t{i}" for i in range(T)]


def ensure_roc_endpoints(fpr: np.ndarray, tpr: np.ndarray):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    if (fpr[0] > 0.0) or (tpr[0] > 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (fpr[-1] < 1.0) or (tpr[-1] < 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    fpr = np.clip(fpr, 0.0, 1.0)
    tpr = np.clip(tpr, 0.0, 1.0)
    return fpr, tpr


def infer_reverse_from_samples_order(df: pd.DataFrame, T: int) -> bool:
    """
    Infer whether sequence axis is reversed from 'samples_order' column.
    Example reverse: \"3,2,1,0\" for T=4.
    """
    if "samples_order" not in df.columns:
        return False

    ser = df["samples_order"].dropna().astype(str)
    for raw in ser.head(200):
        toks = [t.strip() for t in raw.split(",") if t.strip() != ""]
        vals = []
        ok = True
        for t in toks:
            if t.lstrip("-").isdigit():
                vals.append(int(t))
            else:
                ok = False
                break
        if not ok or len(vals) < 2:
            continue
        if T > 0 and len(vals) != T:
            # still usable for direction, continue unless clearly malformed
            pass
        return vals[0] > vals[-1]
    return False


# =========================
# Data
# =========================
def load_data(cfg: Config):
    print(f"[INFO] Loading H5: {cfg.time_series_h5}")
    with h5py.File(cfg.time_series_h5, "r") as f:
        X_all = np.array(f["X"], dtype=np.float32)  # (N,T,D)

        y_h5 = None
        if "y" in f:
            y_h5 = np.asarray(f["y"], dtype=np.int64)

        label_h5 = None
        if "label_str" in f:
            raw = f["label_str"][:]
            label_h5 = np.array([
                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in raw
            ], dtype=object)

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
        # Convert sequence axis to chronological order (early -> late) before UpTo_* truncation.
        X_all = X_all[:, ::-1, :].copy()
        print("[INFO] Detected reverse samples_order; flipped X time axis to chronological order.")

    selected_labels = list(cfg.target_labels)
    if not selected_labels:
        selected_labels = list(pd.Index(labels).drop_duplicates())

    keep = np.isin(labels, np.array(selected_labels, dtype=object))
    X = X_all[keep]
    labels_kept = labels[keep]
    clones_kept = clones[keep]

    class_names = list(selected_labels)
    if len(class_names) < 2:
        raise ValueError(f"Need at least 2 target labels for classification, got: {class_names}")
    label_to_y = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_y[s] for s in labels_kept], dtype=np.int64)

    print(f"[INFO] Total sequences: {len(X_all)}")
    print(f"[INFO] Kept sequences: {len(X)}")
    for c in class_names:
        print(f"  - {c}: {(labels_kept == c).sum()}")

    T = X.shape[1]
    time_labels = read_time_labels_from_h5(cfg.time_series_h5, T)
    print(f"[INFO] T={T}, time_labels={time_labels}")

    if y_h5 is not None and len(y_h5) == len(X_all):
        y_h5_kept = y_h5[keep]
        mismatch = int((y_h5_kept != y).sum())
        if mismatch > 0:
            print(f"[WARN] y from H5 differs from label mapping in {mismatch} rows; using mapped labels from strings.")

    return X, y, clones_kept, class_names, time_labels


def stratified_split(y: np.ndarray, seed: int, test_frac: float, val_frac: float):
    all_idx = np.arange(len(y))
    try:
        tr_idx, te_idx = train_test_split(all_idx, test_size=test_frac, random_state=seed, stratify=y)
        rel_val = val_frac / max(1e-9, (1.0 - test_frac))
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=rel_val, random_state=seed, stratify=y[tr_idx])
    except ValueError:
        print("[WARN] Stratified split failed; fallback to random split.")
        tr_idx, te_idx = train_test_split(all_idx, test_size=test_frac, random_state=seed, stratify=None)
        rel_val = val_frac / max(1e-9, (1.0 - test_frac))
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=rel_val, random_state=seed, stratify=None)
    return tr_idx2.astype(np.int64), va_idx.astype(np.int64), te_idx.astype(np.int64)


def build_time_settings(time_labels):
    """
    settings[name] = keep_len (int)
    Example:
      UpTo_12.5 -> 1
      UpTo_13.5 -> 2
      UpTo_14.5 -> 3
      All_15.5  -> T
    """
    T = len(time_labels)
    settings = {}
    order = []
    x_labels = []

    for k in range(T - 1):
        end_lab = time_labels[k]
        name = f"UpTo_{end_lab}"
        keep_len = k + 1
        settings[name] = keep_len
        order.append(name)
        x_labels.append(str(end_lab))

    name_all = f"All_{time_labels[-1]}"
    settings[name_all] = T
    order.append(name_all)
    x_labels.append(str(time_labels[-1]))

    return settings, order, x_labels


class SeqDataset(Dataset):
    def __init__(self, X, y, idx, keep_len=None):
        self.X = X
        self.y = y
        self.idx = idx
        self.keep_len = keep_len

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j]
        if self.keep_len is None:
            x2 = x
            L = x.shape[0]
        else:
            L = int(self.keep_len)
            x2 = x[:L]
        return (
            torch.from_numpy(np.asarray(x2, dtype=np.float32)),
            torch.tensor(int(self.y[j]), dtype=torch.long),
            torch.tensor(L, dtype=torch.long),
        )


def collate_pad(batch):
    xs, ys, lens = zip(*batch)
    lens = torch.stack(lens, dim=0)
    ys = torch.stack(ys, dim=0)

    max_len = int(lens.max().item())
    d = xs[0].shape[1]

    Xp = torch.zeros((len(xs), max_len, d), dtype=torch.float32)
    for i, x in enumerate(xs):
        L = x.shape[0]
        Xp[i, :L] = x
    return Xp, ys, lens


# =========================
# Models
# =========================
class BiLSTMModel(nn.Module):
    def __init__(self, d, h, n_layers, dropout, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            d, h, n_layers, batch_first=True, bidirectional=True,
            dropout=(dropout if n_layers > 1 else 0.0)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )

    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(feat)


class RNNModel(nn.Module):
    def __init__(self, d, h, n_layers, dropout, n_classes):
        super().__init__()
        self.rnn = nn.RNN(
            d, h, n_layers, batch_first=True,
            dropout=(dropout if n_layers > 1 else 0.0)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )

    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(h[-1])


class TransformerBlock(nn.Module):
    def __init__(self, d, nhead, ff_mult=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ff_mult, d),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, d, ff, n_layers, dropout, nhead, n_classes):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d, nhead, ff_mult=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, n_classes),
        )

    def forward(self, x, lengths):
        B, T, _ = x.shape
        ar = torch.arange(T, device=x.device)[None, :].expand(B, T)
        key_padding_mask = ar >= lengths[:, None]

        h = x
        for blk in self.blocks:
            h = blk(h, key_padding_mask)

        mask = (~key_padding_mask).float().unsqueeze(-1)
        h_sum = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        feat = h_sum / denom
        return self.head(feat)


# =========================
# Training
# =========================
def train_base_model(model: nn.Module, tr_loader: DataLoader, va_loader: DataLoader,
                     device: torch.device, cfg: Config, name: str):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if cfg.label_smoothing > 0:
        try:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        except TypeError:
            print("[WARN] Current torch version does not support label_smoothing; fallback to 0.")
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.min_lr,
            verbose=False
        )

    best_val = float("inf")
    best_state = None
    pat = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n--- Training {name} ---")
    print(f"{'Epoch':<5} | {'TrLoss':<10} | {'VaLoss':<10} | {'VaAcc':<8} | {'LR':<10} | {'Pat':<4}")

    for ep in range(cfg.epochs):
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0

        for x, y, lengths in tr_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            tr_loss_sum += float(loss.item()) * x.size(0)
            tr_n += x.size(0)

        tr_loss = tr_loss_sum / max(1, tr_n)

        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        correct = 0
        with torch.no_grad():
            for x, y, lengths in va_loader:
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)

                logits = model(x, lengths)
                loss = loss_fn(logits, y)

                va_loss_sum += float(loss.item()) * x.size(0)
                va_n += x.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())

        va_loss = va_loss_sum / max(1, va_n)
        va_acc = correct / max(1, va_n)

        if scheduler is not None:
            scheduler.step(va_loss)

        cur_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(cur_lr)

        improved = (va_loss < (best_val - cfg.min_delta))
        if improved:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        print(f"{ep+1:03d}   | {tr_loss:<10.4f} | {va_loss:<10.4f} | {va_acc:<8.4f} | {cur_lr:<10.2e} | {pat:<4d}")

        if pat >= cfg.patience:
            print(f"[EarlyStop] {name} stopped at epoch {ep+1} (best val_loss={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def get_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, targs = [], []
    for x, y, lengths in loader:
        logits = model(x.to(device), lengths.to(device))
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        targs.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(targs, axis=0)


# =========================
# ROC plots
# =========================
def plot_setting_roc_ovr_macro_style(y_true: np.ndarray, y_prob: np.ndarray,
                                     class_names: List[str], setting: str, out_dir: str):
    C = len(class_names)

    color_map = {
        "Alpha": "#4C72B0",
        "Beta": "#DD8452",
        "Delta": "#55A868",
        "Epsilon": "#C44E52",
    }
    fallback = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(np.int64)
        if np.unique(y_bin).size < 2:
            print(f"[WARN] Skip ROC for {setting}/{cname}: test split contains only one class.")
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        fpr, tpr = ensure_roc_endpoints(fpr, tpr)
        auc_i = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=3, color=color_map.get(cname, fallback[i % len(fallback)]), label=f"AUC={auc_i:.4f}")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.margins(x=0.0, y=0.0)
        ax.set_xmargin(0.0)
        ax.set_ymargin(0.0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(False)

        ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
        ax.set_title(f"ROC - {setting} - {cname}", fontsize=14, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=12, frameon=False)

        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
        save_plot(fig, os.path.join(out_dir, f"ROC_{setting}_{cname}"))
        plt.close(fig)

    fpr_list, tpr_list = [], []
    for i in range(C):
        y_bin = (y_true == i).astype(np.int64)
        if np.unique(y_bin).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        fpr, tpr = ensure_roc_endpoints(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    if not fpr_list:
        raise ValueError(f"Unable to compute ROC curves for setting {setting}: every class is missing in the test split.")

    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= float(C)
    all_fpr, mean_tpr = ensure_roc_endpoints(all_fpr, mean_tpr)
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(all_fpr, mean_tpr, lw=3, color="#777777", label=f"Macro AUC={macro_auc:.4f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.0, y=0.0)
    ax.set_xmargin(0.0)
    ax.set_ymargin(0.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)

    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title(f"ROC - {setting} - Macro", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
    save_plot(fig, os.path.join(out_dir, f"ROC_{setting}_Macro"))
    plt.close(fig)

    return all_fpr, mean_tpr, macro_auc


def plot_macro_roc_all_settings_style(macro_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                                      order: List[str], out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#E24A33", "#348ABD", "#988ED5", "#55A868", "#DD8452", "#777777"]

    for i, s in enumerate(order):
        if s not in macro_curves:
            continue
        fpr_s, tpr_s, auc_s = macro_curves[s]
        fpr_s, tpr_s = ensure_roc_endpoints(fpr_s, tpr_s)
        ax.plot(fpr_s, tpr_s, lw=3, color=colors[i % len(colors)], label=f"{s} (AUC={auc_s:.3f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.0, y=0.0)
    ax.set_xmargin(0.0)
    ax.set_ymargin(0.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)

    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title("Macro-average ROC (Ensemble) Across Timepoints", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
    save_plot(fig, os.path.join(out_dir, "ROC_Macro_AllTimepoints_Ensemble"))
    plt.close(fig)


def plot_performance_trend(results_buffer: Dict[str, dict], order: List[str], x_labels: List[str], out_dir: str):
    xs, accs, losses = [], [], []
    for s, xl in zip(order, x_labels):
        if s in results_buffer:
            xs.append(xl)
            accs.append(results_buffer[s]["acc"])
            losses.append(results_buffer[s]["loss"])

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(xs, accs, marker="o", markersize=10, lw=3)
    ax2 = ax1.twinx()
    ax2.plot(xs, losses, marker="^", markersize=10, lw=3)

    ax1.set_xlabel("Observation End Point", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Log Loss", fontsize=14, fontweight="bold")

    ax1.tick_params(direction="in", top=True)
    ax2.tick_params(direction="in", right=True)
    ax1.grid(False)
    ax2.grid(False)

    plt.title("Performance Trend (Ensemble)", fontsize=16, fontweight="bold", pad=15)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.90)
    save_plot(fig, os.path.join(out_dir, "Performance_Trend"))
    plt.close(fig)


# =========================
# 3D Probabilities (PCA)
# =========================
def plot_3d_pred_points_pca_true_coded(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    out_prefix: str,
    max_points: int,
    seed: int,
    alpha: float,
    size: float,
):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.int64)
    P = np.asarray(y_prob, dtype=np.float32)

    s = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(s, 1e-12)

    N = len(y_true)
    idx = np.arange(N, dtype=np.int64)

    if N > max_points:
        keep_idx = []
        for k in range(len(class_names)):
            ik = idx[y_true == k]
            if len(ik) == 0:
                continue
            nk = int(round(max_points * (len(ik) / float(N))))
            nk = max(1, min(nk, len(ik)))
            keep_idx.append(rng.choice(ik, size=nk, replace=False))
        keep_idx = np.concatenate(keep_idx) if len(keep_idx) else rng.choice(idx, size=max_points, replace=False)
        keep_idx = np.unique(keep_idx)
        idx = keep_idx
        y_true = y_true[idx]
        P = P[idx]

    n_comp = max(1, min(3, P.shape[1], P.shape[0]))
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(P)
    if Z.shape[1] < 3:
        Z = np.pad(Z, ((0, 0), (0, 3 - Z.shape[1])), mode="constant")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    markers = ["o", "^", "s", "D", "P", "X"]

    fig = plt.figure(figsize=(9, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    for k, cname in enumerate(class_names):
        m = (y_true == k)
        if m.sum() == 0:
            continue
        ax.scatter(
            Z[m, 0], Z[m, 1], Z[m, 2],
            s=size,
            c=colors[k % len(colors)],
            marker=markers[k % len(markers)],
            alpha=alpha,
            edgecolors="k",
            linewidths=0.15,
            label=f"True={cname} (n={int(m.sum())})"
        )

    ax.set_xlabel("PC1", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("PC2", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_zlabel("PC3", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor("black")
        except Exception:
            pass
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    save_plot(fig, out_prefix)
    plt.close(fig)

    tab = pd.DataFrame({
        "y_true": y_true.astype(np.int64),
        "label_true": np.array([class_names[i] for i in y_true], dtype=object),
        "pc1": Z[:, 0],
        "pc2": Z[:, 1],
        "pc3": Z[:, 2],
        "p_max": P.max(axis=1),
        "pred_label": np.array([class_names[int(i)] for i in P.argmax(axis=1)], dtype=object),
    })
    for k, cname in enumerate(class_names):
        tab[f"p_{cname}"] = P[:, k]

    tab.to_csv(out_prefix + "_pred3d_table.csv", index=False)
    print(f"   [Table] Saved: {out_prefix}_pred3d_table.csv")


def run(cfg: Config):
    if not cfg.model_dir:
        cfg.model_dir = os.path.join(cfg.out_dir, "saved_models")
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model_dir)

    X, y, clones, class_names, time_labels = load_data(cfg)
    C = len(class_names)

    device = pick_device(cfg)
    print(f"[INFO] Device: {device}")

    settings, setting_order, x_labels = build_time_settings(time_labels)
    print(
        f"[INFO] Version={cfg.version} | random-split | "
        f"hidden={cfg.hidden_dim} dropout={cfg.dropout} ls={cfg.label_smoothing}"
    )

    results_buffer: Dict[str, dict] = {}
    macro_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    print("\n" + "=" * 60)
    print("PHASE 1: Train base models + stacking for each timepoint setting")
    print("=" * 60)

    for setting in setting_order:
        keep_len = int(settings[setting])
        seed = cfg.base_seed
        set_all_seeds(seed)

        print(f"\n>>> Setting: {setting} | seed={seed} | keep_len={keep_len}")
        tr_idx, va_idx, te_idx = stratified_split(y, seed, cfg.test_frac, cfg.val_frac)

        tr_loader = DataLoader(
            SeqDataset(X, y, tr_idx, keep_len),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_pad,
        )
        va_loader = DataLoader(
            SeqDataset(X, y, va_idx, keep_len),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_pad,
        )
        te_loader = DataLoader(
            SeqDataset(X, y, te_idx, keep_len),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_pad,
        )

        d = X.shape[2]
        models = {
            "BiLSTM": BiLSTMModel(d, cfg.hidden_dim, cfg.num_layers, cfg.dropout, C),
            "RNN": RNNModel(d, cfg.hidden_dim, cfg.num_layers, cfg.dropout, C),
            "Trans": TransformerModel(d, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead, C),
        }

        val_feats = []
        test_feats = []

        for name, model in models.items():
            model, _ = train_base_model(
                model=model,
                tr_loader=tr_loader,
                va_loader=va_loader,
                device=device,
                cfg=cfg,
                name=name,
            )

            p_val, _ = get_probs(model, va_loader, device)
            p_te, _ = get_probs(model, te_loader, device)
            val_feats.append(p_val)
            test_feats.append(p_te)

            torch.save(model.state_dict(), os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth"))

        X_v = np.concatenate(val_feats, axis=1)
        X_t = np.concatenate(test_feats, axis=1)
        y_v = y[va_idx]
        y_t = y[te_idx]

        print("\n--- Training Stacking (LogReg) ---")
        lr = LogisticRegression(
            random_state=seed,
            max_iter=cfg.stack_max_iter,
            C=cfg.stack_C,
            multi_class="auto",
            solver="lbfgs"
        )
        lr.fit(X_v, y_v)
        p_stack = lr.predict_proba(X_t)

        y_pred = np.argmax(p_stack, axis=1)
        acc = accuracy_score(y_t, y_pred)
        loss = log_loss(y_t, p_stack, labels=list(range(C)))
        print(f"   [Result] {setting} | Acc={acc:.4f} | LogLoss={loss:.4f}")

        with open(os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl"), "wb") as f:
            pickle.dump(lr, f)

        results_buffer[setting] = {
            "seed": seed,
            "y_true": y_t,
            "y_prob": p_stack,
            "acc": acc,
            "loss": loss,
            "clones": clones[te_idx],
        }

    print("\n" + "=" * 60)
    print("PHASE 2: ROC + 3D + Summary")
    print("=" * 60)

    for setting in setting_order:
        res = results_buffer[setting]
        mfpr, mtpr, mauc = plot_setting_roc_ovr_macro_style(
            res["y_true"], res["y_prob"], class_names, setting, cfg.out_dir
        )
        macro_curves[setting] = (mfpr, mtpr, mauc)

    plot_macro_roc_all_settings_style(macro_curves, setting_order, cfg.out_dir)
    plot_performance_trend(results_buffer, setting_order, x_labels, cfg.out_dir)

    for setting in setting_order:
        res = results_buffer[setting]
        plot_3d_pred_points_pca_true_coded(
            y_true=res["y_true"],
            y_prob=res["y_prob"],
            class_names=class_names,
            title=f"{setting} | 3D Prob PCA (color+marker = TRUE label)",
            out_prefix=os.path.join(cfg.out_dir, f"Pred3D_{setting}_truecoded"),
            max_points=cfg.max_points_3d,
            seed=cfg.base_seed,
            alpha=cfg.alpha_3d,
            size=cfg.size_3d,
        )

    summary = []
    for setting in setting_order:
        res = results_buffer[setting]
        summary.append({
            "Setting": setting,
            "Seed": res["seed"],
            "Accuracy": res["acc"],
            "LogLoss": res["loss"],
            "N_test": int(len(res["y_true"])),
        })
    pd.DataFrame(summary).to_csv(os.path.join(cfg.out_dir, "ensemble_summary.csv"), index=False)
    print(f"   [CSV] Saved: {os.path.join(cfg.out_dir, 'ensemble_summary.csv')}")

    print(f"\n[DONE] All outputs saved to: {cfg.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified sequence classification training")
    parser.add_argument("--time-series-h5", required=True, help="Sequence H5 file with X and optional labels")
    parser.add_argument("--index-csv", default="", help="Optional sequence metadata CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--model-dir", default="", help="Model checkpoint directory (default: <out-dir>/saved_models)")
    parser.add_argument("--label-col", default="label_str", help="Label column name in index CSV")
    parser.add_argument("--clone-col", default="clone_id", help="Clone column name in index CSV")
    parser.add_argument(
        "--target-label",
        dest="target_labels",
        action="append",
        default=[],
        help="Endpoint label to keep. Repeat the flag to provide multiple labels.",
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--use-scheduler", action="store_true", default=True)
    parser.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stack-max-iter", type=int, default=4000)
    parser.add_argument("--stack-c", type=float, default=0.8)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--device", dest="device_prefer", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-points-3d", type=int, default=8000)
    parser.add_argument("--alpha-3d", type=float, default=0.75)
    parser.add_argument("--size-3d", type=float, default=24.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        time_series_h5=args.time_series_h5,
        index_csv=args.index_csv,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        label_col=args.label_col,
        clone_col=args.clone_col,
        target_labels=tuple(args.target_labels),
        base_seed=args.base_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nhead=args.nhead,
        grad_clip_norm=args.grad_clip_norm,
        use_scheduler=args.use_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        stack_max_iter=args.stack_max_iter,
        stack_C=args.stack_c,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        device_prefer=args.device_prefer,
        max_points_3d=args.max_points_3d,
        alpha_3d=args.alpha_3d,
        size_3d=args.size_3d,
    )


def main():
    args = parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
