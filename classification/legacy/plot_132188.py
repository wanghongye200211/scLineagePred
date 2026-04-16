# -*- coding: utf-8 -*-
"""
Click-run plotting script for GSE132188 (4-class):

- Select settings by K_LIST + INCLUDE_ALLDAY
- For each selected setting:
  1) load base models + stacking model
  2) evaluate on fixed split (seed=2026)
  3) save OvR ROC (all classes in one figure)
  4) save OvR PR curve (all classes in one figure)
  5) save normalized confusion matrix
  6) save per-class Precision/Recall/F1 bar chart
- Also save one combined Macro ROC across selected settings
"""

import os
import pickle
import numpy as np
import h5py
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    log_loss,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize


# =========================
# Config
# =========================
SEED = 2026

PROCESSED_DIR = "/Users/wanghongye/python/scLineagetracer/GSE132188/processed"
OUT_PREFIX = "GSE132188_DeepLineage_all_generated"

OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188"
SAVED_MODELS_DIR = os.path.join(OUT_DIR, "saved_models")

# selected settings (0-based k over time labels)
K_LIST = [1, 2]
INCLUDE_ALLDAY = True

# split (must match training script)
TEST_FRAC = 0.10
VAL_FRAC = 0.10

# must match class_132188.py
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.10
NHEAD = 4

BATCH_SIZE = 512

SAVE_PER_SETTING_OVR = True
SAVE_PER_SETTING_PR = True
SAVE_PER_SETTING_CM = True
SAVE_PER_SETTING_CLASS_METRICS = True

CLASS_COLOR = {
    "Alpha": "#4C72B0",
    "Beta": "#DD8452",
    "Delta": "#55A868",
    "Epsilon": "#C44E52",
}

LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e", "#7f7f7f"]
GREY = "#444444"


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_all_seeds(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_reverse_from_samples_order(index_csv: str, T: int) -> bool:
    if not os.path.isfile(index_csv):
        return False
    try:
        df = pd.read_csv(index_csv, usecols=["samples_order"])
    except Exception:
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
            pass
        return vals[0] > vals[-1]
    return False


# =========================
# Load dataset
# =========================
def load_h5_dataset(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)

        if "classes" in f:
            classes = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["classes"][:]]
        else:
            classes = [f"class{i}" for i in range(int(y.max()) + 1)]

        if "time_labels" in f:
            time_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["time_labels"][:]]
        else:
            time_labels = [f"t{i}" for i in range(X.shape[1])]

    return X, y, classes, time_labels


def stratified_split(y: np.ndarray, seed: int, test_frac: float, val_frac: float):
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
    rel_val = val_frac / max(1e-9, (1.0 - test_frac))
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=rel_val, random_state=seed, stratify=y[tr_idx])
    return tr_idx2.astype(np.int64), va_idx.astype(np.int64), te_idx.astype(np.int64)


# =========================
# Truncation + padding
# =========================
class SeqDataset(Dataset):
    def __init__(self, X, y, idx, keep_len: int):
        self.X = X
        self.y = y
        self.idx = idx
        self.keep_len = int(keep_len)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j][: self.keep_len]
        return torch.from_numpy(x), torch.tensor(int(self.y[j]), dtype=torch.long), torch.tensor(self.keep_len, dtype=torch.long)


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
        self.lstm = nn.LSTM(d, h, n_layers, batch_first=True, bidirectional=True,
                            dropout=(dropout if n_layers > 1 else 0.0))
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
        self.rnn = nn.RNN(d, h, n_layers, batch_first=True,
                          dropout=(dropout if n_layers > 1 else 0.0))
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
        self.blocks = nn.ModuleList([TransformerBlock(d, nhead, ff_mult=2, dropout=dropout) for _ in range(n_layers)])
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
        feat = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(feat)


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
# ROC plotting
# =========================
def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)

    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)

    mask_one = (fpr == 1.0)
    if np.any(mask_one):
        t1 = float(np.max(tpr[mask_one]))
        keep = ~mask_one
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]

    return fpr, tpr


def _style_axes(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def plot_roc_all_classes_one_fig(y_true, y_prob, class_names, title, out_no_ext):
    C = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(C))

    fig, ax = plt.subplots(figsize=(6.8, 6.1))

    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr, tpr = _roc_endpoints_clean(fpr, tpr)
        a = auc(fpr, tpr)
        col = CLASS_COLOR.get(cname, LINE_COLORS[i % len(LINE_COLORS)])
        ax.plot(fpr, tpr, lw=2.8, color=col, label=f"{cname} (AUC={a:.3f})")

    ax.plot([0, 1], [0, 1], lw=1.5, color=GREY, ls="--", alpha=0.55)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _style_axes(ax)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def compute_macro_roc(y_true, y_prob, n_classes: int):
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr_list, tpr_list = [], []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr, tpr = _roc_endpoints_clean(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= float(n_classes)

    all_fpr, mean_tpr = _roc_endpoints_clean(all_fpr, mean_tpr)
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, float(macro_auc)


def plot_macro_multi_settings(macro_dict, title, out_no_ext):
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.plot([0, 1], [0, 1], lw=1.4, color=GREY, ls="--", alpha=0.5)

    settings = list(macro_dict.keys())
    for i, s in enumerate(settings):
        fpr, tpr, a = macro_dict[s]
        ax.plot(fpr, tpr, lw=2.7, color=LINE_COLORS[i % len(LINE_COLORS)], label=f"{s} (AUC={a:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style_axes(ax)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


# =========================
# PR / Confusion / Per-class metrics
# =========================
def plot_pr_all_classes_one_fig(y_true, y_prob, class_names, title, out_no_ext):
    C = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(C))

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    for i, cname in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        col = CLASS_COLOR.get(cname, LINE_COLORS[i % len(LINE_COLORS)])
        ax.plot(rec, prec, lw=2.8, color=col, label=f"{cname} (AP={ap:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    _style_axes(ax)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_confusion_matrix_normalized(y_true, y_pred, class_names, title, out_no_ext):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cmn = cm / np.maximum(row_sum, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis="both", direction="out", top=False, right=False)

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            v = cmn[i, j]
            txt = f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color=("white" if v > 0.5 else "black"), fontsize=15)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_per_class_metrics(y_true, y_pred, class_names, title, out_no_ext):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    x = np.arange(len(class_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.bar(x - w, p, width=w, label="Precision", color="#4C72B0")
    ax.bar(x, r, width=w, label="Recall", color="#55A868")
    ax.bar(x + w, f1, width=w, label="F1", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    return p, r, f1


# =========================
# Main
# =========================
def main():
    set_all_seeds(SEED)
    device = pick_device()
    print(f"[INFO] device={device}, seed={SEED}")

    h5_path = os.path.join(PROCESSED_DIR, f"{OUT_PREFIX}_sequences.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"[ERROR] missing h5: {h5_path}")

    X, y, class_names, time_labels = load_h5_dataset(h5_path)
    N, T, D = X.shape

    index_csv = os.path.join(PROCESSED_DIR, f"{OUT_PREFIX}_index.csv")
    if infer_reverse_from_samples_order(index_csv, T):
        X = X[:, ::-1, :].copy()
        print("[INFO] Detected reverse samples_order; flipped X time axis to chronological order.")

    print(f"[INFO] X: N={N}, T={T}, D={D}")
    print(f"[INFO] classes={class_names}")
    print(f"[INFO] time_labels={time_labels}")

    settings = []
    if not isinstance(K_LIST, (list, tuple)) or len(K_LIST) == 0:
        raise ValueError("[ERROR] K_LIST must be a non-empty list like [0,2].")

    for k in K_LIST:
        if not isinstance(k, int):
            raise TypeError(f"[ERROR] K_LIST contains non-int: {k} ({type(k)})")
        if k < 0 or k >= T:
            raise ValueError(f"[ERROR] k out of range: {k}, T={T}")
        settings.append((f"UpTo_{time_labels[k]}", k + 1))

    if INCLUDE_ALLDAY:
        settings.append((f"All_{time_labels[-1]}", T))

    print("[INFO] Selected settings:")
    for s, L in settings:
        print(f"  - {s} (keep_len={L})")

    _, _, te_idx = stratified_split(y, seed=SEED, test_frac=TEST_FRAC, val_frac=VAL_FRAC)

    roc_dir = os.path.join(OUT_DIR, "roc")
    upto_dir = os.path.join(roc_dir, "uptoday")
    pr_dir = os.path.join(OUT_DIR, "pr")
    cm_dir = os.path.join(OUT_DIR, "confusion")
    metrics_dir = os.path.join(OUT_DIR, "metrics")
    ensure_dir(roc_dir)
    ensure_dir(upto_dir)
    ensure_dir(pr_dir)
    ensure_dir(cm_dir)
    ensure_dir(metrics_dir)

    def load_model(kind: str, setting: str):
        if kind == "BiLSTM":
            m = BiLSTMModel(D, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(class_names))
        elif kind == "RNN":
            m = RNNModel(D, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(class_names))
        elif kind == "Trans":
            m = TransformerModel(D, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NHEAD, len(class_names))
        else:
            raise ValueError(kind)

        pth = os.path.join(SAVED_MODELS_DIR, f"{setting}_{kind}_s{SEED}.pth")
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"[ERROR] missing model: {pth}")
        sd = torch.load(pth, map_location="cpu")
        m.load_state_dict(sd)
        m.to(device)
        m.eval()
        return m

    macro_dict = {}
    metric_rows = []

    for setting, keep_len in settings:
        print(f"\n[RUN] setting={setting}, keep_len={keep_len}")

        te_loader = DataLoader(
            SeqDataset(X, y, te_idx, keep_len),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pad,
        )

        m_bilstm = load_model("BiLSTM", setting)
        m_rnn = load_model("RNN", setting)
        m_trans = load_model("Trans", setting)

        p1, y_te = get_probs(m_bilstm, te_loader, device)
        p2, _ = get_probs(m_rnn, te_loader, device)
        p3, _ = get_probs(m_trans, te_loader, device)
        X_stack = np.concatenate([p1, p2, p3], axis=1)

        pkl = os.path.join(SAVED_MODELS_DIR, f"{setting}_Stacking_s{SEED}.pkl")
        if not os.path.isfile(pkl):
            raise FileNotFoundError(f"[ERROR] missing stacking pkl: {pkl}")
        with open(pkl, "rb") as f:
            stacker = pickle.load(f)

        p_stack = stacker.predict_proba(X_stack)
        y_pred = p_stack.argmax(axis=1)

        acc = accuracy_score(y_te, y_pred)
        ll = log_loss(y_te, p_stack, labels=list(range(len(class_names))))
        print(f"[RESULT] {setting} | Acc={acc:.4f} | LogLoss={ll:.5f}")

        fpr, tpr, mauc = compute_macro_roc(y_te, p_stack, n_classes=len(class_names))
        macro_dict[setting] = (fpr, tpr, mauc)

        if SAVE_PER_SETTING_OVR:
            plot_roc_all_classes_one_fig(
                y_true=y_te,
                y_prob=p_stack,
                class_names=class_names,
                title=f"Stacking ROC (OvR) - {setting} - seed={SEED}",
                out_no_ext=os.path.join(upto_dir, f"ROC_{setting}_AllClasses"),
            )

        if SAVE_PER_SETTING_PR:
            plot_pr_all_classes_one_fig(
                y_true=y_te,
                y_prob=p_stack,
                class_names=class_names,
                title=f"Stacking PR (OvR) - {setting} - seed={SEED}",
                out_no_ext=os.path.join(pr_dir, f"PR_{setting}_AllClasses"),
            )

        if SAVE_PER_SETTING_CM:
            plot_confusion_matrix_normalized(
                y_true=y_te,
                y_pred=y_pred,
                class_names=class_names,
                title=f"Confusion Matrix (Normalized) - {setting}",
                out_no_ext=os.path.join(cm_dir, f"CM_{setting}_Norm"),
            )

        if SAVE_PER_SETTING_CLASS_METRICS:
            p_cls, r_cls, f1_cls = plot_per_class_metrics(
                y_true=y_te,
                y_pred=y_pred,
                class_names=class_names,
                title=f"Per-class Metrics - {setting}",
                out_no_ext=os.path.join(metrics_dir, f"PerClass_{setting}"),
            )
            for i, cname in enumerate(class_names):
                metric_rows.append({
                    "setting": setting,
                    "class": cname,
                    "precision": float(p_cls[i]),
                    "recall": float(r_cls[i]),
                    "f1": float(f1_cls[i]),
                })

    out_no_ext = os.path.join(roc_dir, f"ROC_Macro_SelectedSettings_Seed{SEED}")
    title = f"Stacking Macro ROC (Selected UpTo + AllDay) - seed={SEED}"
    plot_macro_multi_settings(macro_dict, title, out_no_ext)

    if metric_rows:
        out_csv = os.path.join(metrics_dir, "per_class_metrics.csv")
        pd.DataFrame(metric_rows).to_csv(out_csv, index=False)
        print(f"[DONE] Per-class metrics CSV saved: {out_csv}")

    print(f"\n[DONE] Combined Macro ROC saved: {out_no_ext}.png/pdf")
    print(f"[DONE] Per-setting OvR saved to: {upto_dir}")
    print(f"[DONE] Per-setting PR saved to: {pr_dir}")
    print(f"[DONE] Per-setting confusion matrices saved to: {cm_dir}")
    print(f"[DONE] Per-setting class metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
