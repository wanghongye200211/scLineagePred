# -*- coding: utf-8 -*-
"""
Official benchmark for remaining datasets:
- GSE114412 (UpTo_0, 3-class)
- GSE175634 (Obs_Day1, binary)
- GSE99915 (Obs_Day9, binary)

Compared methods:
- scLineagetracer
- CellRank (official)
- WOT (official)
- CoSpar (official)
"""

import os
import re
import argparse
import pickle
import warnings
import random
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

# runtime env for scanpy/cellrank/cospar on this machine
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import scanpy as sc
import cellrank as cr
import wot
import cospar as cs


GREY = "#444444"
LINE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_fig(fig, out_no_ext):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _style_axes_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)
    m1 = fpr == 1.0
    if np.any(m1):
        t1 = float(np.max(tpr[m1]))
        keep = ~m1
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]
    return fpr, tpr


def normalize_rows(P):
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, 1e-9, np.inf)
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


def _safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return np.nan
    return float(num) / den


def _auprc_trapz(y_true, p):
    precision, recall, _ = precision_recall_curve(y_true, p)
    # sklearn returns recall from 1 -> 0; reverse to make x increasing.
    return float(auc(recall[::-1], precision[::-1]))


def eval_binary(y_true, p_pos):
    p = np.clip(np.asarray(p_pos, dtype=np.float64), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(np.int64)
    fpr, tpr, _ = roc_curve(y_true, p)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    auroc = float(auc(fpr, tpr))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = np.nan
    return {
        "AUC": auroc,
        "AUROC": auroc,
        "AUPRC": float(_auprc_trapz(y_true, p)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "ACC": float(accuracy_score(y_true, pred)),
        "F1": float(f1) if np.isfinite(f1) else np.nan,
        "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "Precision": float(precision) if np.isfinite(precision) else np.nan,
        "Recall": float(recall) if np.isfinite(recall) else np.nan,
        "LogLoss": float(log_loss(y_true, p)),
        "fpr": fpr,
        "tpr": tpr,
    }


def compute_macro_curve_auc(y_true, prob):
    c = prob.shape[1]
    y_bin = label_binarize(y_true, classes=np.arange(c))
    grid = np.linspace(0.0, 1.0, 201)
    tprs = []
    aucs = []
    for i in range(c):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        fpr, tpr = _roc_endpoints_clean(fpr, tpr)
        aucs.append(float(auc(fpr, tpr)))
        tpr_i = np.interp(grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(auc(grid, mean_tpr))
    return grid, mean_tpr, macro_auc, aucs


def eval_multiclass(y_true, prob):
    p = normalize_rows(prob)
    pred = np.argmax(p, axis=1)
    grid, macro_tpr, macro_auc, aucs = compute_macro_curve_auc(y_true, p)
    return {
        "AUC_macro": macro_auc,
        "Accuracy": float(accuracy_score(y_true, pred)),
        "LogLoss": float(log_loss(y_true, p, labels=list(range(p.shape[1])))),
        "fpr_macro": grid,
        "tpr_macro": macro_tpr,
        "AUC_per_class": aucs,
    }


def plot_dual_binary(curves, order, out_no_ext, title_full):
    color = {m: LINE_COLORS[i % len(LINE_COLORS)] for i, m in enumerate(order)}

    def draw(ax, clean):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=GREY, alpha=0.55)
        for m in order:
            if m not in curves:
                continue
            c = curves[m]
            ax.plot(c["fpr"], c["tpr"], lw=2.7, color=color[m], label=f"{m} (AUC={c['AUC']:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        if clean:
            _style_axes_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.set_title(title_full, fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(False)

    fig1, ax1 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax1, clean=False)
    fig1.tight_layout()
    save_fig(fig1, out_no_ext + "_full")

    fig2, ax2 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax2, clean=True)
    fig2.tight_layout()
    save_fig(fig2, out_no_ext + "_clean")


def plot_dual_multiclass(curves, order, out_no_ext, title_full):
    color = {m: LINE_COLORS[i % len(LINE_COLORS)] for i, m in enumerate(order)}

    def draw(ax, clean):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=GREY, alpha=0.55)
        for m in order:
            if m not in curves:
                continue
            c = curves[m]
            ax.plot(c["fpr_macro"], c["tpr_macro"], lw=2.7, color=color[m], label=f"{m} (Macro AUC={c['AUC_macro']:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        if clean:
            _style_axes_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.set_title(title_full, fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(False)

    fig1, ax1 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax1, clean=False)
    fig1.tight_layout()
    save_fig(fig1, out_no_ext + "_full")

    fig2, ax2 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax2, clean=True)
    fig2.tight_layout()
    save_fig(fig2, out_no_ext + "_clean")


class SeqDatasetVarLen(Dataset):
    def __init__(self, X, y, idx, keep_len):
        self.X = X
        self.y = y
        self.idx = np.asarray(idx, dtype=np.int64)
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
        Xp[i, : x.shape[0]] = x
    return Xp, ys, lens


class BiLSTMVar(nn.Module):
    def __init__(self, d, h, n_layers, dropout, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(d, h, n_layers, batch_first=True, bidirectional=True, dropout=(dropout if n_layers > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_classes))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(feat)


class RNNVar(nn.Module):
    def __init__(self, d, h, n_layers, dropout, n_classes):
        super().__init__()
        self.rnn = nn.RNN(d, h, n_layers, batch_first=True, dropout=(dropout if n_layers > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_classes))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(h[-1])


class TransformerBlock(nn.Module):
    def __init__(self, d, nhead, ff_mult=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * ff_mult, d), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerVar(nn.Module):
    def __init__(self, d, ff, n_layers, dropout, nhead, n_classes):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d, nhead, ff_mult=2, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, n_classes))

    def forward(self, x, lengths):
        b, t, _ = x.shape
        ar = torch.arange(t, device=x.device)[None, :].expand(b, t)
        key_padding_mask = ar >= lengths[:, None]
        h = x
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        mask = (~key_padding_mask).float().unsqueeze(-1)
        feat = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(feat)


class SeqDatasetMask(Dataset):
    def __init__(self, X, y, idx, mask_t=None):
        self.X = X
        self.y = y
        self.idx = np.asarray(idx, dtype=np.int64)
        self.mask_t = list(mask_t) if mask_t is not None else []

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j].copy()
        for t in self.mask_t:
            x[t] = 0.0
        return torch.from_numpy(x), torch.tensor(int(self.y[j]), dtype=torch.long)


class LSTMFix(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True, bidirectional=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(0.5 * (h[-2] + h[-1]))


class RNNFix(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.rnn = nn.RNN(d, h, l, batch_first=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        return self.head(self.rnn(x)[1][-1])


class TransformerFix(nn.Module):
    def __init__(self, d, h, l, dr, nhead):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, dim_feedforward=h * 2, dropout=dr, batch_first=True),
            num_layers=l,
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x):
        return self.head(self.enc(x).mean(dim=1))


@torch.no_grad()
def get_probs_var(model, loader, device):
    model.eval()
    probs = []
    for x, _, lengths in loader:
        logits = model(x.to(device), lengths.to(device))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def get_prob_pos_fix(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        logits = model(x.to(device))
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs, axis=0)


def run_sclineage_114412():
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed/GSE114412_all_generated_sequences.h5"
    idx_csv = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed/GSE114412_all_generated_index.csv"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE114412/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)
        classes = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["classes"][:]]

    seed = 2026
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=seed, stratify=y)
    rel_val = 0.1 / (1.0 - 0.1)
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=rel_val, random_state=seed, stratify=y[tr_idx])
    _ = (tr_idx2, va_idx)

    ds = SeqDatasetVarLen(X, y, te_idx.astype(np.int64), keep_len=1)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_pad)

    device = torch.device("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    d = X.shape[2]
    c = len(classes)

    m1 = BiLSTMVar(d, 256, 2, 0.10, c).to(device)
    m2 = RNNVar(d, 256, 2, 0.10, c).to(device)
    m3 = TransformerVar(d, 256, 2, 0.10, 4, c).to(device)

    m1.load_state_dict(torch.load(os.path.join(model_dir, "UpTo_0_BiLSTM_s2026.pth"), map_location=device))
    m2.load_state_dict(torch.load(os.path.join(model_dir, "UpTo_0_RNN_s2026.pth"), map_location=device))
    m3.load_state_dict(torch.load(os.path.join(model_dir, "UpTo_0_Trans_s2026.pth"), map_location=device))

    p1 = get_probs_var(m1, dl, device)
    p2 = get_probs_var(m2, dl, device)
    p3 = get_probs_var(m3, dl, device)

    with open(os.path.join(model_dir, "UpTo_0_Stacking_s2026.pkl"), "rb") as f:
        stk = pickle.load(f)
    p_stack = stk.predict_proba(np.concatenate([p1, p2, p3], axis=1))

    df = pd.read_csv(idx_csv)
    src_ids = df.iloc[te_idx]["id_t0"].astype(str).to_numpy()
    y_true = y[te_idx]

    return {
        "dataset": "GSE114412",
        "task": "multiclass",
        "setting": "UpTo_0",
        "target_classes": classes,
        "y_true": y_true,
        "p_sc": p_stack,
        "source_ids": src_ids,
        "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE114412/preprocess_final/processed_norm_log_hvg1000.h5ad",
        "out_dir": "/Users/wanghongye/python/scLineagetracer/classification/GSE114412/roc/benchmark_upto_0_official_plus",
        "state_col_raw": "Assigned_cluster",
        "time_col_raw": "_week",
        "source_time_raw": "0",
        "target_time_raw": "5",
        "mid_times_raw": ["1", "2", "3", "4"],
        "target_cap_each": 3000,
        "source_cap": 4000,
        "time_map": {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0},
    }


def run_sclineage_175634():
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_sequences.h5"
    idx_csv = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_index.csv"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)

    seed = 7
    idx = np.arange(len(y))
    tr_idx, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
    va_idx, te_idx = train_test_split(tmp, test_size=0.5, random_state=seed)
    _ = (tr_idx, va_idx)

    mask = list(range(2, X.shape[1]))
    ds = SeqDatasetMask(X, y, te_idx.astype(np.int64), mask_t=mask)
    dl = DataLoader(ds, batch_size=512, shuffle=False)

    device = torch.device("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    d = X.shape[2]

    m1 = LSTMFix(d, 256, 2, 0.3).to(device)
    m2 = RNNFix(d, 256, 2, 0.3).to(device)
    m3 = TransformerFix(d, 256, 2, 0.3, 4).to(device)

    m1.load_state_dict(torch.load(os.path.join(model_dir, "Obs_Day1_BiLSTM_s7.pth"), map_location=device))
    m2.load_state_dict(torch.load(os.path.join(model_dir, "Obs_Day1_RNN_s7.pth"), map_location=device))
    m3.load_state_dict(torch.load(os.path.join(model_dir, "Obs_Day1_Trans_s7.pth"), map_location=device))

    p1 = get_prob_pos_fix(m1, dl, device)
    p2 = get_prob_pos_fix(m2, dl, device)
    p3 = get_prob_pos_fix(m3, dl, device)

    with open(os.path.join(model_dir, "Obs_Day1_Stacking_s7.pkl"), "rb") as f:
        stk = pickle.load(f)
    p_pos = stk.predict_proba(np.c_[p1, p2, p3])[:, 1]

    df = pd.read_csv(idx_csv)
    src_ids = df.iloc[te_idx]["id_t1"].astype(str).to_numpy()
    y_true = y[te_idx]

    return {
        "dataset": "GSE175634",
        "task": "binary",
        "setting": "Obs_Day1",
        "target_classes": ["CM", "CF"],
        "positive_class": "CF",
        "y_true": y_true,
        "p_sc": p_pos,
        "source_ids": src_ids,
        "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE175634/preprocess_final/processed_norm_log_hvg1000.h5ad",
        "out_dir": "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_obs_day1_official",
        "state_col_raw": "type",
        "time_col_raw": "diffday",
        "source_time_raw": "day1",
        "target_time_raw": "day15",
        "mid_times_raw": ["day3", "day5", "day7", "day11"],
        "target_cap_each": 2000,
        "source_cap": 1200,
        "time_map": {
            "day0": 0.0,
            "day1": 1.0,
            "day3": 3.0,
            "day5": 5.0,
            "day7": 7.0,
            "day11": 11.0,
            "day15": 15.0,
        },
    }


def run_sclineage_99915():
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_sequences.h5"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)
        indices = np.asarray(f["indices"], dtype=np.int64)

    seed = 999
    idx = np.arange(len(y))
    tr_idx, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
    va_idx, te_idx = train_test_split(tmp, test_size=0.5, random_state=seed)
    _ = (tr_idx, va_idx)

    mask = [2, 3, 4, 5]
    ds = SeqDatasetMask(X, y, te_idx.astype(np.int64), mask_t=mask)
    dl = DataLoader(ds, batch_size=512, shuffle=False)

    device = torch.device("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    d = X.shape[2]

    m1 = LSTMFix(d, 256, 2, 0.3).to(device)
    m2 = RNNFix(d, 256, 2, 0.3).to(device)
    m3 = TransformerFix(d, 256, 2, 0.3, 4).to(device)

    m1.load_state_dict(torch.load(os.path.join(model_dir, "Obs_Day9_BiLSTM_s999.pth"), map_location=device))
    m2.load_state_dict(torch.load(os.path.join(model_dir, "Obs_Day9_RNN_s999.pth"), map_location=device))
    m3.load_state_dict(torch.load(os.path.join(model_dir, "Obs_Day9_Trans_s999.pth"), map_location=device))

    p1 = get_prob_pos_fix(m1, dl, device)
    p2 = get_prob_pos_fix(m2, dl, device)
    p3 = get_prob_pos_fix(m3, dl, device)

    with open(os.path.join(model_dir, "Obs_Day9_Stacking_s999.pkl"), "rb") as f:
        stk = pickle.load(f)
    p_pos = stk.predict_proba(np.c_[p1, p2, p3])[:, 1]

    src_ids = []
    src_times = []
    for i in te_idx:
        s1 = int(indices[i, 1])
        s0 = int(indices[i, 0])
        if s1 >= 0:
            src_ids.append(str(s1))
            src_times.append("Day9")
        elif s0 >= 0:
            src_ids.append(str(s0))
            src_times.append("Day6")
        else:
            src_ids.append(None)
            src_times.append(None)

    y_true = y[te_idx]

    return {
        "dataset": "GSE99915",
        "task": "binary",
        "setting": "Obs_Day9",
        "target_classes": ["Failed", "Reprogrammed"],
        "positive_class": "Reprogrammed",
        "y_true": y_true,
        "p_sc": p_pos,
        "source_ids": np.array(src_ids, dtype=object),
        "source_time_per_sample": np.array(src_times, dtype=object),
        "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE99915/preprocess_final/processed_norm_log_hvg1000.h5ad",
        "out_dir": "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/benchmark_obs_day9_official_plus",
        "state_col_raw": "state_info",
        "time_col_raw": "time_info",
        "source_time_raw": "Day9",
        "target_time_raw": "Day28",
        "mid_times_raw": ["Day12", "Day15", "Day21"],
        "target_cap_each": 3000,
        "source_cap": 5000,
        "time_map": {
            "Day6": 6.0,
            "Day9": 9.0,
            "Day12": 12.0,
            "Day15": 15.0,
            "Day21": 21.0,
            "Day28": 28.0,
        },
    }


def to_time_num(v, time_map):
    s = str(v)
    if s in time_map:
        return float(time_map[s])
    try:
        s2 = str(int(float(s)))
        if s2 in time_map:
            return float(time_map[s2])
        return float(s)
    except Exception:
        return np.nan


def prepare_adata_subset(meta):
    adata = ad.read_h5ad(meta["h5ad"])
    adata.obs_names = pd.Index([str(x) for x in adata.obs_names])

    time_raw = adata.obs[meta["time_col_raw"]].astype(str).to_numpy()
    state_raw = adata.obs[meta["state_col_raw"]].astype(str).to_numpy()
    state_map = meta.get("state_map", None)
    if isinstance(state_map, dict) and len(state_map) > 0:
        state_raw = np.array([state_map.get(x, x) for x in state_raw], dtype=object)

    target_classes = set(meta["target_classes"])
    valid_source = [str(x) for x in meta["source_ids"] if (x is not None) and (str(x) != "None")]
    source_cap = int(meta.get("source_cap", 0))
    if (source_cap > 0) and (len(valid_source) > source_cap):
        # keep deterministic subset to limit OT solve size on large datasets
        valid_source = list(np.random.default_rng(2026).choice(np.array(valid_source, dtype=object), size=source_cap, replace=False))
    source_set = set(valid_source)

    source_mask = np.array([str(x) in source_set for x in adata.obs_names], dtype=bool)
    target_mask = np.zeros(adata.n_obs, dtype=bool)
    target_cap_each = int(meta.get("target_cap_each", 0))
    for cls in meta["target_classes"]:
        idx = np.where((time_raw == str(meta["target_time_raw"])) & (state_raw == str(cls)))[0]
        if (target_cap_each > 0) and (len(idx) > target_cap_each):
            idx = np.random.default_rng(2026).choice(idx, size=target_cap_each, replace=False)
        target_mask[idx] = True

    mid_mask = np.zeros(adata.n_obs, dtype=bool)
    rng = np.random.default_rng(2026)
    mid_cap_each = int(meta.get("mid_cap_each", 3000))
    for mt in meta["mid_times_raw"]:
        idx = np.where(time_raw == str(mt))[0]
        if (mid_cap_each > 0) and (len(idx) > mid_cap_each):
            idx = rng.choice(idx, size=mid_cap_each, replace=False)
        mid_mask[idx] = True

    keep = source_mask | target_mask | mid_mask
    sub = adata[keep].copy()
    sub.obs_names = pd.Index([str(x) for x in sub.obs_names])
    if isinstance(state_map, dict) and len(state_map) > 0:
        sub.obs["state_model"] = pd.Categorical(np.array([state_map.get(x, x) for x in sub.obs[meta["state_col_raw"]].astype(str).to_numpy()], dtype=object))
    else:
        sub.obs["state_model"] = sub.obs[meta["state_col_raw"]].astype(str)
    sub.obs["time_str"] = sub.obs[meta["time_col_raw"]].astype(str)
    sub.obs["time_num"] = [to_time_num(v, meta["time_map"]) for v in sub.obs["time_str"].to_numpy()]

    nan_time = np.isnan(np.asarray(sub.obs["time_num"], dtype=np.float64)).sum()
    if nan_time > 0:
        raise RuntimeError(f"Found {nan_time} cells with unmapped time values.")

    return sub


def run_cellrank_multiclass(adata_sub, target_classes, target_time_num):
    work = adata_sub.copy()
    sc.pp.pca(work, n_comps=min(50, work.n_obs - 1, work.n_vars - 1))
    sc.pp.neighbors(work, n_neighbors=min(30, work.n_obs - 1), use_rep="X_pca")

    ck = cr.kernels.ConnectivityKernel(work).compute_transition_matrix()
    temporal_kernel = None
    kernel_note = "realtime"
    try:
        # RealTimeKernel in CellRank 2.x needs pre-defined couplings for each time pair.
        # If unavailable in the current setup, we fall back to PseudotimeKernel.
        temporal_kernel = cr.kernels.RealTimeKernel(work, time_key="time_num").compute_transition_matrix()
    except Exception:
        kernel_note = "pseudotime_fallback"
        temporal_kernel = cr.kernels.PseudotimeKernel(work, time_key="time_num").compute_transition_matrix(
            threshold_scheme="hard",
            n_jobs=1,
            backend="threading",
            show_progress_bar=False,
        )
    ker = 0.5 * ck + 0.5 * temporal_kernel
    g = cr.estimators.GPCCA(ker)

    t = work.obs["time_num"].astype(float).to_numpy()
    s = work.obs["state_model"].astype(str).to_numpy()
    term = {}
    for c in target_classes:
        term[c] = list(work.obs_names[(t == float(target_time_num)) & (s == c)])
    g.set_terminal_states(term)
    g.compute_fate_probabilities(
        solver="direct",
        use_petsc=False,
        n_jobs=1,
        backend="threading",
        show_progress_bar=False,
    )

    fp = g.fate_probabilities
    names = list(fp.names)
    mat = np.asarray(fp.X, dtype=np.float64)
    col_idx = [names.index(c) for c in target_classes]
    mat = mat[:, col_idx]
    ids = np.asarray(work.obs_names, dtype=str)
    return {str(cid): mat[i] for i, cid in enumerate(ids)}, kernel_note


def run_wot_multiclass(adata_sub, target_classes, source_time_num, target_time_num):
    work = adata_sub.copy()
    work.obs["day_wot"] = work.obs["time_num"].astype(float)
    model = wot.ot.OTModel(work, day_field="day_wot")
    tm = model.compute_transport_map(float(source_time_num), float(target_time_num))

    M = np.asarray(tm.X, dtype=np.float64)
    target_labels = work.obs.loc[tm.var_names, "state_model"].astype(str).to_numpy()

    P = np.zeros((M.shape[0], len(target_classes)), dtype=np.float64)
    for j, cls in enumerate(target_classes):
        m = target_labels == cls
        if np.any(m):
            P[:, j] = M[:, m].sum(axis=1)
    P = normalize_rows(P)

    src = np.asarray(tm.obs_names, dtype=str)
    return {str(cid): P[i] for i, cid in enumerate(src)}


def run_cospar_multiclass(adata_sub, target_classes, source_time_str, target_time_str, out_dir, data_des, time_order=None):
    work = adata_sub.copy()
    work.obs["time_str"] = work.obs["time_str"].astype(str)
    sc.pp.pca(work, n_comps=min(30, work.n_obs - 1, work.n_vars - 1))

    cs.settings.data_path = out_dir
    cs.settings.figure_path = out_dir
    cs.settings.verbosity = 1

    ac = cs.pp.initialize_adata_object(
        adata=None,
        X_state=work.X,
        X_pca=work.obsm["X_pca"],
        cell_names=np.array(work.obs_names),
        gene_names=np.array(work.var_names),
        time_info=work.obs["time_str"].to_numpy(),
        state_info=work.obs["state_model"].astype(str).to_numpy(),
        data_des=data_des,
    )
    if time_order is not None and len(time_order) > 0:
        try:
            cs.hf.update_time_ordering(ac, updated_ordering=np.array([str(x) for x in time_order], dtype=object))
        except Exception as e:
            raise RuntimeError(f"CoSpar update_time_ordering failed: {e}") from e
    if ("X_emb" not in ac.obsm) and ("X_pca" in ac.obsm):
        ac.obsm["X_emb"] = np.asarray(ac.obsm["X_pca"][:, :2], dtype=np.float64)

    ac = cs.tmap.infer_Tmap_from_state_info_alone(
        ac,
        initial_time_points=[str(source_time_str)],
        later_time_point=str(target_time_str),
        max_iter_N=[3, 1],
        compute_new=True,
        CoSpar_KNN=15,
        smooth_array=[10, 5],
    )

    T = ac.uns["transition_map"]
    src_idx = np.asarray(ac.uns["Tmap_cell_id_t1"], dtype=np.int64)
    tgt_idx = np.asarray(ac.uns["Tmap_cell_id_t2"], dtype=np.int64)
    tgt_labels = ac.obs.iloc[tgt_idx]["state_info"].astype(str).to_numpy()

    P = np.zeros((T.shape[0], len(target_classes)), dtype=np.float64)
    for j, cls in enumerate(target_classes):
        m = tgt_labels == cls
        if np.any(m):
            P[:, j] = np.asarray(T[:, m].sum(axis=1), dtype=np.float64).reshape(-1)
    P = normalize_rows(P)

    src_ids = np.asarray(ac.obs_names[src_idx], dtype=str)
    return {str(cid): P[i] for i, cid in enumerate(src_ids)}


def mean_prob_from_map(prob_map, n_classes):
    if (prob_map is None) or (len(prob_map) == 0):
        return np.ones(n_classes, dtype=np.float64) / float(n_classes)
    mat = np.asarray(list(prob_map.values()), dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[1] != int(n_classes):
        return np.ones(n_classes, dtype=np.float64) / float(n_classes)
    row = np.asarray(mat.mean(axis=0), dtype=np.float64).reshape(1, -1)
    return normalize_rows(row).reshape(-1)


def extract_prob_matrix(prob_map, source_ids, n_classes, missing_fill=None):
    out = np.zeros((len(source_ids), n_classes), dtype=np.float64)
    miss = 0
    if missing_fill is None:
        uni = np.ones(n_classes, dtype=np.float64) / float(n_classes)
    else:
        uni = normalize_rows(np.asarray(missing_fill, dtype=np.float64).reshape(1, -1)).reshape(-1)
    for i, sid in enumerate(source_ids):
        if (sid is None) or (str(sid) == "None"):
            out[i] = uni
            miss += 1
            continue
        row = prob_map.get(str(sid))
        if row is None:
            out[i] = uni
            miss += 1
        else:
            out[i] = np.asarray(row, dtype=np.float64)
    out = normalize_rows(out)
    return out, miss


def is_same_time(a, b, eps=1e-9):
    try:
        a = float(a)
        b = float(b)
        if (not np.isfinite(a)) or (not np.isfinite(b)):
            return False
        return abs(a - b) <= float(eps)
    except Exception:
        return False


def build_state_identity_prob_map(adata_sub, target_classes):
    c2i = {str(c): i for i, c in enumerate(target_classes)}
    c = len(target_classes)
    uni = np.ones(c, dtype=np.float64) / float(c)
    out = {}
    labels = adata_sub.obs["state_model"].astype(str).to_numpy()
    for cid, lab in zip(np.asarray(adata_sub.obs_names, dtype=str), labels):
        row = np.zeros(c, dtype=np.float64)
        j = c2i.get(str(lab), None)
        if j is None:
            row[:] = uni
        else:
            row[j] = 1.0
        out[str(cid)] = row
    return out


def build_probs_grouped_by_source_time(
    meta,
    source_ids,
    source_time_per_sample,
    target_time_num,
    n_classes,
    identity_map,
    runner_fn,
    strict_errors=False,
    allow_same_time_identity=True,
):
    source_ids = np.asarray(source_ids, dtype=object)
    n = len(source_ids)
    uni = np.ones((n, n_classes), dtype=np.float64) / float(n_classes)
    out = uni.copy()
    miss = 0
    notes = []
    errs = []

    default_time = meta.get("source_time_raw", None)
    default_time = None if default_time is None else str(default_time)

    if source_time_per_sample is None:
        src_times = np.array([default_time] * n, dtype=object)
    else:
        src_times = np.asarray(source_time_per_sample, dtype=object)
        if len(src_times) != n:
            src_times = np.array([default_time] * n, dtype=object)
            notes.append("source_time_per_sample_len_mismatch_use_default")
        else:
            src_times = src_times.copy()
            for i, v in enumerate(src_times):
                if v is None:
                    src_times[i] = default_time
                else:
                    s = str(v)
                    if s in ("", "None", "nan", "NaN"):
                        src_times[i] = default_time
                    else:
                        src_times[i] = s

    valid = np.array([x is not None for x in src_times], dtype=bool)
    if np.any(~valid):
        miss += int((~valid).sum())
        notes.append("missing_source_time_use_uniform")

    groups = {}
    for i, st in enumerate(src_times):
        if st is None:
            continue
        groups.setdefault(str(st), []).append(i)

    def _sort_key(st):
        return to_time_num(st, meta["time_map"])

    for st in sorted(groups.keys(), key=_sort_key):
        idx = np.asarray(groups[st], dtype=np.int64)
        st_num = to_time_num(st, meta["time_map"])
        if not np.isfinite(st_num):
            miss += int(len(idx))
            notes.append(f"bad_source_time_uniform@{st}")
            continue
        try:
            if allow_same_time_identity and is_same_time(st_num, target_time_num):
                pmap = identity_map
                notes.append(f"same_time_identity@{st}")
            else:
                pmap = runner_fn(str(st), float(st_num))
                notes.append(f"official@{st}")
            p_default = mean_prob_from_map(pmap, n_classes)
            p_grp, miss_grp = extract_prob_matrix(pmap, source_ids[idx], n_classes, missing_fill=p_default)
            out[idx] = p_grp
            miss += int(miss_grp)
        except Exception as e:
            miss += int(len(idx))
            notes.append(f"uniform_fallback@{st}")
            errs.append(f"{st}: {e}")

    if bool(strict_errors) and len(errs) > 0:
        raise RuntimeError(" ; ".join(errs))

    out = normalize_rows(out)
    note = "|".join(notes) if len(notes) > 0 else "official"
    return out, int(miss), note, errs


def sanitize_key(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def run_one(meta, run_cospar):
    ensure_dir(meta["out_dir"])
    ensure_dir(os.path.join(meta["out_dir"], "cospar_data"))

    adata_sub = prepare_adata_subset(meta)

    source_time_num = to_time_num(meta["source_time_raw"], meta["time_map"])
    target_time_num = to_time_num(meta["target_time_raw"], meta["time_map"])
    source_time_per_sample = meta.get("source_time_per_sample", None)

    method_probs = {}
    method_err = []
    miss_rows = {}
    method_note = {}

    if meta["task"] == "binary":
        p_sc_mat = np.c_[1.0 - meta["p_sc"], meta["p_sc"]]
    else:
        p_sc_mat = normalize_rows(meta["p_sc"])

    method_probs["scLineagetracer"] = p_sc_mat
    n_classes = len(meta["target_classes"])
    identity_map = build_state_identity_prob_map(adata_sub, meta["target_classes"])

    try:
        if is_same_time(source_time_num, target_time_num):
            p_default = mean_prob_from_map(identity_map, n_classes)
            P, miss = extract_prob_matrix(identity_map, meta["source_ids"], n_classes, missing_fill=p_default)
            method_probs["CellRank"] = P
            miss_rows["CellRank"] = miss
            method_note["CellRank"] = "same_time_state_identity"
        else:
            pmap, cellrank_kernel_note = run_cellrank_multiclass(adata_sub, meta["target_classes"], target_time_num)
            p_default = mean_prob_from_map(pmap, n_classes)
            P, miss = extract_prob_matrix(pmap, meta["source_ids"], n_classes, missing_fill=p_default)
            method_probs["CellRank"] = P
            miss_rows["CellRank"] = miss
            method_note["CellRank"] = f"official_{cellrank_kernel_note}"
    except Exception as e:
        method_err.append(("CellRank", str(e)))
        miss_rows["CellRank"] = int(len(meta["y_true"]))
        method_note["CellRank"] = "official_failed_no_result"

    try:
        P, miss, note, errs = build_probs_grouped_by_source_time(
            meta=meta,
            source_ids=meta["source_ids"],
            source_time_per_sample=source_time_per_sample,
            target_time_num=target_time_num,
            n_classes=n_classes,
            identity_map=identity_map,
            runner_fn=lambda st_raw, st_num: run_wot_multiclass(adata_sub, meta["target_classes"], st_num, target_time_num),
            strict_errors=True,
            allow_same_time_identity=True,
        )
        method_probs["WOT"] = P
        miss_rows["WOT"] = miss
        method_note["WOT"] = note
        if len(errs) > 0:
            method_err.append(("WOT(grouped)", " ; ".join(errs)))
    except Exception as e:
        method_err.append(("WOT", str(e)))
        miss_rows["WOT"] = int(len(meta["y_true"]))
        method_note["WOT"] = "official_failed_no_result"

    # Force official CoSpar run. Do NOT copy WOT.
    try:
        obs_times = set(adata_sub.obs["time_str"].astype(str).tolist())
        cospar_time_order = [str(k) for k, _ in sorted(meta["time_map"].items(), key=lambda kv: float(kv[1])) if str(k) in obs_times]
        if len(cospar_time_order) == 0:
            cospar_time_order = sorted(list(obs_times))
        P, miss, note, errs = build_probs_grouped_by_source_time(
            meta=meta,
            source_ids=meta["source_ids"],
            source_time_per_sample=source_time_per_sample,
            target_time_num=target_time_num,
            n_classes=n_classes,
            identity_map=identity_map,
            runner_fn=lambda st_raw, st_num: run_cospar_multiclass(
                adata_sub,
                meta["target_classes"],
                str(st_raw),
                str(meta["target_time_raw"]),
                os.path.join(meta["out_dir"], "cospar_data"),
                data_des=f"{meta['dataset'].lower()}_{meta['setting'].lower()}_{sanitize_key(str(st_raw)).lower()}_official",
                time_order=cospar_time_order,
            ),
            strict_errors=True,
            allow_same_time_identity=True,
        )
        method_probs["CoSpar"] = P
        miss_rows["CoSpar"] = miss
        method_note["CoSpar"] = note
        if len(errs) > 0:
            method_err.append(("CoSpar(grouped)", " ; ".join(errs)))
    except Exception as e:
        method_err.append(("CoSpar", str(e)))
        miss_rows["CoSpar"] = int(len(meta["y_true"]))
        method_note["CoSpar"] = "official_failed_no_result"

    rows = []
    curves = {}
    if meta["task"] == "binary":
        pos_idx = meta["target_classes"].index(meta["positive_class"])
        for m, P in method_probs.items():
            c = eval_binary(meta["y_true"], P[:, pos_idx])
            curves[m] = c
            rows.append(
                {
                    "Setting": meta["setting"],
                    "Method": m,
                    "AUC": c["AUC"],
                    "AUROC": c["AUROC"],
                    "AUPRC": c["AUPRC"],
                    "Accuracy": c["Accuracy"],
                    "ACC": c["ACC"],
                    "F1": c["F1"],
                    "Specificity": c["Specificity"],
                    "Precision": c["Precision"],
                    "Recall": c["Recall"],
                    "LogLoss": c["LogLoss"],
                    "N_test": int(len(meta["y_true"])),
                    "MissingSourceFallback": int(miss_rows.get(m, 0)),
                    "MethodNote": str(method_note.get(m, "")),
                    "Pipeline": "official",
                }
            )
        expected = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
        present = set([r["Method"] for r in rows])
        for m in expected:
            if m in present:
                continue
            rows.append(
                {
                    "Setting": meta["setting"],
                    "Method": m,
                    "AUC": np.nan,
                    "AUROC": np.nan,
                    "AUPRC": np.nan,
                    "Accuracy": np.nan,
                    "ACC": np.nan,
                    "F1": np.nan,
                    "Specificity": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "LogLoss": np.nan,
                    "N_test": int(len(meta["y_true"])),
                    "MissingSourceFallback": int(miss_rows.get(m, len(meta["y_true"]))),
                    "MethodNote": str(method_note.get(m, "official_failed_no_result")),
                    "Pipeline": "official",
                }
            )
    else:
        per_class_rows = []
        for m, P in method_probs.items():
            c = eval_multiclass(meta["y_true"], P)
            curves[m] = c
            rows.append(
                {
                    "Setting": meta["setting"],
                    "Method": m,
                    "AUC_macro": c["AUC_macro"],
                    "Accuracy": c["Accuracy"],
                    "LogLoss": c["LogLoss"],
                    "N_test": int(len(meta["y_true"])),
                    "MissingSourceFallback": int(miss_rows.get(m, 0)),
                    "MethodNote": str(method_note.get(m, "")),
                    "Pipeline": "official",
                }
            )
            for cls, a in zip(meta["target_classes"], c["AUC_per_class"]):
                per_class_rows.append({"Method": m, "Class": cls, "AUC": float(a)})
        expected = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
        present = set([r["Method"] for r in rows])
        for m in expected:
            if m in present:
                continue
            rows.append(
                {
                    "Setting": meta["setting"],
                    "Method": m,
                    "AUC_macro": np.nan,
                    "Accuracy": np.nan,
                    "LogLoss": np.nan,
                    "N_test": int(len(meta["y_true"])),
                    "MissingSourceFallback": int(miss_rows.get(m, len(meta["y_true"]))),
                    "MethodNote": str(method_note.get(m, "official_failed_no_result")),
                    "Pipeline": "official",
                }
            )
        pd.DataFrame(per_class_rows).to_csv(os.path.join(meta["out_dir"], "metrics_auc_per_class.csv"), index=False)

    if meta["task"] == "binary":
        summary = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        summary.to_csv(os.path.join(meta["out_dir"], "metrics_summary.csv"), index=False)
        summary[["Method", "Accuracy"]].to_csv(os.path.join(meta["out_dir"], "metrics_accuracy_only.csv"), index=False)
        summary[["Method", "AUC"]].to_csv(os.path.join(meta["out_dir"], "metrics_auc_only.csv"), index=False)

        curve_rows = []
        for m, c in curves.items():
            for i in range(len(c["fpr"])):
                curve_rows.append({"Method": m, "point_id": i, "fpr": float(c["fpr"][i]), "tpr": float(c["tpr"][i])})
        pd.DataFrame(curve_rows).to_csv(os.path.join(meta["out_dir"], "roc_curve_points.csv"), index=False)

        order = [
            "scLineagetracer",
            "CellRank",
            "WOT",
            "CoSpar",
        ]
        roc_suffix = "OfficialPlus" if "official_plus" in str(meta["out_dir"]).lower() else "Official"
        base = os.path.join(meta["out_dir"], f"ROC_Comparison_{meta['setting']}_{roc_suffix}")
        plot_dual_binary(curves, order, base, f"Selected ROC Curves ({meta['setting']} Official)")
    else:
        summary = pd.DataFrame(rows).sort_values("AUC_macro", ascending=False)
        summary.to_csv(os.path.join(meta["out_dir"], "metrics_summary.csv"), index=False)
        summary[["Method", "Accuracy"]].to_csv(os.path.join(meta["out_dir"], "metrics_accuracy_only.csv"), index=False)
        summary[["Method", "AUC_macro"]].to_csv(os.path.join(meta["out_dir"], "metrics_auc_macro_only.csv"), index=False)

        curve_rows = []
        for m, c in curves.items():
            for i in range(len(c["fpr_macro"])):
                curve_rows.append(
                    {
                        "Method": m,
                        "point_id": i,
                        "fpr_macro": float(c["fpr_macro"][i]),
                        "tpr_macro": float(c["tpr_macro"][i]),
                    }
                )
        pd.DataFrame(curve_rows).to_csv(os.path.join(meta["out_dir"], "roc_curve_points_macro.csv"), index=False)

        order = [
            "scLineagetracer",
            "CellRank",
            "WOT",
            "CoSpar",
        ]
        roc_suffix = "OfficialPlus" if "official_plus" in str(meta["out_dir"]).lower() else "Official"
        base = os.path.join(meta["out_dir"], f"ROC_Comparison_{meta['setting']}_{roc_suffix}")
        plot_dual_multiclass(curves, order, base, f"Macro ROC ({meta['setting']} Official)")

    npz_data = {
        "y_true": np.asarray(meta["y_true"], dtype=np.int64),
    }
    for m, P in method_probs.items():
        npz_data[f"p_{sanitize_key(m)}"] = np.asarray(P, dtype=np.float64)
    np.savez(os.path.join(meta["out_dir"], "benchmark_probs.npz"), **npz_data)

    with open(os.path.join(meta["out_dir"], "run_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Official run log ({meta['dataset']} {meta['setting']})\n")
        f.write(f"N_test={len(meta['y_true'])}\n")
        if miss_rows:
            f.write("Missing source fallback rows:\n")
            for k, v in miss_rows.items():
                f.write(f"- {k}: {int(v)}\n")
        if method_note:
            f.write("Method notes:\n")
            for k, v in method_note.items():
                f.write(f"- {k}: {v}\n")
        if method_err:
            f.write("Failed methods:\n")
            for m, e in method_err:
                f.write(f"- {m}: {e}\n")
        else:
            f.write("All methods succeeded.\n")

    print(f"[DONE] Output folder: {meta['out_dir']}")
    print(summary.to_string(index=False))
    if method_err:
        print("[WARN] Some methods failed:")
        for m, e in method_err:
            print(f"- {m}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GSE114412", "GSE175634", "GSE99915"])
    parser.add_argument("--run_cospar", type=int, default=1, help="kept for compatibility; CoSpar is always forced official.")
    args = parser.parse_args()

    set_seed(2026)

    if args.dataset == "GSE114412":
        meta = run_sclineage_114412()
    elif args.dataset == "GSE175634":
        meta = run_sclineage_175634()
    else:
        meta = run_sclineage_99915()

    run_one(meta, run_cospar=int(args.run_cospar))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
