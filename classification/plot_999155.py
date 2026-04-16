# -*- coding: utf-8 -*-
"""
Click-run plotting for GSE99915 (binary):
- Load saved base models + stacking
- Plot ROC (114412 style) + 2D prob-plane (prob vs prob, y=x)

Outputs:
- <OUT_DIR>/roc_click/ROC_<setting>_Stacking_s<seed>.png/pdf
- <OUT_DIR>/roc_click/2d/Pred2D_<setting>_<X>vs<Y>_s<seed>.png/pdf
- <OUT_DIR>/roc_click/plot_summary.csv
"""

import os
import pickle
import random
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss


# ============================================================
# ✅ 配置区（按需改）
# ============================================================
TIME_SERIES_H5 = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_sequences.h5"
INDEX_CSV      = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_index.csv"

OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915"
SAVED_MODELS_DIR = os.path.join(OUT_DIR, "saved_models")

POS_LABEL = "Reprogrammed"
NEG_LABEL = "Failed"

SEEDS = {
    "All_Days": 2026,
    "Obs_Day21": 2024,
    "Obs_Day15": 42,
    "Obs_Day12": 123,
    "Obs_Day9": 999
}

# 只画这些 setting（分别输出 ROC + 2D）
SELECT_SETTINGS = ["All_Days", "Obs_Day21", "Obs_Day15", "Obs_Day12", "Obs_Day9"]

# ✅ 总和/对比 ROC：你自己选（空列表=不画）
PLOT_SELECTED_ROC = True
SELECTED_ROC_SETTINGS = ["Obs_Day21", "Obs_Day15", "Obs_Day12"]  # <- 改这里

# split（必须与训练一致：random 80/10/10）
SPLIT_MODE = "random"
SPLIT_RATIO = (0.8, 0.1, 0.1)

# 模型超参（必须与训练一致）
BATCH_SIZE = 512
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
NHEAD = 4

# 输出目录（不覆盖训练脚本的原图）
ROC_DIR = os.path.join(OUT_DIR, "roc_click")
D2_DIR  = os.path.join(ROC_DIR, "2d")

# -------- 2D 平面：选两轴（默认 BiLSTM vs Trans）--------
# 可选： "BiLSTM", "RNN", "Trans", "Stacking"
PLANE_X = "Trans"
PLANE_Y = "Stacking"

# 2D 抽样点数（<=0=全量）
N_POINTS_2D = 12000
SAMPLE_SEED_2D = 2026

# 点更大（你要的）
POINT_SIZE_CM = 35
POINT_SIZE_CF = 35
ALPHA_2D = 0.70

# 坐标轴文案
AXIS_LABEL_MODE = "symbol"  # "symbol" / "paper"
SHOW_2D_TITLE = False
# ---------------------------------------------------------

# ROC 标题开关
SHOW_ROC_TITLE = True

# CM/CF 颜色形状（按你当前格式）
COLOR_NEG =  "#d62728" # Failed (NEG)
COLOR_POS = "#1f77b4" # Reprogrammed (POS)
MARK_NEG  = "s"
MARK_POS  = "o"
# ============================================================


# ============================================================
# 114412 ROC style (直接复刻)
# ============================================================
GREY = "#444444"
LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e", "#7f7f7f"]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def _style_axes(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    # start (0,0)
    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)

    # end (1,1)
    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)

    # compress x==1 duplicates (avoid right vertical)
    mask_one = (fpr == 1.0)
    if np.any(mask_one):
        t1 = float(np.max(tpr[mask_one]))
        keep = ~mask_one
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]

    return fpr, tpr


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_binary_roc_114412(y_true, y_prob, title, out_no_ext, color="#1f77b4"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    a = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    ax.plot(fpr, tpr, lw=2.8, color=color, label=f"AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], lw=1.5, color=GREY, ls="--", alpha=0.55)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    if SHOW_ROC_TITLE:
        ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _style_axes(ax)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, out_no_ext)
    return float(a), fpr, tpr


def plot_multi_settings_rocs_114412(roc_dict, title, out_no_ext):
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.plot([0, 1], [0, 1], lw=1.4, color=GREY, ls="--", alpha=0.5)

    for i, (s, (fpr, tpr, a)) in enumerate(roc_dict.items()):
        ax.plot(fpr, tpr, lw=2.7, color=LINE_COLORS[i % len(LINE_COLORS)],
                label=f"{s} (AUC={a:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    if SHOW_ROC_TITLE:
        ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style_axes(ax)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, out_no_ext)


# ============================================================
# Data + Models
# ============================================================
def load_data_binary():
    with h5py.File(TIME_SERIES_H5, "r") as f:
        X_all = np.asarray(f["X"], dtype=np.float32)

    df = pd.read_csv(INDEX_CSV)
    labels = df["label_str"].values if "label_str" in df.columns else df["label"].values

    if "clone_id" in df.columns:
        clones = df["clone_id"].values
    else:
        with h5py.File(TIME_SERIES_H5, "r") as f:
            clones = np.asarray(f["seq_clone"], dtype=np.int64)

    keep = (labels == POS_LABEL) | (labels == NEG_LABEL)
    idx = np.where(keep)[0]

    X = X_all[idx]
    y = np.where(labels[idx] == POS_LABEL, 1, 0).astype(np.int64)
    clones = clones[idx]
    return X, y, clones


def get_split(n_samples, seed):
    all_idx = np.arange(n_samples)
    tr, tmp = train_test_split(all_idx, test_size=0.2, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    return tr, va, te


class SeqDataset(Dataset):
    def __init__(self, X, y, idx, mask=None):
        self.X, self.y, self.idx = X, y, idx
        self.mask = set(mask) if mask else set()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j].copy()
        for t in self.mask:
            x[t] = 0.0
        return torch.from_numpy(x), torch.tensor(int(self.y[j]), dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True, bidirectional=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(0.5 * (h[-2] + h[-1]))


class RNNModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.rnn = nn.RNN(d, h, l, batch_first=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        return self.head(self.rnn(x)[1][-1])


class TransformerModel(nn.Module):
    def __init__(self, d, h, l, dr, nhead):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, dim_feedforward=h * 2, dropout=dr, batch_first=True),
            num_layers=l
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x):
        return self.head(self.enc(x).mean(dim=1))


def load_model(kind, setting, seed, D, device):
    if kind == "BiLSTM":
        m = LSTMModel(D, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    elif kind == "RNN":
        m = RNNModel(D, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    elif kind == "Trans":
        m = TransformerModel(D, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NHEAD)
    else:
        raise ValueError(kind)

    pth = os.path.join(SAVED_MODELS_DIR, f"{setting}_{kind}_s{seed}.pth")
    if not os.path.isfile(pth):
        raise FileNotFoundError(f"[ERROR] missing model: {pth}")

    sd = torch.load(pth, map_location="cpu")
    m.load_state_dict(sd)
    m.to(device)
    m.eval()
    return m


def load_stacker(setting, seed):
    pkl = os.path.join(SAVED_MODELS_DIR, f"{setting}_Stacking_s{seed}.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"[ERROR] missing stacking pkl: {pkl}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def get_prob_pos(model, loader, device):
    model.eval()
    probs, targs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        targs.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targs)


def axis_label(model_tag: str) -> str:
    if AXIS_LABEL_MODE == "paper":
        return f"Predicted probability of {POS_LABEL} ({model_tag})"
    return f"{model_tag} p({POS_LABEL})"


def pick_axis_values(axis_name, p_bilstm, p_rnn, p_trans, p_stack):
    if axis_name == "BiLSTM":
        return p_bilstm, axis_label("BiLSTM")
    if axis_name == "RNN":
        return p_rnn, axis_label("RNN")
    if axis_name == "Trans":
        return p_trans, axis_label("Trans")
    if axis_name == "Stacking":
        return p_stack, axis_label("Stacking")
    raise ValueError(axis_name)


def plot_2d_prob_plane(xv, yv, y_true, out_no_ext, xlab, ylab, title=""):
    xv = np.clip(np.asarray(xv, dtype=np.float64), 0.0, 1.0)
    yv = np.clip(np.asarray(yv, dtype=np.float64), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=np.int64)

    # sampling
    if N_POINTS_2D is not None and int(N_POINTS_2D) > 0 and len(y_true) > int(N_POINTS_2D):
        rng = np.random.default_rng(SAMPLE_SEED_2D)
        idx = rng.choice(np.arange(len(y_true)), size=int(N_POINTS_2D), replace=False)
        xv, yv, y_true = xv[idx], yv[idx], y_true[idx]

    m0 = (y_true == 0)  # NEG
    m1 = (y_true == 1)  # POS

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot([0, 1], [0, 1], lw=1.8, color=GREY, ls="--", alpha=0.55, zorder=1)  # ✅ y=x

    ax.scatter(xv[m0], yv[m0], s=POINT_SIZE_CM, alpha=ALPHA_2D,
               c=COLOR_NEG, marker=MARK_NEG, edgecolors="none", label=NEG_LABEL, zorder=2)
    ax.scatter(xv[m1], yv[m1], s=POINT_SIZE_CF, alpha=ALPHA_2D,
               c=COLOR_POS, marker=MARK_POS, edgecolors="none", label=POS_LABEL, zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    if SHOW_2D_TITLE and title:
        ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    _style_axes(ax)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, out_no_ext)


def main():
    ensure_dir(ROC_DIR)
    ensure_dir(D2_DIR)
    device = pick_device()
    print(f"[INFO] device={device}")

    X, y, clones = load_data_binary()
    N, T, D = X.shape
    print(f"[INFO] X={X.shape}, y={y.shape} | POS={POS_LABEL} NEG={NEG_LABEL}")

    # masks（与训练一致）
    settings_mask = {
        "All_Days": None,
        "Obs_Day21": [5],
        "Obs_Day15": [4, 5],
        "Obs_Day12": [3, 4, 5],
        "Obs_Day9":  [2, 3, 4, 5],
    }

    summary_rows = []
    roc_curves_selected = {}

    for i, setting in enumerate(SELECT_SETTINGS):
        seed = int(SEEDS.get(setting, 2026))
        mask = settings_mask.get(setting, None)
        set_all_seeds(seed)

        _, _, te_idx = get_split(len(X), seed)
        te_loader = DataLoader(
            SeqDataset(X, y, te_idx, mask),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        # base probs
        m_bilstm = load_model("BiLSTM", setting, seed, D, device)
        m_rnn    = load_model("RNN",    setting, seed, D, device)
        m_trans  = load_model("Trans",  setting, seed, D, device)

        p_bilstm, y_te = get_prob_pos(m_bilstm, te_loader, device)
        p_rnn,   _     = get_prob_pos(m_rnn,    te_loader, device)
        p_trans, _     = get_prob_pos(m_trans,  te_loader, device)

        feat3 = np.stack([p_bilstm, p_rnn, p_trans], axis=1)

        stacker = load_stacker(setting, seed)
        p_stack = stacker.predict_proba(feat3)[:, 1]
        p_stack = np.clip(p_stack, 0.0, 1.0)

        # ROC (stacking)
        roc_out = os.path.join(ROC_DIR, f"ROC_{setting}_Stacking_s{seed}")
        auc_val, fpr, tpr = plot_binary_roc_114412(
            y_true=y_te,
            y_prob=p_stack,
            title=f"ROC - {setting}",
            out_no_ext=roc_out,
            color=LINE_COLORS[i % len(LINE_COLORS)]
        )

        # 2D (prob vs prob)
        xv, xlab = pick_axis_values(PLANE_X, p_bilstm, p_rnn, p_trans, p_stack)
        yv, ylab = pick_axis_values(PLANE_Y, p_bilstm, p_rnn, p_trans, p_stack)
        d2_out = os.path.join(D2_DIR, f"Pred2D_{setting}_{PLANE_X}vs{PLANE_Y}_s{seed}")
        plot_2d_prob_plane(xv, yv, y_te, d2_out, xlab, ylab)

        y_pred = (p_stack >= 0.5).astype(np.int64)
        acc = accuracy_score(y_te, y_pred)
        ll = log_loss(y_te, np.stack([1 - p_stack, p_stack], axis=1), labels=[0, 1])

        print(f"[RESULT] {setting} | AUC={auc_val:.4f} | Acc={acc:.4f} | LogLoss={ll:.4f}")

        summary_rows.append({
            "Setting": setting,
            "Seed": seed,
            "AUC": float(auc_val),
            "Accuracy": float(acc),
            "LogLoss": float(ll),
            "PlaneX": PLANE_X,
            "PlaneY": PLANE_Y,
            "NPoints2D": int(N_POINTS_2D),
        })

        if PLOT_SELECTED_ROC and (setting in set(SELECTED_ROC_SETTINGS)):
            roc_curves_selected[setting] = (fpr, tpr, float(auc_val))

    # selected ROC（按你给的顺序）
    if PLOT_SELECTED_ROC and SELECTED_ROC_SETTINGS:
        ordered = {}
        for s in SELECTED_ROC_SETTINGS:
            if s in roc_curves_selected:
                ordered[s] = roc_curves_selected[s]
        if len(ordered) > 0:
            out_no_ext = os.path.join(ROC_DIR, "ROC_Selected_Comparison_Stacking")
            plot_multi_settings_rocs_114412(ordered, "Selected ROC Curves", out_no_ext)

    out_csv = os.path.join(ROC_DIR, "plot_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"[DONE] Summary: {out_csv}")
    print(f"[DONE] Plots:   {ROC_DIR}")


if __name__ == "__main__":
    main()
