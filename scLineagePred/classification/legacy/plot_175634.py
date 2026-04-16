# -*- coding: utf-8 -*-
"""
GSE175634 (CM vs CF) - Plot ROC + 2D plane (prob vs prob)

✅ ROC 风格：严格使用 plot_114412.py 的 ROC 风格
- tick 朝外；上/右无 tick；隐藏刻度数字
- 灰色对角线 y=x
- 不贴边：set_xlim/ylim + margins
- lw/字号/figsize/pad_inches 与 114412 一致
- ROC 端点清理：压缩 FPR==1，避免右侧“多一根竖线”

✅ 2D：你截图那种更好（两轴都是 p(CF)）
- x = 模型A 预测 CF 的概率
- y = 模型B 预测 CF 的概率
- CM：蓝色方块；CF：红色圆圈
- 虚线：y = x
- 坐标轴文案可切换（更论文式）
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
# ✅ 配置区：只改这里
# ============================================================
TIME_SERIES_H5 = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_sequences.h5"
INDEX_CSV      = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_index.csv"

OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF"
SAVED_MODELS_DIR = os.path.join(OUT_DIR, "saved_models")

NEG_LABEL = "CM"
POS_LABEL = "CF"

SEEDS = {
    "All_Days": 2026,
    "Obs_Day11": 2024,
    "Obs_Day7": 42,
    "Obs_Day5": 123,
    "Obs_Day3": 999,
    "Obs_Day1": 7,
}

# split（必须与训练一致）
SPLIT_MODE  = "random"   # "random" 或 "clone"
SPLIT_RATIO = (0.8, 0.1, 0.1)

# 必须与训练一致（否则 load_state_dict 会失败）
BATCH_SIZE = 512
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
NHEAD      = 4

# 你要分别输出 ROC + 2D 的 setting
SELECT_SETTINGS = ["All_Days", "Obs_Day11", "Obs_Day7", "Obs_Day5", "Obs_Day3", "Obs_Day1"]

# ✅ “总和/对比 ROC”你自己选哪些 setting（空列表=不画）
PLOT_SELECTED_ROC = True
SELECTED_ROC_SETTINGS = ["Obs_Day1", "Obs_Day3", "Obs_Day5", "All_Days"]  # 1d,3d,5d,15d
SELECTED_ROC_LABELS = {
    "Obs_Day1": "1d",
    "Obs_Day3": "3d",
    "Obs_Day5": "5d",
    "All_Days": "15d",
}

# 输出目录
ROC_DIR = os.path.join(OUT_DIR, "roc_click")
D2_DIR  = os.path.join(ROC_DIR, "2d")

# -------- 2D 平面：选两个轴（默认就是你截图）--------
# 可选： "BiLSTM", "RNN", "Trans", "Stacking"
PLANE_X = "BiLSTM"
PLANE_Y = "Stacking"

# 2D 抽样点数（<=0 = 全量）
N_POINTS_2D = 3000
SAMPLE_SEED_2D = 2026

# 2D 点样式
POINT_SIZE_2D = 30
ALPHA_2D = 0.65

# 坐标轴文案（换一种讲法）
# mode="symbol":  BiLSTM p(CF)
# mode="paper":   Predicted probability of CF (BiLSTM)
AXIS_LABEL_MODE = "paper"

SHOW_2D_TITLE = False
# -------------------------------------------------------

# ROC：是否显示标题（114412 默认有 title）
SHOW_ROC_TITLE = True

# 颜色/形状
COLOR_NEG = "#1f77b4"   # CM 蓝
COLOR_POS = "#d62728"   # CF 红
MARK_NEG  = "s"
MARK_POS  = "o"

# ============================================================


# ============================================================
# 114412 ROC style constants (直接沿用)
# ============================================================
SCI_COLORS  = ["#377eb8", "#e41a1c", "#4daf4a"]
LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e", "#7f7f7f"]
GREY = "#444444"


# ============================================================
# Utils
# ============================================================
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


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_fig_114412(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def random_split(n, seed, ratio=(0.8, 0.1, 0.1)):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=(1 - ratio[0]), random_state=seed)
    rel = ratio[2] / (ratio[1] + ratio[2] + 1e-12)
    va, te = train_test_split(tmp, test_size=rel, random_state=seed)
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def group_split_by_clone(clone_ids, seed, ratio=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clone_ids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(ratio[0] * n)
    n_va = int(ratio[1] * n)

    tr_c = set(uniq[:n_tr].tolist())
    va_c = set(uniq[n_tr:n_tr + n_va].tolist())
    te_c = set(uniq[n_tr + n_va:].tolist())

    idx = np.arange(len(clone_ids))
    tr = idx[np.isin(clone_ids, list(tr_c))]
    va = idx[np.isin(clone_ids, list(va_c))]
    te = idx[np.isin(clone_ids, list(te_c))]
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


# ============================================================
# Data
# ============================================================
def load_data(h5_path: str, index_csv: str):
    with h5py.File(h5_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)      # (N,T,D)
        y = np.asarray(f["y"], dtype=np.int64)        # (N,)
        seq_clone = np.asarray(f["seq_clone"], dtype=np.int64) if "seq_clone" in f.keys() else None

    df = pd.read_csv(index_csv)
    if seq_clone is None:
        if "clone_id" in df.columns:
            seq_clone = df["clone_id"].to_numpy(np.int64)
        elif "seq_clone" in df.columns:
            seq_clone = df["seq_clone"].to_numpy(np.int64)
        else:
            seq_clone = np.arange(len(y), dtype=np.int64)
    return X, y, seq_clone


class SeqDataset(Dataset):
    def __init__(self, X, y, idx, mask=None):
        self.X = X
        self.y = y
        self.idx = idx
        self.mask = mask

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j].copy()
        if self.mask:
            for t in self.mask:
                x[t] = 0.0
        return torch.from_numpy(x), torch.tensor(int(self.y[j]), dtype=torch.long)


# ============================================================
# Models (must match class_175634.py)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True, bidirectional=True,
                            dropout=(dr if l > 1 else 0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(0.5 * (h[-2] + h[-1]))


class RNNModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.rnn = nn.RNN(d, h, l, batch_first=True,
                          dropout=(dr if l > 1 else 0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        return self.head(self.rnn(x)[1][-1])


class TransformerModel(nn.Module):
    def __init__(self, d, h, l, dr, nhead):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, dim_feedforward=h * 2,
                                       dropout=dr, batch_first=True),
            num_layers=l
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x):
        return self.head(self.enc(x).mean(dim=1))


def load_base_model(kind: str, setting: str, seed: int, D: int, device: torch.device):
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


def load_stacker(setting: str, seed: int):
    pkl = os.path.join(SAVED_MODELS_DIR, f"{setting}_Stacking_s{seed}.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"[ERROR] missing stacking pkl: {pkl}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def get_prob_cf(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, targs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # p(CF)
        probs.append(p)
        targs.append(y.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(targs)


# ============================================================
# 114412 ROC style helpers
# ============================================================
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


def _style_axes_114412(ax, hide_tick_labels=True):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    if hide_tick_labels:
        ax.tick_params(labelbottom=False, labelleft=False)


def plot_binary_roc_114412(y_true, y_prob, title, out_no_ext, color="#1f77b4"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    a = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    ax.plot(fpr, tpr, lw=2.8, color=color, label=f"AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], lw=1.8, color=GREY, ls="--", alpha=0.55, zorder=1)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    if SHOW_ROC_TITLE:
        ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    _style_axes_114412(ax, hide_tick_labels=True)
    ax.grid(False)

    fig.tight_layout()
    save_fig_114412(fig, out_no_ext)
    return float(a), fpr, tpr


def plot_selected_rocs_114412(curve_dict, title, out_no_ext, display_name_map=None):
    if display_name_map is None:
        display_name_map = {}

    def _draw(ax, clean):
        ax.plot([0, 1], [0, 1], lw=1.4, color=GREY, ls="--", alpha=0.5)

        for i, s in enumerate(list(curve_dict.keys())):
            fpr, tpr, a = curve_dict[s]
            disp = display_name_map.get(s, s)
            ax.plot(
                fpr,
                tpr,
                lw=2.7,
                color=LINE_COLORS[i % len(LINE_COLORS)],
                label=f"{disp} (AUC={a:.3f})",
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)

        if clean:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            if SHOW_ROC_TITLE:
                ax.set_title(title, fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)

        _style_axes_114412(ax, hide_tick_labels=True)
        ax.grid(False)

    fig_full, ax_full = plt.subplots(figsize=(7.2, 6.3))
    _draw(ax_full, clean=False)
    fig_full.tight_layout()
    save_fig_114412(fig_full, out_no_ext + "_full")

    fig_clean, ax_clean = plt.subplots(figsize=(7.2, 6.3))
    _draw(ax_clean, clean=True)
    fig_clean.tight_layout()
    save_fig_114412(fig_clean, out_no_ext + "_clean")


# ============================================================
# 2D plane (prob vs prob)
# ============================================================
def axis_label(model_tag: str) -> str:
    if AXIS_LABEL_MODE == "symbol":
        return f"{model_tag} p({POS_LABEL})"
    # paper mode (更论文式)
    return f"Predicted probability of {POS_LABEL} ({model_tag})"


def pick_axis_values(axis_name, p_bilstm, p_rnn, p_trans, p_stack):
    if axis_name == "BiLSTM":
        return p_bilstm, axis_label("BiLSTM")
    if axis_name == "RNN":
        return p_rnn, axis_label("RNN")
    if axis_name == "Trans":
        return p_trans, axis_label("Trans")
    if axis_name == "Stacking":
        return p_stack, axis_label("Stacking")
    raise ValueError(f"[ERROR] axis must be one of BiLSTM/RNN/Trans/Stacking, got {axis_name}")


def plot_2d_prob_plane(xv, yv, y_true, out_no_ext, xlab, ylab, title=""):
    xv = np.clip(np.asarray(xv, dtype=np.float64), 0.0, 1.0)
    yv = np.clip(np.asarray(yv, dtype=np.float64), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=np.int64)

    # sampling
    if N_POINTS_2D is not None and int(N_POINTS_2D) > 0 and len(y_true) > int(N_POINTS_2D):
        rng = np.random.default_rng(SAMPLE_SEED_2D)
        idx = rng.choice(np.arange(len(y_true)), size=int(N_POINTS_2D), replace=False)
        xv, yv, y_true = xv[idx], yv[idx], y_true[idx]

    m0 = (y_true == 0)  # CM
    m1 = (y_true == 1)  # CF

    fig, ax = plt.subplots(figsize=(6.6, 6.0))

    # ✅ y=x diagonal (你截图那根线)
    ax.plot([0, 1], [0, 1], lw=1.8, color=GREY, ls="--", alpha=0.55, zorder=1)

    ax.scatter(xv[m0], yv[m0], s=POINT_SIZE_2D, alpha=ALPHA_2D,
               c=COLOR_NEG, marker=MARK_NEG, edgecolors="none", label=NEG_LABEL, zorder=2)
    ax.scatter(xv[m1], yv[m1], s=POINT_SIZE_2D, alpha=ALPHA_2D,
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

    # 用 114412 的轴风格（tick 朝外 + 隐藏数字）
    _style_axes_114412(ax, hide_tick_labels=True)
    ax.grid(False)

    fig.tight_layout()
    save_fig_114412(fig, out_no_ext)


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(ROC_DIR)
    ensure_dir(D2_DIR)

    device = pick_device()
    print(f"[INFO] device={device}")

    if not os.path.isfile(TIME_SERIES_H5):
        raise FileNotFoundError(f"[ERROR] missing h5: {TIME_SERIES_H5}")
    if not os.path.isfile(INDEX_CSV):
        raise FileNotFoundError(f"[ERROR] missing index csv: {INDEX_CSV}")

    X, y, clones = load_data(TIME_SERIES_H5, INDEX_CSV)
    N, T, D = X.shape
    print(f"[INFO] X={X.shape}, y={y.shape}, clones={clones.shape}")

    # mask 与训练一致
    settings_mask = {
        "All_Days": None,
        "Obs_Day11": [T - 1],                   # mask day15
        "Obs_Day7":  [T - 2, T - 1],            # mask day11 + day15
        "Obs_Day5":  [T - 3, T - 2, T - 1],     # mask day7 + day11 + day15
        "Obs_Day3":  [T - 4, T - 3, T - 2, T - 1],
        "Obs_Day1":  list(range(2, T)),         # mask day3..day15
    }

    summary_rows = []
    selected_curve_dict = {}

    for si, setting in enumerate(SELECT_SETTINGS):
        if setting not in settings_mask:
            raise ValueError(f"[ERROR] unknown setting: {setting}")

        seed = int(SEEDS.get(setting, 2026))
        mask = settings_mask[setting]
        set_all_seeds(seed)

        if SPLIT_MODE == "clone":
            _, _, te_idx = group_split_by_clone(clones, seed, SPLIT_RATIO)
        else:
            _, _, te_idx = random_split(len(X), seed, SPLIT_RATIO)

        te_loader = DataLoader(
            SeqDataset(X, y, te_idx, mask),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        # base probs
        m_bilstm = load_base_model("BiLSTM", setting, seed, D, device)
        m_rnn    = load_base_model("RNN",    setting, seed, D, device)
        m_trans  = load_base_model("Trans",  setting, seed, D, device)

        p_bilstm, y_te = get_prob_cf(m_bilstm, te_loader, device)
        p_rnn,   _     = get_prob_cf(m_rnn,    te_loader, device)
        p_trans, _     = get_prob_cf(m_trans,  te_loader, device)

        feat3 = np.stack([p_bilstm, p_rnn, p_trans], axis=1)

        # stacking (CF prob)
        stacker = load_stacker(setting, seed)
        p_stack = stacker.predict_proba(feat3)[:, 1]
        p_stack = np.clip(p_stack, 0.0, 1.0)

        # ROC (stacking)
        roc_out = os.path.join(ROC_DIR, f"ROC_{setting}_Stacking_s{seed}")
        color = LINE_COLORS[si % len(LINE_COLORS)]
        auc_val, fpr, tpr = plot_binary_roc_114412(
            y_true=y_te,
            y_prob=p_stack,
            title=f"ROC - {setting}",
            out_no_ext=roc_out,
            color=color
        )

        # 2D plane (prob vs prob) —— 你截图那种
        xv, xlab = pick_axis_values(PLANE_X, p_bilstm, p_rnn, p_trans, p_stack)
        yv, ylab = pick_axis_values(PLANE_Y, p_bilstm, p_rnn, p_trans, p_stack)
        d2_out = os.path.join(D2_DIR, f"Pred2D_{setting}_{PLANE_X}vs{PLANE_Y}_s{seed}")
        plot_2d_prob_plane(xv, yv, y_te, d2_out, xlab, ylab, title=f"{setting}")

        # metrics
        y_pred = (p_stack >= 0.5).astype(np.int64)
        acc = accuracy_score(y_te, y_pred)
        ll  = log_loss(y_te, np.stack([1 - p_stack, p_stack], axis=1), labels=[0, 1])

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
            "SplitMode": SPLIT_MODE,
        })

        # collect for selected ROC
        if PLOT_SELECTED_ROC and (setting in set(SELECTED_ROC_SETTINGS)):
            selected_curve_dict[setting] = (fpr, tpr, float(auc_val))

    # Selected ROC (你自己选的 setting)
    if PLOT_SELECTED_ROC and SELECTED_ROC_SETTINGS:
        # 保持你给的顺序输出
        ordered = {}
        for s in SELECTED_ROC_SETTINGS:
            if s in selected_curve_dict:
                ordered[s] = selected_curve_dict[s]
        if len(ordered) > 0:
            out_no_ext = os.path.join(ROC_DIR, "ROC_Selected_Comparison_Stacking_1d3d5d15d")
            plot_selected_rocs_114412(
                ordered,
                title="scLineagetracer ROC Curves (1d/3d/5d/15d)",
                out_no_ext=out_no_ext,
                display_name_map=SELECTED_ROC_LABELS,
            )
        else:
            print("[WARN] Selected ROC: none matched (check SELECTED_ROC_SETTINGS).")

    # summary csv
    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(ROC_DIR, "plot_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Summary saved: {out_csv}")
    print(f"[DONE] Outputs: {ROC_DIR}")


if __name__ == "__main__":
    main()
