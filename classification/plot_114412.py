# -*- coding: utf-8 -*-
"""
Click-run plotting script (NO bash):

按 K_LIST + INCLUDE_ALLDAY 选择要画的 setting：
- 对每个 setting：
  1) 计算 stacking 预测
  2) 保存 OvR ROC（三类同图）
  3) 保存 3D（Correct-only，抽样50点，三角形更大更清晰）

- 另外：把这些选中的 setting 的 Macro ROC 画在一张总图上对比

风格：
- seed 固定 2026
- ROC: SCI 配色；tick 朝外；上/右无刻度；不贴边
- ROC 端点：不再无条件补 (0,1)（避免假竖线）；压缩 FPR==1 避免右侧竖线段
- 3D：只取预测正确点，抽 50；视角让三角形面积更大更清晰；不画真实点

依赖文件（必须存在）：
- PROCESSED_DIR/{OUT_PREFIX}_sequences.h5  (含 X, y, classes, time_labels)
- SAVED_MODELS_DIR 下（seed=2026）：
    <setting>_BiLSTM_s2026.pth
    <setting>_RNN_s2026.pth
    <setting>_Trans_s2026.pth
    <setting>_Stacking_s2026.pkl
"""

import os
import pickle
import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
from sklearn.preprocessing import label_binarize


# =========================
# ✅ 配置区：只改这里
# =========================
SEED = 2026

PROCESSED_DIR = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed"
OUT_PREFIX = "GSE114412_all_generated"

OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE114412"
SAVED_MODELS_DIR = os.path.join(OUT_DIR, "saved_models")

# ✅ 选多天（0-based）：默认画 w0 / w2（再配合 AllDay => w5）
K_LIST = [0, 2]

# ✅ 是否加入 allday（= All_<最后一个time_label>）
INCLUDE_ALLDAY = True

# split（固定seed的随机stratify split）
TEST_FRAC = 0.10
VAL_FRAC = 0.10
# ===== 3D 专用：按你截图的颜色与顺序 =====
CLASS_ORDER_3D = ["sc_beta", "sc_alpha", "sc_ec"]
CLASS_COLOR_3D = {
    "sc_beta":  "#D55E00",  # 橙红（接近你截图）
    "sc_alpha": "#4DAF4A",  # 绿色
    "sc_ec":    "#56B4E9",  # 浅蓝
}
# 必须与训练时一致
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.10
NHEAD = 4
# ROC plotting (clean)
# =========================
ROC_CLASS_ORDER = CLASS_ORDER_3D[:]  # sc_beta -> sc_alpha -> sc_ec
ROC_CLASS_COLOR = dict(CLASS_COLOR_3D)

# fallback colors (only used if class name not in ROC_CLASS_COLOR)
SCI_COLORS = ["#D55E00", "#4DAF4A", "#56B4E9"]

BATCH_SIZE = 512

# 输出控制
SAVE_PER_SETTING_OVR = True      # 每个 setting 的三类 OvR 同图
SAVE_PER_SETTING_3D = True       # 每个 setting 的 3D（Correct-only 50点）
N_POINTS_3D = 150

# 3D 视角：更大更清晰
VIEW_ELEV = 20
VIEW_AZIM = 110
VIEW_DIST = 10
# =========================


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


# =========================
# Load dataset
# =========================
def load_h5_dataset(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)  # (N,T,D)
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
# Models (MPS-safe Transformer)
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
        key_padding_mask = ar >= lengths[:, None]  # True = PAD
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
# ROC plotting (clean)
# =========================
SCI_COLORS = ["#377eb8", "#e41a1c", "#4daf4a"]
LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e", "#7f7f7f"]
GREY = "#444444"


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


def _style_axes(ax):
    # tick 朝外，上/右无 tick
    ax.tick_params(axis="both", direction="out", top=False, right=False)

    # ✅ 不显示刻度“数字”
    ax.tick_params(labelbottom=False, labelleft=False)


def plot_roc_all_classes_one_fig(y_true, y_prob, class_names, title, out_no_ext):
    C = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(C))

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    # plot in desired class order (sc_beta -> sc_alpha -> sc_ec)
    order = [c for c in ROC_CLASS_ORDER if c in class_names]
    order += [c for c in class_names if c not in order]

    for cname in order:
        i = class_names.index(cname)
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr, tpr = _roc_endpoints_clean(fpr, tpr)
        a = auc(fpr, tpr)
        col = ROC_CLASS_COLOR.get(cname, SCI_COLORS[i % len(SCI_COLORS)])
        ax.plot(fpr, tpr, lw=2.8, color=col,
                label=f"{cname} (AUC={a:.3f})")

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
        ax.plot(fpr, tpr, lw=2.7, color=LINE_COLORS[i % len(LINE_COLORS)],
                label=f"{s} (AUC={a:.3f})")

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


def _to_w_label(raw_label, fallback_idx: int):
    s = str(raw_label).strip().lower()
    if s.startswith("w"):
        s = s[1:]
    try:
        v = float(s)
        if np.isfinite(v):
            return f"w{int(round(v))}"
    except Exception:
        pass

    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        try:
            return f"w{int(digits)}"
        except Exception:
            pass
    return f"w{int(fallback_idx)}"


def plot_macro_multi_settings_full_clean(
    macro_dict,
    setting_order,
    title,
    out_no_ext,
    display_name_map=None,
):
    if display_name_map is None:
        display_name_map = {}

    def _draw(ax, clean: bool):
        ax.plot([0, 1], [0, 1], lw=1.4, color=GREY, ls="--", alpha=0.5)

        for i, s in enumerate(setting_order):
            if s not in macro_dict:
                continue
            fpr, tpr, a = macro_dict[s]
            disp = display_name_map.get(s, s)
            label = None if clean else f"{disp} (AUC={a:.3f})"
            ax.plot(
                fpr,
                tpr,
                lw=2.7,
                color=LINE_COLORS[i % len(LINE_COLORS)],
                label=label,
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
            ax.set_title(title, fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)

        _style_axes(ax)
        ax.grid(False)

    fig_full, ax_full = plt.subplots(figsize=(7.2, 6.3))
    _draw(ax_full, clean=False)
    fig_full.tight_layout()
    fig_full.savefig(out_no_ext + "_full.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig_full.savefig(out_no_ext + "_full.pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig_full)

    fig_clean, ax_clean = plt.subplots(figsize=(7.2, 6.3))
    _draw(ax_clean, clean=True)
    fig_clean.tight_layout()
    fig_clean.savefig(out_no_ext + "_clean.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig_clean.savefig(out_no_ext + "_clean.pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig_clean)


# =========================
# 3D: correct-only 50, bigger simplex
# =========================
def plot_3d_simplex_correct50_big(
    y_true, y_prob, class_names,
    title, out_no_ext,
    n_points=50, seed=SEED,
    elev=35, azim=45, dist=7.0
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    rng = np.random.default_rng(seed)

    P = np.asarray(y_prob, dtype=np.float64)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

    y_true = np.asarray(y_true, dtype=np.int64)
    pred = P.argmax(axis=1)
    correct = (pred == y_true)

    P = P[correct]
    y_true = y_true[correct]
    if len(y_true) == 0:
        print("[WARN] No correct points for 3D. Skip.")
        return

    idx_all = np.arange(len(y_true))
    C = len(class_names)

    if len(idx_all) > n_points:
        keep = []
        for k in range(C):
            ik = idx_all[y_true == k]
            if len(ik) == 0:
                continue
            nk = max(1, int(round(n_points * (len(ik) / float(len(idx_all))))))
            nk = min(nk, len(ik))
            keep.append(rng.choice(ik, size=nk, replace=False))
        keep = np.unique(np.concatenate(keep)) if len(keep) else rng.choice(idx_all, size=n_points, replace=False)
        if len(keep) > n_points:
            keep = rng.choice(keep, size=n_points, replace=False)

        P = P[keep]
        y_true = y_true[keep]

    fig = plt.figure(figsize=(7.6, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = True
            axis.pane.set_alpha(0.14)
            axis.pane.set_edgecolor("#111111")
        except Exception:
            pass
    ax.grid(False)

    tri = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]], dtype=float)
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], lw=2.6, color="#111111", alpha=0.98)

    markers = {"sc_beta": "o", "sc_alpha": "^", "sc_ec": "s"}

    # 只画你数据里实际存在的类，并按你指定顺序
    order = [c for c in CLASS_ORDER_3D if c in class_names]
    # 防止 class_names 里还有其他类（一般不会），追加到末尾
    order += [c for c in class_names if c not in order]

    for cname in order:
        k = class_names.index(cname)  # y_true 的编号仍由 class_names 决定
        m = (y_true == k)
        if m.sum() == 0:
            continue

        ax.scatter(
            P[m, 0], P[m, 1], P[m, 2],
            s=60, alpha=0.90,
            c=CLASS_COLOR_3D.get(cname, "#666666"),
            marker=markers.get(cname, "o"),
            edgecolors="#111111",
            linewidths=0.35,
            label=f"{cname} (n={int(m.sum())})"
        )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    #ax.set_xlabel(class_names[0], labelpad=10)
    #ax.set_ylabel(class_names[1], labelpad=10)
    #ax.set_zlabel(class_names[2], labelpad=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)
    try:
        ax.dist = dist
    except Exception:
        pass

    #ax.set_title(f"{title} | Correct-only, n={len(y_true)}", pad=12, fontsize=12)
    #ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


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
    print(f"[INFO] X: N={N}, T={T}, D={D}")
    print(f"[INFO] classes={class_names}")
    print(f"[INFO] time_labels={time_labels}")

    # ✅ build settings strictly from K_LIST + INCLUDE_ALLDAY
    settings = []
    if not isinstance(K_LIST, (list, tuple)) or len(K_LIST) == 0:
        raise ValueError("[ERROR] K_LIST must be a non-empty list like [0,3].")

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

    # fixed split
    _, _, te_idx = stratified_split(y, seed=SEED, test_frac=TEST_FRAC, val_frac=VAL_FRAC)

    roc_dir = os.path.join(OUT_DIR, "roc")
    d3_dir = os.path.join(roc_dir, "3d")
    upto_dir = os.path.join(roc_dir, "uptoday")
    ensure_dir(roc_dir)
    ensure_dir(d3_dir)
    ensure_dir(upto_dir)

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

    for setting, keep_len in settings:
        print(f"\n[RUN] setting={setting}, keep_len={keep_len}")

        te_loader = DataLoader(
            SeqDataset(X, y, te_idx, keep_len),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pad
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
                title=f"Stacking ROC (OvR) — {setting} — seed={SEED}",
                out_no_ext=os.path.join(upto_dir, f"ROC_{setting}_AllClasses"),
            )

        if SAVE_PER_SETTING_3D:
            plot_3d_simplex_correct50_big(
                y_true=y_te,
                y_prob=p_stack,
                class_names=class_names,
                title=f"3D Simplex Probabilities — {setting} — True-colored",
                out_no_ext=os.path.join(d3_dir, f"Pred3D_{setting}_TrueColor_Correct50"),
                n_points=N_POINTS_3D,
                seed=SEED,
                elev=VIEW_ELEV,
                azim=VIEW_AZIM,
                dist=VIEW_DIST,
            )

    # combined Macro ROC only for selected settings
    out_no_ext = os.path.join(roc_dir, f"ROC_Macro_SelectedSettings_Seed{SEED}")
    title = f"Stacking Macro ROC (Selected UpTo + AllDay) — seed={SEED}"
    plot_macro_multi_settings(macro_dict, title, out_no_ext)

    # ✅ full/clean: one figure with w0, w1, w2, w5
    compare_settings = []
    display_name_map = {}
    if T >= 1:
        s = f"UpTo_{time_labels[0]}"
        compare_settings.append(s)
        display_name_map[s] = _to_w_label(time_labels[0], 0)
    if T >= 2:
        s = f"UpTo_{time_labels[1]}"
        compare_settings.append(s)
        display_name_map[s] = _to_w_label(time_labels[1], 1)
    if T >= 3:
        s = f"UpTo_{time_labels[2]}"
        compare_settings.append(s)
        display_name_map[s] = _to_w_label(time_labels[2], 2)
    s_all = f"All_{time_labels[-1]}"
    compare_settings.append(s_all)
    display_name_map[s_all] = _to_w_label(time_labels[-1], T - 1)

    compare_settings = [s for s in compare_settings if s in macro_dict]
    if len(compare_settings) >= 2:
        show_labels = [display_name_map.get(s, s) for s in compare_settings]
        label_text = " / ".join(show_labels)
        name_suffix = "".join(
            "".join(ch for ch in lbl.upper() if ch.isalnum()) for lbl in show_labels
        )
        out_no_ext_w = os.path.join(roc_dir, f"ROC_Macro_{name_suffix}_Seed{SEED}")
        title_w = f"Stacking Macro ROC ({label_text}) — seed={SEED}"
        plot_macro_multi_settings_full_clean(
            macro_dict=macro_dict,
            setting_order=compare_settings,
            title=title_w,
            out_no_ext=out_no_ext_w,
            display_name_map=display_name_map,
        )
        print(f"[DONE] {label_text} ROC full:  {out_no_ext_w}_full.png/.pdf")
        print(f"[DONE] {label_text} ROC clean: {out_no_ext_w}_clean.png/.pdf")
    else:
        print("[WARN] Skip selected W* ROC full/clean: not enough settings available.")

    print(f"\n[DONE] Combined Macro ROC saved: {out_no_ext}.png/pdf")
    print(f"[DONE] Per-setting OvR saved to: {upto_dir}")
    print(f"[DONE] Per-setting 3D saved to: {d3_dir}")


if __name__ == "__main__":
    main()
