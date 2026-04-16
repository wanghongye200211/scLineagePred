# -*- coding: utf-8 -*-
"""
Step 6 (GSE175634 Final v10 style) - Classification (CM vs CF)
要求：
1) 不改输入文件名（time_series_h5 / index_csv 固定为你给的）
2) 无样本权重（NO sample weight）
3) 训练过程要 print（每个 epoch 打印 loss/val_acc）
4) All_Days = 所有天不做掩码（mask=None）
5) 其它设置 = prefix-k -> 预测最后一天(day15)（mask future incl. day15）
6) 画图/保存风格与以前一致：Training curve / ROC / Selected ROC / Trend / Clonal Corr / summary csv / saved_models
"""

import os
import random
import numpy as np
import pandas as pd
import h5py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns  # 保留依赖以匹配旧脚本

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss
from scipy.stats import pearsonr


# ================= 绘图风格 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300


# ================= 配置区域（只改 out_dir 就行） =================
class Config:
    # !!! 不允许改这两个文件名（按你的要求固定） !!!
    time_series_h5 = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_sequences.h5"
    index_csv      = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_index.csv"

    out_dir   = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF"
    model_dir = os.path.join(out_dir, "saved_models")

    negative_label = "CM"
    positive_label = "CF"

    # seeds（保持你以前多 setting 的结构）
    SEEDS = {
        "All_Days": 2024,
        "Obs_Day11": 2024,
        "Obs_Day7": 2024,
        "Obs_Day5": 2024,
        "Obs_Day3": 2024,
        "Obs_Day1": 2024,
    }

    batch_size = 512
    epochs = 60
    patience = 15
    lr = 1e-3
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    nhead = 4

    # 优化（不引入样本权重）
    USE_CLASS_WEIGHT = False       # 类别不平衡时很有用；不想用就 False（其余都不变）
    GRAD_CLIP_NORM = 1.0           # 防止发散

    # split
    # 默认 random（跟你以前一致）
    SPLIT_MODE = "random"          # "random" 或 "clone"
    SPLIT_RATIO = (0.8, 0.1, 0.1)

    num_workers = 0                # Mac/MPS 建议 0


# ================= 工具函数 =================
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_plot(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f"{path_no_ext}.png", format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"   [Plot] Saved: {path_no_ext}.png")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def compute_class_weight(y_train):
    # balanced CE weight
    n = len(y_train)
    c0 = max(int((y_train == 0).sum()), 1)
    c1 = max(int((y_train == 1).sum()), 1)
    w0 = n / (2.0 * c0)
    w1 = n / (2.0 * c1)
    return torch.tensor([w0, w1], dtype=torch.float32)


# ================= 绘图函数集（保持你以前一致） =================
def plot_training_curve(history, out_dir, setting, model_name):
    epochs = range(1, len(history['loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))

    c1 = '#E24A33'
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', color=c1, fontsize=12, fontweight='bold')
    ax1.plot(epochs, history['loss'], color=c1, lw=2)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = '#348ABD'
    ax2.set_ylabel('Accuracy', color=c2, fontsize=12, fontweight='bold')
    ax2.plot(epochs, history['val_acc'], color=c2, lw=2)
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.title(f"{setting} - {model_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, f"Training_{setting}_{model_name}"))


def plot_single_roc(y_true, y_prob, setting_name, auc_score, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = {
        "All_Days": "#988ED5",
        "Obs_Day11": "#E24A33",
        "Obs_Day7": "#348ABD",
        "Obs_Day5": "#FBC15E",
        "Obs_Day3": "#777777",
        "Obs_Day1": "#8E8E8E"
    }
    color = colors.get(setting_name, 'k')

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, lw=3, color=color, label=f"AUC = {auc_score:.4f}")

    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    ax.set_title(f"ROC - {setting_name}", fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, frameon=False)

    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, f"ROC_{setting_name}"))


def plot_selected_roc(results_buffer, setting_names, filename, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = {
        "All_Days": "#988ED5",
        "Obs_Day11": "#E24A33",
        "Obs_Day7": "#348ABD",
        "Obs_Day5": "#FBC15E",
        "Obs_Day3": "#777777",
        "Obs_Day1": "#8E8E8E"
    }

    for s in setting_names:
        if s not in results_buffer:
            continue
        data = results_buffer[s]
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        score = auc(fpr, tpr)

        lw = 3.5 if s == "All_Days" else 2.5
        ax.plot(fpr, tpr, lw=lw, color=colors.get(s, 'k'),
                label=f"{s} (AUC={score:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    ax.set_title("Selected ROC Curves", fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, frameon=False)

    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, filename))


def plot_performance_trend_with_all(results_buffer, out_dir):
    order = ["Obs_Day1", "Obs_Day3", "Obs_Day5", "Obs_Day7", "Obs_Day11", "All_Days"]
    x_labels = ["<=day1", "<=day3", "<=day5", "<=day7", "<=day11", "All"]

    xs, accs, losses = [], [], []
    for s, lab in zip(order, x_labels):
        if s in results_buffer:
            xs.append(lab)
            accs.append(results_buffer[s]['acc'])
            losses.append(results_buffer[s]['loss'])

    if not xs:
        return

    fig, ax1 = plt.subplots(figsize=(8, 6))

    c1 = '#E24A33'
    ax1.set_xlabel('Observed Prefix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', color=c1, fontsize=14, fontweight='bold')
    ax1.plot(xs, accs, color=c1, marker='o', markersize=12, lw=3)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = '#348ABD'
    ax2.set_ylabel('Log Loss', color=c2, fontsize=14, fontweight='bold')
    ax2.plot(xs, losses, color=c2, marker='^', markersize=12, lw=3)
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.title("Performance Trend (prefix -> day15 fate)", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, "Performance_Trend_Full"))


def plot_clonal_correlation_jitter(y_true, y_prob, clone_ids, labels, title, out_path):
    df = pd.DataFrame({"clone_id": clone_ids, "true": y_true, "prob": y_prob})
    stats = df.groupby("clone_id").agg(
        obs=("true", "mean"), pred=("prob", "mean"), count=("true", "count")
    ).reset_index()

    if len(stats) > 1:
        corr, _ = pearsonr(stats["obs"], stats["pred"])
    else:
        corr = 0.0

    rng = np.random.default_rng(42)
    jitter_obs = np.clip(stats["obs"] + rng.normal(0, 0.02, size=len(stats)), -0.05, 1.05)
    jitter_pred = np.clip(stats["pred"] + rng.normal(0, 0.02, size=len(stats)), -0.05, 1.05)

    fig = plt.figure(figsize=(7, 7))
    plt.scatter(
        jitter_obs, jitter_pred,
        s=stats["count"] * 10 + 50,
        c=stats["obs"], cmap="coolwarm",
        alpha=0.5, edgecolor="k", linewidth=0.5
    )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=2)

    t_str = f"{title}\nR={corr:.3f}"
    plt.title(t_str, fontsize=14, fontweight='bold')
    plt.xlabel(f"Observed Fraction ({labels[1]})", fontsize=12, fontweight='bold')
    plt.ylabel(f"Predicted Probability ({labels[1]})", fontsize=12, fontweight='bold')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    save_plot(fig, out_path)


# ================= 数据与模型 =================
def load_data(cfg: Config):
    print(f"[INFO] Loading H5: {cfg.time_series_h5}")
    with h5py.File(cfg.time_series_h5, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        y = np.array(f["y"], dtype=np.int64)
        seq_clone = np.array(f["seq_clone"], dtype=np.int64) if "seq_clone" in f.keys() else None

    df = pd.read_csv(cfg.index_csv)

    if seq_clone is None:
        if "clone_id" in df.columns:
            seq_clone = df["clone_id"].to_numpy(np.int64)
        elif "seq_clone" in df.columns:
            seq_clone = df["seq_clone"].to_numpy(np.int64)
        else:
            seq_clone = np.arange(len(y), dtype=np.int64)

    print(f"[INFO] X={X.shape}, y={y.shape}, clones={seq_clone.shape}")
    print("[INFO] Label counts:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})
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
        j = self.idx[i]
        x = self.X[j].copy()
        if self.mask:
            for t in self.mask:
                x[t] = 0.0
        return torch.from_numpy(x), torch.tensor(int(self.y[j]), dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True, bidirectional=True, dropout=(dr if l > 1 else 0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(0.5 * (h[-2] + h[-1]))


class RNNModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.rnn = nn.RNN(d, h, l, batch_first=True, dropout=(dr if l > 1 else 0))
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


def train_base_model(model, tr_l, va_l, device, cfg, name, class_w=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_acc = 0.0
    best_state = None
    pat = 0
    history = {'loss': [], 'val_acc': []}

    print(f"\n--- Training {name} ---")
    print(f"{'Epoch':<5} | {'Loss':<8} | {'Val_Acc':<8}")

    for ep in range(cfg.epochs):
        model.train()
        loss_sum, total = 0.0, 0

        for x, y in tr_l:
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = F.cross_entropy(out, y.to(device), weight=class_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

            loss_sum += float(loss.item()) * x.size(0)
            total += x.size(0)

        tr_loss = loss_sum / max(total, 1)

        model.eval()
        cor, v_tot = 0, 0
        with torch.no_grad():
            for x, y in va_l:
                cor += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                v_tot += x.size(0)

        val_acc = cor / max(v_tot, 1)

        history['loss'].append(tr_loss)
        history['val_acc'].append(val_acc)
        print(f"{ep + 1:03d}   | {tr_loss:.4f}   | {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= cfg.patience:
                print(f"[EarlyStop] {name} stop at epoch {ep+1}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    probs, targs = [], []
    for x, y in loader:
        out = model(x.to(device))
        p = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        targs.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targs)


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model_dir)

    X, y, clones = load_data(cfg)
    device = get_device()
    print(f"[INFO] device={device}")

    T = X.shape[1]
    if T != 7:
        print(f"[WARN] Expected T=7 (day0/day1/day3/day5/day7/day11/day15), got T={T}. Still run with index-based masks.")

    # All_Days = no mask (你要求)
    settings = {
        "All_Days": None,
        "Obs_Day11": [T - 1],                   # mask day15
        "Obs_Day7":  [T - 2, T - 1],            # mask day11 + day15
        "Obs_Day5":  [T - 3, T - 2, T - 1],     # mask day7 + day11 + day15
        "Obs_Day3":  [T - 4, T - 3, T - 2, T - 1],
        "Obs_Day1":  list(range(2, T)),         # mask day3..day15
    }

    results_buffer = {}

    print("\n" + "=" * 50)
    print("   PHASE 1: Training Models (Boosted, NO sample weight)")
    print("=" * 50)

    for setting, mask in settings.items():
        seed = cfg.SEEDS.get(setting, 2026)
        set_all_seeds(seed)

        print(f"\n>>> Setting: {setting} (Seed={seed}) | mask={mask}")

        if cfg.SPLIT_MODE == "clone":
            tr_idx, va_idx, te_idx = group_split_by_clone(clones, seed, cfg.SPLIT_RATIO)
        else:
            tr_idx, va_idx, te_idx = random_split(len(X), seed, cfg.SPLIT_RATIO)

        print(f"    Split sizes: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")
        print(f"    Train label counts: CM={int((y[tr_idx]==0).sum())}, CF={int((y[tr_idx]==1).sum())}")

        tr_l = DataLoader(SeqDataset(X, y, tr_idx, mask), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        va_l = DataLoader(SeqDataset(X, y, va_idx, mask), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        te_l = DataLoader(SeqDataset(X, y, te_idx, mask), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        class_w = None
        if cfg.USE_CLASS_WEIGHT:
            class_w = compute_class_weight(y[tr_idx]).to(device)
            print(f"    Class weight: w_CM={float(class_w[0]):.4f}, w_CF={float(class_w[1]):.4f}")

        dim = X.shape[2]
        models = {
            "BiLSTM": LSTMModel(dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
            "RNN": RNNModel(dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
            "Trans": TransformerModel(dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead).to(device)
        }

        val_probs, test_probs = {}, {}
        y_va_ref, y_te_ref = None, None

        for name, model in models.items():
            model, hist = train_base_model(model, tr_l, va_l, device, cfg, name, class_w=class_w)
            plot_training_curve(hist, cfg.out_dir, setting, name)

            p_va, y_va = get_probs(model, va_l, device)
            p_te, y_te = get_probs(model, te_l, device)
            val_probs[name] = p_va
            test_probs[name] = p_te
            y_va_ref, y_te_ref = y_va, y_te

            torch.save(model.state_dict(), os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth"))

        print(f"\n--- Training Stacking ---")
        Xv = np.stack([val_probs[k] for k in models.keys()], axis=1)
        Xt = np.stack([test_probs[k] for k in models.keys()], axis=1)

        lr = LogisticRegression(random_state=seed, max_iter=300)
        lr.fit(Xv, y_va_ref)
        p_stack = lr.predict_proba(Xt)[:, 1]

        acc = accuracy_score(y_te_ref, p_stack > 0.5)
        loss = log_loss(y_te_ref, p_stack)
        fpr, tpr, _ = roc_curve(y_te_ref, p_stack)
        auc_score = auc(fpr, tpr)

        with open(os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl"), "wb") as f:
            pickle.dump(lr, f)

        results_buffer[setting] = {
            "y_true": y_te_ref,
            "y_prob": p_stack,
            "acc": float(acc),
            "loss": float(loss),
            "auc": float(auc_score),
            "clones": clones[te_idx]
        }

        plot_single_roc(y_te_ref, p_stack, setting, auc_score, cfg.out_dir)
        plot_clonal_correlation_jitter(
            y_te_ref, p_stack, clones[te_idx],
            [cfg.negative_label, cfg.positive_label],
            title=f"{setting} Clonal Fate (Seed {seed})",
            out_path=os.path.join(cfg.out_dir, f"Summary_{setting}_Clonal_Corr")
        )

        print(f"   [Result] {setting} AUC={auc_score:.4f} Acc={acc:.4f} LogLoss={loss:.4f}")

    print("\n" + "=" * 50)
    print("   PHASE 2: Final Plots + Summary")
    print("=" * 50)

    plot_selected_roc(results_buffer, ["All_Days", "Obs_Day11", "Obs_Day5"], "ROC_Selected_Comparison", cfg.out_dir)
    plot_performance_trend_with_all(results_buffer, cfg.out_dir)

    summary = []
    for k, v in results_buffer.items():
        summary.append({
            "Setting": k,
            "Seed": cfg.SEEDS.get(k, 0),
            "AUC": v["auc"],
            "Accuracy": v["acc"],
            "LogLoss": v["loss"],
            "SplitMode": cfg.SPLIT_MODE,
            "UseClassWeight": cfg.USE_CLASS_WEIGHT
        })

    df = pd.DataFrame(summary).sort_values("AUC", ascending=False)
    out_csv = os.path.join(cfg.out_dir, "ensemble_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved: {out_csv}")
    print(f"[DONE] All plots/models saved in: {cfg.out_dir}")


if __name__ == "__main__":
    main()
