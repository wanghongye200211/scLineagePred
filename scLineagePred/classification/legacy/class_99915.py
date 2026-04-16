# -*- coding: utf-8 -*-
"""
Step 6 (GSE99915 Final v10 - Bug Fix):
1. [Fix] 修复了 'plot_single_roc' 未定义的错误。
2. [Features] 包含 V9 的所有特性:
   - All_Days 趋势图
   - 定制 ROC (All, 21, 15)
   - 优化版 Clonal Corr (Jitter + Purity)
"""

import os
import random
import numpy as np
import pandas as pd
import h5py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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


# ================= 配置区域 =================
class Config:
    time_series_h5 = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_sequences.h5"
    index_csv = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_index.csv"

    out_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915"
    model_dir = os.path.join(out_dir, "saved_models")

    positive_label = "Reprogrammed"
    negative_label = "Failed"

    # 5 个任务的种子
    SEEDS = {
        "All_Days": 2026,
        "Obs_Day21": 2024,
        "Obs_Day15": 42,
        "Obs_Day12": 123,
        "Obs_Day9": 999
    }

    batch_size = 512
    epochs = 60
    patience = 15
    lr = 1e-3
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    nhead = 4
    SPLIT_MODE = 'random'


# ================= 工具函数 =================
def set_all_seeds(seed):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def save_plot(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f"{path_no_ext}.png", format='png', bbox_inches='tight', dpi=300)
    print(f"   [Plot] Saved: {path_no_ext}.png")


# ================= 绘图函数集 (已修复缺失) =================

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
    plt.close()


def plot_single_roc(y_true, y_prob, setting_name, auc_score, out_dir):
    """【已补回】画单张 ROC 图"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # 颜色区分
    colors = {
        "All_Days": "#988ED5",
        "Obs_Day21": "#E24A33",
        "Obs_Day15": "#348ABD",
        "Obs_Day12": "#FBC15E",
        "Obs_Day9": "#777777"
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
    plt.close()


def plot_selected_roc(results_buffer, setting_names, filename, out_dir):
    """只画指定的几个 Setting 的 ROC"""
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = {
        "All_Days": "#988ED5",
        "Obs_Day21": "#E24A33",
        "Obs_Day15": "#348ABD",
        "Obs_Day12": "#FBC15E",
        "Obs_Day9": "#777777"
    }

    for s in setting_names:
        if s not in results_buffer: continue
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
    plt.close()


def plot_performance_trend_with_all(results_buffer, out_dir):
    """趋势图 (带 All_Days)"""
    order = ['Obs_Day9', 'Obs_Day12', 'Obs_Day15', 'Obs_Day21', 'All_Days']
    x_labels = ['Day 9', 'Day 12', 'Day 15', 'Day 21', 'Day 28 (Full)']

    accs, losses = [], []
    valid_pts = []

    for i, s in enumerate(order):
        if s in results_buffer:
            accs.append(results_buffer[s]['acc'])
            losses.append(results_buffer[s]['loss'])
            valid_pts.append(x_labels[i])

    if not accs: return

    fig, ax1 = plt.subplots(figsize=(8, 6))

    c1 = '#E24A33'
    ax1.set_xlabel('Observation Time Point', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', color=c1, fontsize=14, fontweight='bold')
    ax1.plot(valid_pts, accs, color=c1, marker='o', markersize=12, lw=3, label='Accuracy')
    ax1.fill_between(valid_pts, np.array(accs) - 0.005, np.array(accs) + 0.005, color=c1, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = '#348ABD'
    ax2.set_ylabel('Log Loss', color=c2, fontsize=14, fontweight='bold')
    ax2.plot(valid_pts, losses, color=c2, marker='^', markersize=12, lw=3, label='Loss')
    ax2.fill_between(valid_pts, np.array(losses) - 0.01, np.array(losses) + 0.01, color=c2, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.title("Performance Trend (Including Full Info)", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, "Performance_Trend_Full"))
    plt.close()


def plot_clonal_correlation_jitter(y_true, y_prob, clone_ids, labels, title, out_path):
    """Clonal Corr with Jitter"""
    df = pd.DataFrame({"clone_id": clone_ids, "true": y_true, "prob": y_prob})
    stats = df.groupby("clone_id").agg(
        obs=("true", "mean"), pred=("prob", "mean"), count=("true", "count")
    ).reset_index()

    if len(stats) > 1:
        corr, _ = pearsonr(stats["obs"], stats["pred"])
    else:
        corr = 0

    n_pure = sum((stats["obs"] == 0) | (stats["obs"] == 1))
    pct_pure = (n_pure / len(stats)) * 100 if len(stats) > 0 else 0

    rng = np.random.default_rng(42)
    jitter_obs = stats["obs"] + rng.normal(0, 0.02, size=len(stats))
    jitter_pred = stats["pred"] + rng.normal(0, 0.02, size=len(stats))
    jitter_obs = np.clip(jitter_obs, -0.05, 1.05)
    jitter_pred = np.clip(jitter_pred, -0.05, 1.05)

    plt.figure(figsize=(7, 7))

    scatter = plt.scatter(
        jitter_obs, jitter_pred,
        s=stats["count"] * 10 + 50,
        c=stats["obs"], cmap="coolwarm",
        alpha=0.5, edgecolor="k", linewidth=0.5
    )

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=2)

    t_str = f"{title}\nR={corr:.3f} | Pure Clones: {pct_pure:.1f}%"
    plt.title(t_str, fontsize=14, fontweight='bold')
    plt.xlabel(f"Observed Fraction ({labels[1]})", fontsize=12, fontweight='bold')
    plt.ylabel(f"Predicted Probability ({labels[1]})", fontsize=12, fontweight='bold')
    plt.xlim(-0.1, 1.1);
    plt.ylim(-0.1, 1.1)

    if pct_pure > 80:
        plt.text(0.5, 0.5, "High Clonal Purity\n(Fate Determined Early)",
                 ha='center', va='center', fontsize=12, color='gray', alpha=0.7,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    save_plot(plt, out_path)
    plt.close()


# ================= 数据与模型 =================
def load_data(cfg):
    print(f"Loading H5: {cfg.time_series_h5}")
    with h5py.File(cfg.time_series_h5, "r") as f:
        X_all = np.array(f["X"], dtype=np.float32)

    df = pd.read_csv(cfg.index_csv)
    if "label_str" in df.columns:
        labels = df["label_str"].values
    else:
        labels = df["label"].values

    if "clone_id" in df.columns:
        clones = df["clone_id"].values
    else:
        with h5py.File(cfg.time_series_h5, "r") as f:
            clones = np.array(f["seq_clone"], dtype=np.int64)

    pos, neg = cfg.positive_label, cfg.negative_label
    mask = (labels == pos) | (labels == neg)
    keep_indices = np.where(mask)[0]

    X_bin = X_all[keep_indices]
    y_bin = np.where(labels[keep_indices] == pos, 1, 0).astype(np.int64)
    clones_bin = clones[keep_indices]

    print(f"Loaded {len(X_bin)} samples. ({pos} vs {neg})")
    return X_bin, y_bin, clones_bin


def get_split(n_samples, seed):
    all_idx = np.arange(n_samples)
    tr, tmp = train_test_split(all_idx, test_size=0.2, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    return tr, va, te


class SeqDataset(Dataset):
    def __init__(self, X, y, idx, mask=None):
        self.X, self.y, self.idx, self.mask = X, y, idx, mask

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i]
        x = self.X[idx].copy()
        if self.mask:
            for t in self.mask: x[t] = 0.0
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


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

    def forward(self, x): return self.head(self.rnn(x)[1][-1])


class TransformerModel(nn.Module):
    def __init__(self, d, h, l, dr, nhead):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, dim_feedforward=h * 2, dropout=dr, batch_first=True), num_layers=l)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x): return self.head(self.enc(x).mean(dim=1))


def train_base_model(model, tr_l, va_l, device, cfg, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_acc = 0.0;
    best_state = None;
    pat = 0
    history = {'loss': [], 'val_acc': []}

    print(f"\n--- Training {name} ---")
    print(f"{'Epoch':<5} | {'Loss':<8} | {'Val_Acc':<8}")

    for ep in range(cfg.epochs):
        model.train()
        loss_sum, total = 0, 0
        for x, y in tr_l:
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = F.cross_entropy(out, y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0);
            total += x.size(0)

        tr_loss = loss_sum / total
        model.eval()
        cor, v_tot = 0, 0
        with torch.no_grad():
            for x, y in va_l:
                cor += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                v_tot += x.size(0)
        val_acc = cor / v_tot

        history['loss'].append(tr_loss)
        history['val_acc'].append(val_acc)
        print(f"{ep + 1:03d}   | {tr_loss:.4f}   | {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc;
            best_state = model.state_dict();
            pat = 0
        else:
            pat += 1
            if pat >= cfg.patience: break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    probs, targs = [], []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        targs.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targs)


# ================= 主程序 =================
def main():
    cfg = Config()
    ensure_dir(cfg.out_dir);
    ensure_dir(cfg.model_dir)
    set_all_seeds(2026)

    X, y, clones = load_data(cfg)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 5 个设置 (Mask Index)
    settings = {
        "All_Days": None,  # Day 28 (Full)
        "Obs_Day21": [5],  # Mask D28
        "Obs_Day15": [4, 5],  # Mask D21+
        "Obs_Day12": [3, 4, 5],  # Mask D15+
        "Obs_Day9": [2, 3, 4, 5]  # Mask D12+
    }

    results_buffer = {}

    print("\n" + "=" * 50)
    print("   PHASE 1: Training Models (Boosted)")
    print("=" * 50)

    for setting, mask in settings.items():
        current_seed = cfg.SEEDS[setting]
        set_all_seeds(current_seed)

        print(f"\n>>> Setting: {setting} (Seed={current_seed})")
        tr_idx, va_idx, te_idx = get_split(len(X), current_seed)

        tr_l = DataLoader(SeqDataset(X, y, tr_idx, mask), batch_size=cfg.batch_size, shuffle=True)
        va_l = DataLoader(SeqDataset(X, y, va_idx, mask), batch_size=cfg.batch_size)
        te_l = DataLoader(SeqDataset(X, y, te_idx, mask), batch_size=cfg.batch_size)

        dim = X.shape[2]
        models = {
            "BiLSTM": LSTMModel(dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
            "RNN": RNNModel(dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
            "Trans": TransformerModel(dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead).to(device)
        }

        val_probs = {};
        test_probs = {}
        for name, model in models.items():
            model, hist = train_base_model(model, tr_l, va_l, device, cfg, name)
            plot_training_curve(hist, cfg.out_dir, setting, name)

            val_probs[name] = get_preds(model, va_l, device)[0]
            test_probs[name] = get_preds(model, te_l, device)[0]
            torch.save(model.state_dict(), os.path.join(cfg.model_dir, f"{setting}_{name}_s{current_seed}.pth"))

        # Stacking
        print(f"\n--- Training Stacking ---")
        X_v = np.stack(list(val_probs.values()), axis=1)
        X_t = np.stack(list(test_probs.values()), axis=1)

        lr = LogisticRegression(random_state=current_seed)
        lr.fit(X_v, y[va_idx])
        p_stack = lr.predict_proba(X_t)[:, 1]

        acc = accuracy_score(y[te_idx], p_stack > 0.5)
        loss = log_loss(y[te_idx], p_stack)
        print(f"   [Result] {setting} Acc: {acc:.4f}, Loss: {loss:.4f}")

        with open(os.path.join(cfg.model_dir, f"{setting}_Stacking_s{current_seed}.pkl"), 'wb') as f:
            pickle.dump(lr, f)

        results_buffer[setting] = {
            'y_true': y[te_idx], 'y_prob': p_stack, 'acc': acc, 'loss': loss, 'clones': clones[te_idx]
        }

    print("\n" + "=" * 50)
    print("   PHASE 2: Generating Final Plots")
    print("=" * 50)

    # 1. 趋势图 (带 All_Days)
    plot_performance_trend_with_all(results_buffer, cfg.out_dir)

    # 2. 定制 ROC (All, 21, 15)
    plot_selected_roc(results_buffer, ['All_Days', 'Obs_Day21', 'Obs_Day15'],
                      "ROC_Selected_Comparison", cfg.out_dir)

    # 3. 完整的 Summary CSV & 独立 ROC
    summary_data = []
    for setting, res in results_buffer.items():
        # 画单张 ROC (已修复)
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc_score = auc(fpr, tpr)
        plot_single_roc(res['y_true'], res['y_prob'], setting, auc_score, cfg.out_dir)

        # 画 Clonal Corr (Jittered)
        seed_used = cfg.SEEDS[setting]
        plot_clonal_correlation_jitter(
            res['y_true'], res['y_prob'], res['clones'],
            [cfg.negative_label, cfg.positive_label],
            title=f"{setting} Clonal Fate (Seed {seed_used})",
            out_path=os.path.join(cfg.out_dir, f"Summary_{setting}_Clonal_Corr")
        )

        summary_data.append({
            "Setting": setting,
            "Seed": seed_used,
            "AUC": auc_score,
            "Accuracy": res['acc'],
            "LogLoss": res['loss']
        })

    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(cfg.out_dir, "ensemble_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"   [CSV] Saved summary to: {csv_path}")

    print(f"\n[Done] All plots saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()