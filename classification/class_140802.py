# -*- coding: utf-8 -*-
"""
Step 6 (Final v7 - Separate ROCs & CSV Summary):
1. [ROC] "只写day2的来一张，只写day4的来一张" -> 生成独立的 ROC 文件。
2. [CSV] "csv总结的是ensemble的" -> 自动保存 Stacking 模型的指标到 CSV。
3. [Logic] 保持 Multi-Seed 和 Deferred Plotting 逻辑不变。
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

# ================= 绘图风格全局设置 =================
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
    time_series_h5 = "/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_DeepLineage_sequences.h5"
    index_csv = "/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_DeepLineage_index.csv"
    out_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7"
    model_dir = os.path.join(out_dir, "saved_models")

    positive_label = "Monocyte"
    negative_label = "Neutrophil"

    # 独立种子
    SEEDS = {
        "Day2_Only": 42,      # Seed 1
        "Day2_Day4": 2024,    # Seed 2
        "All_Days": 2024      # Seed 2
    }

    batch_size = 512
    epochs = 100
    patience = 10
    lr = 1e-3

    hidden_dim = 128
    num_layers = 2
    dropout = 0.2
    nhead = 4

    SPLIT_MODE = 'random'


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
    print(f"   [Plot] Saved: {path_no_ext}.png")


# ================= 绘图函数集 (修改版) =================
def plot_training_curve(history, out_dir, setting, model_name):
    epochs = range(1, len(history['loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))

    c1 = '#E24A33'
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', color=c1, fontsize=12, fontweight='bold')
    ax1.plot(epochs, history['loss'], color=c1, lw=2)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = '#348ABD'
    ax2.set_ylabel('Val Accuracy', color=c2, fontsize=12, fontweight='bold')
    ax2.plot(epochs, history['val_acc'], color=c2, lw=2)
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.title(f"{setting} - {model_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, f"Training_{setting}_{model_name}"))
    plt.close()


# ================= ROC 绘图（114412 风格） =================
LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e", "#7f7f7f"]
GREY = "#444444"


def _roc_endpoints_clean(fpr, tpr):
    """
    114412 风格：
    - 端点补齐 (0,0) 与 (1,1)
    - 压缩 FPR==1 的重复点，避免右侧“假竖线”
    """
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

    # compress x==1 duplicates
    mask_one = (fpr == 1.0)
    if np.any(mask_one):
        t1 = float(np.max(tpr[mask_one]))
        keep = ~mask_one
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]

    return fpr, tpr


def _style_axes_114412(ax):
    # tick 朝外，上/右无 tick；不显示刻度数字（114412）
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def plot_single_roc(y_true, y_prob, setting_name, auc_score, out_dir):
    """
    画单张 ROC 图（114412 风格，Stacking binary）。
    注意：auc_score 参数保留但不依赖，实际 AUC 以清洗后的曲线重新计算为准。
    """
    fig, ax = plt.subplots(figsize=(6.8, 6.1))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    auc_val = auc(fpr, tpr)

    # 稳定颜色映射
    if setting_name == "Day2_Only":
        color = LINE_COLORS[0]
    elif setting_name == "All_Days":
        color = LINE_COLORS[1]
    else:
        color = LINE_COLORS[2]

    ax.plot(fpr, tpr, lw=2.8, color=color, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], lw=1.5, color=GREY, ls="--", alpha=0.55)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"Stacking ROC — {setting_name}", fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    _style_axes_114412(ax)
    ax.grid(False)
    plt.tight_layout()

    save_plot(fig, os.path.join(out_dir, f"ROC_{setting_name}"))
    plt.close()


def plot_day2only_vs_all_roc(results_buffer, out_dir):
    """
    【新增】把 Day2_Only 与 All_Days 的 ROC 放在一张图（114412 风格）
    输出：ROC_Day2Only_vs_AllDays.png/pdf
    """
    if ("Day2_Only" not in results_buffer) or ("All_Days" not in results_buffer):
        return

    a = results_buffer["Day2_Only"]
    b = results_buffer["All_Days"]

    fpr1, tpr1, _ = roc_curve(a["y_true"], a["y_prob"])
    fpr1, tpr1 = _roc_endpoints_clean(fpr1, tpr1)
    auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(b["y_true"], b["y_prob"])
    fpr2, tpr2 = _roc_endpoints_clean(fpr2, tpr2)
    auc2 = auc(fpr2, tpr2)

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    ax.plot([0, 1], [0, 1], lw=1.5, color=GREY, ls="--", alpha=0.55)

    ax.plot(fpr1, tpr1, lw=2.8, color=LINE_COLORS[0], label=f"Day2_Only (AUC={auc1:.3f})")
    ax.plot(fpr2, tpr2, lw=2.8, color=LINE_COLORS[1], label=f"All_Days (AUC={auc2:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Stacking ROC — Day2_Only vs All_Days", fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    _style_axes_114412(ax)
    ax.grid(False)
    plt.tight_layout()

    save_plot(fig, os.path.join(out_dir, "ROC_Day2Only_vs_AllDays"))
    plt.close()
# ============================================================


def plot_performance_trend(results_buffer, out_dir):
    settings_order = ['Day2_Only', 'Day2_Day4', 'All_Days']
    x_labels = ['Day 2', 'Day 4', 'Day 6']

    accs, losses = [], []
    for s in settings_order:
        if s in results_buffer:
            accs.append(results_buffer[s]['acc'])
            losses.append(results_buffer[s]['loss'])

    fig, ax1 = plt.subplots(figsize=(8, 6))

    c1 = '#E24A33'
    ax1.set_xlabel('Observation End Point', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', color=c1, fontsize=14, fontweight='bold')
    ax1.plot(x_labels, accs, color=c1, marker='o', markersize=12, lw=3)
    ax1.fill_between(x_labels, np.array(accs) - 0.005, np.array(accs) + 0.005, color=c1, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = '#348ABD'
    ax2.set_ylabel('Log Loss', color=c2, fontsize=14, fontweight='bold')
    ax2.plot(x_labels, losses, color=c2, marker='^', markersize=12, lw=3)
    ax2.fill_between(x_labels, np.array(losses) - 0.01, np.array(losses) + 0.01, color=c2, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.title("Performance Trend", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, "Performance_Trend"))
    plt.close()


def plot_clonal_correlation(y_true, y_prob, clone_ids, labels, title, out_path):
    df = pd.DataFrame({"clone_id": clone_ids, "true": y_true, "prob": y_prob})
    stats = df.groupby("clone_id").agg(
        obs=("true", "mean"), pred=("prob", "mean"), count=("true", "count")
    ).reset_index()

    if len(stats) > 1:
        corr, _ = pearsonr(stats["obs"], stats["pred"])
    else:
        corr = 0

    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=stats, x="obs", y="pred", size="count", sizes=(60, 500),
        hue="obs", palette="coolwarm", alpha=0.7, edgecolor="k", legend=False
    )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    plt.title(f"{title}\nPearson r = {corr:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Observed Fraction ({labels[1]})", fontsize=12, fontweight='bold')
    plt.ylabel(f"Predicted Probability ({labels[1]})", fontsize=12, fontweight='bold')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    save_plot(plt, out_path)  # 使用 plt.savefig 保存当前图
    plt.close()


# ================= 数据与模型 =================
def load_data(cfg):
    print(f"Loading H5: {cfg.time_series_h5}")
    with h5py.File(cfg.time_series_h5, "r") as f:
        X_all = np.array(f["X"], dtype=np.float32)

    df = pd.read_csv(cfg.index_csv)
    col = "label_str" if "label_str" in df.columns else "y_str"
    labels = df[col].values

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

    print(f"Total Samples: {len(X_bin)}")
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
            for t in self.mask:
                x[t] = 0.0  # zero-mask
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


def train_base_model(model, tr_l, va_l, device, cfg, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_acc = 0.0
    best_state = None
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
            loss_sum += loss.item() * x.size(0)
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
            best_acc = val_acc
            best_state = model.state_dict()
            pat = 0
        else:
            pat += 1
            if pat >= cfg.patience:
                break

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
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model_dir)

    X, y, clones = load_data(cfg)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    settings = {
        "Day2_Only": [1, 2],
        "Day2_Day4": [2],
        "All_Days": None
    }

    results_buffer = {}

    print("\n" + "=" * 50)
    print("   PHASE 1: Training Models (Multi-Seed)")
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

        val_probs = {}
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
            'y_true': y[te_idx],
            'y_prob': p_stack,
            'acc': acc,
            'loss': loss,
            'clones': clones[te_idx]
        }

    print("\n" + "=" * 50)
    print("   PHASE 2: Generating Final Plots & CSV")
    print("=" * 50)

    # 1. 独立的 ROC（114412 风格）
    for setting, res in results_buffer.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc_score = auc(fpr, tpr)
        plot_single_roc(res['y_true'], res['y_prob'], setting, auc_score, cfg.out_dir)

    # 1.5 【新增】Day2_Only vs All_Days 同图 ROC（114412 风格）
    plot_day2only_vs_all_roc(results_buffer, cfg.out_dir)

    # 2. 趋势图
    plot_performance_trend(results_buffer, cfg.out_dir)

    # 3. 克隆相关性图
    for setting, res in results_buffer.items():
        seed_used = cfg.SEEDS[setting]
        plot_clonal_correlation(
            res['y_true'], res['y_prob'], res['clones'],
            [cfg.negative_label, cfg.positive_label],
            title=f"{setting} Clonal Fate (Seed {seed_used})",
            out_path=os.path.join(cfg.out_dir, f"Summary_{setting}_Clonal_Corr")
        )

    # 4. 【新增】保存 CSV 总结
    summary_data = []
    for setting, res in results_buffer.items():
        summary_data.append({
            "Setting": setting,
            "Seed": cfg.SEEDS[setting],
            "Stacking_Accuracy": res['acc'],
            "Stacking_LogLoss": res['loss']
        })
    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(cfg.out_dir, "ensemble_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"   [CSV] Saved summary to: {csv_path}")

    print(f"\n[Done] All plots saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()
