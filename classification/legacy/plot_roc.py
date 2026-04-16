# -*- coding: utf-8 -*-
"""
Plot ROC curves from *trained* ensemble models (no training).
- Load X from H5, labels from index CSV
- Rebuild BiLSTM / RNN / Transformer, load .pth
- Load stacking LR from .pkl
- Predict on test split (same split logic as training script) and plot ROC
"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


# ------------------ Style ------------------
def set_sci_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 300


# A common “SCI / Nature-like” 4-color palette
DEFAULT_SCI4 = ["#3C5488", "#E64B35", "#00A087", "#F39B7F"]  # blue, red, green, orange


# ------------------ Data ------------------
def load_data(time_series_h5, index_csv, positive_label, negative_label):
    print(f"[Load] H5: {time_series_h5}")
    with h5py.File(time_series_h5, "r") as f:
        X_all = np.array(f["X"], dtype=np.float32)

    df = pd.read_csv(index_csv)
    if "label_str" in df.columns:
        labels = df["label_str"].values
    elif "label" in df.columns:
        labels = df["label"].values
    else:
        raise ValueError("index_csv must contain 'label_str' or 'label' column.")

    mask = (labels == positive_label) | (labels == negative_label)
    keep = np.where(mask)[0]
    X = X_all[keep]
    y = np.where(labels[keep] == positive_label, 1, 0).astype(np.int64)

    print(f"[Load] Kept {len(X)} samples ({positive_label}=1 vs {negative_label}=0)")
    return X, y


def get_split(n_samples, seed):
    all_idx = np.arange(n_samples)
    tr, tmp = train_test_split(all_idx, test_size=0.2, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    return tr, va, te


class SeqDataset(Dataset):
    def __init__(self, X, y, idx, mask_idx=None):
        self.X = X
        self.y = y
        self.idx = idx
        self.mask_idx = mask_idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j].copy()
        if self.mask_idx is not None:
            for t in self.mask_idx:
                x[t] = 0.0
        return torch.from_numpy(x), torch.tensor(self.y[j], dtype=torch.long)


# ------------------ Models (must match training) ------------------
class LSTMModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.lstm = nn.LSTM(
            d, h, l,
            batch_first=True,
            bidirectional=True,
            dropout=(dr if l > 1 else 0.0)
        )
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)  # h: [2*l, B, H]
        feat = 0.5 * (h[-2] + h[-1])  # [B, H]
        return self.head(feat)


class RNNModel(nn.Module):
    def __init__(self, d, h, l, dr):
        super().__init__()
        self.rnn = nn.RNN(d, h, l, batch_first=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        h = self.rnn(x)[1][-1]  # [B, H]
        return self.head(h)


class TransformerModel(nn.Module):
    def __init__(self, d, h, l, dr, nhead):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=nhead,
                dim_feedforward=h * 2,
                dropout=dr,
                batch_first=True
            ),
            num_layers=l
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x):
        z = self.enc(x).mean(dim=1)  # [B, d]
        return self.head(z)


@torch.no_grad()
def predict_prob(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        out = model(x.to(device))
        p = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def build_mask_indices(timepoints, obs_day):
    """Mask all time steps whose timepoint > obs_day."""
    if obs_day >= max(timepoints):
        return None
    return [i for i, tp in enumerate(timepoints) if tp > obs_day]


def setting_name_from_obs(timepoints, obs_day):
    if obs_day >= max(timepoints):
        return "All_Days"
    return f"Obs_Day{obs_day}"


def resolve_seeds(obs_days, seeds, seed_default_map):
    if seeds is not None:
        if len(seeds) != len(obs_days):
            raise ValueError("--seeds length must match --obs_days length.")
        return {d: s for d, s in zip(obs_days, seeds)}

    resolved = {}
    for d in obs_days:
        if d in seed_default_map:
            resolved[d] = seed_default_map[d]
        else:
            # fallback
            resolved[d] = list(seed_default_map.values())[0]
            print(f"[WARN] No default seed for obs_day={d}, fallback seed={resolved[d]}")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    # Paths (defaults follow your current GSE99915 script)
    parser.add_argument("--time_series_h5", type=str,
                        default="/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_sequences.h5")
    parser.add_argument("--index_csv", type=str,
                        default="/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_index.csv")
    parser.add_argument("--model_dir", type=str,
                        default="/Users/wanghongye/python/scLineagetracer/classification/GSE99915/saved_models")
    parser.add_argument("--out_dir", type=str,
                        default="/Users/wanghongye/python/scLineagetracer/classification/GSE99915")

    # Labels
    parser.add_argument("--positive_label", type=str, default="Reprogrammed")
    parser.add_argument("--negative_label", type=str, default="Failed")

    # Time controls
    parser.add_argument("--timepoints", type=str, default="6,9,12,15,21,28",
                        help="Comma-separated real timepoints; index order must match sequence axis.")
    parser.add_argument("--obs_days", nargs="+", type=int, default=[28, 21, 15, 12],
                        help="Which observation days to plot (e.g. 28 21 15 12).")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Optional seeds list aligned to obs_days (for loading filenames + split).")

    # Model hyperparams (MUST match training)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--nhead", type=int, default=4)

    # Plot
    parser.add_argument("--colors", type=str, default=",".join(DEFAULT_SCI4),
                        help="Comma-separated hex colors (4 recommended).")
    parser.add_argument("--figsize", type=float, nargs=2, default=[6, 6])
    parser.add_argument("--title", type=str, default="ROC Curves (Selected Observation Times)")
    parser.add_argument("--out_name", type=str, default="ROC_Selected_FromTrained")

    # Inference
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    set_sci_style()
    os.makedirs(args.out_dir, exist_ok=True)

    timepoints = [int(x) for x in args.timepoints.split(",") if x.strip() != ""]
    colors = [c.strip() for c in args.colors.split(",") if c.strip() != ""]
    if len(colors) < 1:
        raise ValueError("--colors parsed empty.")
    # cycle colors if needed
    def color_at(i): return colors[i % len(colors)]

    # Default seed map (same as your training script naming habit)
    seed_default_map = {
        28: 2026,  # All_Days
        21: 2024,
        15: 42,
        12: 123,
        9: 999
    }
    seed_by_day = resolve_seeds(args.obs_days, args.seeds, seed_default_map)

    # Load data once
    X, y = load_data(args.time_series_h5, args.index_csv, args.positive_label, args.negative_label)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # Plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for i, obs_day in enumerate(args.obs_days):
        setting = setting_name_from_obs(timepoints, obs_day)
        seed = seed_by_day[obs_day]
        mask_idx = build_mask_indices(timepoints, obs_day)

        # split
        _, _, te_idx = get_split(len(X), seed)
        y_true = y[te_idx]
        if len(np.unique(y_true)) < 2:
            print(f"[WARN] {setting}: test split has only one class, skip ROC.")
            continue

        te_loader = DataLoader(
            SeqDataset(X, y, te_idx, mask_idx=mask_idx),
            batch_size=args.batch_size,
            shuffle=False
        )

        d = X.shape[2]
        bilstm = LSTMModel(d, args.hidden_dim, args.num_layers, args.dropout).to(device)
        rnn = RNNModel(d, args.hidden_dim, args.num_layers, args.dropout).to(device)
        trans = TransformerModel(d, args.hidden_dim, args.num_layers, args.dropout, args.nhead).to(device)

        # load weights
        def must_exist(p):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")
            return p

        p_bilstm = must_exist(os.path.join(args.model_dir, f"{setting}_BiLSTM_s{seed}.pth"))
        p_rnn = must_exist(os.path.join(args.model_dir, f"{setting}_RNN_s{seed}.pth"))
        p_trans = must_exist(os.path.join(args.model_dir, f"{setting}_Trans_s{seed}.pth"))
        p_stack = must_exist(os.path.join(args.model_dir, f"{setting}_Stacking_s{seed}.pkl"))

        bilstm.load_state_dict(torch.load(p_bilstm, map_location="cpu"))
        rnn.load_state_dict(torch.load(p_rnn, map_location="cpu"))
        trans.load_state_dict(torch.load(p_trans, map_location="cpu"))

        with open(p_stack, "rb") as f:
            lr = pickle.load(f)

        # predict
        prob_b = predict_prob(bilstm, te_loader, device)
        prob_r = predict_prob(rnn, te_loader, device)
        prob_t = predict_prob(trans, te_loader, device)

        X_stack = np.stack([prob_b, prob_r, prob_t], axis=1)
        prob_ens = lr.predict_proba(X_stack)[:, 1]

        fpr, tpr, _ = roc_curve(y_true, prob_ens)
        auc_score = auc(fpr, tpr)

        label = f"{setting} (AUC={auc_score:.4f})"
        ax.plot(fpr, tpr, lw=3, color=color_at(i), label=label)

        print(f"[ROC] {setting} | seed={seed} | AUC={auc_score:.4f} | mask_idx={mask_idx}")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title(args.title, fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    plt.tight_layout()
    out_no_ext = os.path.join(args.out_dir, args.out_name)
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Done] Saved:\n  {out_no_ext}.png\n  {out_no_ext}.pdf")


if __name__ == "__main__":
    main()
