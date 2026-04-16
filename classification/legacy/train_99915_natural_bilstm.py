#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GSE99915 (natural sampling) with BiLSTM on cell-level random splits.
Outputs per-setting metrics with both default threshold (0.5) and
validation-tuned threshold (for balanced accuracy).
"""

import os
import random
import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    f1_score,
    recall_score,
    precision_score,
)


H5_PATH = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_CellLevel_Terminal_Natural_sequences.h5"
CSV_PATH = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_CellLevel_Terminal_Natural_index.csv"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915_natural_bilstm"

SETTINGS = [
    ("Obs_Day21", 2024, [5]),
    ("Obs_Day15", 42, [4, 5]),
    ("Obs_Day12", 123, [3, 4, 5]),
    ("Obs_Day9", 999, [2, 3, 4, 5]),
]

BATCH_SIZE = 1024
EPOCHS = 40
PATIENCE = 8
LR = 1e-3
HIDDEN = 256
LAYERS = 2
DROPOUT = 0.2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, X, y, idx):
        self.X = X
        self.y = y
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(self.X[j]), torch.tensor(int(self.y[j]), dtype=torch.long)


class BiLSTM(nn.Module):
    def __init__(self, d_in, h=256, l=2, dr=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            d_in,
            h,
            l,
            batch_first=True,
            bidirectional=True,
            dropout=(dr if l > 1 else 0.0),
        )
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(0.5 * (h[-2] + h[-1]))


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs, y_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        p = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        y_true.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(y_true)


def compute_metrics(y_true, prob, thr):
    y_hat = (prob >= thr).astype(np.int64)
    return {
        "acc": float(accuracy_score(y_true, y_hat)),
        "bacc": float(balanced_accuracy_score(y_true, y_hat)),
        "f1_failed": float(f1_score(y_true, y_hat, pos_label=0)),
        "recall_failed": float(recall_score(y_true, y_hat, pos_label=0)),
        "precision_failed": float(precision_score(y_true, y_hat, pos_label=0, zero_division=0)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(H5_PATH, "r") as f:
        X0 = np.array(f["X"], dtype=np.float32)
        M0 = np.array(f["mask"], dtype=np.float32)
    y = np.where(pd.read_csv(CSV_PATH)["label_str"].values == "Reprogrammed", 1, 0).astype(np.int64)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] samples={len(y)}, pos_rate={y.mean():.4f}")

    rows = []

    for setting, seed, mask_idx in SETTINGS:
        set_seed(seed)
        print(f"\n[RUN] {setting} seed={seed} mask_idx={mask_idx}")

        X = X0.copy()
        M = M0.copy()
        for i in mask_idx:
            X[:, i, :] = 0.0
            M[:, i] = 0.0
        X = np.concatenate([X, M[..., None]], axis=2)

        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
        va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])

        tr_loader = DataLoader(SeqDataset(X, y, tr), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(SeqDataset(X, y, va), batch_size=BATCH_SIZE, shuffle=False)
        te_loader = DataLoader(SeqDataset(X, y, te), batch_size=BATCH_SIZE, shuffle=False)

        n0 = int((y[tr] == 0).sum())
        n1 = int((y[tr] == 1).sum())
        w0 = (n0 + n1) / (2.0 * max(n0, 1))
        w1 = (n0 + n1) / (2.0 * max(n1, 1))
        cls_w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"[INFO] train class count failed={n0}, reprog={n1}, class_weight=({w0:.3f},{w1:.3f})")

        model = BiLSTM(X.shape[2], h=HIDDEN, l=LAYERS, dr=DROPOUT).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        best_state = None
        best_bacc = -1.0
        best_val_prob = None
        best_val_y = None
        bad = 0

        for ep in range(1, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            n_seen = 0
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb, weight=cls_w)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            tr_loss = loss_sum / max(n_seen, 1)

            val_prob, val_y = collect_probs(model, va_loader, device)
            val_metrics = compute_metrics(val_y, val_prob, thr=0.5)
            val_bacc = val_metrics["bacc"]
            print(f"ep={ep:03d} loss={tr_loss:.4f} va_acc={val_metrics['acc']:.4f} va_bacc={val_bacc:.4f}", flush=True)

            if val_bacc > best_bacc:
                best_bacc = val_bacc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_prob = val_prob.copy()
                best_val_y = val_y.copy()
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        test_prob, test_y = collect_probs(model, te_loader, device)

        m05 = compute_metrics(test_y, test_prob, thr=0.5)
        auc = float(roc_auc_score(test_y, test_prob))
        ll = float(log_loss(test_y, test_prob))

        thrs = np.linspace(0.05, 0.95, 181)
        bvals = [balanced_accuracy_score(best_val_y, (best_val_prob >= t).astype(np.int64)) for t in thrs]
        best_thr = float(thrs[int(np.argmax(bvals))])
        mt = compute_metrics(test_y, test_prob, thr=best_thr)

        row = {
            "setting": setting,
            "seed": seed,
            "n_samples": int(len(y)),
            "pos_rate_all": float(y.mean()),
            "pos_rate_test": float(test_y.mean()),
            "auc": auc,
            "logloss": ll,
            "acc_05": m05["acc"],
            "bacc_05": m05["bacc"],
            "f1_failed_05": m05["f1_failed"],
            "recall_failed_05": m05["recall_failed"],
            "precision_failed_05": m05["precision_failed"],
            "best_val_bacc": float(best_bacc),
            "best_thr_bacc": best_thr,
            "acc_thr": mt["acc"],
            "bacc_thr": mt["bacc"],
            "f1_failed_thr": mt["f1_failed"],
            "recall_failed_thr": mt["recall_failed"],
            "precision_failed_thr": mt["precision_failed"],
        }
        rows.append(row)

        print(
            "[TEST] "
            f"{setting} auc={auc:.4f} acc05={m05['acc']:.4f} bacc05={m05['bacc']:.4f} "
            f"thr={best_thr:.3f} bacc_thr={mt['bacc']:.4f}"
        )

    out_csv = os.path.join(OUT_DIR, "natural_bilstm_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[DONE] saved summary: {out_csv}")


if __name__ == "__main__":
    main()
