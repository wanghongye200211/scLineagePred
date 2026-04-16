# -*- coding: utf-8 -*-
"""
[Part 1] Random-split regression (Deep_Lineage style) with ensemble (RNN / BiLSTM / Transformer),
MPS acceleration, and a SIGN-FIX patch for rare "anti-correlated but high R^2" genes.

What you asked for
- Random shuffle sequences and split train/val/test (no clone split)
- Base models: RNN / BiLSTM / Trans
- Stacking ensemble that should beat single models
- Detect and PATCH sign-flip genes on VAL (clone-mean correlation r<0):
    If stacked r is negative but some base model has positive r, we fallback that gene to the best positive base model.
- Save final test outputs for plotting.

Outputs per task:
- ckpt/*.pt
- norm_mu.npy / norm_sd.npy
- stacking_W.npy / stacking_b.npy   (after sign-fix)
- signfix_report.json
- test_outputs.npz  (pred_log, true_log, clone_id, label, gene_names)
"""

import os
import random
import json
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import anndata as ad
import scipy.sparse as sp

from tqdm import tqdm


# ===================== Config =====================
class Config:
    # ===== paths =====
    ae_result_dir = "/Users/wanghongye/python/scLineagetracer/autoencoder/results/GSE140802"   # genes.txt (HVG list)
    time_series_h5 = "/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_DeepLineage_sequences.h5"
    index_csv = "/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_DeepLineage_index.csv"
    adata_h5ad = "/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_final_integrated.h5ad"
    out_dir = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE140802"

    # Expression source in LOG space:
    #   "X" | "raw" | "layer:<name>"
    adata_expr_source = "X"

    # obs column names
    adata_label_col = "state_info"     # target cell type column in adata.obs

    # labels to keep (filter by TARGET timepoint label)
    keep_types = ("Neutrophil", "Monocyte")

    # tasks: (name, in_pos, tgt_pos) within a 3-timepoint sequence
    tasks = [
        ("Reg_D4_from_D2_D6", [0, 2], 1),
        ("Reg_D6_from_D2_D4", [0, 1], 2),
    ]

    # ===== random split over sequences =====
    seed = 42
    split_train = 0.80
    split_val = 0.10
    split_test = 0.10

    # ===== training =====
    device = "auto"      # "auto" / "mps" / "cuda" / "cpu"
    batch_size = 256
    lr = 1e-3
    epochs = 80
    patience = 12
    hidden = 512
    dropout = 0.2

    # stacking ridge strength
    stack_alpha = 5.0

    # ===== sign-fix patch =====
    sign_fix_enable = True
    # if stacked clone-mean r < this threshold, and some base has r > min_pos_r, fallback
    sign_fix_r_threshold = -0.01
    sign_fix_min_pos_r = 0.05
    # report up to N genes
    sign_fix_report_topn = 25
    # watch genes
    watch_genes = ("Ddc", "Pdzd4")


# ===================== utilities =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            print("[Device] Using CUDA")
            return "cuda"
        if torch.backends.mps.is_available():
            print("[Device] Using MPS")
            return "mps"
        print("[Device] Using CPU")
        return "cpu"
    return device


def safe_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def load_h5_sequences(path: str):
    with h5py.File(path, "r") as f:
        X = f["X"][:] if "X" in f else f["data"][:]
        indices = f["indices"][:] if "indices" in f else None
    return np.asarray(X, dtype=np.float32), indices


def read_gene_list(ae_result_dir: str):
    gene_path = os.path.join(ae_result_dir, "genes.txt")
    if not os.path.exists(gene_path):
        raise FileNotFoundError(f"Missing genes.txt: {gene_path}")
    with open(gene_path, "r", encoding="utf-8") as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    return np.array(genes, dtype=object)


def get_expr_matrix(adata, source: str):
    if source == "X":
        X = adata.X
    elif source == "raw":
        if adata.raw is None:
            raise ValueError("adata.raw is None but adata_expr_source='raw'")
        X = adata.raw.X
    elif source.startswith("layer:"):
        key = source.split("layer:", 1)[1]
        if key not in adata.layers:
            raise KeyError(f"adata.layers['{key}'] not found")
        X = adata.layers[key]
    else:
        raise ValueError(f"Unknown adata_expr_source: {source}")
    return safe_dense(X).astype(np.float32)


def compute_clone_means(expr, clone_ids):
    clone_ids = np.asarray(clone_ids)
    uniq, inv = np.unique(clone_ids, return_inverse=True)
    C, G = len(uniq), expr.shape[1]
    sums = np.zeros((C, G), dtype=np.float64)
    np.add.at(sums, inv, expr.astype(np.float64))
    cnt = np.bincount(inv).astype(np.float64)
    return (sums / np.maximum(cnt, 1.0)[:, None]).astype(np.float32), uniq


def corr_cols(A, B, eps=1e-12):
    """Vectorized Pearson r per column (genes). A,B: [C,G]."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    Am = A - A.mean(axis=0, keepdims=True)
    Bm = B - B.mean(axis=0, keepdims=True)
    cov = (Am * Bm).sum(axis=0)
    denom = np.sqrt((Am * Am).sum(axis=0) * (Bm * Bm).sum(axis=0)) + eps
    r = cov / denom
    r[~np.isfinite(r)] = 0.0
    return r.astype(np.float32)


# ===================== model =====================
class DirectPredictor(nn.Module):
    """Sequence encoder (RNN/BiLSTM/Transformer) + MLP head -> gene expression (log space)."""
    def __init__(self, kind: str, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.kind = kind

        if kind == "RNN":
            self.net = nn.RNN(in_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
            enc_out = hidden
        elif kind == "BiLSTM":
            self.net = nn.LSTM(in_dim, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
            enc_out = hidden * 2
        elif kind == "Trans":
            self.proj = nn.Linear(in_dim, 128)
            enc = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512,
                                             dropout=dropout, activation="gelu", batch_first=True)
            self.net = nn.TransformerEncoder(enc, num_layers=2)
            self.pos = nn.Parameter(torch.randn(1, 8, 128) * 0.02)
            enc_out = 128
        else:
            raise ValueError(f"Unknown kind: {kind}")

        self.head = nn.Sequential(
            nn.Linear(enc_out, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        if self.kind == "Trans":
            h = self.proj(x) + self.pos[:, :x.size(1), :]
            feat = self.net(h).mean(1)
        else:
            out, _ = self.net(x)
            feat = out[:, -1]
        return self.head(feat)


# ===================== dataset =====================
class RegDataset(Dataset):
    def __init__(self, X_in: np.ndarray, tgt_cell_idx: np.ndarray, labels_tgt: np.ndarray, clone_ids: np.ndarray, X_expr: np.ndarray):
        self.X = torch.from_numpy(X_in.astype(np.float32))
        self.tgt = tgt_cell_idx.astype(np.int64)
        self.lbl = labels_tgt.astype(object)
        self.clone = clone_ids.astype(object)
        self.X_expr = X_expr

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        y = torch.from_numpy(self.X_expr[self.tgt[i]].astype(np.float32))
        return self.X[i], y, self.clone[i], self.lbl[i], self.tgt[i]


def build_loaders(X_in, tgt_cell_idx, labels_tgt, clone_ids, X_expr, idx_tr, idx_va, idx_te, batch_size):
    mu = X_in[idx_tr].mean(axis=(0, 1))
    sd = X_in[idx_tr].std(axis=(0, 1)) + 1e-6

    def _dl(idxs, shuffle):
        Xn = (X_in[idxs] - mu) / sd
        ds = RegDataset(Xn, tgt_cell_idx[idxs], labels_tgt[idxs], clone_ids[idxs], X_expr)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    return _dl(idx_tr, True), _dl(idx_va, False), _dl(idx_te, False), mu.astype(np.float32), sd.astype(np.float32)


# ===================== train / predict =====================
def train_one(model, tr_loader, va_loader, device, cfg, save_path):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_losses = []
        for x, y, *_ in tr_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y, *_ in va_loader:
                pred = model(x.to(device))
                va_losses.append(loss_fn(pred, y.to(device)).item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        print(f"  [ep {ep:03d}] tr={tr_loss:.5f} va={va_loss:.5f}")

        if va_loss < best - 1e-6:
            best = va_loss
            bad = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  [EarlyStop] patience reached at ep={ep}, best_va={best:.5f}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    metas = {"clone": [], "label": [], "tgt": []}
    for x, _, c, l, tgt in loader:
        out = model(x.to(device)).detach().cpu().numpy()
        preds.append(out)
        metas["clone"].append(np.array(c, dtype=object))
        metas["label"].append(np.array(l, dtype=object))
        metas["tgt"].append(np.array(tgt, dtype=np.int64))
    return np.concatenate(preds, axis=0), {k: np.concatenate(v, axis=0) for k, v in metas.items()}


def fit_stacking(preds_val_list, y_val, alpha):
    """Per-gene ridge stacking."""
    M = len(preds_val_list)
    N, G = y_val.shape
    ones = np.ones((N, 1), dtype=np.float32)

    W = np.zeros((M, G), dtype=np.float32)
    b = np.zeros((G,), dtype=np.float32)

    for g in tqdm(range(G), desc="Stacking (per gene)"):
        Xg = np.stack([p[:, g] for p in preds_val_list], axis=1).astype(np.float32)  # [N,M]
        Xb = np.concatenate([Xg, ones], axis=1)  # [N,M+1]
        A = Xb.T @ Xb + np.eye(M + 1, dtype=np.float32) * alpha
        A[-1, -1] = 0.0
        sol = np.linalg.solve(A, Xb.T @ y_val[:, g].astype(np.float32))
        W[:, g] = sol[:M]
        b[g] = sol[M]
    return W, b


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    genes = read_gene_list(cfg.ae_result_dir)
    gene_to_idx = {str(g): i for i, g in enumerate(genes)}

    # load adata and align
    print(f"[Data] Loading adata: {cfg.adata_h5ad}")
    adata = ad.read_h5ad(cfg.adata_h5ad)
    adata = adata[:, genes].copy()
    X_expr = get_expr_matrix(adata, cfg.adata_expr_source)
    x_min, x_max = float(np.min(X_expr)), float(np.max(X_expr))
    print(f"[Expr] source={cfg.adata_expr_source} min={x_min:.4f} max={x_max:.4f}")

    # sequences + indices
    print(f"[Data] Loading sequences: {cfg.time_series_h5}")
    X_seq, indices = load_h5_sequences(cfg.time_series_h5)
    if indices is None:
        raise ValueError("H5 missing 'indices'. Please regenerate sequences with indices saved.")
    df_idx = pd.read_csv(cfg.index_csv)
    if len(df_idx) != X_seq.shape[0]:
        raise ValueError(f"index_csv rows ({len(df_idx)}) != sequences ({X_seq.shape[0]}). Use matching files.")
    clone_ids_seq = df_idx["clone_id"].astype(str).values

    print(f"[Seq] X_seq={X_seq.shape} indices={indices.shape}")

    for task_name, in_pos, tgt_pos in cfg.tasks:
        print(f"\n=== Task: {task_name} | in_pos={in_pos} -> tgt_pos={tgt_pos} ===")
        tdir = os.path.join(cfg.out_dir, task_name)
        ensure_dir(tdir)
        ensure_dir(os.path.join(tdir, "ckpt"))

        tgt_cell_idx = indices[:, tgt_pos].astype(np.int64)
        labels_tgt = adata.obs[cfg.adata_label_col].iloc[tgt_cell_idx].astype(str).values

        keep_mask = np.isin(labels_tgt, cfg.keep_types)
        valid = np.where(keep_mask)[0]
        print(f"[Filter] valid_sequences={len(valid)}/{len(labels_tgt)} keep={cfg.keep_types}")

        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(valid)
        n = len(valid)
        n_tr = int(n * cfg.split_train)
        n_va = int(n * cfg.split_val)
        idx_tr = valid[:n_tr]
        idx_va = valid[n_tr:n_tr + n_va]
        idx_te = valid[n_tr + n_va:]
        print(f"[Split-Random] train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} (n={n})")

        X_in = X_seq[:, in_pos, :]
        tr_loader, va_loader, te_loader, mu, sd = build_loaders(
            X_in, tgt_cell_idx, labels_tgt, clone_ids_seq, X_expr, idx_tr, idx_va, idx_te, cfg.batch_size
        )
        np.save(os.path.join(tdir, "norm_mu.npy"), mu)
        np.save(os.path.join(tdir, "norm_sd.npy"), sd)

        base_names = ["RNN", "BiLSTM", "Trans"]
        preds_val_list = []
        models = {}

        for name in base_names:
            print(f"[Train] {name}")
            model = DirectPredictor(name, in_dim=X_in.shape[-1], out_dim=X_expr.shape[1], hidden=cfg.hidden, dropout=cfg.dropout)
            ckpt_path = os.path.join(tdir, "ckpt", f"{name}.pt")
            model = train_one(model, tr_loader, va_loader, device, cfg, ckpt_path)
            models[name] = model
            p_val, _ = predict(model, va_loader, device)
            preds_val_list.append(p_val)

        y_val = X_expr[tgt_cell_idx[idx_va]]
        print("[Stacking] fitting per-gene ridge stacking...")
        W, b = fit_stacking(preds_val_list, y_val, alpha=cfg.stack_alpha)

        # -------- sign-fix patch on VAL (clone-mean correlation) --------
        report = {"task": task_name, "sign_fix_enable": cfg.sign_fix_enable}
        if cfg.sign_fix_enable:
            # stacked val predictions
            p_stack_val = np.zeros_like(preds_val_list[0], dtype=np.float32)
            for i, p in enumerate(preds_val_list):
                p_stack_val += p * W[i][None, :]
            p_stack_val += b[None, :]

            clone_val = clone_ids_seq[idx_va]
            Pc_stack, _ = compute_clone_means(p_stack_val, clone_val)
            Tc_val, _ = compute_clone_means(y_val, clone_val)
            r_stack = corr_cols(Pc_stack, Tc_val)
            r_base = []
            for p in preds_val_list:
                Pc_i, _ = compute_clone_means(p, clone_val)
                r_base.append(corr_cols(Pc_i, Tc_val))
            r_base = np.stack(r_base, axis=0)  # [M,G]

            # genes to patch
            bad = np.where(r_stack < cfg.sign_fix_r_threshold)[0]
            fixed = []
            for g in bad:
                rb = r_base[:, g]
                # best positive base
                pos = np.where(rb > cfg.sign_fix_min_pos_r)[0]
                if pos.size == 0:
                    continue
                k = pos[np.argmax(rb[pos])]
                # fallback: use base k only
                W[:, g] = 0.0
                W[k, g] = 1.0
                b[g] = 0.0
                fixed.append((int(g), int(k), float(r_stack[g]), float(rb[k])))

            report["sign_fix_threshold"] = cfg.sign_fix_r_threshold
            report["sign_fix_min_pos_r"] = cfg.sign_fix_min_pos_r
            report["fixed_genes_n"] = len(fixed)

            # print top fixed by |r_stack| (most negative)
            if fixed:
                fixed_sorted = sorted(fixed, key=lambda x: x[2])  # most negative first
                print(f"[SignFix] fixed {len(fixed_sorted)} genes on VAL (fallback to best positive base). Top cases:")
                for g, k, rs, rk in fixed_sorted[:cfg.sign_fix_report_topn]:
                    print(f"  {genes[g]}  r_stack={rs:.3f} -> base[{base_names[k]}] r={rk:.3f}")

            # watch genes (val)
            for wg in cfg.watch_genes:
                if wg in gene_to_idx:
                    gi = gene_to_idx[wg]
                    print(f"[Watch][VAL] {wg}: r_stack(before-fix may differ)={r_stack[gi]:.3f}, "
                          f"r_base={[(base_names[i], float(r_base[i,gi])) for i in range(len(base_names))]} "
                          f"W={W[:,gi].round(3).tolist()} b={float(b[gi]):.3f}")

        # save stacking params (after sign-fix)
        np.save(os.path.join(tdir, "stacking_W.npy"), W)
        np.save(os.path.join(tdir, "stacking_b.npy"), b)
        with open(os.path.join(tdir, "signfix_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # -------- TEST predict with final stacking --------
        print("[Test] predicting stacked output...")
        p_list = []
        meta = None
        for name in base_names:
            p_te, m = predict(models[name], te_loader, device)
            p_list.append(p_te)
            meta = m if meta is None else meta

        p_final = np.zeros_like(p_list[0], dtype=np.float32)
        for i, p in enumerate(p_list):
            p_final += p * W[i][None, :]
        p_final += b[None, :]

        y_true = X_expr[meta["tgt"]]

        # watch genes on TEST clone-mean
        if cfg.sign_fix_enable:
            clone_te = meta["clone"]
            Pc_te, _ = compute_clone_means(p_final, clone_te)
            Tc_te, _ = compute_clone_means(y_true, clone_te)
            r_te = corr_cols(Pc_te, Tc_te)
            for wg in cfg.watch_genes:
                if wg in gene_to_idx:
                    gi = gene_to_idx[wg]
                    print(f"[Watch][TEST] {wg}: r={float(r_te[gi]):.3f}")

        out_npz = os.path.join(tdir, "test_outputs.npz")
        np.savez_compressed(
            out_npz,
            pred_log=p_final.astype(np.float32),
            true_log=y_true.astype(np.float32),
            tgt_cell_idx=meta["tgt"].astype(np.int64),
            clone_id=meta["clone"].astype(object),
            label=meta["label"].astype(object),
            gene_names=genes.astype(object),
            task=np.array([task_name], dtype=object),
        )
        print(f"[Saved] {out_npz}")

    print("\nDone.")


if __name__ == "__main__":
    main()
