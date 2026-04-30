# -*- coding: utf-8 -*-
"""
Unified regression training in gene-expression space for scLineagePred.

Outputs per task:
- ckpt/*.pt
- norm_mu.npy / norm_sd.npy
- stacking_W.npy / stacking_b.npy
- signfix_report.json
- test_outputs.npz
"""

import argparse
import os
import json
import random
import re
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import anndata as ad
import scipy.sparse as sp

from tqdm import tqdm


# ===================== Config =====================
class Config:
    # ===== paths =====
    ae_result_dir = ""   # must contain genes.txt
    time_series_h5 = ""
    index_csv = ""
    index_clone_col = "clone_id"
    csv_label_col = "label_str"
    adata_h5ad = ""
    out_dir = "./outputs/regression"

    # Expression source in LOG space:
    #   "X" | "raw" | "layer:<name>"
    adata_expr_source = "X"

    # keep only these endpoint cell types (细胞类型 cell types)
    keep_labels = tuple()

    # tasks mode:
    #   "all_prev_only"  -> predict LAST timepoint using ALL previous inputs (1 task, same as 99915 style)
    #   "each_prefix"    -> predict LAST timepoint using prefix inputs (T-1 tasks, heavier)
    tasks_mode = "all_prev_only"
    tasks = None  # leave None to auto-build from timepoints

    # Optional: if your sequences.h5 contains a mask dataset (N,T) of 0/1,
    # require target and ALL inputs present.
    # NOTE: for masked data we usually want False
    require_all_inputs_present = False

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
    sign_fix_enable = False
    sign_fix_r_threshold = -0.01
    sign_fix_min_pos_r = 0.05
    sign_fix_report_topn = 25

    # watch genes (optional)
    watch_genes = tuple()


# ===================== utilities =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            print("[Device] Using CUDA")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device] Using MPS")
            return "mps"
        print("[Device] Using CPU")
        return "cpu"
    return device


def safe_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def read_time_labels_from_h5(h5_path: str, T: int):
    """Support both v2-style 'timepoints' and v1-style 'time_labels'/'time_values'."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "time_labels" in f:
                tl = f["time_labels"][:]
                out = []
                for x in tl:
                    if isinstance(x, (bytes, np.bytes_)):
                        out.append(x.decode("utf-8"))
                    else:
                        out.append(str(x))
                if len(out) == T:
                    return out
            if "timepoints" in f:
                tp = np.asarray(f["timepoints"][:]).tolist()
                return [str(float(x)) for x in tp]
            if "time_values" in f:
                tv = np.asarray(f["time_values"][:]).tolist()
                return [str(int(x)) for x in tv]
    except Exception:
        pass
    return [f"t{i}" for i in range(T)]


def sanitize_name(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def infer_reverse_from_samples_order(df_idx: pd.DataFrame, T: int) -> bool:
    """
    Infer sequence direction from index_csv.samples_order.
    Example reverse order: "3,2,1,0" when T=4.
    """
    if "samples_order" not in df_idx.columns:
        return False

    ser = df_idx["samples_order"].dropna().astype(str)
    for raw in ser.head(200):
        toks = [t.strip() for t in raw.split(",") if t.strip() != ""]
        vals = []
        ok = True
        for t in toks:
            if t.lstrip("-").isdigit():
                vals.append(int(t))
            else:
                ok = False
                break
        if (not ok) or (len(vals) < 2):
            continue
        if T > 0 and len(vals) != T:
            # still usable for direction
            pass
        return vals[0] > vals[-1]
    return False


def maybe_flip_time_labels(timepoints):
    """
    Flip time labels only if they are strictly descending numerically.
    """
    try:
        ts = np.array([float(x) for x in timepoints], dtype=np.float64)
    except Exception:
        return timepoints, False
    if len(ts) >= 2 and np.all(np.diff(ts) < 0):
        return list(timepoints[::-1]), True
    return timepoints, False


def load_h5_sequences(path: str):
    """
    Expected datasets:
      - X (or data): float32 [N,T,D]
      - indices: int [N,T]  (REQUIRED for 140802-style training)
      - mask: optional int8 [N,T]
      - label_str: optional bytes/str [N]
    """
    with h5py.File(path, "r") as f:
        if "X" in f:
            X = f["X"][:]
        elif "data" in f:
            X = f["data"][:]
        else:
            raise KeyError("H5 missing dataset 'X' (or 'data').")

        indices = f["indices"][:] if "indices" in f else None
        mask = f["mask"][:] if "mask" in f else None
        label_str = f["label_str"][:] if "label_str" in f else None

    X = np.asarray(X, dtype=np.float32)
    if indices is not None:
        indices = np.asarray(indices, dtype=np.int64)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.int8)
    return X, indices, mask, label_str


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


def _masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor):
    """x: [B,T,D], key_padding_mask: [B,T] bool True=invalid."""
    if key_padding_mask is None:
        return x.mean(dim=1)
    valid = (~key_padding_mask).unsqueeze(-1).type_as(x)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


def _is_suffix_padding_mask(key_padding_mask: torch.Tensor) -> bool:
    """True if valid positions are prefix-contiguous (111000...)."""
    if key_padding_mask is None:
        return True
    valid = (~key_padding_mask).to(torch.int32)
    return bool(((valid[:, :-1] - valid[:, 1:]) >= 0).all().item())


# ===================== model =====================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pe[:, :t, :].to(x.dtype)
        return self.dropout(x)


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
            d_model = 128
            self.proj = nn.Linear(in_dim, d_model)
            self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=512)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=512,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.net = nn.TransformerEncoder(enc, num_layers=2, enable_nested_tensor=False)  # MPS: avoid nested tensor path
            enc_out = d_model
        else:
            raise ValueError(f"Unknown kind: {kind}")

        self.head = nn.Sequential(
            nn.Linear(enc_out, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        if self.kind == "Trans":
            h = self.proj(x)
            h = self.pos(h)
            h = self.net(h, src_key_padding_mask=key_padding_mask)
            feat = _masked_mean(h, key_padding_mask)
            return self.head(feat)

        # RNN / BiLSTM
        if key_padding_mask is not None and _is_suffix_padding_mask(key_padding_mask):
            lengths = (~key_padding_mask).sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            if self.kind == "RNN":
                _, h = self.net(packed)
                feat = h[-1]
            else:
                _, (h, _) = self.net(packed)
                feat = torch.cat([h[-2], h[-1]], dim=1)
        else:
            if self.kind == "RNN":
                out, _ = self.net(x)
            else:
                out, _ = self.net(x)
            feat = _masked_mean(out, key_padding_mask)

        return self.head(feat)


# ===================== dataset =====================
class RegDataset(Dataset):
    def __init__(
        self,
        X_in: np.ndarray,
        mask_in: np.ndarray,
        tgt_cell_idx: np.ndarray,
        labels_tgt: np.ndarray,
        clone_ids: np.ndarray,
        X_expr: np.ndarray,
    ):
        self.X = torch.from_numpy(X_in.astype(np.float32))
        if mask_in is None:
            self.kpm = None
        else:
            # key_padding_mask: True=missing
            self.kpm = torch.from_numpy((mask_in.astype(np.int8) == 0))
        self.tgt = tgt_cell_idx.astype(np.int64)
        self.lbl = labels_tgt.astype(object)
        self.clone = clone_ids.astype(object)
        self.X_expr = X_expr

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        y = torch.from_numpy(self.X_expr[self.tgt[i]].astype(np.float32))
        if self.kpm is None:
            return self.X[i], None, y, self.clone[i], self.lbl[i], self.tgt[i]
        return self.X[i], self.kpm[i].to(torch.bool), y, self.clone[i], self.lbl[i], self.tgt[i]


def build_loaders(X_in, mask_in, tgt_cell_idx, labels_tgt, clone_ids, X_expr, idx_tr, idx_va, idx_te, batch_size):
    """
    Mask-aware normalization (掩码感知归一化 mask-aware normalization):
    - mu/sd computed ONLY on observed positions (mask==1) from train
    - missing positions kept at 0 after normalization
    """
    eps = 1e-6

    if mask_in is None:
        mu = X_in[idx_tr].mean(axis=(0, 1))
        sd = X_in[idx_tr].std(axis=(0, 1)) + eps

        def _norm(X):
            return (X - mu) / sd

        def _dl(idxs, shuffle):
            Xn = _norm(X_in[idxs])
            ds = RegDataset(Xn, None, tgt_cell_idx[idxs], labels_tgt[idxs], clone_ids[idxs], X_expr)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    else:
        mtr = mask_in[idx_tr].astype(np.float32)  # [n_tr, T_in]
        den = float(mtr.sum())
        if den <= 0:
            raise ValueError("No observed inputs in training set after mask filter.")

        mu = (X_in[idx_tr] * mtr[:, :, None]).sum(axis=(0, 1)) / (den + eps)
        var = ((X_in[idx_tr] - mu) * mtr[:, :, None]) ** 2
        var = var.sum(axis=(0, 1)) / (den + eps)
        sd = np.sqrt(var) + eps

        def _norm(X, m):
            Xn = (X - mu) / sd
            return Xn * m[:, :, None]

        def _dl(idxs, shuffle):
            m = mask_in[idxs].astype(np.float32)
            Xn = _norm(X_in[idxs], m)
            ds = RegDataset(Xn, mask_in[idxs], tgt_cell_idx[idxs], labels_tgt[idxs], clone_ids[idxs], X_expr)
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
        for x, kpm, y, *_ in tr_loader:
            x = x.to(device)
            y = y.to(device)
            kpm_t = None if kpm is None else kpm.to(device)
            pred = model(x, kpm_t)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, kpm, y, *_ in va_loader:
                x = x.to(device)
                y = y.to(device)
                kpm_t = None if kpm is None else kpm.to(device)
                pred = model(x, kpm_t)
                va_losses.append(loss_fn(pred, y).item())

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
    for x, kpm, _, c, l, tgt in loader:
        x = x.to(device)
        kpm_t = None if kpm is None else kpm.to(device)
        out = model(x, kpm_t).detach().cpu().numpy()
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
        Xg = np.stack([p[:, g] for p in preds_val_list], axis=1).astype(np.float32)
        Xb = np.concatenate([Xg, ones], axis=1)
        A = Xb.T @ Xb + np.eye(M + 1, dtype=np.float32) * alpha
        A[-1, -1] = 0.0
        sol = np.linalg.solve(A, Xb.T @ y_val[:, g].astype(np.float32))
        W[:, g] = sol[:M]
        b[g] = sol[M]
    return W, b


def build_tasks_from_timepoints(timepoints, mode: str):
    T = len(timepoints)
    if T < 2:
        raise ValueError(f"Need at least 2 timepoints, got T={T}")
    tgt_pos = T - 1

    def _mk_name(in_pos):
        tgt = sanitize_name(timepoints[tgt_pos])
        ins = [sanitize_name(timepoints[i]) for i in in_pos]
        # keep folder name reasonable
        if len(ins) > 4:
            name = f"Reg_{tgt}_from_{ins[0]}_to_{ins[-1]}"
        else:
            name = f"Reg_{tgt}_from_" + "_".join(ins)
        return name

    if mode == "all_prev_only":
        in_pos = list(range(T - 1))
        return [(_mk_name(in_pos), in_pos, tgt_pos)]
    if mode == "each_prefix":
        tasks = []
        for k in range(T - 1):
            in_pos = list(range(k + 1))
            tasks.append((_mk_name(in_pos), in_pos, tgt_pos))
        return tasks
    raise ValueError(f"Unknown tasks_mode: {mode}")


def run(cfg: Config):
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    genes = read_gene_list(cfg.ae_result_dir)
    gene_to_idx = {str(g): i for i, g in enumerate(genes)}

    # load adata and align to genes
    print(f"[Data] Loading adata: {cfg.adata_h5ad}")
    adata = ad.read_h5ad(cfg.adata_h5ad)
    adata = adata[:, genes].copy()
    X_expr = get_expr_matrix(adata, cfg.adata_expr_source)
    print(f"[Expr] source={cfg.adata_expr_source} shape={X_expr.shape} min={float(X_expr.min()):.4f} max={float(X_expr.max()):.4f}")

    # sequences + indices (+ optional mask + optional label_str)
    print(f"[Data] Loading sequences: {cfg.time_series_h5}")
    X_seq, indices, mask, label_str_h5 = load_h5_sequences(cfg.time_series_h5)
    if indices is None:
        raise ValueError(
            "Your sequences.h5 does NOT contain `indices`, so we cannot fetch real gene expression from adata by index.\n"
            "This unified regression pipeline requires indices for expression alignment.\n"
            "Please regenerate sequences with indices saved."
        )
    N, T, D = X_seq.shape
    print(f"[Seq] X_seq={X_seq.shape} indices={indices.shape} mask={'None' if mask is None else mask.shape}")

    timepoints = read_time_labels_from_h5(cfg.time_series_h5, T)
    print(f"[Time] T={T} timepoints={timepoints}")

    # load index CSV (labels + clone ids)
    df_idx = pd.read_csv(cfg.index_csv)
    if len(df_idx) != N:
        raise ValueError(f"index_csv rows ({len(df_idx)}) != sequences ({N}). Use matching files.")

    reverse_build = infer_reverse_from_samples_order(df_idx, T)
    if reverse_build:
        X_seq = X_seq[:, ::-1, :].copy()
        indices = indices[:, ::-1].copy()
        if mask is not None:
            mask = mask[:, ::-1].copy()
        tp_new, flipped_tp = maybe_flip_time_labels(timepoints)
        if flipped_tp:
            timepoints = tp_new
            print("[Time] Detected descending time labels; flipped to ascending after sequence flip.")
        print("[Seq] Detected reverse samples_order; flipped X_seq/indices/mask to chronological order.")

    if cfg.index_clone_col not in df_idx.columns:
        raise KeyError(f"Missing clone id column in index_csv: {cfg.index_clone_col}")
    clone_ids_seq = df_idx[cfg.index_clone_col].astype(str).values

    # prefer CSV labels; fallback to H5 label_str
    if cfg.csv_label_col in df_idx.columns:
        label_str_seq = df_idx[cfg.csv_label_col].astype(str).values
    else:
        if label_str_h5 is None:
            raise KeyError(f"Missing {cfg.csv_label_col} in CSV and label_str in H5.")
        label_str_seq = np.array([
            (x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)).strip()
            for x in label_str_h5
        ], dtype=object)

    # build tasks
    if cfg.tasks is None:
        cfg.tasks = build_tasks_from_timepoints(timepoints, cfg.tasks_mode)
    print(f"[Tasks] {len(cfg.tasks)} task(s). mode={cfg.tasks_mode}")

    # run tasks
    for task_name, in_pos, tgt_pos in cfg.tasks:
        if tgt_pos >= T:
            raise ValueError(f"tgt_pos={tgt_pos} out of range for T={T}")
        if any(p >= T for p in in_pos):
            raise ValueError(f"in_pos has out-of-range index for T={T}: {in_pos}")

        print(f"\n=== Task: {task_name} | in_pos={in_pos} -> tgt_pos={tgt_pos} ({timepoints[tgt_pos]}) ===")
        tdir = os.path.join(cfg.out_dir, task_name)
        ensure_dir(tdir)
        ensure_dir(os.path.join(tdir, "ckpt"))

        # ===== validity =====
        m = (indices[:, tgt_pos] >= 0)

        if mask is not None:
            m &= (mask[:, tgt_pos] == 1)
            if cfg.require_all_inputs_present:
                m &= (mask[:, in_pos].sum(axis=1) == len(in_pos))
            else:
                m &= (mask[:, in_pos].sum(axis=1) > 0)
        else:
            if cfg.require_all_inputs_present:
                m &= (indices[:, in_pos] >= 0).all(axis=1)
            else:
                m &= (indices[:, in_pos] >= 0).any(axis=1)

        valid0 = np.where(m)[0]
        if valid0.size == 0:
            print("[Skip] no valid sequences for this task (check indices/mask).")
            continue

        # ===== keep only requested endpoint labels =====
        keep_labels = np.array(cfg.keep_labels, dtype=object)
        keep = np.isin(label_str_seq[valid0], keep_labels) if keep_labels.size > 0 else np.ones(valid0.shape[0], dtype=bool)
        valid = valid0[keep]

        if valid.size == 0:
            vc = pd.Series(label_str_seq[valid0]).value_counts().to_dict()
            print(f"[Skip] 0 sequences after label filter. keep={cfg.keep_labels} | label counts={vc}")
            continue

        # subset arrays to valid
        X_in = X_seq[valid][:, in_pos, :]                       # [n, T_in, D]
        tgt_cell_idx = indices[valid, tgt_pos].astype(np.int64) # [n]
        labels_tgt = label_str_seq[valid]
        clone_ids = clone_ids_seq[valid]

        # input mask for normalization + key_padding_mask
        if mask is not None:
            mask_in = mask[valid][:, in_pos].astype(np.int8)  # [n, T_in]
        else:
            mask_in = (indices[valid][:, in_pos] >= 0).astype(np.int8)

        # random split on valid subset
        rng = np.random.default_rng(cfg.seed)
        idx_all = np.arange(valid.size)
        rng.shuffle(idx_all)
        n = idx_all.size
        n_tr = int(n * cfg.split_train)
        n_va = int(n * cfg.split_val)
        idx_tr = idx_all[:n_tr]
        idx_va = idx_all[n_tr:n_tr + n_va]
        idx_te = idx_all[n_tr + n_va:]
        print(f"[Split-Random] train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} (n={n})")

        tr_loader, va_loader, te_loader, mu, sd = build_loaders(
            X_in, mask_in, tgt_cell_idx, labels_tgt, clone_ids, X_expr, idx_tr, idx_va, idx_te, cfg.batch_size
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
            preds_val_list.append(p_val.astype(np.float32))

        # fit stacking on VAL in gene space (log)
        y_val = X_expr[tgt_cell_idx[idx_va]]
        print("[Stacking] fitting per-gene ridge stacking...")
        W, b = fit_stacking(preds_val_list, y_val, alpha=cfg.stack_alpha)

        # -------- sign-fix patch on VAL (clone-mean correlation) --------
        report = {"task": task_name, "sign_fix_enable": cfg.sign_fix_enable}
        if cfg.sign_fix_enable:
            p_stack_val = np.zeros_like(preds_val_list[0], dtype=np.float32)
            for i, p in enumerate(preds_val_list):
                p_stack_val += p * W[i][None, :]
            p_stack_val += b[None, :]

            clone_val = clone_ids[idx_va]
            Pc_stack, _ = compute_clone_means(p_stack_val, clone_val)
            Tc_val, _ = compute_clone_means(y_val, clone_val)
            r_stack = corr_cols(Pc_stack, Tc_val)

            r_base = []
            for p in preds_val_list:
                Pc_i, _ = compute_clone_means(p, clone_val)
                r_base.append(corr_cols(Pc_i, Tc_val))
            r_base = np.stack(r_base, axis=0)  # [M,G]

            bad = np.where(r_stack < cfg.sign_fix_r_threshold)[0]
            fixed = []
            for g in bad:
                rb = r_base[:, g]
                pos = np.where(rb > cfg.sign_fix_min_pos_r)[0]
                if pos.size == 0:
                    continue
                k = pos[np.argmax(rb[pos])]
                W[:, g] = 0.0
                W[k, g] = 1.0
                b[g] = 0.0
                fixed.append((int(g), int(k), float(r_stack[g]), float(rb[k])))

            report.update({
                "sign_fix_threshold": cfg.sign_fix_r_threshold,
                "sign_fix_min_pos_r": cfg.sign_fix_min_pos_r,
                "fixed_genes_n": len(fixed),
            })

            if fixed:
                fixed_sorted = sorted(fixed, key=lambda x: x[2])
                print(f"[SignFix] fixed {len(fixed_sorted)} genes on VAL. Top cases:")
                for g, k, rs, rk in fixed_sorted[:cfg.sign_fix_report_topn]:
                    print(f"  {genes[g]}  r_stack={rs:.3f} -> base[{base_names[k]}] r={rk:.3f}")

            for wg in cfg.watch_genes:
                if wg in gene_to_idx:
                    gi = gene_to_idx[wg]
                    print(
                        f"[Watch][VAL] {wg}: r_stack={float(r_stack[gi]):.3f}, "
                        f"r_base={[(base_names[i], float(r_base[i, gi])) for i in range(len(base_names))]} "
                        f"W={W[:, gi].round(3).tolist()} b={float(b[gi]):.3f}"
                    )

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
            p_te, m_meta = predict(models[name], te_loader, device)
            p_list.append(p_te.astype(np.float32))
            meta = m_meta if meta is None else meta

        p_final = np.zeros_like(p_list[0], dtype=np.float32)
        for i, p in enumerate(p_list):
            p_final += p * W[i][None, :]
        p_final += b[None, :]

        y_true = X_expr[meta["tgt"]]

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
            timepoints=np.array(timepoints, dtype=object),
            in_pos=np.array(in_pos, dtype=np.int64),
            tgt_pos=np.array([tgt_pos], dtype=np.int64),
        )
        print(f"[Saved] {out_npz}")

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified gene-space regression training")
    parser.add_argument("--ae-result-dir", required=True, help="Directory containing genes.txt from embedding training")
    parser.add_argument("--time-series-h5", required=True, help="Sequence H5 file")
    parser.add_argument("--index-csv", required=True, help="Sequence metadata CSV")
    parser.add_argument("--adata-h5ad", required=True, help="Integrated AnnData with matched cells")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--index-clone-col", default="clone_id")
    parser.add_argument("--csv-label-col", default="label_str")
    parser.add_argument("--adata-expr-source", default="X")
    parser.add_argument(
        "--keep-label",
        dest="keep_labels",
        action="append",
        default=[],
        help="Endpoint label to keep. Repeat the flag to provide multiple labels.",
    )
    parser.add_argument("--tasks-mode", default="all_prev_only", choices=["all_prev_only", "each_prefix"])
    parser.add_argument("--require-all-inputs-present", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-train", type=float, default=0.80)
    parser.add_argument("--split-val", type=float, default=0.10)
    parser.add_argument("--split-test", type=float, default=0.10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stack-alpha", type=float, default=5.0)
    parser.add_argument("--sign-fix-enable", action="store_true")
    parser.add_argument("--sign-fix-r-threshold", type=float, default=-0.01)
    parser.add_argument("--sign-fix-min-pos-r", type=float, default=0.05)
    parser.add_argument("--sign-fix-report-topn", type=int, default=25)
    parser.add_argument(
        "--watch-gene",
        dest="watch_genes",
        action="append",
        default=[],
        help="Gene to print in sign-fix diagnostics. Repeat the flag to provide multiple genes.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.ae_result_dir = args.ae_result_dir
    cfg.time_series_h5 = args.time_series_h5
    cfg.index_csv = args.index_csv
    cfg.index_clone_col = args.index_clone_col
    cfg.csv_label_col = args.csv_label_col
    cfg.adata_h5ad = args.adata_h5ad
    cfg.out_dir = args.out_dir
    cfg.adata_expr_source = args.adata_expr_source
    cfg.keep_labels = tuple(args.keep_labels)
    cfg.tasks_mode = args.tasks_mode
    cfg.require_all_inputs_present = args.require_all_inputs_present
    cfg.seed = args.seed
    cfg.split_train = args.split_train
    cfg.split_val = args.split_val
    cfg.split_test = args.split_test
    cfg.device = args.device
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.hidden = args.hidden
    cfg.dropout = args.dropout
    cfg.stack_alpha = args.stack_alpha
    cfg.sign_fix_enable = args.sign_fix_enable
    cfg.sign_fix_r_threshold = args.sign_fix_r_threshold
    cfg.sign_fix_min_pos_r = args.sign_fix_min_pos_r
    cfg.sign_fix_report_topn = args.sign_fix_report_topn
    cfg.watch_genes = tuple(args.watch_genes)
    return cfg


def main():
    args = parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
