
# -*- coding: utf-8 -*-
"""
GSE114412 - Perturbation Scan (3-class) + Decoder Mapping + (Optional) CellChat Export
===================================================================================

This script is the GSE114412 counterpart of your GSE140802 perturbation pipeline:

[A] Perturbation scan (scan ALL latent dims)
    - On sequence inputs (N, T, D) produced by step5_build_sequence.py
    - Uses your already-trained ensemble (BiLSTM / RNN / Transformer) + stacking LR
      produced by classification/class_114412.py
    - Adaptive observation window: perturb at time t -> observe up to time t (keep_len=t+1)

[B] Decoder mapping to genes
    - Uses autoencoder decoder weights: genes.txt + Z_genes.npy
    - Produces class-specific marker gene rankings per target timepoint
    - Builds class-specific driver gene master tables via RRF (Reciprocal Rank Fusion)

[C] (Optional) CellChat inputs export (from RAW full-gene h5ad)
    - expr.mtx.gz + genes.tsv + cells.tsv + meta.csv
    - optionally split by timepoint

Notes
-----
- This script only READS your assets (processed sequences, saved models, decoder files, raw h5ad).
  It does NOT modify your training assets.
- Default paths follow your project layout. If your obs column names differ in RAW h5ad,
  just edit Config.time_key / Config.state_key / Config.clone_key.

Key terms (中英对照)
------------------
- 扰动 perturbation
- 潜在空间 latent space
- 观测窗口 observation window
- 集成 stacking ensemble
- 倒数排名融合 Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

import os
import re
import json
import gzip
import shutil
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

import anndata as ad
import scipy.sparse as sp
from scipy.io import mmwrite


# ===================== Config =====================
@dataclass
class Config:
    # ---------- dataset ----------
    OUT_PREFIX: str = "GSE114412_all_generated"
    processed_dir: str = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed"

    # sequences built by Step5
    time_series_h5: str = ""   # auto from processed_dir + OUT_PREFIX
    index_csv: str = ""        # auto from processed_dir + OUT_PREFIX

    # endpoint classes (MUST match class_114412.py)
    TARGET_ENDPOINT_CLUSTERS: Tuple[str, ...] = ("sc_beta", "sc_ec", "sc_alpha")
    csv_label_col: str = "label_str"

    # ---------- classifier ensemble ----------
    model_dir: str = "/Users/wanghongye/python/scLineagetracer/classification/GSE114412/saved_models"
    base_seed: int = 2026

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    nhead: int = 4

    # ---------- perturbation ----------
    # scale factors applied to one latent dimension at a chosen timepoint
    folds: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

    max_ood_rate_for_ranking: float = 0.10
    top_k_dims: int = 30
    top_k_genes_per_dim: int = 50

    # To keep runtime reasonable on laptop:
    # - If None: use ALL sequences in the H5
    # - If int: stratified downsample to at most this many sequences
    max_sequences_scan: Optional[int] = 200000


    # Choose which time indices to perturb:
    # - None: perturb ALL timepoints
    # - e.g. (0, 2, 3): only perturb these indices
    perturb_target_indices: Optional[Tuple[int, ...]] = None

    # ---------- decoder (STRICT: GSE114412 autoencoder outputs) ----------
    decoder_dir: str = "/Users/wanghongye/python/scLineagetracer/autoencoder/results/GSE114412"
    genes_txt: str = field(init=False)
    z_genes_npy: str = field(init=False)

    # HVG h5ad to sanity-check decoder gene overlap
    hvg_h5ad: str = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed/GSE114412_with_latent.h5ad"
    latent_key: str = "X_latent"

    # ---------- outputs ----------
    out_dir: str = "/Users/wanghongye/python/scLineagetracer/Downstream_114412/Perturbation"

    # ---------- driver master (RRF) ----------
    rrf_k: int = 50
    top_union_marker: int = 800  # per target, per class
    save_driver_master: bool = True

    # ---------- transition genes ----------
    top_union_transition: int = 300  # per transition file (used to build union)

    # ---------- CellChat export (optional) ----------
    raw_full_h5ad: str = "/Users/wanghongye/python/scLineagetracer/GSE114412/GSE114412_stage6.h5ad"
    raw_counts_layer: str = "counts"   # preferred counts layer; fallback to X if missing
    export_cellchat_inputs: bool = False
    split_cellchat_by_time: bool = True
    cellchat_genes_mode: str = "full"  # "full" or "driver"
    export_expr_summary: bool = True

    # meta columns in raw h5ad (edit if needed)
    time_key: str = "CellWeek"
    state_key: str = "Assigned_cluster"
    clone_key: str = "clone_id"  # optional, if present

    # ---------- device ----------
    device: str = "auto"
    batch_size: int = 2048

    def __post_init__(self):
        self.time_series_h5 = os.path.join(self.processed_dir, f"{self.OUT_PREFIX}_sequences.h5")
        self.index_csv = os.path.join(self.processed_dir, f"{self.OUT_PREFIX}_index.csv")
        self.genes_txt = os.path.join(self.decoder_dir, "genes.txt")
        self.z_genes_npy = os.path.join(self.decoder_dir, "Z_genes.npy")


# ===================== utils =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pick_device(cfg: Config) -> str:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return cfg.device


def sanitize_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = s.strip("_")
    return s if s else "NA"


def stratified_subsample_indices(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    idx_all = np.arange(len(y), dtype=np.int64)
    if len(idx_all) <= max_n:
        return idx_all

    keep = []
    C = int(y.max()) + 1
    for k in range(C):
        ik = idx_all[y == k]
        if len(ik) == 0:
            continue
        nk = int(round(max_n * (len(ik) / float(len(idx_all)))))
        nk = max(1, min(nk, len(ik)))
        keep.append(rng.choice(ik, size=nk, replace=False))
    keep = np.unique(np.concatenate(keep)) if keep else rng.choice(idx_all, size=max_n, replace=False)
    if len(keep) > max_n:
        keep = rng.choice(keep, size=max_n, replace=False)
    return np.sort(keep.astype(np.int64))


def read_time_labels_from_h5(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        if "time_labels" in f:
            tl = f["time_labels"][:]
            out = []
            for x in tl:
                if isinstance(x, (bytes, np.bytes_)):
                    out.append(x.decode("utf-8"))
                else:
                    out.append(str(x))
            return out
        if "time_values" in f:
            tv = f["time_values"][:]
            return [str(int(x)) for x in np.asarray(tv).tolist()]
    # fallback: infer from X
    with h5py.File(h5_path, "r") as f:
        T = int(f["X"].shape[1])
    return [f"t{i}" for i in range(T)]


def setting_name_for_time(time_labels: List[str], t_idx: int) -> str:
    """Match class_114412.py build_time_settings()."""
    if t_idx < len(time_labels) - 1:
        return f"UpTo_{time_labels[t_idx]}"
    return f"All_{time_labels[-1]}"


# ===================== Models (match class_114412.py) =====================
class BiLSTMModel(nn.Module):
    def __init__(self, d: int, h: int, n_layers: int, dropout: float, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            d, h, n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if n_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(feat)


class RNNModel(nn.Module):
    def __init__(self, d: int, h: int, n_layers: int, dropout: float, n_classes: int):
        super().__init__()
        self.rnn = nn.RNN(
            d, h, n_layers,
            batch_first=True,
            dropout=(dropout if n_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(h[-1])


class TransformerBlock(nn.Module):
    def __init__(self, d: int, nhead: int, ff_mult: int = 2, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, d: int, ff: int, n_layers: int, dropout: float, nhead: int, n_classes: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d, nhead, ff_mult=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, n_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        ar = torch.arange(T, device=x.device)[None, :].expand(B, T)
        key_padding_mask = ar >= lengths[:, None]  # True=PAD

        h = x
        for blk in self.blocks:
            h = blk(h, key_padding_mask)

        mask = (~key_padding_mask).float().unsqueeze(-1)
        h_sum = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        feat = h_sum / denom
        return self.head(feat)


def load_ensemble(cfg: Config, input_dim: int, n_classes: int, device: str, setting: str):
    seed = int(cfg.base_seed)

    models = {
        "BiLSTM": BiLSTMModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, n_classes).to(device),
        "RNN": RNNModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, n_classes).to(device),
        "Trans": TransformerModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead, n_classes).to(device),
    }

    for name, m in models.items():
        pth = os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth")
        if not os.path.exists(pth):
            raise FileNotFoundError(f"[ERROR] missing model: {pth}")
        state = torch.load(pth, map_location=device)
        m.load_state_dict(state, strict=True)
        m.eval()

    pkl = os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"[ERROR] missing stacking LR: {pkl}")
    with open(pkl, "rb") as f:
        lr = pickle.load(f)

    return models, lr


@torch.no_grad()
def predict_proba_stack(models: Dict[str, nn.Module], lr, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """
    Multi-class stacking:
      - base model -> softmax probs (N,C)
      - concat features [BiLSTM_probs, RNN_probs, Trans_probs] -> (N, 3*C)
      - lr.predict_proba -> (N,C)

    IMPORTANT: feature order MUST match class_114412.py training:
      BiLSTM then RNN then Trans (each contributes C probabilities).
    """
    X = np.asarray(X, dtype=np.float32)
    N, T, _ = X.shape

    probs_parts = {"BiLSTM": [], "RNN": [], "Trans": []}

    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        xb = torch.from_numpy(X[st:ed]).to(device)
        lengths = torch.full((ed - st,), T, dtype=torch.long, device=device)

        for name in ["BiLSTM", "RNN", "Trans"]:
            logits = models[name](xb, lengths)
            p = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            probs_parts[name].append(p)

    feats = np.concatenate(
        [np.concatenate(probs_parts["BiLSTM"], axis=0),
         np.concatenate(probs_parts["RNN"], axis=0),
         np.concatenate(probs_parts["Trans"], axis=0)],
        axis=1
    )  # (N, 3*C)

    return lr.predict_proba(feats).astype(np.float32)


# ===================== Data loading =====================
def load_sequences(cfg: Config):
    if not os.path.isfile(cfg.index_csv):
        raise FileNotFoundError(f"[ERROR] missing index csv: {cfg.index_csv}")
    if not os.path.isfile(cfg.time_series_h5):
        raise FileNotFoundError(f"[ERROR] missing sequences h5: {cfg.time_series_h5}")

    df = pd.read_csv(cfg.index_csv)
    if cfg.csv_label_col not in df.columns:
        raise KeyError(f"[ERROR] index csv missing label col: {cfg.csv_label_col}")

    labels = df[cfg.csv_label_col].astype(str).values
    keep = np.isin(labels, np.array(cfg.TARGET_ENDPOINT_CLUSTERS, dtype=object))

    if not keep.any():
        raise ValueError(f"[ERROR] No sequences match TARGET_ENDPOINT_CLUSTERS={cfg.TARGET_ENDPOINT_CLUSTERS}")

    # class mapping (stable order)
    class_names = list(cfg.TARGET_ENDPOINT_CLUSTERS)
    label_to_y = {c: i for i, c in enumerate(class_names)}
    y_all = np.array([label_to_y.get(s, -1) for s in labels], dtype=np.int64)
    idx_keep = np.where(keep & (y_all >= 0))[0].astype(np.int64)

    # optional stratified subsample
    y_keep = y_all[idx_keep]
    if cfg.max_sequences_scan is not None:
        idx_sub_local = stratified_subsample_indices(y_keep, int(cfg.max_sequences_scan), seed=cfg.base_seed)
        idx_keep = idx_keep[idx_sub_local]
        y_keep = y_all[idx_keep]

    # load X subset from H5 (sorted indices for efficient I/O)
    idx_keep = np.sort(idx_keep.astype(np.int64))
    y_keep = y_all[idx_keep]

    with h5py.File(cfg.time_series_h5, "r") as f:
        X = np.asarray(f["X"][idx_keep], dtype=np.float32)

    time_labels = read_time_labels_from_h5(cfg.time_series_h5)
    T = X.shape[1]
    if len(time_labels) != T:
        time_labels = time_labels[:T] if len(time_labels) > T else (time_labels + [f"t{i}" for i in range(len(time_labels), T)])

    # summary
    print(f"[INFO] Loaded sequences: N={len(X)} T={T} D={X.shape[2]}")
    counts = {c: int((y_keep == i).sum()) for i, c in enumerate(class_names)}
    print(f"[INFO] Label counts: {counts}")
    print(f"[INFO] time_labels={time_labels}")

    return X, y_keep.astype(np.int64), class_names, time_labels


# ===================== Perturbation scan =====================
def ood_rate_1d(ref: np.ndarray, new: np.ndarray) -> float:
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    mn, mx = float(r.min()), float(r.max())
    v = np.asarray(new, dtype=np.float32).reshape(-1)
    return float(((v < mn) | (v > mx)).mean())


def scan_target_timepoint(
    cfg: Config,
    target_t: int,
    target_label: str,
    X_all: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    time_labels: List[str],
    ens_cache: Dict[str, Tuple[Dict[str, nn.Module], object]],
    device: str,
):
    """
    For a fixed target time index t:
      - use keep_len=t+1 (observe up to t)
      - load the matching ensemble by setting name
      - scan dims and folds
    """
    N, T, D = X_all.shape
    keep_len = int(target_t + 1)
    if keep_len < 1 or keep_len > T:
        raise ValueError("keep_len out of range")

    setting = setting_name_for_time(time_labels, target_t)
    if setting not in ens_cache:
        ens_cache[setting] = load_ensemble(cfg, input_dim=D, n_classes=len(class_names), device=device, setting=setting)
    models, lr = ens_cache[setting]

    X_base = np.asarray(X_all[:, :keep_len, :], dtype=np.float32)
    # baseline prediction
    p0 = predict_proba_stack(models, lr, X_base, device=device, batch_size=cfg.batch_size)
    y0 = np.argmax(p0, axis=1).astype(np.int64)

    base_mean = p0.mean(axis=0)  # (C,)
    base_frac = np.array([(y0 == k).mean() for k in range(len(class_names))], dtype=np.float32)

    rows = []
    # mutable working copy for in-place dim scaling
    Xw = X_base.copy()

    for dim in range(D):
        ref = X_base[:, target_t, dim].copy()
        for fold in cfg.folds:
            # apply perturbation: scale ONE dimension at time t
            new_vals = ref * float(fold)
            Xw[:, target_t, dim] = new_vals

            p = predict_proba_stack(models, lr, Xw, device=device, batch_size=cfg.batch_size)
            yp = np.argmax(p, axis=1).astype(np.int64)

            mean_prob = p.mean(axis=0)
            delta = mean_prob - base_mean
            delta_l1 = float(np.abs(delta).sum())
            delta_maxabs = float(np.abs(delta).max())

            frac = np.array([(yp == k).mean() for k in range(len(class_names))], dtype=np.float32)
            delta_frac = frac - base_frac

            flip_rate = float((yp != y0).mean())
            ood = ood_rate_1d(ref, new_vals)

            push_k = int(np.argmax(delta))
            push_class = class_names[push_k]
            push_delta = float(delta[push_k])

            row = {
                "target_t": int(target_t),
                "target_label": str(target_label),
                "setting": str(setting),
                "keep_len": int(keep_len),
                "dim": int(dim),
                "fold": float(fold),
                "flip_rate": float(flip_rate),
                "ood_rate": float(ood),
                "delta_l1": float(delta_l1),
                "delta_maxabs": float(delta_maxabs),
                "push_class": str(push_class),
                "push_delta": float(push_delta),
            }
            for k, cname in enumerate(class_names):
                row[f"mean_prob_{cname}"] = float(mean_prob[k])
                row[f"delta_mean_prob_{cname}"] = float(delta[k])
                row[f"pred_frac_{cname}"] = float(frac[k])
                row[f"delta_pred_frac_{cname}"] = float(delta_frac[k])

            rows.append(row)

        # restore this dim column
        Xw[:, target_t, dim] = ref

    df = pd.DataFrame(rows)

    # summarize best fold per dim
    sums = []
    for dim, g in df.groupby("dim", sort=True):
        g_in = g[g["ood_rate"] <= cfg.max_ood_rate_for_ranking]
        if len(g_in) == 0:
            g_in = g
        best = g_in.sort_values(["flip_rate", "delta_l1", "delta_maxabs"], ascending=[False, False, False]).iloc[0]
        rec = {
            "target_t": int(target_t),
            "target_label": str(target_label),
            "setting": str(setting),
            "keep_len": int(keep_len),
            "dim": int(dim),
            "best_fold": float(best["fold"]),
            "best_flip_rate": float(best["flip_rate"]),
            "best_ood_rate": float(best["ood_rate"]),
            "best_delta_l1": float(best["delta_l1"]),
            "best_delta_maxabs": float(best["delta_maxabs"]),
            "best_push_class": str(best["push_class"]),
            "best_push_delta": float(best["push_delta"]),
        }
        for cname in class_names:
            rec[f"best_delta_mean_prob_{cname}"] = float(best[f"delta_mean_prob_{cname}"])
            rec[f"best_delta_pred_frac_{cname}"] = float(best[f"delta_pred_frac_{cname}"])
        sums.append(rec)

    df_sum = pd.DataFrame(sums).sort_values(
        ["best_flip_rate", "best_delta_l1", "best_delta_maxabs"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df, df_sum


# ===================== Decoder mapping =====================
def load_hvgs(cfg: Config) -> set:
    if not os.path.exists(cfg.hvg_h5ad):
        raise FileNotFoundError(f"[ERROR] hvg_h5ad not found: {cfg.hvg_h5ad}")
    a = ad.read_h5ad(cfg.hvg_h5ad, backed="r")
    hvgs = set(a.var_names.astype(str).tolist())
    try:
        a.file.close()
    except Exception:
        pass
    return hvgs


def load_decoder(cfg: Config, latent_dim: int, hvgs: set):
    if not (os.path.exists(cfg.genes_txt) and os.path.exists(cfg.z_genes_npy)):
        raise FileNotFoundError(f"[ERROR] missing decoder files in {cfg.decoder_dir}: genes.txt or Z_genes.npy")

    genes = [ln.strip() for ln in open(cfg.genes_txt, "r", encoding="utf-8") if ln.strip()]
    Zg = np.load(cfg.z_genes_npy).astype(np.float32)

    if Zg.ndim != 2:
        raise ValueError(f"[ERROR] Z_genes must be 2D, got {Zg.shape}")
    if Zg.shape[1] != latent_dim:
        raise ValueError(f"[ERROR] Z_genes dim mismatch: {Zg.shape[1]} vs latent_dim={latent_dim}")
    if len(genes) != Zg.shape[0]:
        m = min(len(genes), Zg.shape[0])
        genes = genes[:m]
        Zg = Zg[:m, :]

    overlap = len(set(genes) & hvgs) / max(len(set(genes)), 1)
    if overlap < 0.80:
        raise ValueError(
            f"[ERROR] Decoder genes overlap with HVG is too low ({overlap:.3f}). "
            f"Likely wrong decoder_dir. Expected: {cfg.decoder_dir}"
        )
    return np.array(genes, dtype=object), Zg


def top_genes_by_dim(genes: np.ndarray, Zg: np.ndarray, dim: int, k: int) -> pd.DataFrame:
    w = Zg[:, dim]
    idx = np.argsort(-np.abs(w))[:k]
    return pd.DataFrame({"gene": genes[idx], "loading": w[idx], "abs_loading": np.abs(w[idx]), "dim": int(dim)})


def integrate_marker_genes_for_class(
    genes: np.ndarray,
    Zg: np.ndarray,
    dim_sum: pd.DataFrame,
    top_dims: List[int],
    class_name: str,
):
    """
    Class-specific marker integration:
      weights per dim = best_delta_mean_prob_{class_name}  (signed)
      score_abs   = sum_d |Zg[g,d]| * |w_d|
      score_signed= sum_d  Zg[g,d]  *  w_d
    """
    if dim_sum.empty:
        return pd.DataFrame()

    col = f"best_delta_mean_prob_{class_name}"
    if col not in dim_sum.columns:
        raise KeyError(f"[ERROR] dim_summary missing column: {col}")

    s = dim_sum.set_index("dim")
    dims = [int(d) for d in top_dims if int(d) in s.index]
    if not dims:
        return pd.DataFrame()

    w_signed = np.array([float(s.loc[d][col]) for d in dims], dtype=np.float32)
    w_abs = np.abs(w_signed)

    M = Zg[:, dims]  # [G, K]
    score_abs = (np.abs(M) * w_abs[None, :]).sum(axis=1)
    score_signed = (M * w_signed[None, :]).sum(axis=1)

    df = pd.DataFrame({"gene": genes, "score_abs": score_abs, "score_signed": score_signed})
    return df.sort_values("score_abs", ascending=False).reset_index(drop=True)


def transition_genes(genes: np.ndarray, Zg: np.ndarray, deltaZ: np.ndarray, topk: int) -> pd.DataFrame:
    """
    Project deltaZ onto gene loadings in latent space.

    score = |cosine(Zg_i, deltaZ)|, proj = |Zg_i dot deltaZ_hat|
    """
    dz = np.asarray(deltaZ, dtype=np.float64).reshape(-1)
    dz_norm = float(np.linalg.norm(dz))
    if dz_norm < 1e-12:
        s = np.zeros((Zg.shape[0],), dtype=np.float64)
        proj = np.zeros_like(s)
        cos_signed = np.zeros_like(s)
        idx = np.argsort(-s)[:topk]
        return pd.DataFrame({"gene": genes[idx], "score": s[idx], "proj": proj[idx], "cos_signed": cos_signed[idx]})

    dz_hat = dz / dz_norm
    Z = np.asarray(Zg, dtype=np.float64)  # [G,D]
    dot = Z @ dz
    dot_hat = Z @ dz_hat

    zn = np.linalg.norm(Z, axis=1) + 1e-12
    cos_signed = dot / (zn * dz_norm)
    score = np.abs(cos_signed)
    proj = np.abs(dot_hat)

    order = np.lexsort((-proj, -score))
    idx = order[:topk]

    return pd.DataFrame({
        "gene": genes[idx],
        "score": score[idx],
        "proj": proj[idx],
        "cos_signed": cos_signed[idx],
    })


# ===================== Driver master (RRF) =====================
def build_driver_master_rrf(
    cfg: Config,
    downstream_dir: str,
    targets: List[str],
    class_name: str,
) -> pd.DataFrame:
    """
    Build driver genes master for ONE class via RRF over per-target marker lists:
      downstream/decoder_<target>/marker_genes_ranked_<class>.csv

    driver_score = sum_t 1/(k + rank_t)
    signed_score accumulates sign(score_signed) with same weight.

    direction:
      signed_score > 0  -> {class}_push
      signed_score < 0  -> {class}_suppress
      else              -> Neutral
    """
    k = int(cfg.rrf_k)
    score: Dict[str, float] = {}
    signed: Dict[str, float] = {}
    sources: Dict[str, set] = {}

    for tgt in targets:
        p = os.path.join(downstream_dir, f"decoder_{tgt}", f"marker_genes_ranked_{class_name}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df = df.head(int(cfg.top_union_marker)).copy()
        genes_list = df["gene"].astype(str).tolist()
        sign_list = df["score_signed"].to_numpy(dtype=np.float32) if "score_signed" in df.columns else None

        for r, g in enumerate(genes_list, start=1):
            w = 1.0 / float(k + r)
            score[g] = score.get(g, 0.0) + w
            if sign_list is not None:
                signed[g] = signed.get(g, 0.0) + w * float(np.sign(sign_list[r - 1]))
            sources.setdefault(g, set()).add(f"marker_{tgt}")

    if not score:
        return pd.DataFrame(columns=["gene", "driver_score", "signed_score", "direction", "sources", "rank"])

    rows = []
    for g, sc in score.items():
        ss = float(signed.get(g, 0.0))
        if ss > 0:
            direction = f"{class_name}_push"
        elif ss < 0:
            direction = f"{class_name}_suppress"
        else:
            direction = "Neutral"
        rows.append({
            "gene": g,
            "driver_score": float(sc),
            "signed_score": float(ss),
            "direction": direction,
            "sources": "|".join(sorted(list(sources.get(g, set())))) if sources.get(g, None) else "",
        })

    dfm = pd.DataFrame(rows).sort_values(["driver_score", "gene"], ascending=[False, True]).reset_index(drop=True)
    dfm["rank"] = np.arange(1, len(dfm) + 1)
    return dfm


# ===================== CellChat export (optional) =====================
def get_X(raw: ad.AnnData, layer: Optional[str] = None):
    if layer and (layer in getattr(raw, "layers", {})):
        X = raw.layers[layer]
    else:
        X = raw.X
    if sp.issparse(X):
        return X.tocsr()
    return np.asarray(X)


def export_cellchat_inputs(raw: ad.AnnData, cfg: Config, genes_use: List[str], out_dir: str):
    ensure_dir(out_dir)

    # filter timepoints if possible (keep only those appearing in time_labels)
    tvals = raw.obs[cfg.time_key].astype(str).to_numpy()
    # keep all by default (user can pre-filter raw)
    keep = np.ones_like(tvals, dtype=bool)

    raw2 = raw[keep].copy()

    if cfg.cellchat_genes_mode == "driver":
        genes = [g for g in genes_use if g in raw2.var_names]
    else:
        genes = raw2.var_names.astype(str).tolist()

    if len(genes) == 0:
        raise ValueError("[ERROR] No genes selected for CellChat export (empty).")

    gene_idx = np.array([raw2.var_names.get_loc(g) for g in genes], dtype=int)

    X = get_X(raw2, layer=(cfg.raw_counts_layer if cfg.raw_counts_layer else None))
    if sp.issparse(X):
        X_sel = X[:, gene_idx].T.tocoo()  # genes×cells
    else:
        X_sel = sp.coo_matrix(np.asarray(X)[:, gene_idx].T)

    cell_ids = raw2.obs_names.astype(str).to_numpy()
    meta = pd.DataFrame({
        "cell_id": cell_ids,
        cfg.time_key: raw2.obs[cfg.time_key].astype(str).values,
        cfg.state_key: raw2.obs[cfg.state_key].astype(str).values,
    })
    if cfg.clone_key in raw2.obs.columns:
        meta[cfg.clone_key] = raw2.obs[cfg.clone_key].astype(str).values

    tmp = os.path.join(out_dir, "expr.mtx")
    mmwrite(tmp, X_sel)
    with open(tmp, "rb") as f_in, gzip.open(os.path.join(out_dir, "expr.mtx.gz"), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(tmp)

    pd.DataFrame({"gene": genes}).to_csv(os.path.join(out_dir, "genes.tsv"), sep="\t", index=False, header=False)
    pd.DataFrame({"cell_id": cell_ids}).to_csv(os.path.join(out_dir, "cells.tsv"), sep="\t", index=False, header=False)
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)

    if not cfg.split_cellchat_by_time:
        return

    for day in np.unique(meta[cfg.time_key].astype(str).values):
        m = (meta[cfg.time_key].astype(str).values == str(day))
        if m.sum() == 0:
            continue
        sub_dir = os.path.join(os.path.dirname(out_dir), f"cellchat_input_{sanitize_name(day)}")
        ensure_dir(sub_dir)

        sub_cells = np.where(m)[0]
        sub_meta = meta.iloc[sub_cells].copy()
        X_day = X_sel.tocsr()[:, sub_cells].tocoo()

        tmp2 = os.path.join(sub_dir, "expr.mtx")
        mmwrite(tmp2, X_day)
        with open(tmp2, "rb") as f_in, gzip.open(os.path.join(sub_dir, "expr.mtx.gz"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(tmp2)

        pd.DataFrame({"gene": genes}).to_csv(os.path.join(sub_dir, "genes.tsv"), sep="\t", index=False, header=False)
        pd.DataFrame({"cell_id": sub_meta["cell_id"].values}).to_csv(os.path.join(sub_dir, "cells.tsv"), sep="\t", index=False, header=False)
        sub_meta.to_csv(os.path.join(sub_dir, "meta.csv"), index=False)


def summarize_driver_expr_raw(raw: ad.AnnData, cfg: Config, genes: List[str], out_csv: str):
    genes = [g for g in genes if g in raw.var_names]
    if len(genes) == 0:
        print("[WARN] No driver genes found in raw var_names, skip expr summary.")
        return

    gene_idx = np.array([raw.var_names.get_loc(g) for g in genes], dtype=int)
    X = get_X(raw, layer=None)  # use X (typically log/normalized) for summary

    obs = raw.obs
    rows = []
    time_vals = obs[cfg.time_key].astype(str).values
    state_vals = obs[cfg.state_key].astype(str).values

    for day in np.unique(time_vals):
        m_day = (time_vals == str(day))
        if m_day.sum() == 0:
            continue
        for ct in np.unique(state_vals[m_day]):
            m = m_day & (state_vals == str(ct))
            n = int(m.sum())
            if n == 0:
                continue
            idx = np.where(m)[0]
            if sp.issparse(X):
                Xg = X[idx, :][:, gene_idx].tocsr()
                mean_expr = np.asarray(Xg.mean(axis=0)).ravel()
                pct_expr = Xg.getnnz(axis=0) / float(n)
            else:
                Xg = np.asarray(X)[idx, :][:, gene_idx]
                mean_expr = Xg.mean(axis=0)
                pct_expr = (Xg > 0).mean(axis=0)

            for i, g in enumerate(genes):
                rows.append({
                    "timepoint": str(day),
                    "cell_type": str(ct),
                    "gene": g,
                    "mean_expr": float(mean_expr[i]),
                    "pct_expr": float(pct_expr[i]),
                    "n_cells": n
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ===================== Main =====================
def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "perturbation"))
    ensure_dir(os.path.join(cfg.out_dir, "downstream"))

    device = pick_device(cfg)
    print(f"[INFO] Device: {device}")

    # ---- load sequences (optionally downsample) ----
    X, y, class_names, time_labels = load_sequences(cfg)
    N, T, D = X.shape

    # ---- decide perturb targets ----
    if cfg.perturb_target_indices is None:
        t_list = list(range(T))
    else:
        t_list = [int(x) for x in cfg.perturb_target_indices if 0 <= int(x) < T]
        if len(t_list) == 0:
            raise ValueError(f"[ERROR] perturb_target_indices is empty or out of range (0..{T-1}).")

    targets = []
    for t in t_list:
        tag = f"t{t}_{sanitize_name(time_labels[t])}"
        targets.append((t, time_labels[t], tag))

    print(f"[INFO] perturb targets: {[tag for _, _, tag in targets]}")

    # ---- run perturbation scans ----
    ens_cache: Dict[str, Tuple[Dict[str, nn.Module], object]] = {}
    all_sum = []

    for t_idx, t_lab, tag in targets:
        out_p = os.path.join(cfg.out_dir, "perturbation", tag)
        ensure_dir(out_p)

        print(f"\n[Perturb] target={tag} (t={t_idx}, label={t_lab}) | keep_len={t_idx+1}")
        df, df_sum = scan_target_timepoint(
            cfg=cfg,
            target_t=int(t_idx),
            target_label=str(t_lab),
            X_all=X,
            y_true=y,
            class_names=class_names,
            time_labels=time_labels,
            ens_cache=ens_cache,
            device=device,
        )
        df.to_csv(os.path.join(out_p, "dose_response_all_dims.csv"), index=False)
        df_sum.to_csv(os.path.join(out_p, "dim_summary.csv"), index=False)

        top_dims = df_sum["dim"].head(int(cfg.top_k_dims)).astype(int).tolist()
        with open(os.path.join(out_p, "top_dims.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, top_dims)))

        all_sum.append(df_sum.assign(target_tag=tag))

    pd.concat(all_sum, ignore_index=True).to_csv(
        os.path.join(cfg.out_dir, "perturbation", "dim_summary_all_targets.csv"), index=False
    )

    # ---- decoder mapping ----
    hvgs = load_hvgs(cfg)
    genes, Zg = load_decoder(cfg, latent_dim=D, hvgs=hvgs)

    downstream = os.path.join(cfg.out_dir, "downstream")
    ensure_dir(downstream)

    # ---- transition genes from sequences (mean deltas) ----
    # (Optional but useful as a complementary candidate set)
    union_genes = set()

    for i in range(T - 1):
        lab_a = sanitize_name(time_labels[i])
        lab_b = sanitize_name(time_labels[i + 1])
        dz_mean = (X[:, i + 1, :] - X[:, i, :]).mean(axis=0)
        df_tr = transition_genes(genes, Zg, dz_mean, topk=max(200, cfg.top_union_transition))
        p_tr = os.path.join(downstream, f"genes_transition_{lab_a}to{lab_b}_mean.csv")
        df_tr.to_csv(p_tr, index=False, float_format="%.10f")
        union_genes.update(df_tr["gene"].astype(str).head(int(cfg.top_union_transition)).tolist())

        # one-vs-rest per class
        for k, cname in enumerate(class_names):
            mk = (y == k)
            mo = ~mk
            if mk.sum() == 0 or mo.sum() == 0:
                continue
            dz_k = (X[mk, i + 1, :] - X[mk, i, :]).mean(axis=0)
            dz_o = (X[mo, i + 1, :] - X[mo, i, :]).mean(axis=0)
            dz_diff = dz_k - dz_o
            df_tr2 = transition_genes(genes, Zg, dz_diff, topk=max(200, cfg.top_union_transition))
            p_tr2 = os.path.join(downstream, f"genes_transition_{lab_a}to{lab_b}_{cname}_vs_rest.csv")
            df_tr2.to_csv(p_tr2, index=False, float_format="%.10f")
            union_genes.update(df_tr2["gene"].astype(str).head(int(cfg.top_union_transition)).tolist())

    # ---- per-target decoder integration ----
    target_tags = [tag for _, _, tag in targets]
    for t_idx, t_lab, tag in targets:
        out_p = os.path.join(cfg.out_dir, "perturbation", tag)
        dim_sum_path = os.path.join(out_p, "dim_summary.csv")
        if not os.path.exists(dim_sum_path):
            continue
        df_sum = pd.read_csv(dim_sum_path)
        top_dims = df_sum["dim"].head(int(cfg.top_k_dims)).astype(int).tolist()

        sub = os.path.join(downstream, f"decoder_{tag}")
        ensure_dir(sub)

        # per-dim top genes
        per_dim = []
        for d in top_dims:
            per_dim.append(top_genes_by_dim(genes, Zg, int(d), int(cfg.top_k_genes_per_dim)))
        per_dim_df = pd.concat(per_dim, ignore_index=True) if per_dim else pd.DataFrame()
        if not per_dim_df.empty:
            per_dim_df.to_csv(os.path.join(sub, "top_genes_per_dim.csv"), index=False)
            union_genes.update(per_dim_df["gene"].astype(str).tolist())

        # class-specific marker rankings
        for cname in class_names:
            df_rank = integrate_marker_genes_for_class(genes, Zg, df_sum, top_dims, class_name=cname)
            out_rank = os.path.join(sub, f"marker_genes_ranked_{cname}.csv")
            df_rank.to_csv(out_rank, index=False)
            # union add
            union_genes.update(df_rank["gene"].astype(str).head(int(cfg.top_union_marker)).tolist())

    # save union
    pd.DataFrame({"gene": sorted(list(union_genes))}).to_csv(
        os.path.join(downstream, "marker_gene_candidates_union.csv"), index=False
    )

    # ---- build driver masters per class ----
    if cfg.save_driver_master:
        for cname in class_names:
            dm = build_driver_master_rrf(cfg, downstream_dir=downstream, targets=target_tags, class_name=cname)
            dm.to_csv(os.path.join(downstream, f"driver_genes_master_{cname}.csv"), index=False)
            dm[["gene", "driver_score", "direction", "sources", "rank"]].to_csv(
                os.path.join(downstream, f"driver_genes_{cname}.csv"), index=False
            )
            # also export to out_dir root for convenience
            dm.to_csv(os.path.join(cfg.out_dir, f"driver_genes_master_{cname}.csv"), index=False)
            dm[["gene", "driver_score"]].to_csv(os.path.join(cfg.out_dir, f"driver_genes_{cname}.csv"), index=False)

        # union driver list (top K from each class master)
        union_driver = set()
        for cname in class_names:
            p = os.path.join(downstream, f"driver_genes_master_{cname}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                union_driver.update(df["gene"].astype(str).head(300).tolist())
        pd.DataFrame({"gene": sorted(list(union_driver))}).to_csv(
            os.path.join(downstream, "driver_genes_union_top300_eachclass.csv"), index=False
        )
    else:
        union_driver = set()

    # ---- optional: CellChat export + expr summary ----
    if (cfg.export_cellchat_inputs or cfg.export_expr_summary) and cfg.raw_full_h5ad:
        if not os.path.exists(cfg.raw_full_h5ad):
            print(f"[WARN] raw_full_h5ad not found, skip cellchat/expr export: {cfg.raw_full_h5ad}")
        else:
            print(f"[Raw] {cfg.raw_full_h5ad}")
            raw = ad.read_h5ad(cfg.raw_full_h5ad)

            # genes for export / summary
            genes_use = sorted(list(union_driver)) if union_driver else sorted(list(union_genes))

            if cfg.export_expr_summary:
                out_expr = os.path.join(downstream, "expr_driver_by_state_time_fullgenes.csv")
                summarize_driver_expr_raw(raw, cfg, genes=genes_use[:300], out_csv=out_expr)
                print(f"[Expr] Saved: {out_expr}")

            if cfg.export_cellchat_inputs:
                out_cc = os.path.join(cfg.out_dir, "cellchat_input_all")
                export_cellchat_inputs(raw, cfg, genes_use=genes_use, out_dir=out_cc)
                print(f"[CellChat] Inputs saved under: {os.path.dirname(out_cc)}")

    # ---- save run config ----
    run_meta = cfg.__dict__.copy()
    for _k in ["TARGET_ENDPOINT_CLUSTERS", "folds"]:
        if _k in run_meta and isinstance(run_meta[_k], tuple):
            run_meta[_k] = list(run_meta[_k])
    with open(os.path.join(cfg.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n[Done] Outputs at: {cfg.out_dir}")


if __name__ == "__main__":
    main()
