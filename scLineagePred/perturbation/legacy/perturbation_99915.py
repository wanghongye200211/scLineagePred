# -*- coding: utf-8 -*-
"""
perturbation_99915.py  (Perturbation + downstream exports)
==========================================================

Clone-centric perturbation scan for GSE99915 (Reprogrammed vs Failed) on latent space (latent space / 潜在空间).

What this script does
---------------------
[Part A] Perturbation scan (扰动扫描):
    For a chosen timepoint and latent dimension j:
        z(timepoint)[:, j] <- z(timepoint)[:, j] * fold
    Other dims/timepoints unchanged.
    Then feed sequences into your trained ensemble (BiLSTM + RNN + Transformer + stacking LR)
    and compute flip_rate / mean_prob changes.

[Part B] Gene mapping (基因映射):
    Prefer decoder weights (genes.txt + Z_genes.npy) if available (decoder / 解码器).
    If decoder files are missing, fallback to gene–latent correlation mapping on HVG matrix.

[Part C] Downstream exports:
    - downstream/driver_genes_master.csv  (gene + driver_score + direction + sources + rank)
    - downstream/driver_genes.csv         (subset columns)
    - (optional) cellchat_input_all/      (expr.mtx.gz + genes.tsv + cells.tsv + meta.csv) from RAW full-gene h5ad
    - (optional) downstream/expr_driver_by_state_time_fullgenes.csv

It DOES NOT modify your training assets.

References (your existing scripts):
- 140802 perturbation pipeline for structure
- class_99915.py for model architectures & filenames

Run
---
python perturbation_99915.py

Then (plots):
python plot_perturbation_99915.py
python boxplot_marker_99915.py
"""

from __future__ import annotations
import os
import json
import pickle
import gzip
import shutil
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import anndata as ad
import scipy.sparse as sp
from scipy.io import mmwrite


# ===================== Config =====================
@dataclass
class Config:
    # --- data (HVG-integrated, training asset) ---
    adata_h5ad: str = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_final_integrated.h5ad"
    latent_key: str = "X_latent"
    clone_key: str = "clone_id"
    time_key: str = "time_info"
    state_key: str = "state_info"
    time_order: Tuple[str, ...] = ("Day6", "Day9", "Day12", "Day15", "Day21", "Day28")
    final_timepoint: str = "Day28"

    # label names used in your 99915 tasks
    positive_label: str = "Reprogrammed"
    negative_label: str = "Failed"

    # --- classification ensemble (from classification/GSE99915/saved_models) ---
    model_dir: str = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/saved_models"
    setting: str = "All_Days"  # All_Days / Obs_Day21 / Obs_Day15 / Obs_Day12 / Obs_Day9
    seeds: Dict[str, int] = field(default_factory=dict)

    # must match class_99915.py
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    nhead: int = 4

    # --- perturbation ---
    # choose any subset of: Day6/Day9/Day12/Day15/Day21/Day28 (case-insensitive, 'day12' ok)
    perturb_targets: Tuple[str, ...] = ("Day6", "Day9", "Day12", "Day15", "Day21", "Day28")
    # --- marker master (global) ---
    # Only these timepoints are used to build the GLOBAL marker genes table.
    # 只用这些时间点做“总 marker 基因”融合排序（RRF）。
    marker_master_targets: Tuple[str, ...] = ("Day6", "Day9", "Day12", "Day15", "Day21", "Day28")
    rrf_k: int = 50  # Reciprocal Rank Fusion k (排名融合 RRF 的 k)

    folds: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    max_ood_rate_for_ranking: float = 0.10
    top_k_dims: int = 30
    top_k_genes_per_dim: int = 50
    top_k_transition_genes: int = 200

    # --- gene mapping (decoder preferred; correlation fallback) ---
    decoder_dir: str = "/Users/wanghongye/python/scLineagetracer/autoencoder/results/GSE99915"
    genes_txt: str = field(init=False)
    z_genes_npy: str = field(init=False)
    gene_map_fallback: str = "corr"  # "corr" or "none"
    max_cells_for_corr_map: int = 50000
    corr_map_seed: int = 0

    # --- outputs ---
    out_dir: str = "/Users/wanghongye/python/scLineagetracer/Downstream_99915"

    # --- raw full-gene for downstream exports (optional, NO change to training assets) ---
    raw_full_h5ad: str = "/Users/wanghongye/python/scLineagetracer/GSE99915/CellTagging_adata_preprocessed.h5ad"
    raw_counts_layer: str = "counts"  # preferred counts layer for CellChat export; fallback to X if missing
    export_cellchat_inputs: bool = False
    split_cellchat_by_time: bool = True
    cellchat_genes_mode: str = "driver"  # "full" or "driver"
    export_expr_summary: bool = True

    # --- union / driver master export ---
    top_union_transition: int = 300
    top_union_marker: int = 800
    per_dim_use_top_n_dims: int = 30
    save_driver_master: bool = True

    # --- device ---
    device: str = "auto"
    batch_size: int = 2048

    def __post_init__(self):
        if not self.seeds:
            # must match class_99915.py
            self.seeds = {
                "All_Days": 2026,
                "Obs_Day21": 2024,
                "Obs_Day15": 42,
                "Obs_Day12": 123,
                "Obs_Day9": 999,
            }
        self.genes_txt = os.path.join(self.decoder_dir, "genes.txt")
        self.z_genes_npy = os.path.join(self.decoder_dir, "Z_genes.npy")
        if 0.0 not in self.folds or 1.0 not in self.folds:
            raise ValueError("folds must include 0.0 and 1.0")


# ===================== utils =====================
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def np_lower(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="U")
    return np.char.lower(arr)

def pick_device(cfg: Config) -> str:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return cfg.device

def normalize_timepoint(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    if s.lower().startswith("day"):
        return "Day" + s[3:]
    # allow pure numeric like "12" -> "Day12"
    if s.isdigit():
        return "Day" + s
    return s

def setting_mask_indices(setting: str) -> Optional[List[int]]:
    # time indices: [Day6, Day9, Day12, Day15, Day21, Day28] -> [0..5]
    if setting == "All_Days": return None
    if setting == "Obs_Day21": return [5]
    if setting == "Obs_Day15": return [4, 5]
    if setting == "Obs_Day12": return [3, 4, 5]
    if setting == "Obs_Day9": return [2, 3, 4, 5]
    raise ValueError(f"Unknown setting: {setting}")

def apply_setting_mask(X: np.ndarray, setting: str) -> np.ndarray:
    Xm = X.copy()
    idxs = setting_mask_indices(setting)
    if idxs is not None:
        Xm[:, idxs, :] = 0.0
    return Xm


# ===================== models (must match class_99915.py) =====================
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


def load_ensemble(cfg: Config, input_dim: int, device: str):
    setting = cfg.setting
    seed = cfg.seeds[setting]

    models = {
        "BiLSTM": LSTMModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
        "RNN":   RNNModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
        "Trans": TransformerModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead).to(device),
    }
    for name in models:
        pth = os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth")
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing base model: {pth}")
        models[name].load_state_dict(torch.load(pth, map_location=device))
        models[name].eval()

    pkl = os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Missing stacking LR: {pkl}")
    with open(pkl, "rb") as f:
        lr = pickle.load(f)
    return models, lr


@torch.no_grad()
def predict_prob_stack(models: Dict[str, nn.Module], lr, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """
    Predict P(class=1) for all samples in X, using stacking LR over base model probabilities.
    """
    N = X.shape[0]
    probs = {k: [] for k in models.keys()}
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device)
        for name, m in models.items():
            out = m(xb)
            probs[name].append(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy())
    feats = np.stack([np.concatenate(probs[k]) for k in ["BiLSTM", "RNN", "Trans"]], axis=1)
    return lr.predict_proba(feats)[:, 1].astype(np.float32)


@torch.no_grad()
def predict_prob_stack_perturb(models: Dict[str, nn.Module], lr, X_base: np.ndarray, device: str, batch_size: int,
                              t_index: int, dim: int, fold: float) -> np.ndarray:
    """
    Like predict_prob_stack, but apply perturbation on-the-fly:
        xb[:, t_index, dim] *= fold
    Avoids copying the whole X array for each (dim, fold).
    """
    N = X_base.shape[0]
    probs = {k: [] for k in models.keys()}
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X_base[i:i+batch_size].astype(np.float32)).to(device)
        xb[:, t_index, dim] = xb[:, t_index, dim] * float(fold)
        for name, m in models.items():
            out = m(xb)
            probs[name].append(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy())
    feats = np.stack([np.concatenate(probs[k]) for k in ["BiLSTM", "RNN", "Trans"]], axis=1)
    return lr.predict_proba(feats)[:, 1].astype(np.float32)


# ===================== data (clone prototypes) =====================
def build_clone_prototypes(adata: ad.AnnData, cfg: Config):
    """
    Build clone-wise mean latent per timepoint (missing filled with 0) + presence mask.
    Also infer each clone's final fate label using final_timepoint majority vote; fallback to all-time majority.

    Returns
    -------
    clones: (C,) str
    X_seq: (C,T,D) float32
    mask_seq: (C,T) int8  (1=present, 0=missing)
    fate: (C,) str
    """
    obs = adata.obs
    Z = np.asarray(adata.obsm[cfg.latent_key], dtype=np.float32)

    t = obs[cfg.time_key].astype(str).to_numpy()
    c = obs[cfg.clone_key].astype(str).to_numpy()
    s = obs[cfg.state_key].astype(str).to_numpy()

    c_low = np_lower(c)
    valid_clone = ~np.isin(c_low, ["-1", "nan", "none", ""])
    t, c, s, Z = t[valid_clone], c[valid_clone], s[valid_clone], Z[valid_clone]

    # keep only target timepoints
    t_norm = np.array([normalize_timepoint(x) for x in t], dtype=object)
    time_order = tuple(normalize_timepoint(x) for x in cfg.time_order)
    keep_t = np.isin(t_norm, list(time_order))
    t_norm, c, s, Z = t_norm[keep_t], c[keep_t], s[keep_t], Z[keep_t]

    if Z.shape[0] == 0:
        raise ValueError("No cells left after filtering by clone/time_order.")

    clones, inv = np.unique(c, return_inverse=True)
    C = len(clones)
    T = len(time_order)
    D = Z.shape[1]

    X_seq = np.zeros((C, T, D), dtype=np.float32)
    mask_seq = np.zeros((C, T), dtype=np.int8)

    # mean latent per (clone, time)
    for ti, tp in enumerate(time_order):
        m = (t_norm == tp)
        if not np.any(m):
            continue
        inv_m = inv[m]
        Zm = Z[m].astype(np.float64)
        sums = np.zeros((C, D), dtype=np.float64)
        np.add.at(sums, inv_m, Zm)
        cnt = np.bincount(inv_m, minlength=C).astype(np.float64)
        ok = cnt > 0
        X_seq[ok, ti, :] = (sums[ok] / cnt[ok, None]).astype(np.float32)
        mask_seq[ok, ti] = 1

    # fate label
    df = pd.DataFrame({"clone": c, "time": t_norm, "state": s})
    df = df[df["state"].isin([cfg.positive_label, cfg.negative_label])].copy()
    fate_final = {}
    final_tp = normalize_timepoint(cfg.final_timepoint)
    if (df["time"] == final_tp).any():
        df_f = df[df["time"] == final_tp]
        fate_final = df_f.groupby("clone")["state"].agg(lambda x: x.value_counts().index[0]).to_dict()
    fate_all = df.groupby("clone")["state"].agg(lambda x: x.value_counts().index[0]).to_dict()

    fate = np.array([fate_final.get(cl, fate_all.get(cl, "Unknown")) for cl in clones], dtype=object)

    # keep clones with known fate
    keep_f = (fate == cfg.positive_label) | (fate == cfg.negative_label)
    keep_obs = (mask_seq.sum(axis=1) > 0)
    keep = keep_f & keep_obs

    clones = clones[keep]
    X_seq = X_seq[keep]
    mask_seq = mask_seq[keep]
    fate = fate[keep]

    return clones, X_seq, mask_seq, fate, time_order


def infer_positive_label(p0: np.ndarray, fate: np.ndarray, cfg: Config) -> str:
    """
    Determine which biological label corresponds to p(class=1).
    """
    p0 = np.asarray(p0, dtype=np.float32).reshape(-1)
    fate = np.asarray(fate, dtype=object).reshape(-1)

    m_pos = (fate == cfg.positive_label)
    m_neg = (fate == cfg.negative_label)
    if m_pos.any() and m_neg.any():
        return cfg.positive_label if float(p0[m_pos].mean()) > float(p0[m_neg].mean()) else cfg.negative_label
    return "pos_class"


# ===================== gene mapping (decoder or correlation fallback) =====================
def load_decoder(cfg: Config, latent_dim: int, hvgs: set):
    if not (os.path.exists(cfg.genes_txt) and os.path.exists(cfg.z_genes_npy)):
        raise FileNotFoundError(f"Missing decoder files: {cfg.genes_txt} or {cfg.z_genes_npy}")

    genes = [ln.strip() for ln in open(cfg.genes_txt, "r", encoding="utf-8") if ln.strip()]
    Zg = np.load(cfg.z_genes_npy).astype(np.float32)

    if Zg.ndim != 2:
        raise ValueError(f"Z_genes must be 2D, got {Zg.shape}")
    if Zg.shape[1] != latent_dim:
        raise ValueError(f"Z_genes dim mismatch: {Zg.shape[1]} vs latent_dim={latent_dim}")
    if len(genes) != Zg.shape[0]:
        m = min(len(genes), Zg.shape[0])
        genes = genes[:m]
        Zg = Zg[:m, :]

    overlap = len(set(genes) & hvgs) / max(len(set(genes)), 1)
    if overlap < 0.50:
        raise ValueError(
            f"Decoder genes overlap with HVG is too low ({overlap:.3f}). "
            f"Likely wrong decoder_dir: {cfg.decoder_dir}"
        )
    return np.array(genes, dtype=object), Zg


def corr_gene_map(adata: ad.AnnData, cfg: Config, latent_dim: int):
    """
    Fallback mapping: gene–latent Pearson correlation on HVG matrix.

    Zg[g, d] = corr(expr_gene, latent_dim_d)
    """
    rng = np.random.default_rng(cfg.corr_map_seed)
    n = adata.n_obs
    if n == 0:
        raise ValueError("Empty adata for corr mapping.")
    take = min(int(cfg.max_cells_for_corr_map), n)
    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)

    Z = np.asarray(adata.obsm[cfg.latent_key], dtype=np.float32)[idx]
    if Z.shape[1] != latent_dim:
        raise ValueError("latent_dim mismatch in corr_gene_map")

    X = adata.X[idx]
    if sp.issparse(X):
        X = X.tocsr()
        # mean and std for genes
        mu_x = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
        ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float64)
        var_x = np.maximum(ex2 - mu_x * mu_x, 1e-12)
        sd_x = np.sqrt(var_x)
    else:
        X = np.asarray(X, dtype=np.float64)
        mu_x = X.mean(axis=0)
        sd_x = X.std(axis=0) + 1e-12

    mu_z = Z.mean(axis=0).astype(np.float64)
    sd_z = Z.std(axis=0).astype(np.float64) + 1e-12

    # compute corr for each dim via sparse/dense matmul
    G = adata.n_vars
    Zg = np.zeros((G, latent_dim), dtype=np.float32)
    n_float = float(take)

    if sp.issparse(adata.X):
        Xt = X.T.tocsr()
        for d in range(latent_dim):
            z = Z[:, d].astype(np.float64)
            exz = (Xt @ z) / n_float  # E[x*z] per gene
            cov = exz - mu_x * float(mu_z[d])
            corr = cov / (sd_x * float(sd_z[d]))
            Zg[:, d] = np.clip(corr, -1.0, 1.0).astype(np.float32)
    else:
        for d in range(latent_dim):
            z = Z[:, d].astype(np.float64)
            cov = ((X - mu_x[None, :]) * (z[:, None] - float(mu_z[d]))).mean(axis=0)
            corr = cov / (sd_x * float(sd_z[d]))
            Zg[:, d] = np.clip(corr, -1.0, 1.0).astype(np.float32)

    genes = adata.var_names.astype(str).to_numpy(dtype=object)
    return genes, Zg


def top_genes_by_dim(genes: np.ndarray, Zg: np.ndarray, dim: int, k: int) -> pd.DataFrame:
    w = Zg[:, dim]
    idx = np.argsort(-np.abs(w))[:k]
    return pd.DataFrame({"gene": genes[idx], "loading": w[idx], "abs_loading": np.abs(w[idx]), "dim": dim})


def integrate_marker_genes(genes: np.ndarray, Zg: np.ndarray, dim_sum: pd.DataFrame, top_dims: List[int]):
    s = dim_sum.set_index("dim")
    dims = [d for d in top_dims if d in s.index]
    if not dims:
        return pd.DataFrame()

    # prefer *_present columns if exist
    col = "best_delta_mean_prob_present" if "best_delta_mean_prob_present" in s.columns else "best_delta_mean_prob"
    w_signed = np.array([float(s.loc[d][col]) for d in dims], dtype=np.float32)
    w_abs = np.abs(w_signed)

    M = Zg[:, dims]  # [G,K]
    score_abs = (np.abs(M) * w_abs[None, :]).sum(axis=1)
    score_signed = (M * w_signed[None, :]).sum(axis=1)

    df = pd.DataFrame({"gene": genes, "score_abs": score_abs, "score_signed": score_signed})
    return df.sort_values("score_abs", ascending=False).reset_index(drop=True)


def transition_genes(genes: np.ndarray, Zg: np.ndarray, deltaZ: np.ndarray, topk: int):
    dz = np.asarray(deltaZ, dtype=np.float64).reshape(-1)
    dz_norm = float(np.linalg.norm(dz))
    if dz_norm < 1e-12:
        s = np.zeros((Zg.shape[0],), dtype=np.float64)
        idx = np.argsort(-s)[:topk]
        return pd.DataFrame({"gene": genes[idx], "score": s[idx], "proj": s[idx], "cos_signed": s[idx]})

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


# ===================== driver master + exports =====================
def build_driver_master(union_genes: List[str], pos_label: str, downstream_dir: str, cfg: Config) -> pd.DataFrame:
    """
    Global marker genes table via Reciprocal Rank Fusion (RRF / 倒数排名融合).

    IMPORTANT (关键点):
      - ONLY uses timepoint-specific perturbation marker lists:
            downstream/decoder_<DayX>/marker_genes_ranked.csv
      - Only the timepoints in cfg.marker_master_targets are considered.
      - RRF score: sum_t 1 / (k + rank_t(g)), with k = cfg.rrf_k (default 50)

    This matches your requested Scheme-B (方案B：排名融合).
    """
    k = int(getattr(cfg, "rrf_k", 50))
    targets = getattr(cfg, "marker_master_targets", None)
    targets = targets if (targets is not None and len(targets) > 0) else cfg.perturb_targets
    targets = [normalize_timepoint(t) for t in targets]
    # If using an observation setting (e.g. Obs_Day15), timepoints after the observed window are masked to 0.
    # Perturbing masked timepoints has no effect, so we exclude them from the GLOBAL marker fusion.
    mask_idxs = setting_mask_indices(cfg.setting)
    if mask_idxs is not None:
        allowed = {normalize_timepoint(cfg.time_order[i]) for i in range(len(cfg.time_order)) if i not in mask_idxs}
        targets = [t for t in targets if normalize_timepoint(t) in allowed]

    score: Dict[str, float] = {}
    signed: Dict[str, float] = {}
    sources: Dict[str, set] = {}

    for tgt in targets:
        p = os.path.join(downstream_dir, f"decoder_{tgt}", "marker_genes_ranked.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue

        # list is already ranked (descending score_abs)
        df = df.head(cfg.top_union_marker).copy()
        genes_list = df["gene"].astype(str).tolist()

        sign_list = None
        if "score_signed" in df.columns:
            sign_list = df["score_signed"].to_numpy(dtype=np.float32)

        for r, g in enumerate(genes_list, start=1):
            g = str(g)
            w = 1.0 / float(k + r)
            score[g] = score.get(g, 0.0) + w
            if sign_list is not None:
                signed[g] = signed.get(g, 0.0) + w * float(np.sign(sign_list[r - 1]))
            sources.setdefault(g, set()).add(f"marker_{tgt}")

    # union set to output
    union_set = set(map(str, union_genes))
    # ensure any missing gene has 0 score (usually shouldn't happen)
    for g in union_set:
        score.setdefault(g, 0.0)
        signed.setdefault(g, 0.0)
        sources.setdefault(g, set())

    neg_label = cfg.negative_label if pos_label == cfg.positive_label else cfg.positive_label

    rows = []
    for g in union_set:
        ds = float(score.get(g, 0.0))
        ss = float(signed.get(g, 0.0))
        if ss > 0:
            direction = f"{pos_label}_push"
        elif ss < 0:
            direction = f"{neg_label}_push"
        else:
            direction = "Neutral"
        rows.append({
            "gene": g,
            "driver_score": ds,
            "signed_score": ss,
            "direction": direction,
            "sources": "|".join(sorted(list(sources.get(g, set())))) if sources.get(g, None) else "",
        })

    dfm = pd.DataFrame(rows).sort_values(["driver_score", "gene"], ascending=[False, True]).reset_index(drop=True)
    dfm["rank"] = np.arange(1, len(dfm) + 1)
    return dfm


def get_X(adata: ad.AnnData, layer: str = None):
    if layer and (layer in getattr(adata, "layers", {})):
        X = adata.layers[layer]
    else:
        X = adata.X
    if sp.issparse(X):
        return X.tocsr()
    return np.asarray(X)


def export_cellchat_inputs_from_raw(raw: ad.AnnData, cfg: Config, genes_use: List[str], out_dir: str):
    ensure_dir(out_dir)

    tvals = raw.obs[cfg.time_key].astype(str).to_numpy()
    time_order = [normalize_timepoint(x) for x in cfg.time_order]
    tvals = np.array([normalize_timepoint(x) for x in tvals], dtype=object)

    keep = np.isin(tvals, time_order)
    raw = raw[keep].copy()

    if cfg.cellchat_genes_mode == "driver":
        genes = [g for g in genes_use if g in raw.var_names]
    else:
        genes = raw.var_names.astype(str).tolist()
    if len(genes) == 0:
        raise ValueError("No genes selected for CellChat export (empty).")

    gene_idx = np.array([raw.var_names.get_loc(g) for g in genes], dtype=int)

    X = get_X(raw, layer=(cfg.raw_counts_layer if cfg.raw_counts_layer else None))
    if sp.issparse(X):
        X_sel = X[:, gene_idx].T.tocoo()  # genes×cells
    else:
        X_sel = sp.coo_matrix(np.asarray(X)[:, gene_idx].T)

    cell_ids = raw.obs_names.astype(str).to_numpy()
    meta = pd.DataFrame({
        "cell_id": cell_ids,
        cfg.time_key: raw.obs[cfg.time_key].astype(str).values,
        cfg.state_key: raw.obs[cfg.state_key].astype(str).values,
    })
    if cfg.clone_key in raw.obs.columns:
        meta[cfg.clone_key] = raw.obs[cfg.clone_key].astype(str).values

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

    # split
    for day in time_order:
        m = np.array([normalize_timepoint(x) for x in meta[cfg.time_key].astype(str).values], dtype=object) == day
        if m.sum() == 0:
            continue
        sub_dir = os.path.join(os.path.dirname(out_dir), f"cellchat_input_{day}")
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


def summarize_driver_expr_raw(raw: ad.AnnData, cfg: Config, driver_df: pd.DataFrame, out_csv: str, top_n: int = 300):
    time_order = [normalize_timepoint(x) for x in cfg.time_order]
    tvals = np.array([normalize_timepoint(x) for x in raw.obs[cfg.time_key].astype(str).values], dtype=object)
    keep = np.isin(tvals, time_order)
    raw = raw[keep].copy()

    genes = driver_df["gene"].astype(str).head(top_n).tolist()
    genes = [g for g in genes if g in raw.var_names]
    if len(genes) == 0:
        print("[WARN] No driver genes found in raw var_names, skip expr summary.")
        return

    gene_idx = np.array([raw.var_names.get_loc(g) for g in genes], dtype=int)
    X = get_X(raw, layer=None)  # use X for summary (often log/normalized)

    obs = raw.obs
    rows = []
    for day in time_order:
        m_day = np.array([normalize_timepoint(x) for x in obs[cfg.time_key].astype(str).values], dtype=object) == day
        if m_day.sum() == 0:
            continue
        for ct in np.unique(obs.loc[m_day, cfg.state_key].astype(str).values):
            m = m_day & (obs[cfg.state_key].astype(str).values == ct)
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


# ===================== perturbation core =====================
def run_target(cfg: Config, target: str, X_seq: np.ndarray, mask_seq: np.ndarray, fate_true: np.ndarray,
               models, lr, device: str, time_order: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = normalize_timepoint(target)
    if target not in time_order:
        raise ValueError(f"Unknown perturb_target={target}. Must be in {time_order}")
    t_idx = list(time_order).index(target)

    X0 = apply_setting_mask(X_seq, cfg.setting)
    p0 = predict_prob_stack(models, lr, X0, device, cfg.batch_size)
    y0 = (p0 > 0.5).astype(np.int64)
    base_mean_all = float(p0.mean())

    present = (mask_seq[:, t_idx] == 1)
    n_total = int(X0.shape[0])
    n_present = int(present.sum())
    base_mean_present = float(p0[present].mean()) if n_present > 0 else float("nan")

    rows = []
    D = X0.shape[2]
    for dim in range(D):
        # reference range computed on present clones only, from the *unmasked* latent at this timepoint
        ref_vals = X_seq[present, t_idx, dim].astype(np.float32) if n_present > 0 else None
        ref_min = float(ref_vals.min()) if n_present > 0 else float("nan")
        ref_max = float(ref_vals.max()) if n_present > 0 else float("nan")

        for fold in cfg.folds:
            p = predict_prob_stack_perturb(models, lr, X0, device, cfg.batch_size, t_idx, dim, float(fold))
            y = (p > 0.5).astype(np.int64)

            # all clones metrics
            frac_pos_all = float(y.mean())
            mean_prob_all = float(p.mean())
            delta_mean_prob_all = float(mean_prob_all - base_mean_all)
            flip_rate_all = float((y != y0).mean())

            # present-only metrics
            if n_present > 0:
                yp = y[present]
                pp = p[present]
                y0p = y0[present]
                frac_pos_present = float(yp.mean())
                mean_prob_present = float(pp.mean())
                delta_mean_prob_present = float(mean_prob_present - base_mean_present)
                flip_rate_present = float((yp != y0p).mean())

                new_vals = (X_seq[present, t_idx, dim] * float(fold)).astype(np.float32)
                ood = float(((new_vals < ref_min) | (new_vals > ref_max)).mean())
            else:
                frac_pos_present = float("nan")
                mean_prob_present = float("nan")
                delta_mean_prob_present = float("nan")
                flip_rate_present = float("nan")
                ood = float("nan")

            rows.append({
                "perturb_target": target,
                "t_index": int(t_idx),
                "dim": int(dim),
                "fold": float(fold),
                "n_total": n_total,
                "n_present": n_present,

                "frac_pos_all": frac_pos_all,
                "frac_neg_all": 1.0 - frac_pos_all,
                "mean_prob_all": mean_prob_all,
                "delta_mean_prob_all": delta_mean_prob_all,
                "flip_rate_all": flip_rate_all,

                "frac_pos_present": frac_pos_present,
                "frac_neg_present": 1.0 - frac_pos_present if np.isfinite(frac_pos_present) else float("nan"),
                "mean_prob_present": mean_prob_present,
                "delta_mean_prob_present": delta_mean_prob_present,
                "flip_rate_present": flip_rate_present,

                "ood_rate_present": ood,
                "ref_min_present": ref_min,
                "ref_max_present": ref_max,
            })

    df = pd.DataFrame(rows)

    # dim summary (rank by present metrics if available)
    sums = []
    for dim, g in df.groupby("dim"):
        g_in = g[g["ood_rate_present"] <= cfg.max_ood_rate_for_ranking]
        if len(g_in) == 0:
            g_in = g
        best = g_in.sort_values(["flip_rate_present", "delta_mean_prob_present"], ascending=[False, False]).iloc[0]
        sums.append({
            "perturb_target": target,
            "dim": int(dim),

            "best_fold": float(best["fold"]),
            "best_flip_rate_present": float(best["flip_rate_present"]),
            "best_delta_mean_prob_present": float(best["delta_mean_prob_present"]),
            "best_abs_delta_mean_prob_present": float(abs(best["delta_mean_prob_present"])),

            "best_flip_rate_all": float(best["flip_rate_all"]),
            "best_delta_mean_prob_all": float(best["delta_mean_prob_all"]),
            "best_abs_delta_mean_prob_all": float(abs(best["delta_mean_prob_all"])),

            "best_ood_rate_present": float(best["ood_rate_present"]),
            "n_total": int(best["n_total"]),
            "n_present": int(best["n_present"]),
        })
    df_sum = pd.DataFrame(sums).sort_values(["best_flip_rate_present", "best_abs_delta_mean_prob_present"],
                                            ascending=[False, False])
    return df, df_sum


# ===================== main =====================
def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    device = pick_device(cfg)

    print(f"[Load] {cfg.adata_h5ad}")
    adata = ad.read_h5ad(cfg.adata_h5ad)
    if cfg.latent_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{cfg.latent_key}']")

    hvgs = set(adata.var_names.astype(str).tolist())

    clones, X_seq, mask_seq, fate, time_order = build_clone_prototypes(adata, cfg)
    C, T, D = X_seq.shape
    print(f"[Clone] n={C} T={T} latent_dim={D}")
    print(f"[Missing] per timepoint present clones: " +
          ", ".join([f"{tp}:{int(mask_seq[:,i].sum())}" for i,tp in enumerate(time_order)]))

    models, lr = load_ensemble(cfg, input_dim=D, device=device)

    # infer which biological label corresponds to p(class=1)
    X0_base = apply_setting_mask(X_seq, cfg.setting)
    p0_base = predict_prob_stack(models, lr, X0_base, device, cfg.batch_size)
    pos_label = infer_positive_label(p0_base, fate, cfg)
    neg_label = cfg.negative_label if pos_label == cfg.positive_label else cfg.positive_label
    print(f"[Info] p(class=1) seems to correspond to: {pos_label} (other={neg_label})")

    # run perturbation targets
    perturb_root = os.path.join(cfg.out_dir, "perturbation")
    ensure_dir(perturb_root)

    all_sum = []
    for target in cfg.perturb_targets:
        # skip masked timepoints under observation setting (e.g. Obs_Day15 masks Day21/Day28)
        mask_idxs_p = setting_mask_indices(cfg.setting)
        if mask_idxs_p is not None:
            masked_tp = {normalize_timepoint(cfg.time_order[i]) for i in mask_idxs_p if i < len(cfg.time_order)}
        else:
            masked_tp = set()
        tgt = normalize_timepoint(target)
        if tgt in masked_tp:
            print(f"[Skip] target={tgt} is masked under setting={cfg.setting}. Use setting=All_Days/Obs_Day21 if you want to perturb it.")
            continue
        out_p = os.path.join(perturb_root, tgt)
        ensure_dir(out_p)

        print(f"[Perturb] target={tgt} setting={cfg.setting} folds={len(cfg.folds)} dims={D}")
        df, df_sum = run_target(cfg, tgt, X_seq, mask_seq, fate, models, lr, device, time_order)

        df.to_csv(os.path.join(out_p, "dose_response_all_dims.csv"), index=False)
        df_sum.to_csv(os.path.join(out_p, "dim_summary.csv"), index=False)

        top_dims = df_sum["dim"].head(cfg.top_k_dims).astype(int).tolist()
        with open(os.path.join(out_p, "top_dims.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, top_dims)))

        all_sum.append(df_sum)

    pd.concat(all_sum, ignore_index=True).to_csv(
        os.path.join(perturb_root, "dim_summary_all_targets.csv"), index=False
    )

    # ---- gene mapping ----
    downstream = os.path.join(cfg.out_dir, "downstream")
    ensure_dir(downstream)

    genes = None
    Zg = None
    used_map = None

    # try decoder first
    try:
        genes, Zg = load_decoder(cfg, latent_dim=D, hvgs=hvgs)
        used_map = "decoder"
        print(f"[GeneMap] Using decoder weights from: {cfg.decoder_dir}")
    except Exception as e:
        print(f"[WARN] Decoder mapping failed: {e}")
        if cfg.gene_map_fallback == "corr":
            genes, Zg = corr_gene_map(adata, cfg, latent_dim=D)
            used_map = "corr"
            print(f"[GeneMap] Using correlation fallback on HVG matrix, cells<= {cfg.max_cells_for_corr_map}")
        else:
            print("[GeneMap] Disabled (no decoder, no fallback). Skip gene-level exports.")
            used_map = "none"

    # transitions + markers + union genes
    union = set()

    if used_map != "none":
        # transition genes for each consecutive pair in time_order
        for i in range(len(time_order) - 1):
            a, b = time_order[i], time_order[i+1]
            ma = (mask_seq[:, i] == 1)
            mb = (mask_seq[:, i+1] == 1)
            both = ma & mb
            if int(both.sum()) == 0:
                continue

            dz_mean = (X_seq[both, i+1, :] - X_seq[both, i, :]).mean(axis=0)
            fn_mean = f"genes_transition_{a}to{b}_mean.csv"
            transition_genes(genes, Zg, dz_mean, cfg.top_k_transition_genes).to_csv(
                os.path.join(downstream, fn_mean), index=False, float_format="%.10f"
            )
            union.update(pd.read_csv(os.path.join(downstream, fn_mean))["gene"].head(cfg.top_union_transition).astype(str).tolist())

            # fateDiff (pos vs neg), only if both groups exist
            mpos = (fate == cfg.positive_label) & both
            mneg = (fate == cfg.negative_label) & both
            if mpos.any() and mneg.any():
                dz_pos = (X_seq[mpos, i+1, :] - X_seq[mpos, i, :]).mean(axis=0)
                dz_neg = (X_seq[mneg, i+1, :] - X_seq[mneg, i, :]).mean(axis=0)
                dz_diff = dz_pos - dz_neg
                fn_diff = f"genes_transition_{a}to{b}_fateDiff.csv"
                transition_genes(genes, Zg, dz_diff, cfg.top_k_transition_genes).to_csv(
                    os.path.join(downstream, fn_diff), index=False, float_format="%.10f"
                )

        # per target: per-dim top genes + integrated markers
        for target in cfg.perturb_targets:
            tgt = normalize_timepoint(target)
            out_p = os.path.join(perturb_root, tgt)
            df_sum = pd.read_csv(os.path.join(out_p, "dim_summary.csv"))
            # rank dims by present metrics
            df_sum = df_sum.sort_values(["best_flip_rate_present", "best_abs_delta_mean_prob_present"], ascending=[False, False])
            top_dims = df_sum["dim"].head(cfg.top_k_dims).astype(int).tolist()

            sub = os.path.join(downstream, f"decoder_{tgt}")
            ensure_dir(sub)

            per_dim = []
            for d in top_dims[:min(cfg.per_dim_use_top_n_dims, len(top_dims))]:
                per_dim.append(top_genes_by_dim(genes, Zg, d, cfg.top_k_genes_per_dim))
            if per_dim:
                per_dim_df = pd.concat(per_dim, ignore_index=True)
                per_dim_df.to_csv(os.path.join(sub, "top_genes_per_dim.csv"), index=False)
                union.update(per_dim_df["gene"].astype(str).tolist())

            df_rank = integrate_marker_genes(genes, Zg, df_sum.set_index("dim", drop=False), top_dims)
            df_rank.to_csv(os.path.join(sub, "marker_genes_ranked.csv"), index=False)
            union.update(df_rank["gene"].head(cfg.top_union_marker).astype(str).tolist())

        pd.DataFrame({"gene": sorted(list(union))}).to_csv(os.path.join(downstream, "marker_gene_candidates_union.csv"), index=False)

        # driver master
        if cfg.save_driver_master:
            # Build GLOBAL marker genes only from selected timepoints (cfg.marker_master_targets)
            master_targets = [normalize_timepoint(t) for t in getattr(cfg, "marker_master_targets", cfg.perturb_targets)]
            mask_idxs_m = setting_mask_indices(cfg.setting)
            if mask_idxs_m is not None:
                allowed_m = {normalize_timepoint(cfg.time_order[i]) for i in range(len(cfg.time_order)) if i not in mask_idxs_m}
                master_targets = [t for t in master_targets if normalize_timepoint(t) in allowed_m]
            union_master = set()
            for mt in master_targets:
                p = os.path.join(downstream, f"decoder_{mt}", "marker_genes_ranked.csv")
                if not os.path.exists(p):
                    continue
                df_mt = pd.read_csv(p)
                if df_mt.empty:
                    continue
                union_master.update(df_mt["gene"].astype(str).head(cfg.top_union_marker).tolist())

            union_list = sorted(list(union_master)) if len(union_master) > 0 else sorted(list(union))
            driver_master = build_driver_master(union_list, pos_label=pos_label, downstream_dir=downstream, cfg=cfg)
            driver_master.to_csv(os.path.join(downstream, "driver_genes_master.csv"), index=False)
            driver_master[["gene", "driver_score", "direction", "sources", "rank"]].to_csv(os.path.join(downstream, "driver_genes.csv"), index=False)

            # also copy to out_dir root for convenience
            driver_master.to_csv(os.path.join(cfg.out_dir, "driver_genes_master.csv"), index=False)
            driver_master[["gene", "driver_score"]].to_csv(os.path.join(cfg.out_dir, "driver_genes.csv"), index=False)
            print(f"[Driver] Saved: {os.path.join(downstream, 'driver_genes_master.csv')}")
        else:
            driver_master = pd.DataFrame({"gene": sorted(list(union))})
            driver_master.to_csv(os.path.join(cfg.out_dir, "driver_genes.csv"), index=False)
    else:
        driver_master = pd.DataFrame()

    # optional: export CellChat inputs + expression summary from RAW full-gene AnnData
    raw_path = cfg.raw_full_h5ad
    if (cfg.export_cellchat_inputs or cfg.export_expr_summary) and raw_path:
        if not os.path.exists(raw_path):
            print(f"[WARN] raw_full_h5ad not found, skip cellchat/expr export: {raw_path}")
        else:
            print(f"[Raw] {raw_path}")
            raw = ad.read_h5ad(raw_path)

            if cfg.export_expr_summary and (not driver_master.empty):
                out_expr = os.path.join(downstream, "expr_driver_by_state_time_fullgenes.csv")
                summarize_driver_expr_raw(raw, cfg, driver_master, out_expr, top_n=min(300, len(driver_master)))
                print(f"[Expr] Saved: {out_expr}")

            if cfg.export_cellchat_inputs and (not driver_master.empty):
                out_cc = os.path.join(cfg.out_dir, "cellchat_input_all")
                genes_use = driver_master["gene"].astype(str).tolist()
                export_cellchat_inputs_from_raw(raw, cfg, genes_use=genes_use, out_dir=out_cc)
                print(f"[CellChat] Inputs saved under: {os.path.dirname(out_cc)}")

    # save run config (include inferred labels)
    run_meta = cfg.__dict__.copy()
    run_meta["pos_label_inferred"] = pos_label
    run_meta["neg_label_inferred"] = neg_label
    run_meta["gene_map_used"] = used_map

    # make json-serializable (tuples -> lists)
    for _k in ["time_order", "perturb_targets", "marker_master_targets", "folds"]:
        if _k in run_meta and isinstance(run_meta[_k], tuple):
            run_meta[_k] = list(run_meta[_k])

    with open(os.path.join(cfg.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[Done] Outputs at: {cfg.out_dir}")


if __name__ == "__main__":
    main()