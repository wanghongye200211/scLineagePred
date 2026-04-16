# -*- coding: utf-8 -*-
"""
Unified latent-space perturbation scan for scLineagePred.

This entry point keeps the downstream logic from the original per-dataset
perturbation scripts, but removes hard-coded dataset paths. It now accepts
explicit sequence, classifier, decoder, and h5ad inputs and builds the same
two endpoint-focused scenarios generically from the final two timepoints:

1) observe up to the penultimate timepoint, perturb each kept step, predict the
   final endpoint
2) observe up to the final timepoint, perturb each kept step, predict the final
   endpoint
"""

from __future__ import annotations

import argparse
import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

import anndata as ad


# ===================== Config =====================
@dataclass
class Config:
    # ---------- dataset ----------
    time_series_h5: str = ""
    index_csv: str = ""

    # endpoint classes. Empty means "use every label found in data".
    target_labels: Tuple[str, ...] = ()
    csv_label_col: str = "label_str"

    # ---------- classifier ensemble ----------
    model_dir: str = ""
    base_seed: int = 2026

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    nhead: int = 4

    # ---------- perturbation ----------
    # fold scaling on one latent dimension at chosen timepoint
    folds: Tuple[float, ...] = (
        0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.25, 1.5,
        2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
    )

    max_ood_rate_for_ranking: float = 0.10
    top_k_dims: int = 30
    top_k_genes_per_dim: int = 50

    # if None: use all selected sequences
    max_sequences_scan: Optional[int] = 200000

    # ---------- decoder ----------
    decoder_dir: str = ""
    genes_txt: str = ""
    z_genes_npy: str = ""
    hvg_h5ad: str = ""

    # ---------- marker/driver rebuild behavior ----------
    # "push_class": keep dims where best_push_class == class and best_push_delta > 0
    # "pos_only"  : keep dims where best_delta_mean_prob_{class} > 0
    # "all"       : keep all top dims
    marker_dim_mode: str = "push_class"
    # If marker_dim_mode="push_class" yields empty dims, fall back to pos_only first.
    marker_fallback_pos_only: bool = True
    # Normalize decoder gene vectors to reduce "high-norm genes always top" effect.
    normalize_gene_vectors: bool = True

    # ---------- outputs ----------
    out_dir: str = "./outputs/perturbation"
    scenario_mode: str = "last_two"

    # ---------- driver master ----------
    rrf_k: int = 50
    top_union_marker: int = 800
    top_union_transition: int = 300
    save_driver_master: bool = True

    # ---------- runtime ----------
    device: str = "auto"
    batch_size: int = 2048

    def __post_init__(self):
        self.time_series_h5 = os.path.expanduser(self.time_series_h5)
        self.index_csv = os.path.expanduser(self.index_csv)
        self.model_dir = os.path.expanduser(self.model_dir)
        self.decoder_dir = os.path.expanduser(self.decoder_dir)
        self.hvg_h5ad = os.path.expanduser(self.hvg_h5ad)
        self.out_dir = os.path.expanduser(self.out_dir)
        if self.decoder_dir:
            if not self.genes_txt:
                self.genes_txt = os.path.join(self.decoder_dir, "genes.txt")
            if not self.z_genes_npy:
                self.z_genes_npy = os.path.join(self.decoder_dir, "Z_genes.npy")
        self.genes_txt = os.path.expanduser(self.genes_txt)
        self.z_genes_npy = os.path.expanduser(self.z_genes_npy)


# ===================== Utils =====================
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

    with h5py.File(h5_path, "r") as f:
        T = int(f["X"].shape[1])
    return [f"t{i}" for i in range(T)]


def infer_reverse_from_samples_order(df: pd.DataFrame, T: int) -> bool:
    """
    Infer whether sequence axis is reversed from 'samples_order' in index CSV.
    Example reverse string: "3,2,1,0" for T=4.
    """
    if "samples_order" not in df.columns:
        return False

    ser = df["samples_order"].dropna().astype(str)
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
        if not ok or len(vals) < 2:
            continue
        if T > 0 and len(vals) != T:
            pass
        return vals[0] > vals[-1]
    return False


def ood_rate_1d(ref: np.ndarray, new: np.ndarray) -> float:
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    mn, mx = float(r.min()), float(r.max())
    v = np.asarray(new, dtype=np.float32).reshape(-1)
    return float(((v < mn) | (v > mx)).mean())


# ===================== Models (must match class_132188.py) =====================
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
      - concat [BiLSTM_probs, RNN_probs, Trans_probs] -> (N, 3*C)
      - lr.predict_proba -> (N,C)

    IMPORTANT: feature order must match class_132188.py training:
      BiLSTM then RNN then Trans.
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
        [
            np.concatenate(probs_parts["BiLSTM"], axis=0),
            np.concatenate(probs_parts["RNN"], axis=0),
            np.concatenate(probs_parts["Trans"], axis=0),
        ],
        axis=1,
    )

    return lr.predict_proba(feats).astype(np.float32)


# ===================== Data loading =====================
def load_sequences(cfg: Config):
    if not os.path.isfile(cfg.index_csv):
        raise FileNotFoundError(f"[ERROR] missing index csv: {cfg.index_csv}")
    if not os.path.isfile(cfg.time_series_h5):
        raise FileNotFoundError(f"[ERROR] missing sequences h5: {cfg.time_series_h5}")

    df = pd.read_csv(cfg.index_csv)

    with h5py.File(cfg.time_series_h5, "r") as f:
        X_all = np.asarray(f["X"], dtype=np.float32)
        label_h5 = None
        if "label_str" in f:
            raw = f["label_str"][:]
            label_h5 = np.array(
                [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw],
                dtype=object,
            )

    T = X_all.shape[1]
    time_labels = read_time_labels_from_h5(cfg.time_series_h5)
    if len(time_labels) != T:
        time_labels = time_labels[:T] if len(time_labels) > T else (time_labels + [f"t{i}" for i in range(len(time_labels), T)])

    # Keep same reverse handling as class_132188.py
    reverse_time_axis = infer_reverse_from_samples_order(df, T)
    if reverse_time_axis:
        X_all = X_all[:, ::-1, :].copy()
        print("[INFO] Detected reverse samples_order; flipped X time axis to chronological order.")

    if cfg.csv_label_col in df.columns:
        labels_all = df[cfg.csv_label_col].astype(str).to_numpy()
    elif label_h5 is not None:
        labels_all = label_h5
    else:
        raise KeyError(f"[ERROR] label column missing in index csv ({cfg.csv_label_col}) and h5(label_str)")

    if len(labels_all) != len(X_all):
        raise ValueError(f"[ERROR] Label length mismatch: labels={len(labels_all)} vs X={len(X_all)}")

    if cfg.target_labels:
        class_names = list(cfg.target_labels)
        keep = np.isin(labels_all, np.array(class_names, dtype=object))
        if not keep.any():
            raise ValueError(f"[ERROR] No sequences match target_labels={class_names}")
        X = X_all[keep]
        labels = labels_all[keep]
    else:
        X = X_all
        labels = labels_all
        class_names = [str(x) for x in pd.unique(labels)]

    label_to_y = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_y[s] for s in labels], dtype=np.int64)

    # optional stratified subsample
    if cfg.max_sequences_scan is not None:
        idx_sub = stratified_subsample_indices(y, int(cfg.max_sequences_scan), seed=cfg.base_seed)
        X = X[idx_sub]
        y = y[idx_sub]
        labels = labels[idx_sub]

    print(f"[INFO] Loaded sequences: N={len(X)} T={X.shape[1]} D={X.shape[2]}")
    counts = {c: int((labels == c).sum()) for c in class_names}
    print(f"[INFO] Label counts: {counts}")
    print(f"[INFO] time_labels={time_labels}")

    return X, y.astype(np.int64), class_names, time_labels


# ===================== Strategy =====================
def build_endpoint_strategies(time_labels: List[str]):
    """
    Build two generic endpoint-focused strategies:
      A: UpTo_<T-2>, perturb [0..T-2]
      B: All_<T-1>, perturb [0..T-1]
    """
    T = len(time_labels)
    if T < 2:
        raise ValueError(f"Need at least 2 timepoints, got {T}")

    prev_idx = T - 2
    last_idx = T - 1

    prev_lab = str(time_labels[prev_idx])
    last_lab = str(time_labels[last_idx])

    s1 = {
        "scenario_id": "obs_to_prev_pred_last",
        "scenario_label": f"ObsTo_{prev_lab}_Pred_{last_lab}",
        "setting": f"UpTo_{prev_lab}",
        "keep_len": int(prev_idx + 1),
        "perturb_t_indices": list(range(prev_idx + 1)),
    }
    s2 = {
        "scenario_id": "obs_to_last_pred_last",
        "scenario_label": f"ObsTo_{last_lab}_Pred_{last_lab}",
        "setting": f"All_{last_lab}",
        "keep_len": int(last_idx + 1),
        "perturb_t_indices": list(range(last_idx + 1)),
    }
    return [s1, s2]


def build_strategies(cfg: Config, time_labels: List[str]):
    mode = str(cfg.scenario_mode).strip().lower()
    if mode == "last_two":
        return build_endpoint_strategies(time_labels)
    raise ValueError(f"Unsupported scenario_mode={cfg.scenario_mode!r}")


# ===================== Perturbation scan =====================
def scan_one_scenario(
    cfg: Config,
    X_all: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    time_labels: List[str],
    models: Dict[str, nn.Module],
    lr,
    device: str,
    scenario: Dict,
):
    """
    Fixed observation window per scenario:
      - setting fixed (e.g. UpTo_14.5 or All_15.5)
      - keep_len fixed
      - scan multiple perturb t indices within this fixed window
    """
    N, T, D = X_all.shape

    keep_len = int(scenario["keep_len"])
    if keep_len < 1 or keep_len > T:
        raise ValueError(f"Invalid keep_len={keep_len} for T={T}")

    target_t_list = [int(t) for t in scenario["perturb_t_indices"] if 0 <= int(t) < keep_len]
    if len(target_t_list) == 0:
        raise ValueError(f"Scenario has empty perturb targets after filtering: {scenario}")

    X_base = np.asarray(X_all[:, :keep_len, :], dtype=np.float32)

    # baseline
    p0 = predict_proba_stack(models, lr, X_base, device=device, batch_size=cfg.batch_size)
    y0 = np.argmax(p0, axis=1).astype(np.int64)

    base_mean = p0.mean(axis=0)
    base_frac = np.array([(y0 == k).mean() for k in range(len(class_names))], dtype=np.float32)

    rows = []
    Xw = X_base.copy()

    for target_t in target_t_list:
        target_label = str(time_labels[target_t])

        for dim in range(D):
            ref = X_base[:, target_t, dim].copy()

            for fold in cfg.folds:
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
                    "scenario_id": str(scenario["scenario_id"]),
                    "scenario_label": str(scenario["scenario_label"]),
                    "setting": str(scenario["setting"]),
                    "keep_len": int(keep_len),
                    "target_t": int(target_t),
                    "target_label": str(target_label),
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

            # restore this dim
            Xw[:, target_t, dim] = ref

    df = pd.DataFrame(rows)

    # summarize best fold per (target_t, dim)
    sums = []
    for (target_t, dim), g in df.groupby(["target_t", "dim"], sort=True):
        g_in = g[g["ood_rate"] <= cfg.max_ood_rate_for_ranking]
        if len(g_in) == 0:
            g_in = g
        best = g_in.sort_values(["flip_rate", "delta_l1", "delta_maxabs"], ascending=[False, False, False]).iloc[0]

        rec = {
            "scenario_id": str(scenario["scenario_id"]),
            "scenario_label": str(scenario["scenario_label"]),
            "setting": str(scenario["setting"]),
            "keep_len": int(keep_len),
            "target_t": int(target_t),
            "target_label": str(best["target_label"]),
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
        ["target_t", "best_flip_rate", "best_delta_l1", "best_delta_maxabs"],
        ascending=[True, False, False, False],
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
    if bool(cfg.normalize_gene_vectors):
        n = np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-12
        Zg = Zg / n
    return np.array(genes, dtype=object), Zg


def top_genes_by_dim(genes: np.ndarray, Zg: np.ndarray, dim: int, k: int) -> pd.DataFrame:
    w = Zg[:, dim]
    idx = np.argsort(-np.abs(w))[:k]
    return pd.DataFrame({"gene": genes[idx], "loading": w[idx], "abs_loading": np.abs(w[idx]), "dim": int(dim)})


def pick_dims_for_class(
    dim_sum: pd.DataFrame,
    top_dims: List[int],
    class_name: str,
    mode: str = "push_class",
) -> List[int]:
    """
    Pick class-specific latent dims from dim_summary.
    """
    if dim_sum.empty:
        return []
    s = dim_sum.set_index("dim")
    dims = [int(d) for d in top_dims if int(d) in s.index]
    if not dims:
        return []

    mode = str(mode).strip().lower()
    col = f"best_delta_mean_prob_{class_name}"
    if col not in dim_sum.columns:
        raise KeyError(f"[ERROR] dim_summary missing column: {col}")

    if mode == "push_class":
        out = []
        for d in dims:
            push = str(s.loc[d].get("best_push_class", ""))
            pdlt = float(s.loc[d].get("best_push_delta", 0.0))
            if push == class_name and pdlt > 0:
                out.append(int(d))
        return out
    if mode == "pos_only":
        return [int(d) for d in dims if float(s.loc[d].get(col, 0.0)) > 0.0]
    return dims


def integrate_marker_genes_for_class(
    genes: np.ndarray,
    Zg: np.ndarray,
    dim_sum: pd.DataFrame,
    top_dims: List[int],
    class_name: str,
):
    """
    Class-specific marker integration:
      weights per dim = best_delta_mean_prob_{class_name}
      score_abs    = sum_d |Zg[g,d]| * |w_d|
      score_signed = sum_d  Zg[g,d]  *  w_d
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

    M = Zg[:, dims]
    score_abs = (np.abs(M) * w_abs[None, :]).sum(axis=1)
    score_signed = (M * w_signed[None, :]).sum(axis=1)

    df = pd.DataFrame({"gene": genes, "score_abs": score_abs, "score_signed": score_signed})
    return df.sort_values("score_abs", ascending=False).reset_index(drop=True)


def build_local_driver_from_marker(
    df_rank: pd.DataFrame,
    class_name: str,
    rrf_k: int,
    top_n: int,
) -> pd.DataFrame:
    """
    Build per-timepoint local driver table from one marker ranking.
    """
    cols = ["gene", "driver_score", "score_abs", "score_signed", "direction", "rank"]
    if df_rank is None or df_rank.empty:
        return pd.DataFrame(columns=cols)

    d = df_rank.copy()
    if "gene" not in d.columns:
        return pd.DataFrame(columns=cols)
    if "score_abs" not in d.columns:
        d["score_abs"] = 0.0
    if "score_signed" not in d.columns:
        d["score_signed"] = 0.0

    d = d.head(int(max(1, top_n))).copy().reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1, dtype=np.int64)
    d["driver_score"] = 1.0 / (float(rrf_k) + d["rank"].astype(float))
    d["direction"] = np.where(
        d["score_signed"] > 0,
        f"{class_name}_push",
        np.where(d["score_signed"] < 0, f"{class_name}_suppress", "Neutral"),
    )
    return d[cols]


def transition_genes(genes: np.ndarray, Zg: np.ndarray, deltaZ: np.ndarray, topk: int) -> pd.DataFrame:
    dz = np.asarray(deltaZ, dtype=np.float64).reshape(-1)
    dz_norm = float(np.linalg.norm(dz))
    if dz_norm < 1e-12:
        s = np.zeros((Zg.shape[0],), dtype=np.float64)
        proj = np.zeros_like(s)
        cos_signed = np.zeros_like(s)
        idx = np.argsort(-s)[:topk]
        return pd.DataFrame({"gene": genes[idx], "score": s[idx], "proj": proj[idx], "cos_signed": cos_signed[idx]})

    dz_hat = dz / dz_norm
    Z = np.asarray(Zg, dtype=np.float64)
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
    decoder_tags: List[str],
    class_name: str,
) -> pd.DataFrame:
    """
    Build driver genes master for one class via RRF over per-target marker lists:
      downstream/decoder_<tag>/marker_genes_ranked_<class>.csv
    """
    k = int(cfg.rrf_k)
    score: Dict[str, float] = {}
    signed: Dict[str, float] = {}
    sources: Dict[str, set] = {}

    for tag in decoder_tags:
        p = os.path.join(downstream_dir, f"decoder_{tag}", f"marker_genes_ranked_{class_name}.csv")
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
            sources.setdefault(g, set()).add(f"marker_{tag}")

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


# ===================== Main =====================
def main(cfg: Optional[Config] = None):
    cfg = cfg or Config()
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "perturbation"))
    ensure_dir(os.path.join(cfg.out_dir, "downstream"))

    device = pick_device(cfg)
    print(f"[INFO] Device: {device}")

    X, y, class_names, time_labels = load_sequences(cfg)
    N, T, D = X.shape

    strategies = build_strategies(cfg, time_labels)
    print("[INFO] Active endpoint strategies:")
    for s in strategies:
        print(
            f"  - {s['scenario_id']}: setting={s['setting']} keep_len={s['keep_len']} "
            f"targets={s['perturb_t_indices']}"
        )

    # cache models by setting
    ens_cache: Dict[str, Tuple[Dict[str, nn.Module], object]] = {}

    all_sum_global = []
    decoder_tags = []

    for s in strategies:
        setting = str(s["setting"])
        if setting not in ens_cache:
            ens_cache[setting] = load_ensemble(
                cfg,
                input_dim=D,
                n_classes=len(class_names),
                device=device,
                setting=setting,
            )
        models, lr = ens_cache[setting]

        scenario_tag = sanitize_name(s["scenario_label"])
        out_s = os.path.join(cfg.out_dir, "perturbation", scenario_tag)
        ensure_dir(out_s)

        print(
            f"\n[Perturb] scenario={s['scenario_id']} "
            f"setting={setting} keep_len={s['keep_len']} target_t={s['perturb_t_indices']}"
        )

        df, df_sum = scan_one_scenario(
            cfg=cfg,
            X_all=X,
            y_true=y,
            class_names=class_names,
            time_labels=time_labels,
            models=models,
            lr=lr,
            device=device,
            scenario=s,
        )

        # save scenario-level combined tables
        df.to_csv(os.path.join(out_s, "dose_response_all_dims.csv"), index=False)
        df_sum.to_csv(os.path.join(out_s, "dim_summary.csv"), index=False)

        # save per-target tables (114412-like granularity)
        for t_idx in sorted(df_sum["target_t"].unique().tolist()):
            t_idx = int(t_idx)
            t_lab = str(time_labels[t_idx])
            target_tag = f"t{t_idx}_{sanitize_name(t_lab)}"
            out_t = os.path.join(out_s, target_tag)
            ensure_dir(out_t)

            df_t = df[df["target_t"] == t_idx].copy()
            sum_t = df_sum[df_sum["target_t"] == t_idx].copy()

            df_t.to_csv(os.path.join(out_t, "dose_response_all_dims.csv"), index=False)
            sum_t.to_csv(os.path.join(out_t, "dim_summary.csv"), index=False)

            top_dims = sum_t["dim"].head(int(cfg.top_k_dims)).astype(int).tolist()
            with open(os.path.join(out_t, "top_dims.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, top_dims)))

            tag_full = f"{scenario_tag}__{target_tag}"
            decoder_tags.append(tag_full)
            all_sum_global.append(sum_t.assign(scenario_tag=scenario_tag, target_tag=target_tag, decoder_tag=tag_full))

    if all_sum_global:
        pd.concat(all_sum_global, ignore_index=True).to_csv(
            os.path.join(cfg.out_dir, "perturbation", "dim_summary_all_targets.csv"),
            index=False,
        )

    # ---- decoder mapping ----
    hvgs = load_hvgs(cfg)
    genes, Zg = load_decoder(cfg, latent_dim=D, hvgs=hvgs)

    downstream = os.path.join(cfg.out_dir, "downstream")
    ensure_dir(downstream)

    # transition genes from sequence means
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

    # per-target decoder integration
    for s in strategies:
        scenario_tag = sanitize_name(s["scenario_label"])
        out_s = os.path.join(cfg.out_dir, "perturbation", scenario_tag)

        target_dirs = []
        for t_idx in s["perturb_t_indices"]:
            t_idx = int(t_idx)
            t_lab = str(time_labels[t_idx])
            target_tag = f"t{t_idx}_{sanitize_name(t_lab)}"
            target_dirs.append((target_tag, os.path.join(out_s, target_tag)))

        for target_tag, target_dir in target_dirs:
            dim_sum_path = os.path.join(target_dir, "dim_summary.csv")
            if not os.path.exists(dim_sum_path):
                continue

            df_sum = pd.read_csv(dim_sum_path)
            top_dims = df_sum["dim"].head(int(cfg.top_k_dims)).astype(int).tolist()
            dim_all_set = set(int(x) for x in df_sum["dim"].astype(int).tolist())

            dec_tag = f"{scenario_tag}__{target_tag}"
            sub = os.path.join(downstream, f"decoder_{dec_tag}")
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
                dims_cls = pick_dims_for_class(
                    dim_sum=df_sum,
                    top_dims=top_dims,
                    class_name=cname,
                    mode=cfg.marker_dim_mode,
                )
                if (not dims_cls) and bool(cfg.marker_fallback_pos_only) and str(cfg.marker_dim_mode).lower() == "push_class":
                    dims_cls = pick_dims_for_class(
                        dim_sum=df_sum,
                        top_dims=top_dims,
                        class_name=cname,
                        mode="pos_only",
                    )
                if not dims_cls:
                    dims_cls = [int(d) for d in top_dims if int(d) in dim_all_set]

                df_rank = integrate_marker_genes_for_class(
                    genes,
                    Zg,
                    df_sum,
                    dims_cls,
                    class_name=cname,
                )
                out_rank = os.path.join(sub, f"marker_genes_ranked_{cname}.csv")
                df_rank.to_csv(out_rank, index=False)

                # local driver list for this decoder target (timepoint-level)
                df_local_driver = build_local_driver_from_marker(
                    df_rank=df_rank,
                    class_name=cname,
                    rrf_k=int(cfg.rrf_k),
                    top_n=int(cfg.top_union_marker),
                )
                out_local_driver = os.path.join(sub, f"driver_genes_ranked_{cname}.csv")
                df_local_driver.to_csv(out_local_driver, index=False)

                union_genes.update(df_rank["gene"].astype(str).head(int(cfg.top_union_marker)).tolist())

    # save union
    pd.DataFrame({"gene": sorted(list(union_genes))}).to_csv(
        os.path.join(downstream, "marker_gene_candidates_union.csv"),
        index=False,
    )

    # build driver masters per class
    if cfg.save_driver_master:
        for cname in class_names:
            dm = build_driver_master_rrf(cfg, downstream_dir=downstream, decoder_tags=decoder_tags, class_name=cname)
            dm.to_csv(os.path.join(downstream, f"driver_genes_master_{cname}.csv"), index=False)
            dm[["gene", "driver_score", "direction", "sources", "rank"]].to_csv(
                os.path.join(downstream, f"driver_genes_{cname}.csv"),
                index=False,
            )

            # convenience copy
            dm.to_csv(os.path.join(cfg.out_dir, f"driver_genes_master_{cname}.csv"), index=False)
            dm[["gene", "driver_score"]].to_csv(
                os.path.join(cfg.out_dir, f"driver_genes_{cname}.csv"),
                index=False,
            )

        # union top300 from each class
        union_driver = set()
        for cname in class_names:
            p = os.path.join(downstream, f"driver_genes_master_{cname}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                union_driver.update(df["gene"].astype(str).head(300).tolist())
        pd.DataFrame({"gene": sorted(list(union_driver))}).to_csv(
            os.path.join(downstream, "driver_genes_union_top300_eachclass.csv"),
            index=False,
        )

    # save run config
    run_meta = cfg.__dict__.copy()
    for key, value in list(run_meta.items()):
        if isinstance(value, tuple):
            run_meta[key] = list(value)
    run_meta["time_labels"] = list(time_labels)
    run_meta["active_strategies"] = strategies

    with open(os.path.join(cfg.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n[Done] Outputs at: {cfg.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified perturbation scan for scLineagePred")
    parser.add_argument("--time-series-h5", required=True, help="Sequence H5 produced by trajectory reconstruction")
    parser.add_argument("--index-csv", required=True, help="Index CSV paired with the sequence H5")
    parser.add_argument("--model-dir", required=True, help="Directory containing classification ensemble weights")
    parser.add_argument("--decoder-dir", required=True, help="Directory containing decoder outputs such as genes.txt and Z_genes.npy")
    parser.add_argument("--hvg-h5ad", required=True, help="H5AD file used for HVG overlap validation")
    parser.add_argument("--out-dir", required=True, help="Output directory for perturbation results")
    parser.add_argument("--target-label", action="append", dest="target_labels", default=None, help="Endpoint label to keep; repeat for multiple labels")
    parser.add_argument("--csv-label-col", default="label_str", help="Label column in index CSV")
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--folds", nargs="+", type=float, help="Perturbation scale factors")
    parser.add_argument("--max-ood-rate-for-ranking", type=float, default=0.10)
    parser.add_argument("--top-k-dims", type=int, default=30)
    parser.add_argument("--top-k-genes-per-dim", type=int, default=50)
    parser.add_argument("--max-sequences-scan", type=int, default=200000)
    parser.add_argument("--marker-dim-mode", choices=["push_class", "pos_only", "all"], default="push_class")
    parser.add_argument("--scenario-mode", choices=["last_two"], default="last_two")
    parser.add_argument("--rrf-k", type=int, default=50)
    parser.add_argument("--top-union-marker", type=int, default=800)
    parser.add_argument("--top-union-transition", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--genes-txt", help="Optional explicit genes.txt path")
    parser.add_argument("--z-genes-npy", help="Optional explicit Z_genes.npy path")
    parser.add_argument("--no-marker-fallback-pos-only", action="store_true", help="Disable pos_only fallback when push_class yields no dims")
    parser.add_argument("--no-normalize-gene-vectors", action="store_true", help="Disable decoder gene vector normalization")
    parser.add_argument("--no-save-driver-master", action="store_true", help="Skip aggregated driver gene exports")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    kwargs = {
        "time_series_h5": args.time_series_h5,
        "index_csv": args.index_csv,
        "csv_label_col": args.csv_label_col,
        "model_dir": args.model_dir,
        "base_seed": args.base_seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "nhead": args.nhead,
        "max_ood_rate_for_ranking": args.max_ood_rate_for_ranking,
        "top_k_dims": args.top_k_dims,
        "top_k_genes_per_dim": args.top_k_genes_per_dim,
        "max_sequences_scan": args.max_sequences_scan,
        "decoder_dir": args.decoder_dir,
        "genes_txt": args.genes_txt or "",
        "z_genes_npy": args.z_genes_npy or "",
        "hvg_h5ad": args.hvg_h5ad,
        "marker_dim_mode": args.marker_dim_mode,
        "marker_fallback_pos_only": not args.no_marker_fallback_pos_only,
        "normalize_gene_vectors": not args.no_normalize_gene_vectors,
        "out_dir": args.out_dir,
        "scenario_mode": args.scenario_mode,
        "rrf_k": args.rrf_k,
        "top_union_marker": args.top_union_marker,
        "top_union_transition": args.top_union_transition,
        "save_driver_master": not args.no_save_driver_master,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if args.target_labels:
        kwargs["target_labels"] = tuple(args.target_labels)
    if args.folds:
        kwargs["folds"] = tuple(args.folds)
    return Config(**kwargs)


if __name__ == "__main__":
    main(config_from_args(parse_args()))
