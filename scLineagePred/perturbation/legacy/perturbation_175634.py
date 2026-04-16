# -*- coding: utf-8 -*-
"""
/Users/wanghongye/python/scLineagetracer/Downstream_175634/Perturbation.py

GSE175634 - Perturbation -> Driver genes (direction-free)
=========================================================
- 只输出 driver genes（驱动基因），不输出 push 方向 direction
- 维度权重使用概率分布变化 probability shift（L1 distance），适配 binary / multi-class 的“方向无关”定义
- 默认包含最后一个 timepoint：Day15 (t_idx=6) -> 输出 decoder_day6（修复你说漏掉的 target idx=6）
- 可选抑制 decoder norm dominance：gene L2 norm 归一化

输出：
out_dir/
  perturbation/dayX/...
  downstream/decoder_day{t_idx}/marker_genes_ranked.csv
  downstream/marker_gene_candidates_union.csv
  downstream/driver_genes_master.csv
  run_config.json
"""

from __future__ import annotations
import os
import re
import json
import pickle
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Config =====================
@dataclass
class Config:
    # --- classification assets (匹配 class_175634.py) ---
    time_series_h5: str = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_sequences.h5"
    cls_out_dir: str = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF"
    model_dir: str = field(init=False)

    # class_175634.py 超参
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    nhead: int = 4

    # seeds（必须与 class_175634.py 一致）
    SEEDS: Dict[str, int] = field(default_factory=lambda: {
        "All_Days": 2026,
        "Obs_Day11": 2024,
        "Obs_Day7": 42,
        "Obs_Day5": 123,
        "Obs_Day3": 999,
        "Obs_Day1": 7,
    })

    # --- perturbation（扰动） ---
    folds: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)
    max_ood_rate_for_ranking: float = 0.10
    top_k_dims: int = 30

    # ✅ 默认包含最后一个 timepoint（修复你说漏掉的 idx=6 / Day15）
    exclude_last_timepoint: bool = False
    perturb_targets: Optional[Tuple[str, ...]] = None  # None=自动从 time_labels 推断 day0/day1/.../day15

    # clone 汇总
    min_clone_count: int = 1
    sample_n_clones: int = 0  # >0 抽样 clone 加速

    # --- decoder mapping（可选） ---
    decoder_dir: str = "/Users/wanghongye/python/scLineagetracer/autoencoder/results/GSE175634"
    genes_txt: str = field(init=False)
    z_genes_npy: str = field(init=False)
    hvgs_h5ad_for_check: str = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_with_latent.h5ad"
    min_decoder_hvg_overlap: float = 0.50

    top_k_genes_per_dim: int = 50
    top_union_marker: int = 800

    # ✅ gene norm 抑制“高范数基因统治”
    gene_norm_mode: str = "l2"   # "none" | "l2"

    # Driver master（RRF）
    rrf_k: int = 50
    save_driver_master: bool = True

    # --- outputs ---
    out_dir: str = "/Users/wanghongye/python/scLineagetracer/Downstream_175634"

    # --- device / batching ---
    device: str = "auto"   # auto / mps / cuda / cpu
    batch_size: int = 2048

    def __post_init__(self):
        self.model_dir = os.path.join(self.cls_out_dir, "saved_models")
        self.genes_txt = os.path.join(self.decoder_dir, "genes.txt")
        self.z_genes_npy = os.path.join(self.decoder_dir, "Z_genes.npy")
        if 0.0 not in self.folds or 1.0 not in self.folds:
            raise ValueError("folds must include 0.0 and 1.0")


# ===================== utils =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pick_device(cfg: Config) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def decode_bytes_arr(x) -> List[str]:
    out = []
    for v in x:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out

_DAY_RE = re.compile(r"[-+]?\d*\.?\d+")
def parse_day_number(s: str) -> float:
    s = str(s)
    m = _DAY_RE.findall(s)
    if not m:
        return 0.0
    try:
        return float(m[0])
    except Exception:
        return 0.0

def ood_rate_1d(ref_vec: np.ndarray, new_vec: np.ndarray) -> float:
    mn, mx = float(ref_vec.min()), float(ref_vec.max())
    return float(((new_vec < mn) | (new_vec > mx)).mean())


# ===================== setting mask（与 class_175634.py 一致） =====================
def setting_mask_indices(setting: str, T: int) -> Optional[List[int]]:
    if setting == "All_Days":
        return None
    if setting == "Obs_Day11":
        return [T - 1]
    if setting == "Obs_Day7":
        return [T - 2, T - 1]
    if setting == "Obs_Day5":
        return [T - 3, T - 2, T - 1]
    if setting == "Obs_Day3":
        return [T - 4, T - 3, T - 2, T - 1]
    if setting == "Obs_Day1":
        return list(range(2, T))
    raise ValueError(f"Unknown setting: {setting}")

def target_to_setting_by_index(t_idx: int, T: int) -> str:
    if T < 2:
        return "All_Days"
    if t_idx <= 1:
        return "Obs_Day1"
    if t_idx == 2:
        return "Obs_Day3"
    if t_idx == 3:
        return "Obs_Day5"
    if t_idx == 4:
        return "Obs_Day7"
    if t_idx == 5:
        return "Obs_Day11"
    return "All_Days"


# ===================== models（必须和 class_175634.py 一致） =====================
class LSTMModel(nn.Module):
    def __init__(self, d: int, h: int, l: int, dr: float):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True, bidirectional=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(0.5 * (h[-2] + h[-1]))

class RNNModel(nn.Module):
    def __init__(self, d: int, h: int, l: int, dr: float):
        super().__init__()
        self.rnn = nn.RNN(d, h, l, batch_first=True, dropout=(dr if l > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(x)[1][-1])

def build_transformer_encoder(enc_layer: nn.Module, num_layers: int) -> nn.TransformerEncoder:
    sig = inspect.signature(nn.TransformerEncoder)
    kwargs = {}
    if "enable_nested_tensor" in sig.parameters:
        kwargs["enable_nested_tensor"] = False
    if "mask_check" in sig.parameters:
        kwargs["mask_check"] = False
    return nn.TransformerEncoder(enc_layer, num_layers=num_layers, **kwargs)

class TransformerModel(nn.Module):
    def __init__(self, d: int, h: int, l: int, dr: float, nhead: int):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=h * 2, dropout=dr, batch_first=True
        )
        self.enc = build_transformer_encoder(enc_layer, num_layers=l)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x).mean(dim=1))


def load_ensemble(cfg: Config, input_dim: int, device: torch.device, setting: str):
    seed = cfg.SEEDS.get(setting, None)
    if seed is None:
        raise ValueError(f"Missing seed for setting={setting}")

    models = {
        "BiLSTM": LSTMModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
        "RNN": RNNModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device),
        "Trans": TransformerModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead).to(device),
    }
    for name in models:
        pth = os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth")
        if not os.path.exists(pth):
            raise FileNotFoundError(f"[ERROR] missing model: {pth}")
        state = torch.load(pth, map_location=device)
        models[name].load_state_dict(state, strict=True)
        models[name].eval()

    pkl = os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"[ERROR] missing stacking LR: {pkl}")
    with open(pkl, "rb") as f:
        lr = pickle.load(f)
    return models, lr


@torch.no_grad()
def predict_proba_stack(
    models: Dict[str, nn.Module],
    lr,
    X: np.ndarray,                       # [N,T,D]
    device: torch.device,
    batch_size: int,
    mask_indices: Optional[List[int]] = None,
    perturb: Optional[Tuple[int, int, float]] = None,  # (t_idx, dim, fold)
) -> np.ndarray:
    """
    返回 stacking 输出概率 proba: [N,2]
    """
    N = X.shape[0]
    probs = {k: [] for k in models.keys()}

    t_idx_p, dim_p, fold_p = (-1, -1, 1.0)
    if perturb is not None:
        t_idx_p, dim_p, fold_p = int(perturb[0]), int(perturb[1]), float(perturb[2])

    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        xb = X[st:ed].astype(np.float32, copy=True)

        if mask_indices is not None and len(mask_indices) > 0:
            xb[:, mask_indices, :] = 0.0

        if perturb is not None:
            xb[:, t_idx_p, dim_p] *= fold_p

        xt = torch.from_numpy(xb).to(device)

        for name, m in models.items():
            out = m(xt)
            probs[name].append(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy())

    feats = np.stack([np.concatenate(probs[k]) for k in ["BiLSTM", "RNN", "Trans"]], axis=1)
    return lr.predict_proba(feats).astype(np.float32)  # [N,2]


# ===================== data: clone prototypes from H5 =====================
def load_time_labels_from_h5(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        if "time_labels" in f:
            return decode_bytes_arr(f["time_labels"][...])
    return []

def build_clone_prototypes_from_h5(cfg: Config):
    if not os.path.exists(cfg.time_series_h5):
        raise FileNotFoundError(f"[ERROR] missing H5: {cfg.time_series_h5}")

    time_labels = load_time_labels_from_h5(cfg.time_series_h5)

    with h5py.File(cfg.time_series_h5, "r") as f:
        X_ds = f["X"]
        y = f["y"][...].astype(np.int64)
        if "seq_clone" not in f:
            raise KeyError("[ERROR] H5 missing key 'seq_clone'")
        seq_clone = f["seq_clone"][...].astype(np.int64)

        N, T, D = X_ds.shape
        print(f"[INFO] H5 X shape = {X_ds.shape} (N={N}, T={T}, D={D})")

        uniq = np.unique(seq_clone)
        uniq.sort()
        C = len(uniq)

        def remap(ids: np.ndarray) -> np.ndarray:
            j = np.searchsorted(uniq, ids)
            if not np.all(uniq[j] == ids):
                raise RuntimeError("[ERROR] clone id remap failed.")
            return j.astype(np.int64)

        sum_X = np.zeros((C, T, D), dtype=np.float64)
        cnt = np.zeros((C,), dtype=np.int64)

        chunk = 2048
        for st in range(0, N, chunk):
            ed = min(st + chunk, N)
            Xb = X_ds[st:ed].astype(np.float32)
            cb = remap(seq_clone[st:ed])
            np.add.at(sum_X, cb, Xb)
            np.add.at(cnt, cb, 1)

        keep = cnt >= int(cfg.min_clone_count)
        sum_X = sum_X[keep]
        cnt = cnt[keep]
        X_clone = (sum_X / cnt[:, None, None]).astype(np.float32)

    if not time_labels:
        time_labels = [str(i) for i in range(X_clone.shape[1])]

    # map day label -> index
    day_nums = [parse_day_number(s) for s in time_labels]
    day_to_index = {}
    for i, dn in enumerate(day_nums):
        k = f"day{int(dn) if float(dn).is_integer() else dn}".lower()
        if k not in day_to_index:
            day_to_index[k] = i

    return X_clone, time_labels, day_to_index


# ===================== decoder mapping (optional) =====================
def load_hvgs_for_overlap_check(h5ad_path: str) -> Optional[set]:
    if not h5ad_path or (not os.path.exists(h5ad_path)):
        return None
    try:
        import anndata as ad
        a = ad.read_h5ad(h5ad_path)
        return set(a.var_names.astype(str).tolist())
    except Exception:
        return None

def load_decoder(cfg: Config, latent_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (os.path.exists(cfg.genes_txt) and os.path.exists(cfg.z_genes_npy)):
        raise FileNotFoundError(f"[Decoder] Missing files: {cfg.genes_txt} or {cfg.z_genes_npy}")

    genes = [ln.strip() for ln in open(cfg.genes_txt, "r", encoding="utf-8") if ln.strip()]
    Zg = np.load(cfg.z_genes_npy).astype(np.float32)

    if Zg.ndim != 2:
        raise ValueError(f"[Decoder] Z_genes must be 2D, got {Zg.shape}")
    if Zg.shape[1] != latent_dim:
        raise ValueError(f"[Decoder] dim mismatch: Z_genes D={Zg.shape[1]} vs latent_dim={latent_dim}")
    if len(genes) != Zg.shape[0]:
        m = min(len(genes), Zg.shape[0])
        genes = genes[:m]
        Zg = Zg[:m, :]

    hvgs = load_hvgs_for_overlap_check(cfg.hvgs_h5ad_for_check)
    if hvgs is not None:
        overlap = len(set(genes) & hvgs) / max(len(set(genes)), 1)
        if overlap < float(cfg.min_decoder_hvg_overlap):
            raise ValueError(f"[Decoder] overlap too low ({overlap:.3f}) -> likely wrong decoder_dir")

    return np.array(genes, dtype=object), Zg


def integrate_driver_genes(genes: np.ndarray, Zg: np.ndarray, dim_sum: pd.DataFrame, top_dims: List[int], cfg: Config) -> pd.DataFrame:
    """
    direction-free driver gene scoring:
      weight_d = best_delta_l1 (>=0)
      score_raw(g) = sum_d weight_d * |loading(g,d)|
      score = score_raw / ||Zg[g,:]||_2  (optional)
    """
    s = dim_sum.set_index("dim")
    dims = [int(d) for d in top_dims if int(d) in s.index]
    if not dims:
        return pd.DataFrame()

    w = np.array([float(s.loc[d]["best_delta_l1"]) for d in dims], dtype=np.float32)  # >=0
    M = np.abs(Zg[:, dims])  # [G,K]
    score_raw = (M * w[None, :]).sum(axis=1)

    if str(cfg.gene_norm_mode).lower() == "l2":
        gene_norm = np.linalg.norm(Zg, axis=1) + 1e-12
        score = score_raw / gene_norm
    else:
        gene_norm = np.ones_like(score_raw, dtype=np.float32)
        score = score_raw

    df = pd.DataFrame({
        "gene": genes.astype(str),
        "score": score.astype(np.float32),
        "score_raw": score_raw.astype(np.float32),
        "gene_norm": gene_norm.astype(np.float32),
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def build_driver_master(decoder_subdirs: List[str], cfg: Config) -> pd.DataFrame:
    """
    Global driver genes via RRF (倒数排名融合).
    只用每个 decoder_day{t_idx}/marker_genes_ranked.csv 的排序（方向无关）
    """
    k = int(cfg.rrf_k)
    score = {}
    sources = {}

    for sub in decoder_subdirs:
        p = os.path.join(cfg.out_dir, "downstream", sub, "marker_genes_ranked.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if df.empty or ("gene" not in df.columns):
            continue
        genes_list = df["gene"].astype(str).head(cfg.top_union_marker).tolist()

        for r, g in enumerate(genes_list, start=1):
            w = 1.0 / float(k + r)
            score[g] = score.get(g, 0.0) + w
            sources.setdefault(g, set()).add(sub)

    rows = []
    for g, sc in score.items():
        rows.append({
            "gene": g,
            "driver_score": float(sc),
            "sources": "|".join(sorted(list(sources.get(g, set())))),
        })

    dfm = pd.DataFrame(rows).sort_values(["driver_score", "gene"], ascending=[False, True]).reset_index(drop=True)
    dfm["rank"] = np.arange(1, len(dfm) + 1)
    return dfm


# ===================== perturbation core =====================
def run_target(
    cfg: Config,
    target: str,
    t_idx: int,
    X: np.ndarray,            # [N,T,D] clone prototypes
    models: Dict[str, nn.Module],
    lr,
    device: torch.device,
    setting: str,
    mask_indices: Optional[List[int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出：
      df: 每个 (dim, fold) 的 dose response
      df_sum: 每个 dim 的 best fold summary（按 best_delta_l1 排）
    """
    N, T, D = X.shape

    P0 = predict_proba_stack(models, lr, X, device, cfg.batch_size, mask_indices=mask_indices, perturb=None)  # [N,2]
    y0 = np.argmax(P0, axis=1).astype(np.int64)

    ref_t = X[:, t_idx, :]  # reference for OOD

    rows = []
    for dim in range(D):
        ref_dim = ref_t[:, dim]
        for fold in cfg.folds:
            P = predict_proba_stack(
                models, lr, X, device, cfg.batch_size,
                mask_indices=mask_indices,
                perturb=(t_idx, dim, float(fold))
            )
            y = np.argmax(P, axis=1).astype(np.int64)

            delta_l1 = float(np.abs(P - P0).sum(axis=1).mean())   # direction-free magnitude
            flip_rate = float((y != y0).mean())

            new_dim = ref_dim * float(fold)
            ood = ood_rate_1d(ref_dim, new_dim)

            rows.append({
                "perturb_target": target,
                "t_idx": int(t_idx),
                "dim": int(dim),
                "fold": float(fold),
                "delta_l1": delta_l1,
                "flip_rate": flip_rate,
                "ood_rate": ood,
                "setting": setting,
            })

    df = pd.DataFrame(rows)

    # per-dim best fold summary
    sums = []
    for dim, g in df.groupby("dim", sort=False):
        g_in = g[g["ood_rate"] <= cfg.max_ood_rate_for_ranking]
        if len(g_in) == 0:
            g_in = g

        # 先 delta_l1 最大，再 flip_rate 最大
        best = g_in.sort_values(["delta_l1", "flip_rate"], ascending=[False, False]).iloc[0]

        sums.append({
            "perturb_target": target,
            "t_idx": int(t_idx),
            "dim": int(dim),
            "best_fold": float(best["fold"]),
            "best_delta_l1": float(best["delta_l1"]),
            "best_flip_rate": float(best["flip_rate"]),
            "best_ood_rate": float(best["ood_rate"]),
            "setting": setting,
        })

    df_sum = pd.DataFrame(sums).sort_values(
        ["best_delta_l1", "best_flip_rate"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df, df_sum


# ===================== main =====================
def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "perturbation"))
    ensure_dir(os.path.join(cfg.out_dir, "downstream"))

    device = pick_device(cfg)
    print(f"[INFO] device={device}")

    Xc, time_labels, day_to_index = build_clone_prototypes_from_h5(cfg)
    N, T, D = Xc.shape
    print(f"[INFO] clone prototypes: N={N}, T={T}, D={D}")
    print(f"[INFO] time_labels: {time_labels}")
    print(f"[INFO] day_to_index: {day_to_index}")

    if cfg.sample_n_clones and int(cfg.sample_n_clones) > 0 and int(cfg.sample_n_clones) < N:
        rng = np.random.default_rng(2026)
        idx = rng.choice(np.arange(N), size=int(cfg.sample_n_clones), replace=False)
        Xc = Xc[idx]
        N = Xc.shape[0]
        print(f"[INFO] sample_n_clones={cfg.sample_n_clones} -> N={N}")

    # targets
    if cfg.perturb_targets is None:
        nums = [parse_day_number(s) for s in time_labels]
        tgts = [f"day{int(v) if float(v).is_integer() else v}".lower() for v in nums]
        if cfg.exclude_last_timepoint and len(tgts) >= 2:
            tgts = tgts[:-1]
        cfg.perturb_targets = tuple(tgts)

    print(f"[INFO] perturb_targets={cfg.perturb_targets}")

    ens_cache = {}
    all_sum = []
    ran = []

    for tgt in cfg.perturb_targets:
        tgt = str(tgt).lower().strip()
        if tgt not in day_to_index:
            raise KeyError(f"[ERROR] target '{tgt}' not in day_to_index")

        t_idx = int(day_to_index[tgt])
        setting = target_to_setting_by_index(t_idx, T)
        mask_idx = setting_mask_indices(setting, T)

        if mask_idx is not None and t_idx in set(mask_idx):
            print(f"[Skip] target={tgt} masked under setting={setting}")
            continue

        if setting not in ens_cache:
            ens_cache[setting] = load_ensemble(cfg, input_dim=D, device=device, setting=setting)
        models, lr = ens_cache[setting]

        out_p = os.path.join(cfg.out_dir, "perturbation", tgt)
        ensure_dir(out_p)

        print(f"\n[Perturb] target={tgt} (t_idx={t_idx}) setting={setting}")
        df, df_sum = run_target(cfg, tgt, t_idx, Xc, models, lr, device, setting, mask_idx)

        df.to_csv(os.path.join(out_p, "dose_response_all_dims.csv"), index=False)
        df_sum.to_csv(os.path.join(out_p, "dim_summary.csv"), index=False)

        top_dims = df_sum["dim"].head(cfg.top_k_dims).astype(int).tolist()
        with open(os.path.join(out_p, "top_dims.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, top_dims)))

        all_sum.append(df_sum.copy())
        ran.append((tgt, t_idx, top_dims))

    if not all_sum:
        raise RuntimeError("[ERROR] no targets were run")

    pd.concat(all_sum, ignore_index=True).to_csv(
        os.path.join(cfg.out_dir, "perturbation", "dim_summary_all_targets.csv"),
        index=False
    )
    print(f"\n[Done] perturbation saved under: {os.path.join(cfg.out_dir, 'perturbation')}")

    # ===================== downstream (decoder) =====================
    downstream = os.path.join(cfg.out_dir, "downstream")
    ensure_dir(downstream)

    try:
        genes, Zg = load_decoder(cfg, latent_dim=D)
        has_decoder = True
        print(f"[Decoder] loaded: genes={len(genes)} Z_genes={Zg.shape}")
    except Exception as e:
        has_decoder = False
        print(f"[WARN] Decoder skipped: {e}")

    decoder_subdirs = []
    union = set()

    if has_decoder:
        for tgt, t_idx, top_dims in ran:
            dim_sum_path = os.path.join(cfg.out_dir, "perturbation", tgt, "dim_summary.csv")
            if not os.path.exists(dim_sum_path):
                continue
            df_sum = pd.read_csv(dim_sum_path)

            # ✅ 目录命名按 time index：decoder_day{t_idx}
            subname = f"decoder_day{t_idx}"
            sub = os.path.join(downstream, subname)
            ensure_dir(sub)
            decoder_subdirs.append(subname)

            # per-dim top genes（可选输出）
            per_dim = []
            for d in top_dims:
                w = Zg[:, int(d)]
                idx = np.argsort(-np.abs(w))[:cfg.top_k_genes_per_dim]
                per_dim.append(pd.DataFrame({
                    "gene": genes[idx].astype(str),
                    "loading": w[idx],
                    "abs_loading": np.abs(w[idx]),
                    "dim": int(d),
                }))
            if per_dim:
                pd.concat(per_dim, ignore_index=True).to_csv(os.path.join(sub, "top_genes_per_dim.csv"), index=False)

            # driver genes ranked（方向无关）
            df_rank = integrate_driver_genes(genes, Zg, df_sum, top_dims, cfg)
            df_rank.to_csv(os.path.join(sub, "marker_genes_ranked.csv"), index=False)

            union.update(df_rank["gene"].head(cfg.top_union_marker).astype(str).tolist())

        pd.DataFrame({"gene": sorted(list(union))}).to_csv(
            os.path.join(downstream, "marker_gene_candidates_union.csv"),
            index=False
        )

        if cfg.save_driver_master and decoder_subdirs:
            master = build_driver_master(decoder_subdirs, cfg)
            master.to_csv(os.path.join(downstream, "driver_genes_master.csv"), index=False)
            master[["gene", "driver_score", "rank", "sources"]].to_csv(os.path.join(downstream, "driver_genes.csv"), index=False)
            master.to_csv(os.path.join(cfg.out_dir, "driver_genes_master.csv"), index=False)
            master[["gene", "driver_score"]].to_csv(os.path.join(cfg.out_dir, "driver_genes.csv"), index=False)
            print(f"[Driver] saved: {os.path.join(downstream, 'driver_genes_master.csv')}")

    # run config
    meta = cfg.__dict__.copy()
    for k in ["folds", "perturb_targets"]:
        if k in meta and isinstance(meta[k], tuple):
            meta[k] = list(meta[k])
    with open(os.path.join(cfg.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] outputs at: {cfg.out_dir}")


if __name__ == "__main__":
    main()
