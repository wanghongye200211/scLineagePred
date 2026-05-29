from __future__ import annotations

import os
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd

try:
    from .config import Config
except ImportError:
    from config import Config


def load_hvgs(cfg: Config) -> set:
    if not os.path.exists(cfg.hvg_h5ad):
        raise FileNotFoundError(f"[ERROR] hvg_h5ad not found: {cfg.hvg_h5ad}")
    adata = ad.read_h5ad(cfg.hvg_h5ad, backed="r")
    hvgs = set(adata.var_names.astype(str).tolist())
    try:
        adata.file.close()
    except Exception:
        pass
    return hvgs


def load_decoder(cfg: Config, latent_dim: int, hvgs: set):
    if not (os.path.exists(cfg.genes_txt) and os.path.exists(cfg.z_genes_npy)):
        raise FileNotFoundError(f"[ERROR] missing decoder files in {cfg.decoder_dir}: genes.txt or Z_genes.npy")

    with open(cfg.genes_txt, "r", encoding="utf-8") as handle:
        genes = [line.strip() for line in handle if line.strip()]
    Zg = np.load(cfg.z_genes_npy).astype(np.float32)

    if Zg.ndim != 2:
        raise ValueError(f"[ERROR] Z_genes must be 2D, got {Zg.shape}")
    if Zg.shape[1] != latent_dim:
        raise ValueError(f"[ERROR] Z_genes dim mismatch: {Zg.shape[1]} vs latent_dim={latent_dim}")
    if len(genes) != Zg.shape[0]:
        size = min(len(genes), Zg.shape[0])
        genes = genes[:size]
        Zg = Zg[:size, :]

    overlap = len(set(genes) & hvgs) / max(len(set(genes)), 1)
    if overlap < 0.80:
        raise ValueError(
            f"[ERROR] Decoder genes overlap with HVG is too low ({overlap:.3f}). "
            f"Likely wrong decoder_dir. Expected: {cfg.decoder_dir}"
        )
    if cfg.normalize_gene_vectors:
        Zg = Zg / (np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-12)
    return np.array(genes, dtype=object), Zg


def top_genes_by_dim(genes: np.ndarray, Zg: np.ndarray, dim: int, k: int) -> pd.DataFrame:
    weights = Zg[:, dim]
    idx = np.argsort(-np.abs(weights))[:k]
    return pd.DataFrame({"gene": genes[idx], "loading": weights[idx], "abs_loading": np.abs(weights[idx]), "dim": int(dim)})


def pick_dims_for_class(dim_sum: pd.DataFrame, top_dims: List[int], class_name: str, mode: str = "push_class") -> List[int]:
    if dim_sum.empty:
        return []
    by_dim = dim_sum.set_index("dim")
    dims = [int(dim) for dim in top_dims if int(dim) in by_dim.index]
    if not dims:
        return []

    column = f"best_delta_mean_prob_{class_name}"
    if column not in dim_sum.columns:
        raise KeyError(f"[ERROR] dim_summary missing column: {column}")

    mode = str(mode).strip().lower()
    if mode == "push_class":
        selected = []
        for dim in dims:
            push_class = str(by_dim.loc[dim].get("best_push_class", ""))
            push_delta = float(by_dim.loc[dim].get("best_push_delta", 0.0))
            if push_class == class_name and push_delta > 0:
                selected.append(int(dim))
        return selected
    if mode == "pos_only":
        return [int(dim) for dim in dims if float(by_dim.loc[dim].get(column, 0.0)) > 0.0]
    return dims


def integrate_marker_genes_for_class(genes: np.ndarray, Zg: np.ndarray, dim_sum: pd.DataFrame, top_dims: List[int], class_name: str):
    if dim_sum.empty:
        return pd.DataFrame()

    column = f"best_delta_mean_prob_{class_name}"
    if column not in dim_sum.columns:
        raise KeyError(f"[ERROR] dim_summary missing column: {column}")

    by_dim = dim_sum.set_index("dim")
    dims = [int(dim) for dim in top_dims if int(dim) in by_dim.index]
    if not dims:
        return pd.DataFrame()

    signed = np.array([float(by_dim.loc[dim][column]) for dim in dims], dtype=np.float32)
    score_abs = (np.abs(Zg[:, dims]) * np.abs(signed)[None, :]).sum(axis=1)
    score_signed = (Zg[:, dims] * signed[None, :]).sum(axis=1)

    return pd.DataFrame({"gene": genes, "score_abs": score_abs, "score_signed": score_signed}).sort_values(
        "score_abs",
        ascending=False,
    ).reset_index(drop=True)


def build_local_marker_rank(df_rank: pd.DataFrame, class_name: str, rrf_k: int, top_n: int) -> pd.DataFrame:
    columns = ["gene", "marker_score", "score_abs", "score_signed", "direction", "rank"]
    if df_rank is None or df_rank.empty or "gene" not in df_rank.columns:
        return pd.DataFrame(columns=columns)

    df_rank = df_rank.head(int(max(1, top_n))).copy().reset_index(drop=True)
    if "score_abs" not in df_rank.columns:
        df_rank["score_abs"] = 0.0
    if "score_signed" not in df_rank.columns:
        df_rank["score_signed"] = 0.0

    df_rank["rank"] = np.arange(1, len(df_rank) + 1, dtype=np.int64)
    df_rank["marker_score"] = 1.0 / (float(rrf_k) + df_rank["rank"].astype(float))
    df_rank["direction"] = np.where(
        df_rank["score_signed"] > 0,
        f"{class_name}_push",
        np.where(df_rank["score_signed"] < 0, f"{class_name}_suppress", "Neutral"),
    )
    return df_rank[columns]


def transition_genes(genes: np.ndarray, Zg: np.ndarray, deltaZ: np.ndarray, topk: int) -> pd.DataFrame:
    deltaZ = np.asarray(deltaZ, dtype=np.float64).reshape(-1)
    delta_norm = float(np.linalg.norm(deltaZ))
    if delta_norm < 1e-12:
        score = np.zeros((Zg.shape[0],), dtype=np.float64)
        proj = np.zeros_like(score)
        cos_signed = np.zeros_like(score)
        idx = np.argsort(-score)[:topk]
        return pd.DataFrame({"gene": genes[idx], "score": score[idx], "proj": proj[idx], "cos_signed": cos_signed[idx]})

    delta_hat = deltaZ / delta_norm
    Z = np.asarray(Zg, dtype=np.float64)
    dot = Z @ deltaZ
    dot_hat = Z @ delta_hat
    norms = np.linalg.norm(Z, axis=1) + 1e-12
    cos_signed = dot / (norms * delta_norm)
    score = np.abs(cos_signed)
    proj = np.abs(dot_hat)
    order = np.lexsort((-proj, -score))
    idx = order[:topk]
    return pd.DataFrame({"gene": genes[idx], "score": score[idx], "proj": proj[idx], "cos_signed": cos_signed[idx]})


def build_marker_master_rrf(cfg: Config, downstream_dir: str, decoder_tags: List[str], class_name: str) -> pd.DataFrame:
    score: Dict[str, float] = {}
    signed: Dict[str, float] = {}
    sources: Dict[str, set] = {}

    for tag in decoder_tags:
        path = os.path.join(downstream_dir, f"decoder_{tag}", f"marker_genes_ranked_{class_name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.head(int(cfg.top_union_marker)).copy()
        genes = df["gene"].astype(str).tolist()
        signed_values = df["score_signed"].to_numpy(dtype=np.float32) if "score_signed" in df.columns else None
        for rank, gene in enumerate(genes, start=1):
            weight = 1.0 / float(cfg.rrf_k + rank)
            score[gene] = score.get(gene, 0.0) + weight
            if signed_values is not None:
                signed[gene] = signed.get(gene, 0.0) + weight * float(np.sign(signed_values[rank - 1]))
            sources.setdefault(gene, set()).add(f"marker_{tag}")

    if not score:
        return pd.DataFrame(columns=["gene", "marker_score", "signed_score", "direction", "sources", "rank"])

    rows = []
    for gene, marker_score in score.items():
        signed_score = float(signed.get(gene, 0.0))
        if signed_score > 0:
            direction = f"{class_name}_push"
        elif signed_score < 0:
            direction = f"{class_name}_suppress"
        else:
            direction = "Neutral"
        rows.append(
            {
                "gene": gene,
                "marker_score": float(marker_score),
                "signed_score": signed_score,
                "direction": direction,
                "sources": "|".join(sorted(sources.get(gene, set()))),
            }
        )

    df_master = pd.DataFrame(rows).sort_values(["marker_score", "gene"], ascending=[False, True]).reset_index(drop=True)
    df_master["rank"] = np.arange(1, len(df_master) + 1)
    return df_master
