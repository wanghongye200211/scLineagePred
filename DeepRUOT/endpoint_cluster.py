#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Endpoint highlight on a FIXED UMAP reducer (dim_reduction.pkl)
- DO NOT recompute UMAP
- DO NOT overwrite pkl
- Plot style aligned with plot.py (square, robust limits, black frame)
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


RUOT_CSV = "/Users/wanghongye/python/scLineagetracer/GSE175634/preprocess_final/ruot_input_pca30_forward.csv"
MAPPING_TSV = "/Users/wanghongye/python/scLineagetracer/GSE175634/preprocess_final/ruot_mapping_pca30_forward.tsv"

DIM_PKL = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/results/GSE175634/dim_reduction.pkl"

CLUSTER_COL = "state"
#TARGET_ENDPOINT_CLUSTERS = ["Alpha", "Beta", "Delta","Epsilon"]
TARGET_ENDPOINT_CLUSTERS = ["CF", "CM"]
#TARGET_ENDPOINT_CLUSTERS = ["sc_beta", "sc_ec", "sc_alpha"]
#TARGET_ENDPOINT_CLUSTERS = ["Failed", "Reprogrammed"]
#TARGET_ENDPOINT_CLUSTERS = ["Monocyte", "Neutrophil","Baso"]
#TARGET_ENDPOINT_CLUSTERS = ["Monocyte", "Neutrophil"]


# True: 只高亮最后一个 timepoint 的细胞
# False: 高亮所有 timepoint 中属于这些簇的细胞（能看到轨迹走向）
ONLY_SHOW_ENDPOINT = True

# 输出
OUT_DIR = os.path.join(os.path.dirname(DIM_PKL), "figures_csv")
OUT_PNG = os.path.join(OUT_DIR, "endpoint_highlight.png")
OUT_PDF = os.path.join(OUT_DIR, "endpoint_highlight.pdf")


# =========================
# FIGURE STYLE (match plot.py)
# =========================
FIG_INCH = 6.0
DPI = 300
Q = 0.01
PAD = 0.08

AX_PAD = 0.008  # 减少留白，和 plot.py 一致

BG_COLOR = "#CFCFCF"
BG_S = 2.0
BG_ALPHA = 0.10

HI_S = 18.0
HI_ALPHA = 0.95
HI_EDGE_LW = 0.35
HIGHLIGHT_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]


# =========================
# helpers
# =========================
def _finite_rows(A: np.ndarray) -> np.ndarray:
    return np.isfinite(A).all(axis=1)


def compute_square_limits(P2: np.ndarray, q: float = 0.01, pad: float = 0.08):
    """Square x/y limits with robust quantile clipping (same idea as plot.py)."""
    P2 = P2[_finite_rows(P2)]
    x0, x1 = np.quantile(P2[:, 0], [q, 1 - q])
    y0, y1 = np.quantile(P2[:, 1], [q, 1 - q])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = max(x1 - x0, y1 - y0) / 2.0
    r *= (1.0 + pad)
    return (cx - r, cx + r), (cy - r, cy + r)


def apply_square_style(ax, XLIM, YLIM):
    """Square view, fixed limits, thin black frame, no ticks (same style as plot.py)."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([AX_PAD, AX_PAD, 1 - 2 * AX_PAD, 1 - 2 * AX_PAD])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("black")


def load_umap_embedding(dim_pkl: str, X: np.ndarray) -> np.ndarray:
    """
    Load a FIXED reducer.
    - If embedding_ matches current X rows -> use it (most stable)
    - Else -> use transform(X)
    """
    reducer = joblib.load(dim_pkl)

    if hasattr(reducer, "embedding_") and reducer.embedding_ is not None:
        emb = np.asarray(reducer.embedding_)
        if emb.ndim == 2 and emb.shape == (X.shape[0], 2):
            return emb.astype(np.float32)

    # If not matching, require transform
    return reducer.transform(X).astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Read RUOT CSV (PCA dims)
    df = pd.read_csv(RUOT_CSV)
    if "samples" not in df.columns:
        raise RuntimeError("RUOT_CSV must contain column 'samples'.")

    x_cols = sorted([c for c in df.columns if c.startswith("x")], key=lambda c: int(c[1:]))
    if len(x_cols) == 0:
        raise RuntimeError("RUOT_CSV must contain x1..xD columns.")

    X = df[x_cols].to_numpy(dtype=np.float32)
    df["row_id"] = np.arange(df.shape[0])

    # 2) Read mapping and merge labels
    map_df = pd.read_csv(MAPPING_TSV, sep="\t")
    if CLUSTER_COL not in map_df.columns:
        raise RuntimeError(f"Mapping missing '{CLUSTER_COL}'. Available columns: {list(map_df.columns)}")

    merged = df.merge(map_df[["row_id", CLUSTER_COL]], on="row_id", how="left")

    final_sample = int(merged["samples"].max())
    is_endpoint = (merged["samples"].astype(int) == final_sample)

    print("[INFO] final_sample:", final_sample)
    for cname in TARGET_ENDPOINT_CLUSTERS:
        n_all = int((merged[CLUSTER_COL].astype(str) == str(cname)).sum())
        n_end = int((is_endpoint & (merged[CLUSTER_COL].astype(str) == str(cname))).sum())
        print(f"[INFO] {cname}: all={n_all}, endpoint={n_end}")

    # 3) Load fixed UMAP embedding
    X_emb = load_umap_embedding(DIM_PKL, X)
    ok = _finite_rows(X_emb)
    X_emb = X_emb[ok]
    merged = merged.loc[ok].reset_index(drop=True)
    is_endpoint = (merged["samples"].astype(int) == final_sample)

    # 4) Plot
    XLIM, YLIM = compute_square_limits(X_emb, q=Q, pad=PAD)
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))

    # background
    ax.scatter(
        X_emb[:, 0], X_emb[:, 1],
        s=BG_S, marker="x", c=BG_COLOR,
        linewidths=1.0, alpha=BG_ALPHA, zorder=0
    )

    # highlights
    for i, cname in enumerate(TARGET_ENDPOINT_CLUSTERS):
        if ONLY_SHOW_ENDPOINT:
            mask = is_endpoint & (merged[CLUSTER_COL].astype(str) == str(cname))
        else:
            mask = (merged[CLUSTER_COL].astype(str) == str(cname))

        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        ax.scatter(
            X_emb[mask, 0], X_emb[mask, 1],
            s=HI_S, marker="o",
            facecolors=color, edgecolors="black",
            linewidths=HI_EDGE_LW, alpha=HI_ALPHA,
            zorder=3, label=str(cname)
        )

    apply_square_style(ax, XLIM, YLIM)
    ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=1.2)

    fig.savefig(OUT_PNG, dpi=DPI)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print("[DONE] saved:", OUT_PNG)
    print("[DONE] saved:", OUT_PDF)




if __name__ == "__main__":
    main()
