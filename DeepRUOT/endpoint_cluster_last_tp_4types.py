#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Endpoint-focused plot for GSE132188 (4 endocrine cell types), style-aligned
with the original endpoint_cluster.py.

Outputs two variants:
- full  : with legend
- clean : no legend
"""

import argparse
import os
import sys
import types
from typing import Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_RUOT_CSV = "/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/ruot_input_pca50_forward.csv"
DEFAULT_MAPPING_TSV = "/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/ruot_mapping_pca50_forward.tsv"
DEFAULT_OUT_DIR = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/results/GSE132188/figures_csv"

# Prefer current workspace, fallback to old project path if needed.
DEFAULT_DIM_PKL_CANDIDATES = [
    "/Users/wanghongye/python/scLineagetracer/DeepRUOT/results/GSE132188/dim_reduction.pkl",
    "/Users/wanghongye/PycharmProjects/scLineagetracer/DeepRUOT/results/GSE132188/dim_reduction.pkl",
]

DEFAULT_TARGET_CLASSES = ("Alpha", "Beta", "Delta", "Epsilon")


# =========================
# Plot style (match original endpoint_cluster.py)
# =========================
FIG_INCH = 6.0
DPI = 300
Q = 0.01
PAD = 0.08
AX_PAD = 0.008

SHOW_BACKGROUND = True
BG_COLOR = "#CFCFCF"
BG_S = 2.0
BG_ALPHA = 0.10

HI_S = 18.0
HI_ALPHA = 0.95
HI_EDGE_LW = 0.35
HIGHLIGHT_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]
CLASS_COLOR_MAP_SCATTER = {
    "Alpha": "#1f77b4",
    "Beta": "#ff7f0e",
    "Delta": "#2ca02c",
    "Epsilon": "#d62728",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruot-csv", default=DEFAULT_RUOT_CSV)
    parser.add_argument("--mapping-tsv", default=DEFAULT_MAPPING_TSV)
    parser.add_argument("--dim-pkl", default="", help="Path to fixed UMAP reducer pkl. If empty, auto-search candidates.")
    parser.add_argument("--cluster-col", default="state")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Target classes to highlight at the final timepoint.",
    )
    parser.add_argument("--focus-endpoint-only", type=int, default=1, help="1: only plot cells at final timepoint.")
    parser.add_argument("--show-background", type=int, default=1, help="1: show non-target endpoint cells as gray background.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--allow-pca-fallback",
        type=int,
        default=1,
        help="1: if dim_reduction.pkl fails to load, fallback to PCA-2D for plotting.",
    )
    return parser.parse_args()


def _finite_rows(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=1)


def _pick_existing_path(candidates: Iterable[str]) -> str:
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return ""


def _load_umap_embedding(dim_pkl: str, x: np.ndarray) -> np.ndarray:
    try:
        reducer = joblib.load(dim_pkl)
    except ModuleNotFoundError as e:
        # Some environments do not have umap-learn installed.
        # Install a minimal stub so we can still unpickle and read reducer.embedding_.
        if "umap" not in str(e):
            raise
        if "umap" not in sys.modules:
            umap_mod = types.ModuleType("umap")
            umap_umap_mod = types.ModuleType("umap.umap_")

            class _UMAPStub:
                def __setstate__(self, state):
                    self.__dict__.update(state)

                def __getstate__(self):
                    return self.__dict__

            umap_umap_mod.UMAP = _UMAPStub
            umap_mod.umap_ = umap_umap_mod
            sys.modules["umap"] = umap_mod
            sys.modules["umap.umap_"] = umap_umap_mod
        reducer = joblib.load(dim_pkl)

    if hasattr(reducer, "embedding_") and reducer.embedding_ is not None:
        emb = np.asarray(reducer.embedding_)
        if emb.ndim == 2 and emb.shape == (x.shape[0], 2):
            return emb.astype(np.float32)

    if not hasattr(reducer, "transform"):
        raise RuntimeError(
            "dim_reduction.pkl has no matching embedding_ and no transform() method. "
            "Please install umap-learn or regenerate a compatible dim_reduction.pkl."
        )
    return reducer.transform(x).astype(np.float32)


def _compute_square_limits(p2: np.ndarray, q: float = Q, pad: float = PAD) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    p2 = p2[_finite_rows(p2)]
    x0, x1 = np.quantile(p2[:, 0], [q, 1 - q])
    y0, y1 = np.quantile(p2[:, 1], [q, 1 - q])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = max(x1 - x0, y1 - y0) / 2.0
    r *= (1.0 + pad)
    return (cx - r, cx + r), (cy - r, cy + r)


def _apply_square_style(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([AX_PAD, AX_PAD, 1 - 2 * AX_PAD, 1 - 2 * AX_PAD])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("black")


def _render_variant(
    x_plot: np.ndarray,
    m_plot: pd.DataFrame,
    is_target_plot: np.ndarray,
    cluster_col: str,
    target_classes,
    xlim,
    ylim,
    show_background: bool,
    show_legend: bool,
    out_png: str,
    out_pdf: str,
):
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))

    if show_background:
        bg_mask = ~is_target_plot
        if int(bg_mask.sum()) > 0:
            ax.scatter(
                x_plot[bg_mask, 0],
                x_plot[bg_mask, 1],
                s=BG_S,
                marker="x",
                c=BG_COLOR,
                linewidths=1.0,
                alpha=BG_ALPHA,
                zorder=1,
                label="_nolegend_",
            )

    for i, cname in enumerate(target_classes):
        cls_mask = m_plot[cluster_col].to_numpy() == cname
        if int(cls_mask.sum()) <= 0:
            continue
        ax.scatter(
            x_plot[cls_mask, 0],
            x_plot[cls_mask, 1],
            s=HI_S,
            marker="o",
            facecolors=CLASS_COLOR_MAP_SCATTER.get(cname, HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]),
            edgecolors="black",
            linewidths=HI_EDGE_LW,
            alpha=HI_ALPHA,
            zorder=3,
            label=str(cname),
        )

    _apply_square_style(ax, xlim, ylim)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=1.2)

    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    args = parse_args()

    dim_pkl = args.dim_pkl.strip()
    if not dim_pkl:
        dim_pkl = _pick_existing_path(DEFAULT_DIM_PKL_CANDIDATES)
    if not dim_pkl:
        raise FileNotFoundError(
            "No dim_reduction.pkl found. Please pass --dim-pkl explicitly."
        )

    if not os.path.isfile(args.ruot_csv):
        raise FileNotFoundError(f"RUOT CSV not found: {args.ruot_csv}")
    if not os.path.isfile(args.mapping_tsv):
        raise FileNotFoundError(f"Mapping TSV not found: {args.mapping_tsv}")

    out_dir = args.out_dir.strip()
    os.makedirs(out_dir, exist_ok=True)

    out_png_full = os.path.join(out_dir, "endpoint_highlight_last_tp_4types_full.png")
    out_pdf_full = os.path.join(out_dir, "endpoint_highlight_last_tp_4types_full.pdf")
    out_png_clean = os.path.join(out_dir, "endpoint_highlight_last_tp_4types_clean.png")
    out_pdf_clean = os.path.join(out_dir, "endpoint_highlight_last_tp_4types_clean.pdf")

    target_classes = [str(x) for x in args.classes]
    cluster_col = str(args.cluster_col)
    focus_endpoint_only = bool(args.focus_endpoint_only)
    show_background = bool(int(args.show_background))

    # 1) Load RUOT inputs
    df = pd.read_csv(args.ruot_csv)
    if "samples" not in df.columns:
        raise RuntimeError("RUOT CSV must contain column 'samples'.")
    x_cols = sorted([c for c in df.columns if c.startswith("x")], key=lambda c: int(c[1:]))
    if len(x_cols) == 0:
        raise RuntimeError("RUOT CSV must contain x1..xD columns.")
    x = df[x_cols].to_numpy(dtype=np.float32)
    df["row_id"] = np.arange(df.shape[0], dtype=np.int64)

    map_df = pd.read_csv(args.mapping_tsv, sep="\t")
    if cluster_col not in map_df.columns:
        raise RuntimeError(f"Mapping missing '{cluster_col}'. Available: {list(map_df.columns)}")

    merged = df.merge(map_df[["row_id", cluster_col]], on="row_id", how="left")
    merged[cluster_col] = merged[cluster_col].astype(str)

    # 2) Load fixed embedding
    try:
        x_emb = _load_umap_embedding(dim_pkl, x)
    except Exception as e:
        if not bool(int(args.allow_pca_fallback)):
            raise
        print(f"[WARN] Failed to load fixed reducer ({dim_pkl}): {e}")
        print("[WARN] Falling back to PCA-2D for plotting.")
        x_emb = PCA(n_components=2, random_state=0).fit_transform(x).astype(np.float32)
    ok = _finite_rows(x_emb)
    x_emb = x_emb[ok]
    merged = merged.loc[ok].reset_index(drop=True)

    # 3) Endpoint masks
    final_sample = float(pd.to_numeric(merged["samples"], errors="coerce").max())
    endpoint_mask = pd.to_numeric(merged["samples"], errors="coerce").to_numpy(dtype=np.float64) == final_sample
    target_mask = endpoint_mask & merged[cluster_col].isin(target_classes).to_numpy()

    print(f"[INFO] dim_pkl: {dim_pkl}")
    print(f"[INFO] final_sample: {final_sample:g}")
    print(f"[INFO] focus_endpoint_only: {focus_endpoint_only}")
    for cname in target_classes:
        n_end = int((endpoint_mask & (merged[cluster_col].to_numpy() == cname)).sum())
        print(f"[INFO] endpoint count - {cname}: {n_end}")

    if int(target_mask.sum()) <= 0:
        raise RuntimeError("No target class cells found at the final timepoint.")

    if focus_endpoint_only:
        plot_mask = endpoint_mask
    else:
        plot_mask = np.ones(len(merged), dtype=bool)

    x_plot = x_emb[plot_mask]
    m_plot = merged.loc[plot_mask].reset_index(drop=True)
    is_target_plot = m_plot[cluster_col].isin(target_classes).to_numpy()

    # 4) Plot (full/clean)
    xlim, ylim = _compute_square_limits(x_plot, q=Q, pad=PAD)
    _render_variant(
        x_plot=x_plot,
        m_plot=m_plot,
        is_target_plot=is_target_plot,
        cluster_col=cluster_col,
        target_classes=target_classes,
        xlim=xlim,
        ylim=ylim,
        show_background=show_background,
        show_legend=True,
        out_png=out_png_full,
        out_pdf=out_pdf_full,
    )
    _render_variant(
        x_plot=x_plot,
        m_plot=m_plot,
        is_target_plot=is_target_plot,
        cluster_col=cluster_col,
        target_classes=target_classes,
        xlim=xlim,
        ylim=ylim,
        show_background=show_background,
        show_legend=False,
        out_png=out_png_clean,
        out_pdf=out_pdf_clean,
    )

    print(f"[DONE] saved: {out_png_full}")
    print(f"[DONE] saved: {out_pdf_full}")
    print(f"[DONE] saved: {out_png_clean}")
    print(f"[DONE] saved: {out_pdf_clean}")


if __name__ == "__main__":
    main()
