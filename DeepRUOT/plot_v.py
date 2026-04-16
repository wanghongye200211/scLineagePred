#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import torch

import anndata
import scanpy as sc
import scvelo as scv

# ============================================================
# USER PATHS
# ============================================================
CONFIG_PATH = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/config/GSE99915.yaml"
CSV_PATH_OVERRIDE = "/Users/wanghongye/python/scLineagetracer/GSE99915/preprocess_final/ruot_input_pca50_forward.csv"

# ============================================================
# STRATIFIED SUBSAMPLE (方案A)
# ============================================================
SEED = 0
MAX_PER_TIME = 5000  # 每个时间点最多保留多少细胞（建议 5k~15k）

# ============================================================
# SCVELO SETTINGS (不改画法)
# ============================================================
N_NEIGHBORS = 30
N_JOBS = 16
BATCH_SIZE = 8192

# ============================================================
# PLOT.PY SQUARE EXPORT STYLE (照抄)
# ============================================================
FIG_INCH = 6.0
DPI = 300
Q = 0.01
PAD = 0.08
AX_PAD = 0.008  # 和 plot.py 一致

OUT_SUBDIR = "figures_flow"
OUT_NAME = f"v_stream_sq_stratified_max{MAX_PER_TIME}_viridis_nolegend_frame"

# ============================================================
# Import DeepRUOT (same style as plot.py)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "."))
PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
for p in [PROJECT_ROOT, PARENT]:
    if p not in sys.path:
        sys.path.append(p)

from DeepRUOT.utils import load_and_merge_config
from DeepRUOT.models import FNet
from DeepRUOT.constants import RES_DIR, DATA_DIR
from DeepRUOT.exp import setup_exp


# ---------------- plot.py helpers ----------------
def _finite_rows(A: np.ndarray) -> np.ndarray:
    return np.isfinite(A).all(axis=1)

def compute_square_limits(P2: np.ndarray, q: float = 0.01, pad: float = 0.08):
    P2 = P2[np.isfinite(P2).all(axis=1)]
    x0, x1 = np.quantile(P2[:, 0], [q, 1 - q])
    y0, y1 = np.quantile(P2[:, 1], [q, 1 - q])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = max(x1 - x0, y1 - y0) / 2.0
    r *= (1.0 + pad)
    return (cx - r, cx + r), (cy - r, cy + r)

def apply_square_style(ax, XLIM, YLIM):
    """
    强制打开 axis + 铺满画布 + 黑框
    （scvelo/scanpy 有时会 frameon=False/axis off，导致 spines 不画）
    """
    ax.set_axis_on()
    ax.set_frame_on(True)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_position([AX_PAD, AX_PAD, 1 - 2*AX_PAD, 1 - 2*AX_PAD])

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("black")
        sp.set_zorder(50)

    # 保险：再画一层显式矩形边框（永远可见）
    ax.add_patch(Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor="black",
        linewidth=0.8,
        zorder=60,
        clip_on=False
    ))


# ---------------- core helpers ----------------
def stratified_subsample_by_time(times: np.ndarray, max_per_time: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times = np.asarray(times)
    idx_list = []
    for t in np.unique(times):
        ids = np.where(times == t)[0]
        k = min(len(ids), max_per_time)
        idx_list.append(rng.choice(ids, size=k, replace=False))
    idx = np.concatenate(idx_list)
    idx.sort()
    return idx

def build_viridis_palette_from_times(times: np.ndarray):
    uniq = np.array(sorted(np.unique(times.astype(float))), dtype=float)
    K = len(uniq)
    cmap = plt.cm.viridis
    cols = [to_hex(cmap(i / (K - 1 if K > 1 else 1))) for i in range(K)]
    return uniq, cols

def load_dim_reducer_and_umap(exp_dir: str, X_full_n: int, X_sub: np.ndarray, idx: np.ndarray):
    dr_path = os.path.join(exp_dir, "dim_reduction.pkl")
    if not os.path.exists(dr_path):
        raise RuntimeError(f"dim_reduction.pkl not found: {dr_path}（先跑 dim_pkl.py 生成它）")

    dim_reducer = joblib.load(dr_path)

    X_umap = None
    if hasattr(dim_reducer, "embedding_") and getattr(dim_reducer, "embedding_", None) is not None:
        emb = np.asarray(dim_reducer.embedding_)
        if emb.ndim == 2 and emb.shape[0] == X_full_n and emb.shape[1] == 2:
            X_umap = emb[idx].astype(np.float32)

    if X_umap is None:
        X_umap = dim_reducer.transform(X_sub).astype(np.float32)

    return dim_reducer, X_umap

def _is_ckpt_file(path: str) -> bool:
    low = os.path.basename(path).lower()
    if any(low.endswith(s) for s in [".npy",".npz",".csv",".pkl",".png",".pdf",".svg",".log",".txt",".yml",".yaml"]):
        return False
    return os.path.isfile(path)

def load_fnet(exp_dir: str, model_cfg: dict, device: torch.device) -> FNet:
    f_net = FNet(
        in_out_dim=model_cfg["in_out_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hiddens=model_cfg["n_hiddens"],
        activation=model_cfg["activation"],
    ).to(device)

    cands = []
    p0 = os.path.join(exp_dir, "model_final")
    if os.path.isfile(p0):
        cands.append(p0)

    for root, _, files in os.walk(exp_dir):
        for fn in files:
            p = os.path.join(root, fn)
            if not _is_ckpt_file(p):
                continue
            if "model" in fn.lower():
                cands.append(p)

    cands = sorted(list(dict.fromkeys(cands)), key=lambda p: os.path.getmtime(p), reverse=True)
    if len(cands) == 0:
        raise RuntimeError("Cannot find f_net checkpoint under exp_dir.")

    for p in cands:
        obj = torch.load(p, map_location=torch.device("cpu"))
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif isinstance(obj, dict):
            sd = obj
        else:
            continue

        keys = list(sd.keys())
        if len(keys) > 0 and all(k.startswith("module.") for k in keys):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        try:
            f_net.load_state_dict(sd, strict=True)
            print("[OK] loaded f_net:", p)
            f_net.eval()
            return f_net
        except Exception:
            try:
                f_net.load_state_dict(sd, strict=False)
                print("[OK] loaded f_net (strict=False):", p)
                f_net.eval()
                return f_net
            except Exception:
                continue

    raise RuntimeError("Found checkpoints but none could be loaded into f_net.")

def batched_vnet(f_net: FNet, X: np.ndarray, t: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    n = X.shape[0]
    V = np.zeros_like(X, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            xb = torch.tensor(X[i:j], dtype=torch.float32, device=device)
            tb = torch.tensor(t[i:j], dtype=torch.float32, device=device).unsqueeze(1)
            vb = f_net.v_net(tb, xb)
            V[i:j] = vb.detach().cpu().numpy().astype(np.float32)
    return V


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) config & exp_dir
    config = load_and_merge_config(CONFIG_PATH)
    model_cfg = config["model"]

    dev = str(config.get("device", "cpu")).lower()
    if dev == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif dev == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    res_root = config.get("exp", {}).get("output_dir", None)
    if res_root is None:
        res_root = RES_DIR

    exp_name = config["exp"]["name"]
    exp_dir, _ = setup_exp(res_root, config, exp_name)

    print("[INFO] device:", device)
    print("[INFO] exp_dir:", exp_dir)

    # 2) load CSV full
    csv_path = CSV_PATH_OVERRIDE if CSV_PATH_OVERRIDE else config["data"]["file_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(DATA_DIR, csv_path)

    df = pd.read_csv(csv_path)

    x_cols = [c for c in df.columns if c.startswith("x")]
    if "samples" not in df.columns or len(x_cols) == 0:
        raise RuntimeError("CSV must contain 'samples' and x1..xD columns.")

    def x_key(c):
        try:
            return int(c[1:])
        except Exception:
            return 10**9
    x_cols = sorted(x_cols, key=x_key)

    times_full = df["samples"].to_numpy(dtype=float)
    X_full = df[x_cols].to_numpy(dtype=np.float32)

    n_full = X_full.shape[0]
    print("[INFO] full N:", n_full, "dim:", X_full.shape[1])
    print("[INFO] unique times (full):", sorted(np.unique(times_full)))

    # FULL: drop non-finite
    keep_full = _finite_rows(X_full) & np.isfinite(times_full)
    if keep_full.sum() < n_full:
        print(f"[WARN] drop non-finite rows (FULL): {n_full - keep_full.sum()} / {n_full}")
        X_full = X_full[keep_full]
        times_full = times_full[keep_full]
        n_full = X_full.shape[0]

    # 3) stratified subsample
    idx = stratified_subsample_by_time(times_full, MAX_PER_TIME, seed=SEED)
    X = X_full[idx]
    times = times_full[idx]

    print("[INFO] subsampled N:", X.shape[0], f"(max_per_time={MAX_PER_TIME})")

    # 4) dim_reduction.pkl -> X_umap
    _, X_umap = load_dim_reducer_and_umap(exp_dir, X_full_n=n_full, X_sub=X, idx=idx)

    # SUB: drop any non-finite (PDF 必须)
    keep = _finite_rows(X) & _finite_rows(X_umap) & np.isfinite(times)
    if keep.sum() < X.shape[0]:
        print(f"[WARN] drop non-finite rows (SUB): {X.shape[0] - keep.sum()} / {X.shape[0]}")
        X = X[keep]
        X_umap = X_umap[keep]
        times = times[keep]

    # 5) compute velocity = v_net(t, x)
    f_net = load_fnet(exp_dir, model_cfg, device=device)
    V = batched_vnet(f_net, X, times, device=device, batch_size=BATCH_SIZE)

    keep_v = _finite_rows(V)
    if keep_v.sum() < V.shape[0]:
        print(f"[WARN] drop non-finite velocity rows: {V.shape[0] - keep_v.sum()} / {V.shape[0]}")
        X = X[keep_v]
        X_umap = X_umap[keep_v]
        times = times[keep_v]
        V = V[keep_v]

    # 6) AnnData + scVelo pipeline (不改画法)
    adata = anndata.AnnData(X=X)
    adata.layers["Ms"] = X
    adata.layers["velocity"] = V
    adata.obsm["X_umap"] = X_umap
    adata.obs["time"] = times

    if adata.layers["velocity"].shape[1] != 2:
        sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X")
        scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=N_JOBS)
        scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
    else:
        adata.obsm["velocity_umap"] = adata.layers["velocity"]

    # 防御：velocity_umap 非有限会导致 PDF 崩
    if "velocity_umap" in adata.obsm:
        V_umap = np.asarray(adata.obsm["velocity_umap"])
        bad = ~np.isfinite(V_umap).all(axis=1)
        if bad.any():
            print(f"[WARN] velocity_umap non-finite rows: {bad.sum()} / {V_umap.shape[0]} -> set to 0")
            V_umap[bad] = 0.0
            adata.obsm["velocity_umap"] = V_umap

    # 7) colors (viridis, no interpolation) + no legend
    uniq_times, palette_list = build_viridis_palette_from_times(adata.obs["time"].to_numpy())
    adata.obs["time_categorical"] = pd.Categorical(adata.obs["time"], categories=uniq_times, ordered=True)

    # 8) square limits (plot.py style)
    ok_bg = _finite_rows(adata.obsm["X_umap"])
    XLIM, YLIM = compute_square_limits(adata.obsm["X_umap"][ok_bg], q=Q, pad=PAD)

    # 9) plot on our own square canvas (no legend)
    scv.settings.set_figure_params("scvelo")
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH), dpi=DPI)

    ax_ret = scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        color="time_categorical",
        arrowsize=2.0,
        linewidth=1.5,
        density=3,
        title=None,
        legend_loc="none",     # <<< 不要侧面图例
        palette=palette_list,
        ax=ax,
        show=False,
    )

    if isinstance(ax_ret, (list, tuple)):
        ax = ax_ret[0]
    else:
        ax = ax_ret

    # rasterize to keep pdf small & stable
    for coll in ax.collections:
        try:
            coll.set_rasterized(True)
        except Exception:
            pass

    apply_square_style(ax, XLIM, YLIM)

    # 10) save (和 plot.py 一样：不要 bbox_inches="tight"，避免裁边框)
    out_dir = os.path.join(exp_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{OUT_NAME}.png")
    out_pdf = os.path.join(out_dir, f"{OUT_NAME}.pdf")

    fig.savefig(out_png, dpi=DPI)
    try:
        fig.savefig(out_pdf)
    except Exception:
        fig.savefig(out_pdf.replace(".pdf", ".svg"))

    plt.close(fig)

    print("[OK] saved:", out_png)
    print("[OK] saved:", out_pdf)
