#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import torch

# ============================================================
# USER SETTINGS (edit if needed)

# ============================================================

CONFIG_PATH = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/config/GSE175634.yaml"
CSV_PATH_OVERRIDE = "/Users/wanghongye/python/scLineagetracer/GSE175634/preprocess_final/ruot_input_pca30_forward.csv"


# ===== Start time control =====
START_T = 0   # 你想从哪个 samples 开始（注意：这是 samples 空间，不是“真实天数”）

# ============================================================
# Times you want as SINGLE images
# - t0(=0) must be GT-only
# - All others are Pred-only images (each time one image)
# ============================================================
PRED_TIMES = sorted(set([
   0.3,0.5,0.7,1,2,3,4,5, 6,7,8,9,10,11,12,13,14,15
]))
# GT-only 单图（存在于 CSV 的才会输出）。t0 必定会输出（仅真实）
GT_ONLY_CANDIDATES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0 , 7.0, 11,15]

# ============================================================
# PHYSICAL INTEGRATION SETTINGS
# ============================================================
SEED = 0
N_INIT = 2500
DT = 0.05
USE_CACHE = False


# ============================================================
# SQUARE EXPORT SETTINGS (paper-friendly)
# ============================================================
FIG_INCH = 6.0     # 正方形画布 6×6 inch
DPI = 300          # 300dpi -> 1800×1800 px
Q = 0.01           # 分位数裁剪（稳健）
PAD = 0.08         # 额外留白比例（稳健）

# ============================================================
# V1-LIKE STYLE
# ============================================================
# Ground Truth as 'x'
GT_S = 18
GT_LW = 1.6
GT_ALPHA_BG = 0.10
GT_ALPHA_HI = 0.60
GT_ALPHA_BIG = 0.25

# Predicted points (colored by piecewise interpolation) with black edge
PRED_S = 28
PRED_EDGE_LW = 0.9
PRED_ALPHA = 0.90

# Trajectory lines in black
TRAJ_COLOR = "black"
TRAJ_LW = 0.8
TRAJ_ALPHA = 0.2
N_TRAJ_DRAW = 0

# Big predicted figure: reduce point alpha a bit to avoid over-dark
BIG_PRED_ALPHA = 0.55

# ============================================================
# Import DeepRUOT
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "."))
PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
for p in [PROJECT_ROOT, PARENT]:
    if p not in sys.path:
        sys.path.append(p)

from DeepRUOT.utils import load_and_merge_config, euler_sdeint
from DeepRUOT.models import FNet, scoreNet2
from DeepRUOT.constants import RES_DIR, DATA_DIR
from DeepRUOT.exp import setup_exp


def _is_ckpt_file(path: str) -> bool:
    low = os.path.basename(path).lower()
    if any(low.endswith(s) for s in [".npy",".npz",".csv",".pkl",".png",".pdf",".svg",".log",".txt",".yml",".yaml"]):
        return False
    return os.path.isfile(path)

def _fmt_t_for_name(t: float) -> str:
    t = float(t)
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t)))
    return f"{t:g}".replace(".", "p")

def _finite_rows(A: np.ndarray) -> np.ndarray:
    return np.isfinite(A).all(axis=1)

def piecewise_color(t_pred: float, gt_times: np.ndarray, gt_cols: np.ndarray) -> np.ndarray:
    """
    Piecewise linear interpolation between neighboring GT time colors.
    Example: between t=1 and t=2, t=1.3/1.5/1.7 interpolate smoothly.
    """
    t_pred = float(t_pred)
    if t_pred <= gt_times[0]:
        return gt_cols[0].copy()
    if t_pred >= gt_times[-1]:
        return gt_cols[-1].copy()

    i = np.searchsorted(gt_times, t_pred, side="right") - 1
    i = int(np.clip(i, 0, len(gt_times) - 2))

    tL, tR = float(gt_times[i]), float(gt_times[i + 1])
    a = 0.0 if tR == tL else (t_pred - tL) / (tR - tL)
    a = float(np.clip(a, 0.0, 1.0))
    return (1.0 - a) * gt_cols[i] + a * gt_cols[i + 1]

def compute_square_limits(P2: np.ndarray, q: float = 0.01, pad: float = 0.08):
    """Return square limits (xlim, ylim) using robust quantiles."""
    P2 = P2[np.isfinite(P2).all(axis=1)]
    x0, x1 = np.quantile(P2[:, 0], [q, 1 - q])
    y0, y1 = np.quantile(P2[:, 1], [q, 1 - q])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = max(x1 - x0, y1 - y0) / 2.0
    r *= (1.0 + pad)
    return (cx - r, cx + r), (cy - r, cy + r)

AX_PAD = 0.008  # figure fraction：越小白边越少；过小可能导致黑框被裁切一点点

def apply_square_style(ax, XLIM, YLIM):
    """No title, square view, fixed limits, thin frame, no ticks."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    # 关键：让 axes 铺满画布，减少四周留白
    ax.set_position([AX_PAD, AX_PAD, 1 - 2*AX_PAD, 1 - 2*AX_PAD])

    # 黑框（spines）
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("black")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------
    # 1) Load config & locate exp_dir
    # ------------------------------------------------------------
    config = load_and_merge_config(CONFIG_PATH)
    model_cfg = config["model"]

    dev = str(config.get("device", "cpu")).lower()
    if dev == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif dev == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sigma = 0.0
    if isinstance(config.get("score_train", None), dict):
        sigma = float(config["score_train"].get("sigma", 0.0))

    res_root = config.get("exp", {}).get("output_dir", None)
    if res_root is None:
        res_root = RES_DIR

    exp_name = config["exp"]["name"]
    exp_dir, _ = setup_exp(res_root, config, exp_name)

    fig_dir = os.path.join(exp_dir, "figures_csv")
    single_dir = os.path.join(fig_dir, "single_1")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)

    print("[INFO] exp_dir:", exp_dir)
    print("[INFO] fig_dir:", fig_dir)
    print("[INFO] single_dir:", single_dir)
    print("[INFO] device:", device)
    print("[INFO] sigma:", sigma)

    # ------------------------------------------------------------
    # 2) Load CSV (GT)
    # ------------------------------------------------------------
    csv_path = CSV_PATH_OVERRIDE if CSV_PATH_OVERRIDE else config["data"]["file_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(DATA_DIR, csv_path)

    df = pd.read_csv(csv_path)
    x_cols = [c for c in df.columns if c.startswith("x")]
    if "samples" not in df.columns or len(x_cols) == 0:
        raise RuntimeError("CSV must contain 'samples' and x1..xD columns.")

    # sort x columns by numeric suffix
    def x_key(c):
        try:
            return int(c[1:])
        except Exception:
            return 10**9
    x_cols = sorted(x_cols, key=x_key)

    X = df[x_cols].values.astype(np.float32)
    times = df["samples"].values.astype(float)
    uniq_times = sorted(np.unique(times))
    dim = X.shape[1]
    # --- override t0 ---
    t0 = float(START_T)

    # sanity check: START_T must exist in GT times
    if not np.any(np.isclose(times, t0, atol=1e-9)):
        raise RuntimeError(f"START_T={t0} not found in CSV 'samples'. Available: {sorted(np.unique(times))}")

    # filter prediction times >= t0
    PRED_TIMES = sorted([float(t) for t in PRED_TIMES if float(t) >= t0 - 1e-9])

    if len(PRED_TIMES) == 0:
        raise RuntimeError(f"All PRED_TIMES are < START_T={t0}. Please provide PRED_TIMES >= {t0}.")
    print("[INFO] CSV:", csv_path)
    print("[INFO] GT times:", uniq_times)
    print("[INFO] dim:", dim)
    print("[INFO] PRED_TIMES:", PRED_TIMES)

    # t0 = earliest GT time
    t_end = float(max(PRED_TIMES))

    # GT-only times present in CSV
    gt_only = [t for t in GT_ONLY_CANDIDATES if any(abs(float(t)-float(u))<1e-9 for u in uniq_times)]
    if all(abs(float(t0)-float(t)) > 1e-9 for t in gt_only):
        gt_only = [t0] + gt_only  # ensure t0 included

    # ------------------------------------------------------------
    # 3) dim_reduction.pkl (auto-create if missing)
    # ------------------------------------------------------------
    dr_path = os.path.join(exp_dir, "dim_reduction.pkl")
    dim_reducer = None
    X2 = None

    if not os.path.exists(dr_path):
        print("[WARN] dim_reduction.pkl not found. Will create it using UMAP.")
        import umap
        umap_op = umap.UMAP(n_components=2, random_state=42)
        X2 = umap_op.fit_transform(X).astype(np.float32)
        joblib.dump(umap_op, dr_path)
        dim_reducer = umap_op
        print("[OK] created:", dr_path)
    else:
        dim_reducer = joblib.load(dr_path)
        # If embedding_ exists and matches, prefer it (stable)
        if hasattr(dim_reducer, "embedding_") and getattr(dim_reducer, "embedding_", None) is not None:
            emb = np.asarray(dim_reducer.embedding_)
            if emb.ndim == 2 and emb.shape[0] == X.shape[0] and emb.shape[1] == 2:
                X2 = emb.astype(np.float32)

        if X2 is None:
            try:
                X2 = dim_reducer.transform(X).astype(np.float32)
            except Exception as e:
                print("[WARN] dim_reducer.transform failed; fallback to X[:, :2]. Error:", e)
                X2 = X[:, :2].astype(np.float32)

    ok_bg = _finite_rows(X2)
    X2_bg = X2[ok_bg]

    # ------------------------------------------------------------
    # 4) GT colormap & piecewise interpolation basis
    # ------------------------------------------------------------
    K = len(uniq_times)
    cmap = plt.cm.viridis
    gt_cols = np.array([cmap(i / (K - 1 if K > 1 else 1)) for i in range(K)], dtype=float)  # (K,4)
    gt_times = np.array([float(t) for t in uniq_times], dtype=float)                          # (K,)
    t_to_color = {float(t): gt_cols[i] for i, t in enumerate(gt_times)}

    # ------------------------------------------------------------
    # 5) Load checkpoints (f_net and score_model)
    # ------------------------------------------------------------
    f_net = FNet(
        in_out_dim=model_cfg["in_out_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hiddens=model_cfg["n_hiddens"],
        activation=model_cfg["activation"],
    ).to(device)

    score_model = scoreNet2(
        in_out_dim=model_cfg["in_out_dim"],
        hidden_dim=model_cfg["score_hidden_dim"],
        activation=model_cfg["activation"],
    ).float().to(device)

    cand_model, cand_score = [], []
    for root, _, files in os.walk(exp_dir):
        for fn in files:
            p = os.path.join(root, fn)
            if not _is_ckpt_file(p):
                continue
            low = fn.lower()
            if "model" in low:
                cand_model.append(p)
            if "score" in low:
                cand_score.append(p)

    cand_model = sorted(list(dict.fromkeys(cand_model)), key=lambda p: os.path.getmtime(p), reverse=True)
    cand_score = sorted(list(dict.fromkeys(cand_score)), key=lambda p: os.path.getmtime(p), reverse=True)

    def try_load(net, cands, tag):
        for p in cands:
            try:
                obj = torch.load(p, map_location=torch.device("cpu"))
            except Exception:
                continue

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
                net.load_state_dict(sd, strict=True)
                print(f"[OK] loaded {tag}:", p)
                return True
            except Exception:
                try:
                    net.load_state_dict(sd, strict=False)
                    print(f"[OK] loaded {tag} (strict=False):", p)
                    return True
                except Exception:
                    continue
        return False

    if not try_load(f_net, cand_model, "f_net"):
        raise RuntimeError("Cannot find a loadable f_net checkpoint under exp_dir.")
    if not try_load(score_model, cand_score, "score"):
        raise RuntimeError("Cannot find a loadable score checkpoint under exp_dir.")

    f_net.eval()
    score_model.eval()

    # ------------------------------------------------------------
    # 6) Physical integration -> traj2(ts, particle, 2) (cache)
    # ------------------------------------------------------------
    required_times = sorted(set([t0, t_end] + [float(t) for t in PRED_TIMES]))
    times_key = ",".join([f"{t:.4f}" for t in required_times])
    cache_sig = hashlib.md5(
        f"dt={DT};n={N_INIT};t0={t0};tend={t_end};times={times_key};sigma={sigma};dim={dim}".encode()
    ).hexdigest()[:10]
    cache_path = os.path.join(exp_dir, f"phys_traj2_sq_{cache_sig}.npz")

    if USE_CACHE and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        ts = z["ts"].astype(np.float32)
        traj2 = z["traj2"].astype(np.float32)
        print("[INFO] loaded cache:", cache_path)
    else:
        idx0 = np.where(np.isclose(times, t0, atol=1e-9))[0]
        if len(idx0) == 0:
            idx0 = np.arange(X.shape[0])
        if len(idx0) > N_INIT:
            idx0 = np.random.choice(idx0, size=N_INIT, replace=False)

        x0 = torch.tensor(X[idx0], dtype=torch.float32, device=device)
        lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
        init_state = (x0, lnw0)

        class SDE(torch.nn.Module):
            noise_type = "diagonal"
            sde_type = "ito"
            def __init__(self, v_net, score, sigma):
                super().__init__()
                self.v_net = v_net
                self.score = score
                self.sigma = float(sigma)

            def f(self, t, y):
                z, lnw = y
                t_exp = t.reshape(1, 1).expand(z.shape[0], 1)

                drift = self.v_net(t_exp, z)

                if hasattr(self.score, "compute_gradient"):
                    sgrad = self.score.compute_gradient(t_exp, z)
                else:
                    z_req = z.detach().requires_grad_(True)
                    ld = self.score(t_exp, z_req)
                    sgrad = torch.autograd.grad(ld.sum(), z_req, create_graph=False)[0]

                dz = drift + sgrad
                dlnw = torch.zeros_like(lnw)  # use_mass=False
                return (dz, dlnw)

            # DeepRUOT euler_sdeint calls g(t, z)
            def g(self, t, z):
                return torch.ones_like(z) * self.sigma

        sde = SDE(f_net.v_net, score_model, sigma=sigma)

        ts = np.arange(t0, t_end + 1e-9, DT).astype(np.float32)
        ts = np.unique(np.concatenate([ts, np.array(required_times, dtype=np.float32)]))
        ts = np.sort(ts)

        ts_torch = torch.tensor(ts, dtype=torch.float32, device=device)
        traj, _ = euler_sdeint(sde, init_state, dt=DT, ts=ts_torch)  # (len(ts), N_INIT, dim)
        traj_np = traj.detach().cpu().numpy().astype(np.float32)

        flat = traj_np.reshape(-1, dim).astype(np.float32)
        try:
            traj2 = dim_reducer.transform(flat).reshape(len(ts), -1, 2).astype(np.float32)
        except Exception as e:
            print("[WARN] transform trajectory failed; fallback to first2 dims. Error:", e)
            traj2 = flat[:, :2].reshape(len(ts), -1, 2).astype(np.float32)

        if USE_CACHE:
            np.savez_compressed(cache_path, ts=ts, traj2=traj2)
            print("[INFO] saved cache:", cache_path)

    def idx_of(t):
        return int(np.argmin(np.abs(ts - float(t))))

    # sample indices for trajectory lines (consistent across all images)
    n_traj = traj2.shape[1]
    draw_idx = np.arange(n_traj)
    if n_traj > N_TRAJ_DRAW:
        draw_idx = np.random.choice(n_traj, N_TRAJ_DRAW, replace=False)

    # ------------------------------------------------------------
    # 7) Compute global square limits (include GT + predicted points)
    #    This ensures all exported images have consistent scale within dataset.
    # ------------------------------------------------------------
    pred_stack = []
    for t in PRED_TIMES:
        if abs(float(t) - float(t0)) < 1e-9:
            continue
        pred_stack.append(traj2[idx_of(t)])
    if len(pred_stack) > 0:
        pred_all = np.vstack(pred_stack)
        pred_all = pred_all[_finite_rows(pred_all)]
        union = np.vstack([X2_bg, pred_all]) if pred_all.shape[0] > 0 else X2_bg
    else:
        union = X2_bg

    XLIM, YLIM = compute_square_limits(union, q=Q, pad=PAD)

    # ------------------------------------------------------------
    # 8) Export GT-only images (square, no title)
    # ------------------------------------------------------------
    for t in gt_only:
        fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))

        bg = ax.scatter(
            X2_bg[:, 0], X2_bg[:, 1],
            s=GT_S, marker="x", c="#CFCFCF",
            linewidths=GT_LW, alpha=GT_ALPHA_BG, zorder=0
        )
        bg.set_rasterized(True)

        m = np.isclose(times, float(t), atol=1e-9)
        P = X2[m]
        P = P[_finite_rows(P)]
        col = t_to_color.get(float(t), np.array([0.4, 0.4, 0.4, 1.0], dtype=float))

        hi = ax.scatter(
            P[:, 0], P[:, 1],
            s=GT_S, marker="x",
            c=[col], linewidths=GT_LW,
            alpha=GT_ALPHA_HI, zorder=2
        )
        hi.set_rasterized(True)

        apply_square_style(ax, XLIM, YLIM)

        out_png = os.path.join(single_dir, f"gt_t{_fmt_t_for_name(t)}.png")
        out_pdf = os.path.join(single_dir, f"gt_t{_fmt_t_for_name(t)}.pdf")
        fig.savefig(out_png, dpi=DPI)          # IMPORTANT: no bbox_inches="tight"
        try:
            fig.savefig(out_pdf)
        except Exception:
            fig.savefig(out_pdf.replace(".pdf", ".svg"))
        plt.close(fig)
        print("[OK] saved:", out_png)

    # ------------------------------------------------------------
    # 9) Export Pred-only images (square, no title)
    #    - skip t0 (GT-only)
    #    - predicted point color: piecewise interpolation
    # ------------------------------------------------------------
    for t in PRED_TIMES:
        if abs(float(t) - float(t0)) < 1e-9:
            continue

        idx_end = idx_of(t)

        fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))

        bg = ax.scatter(
            X2_bg[:, 0], X2_bg[:, 1],
            s=GT_S, marker="x", c="#CFCFCF",
            linewidths=GT_LW, alpha=GT_ALPHA_BG, zorder=0
        )
        bg.set_rasterized(True)

        # trajectory segments up to time t
        segs = []
        for j in draw_idx:
            for ii in range(idx_end):
                p0 = traj2[ii, j]
                p1 = traj2[ii + 1, j]
                if np.isfinite(p0).all() and np.isfinite(p1).all():
                    segs.append([p0, p1])
        lc = LineCollection(segs, colors=TRAJ_COLOR, linewidths=TRAJ_LW, alpha=TRAJ_ALPHA)
        lc.set_rasterized(True)
        ax.add_collection(lc)

        # predicted points at time t
        Pp = traj2[idx_end]
        Pp = Pp[_finite_rows(Pp)]
        pred_col = piecewise_color(float(t), gt_times, gt_cols)

        pr = ax.scatter(
            Pp[:, 0], Pp[:, 1],
            s=PRED_S, marker="o",
            facecolors=[pred_col],
            edgecolors="black",
            linewidths=PRED_EDGE_LW,
            alpha=PRED_ALPHA, zorder=3
        )
        pr.set_rasterized(True)

        apply_square_style(ax, XLIM, YLIM)

        out_png = os.path.join(single_dir, f"pred_t{_fmt_t_for_name(t)}.png")
        out_pdf = os.path.join(single_dir, f"pred_t{_fmt_t_for_name(t)}.pdf")
        fig.savefig(out_png, dpi=DPI)
        try:
            fig.savefig(out_pdf)
        except Exception:
            fig.savefig(out_pdf.replace(".pdf", ".svg"))
        plt.close(fig)
        print("[OK] saved:", out_png)

    # ------------------------------------------------------------
    # 10) Big figure A (square): continuous predicted trajectories + all predicted points
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))

    bg = ax.scatter(
        X2_bg[:, 0], X2_bg[:, 1],
        s=GT_S, marker="x", c="#CFCFCF",
        linewidths=GT_LW, alpha=0.10, zorder=0
    )
    bg.set_rasterized(True)

    # full trajectories
    segs = []
    for j in draw_idx:
        for ii in range(len(ts) - 1):
            p0 = traj2[ii, j]
            p1 = traj2[ii + 1, j]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                segs.append([p0, p1])
    lc = LineCollection(segs, colors=TRAJ_COLOR, linewidths=TRAJ_LW, alpha=TRAJ_ALPHA)
    lc.set_rasterized(True)
    ax.add_collection(lc)

    # all predicted points (colored by piecewise interpolation)
    for t in PRED_TIMES:
        if abs(float(t) - float(t0)) < 1e-9:
            continue
        idx_t = idx_of(t)
        Pp = traj2[idx_t]
        Pp = Pp[_finite_rows(Pp)]
        pred_col = piecewise_color(float(t), gt_times, gt_cols)

        pr = ax.scatter(
            Pp[:, 0], Pp[:, 1],
            s=PRED_S, marker="o",
            facecolors=[pred_col],
            edgecolors="black",
            linewidths=PRED_EDGE_LW,
            alpha=BIG_PRED_ALPHA, zorder=3
        )
        pr.set_rasterized(True)

    apply_square_style(ax, XLIM, YLIM)

    # optional legend (inside, won't change output size)
    pred_proxy = Line2D([0], [0], marker="o", linestyle="None",
                        markerfacecolor=gt_cols[len(gt_cols)//2], markeredgecolor="black",
                        markersize=8, label="Predicted (interpolated)")
    traj_proxy = Line2D([0], [0], color="black", linewidth=1.5, label="Trajectory")
    #ax.legend(handles=[pred_proxy, traj_proxy], loc="upper right", frameon=True, fontsize=9)

    out_png = os.path.join(fig_dir, "big_pred_continuous_all_points.png")
    out_pdf = os.path.join(fig_dir, "big_pred_continuous_all_points.pdf")
    fig.savefig(out_png, dpi=DPI)
    try:
        fig.savefig(out_pdf)
    except Exception:
        fig.savefig(out_pdf.replace(".pdf", ".svg"))
    plt.close(fig)
    print("[OK] saved:", out_png)

    # ------------------------------------------------------------
    # 11) Big figure B (square): ground truth only (x colored by discrete times)
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))

    for t in uniq_times:
        m = np.isclose(times, float(t), atol=1e-9)
        P = X2[m]
        P = P[_finite_rows(P)]
        col = t_to_color.get(float(t), np.array([0.4, 0.4, 0.4, 1.0], dtype=float))

        sc_gt = ax.scatter(
            P[:, 0], P[:, 1],
            s=GT_S, marker="x",
            c=[col],
            linewidths=GT_LW,
            alpha=GT_ALPHA_BIG, zorder=1
        )
        sc_gt.set_rasterized(True)

    apply_square_style(ax, XLIM, YLIM)

    # optional time legend (inside)
    time_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=gt_cols[i], markeredgecolor="none",
               markersize=7, label=f"T{i}")
        for i in range(K)
    ]
    #ax.legend(handles=time_handles, loc="upper left", frameon=True, fontsize=9)

    out_png = os.path.join(fig_dir, "big_real_only.png")
    out_pdf = os.path.join(fig_dir, "big_real_only.pdf")
    fig.savefig(out_png, dpi=DPI)
    try:
        fig.savefig(out_pdf)
    except Exception:
        fig.savefig(out_pdf.replace(".pdf", ".svg"))
    plt.close(fig)
    print("[OK] saved:", out_png)

    print("[DONE] Single images saved in:", single_dir)
    print("[DONE] Big figures saved in:", fig_dir)
