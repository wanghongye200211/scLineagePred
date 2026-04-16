# -*- coding: utf-8 -*-
"""
decoder_99915.py  (log1p only, seaborn-template style)

Input:
- <reg_out_dir>/<task>/test_outputs.npz
  required keys: pred_log, true_log, gene_names, label

Output:
- <task>/plots_log/scatter_<label>.png      (TWO plots only; different colors)
- <task>/plots_log/violin_<label>.png

Key fixes:
- Scatter: two plots (Failed/Reprogrammed), each with its own color (editable at top)
- Violin: force y-axis include 0 to avoid trimming near zero (violin_bottom_floor=0.0)

Notes:
- Violin "cut"（切割 cut） parameter: Config.violin_cut -> sns.violinplot(cut=...)
- Your "0被裁掉" issue is usually caused by ylim based on percentiles, NOT cut.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

import seaborn as sns
from scipy.stats import pearsonr


# =========================================================
# ===============  COLOR SETTINGS (EDIT HERE) ===============
# =========================================================
# Scatter colors: two plots -> two colors (你只改这里就行)
SCATTER_COLORS_BY_LABEL = {
    "Failed": "#4C78A8",        # blue
    "Reprogrammed": "#F58518",  # orange
}
SCATTER_FALLBACK_COLOR = "#9ecae1"  # if label not found

# Diagonal line color on scatter
SCATTER_DIAG_LINE_COLOR = "#6E6E6E"

# Violin colors (Real vs Pred) if you want to adjust too
VIOLIN_PALETTE = {
    "Real": "#A6CEE3",  # light blue
    "Pred": "#FB9A99",  # light pink
}


# ===================== Config =====================
class Config:
    reg_out_dir = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE99915"
    tasks = None  # None -> auto detect tasks
   # keep_types = ("Neutrophil", "Monocyte")
    keep_labels = ("Reprogrammed", "Failed")
    out_subdir = "plots_log"

    # ---------- YOUR VIOLIN PARAMETERS ----------
    violin_genes_fail = []
    violin_genes_repr = []

    # auto top-k only used if manual violin list becomes empty after filtering
    violin_top_k = 10
    require_positive_r_for_top = True
    min_pos_r = 0.0

    # ---------- scatter labeling by REAL mean targets ----------
    scatter_label_targets = [0.5, 1.0, 1.5, 2.0, 2.5]  # REAL mean (log1p)
    scatter_label_outlier_percentile = 99.5           # do not label extreme outliers

    # ---------- style ----------
    dpi = 300
    seaborn_style = "white"  # clean background

    # violin palette
    palette = VIOLIN_PALETTE

    # scatter colors (two plots)
    scatter_colors_by_label = SCATTER_COLORS_BY_LABEL
    scatter_fallback_color = SCATTER_FALLBACK_COLOR

    # scatter appearance
    scatter_point_size = 60
    scatter_point_alpha = 0.72
    diag_line_color = SCATTER_DIAG_LINE_COLOR
    highlight_point_size = 220

    # draw only points with x>0 and y>0
    scatter_positive_only = True
    scatter_axis_margin_ratio = 0.04
    scatter_tick_step = 1  # x/y tick step must match

    # label offsets (to avoid text being blocked)
    label_dx_ratio = 0.025
    label_dy_ratio = 0.090

    # violin
    violin_inner = "box"
    violin_linewidth = 1.0
    violin_cut = 2            # cut（切割 cut）
    violin_scale = "width"

    violin_ylim_percentiles = (0.0, 95.0)
    violin_ylim_margin = 0.06
    violin_bottom_extra_ratio = 0.08
    violin_top_headroom_ratio = 0.40  # more blank space on top

    # FIX: force y-axis bottom include 0 (避免 0 被裁掉)
    # set None to disable
    violin_bottom_floor = None


    # no overlay points
    overlay_points = False

    # exclude (always)
    exclude_genes = ("Penk", "penk")


# ===================== helpers =====================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_pearsonr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    if np.std(a) < eps or np.std(b) < eps:
        return np.nan, np.nan
    r = pearsonr(a, b)[0]
    if not np.isfinite(r):
        return np.nan, np.nan
    return float(r), float(r * r)

def compute_clone_means(expr, clone_ids):
    # compatibility stub; NO clone aggregation
    return np.asarray(expr, dtype=np.float32), None

def _decode_bytes(arr):
    if arr is None:
        return None
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
        return np.array(
            [(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)).strip() for x in arr],
            dtype=object
        )
    return np.array(arr, dtype=object)

def _robust_ylim(values, qlo, qhi, margin_ratio):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo = np.percentile(v, qlo)
    hi = np.percentile(v, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < 1e-8:
        lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, 1e-6)
    pad = span * margin_ratio
    return float(lo - pad), float(hi + pad)

def _style_ax(ax, set_y_locator=True):
    ax.grid(False)

    # full black frame
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.6)

    # x-axis: keep labels, remove tick marks
    ax.tick_params(
        axis="x", which="both",
        bottom=True, top=False,
        direction="out",
        length=0, width=1.2, colors="black",
        pad=6
    )

    # y-axis: left only, ticks outside
    if set_y_locator:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(
        axis="y", which="major",
        left=True, right=False,
        labelleft=True, labelright=False,
        direction="out",
        length=6, width=1.2, colors="black",
        pad=6
    )
    ax.minorticks_off()

def list_tasks(root):
    tasks = []
    if not os.path.isdir(root):
        return tasks
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "test_outputs.npz")
        if os.path.exists(p):
            tasks.append(name)
    return tasks

def _clean_gene_list(gene_list):
    out = []
    for g in gene_list:
        if g is None:
            continue
        gs = str(g).strip()
        if gs == "":
            continue
        out.append(gs)
    return out

def _genes_to_indices(genes_all, gene_list, exclude_set):
    gene_list = _clean_gene_list(gene_list)
    idx, missing, dropped = [], [], []
    name_to_idx = {str(g): i for i, g in enumerate(genes_all)}
    for g in gene_list:
        gs = str(g)
        if gs.lower() in exclude_set:
            dropped.append(gs)
            continue
        if gs not in name_to_idx:
            missing.append(gs)
            continue
        idx.append(name_to_idx[gs])
    return idx, missing, dropped

def _annotate_labels(ax, x, y, names, cfg: Config):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n == 0:
        return

    xr = max(x.max() - x.min(), 1e-6)
    yr = max(y.max() - y.min(), 1e-6)
    dx = cfg.label_dx_ratio * xr
    dy = cfg.label_dy_ratio * yr

    order = np.argsort(y)
    x = x[order]
    y = y[order]
    names = [names[i] for i in order]

    placed_y = []
    for i in range(n):
        ha = "left" if (i % 2 == 0) else "right"
        x_text = x[i] + (dx if ha == "left" else -dx)

        k = (i // 2)
        sgn = 1 if (i % 2 == 0) else -1
        y_text = y[i] + (sgn * (k * 0.55) * dy)

        for _ in range(10):
            if all(abs(y_text - yy) >= 0.55 * dy for yy in placed_y):
                break
            y_text += 0.65 * dy
        placed_y.append(y_text)

        ax.text(
            x_text, y_text, str(names[i]),
            fontsize=11, fontweight="bold",
            color="black", ha=ha, va="center",
            zorder=50, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88)
        )

def _choose_genes_near_targets(y, x, idx_plot, targets, x_thr, y_thr):
    """
    Select one gene per target based on minimal |y - target|,
    excluding outliers (x>x_thr or y>y_thr).
    """
    candidates = []
    for g in idx_plot:
        if (x[g] <= x_thr) and (y[g] <= y_thr):
            candidates.append(g)
    candidates = np.array(candidates, dtype=int)
    if candidates.size == 0:
        return np.array([], dtype=int)

    selected = []
    used = set()
    for t in targets:
        dist = np.abs(y[candidates] - float(t))
        order = np.argsort(dist)
        pick = None
        for j in order:
            g = int(candidates[j])
            if g not in used:
                pick = g
                break
        if pick is not None:
            selected.append(pick)
            used.add(pick)

    return np.array(selected, dtype=int)


# ===================== plotting =====================
def plot_violin(genes, pred_log, true_log, top_idx, out_path, title, cfg: Config):
    pc, _ = compute_clone_means(pred_log, None)
    tc, _ = compute_clone_means(true_log, None)

    dfs = []
    for gi in top_idx:
        gname = str(genes[gi])
        dfs.append(pd.DataFrame({"E": tc[:, gi], "Type": "Real", "Gene": gname}))
        dfs.append(pd.DataFrame({"E": pc[:, gi], "Type": "Pred", "Gene": gname}))
    df = pd.concat(dfs, ignore_index=True)

    yl = _robust_ylim(df["E"].values, cfg.violin_ylim_percentiles[0], cfg.violin_ylim_percentiles[1], cfg.violin_ylim_margin)
    if yl is not None:
        lo, hi = yl
        span = max(hi - lo, 1e-6)
        lo = lo - span * cfg.violin_bottom_extra_ratio
        hi = hi + span * cfg.violin_top_headroom_ratio

        # FIX: include 0 on y-axis (避免 0 被裁掉)
        if getattr(cfg, "violin_bottom_floor", None) is not None:
            lo = min(lo, float(cfg.violin_bottom_floor))

        yl = (lo, hi)

    fig_w = max(10.0, 1.12 * len(top_idx) + 3.0)
    fig_h = 5.6
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    sns.violinplot(
        x="Gene", y="E", hue="Type", data=df,
        hue_order=["Real", "Pred"],
        palette=cfg.palette,
        inner=cfg.violin_inner,
        linewidth=cfg.violin_linewidth,
        cut=cfg.violin_cut,          # cut（切割 cut）
        scale=cfg.violin_scale,
        ax=ax
    )

    if yl is not None:
        ax.set_ylim(yl[0], yl[1])

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Expression (log1p)")
    _style_ax(ax, set_y_locator=True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], frameon=True, title="")

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()

def plot_scatter(genes, pred_log, true_log, out_path, title, cfg: Config, exclude_set, point_color: str):
    pc, _ = compute_clone_means(pred_log, None)
    tc, _ = compute_clone_means(true_log, None)

    x = pc.mean(axis=0)
    y = tc.mean(axis=0)

    # only plot x>0 & y>0
    if cfg.scatter_positive_only:
        mask = (x > 0) & (y > 0)
    else:
        mask = np.ones_like(x, dtype=bool)

    idx_plot = np.where(mask)[0].astype(int)
    x_plot = x[idx_plot]
    y_plot = y[idx_plot]

    plt.figure(figsize=(7.2, 7.2))
    ax = plt.gca()

    ax.scatter(
        x_plot, y_plot,
        s=cfg.scatter_point_size,
        alpha=cfg.scatter_point_alpha,
        c=point_color,
        edgecolors="none"
    )

    if x_plot.size > 0:
        step = float(cfg.scatter_tick_step)
        mx = float(max(x_plot.max(), y_plot.max()))
        mx = mx * (1.0 + cfg.scatter_axis_margin_ratio)
        mx = np.ceil(mx / step) * step

        ax.plot([0, mx], [0, mx], "--", color=cfg.diag_line_color, alpha=0.75, lw=1.5)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_aspect("equal", adjustable="box")

        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_locator(MultipleLocator(step))

    # choose 5 label genes by REAL mean targets (exclude Penk + extreme outliers)
    if x_plot.size > 0:
        x_thr = float(np.percentile(x_plot, cfg.scatter_label_outlier_percentile))
        y_thr = float(np.percentile(y_plot, cfg.scatter_label_outlier_percentile))

        idx_plot_noex = np.array([g for g in idx_plot if str(genes[g]).lower() not in exclude_set], dtype=int)

        hi = _choose_genes_near_targets(
            y=y, x=x,
            idx_plot=idx_plot_noex,
            targets=cfg.scatter_label_targets,
            x_thr=x_thr, y_thr=y_thr
        )

        if hi.size > 0:
            ax.scatter(
                x[hi], y[hi],
                s=cfg.highlight_point_size,
                c=point_color,
                edgecolors="black",
                linewidth=0.9,
                zorder=10
            )
            _annotate_labels(ax, x[hi], y[hi], [genes[g] for g in hi], cfg)

    r, r2 = safe_pearsonr(x_plot, y_plot)
    ax.set_title(f"{title}\nGene-mean r={r:.3f}, R²={r2:.3f} (x>0,y>0)", fontsize=14)
    ax.set_xlabel("Pred mean (log1p)")
    ax.set_ylabel("Real mean (log1p)")

    _style_ax(ax, set_y_locator=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()


# ===================== main =====================
def main():
    cfg = Config()
    sns.set_theme(style=cfg.seaborn_style)

    tasks = cfg.tasks if cfg.tasks is not None else list_tasks(cfg.reg_out_dir)
    if not tasks:
        raise FileNotFoundError(f"No tasks found under {cfg.reg_out_dir}.")
    print("[Tasks]", tasks)

    exclude_set = set([g.lower() for g in cfg.exclude_genes])

    for task in tasks:
        tdir = os.path.join(cfg.reg_out_dir, task)
        npz_path = os.path.join(tdir, "test_outputs.npz")
        if not os.path.exists(npz_path):
            print(f"[Skip] missing {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        pred = np.asarray(data["pred_log"], dtype=np.float32)  # log1p
        true = np.asarray(data["true_log"], dtype=np.float32)  # log1p
        genes = _decode_bytes(data["gene_names"])
        label = _decode_bytes(data["label"])

        out_dir = os.path.join(tdir, cfg.out_subdir)
        ensure_dir(out_dir)

        print(f"\n[{task}] N={pred.shape[0]} label_counts={pd.Series(label.astype(str)).value_counts().to_dict()}")

        manual_violin = {
            "Failed": cfg.violin_genes_fail,
            "Reprogrammed": cfg.violin_genes_repr
        }

        for ct in cfg.keep_labels:
            m = (label == str(ct))
            n_ct = int(m.sum())
            if n_ct == 0:
                continue

            p_sub = pred[m]
            t_sub = true[m]
            G = p_sub.shape[1]

            # per-gene r/R2 across cells (log1p) for table + auto fallback
            r = np.full((G,), np.nan, dtype=np.float32)
            r2 = np.full((G,), np.nan, dtype=np.float32)
            if n_ct >= 2:
                for g in range(G):
                    rg, r2g = safe_pearsonr(p_sub[:, g], t_sub[:, g])
                    r[g] = rg
                    r2[g] = r2g

            # --- violin genes: manual first ---
            top_idx, missing, dropped = _genes_to_indices(genes, manual_violin.get(ct, []), exclude_set)
            if dropped:
                print(f"[{task}][{ct}] dropped excluded genes from violin: {dropped}")
            if missing:
                print(f"[{task}][{ct}] missing genes (not in gene_names) for violin: {missing}")

            if len(top_idx) == 0:
                cand = np.array([i for i in range(G) if str(genes[i]).lower() not in exclude_set], dtype=int)
                if n_ct >= 2:
                    if cfg.require_positive_r_for_top:
                        cand = cand[np.isfinite(r[cand]) & (r[cand] >= cfg.min_pos_r)]
                    if cand.size < cfg.violin_top_k:
                        cand = np.array([i for i in range(G) if str(genes[i]).lower() not in exclude_set], dtype=int)
                    r2_rank = np.nan_to_num(r2[cand], nan=-1e9)
                    top_idx = cand[np.argsort(-r2_rank)[:cfg.violin_top_k]]
                else:
                    t0 = t_sub[0].copy()
                    for i in range(G):
                        if str(genes[i]).lower() in exclude_set:
                            t0[i] = -1e9
                    top_idx = np.argsort(-t0)[:cfg.violin_top_k]

            # table for violin genes
            mean_true = t_sub.mean(axis=0).astype(np.float32)
            mean_pred = p_sub.mean(axis=0).astype(np.float32)
            std_true = t_sub.std(axis=0).astype(np.float32)
            std_pred = p_sub.std(axis=0).astype(np.float32)

            tab = pd.DataFrame({
                "gene": [str(genes[g]) for g in top_idx],
                "r": [float(r[g]) if np.isfinite(r[g]) else np.nan for g in top_idx],
                "R2": [float(r2[g]) if np.isfinite(r2[g]) else np.nan for g in top_idx],
                "mean_true": [float(mean_true[g]) for g in top_idx],
                "mean_pred": [float(mean_pred[g]) for g in top_idx],
                "std_true": [float(std_true[g]) for g in top_idx],
                "std_pred": [float(std_pred[g]) for g in top_idx],
            })
            print(f"\n[{task}][{ct}] Gene table (log1p) | N={n_ct}")
            print(tab.to_string(index=False))

            title = f"{task} | {ct} | N={n_ct}"

            # color per label (two plots -> two colors)
            point_color = cfg.scatter_colors_by_label.get(str(ct), cfg.scatter_fallback_color)

            # scatter (per-ct)
            plot_scatter(
                genes, p_sub, t_sub,
                out_path=os.path.join(out_dir, f"scatter_{ct}.png"),
                title=title,
                cfg=cfg,
                exclude_set=exclude_set,
                point_color=point_color
            )

            # violin
            plot_violin(
                genes, p_sub, t_sub, top_idx,
                out_path=os.path.join(out_dir, f"violin_{ct}.png"),
                title=title,
                cfg=cfg
            )

        print(f"[OK] plots saved under: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
