# -*- coding: utf-8 -*-
"""
decoder_114412.py (log1p only, seaborn-template style)
(Only Config paths/labels/gene lists are changed.)
- Follow the SAME method as decoder_175634.py:
  * read <reg_out_dir>/<task>/test_outputs.npz
  * use pred_log / true_log / gene_names / label
  * per-celltype plots: scatter(mean_pred vs mean_true) + violin(top genes by R2)
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from matplotlib.ticker import MaxNLocator


# ===================== Config =====================
class Config:
    # ===== paths =====
    # regression outputs root; each task should contain: <task>/test_outputs.npz
    reg_out_dir = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE114412"

    tasks = None  # None -> auto detect tasks under reg_out_dir

    # ===== cell types (labels) =====
    # Must match `label` stored in test_outputs.npz (usually sequence-level label_str).
    keep_labels = ("sc_beta", "sc_ec", "sc_alpha")

    out_subdir = "plots_log"

    # ---------- YOU EDIT HERE (manual gene choices) ----------
    # Leave empty [] to auto-select (recommended first run).
    # Gene names must match `gene_names` in test_outputs.npz.
    scatter_genes_sc_beta = []
    scatter_genes_sc_ec = []
    scatter_genes_sc_alpha = []

    violin_genes_sc_beta = []
    violin_genes_sc_ec = []
    violin_genes_sc_alpha = []

    violin_top_k = 8
    require_positive_r_for_top = True
    min_pos_r = 0.0

    dpi = 300
    seaborn_style = "white"

    _p_hex = sns.color_palette("Pastel1", 8).as_hex()
    palette = {"Real": _p_hex[1], "Pred": _p_hex[0]}

    violin_inner = "box"
    violin_linewidth = 1.0
    violin_cut = 2
    violin_scale = "width"

    violin_ylim_percentiles = (2.0, 98.0)
    violin_ylim_margin = 0.3

    overlay_points = True
    overlay_points_max_n = 2000
    overlay_point_size = 4.5
    overlay_point_alpha = 0.35

    scatter_point_size = 22
    scatter_point_alpha = 0.55
    diag_line_color = "#6E6E6E"

    scatter_color_under = palette["Real"]
    scatter_color_over = palette["Pred"]

    max_gene_labels_on_scatter = 6
    scatter_label_outlier_percentile = 99.5

    label_dx_ratio = 0.018
    label_dy_ratio = 0.060

    # Optional: drop genes you don't want to highlight
    exclude_genes = ("Penk", "penk")


# ===================== helpers =====================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

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
    # Keep original method: NO clone aggregation by default.
    return np.asarray(expr, dtype=np.float32), None

def _decode_bytes(arr):
    if arr is None:
        return None
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
        return np.array([(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)).strip()
                         for x in arr], dtype=object)
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

def _style_ax(ax):
    ax.grid(False)
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.6)

    ax.tick_params(axis="x", which="both", bottom=True, top=False, direction="out",
                   length=0, width=1.2, colors="black", pad=6)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="y", which="major", left=True, right=True,
                   labelleft=True, labelright=False, direction="out",
                   length=6, width=1.2, colors="black", pad=6)
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
    idx = []
    missing = []
    dropped = []
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

def _select_spread_indices(idxs, yvals, k):
    idxs = np.array(idxs, dtype=int)
    yvals = np.array(yvals, dtype=float)
    if idxs.size == 0:
        return idxs
    if idxs.size <= k:
        return idxs
    order = np.argsort(yvals)
    idxs_sorted = idxs[order]
    pos = np.linspace(0, idxs_sorted.size - 1, k).round().astype(int)
    return idxs_sorted[pos]

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

        ax.text(x_text, y_text, str(names[i]), fontsize=11, fontweight="bold",
                color="black", ha=ha, va="center")


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
        cut=cfg.violin_cut,
        scale=cfg.violin_scale,
        ax=ax
    )

    if cfg.overlay_points and (tc.shape[0] <= cfg.overlay_points_max_n):
        sns.stripplot(
            x="Gene", y="E", hue="Type", data=df,
            hue_order=["Real", "Pred"],
            palette=cfg.palette,
            dodge=True, jitter=0.20,
            size=cfg.overlay_point_size, alpha=cfg.overlay_point_alpha,
            linewidth=0, ax=ax
        )

    if yl is not None:
        ax.set_ylim(yl[0], yl[1])

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Expression (log1p)")
    _style_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], frameon=True, title="")

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi)
    plt.close()

def plot_scatter(genes, pred_log, true_log, highlight_idx, out_path, title, cfg: Config):
    pc, _ = compute_clone_means(pred_log, None)
    tc, _ = compute_clone_means(true_log, None)

    x = pc.mean(axis=0)
    y = tc.mean(axis=0)
    diff = x - y

    colors = np.where(diff > 0, cfg.scatter_color_over, cfg.scatter_color_under).astype(object)

    plt.figure(figsize=(7.2, 7.2))
    ax = plt.gca()
    ax.scatter(x, y, s=cfg.scatter_point_size, alpha=cfg.scatter_point_alpha,
               c=colors, edgecolors="none")

    mn = 0.0
    mx = float(max(x.max(), y.max()))
    mx = mx + max(0.06 * mx, 0.1)

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    ax.plot([mn, mx], [mn, mx], "--", color=cfg.diag_line_color, alpha=0.75, lw=1.5)
    ax.set_aspect("equal", adjustable="box")

    hi = np.array(highlight_idx, dtype=int)
    if hi.size > 0:
        x_thr = float(np.percentile(x, cfg.scatter_label_outlier_percentile))
        y_thr = float(np.percentile(y, cfg.scatter_label_outlier_percentile))

        hi_ok = [g for g in hi if (x[g] <= x_thr and y[g] <= y_thr)]
        hi_ok = np.array(hi_ok, dtype=int)

        if hi_ok.size > 0:
            k = min(cfg.max_gene_labels_on_scatter, hi_ok.size)
            hi_lab = _select_spread_indices(hi_ok, y[hi_ok], k)

            ax.scatter(x[hi_ok], y[hi_ok], s=120, c=colors[hi_ok],
                       edgecolors="black", linewidth=0.9, zorder=10)

            _annotate_labels(ax, x[hi_lab], y[hi_lab], [genes[g] for g in hi_lab], cfg)

    r, r2 = safe_pearsonr(x, y)
    ax.set_title(f"{title}\nGene-mean r={r:.3f}, R²={r2:.3f}", fontsize=14)
    ax.set_xlabel("Pred mean (log1p)")
    ax.set_ylabel("Real mean (log1p)")
    _style_ax(ax)

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

    excl = set([g.lower() for g in cfg.exclude_genes])

    manual_scatter = {
        "sc_beta": cfg.scatter_genes_sc_beta,
        "sc_ec": cfg.scatter_genes_sc_ec,
        "sc_alpha": cfg.scatter_genes_sc_alpha,
    }
    manual_violin = {
        "sc_beta": cfg.violin_genes_sc_beta,
        "sc_ec": cfg.violin_genes_sc_ec,
        "sc_alpha": cfg.violin_genes_sc_alpha,
    }

    for task in tasks:
        tdir = os.path.join(cfg.reg_out_dir, task)
        npz_path = os.path.join(tdir, "test_outputs.npz")
        if not os.path.exists(npz_path):
            print(f"[Skip] missing {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        pred = np.asarray(data["pred_log"], dtype=np.float32)
        true = np.asarray(data["true_log"], dtype=np.float32)
        genes = _decode_bytes(data["gene_names"])
        label = _decode_bytes(data["label"])

        out_dir = os.path.join(tdir, cfg.out_subdir)
        ensure_dir(out_dir)

        print(f"\n[{task}] N={pred.shape[0]} label_counts={pd.Series(label.astype(str)).value_counts().to_dict()}")

        for ct in cfg.keep_labels:
            m = (label == str(ct))
            n_ct = int(m.sum())
            if n_ct == 0:
                continue

            p_sub = pred[m]
            t_sub = true[m]
            G = p_sub.shape[1]

            r = np.full((G,), np.nan, dtype=np.float32)
            r2 = np.full((G,), np.nan, dtype=np.float32)
            if n_ct >= 2:
                for g in range(G):
                    rg, r2g = safe_pearsonr(p_sub[:, g], t_sub[:, g])
                    r[g] = rg
                    r2[g] = r2g

            # violin genes
            if len(_clean_gene_list(manual_violin.get(ct, []))) > 0:
                top_idx, missing, dropped = _genes_to_indices(genes, manual_violin[ct], excl)
                if dropped:
                    print(f"[{task}][{ct}] dropped excluded genes from violin: {dropped}")
                if missing:
                    print(f"[{task}][{ct}] missing genes for violin: {missing}")
                if len(top_idx) == 0:
                    top_idx = []
            else:
                top_idx = []

            if len(top_idx) == 0:
                cand = np.array([i for i in range(G) if str(genes[i]).lower() not in excl], dtype=int)
                if n_ct >= 2:
                    if cfg.require_positive_r_for_top:
                        cand = cand[np.isfinite(r[cand]) & (r[cand] >= cfg.min_pos_r)]
                    if cand.size < cfg.violin_top_k:
                        cand = np.array([i for i in range(G) if str(genes[i]).lower() not in excl], dtype=int)
                    r2_rank = np.nan_to_num(r2[cand], nan=-1e9)
                    top_idx = cand[np.argsort(-r2_rank)[:cfg.violin_top_k]]
                else:
                    t0 = t_sub[0].copy()
                    for i in range(G):
                        if str(genes[i]).lower() in excl:
                            t0[i] = -1e9
                    top_idx = np.argsort(-t0)[:cfg.violin_top_k]

            # scatter label genes
            if len(_clean_gene_list(manual_scatter.get(ct, []))) > 0:
                hi_idx, missing2, dropped2 = _genes_to_indices(genes, manual_scatter[ct], excl)
                if dropped2:
                    print(f"[{task}][{ct}] dropped excluded genes from scatter: {dropped2}")
                if missing2:
                    print(f"[{task}][{ct}] missing genes for scatter: {missing2}")
            else:
                hi_idx = list(top_idx)

            # print small table
            mean_true = t_sub.mean(axis=0).astype(np.float32)
            mean_pred = p_sub.mean(axis=0).astype(np.float32)
            tab = pd.DataFrame({
                "gene": [str(genes[g]) for g in top_idx],
                "r": [float(r[g]) if np.isfinite(r[g]) else np.nan for g in top_idx],
                "R2": [float(r2[g]) if np.isfinite(r2[g]) else np.nan for g in top_idx],
                "mean_true": [float(mean_true[g]) for g in top_idx],
                "mean_pred": [float(mean_pred[g]) for g in top_idx],
            })
            print(f"\n[{task}][{ct}] Gene table (log1p) | N={n_ct}")
            print(tab.to_string(index=False))

            title = f"{task} | {ct} | N={n_ct}"
            plot_scatter(genes, p_sub, t_sub, hi_idx,
                         out_path=os.path.join(out_dir, f"scatter_{ct}.png"),
                         title=title, cfg=cfg)
            plot_violin(genes, p_sub, t_sub, top_idx,
                        out_path=os.path.join(out_dir, f"violin_{ct}.png"),
                        title=title, cfg=cfg)

        print(f"[OK] plots saved under: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
