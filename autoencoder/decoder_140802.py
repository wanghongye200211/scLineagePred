# -*- coding: utf-8 -*-
"""
decoder_140802.py  (LOG1P + v1 plot style + auto scatter labels)

Input (per task):
- <reg_out_dir>/<task>/test_outputs.npz
  required keys: pred_log, true_log, clone_id, label, gene_names

Output:
- <reg_out_dir>/<task>/<out_subdir>/scatter_<celltype>.png
- <reg_out_dir>/<task>/<out_subdir>/violin_<celltype>.png

Scatter:
- Two plots (one per celltype), each has its own color (EDIT AT TOP).
"""

# =========================================================
# ===============  COLOR SETTINGS (EDIT HERE) ===============
# =========================================================
# Scatter colors: two plots -> two colors (你只改这里就行)
SCATTER_COLORS_BY_CELLTYPE = {
    "Neutrophil": "#4C78A8",  # blue
    "Monocyte":   "#F58518",  # orange
}
SCATTER_FALLBACK_COLOR = "#9ecae1"     # if celltype key not found
SCATTER_DIAG_LINE_COLOR = "#6E6E6E"    # diagonal line color


import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

import seaborn as sns
from scipy.stats import pearsonr


# ===================== Config =====================
class Config:
    # --- keep original paths / tasks ---
    reg_out_dir = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE140802"
    tasks = ["Reg_D4_from_D2_D6", "Reg_D6_from_D2_D4"]
    keep_types = ("Neutrophil", "Monocyte")
    out_subdir = "plots"

    # ---------- violin genes (manual first; fallback keeps selection logic) ----------
    violin_genes_neutrophil = ["Axl", "Bcl6", "Bmf"]
    violin_genes_monocyte = ["Axl", "Bcl6"]

    violin_top_k = 8
    require_positive_r_for_top = True
    min_pos_r = 0.0

    # sign-flip diagnostics
    neg_r_threshold = -0.7
    high_r2_threshold = 0.5

    # ---------- scatter labeling by REAL mean targets ----------
    scatter_label_targets = [0.5, 1.0, 1.5, 2.0, 2.5]
    scatter_label_outlier_percentile = 99.5

    # ---------- style ----------
    dpi = 300
    seaborn_style = "white"

    _p_hex = sns.color_palette("Pastel1", 8).as_hex()
    palette = {"Real": _p_hex[1], "Pred": _p_hex[0]}  # Real=light blue, Pred=light pink

    # scatter (per celltype colors)
    scatter_colors_by_celltype = SCATTER_COLORS_BY_CELLTYPE
    scatter_fallback_color = SCATTER_FALLBACK_COLOR
    diag_line_color = SCATTER_DIAG_LINE_COLOR

    scatter_point_size = 60
    scatter_point_alpha = 0.72
    highlight_point_size = 220

    scatter_positive_only = True
    scatter_axis_margin_ratio = 0.04
    scatter_tick_step = 0.8

    label_dx_ratio = 0.025
    label_dy_ratio = 0.090

    # violin
    violin_inner = "box"
    violin_linewidth = 1.0
    violin_cut = 0
    violin_scale = "width"

    violin_ylim_percentiles = (2.0, 98.0)
    violin_ylim_margin = 0.06
    violin_bottom_extra_ratio = 0.08
    violin_top_headroom_ratio = 0.40  # more blank space on top

    # violin KDE tuning
    violin_bw_adjust = 1.0
    violin_gridsize = 200
    violin_fill_to_k = True

    # per-gene filtering for violin (ignore cells with true==0 for that gene)
    violin_pos_min_cells = 50
    violin_pos_min_frac = 0.02

    # exclude (always)
    exclude_genes = ("Penk", "penk")


# ===================== helpers =====================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _decode_bytes(arr):
    if arr is None:
        return None
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
        return np.array([(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)).strip()
                         for x in arr], dtype=object)
    return np.array(arr, dtype=object)

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
    return float(r), float(r*r)

def compute_clone_means(expr, clone_ids):
    clone_ids = np.asarray(clone_ids)
    uniq, inv = np.unique(clone_ids, return_inverse=True)
    C, G = len(uniq), expr.shape[1]
    sums = np.zeros((C, G), dtype=np.float64)
    np.add.at(sums, inv, expr.astype(np.float64))
    cnt = np.bincount(inv).astype(np.float64)
    return (sums / np.maximum(cnt, 1.0)[:, None]).astype(np.float32), uniq

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

    # y-axis: left only, ticks outside, not dense
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
def plot_violin(genes, pred_log, true_log, clone_ids, top_idx, out_path, title, cfg: Config):
    pc = np.asarray(pred_log, dtype=np.float32)
    tc = np.asarray(true_log, dtype=np.float32)

    dfs = []
    for gi in top_idx:
        gname = str(genes[gi])

        t = tc[:, gi].astype(np.float64)
        p = pc[:, gi].astype(np.float64)

        mask = np.isfinite(t) & np.isfinite(p) & (t > 0.0)
        if int(mask.sum()) < 2:
            mask = np.isfinite(t) & np.isfinite(p)

        dfs.append(pd.DataFrame({"E": t[mask], "Type": "Real", "Gene": gname}))
        dfs.append(pd.DataFrame({"E": p[mask], "Type": "Pred", "Gene": gname}))

    df = pd.concat(dfs, ignore_index=True)

    yl = _robust_ylim(df["E"].values, cfg.violin_ylim_percentiles[0], cfg.violin_ylim_percentiles[1], cfg.violin_ylim_margin)
    if yl is not None:
        lo, hi = yl
        span = max(hi - lo, 1e-6)
        lo = lo - span * cfg.violin_bottom_extra_ratio
        hi = hi + span * cfg.violin_top_headroom_ratio
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
        cut=cfg.violin_cut,
        scale=cfg.violin_scale,
        bw_adjust=cfg.violin_bw_adjust,
        gridsize=cfg.violin_gridsize,
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

def plot_scatter(genes, pred_log, true_log, clone_ids, out_path, title, cfg: Config, exclude_set, point_color: str):
    pc, _ = compute_clone_means(pred_log, clone_ids)
    tc, _ = compute_clone_means(true_log, clone_ids)

    x = pc.mean(axis=0)  # gene mean across clones
    y = tc.mean(axis=0)

    # only plot x>0 & y>0
    if cfg.scatter_positive_only:
        mask = (x > 0) & (y > 0)
    else:
        mask = np.ones_like(x, dtype=bool)

    # permanently exclude genes from plotting (Penk)
    excl_mask = np.array([str(g).lower() not in exclude_set for g in genes], dtype=bool)
    mask = mask & excl_mask

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

    # auto choose 5 label genes near REAL-mean targets (exclude outliers)
    if x_plot.size > 0:
        x_thr = float(np.percentile(x_plot, cfg.scatter_label_outlier_percentile))
        y_thr = float(np.percentile(y_plot, cfg.scatter_label_outlier_percentile))

        hi = _choose_genes_near_targets(
            y=y, x=x,
            idx_plot=idx_plot,
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

    exclude_set = set([g.lower() for g in cfg.exclude_genes])

    manual_violin = {
        "Neutrophil": cfg.violin_genes_neutrophil,
        "Monocyte": cfg.violin_genes_monocyte,
    }

    for task in cfg.tasks:
        tdir = os.path.join(cfg.reg_out_dir, task)
        npz_path = os.path.join(tdir, "test_outputs.npz")
        if not os.path.exists(npz_path):
            print(f"[Skip] missing {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        pred = np.asarray(data["pred_log"], dtype=np.float32)
        true = np.asarray(data["true_log"], dtype=np.float32)
        clone_id = data["clone_id"]
        label = _decode_bytes(data["label"]).astype(str)
        genes = _decode_bytes(data["gene_names"]).astype(object)

        out_dir = os.path.join(tdir, cfg.out_subdir)
        ensure_dir(out_dir)

        print(f"\n[{task}] label_counts={pd.Series(label.astype(str)).value_counts().to_dict()}")

        for ct in cfg.keep_types:
            m = (label == str(ct))
            if int(m.sum()) < 10:
                continue

            p_sub = pred[m]
            t_sub = true[m]
            c_sub = clone_id[m]

            # clone means for gene ranking
            pc, _ = compute_clone_means(p_sub, c_sub)
            tc, _ = compute_clone_means(t_sub, c_sub)

            G = pc.shape[1]
            r = np.full((G,), np.nan, dtype=np.float32)
            r2 = np.full((G,), np.nan, dtype=np.float32)
            for g in range(G):
                rg, r2g = safe_pearsonr(pc[:, g], tc[:, g])
                r[g] = rg
                r2[g] = r2g

            # cell-level metrics on expressed cells only (true>0), used for violin ranking / diagnostics
            n_cells_ct = int(t_sub.shape[0])
            pos_counts = (t_sub > 0.0).sum(axis=0).astype(np.int32)
            pos_fracs = (pos_counts / max(n_cells_ct, 1)).astype(np.float32)

            r_pos = np.full((G,), np.nan, dtype=np.float32)
            r2_pos = np.full((G,), np.nan, dtype=np.float32)
            for g in range(G):
                if pos_counts[g] >= cfg.violin_pos_min_cells:
                    mask_g = (t_sub[:, g] > 0.0)
                    rg, r2g = safe_pearsonr(p_sub[mask_g, g], t_sub[mask_g, g])
                    r_pos[g] = rg
                    r2_pos[g] = r2g

            # print sign-flip candidates
            bad = np.where((np.isfinite(r)) & (np.isfinite(r2)) & (r < cfg.neg_r_threshold) & (r2 >= cfg.high_r2_threshold))[0]
            if bad.size > 0:
                bad_sorted = bad[np.argsort(-r2[bad])][:10]
                print(f"\n[WARN][{task}][{ct}] potential sign-flip genes (r<{cfg.neg_r_threshold}, R^2>={cfg.high_r2_threshold}):")
                for g in bad_sorted:
                    print(f"  {genes[g]}  r={float(r[g]):.3f}  R^2={float(r2[g]):.3f}")

            # violin genes: manual first; optionally fill to top_k by R2
            top_idx, missing, dropped = _genes_to_indices(genes, manual_violin.get(ct, []), exclude_set)
            if dropped:
                print(f"[{task}][{ct}] dropped excluded genes from violin: {dropped}")
            if missing:
                print(f"[{task}][{ct}] missing genes (not in gene_names) for violin: {missing}")

            cand = np.array([i for i in range(G) if str(genes[i]).lower() not in exclude_set], dtype=int)

            ok_expr = (pos_counts[cand] >= cfg.violin_pos_min_cells) & (pos_fracs[cand] >= cfg.violin_pos_min_frac)
            cand = cand[ok_expr]

            if cfg.require_positive_r_for_top:
                cand = cand[np.isfinite(r_pos[cand]) & (r_pos[cand] >= cfg.min_pos_r)]

            if cand.size == 0:
                cand = np.array([i for i in range(G) if str(genes[i]).lower() not in exclude_set], dtype=int)
                ok_expr = (pos_counts[cand] >= cfg.violin_pos_min_cells) & (pos_fracs[cand] >= cfg.violin_pos_min_frac)
                cand = cand[ok_expr]

            if cand.size == 0:
                cand = np.array([i for i in range(G) if str(genes[i]).lower() not in exclude_set], dtype=int)
                r2_rank = np.nan_to_num(r2[cand], nan=-1e9)
                ranked = cand[np.argsort(-r2_rank)]
            else:
                r2_rank = np.nan_to_num(r2_pos[cand], nan=-1e9)
                ranked = cand[np.argsort(-r2_rank)]

            if len(top_idx) == 0:
                top_idx = list(ranked[:cfg.violin_top_k])
            else:
                if cfg.violin_fill_to_k and (len(top_idx) < cfg.violin_top_k):
                    need = cfg.violin_top_k - len(top_idx)
                    extra = [g for g in ranked if g not in set(top_idx)]
                    top_idx = list(top_idx) + extra[:need]

            print(f"\n[{task}][{ct}] Violin genes (metrics on expressed cells only, plus clone-mean reference):")
            for g in top_idx:
                rr_clone = float(r[g]) if np.isfinite(r[g]) else np.nan
                rr2_clone = float(r2[g]) if np.isfinite(r2[g]) else np.nan
                rr_pos_v = float(r_pos[g]) if np.isfinite(r_pos[g]) else np.nan
                rr2_pos_v = float(r2_pos[g]) if np.isfinite(r2_pos[g]) else np.nan
                npos = int(pos_counts[g])
                pfrac = float(pos_fracs[g]) if np.isfinite(pos_fracs[g]) else np.nan
                print(f"  {genes[g]}  n_pos={npos} ({pfrac*100:.1f}%) | r_pos={rr_pos_v:.3f} R2_pos={rr2_pos_v:.3f} | r_clone={rr_clone:.3f} R2_clone={rr2_clone:.3f}")

            title = f"{task} | {ct}"

            # ---- color per celltype (two scatters -> two colors) ----
            point_color = cfg.scatter_colors_by_celltype.get(str(ct), cfg.scatter_fallback_color)

            plot_scatter(
                genes, p_sub, t_sub, c_sub,
                out_path=os.path.join(out_dir, f"scatter_{ct}.png"),
                title=title,
                cfg=cfg,
                exclude_set=exclude_set,
                point_color=point_color
            )
            plot_violin(
                genes, p_sub, t_sub, c_sub, top_idx,
                out_path=os.path.join(out_dir, f"violin_{ct}.png"),
                title=title,
                cfg=cfg
            )

        print(f"[OK] plots saved under: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
