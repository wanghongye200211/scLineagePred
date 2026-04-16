# -*- coding: utf-8 -*-
"""
GSE140802 Day2 official-method comparison (CellRank/WOT/CoSpar) vs scLineagetracer.

Pipeline:
- scLineagetracer: from Day2 cache (same test split)
- CellRank: ConnectivityKernel + PseudotimeKernel + GPCCA terminal fate probabilities
- WOT: OTModel transport map from day2 -> day6
- CoSpar: infer_Tmap_from_state_info_alone + compute_fate_probability_map

Outputs:
- classification/GSE140802_Final_v7/roc_click/benchmark_day2_official
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss, precision_recall_curve, confusion_matrix


# runtime env for scanpy/cellrank/cospar on this machine
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import scanpy as sc
import cellrank as cr
import wot
import cospar as cs


GREY = "#444444"
LINE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_fig(fig, out_no_ext):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _style_axes_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)
    m1 = fpr == 1.0
    if np.any(m1):
        t1 = float(np.max(tpr[m1]))
        keep = ~m1
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]
    return fpr, tpr


def _safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return np.nan
    return float(num) / den


def _auprc_trapz(y_true, p):
    precision, recall, _ = precision_recall_curve(y_true, p)
    return float(auc(recall[::-1], precision[::-1]))


def eval_binary(y_true, p_pos):
    p = np.clip(np.asarray(p_pos, dtype=np.float64), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(np.int64)
    fpr, tpr, _ = roc_curve(y_true, p)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    auroc = float(auc(fpr, tpr))
    acc = float(accuracy_score(y_true, pred))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = np.nan
    return {
        "AUC": auroc,
        "AUROC": auroc,
        "Accuracy@0.5": acc,
        "Accuracy": acc,
        "ACC": acc,
        "AUPRC": float(_auprc_trapz(y_true, p)),
        "F1": float(f1) if np.isfinite(f1) else np.nan,
        "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "Precision": float(precision) if np.isfinite(precision) else np.nan,
        "Recall": float(recall) if np.isfinite(recall) else np.nan,
        "LogLoss": float(log_loss(y_true, p)),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_dual(curves, out_dir):
    base = os.path.join(out_dir, "ROC_Comparison_Day2_Official")
    order = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
    color = {m: LINE_COLORS[i] for i, m in enumerate(order)}

    def draw(ax, clean):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=GREY, alpha=0.55)
        for m in order:
            if m not in curves:
                continue
            c = curves[m]
            ax.plot(c["fpr"], c["tpr"], lw=2.7, color=color[m], label=f"{m} (AUC={c['AUC']:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        if clean:
            _style_axes_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.set_title("Selected ROC Curves (Day2 Official)", fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(False)

    fig1, ax1 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax1, clean=False)
    fig1.tight_layout()
    save_fig(fig1, base + "_full")

    fig2, ax2 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax2, clean=True)
    fig2.tight_layout()
    save_fig(fig2, base + "_clean")
    return base


def prepare_subset(adata, source_ids, time_col, state_col, source_time, target_time, pos_label, neg_label, mid_time=4, mid_cap=4000):
    time = adata.obs[time_col].astype(int).to_numpy()
    state = adata.obs[state_col].astype(str).to_numpy()

    source_set = set(int(x) for x in source_ids)
    source_mask = np.array([int(x) in source_set for x in adata.obs_names], dtype=bool)
    target_mask = (time == target_time) & np.isin(state, [pos_label, neg_label])
    mid_idx = np.where(time == mid_time)[0]
    if len(mid_idx) > mid_cap:
        rng = np.random.default_rng(2026)
        mid_idx = rng.choice(mid_idx, size=mid_cap, replace=False)
    mid_mask = np.zeros(adata.n_obs, dtype=bool)
    mid_mask[mid_idx] = True

    keep = source_mask | target_mask | mid_mask
    sub = adata[keep].copy()
    sub.obs_names = pd.Index([str(x) for x in sub.obs_names])
    return sub


def extract_source_probs_by_ids(prob_by_cell, source_ids):
    out = np.array([prob_by_cell.get(int(cid), np.nan) for cid in source_ids], dtype=np.float64)
    if np.isnan(out).any():
        miss = int(np.isnan(out).sum())
        raise RuntimeError(f"Missing probabilities for {miss} source cells.")
    return out


def run_cellrank_binary(adata_sub, time_col, state_col, source_time, target_time, pos_label, neg_label):
    work = adata_sub.copy()
    sc.pp.pca(work, n_comps=min(50, work.n_obs - 1, work.n_vars - 1))
    sc.pp.neighbors(work, n_neighbors=min(30, work.n_obs - 1), use_rep="X_pca")

    ck = cr.kernels.ConnectivityKernel(work).compute_transition_matrix()
    pk = cr.kernels.PseudotimeKernel(work, time_key=time_col).compute_transition_matrix(
        threshold_scheme="hard",
        n_jobs=1,
        backend="threading",
        show_progress_bar=False,
    )
    kernel = 0.5 * ck + 0.5 * pk
    g = cr.estimators.GPCCA(kernel)

    state = work.obs[state_col].astype(str).to_numpy()
    time = work.obs[time_col].astype(int).to_numpy()
    term_pos = list(work.obs_names[(time == target_time) & (state == pos_label)])
    term_neg = list(work.obs_names[(time == target_time) & (state == neg_label)])
    g.set_terminal_states({pos_label: term_pos, neg_label: term_neg})
    g.compute_fate_probabilities(
        solver="direct",
        use_petsc=False,
        n_jobs=1,
        backend="threading",
        show_progress_bar=False,
    )

    fp = g.fate_probabilities
    names = list(fp.names)
    pos_idx = names.index(pos_label)
    mat = np.asarray(fp.X)
    p_pos = mat[:, pos_idx].astype(np.float64)
    cell_ids = np.array(work.obs_names, dtype=str).astype(int)
    return {int(c): float(p) for c, p in zip(cell_ids, p_pos)}


def run_wot_binary(adata_sub, time_col, state_col, source_time, target_time, pos_label):
    work = adata_sub.copy()
    work.obs["day"] = work.obs[time_col].astype(float)
    model = wot.ot.OTModel(work, day_field="day")
    tm = model.compute_transport_map(float(source_time), float(target_time))
    M = np.asarray(tm.X, dtype=np.float64)
    target_labels = work.obs.loc[tm.var_names, state_col].astype(str).to_numpy()
    pos_mask = target_labels == pos_label
    p = M[:, pos_mask].sum(axis=1) / (M.sum(axis=1) + 1e-12)
    source_ids = np.array(tm.obs_names, dtype=str).astype(int)
    return {int(c): float(x) for c, x in zip(source_ids, p)}


def run_cospar_binary(adata_sub, time_col, state_col, source_time, target_time, pos_label, neg_label, out_dir):
    work = adata_sub.copy()
    work.obs[time_col] = work.obs[time_col].astype(str)
    sc.pp.pca(work, n_comps=min(30, work.n_obs - 1, work.n_vars - 1))

    cs.settings.data_path = out_dir
    cs.settings.figure_path = out_dir
    cs.settings.verbosity = 1

    ac = cs.pp.initialize_adata_object(
        adata=None,
        X_state=work.X,
        X_pca=work.obsm["X_pca"],
        cell_names=np.array(work.obs_names),
        gene_names=np.array(work.var_names),
        time_info=work.obs[time_col].to_numpy(),
        state_info=work.obs[state_col].astype(str).to_numpy(),
        data_des="gse140802_day2_official",
    )
    ac = cs.tmap.infer_Tmap_from_state_info_alone(
        ac,
        initial_time_points=[str(source_time)],
        later_time_point=str(target_time),
        max_iter_N=[1, 2],
        compute_new=True,
        CoSpar_KNN=20,
        smooth_array=[15, 10, 5],
    )
    cs.tl.compute_fate_probability_map(
        ac,
        selected_fates=[pos_label, neg_label],
        used_Tmap="transition_map",
        map_backward=True,
        method="norm-sum",
        fate_count=True,
    )
    key = f"fate_map_transition_map_{pos_label}"
    vals = np.asarray(ac.obs[key], dtype=np.float64)
    src_mask = ac.obs["time_info"].astype(str).to_numpy() == str(source_time)
    source_ids = np.array(ac.obs_names[src_mask], dtype=str).astype(int)
    p = vals[src_mask]
    return {int(c): float(x) for c, x in zip(source_ids, p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", default="/Users/wanghongye/python/scLineagetracer/GSE140802/preprocess_final/processed_norm_log_hvg1000.h5ad")
    parser.add_argument("--index_csv", default="/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_DeepLineage_index.csv")
    parser.add_argument("--day2_cache", default="/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/saved_models/Day2_Only_TestCache_s42.npz")
    parser.add_argument("--out_root", default="/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click")
    parser.add_argument("--run_cospar", type=int, default=0, help="1 to run official CoSpar (can be slow)")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_root, "benchmark_day2_official")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "cospar_data"))

    if not os.path.isfile(args.day2_cache):
        raise FileNotFoundError(f"Missing day2 cache: {args.day2_cache}")

    cache = np.load(args.day2_cache, allow_pickle=True)
    te_idx = cache["te_idx"].astype(np.int64)
    y_true = cache["y_true"].astype(np.int64)
    p_sc = np.asarray(cache["p_stack"], dtype=np.float64)

    idx_df = pd.read_csv(args.index_csv)
    idx_bin = idx_df[idx_df["label_str"].isin(["Monocyte", "Neutrophil"])].reset_index(drop=True)
    te_rows = idx_bin.iloc[te_idx]
    source_ids = te_rows["idx_t0"].astype(int).to_numpy()

    adata = ad.read_h5ad(args.h5ad)
    adata.obs["time_info"] = adata.obs["time_info"].astype(int)
    adata.obs["state_info"] = adata.obs["state_info"].astype(str)
    adata.obs_names = pd.Index([str(x) for x in adata.obs_names])

    adata_sub = prepare_subset(
        adata=adata,
        source_ids=source_ids,
        time_col="time_info",
        state_col="state_info",
        source_time=2,
        target_time=6,
        pos_label="Monocyte",
        neg_label="Neutrophil",
        mid_time=4,
        mid_cap=4000,
    )

    curves = {"scLineagetracer": eval_binary(y_true, p_sc)}
    method_err = []

    # CellRank
    try:
        prob_cellrank = run_cellrank_binary(
            adata_sub, "time_info", "state_info", 2, 6, "Monocyte", "Neutrophil"
        )
        p = extract_source_probs_by_ids(prob_cellrank, source_ids)
        curves["CellRank"] = eval_binary(y_true, p)
    except Exception as e:
        method_err.append(("CellRank", str(e)))

    # WOT
    try:
        prob_wot = run_wot_binary(
            adata_sub, "time_info", "state_info", 2, 6, "Monocyte"
        )
        p = extract_source_probs_by_ids(prob_wot, source_ids)
        curves["WOT"] = eval_binary(y_true, p)
    except Exception as e:
        method_err.append(("WOT", str(e)))

    # CoSpar (optional, slower)
    if int(args.run_cospar) == 1:
        try:
            prob_cospar = run_cospar_binary(
                adata_sub, "time_info", "state_info", 2, 6, "Monocyte", "Neutrophil", os.path.join(out_dir, "cospar_data")
            )
            p = extract_source_probs_by_ids(prob_cospar, source_ids)
            curves["CoSpar"] = eval_binary(y_true, p)
        except Exception as e:
            method_err.append(("CoSpar", str(e)))

    rows = []
    for m, c in curves.items():
        rows.append({
            "Setting": "Day2_Only",
            "Method": m,
            "AUC": c["AUC"],
            "AUROC": c["AUROC"],
            "Accuracy@0.5": c["Accuracy@0.5"],
            "Accuracy": c["Accuracy"],
            "ACC": c["ACC"],
            "AUPRC": c["AUPRC"],
            "F1": c["F1"],
            "Specificity": c["Specificity"],
            "Precision": c["Precision"],
            "Recall": c["Recall"],
            "LogLoss": c["LogLoss"],
            "N_test": int(len(y_true)),
            "Pipeline": "official",
        })
    summary = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    summary[["Method", "Accuracy@0.5"]].to_csv(os.path.join(out_dir, "metrics_accuracy_only.csv"), index=False)
    summary[["Method", "AUC"]].to_csv(os.path.join(out_dir, "metrics_auc_only.csv"), index=False)

    curve_rows = []
    for m, c in curves.items():
        for i in range(len(c["fpr"])):
            curve_rows.append({"Method": m, "point_id": i, "fpr": float(c["fpr"][i]), "tpr": float(c["tpr"][i])})
    pd.DataFrame(curve_rows).to_csv(os.path.join(out_dir, "roc_curve_points.csv"), index=False)

    np.savez(
        os.path.join(out_dir, "benchmark_probs.npz"),
        te_idx=te_idx,
        y_true=y_true,
        p_sclineagetracer=p_sc,
    )

    base = plot_dual(curves, out_dir)
    with open(os.path.join(out_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("Official method run log (GSE140802 Day2)\n")
        if method_err:
            f.write("Failed methods:\n")
            for m, e in method_err:
                f.write(f"- {m}: {e}\n")
        else:
            f.write("All methods succeeded.\n")

    print(f"[DONE] Output folder: {out_dir}")
    print(f"[DONE] ROC full: {base}_full.png/.pdf")
    print(f"[DONE] ROC clean: {base}_clean.png/.pdf")
    print(summary.to_string(index=False))
    if method_err:
        print("[WARN] Some methods failed:")
        for m, e in method_err:
            print(f"- {m}: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
