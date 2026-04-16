# -*- coding: utf-8 -*-
"""
GSE140802 official 4-method comparison (multi-metric grouped bar chart).

Default metrics:
  ACC, AUPRC

Input (preferred):
  classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus/<SETTING>/benchmark_probs.npz
    keys: y_true, p_scLineagetracer, p_CellRank, p_WOT, p_CoSpar

Output:
  classification/GSE140802_Final_v7/roc_click/official_methods_plot_v3/
    - Method_Compare_Metrics_<SETTING>_values.csv
    - Method_Compare_Metrics_<SETTING>_full.png/.pdf
    - Method_Compare_Metrics_<SETTING>_clean.png/.pdf
    - Method_Compare_MetricLines_<SETTING>_full.png/.pdf
    - Method_Compare_MetricLines_<SETTING>_clean.png/.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/official_methods_plot_v3"
DEFAULT_SETTING = "Day2_Only"
FIG_SIDE = 9.0

METHOD_ORDER = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
METHOD_COLORS = {
    # Match the legacy method-color palette used in ROC comparison figures.
    "scLineagetracer": "#C43C35",  # red
    "CellRank": "#4F81BD",         # blue
    "WOT": "#4E9A3E",              # green
    "CoSpar": "#8E6BB7",           # purple
}

DEFAULT_METRICS = ["ACC", "AUPRC"]
METRIC_ALIASES = {
    "acc": "ACC",
    "accuracy": "ACC",
    "accuracy@0.5": "ACC",
    "auroc": "AUROC",
    "auc": "AUROC",
    "roc_auc": "AUROC",
    "auprc": "AUPRC",
    "pr_auc": "AUPRC",
    "f1": "F1",
    "specificity": "Specificity",
    "tnr": "Specificity",
    "precision": "Precision",
    "ppv": "Precision",
    "recall": "Recall",
    "sensitivity": "Recall",
    "tpr": "Recall",
}
METRIC_LINE_COLORS = {
    # High-contrast palette for metric lines.
    "ACC": "#377EB8",          # blue
    "AUROC": "#E41A1C",        # red
    "AUPRC": "#4DAF4A",        # green
    "F1": "#984EA3",           # purple
    "Specificity": "#FF7F00",  # orange
    "Precision": "#A65628",    # brown
    "Recall": "#F781BF",       # pink
}
METRIC_LINE_MARKERS = {
    "ACC": "o",
    "AUROC": "s",
    "AUPRC": "^",
    "F1": "D",
    "Specificity": "P",
    "Precision": "X",
    "Recall": "v",
}
METRIC_LINE_STYLES = {
    "ACC": "-",
    "AUROC": "-",
    "AUPRC": "--",
    "F1": "-.",
    "Specificity": ":",
    "Precision": "--",
    "Recall": "-.",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def sanitize_key(s: str) -> str:
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def _safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return np.nan
    return float(num) / den


def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    fpr = np.clip(fpr, 0.0, 1.0)
    tpr = np.clip(tpr, 0.0, 1.0)
    m1 = fpr == 1.0
    if np.any(m1):
        t1 = float(np.max(tpr[m1]))
        keep = ~m1
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]
    return fpr, tpr


def _auprc_trapz(y_true, p):
    precision, recall, _ = precision_recall_curve(y_true, p)
    return float(auc(recall[::-1], precision[::-1]))


def compute_binary_metrics(y_true, p_pos, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(p_pos, dtype=np.float64), 1e-6, 1 - 1e-6)
    y_pred = (p >= float(threshold)).astype(np.int64)

    fpr, tpr, _ = roc_curve(y_true, p)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    auroc = float(auc(fpr, tpr))
    auprc = _auprc_trapz(y_true, p)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = np.nan

    acc = float(np.mean(y_pred == y_true))
    return {
        "ACC": acc,
        "AUROC": auroc,
        "AUPRC": float(auprc),
        "F1": float(f1) if np.isfinite(f1) else np.nan,
        "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "Precision": float(precision) if np.isfinite(precision) else np.nan,
        "Recall": float(recall) if np.isfinite(recall) else np.nan,
    }


def parse_metric_list(metric_str: str):
    if metric_str is None or metric_str.strip() == "":
        return list(DEFAULT_METRICS)
    out = []
    for raw in metric_str.split(","):
        k = raw.strip()
        if not k:
            continue
        canon = METRIC_ALIASES.get(k.lower())
        if canon is None:
            valid = ", ".join(sorted(set(METRIC_ALIASES.values())))
            raise ValueError(f"Unknown metric '{k}'. Valid metrics: {valid}")
        if canon not in out:
            out.append(canon)
    if not out:
        raise ValueError("No valid metrics selected.")
    return out


def load_benchmark_probs(benchmark_root: str, setting: str):
    npz_path = os.path.join(benchmark_root, setting, "benchmark_probs.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing benchmark_probs.npz: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)
    if "y_true" not in z:
        raise KeyError(f"Missing y_true in: {npz_path}")
    y_true = np.asarray(z["y_true"], dtype=np.int64)

    probs = {}
    for m in METHOD_ORDER:
        key = "p_" + sanitize_key(m)
        if key not in z:
            continue
        arr = np.asarray(z[key], dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] < 2:
                continue
            p = arr[:, 1]
        elif arr.ndim == 1:
            p = arr
        else:
            continue
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        if len(p) != len(y_true):
            raise ValueError(f"Length mismatch for {m}: len(p)={len(p)} != len(y_true)={len(y_true)}")
        probs[m] = p

    return npz_path, y_true, probs


def build_metrics_table(benchmark_root: str, setting: str, metric_order):
    npz_path, y_true, probs = load_benchmark_probs(benchmark_root, setting)

    rows = []
    for m in METHOD_ORDER:
        if m not in probs:
            row = {"Method": m, "N_test": int(len(y_true)), "BenchmarkFile": os.path.basename(npz_path)}
            for metric in metric_order:
                row[metric] = np.nan
            rows.append(row)
            continue

        stats = compute_binary_metrics(y_true, probs[m], threshold=0.5)
        row = {
            "Method": m,
            "N_test": int(len(y_true)),
            "BenchmarkFile": os.path.basename(npz_path),
            "Accuracy": float(stats["ACC"]),
            "ACC": float(stats["ACC"]),
            "AUC": float(stats["AUROC"]),
            "AUROC": float(stats["AUROC"]),
        }
        for metric in metric_order:
            row[metric] = float(stats[metric]) if np.isfinite(stats[metric]) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def draw_bar(df: pd.DataFrame, metric_order, out_no_ext: str, setting: str, clean: bool):
    methods = df["Method"].astype(str).tolist()
    n_methods = len(methods)
    n_metrics = len(metric_order)
    x = np.arange(n_methods, dtype=float)

    cluster_width = 0.66 if n_metrics <= 2 else 0.82
    width = cluster_width / max(1, n_metrics)
    center_shift = (n_metrics - 1) * 0.5

    fig, ax = plt.subplots(figsize=(FIG_SIDE, max(6.6, FIG_SIDE * 0.78)))

    hatches = ["", "//", "\\\\", "xx", "..", "++", "--"]
    metric_display = {
        "ACC": "Accuracy",
        "AUROC": "AUROC",
        "AUPRC": "AUPRC",
        "F1": "F1",
        "Specificity": "Specificity",
        "Precision": "Precision",
        "Recall": "Recall",
    }

    all_vals = []
    for k, metric in enumerate(metric_order):
        vals = df[metric].to_numpy(dtype=float) * 100.0
        pos = x + (k - center_shift) * width
        hatch = hatches[k % len(hatches)]
        alpha = 0.92 if hatch == "" else 0.42
        ax.bar(
            pos,
            vals,
            width=width * 0.94,
            color=[METHOD_COLORS.get(m, "#7F7F7F") for m in methods],
            edgecolor="#2B2B2B",
            linewidth=0.55,
            hatch=hatch,
            label=metric_display.get(metric, metric),
            alpha=alpha,
        )
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size > 0:
            all_vals.append(finite_vals)

    if all_vals:
        vv = np.concatenate(all_vals)
        y_min = max(0.0, float(np.floor(np.nanmin(vv) - 2.0)))
        y_max = min(100.0, float(np.ceil(np.nanmax(vv) + 2.0)))
        if y_max <= y_min:
            y_min, y_max = 0.0, 100.0
    else:
        y_min, y_max = 0.0, 100.0
    ax.set_ylim(y_min, y_max)

    if clean:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_methods)
        ax.grid(False)
        ax.tick_params(axis="x", labelbottom=False, length=4)
        ax.tick_params(axis="y", labelleft=False, length=4)
    else:
        ax.set_title(f"4-Method Comparison ({setting})", fontsize=13, pad=10)
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10, rotation=-15, ha="right")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        legend_face = "#6C89B3"
        proxies = []
        labels = []
        for k, metric in enumerate(metric_order):
            hatch = hatches[k % len(hatches)]
            alpha = 0.92 if hatch == "" else 0.42
            proxies.append(Patch(facecolor=legend_face, edgecolor="#2B2B2B", linewidth=0.55, hatch=hatch, alpha=alpha))
            labels.append(metric_display.get(metric, metric))
        ax.legend(proxies, labels, loc="upper right", frameon=False, fontsize=10, ncol=1)

    ax.tick_params(axis="both", direction="out", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def draw_metric_lines(df: pd.DataFrame, metric_order, out_no_ext: str, setting: str, clean: bool):
    methods = df["Method"].astype(str).tolist()
    x = np.arange(len(methods), dtype=float)

    # Force 1:1 aspect figure for line-profile comparisons.
    fig, ax = plt.subplots(figsize=(FIG_SIDE, FIG_SIDE))
    all_vals = []

    for metric in metric_order:
        vals = df[metric].to_numpy(dtype=float) * 100.0
        finite = np.isfinite(vals)
        if np.any(finite):
            all_vals.append(vals[finite])
        ax.plot(
            x,
            vals,
            marker=METRIC_LINE_MARKERS.get(metric, "o"),
            markersize=8.2,
            linewidth=2.4,
            linestyle=METRIC_LINE_STYLES.get(metric, "-"),
            color=METRIC_LINE_COLORS.get(metric, "#6C89B3"),
            alpha=0.95,
            label=metric,
        )

    if all_vals:
        vv = np.concatenate(all_vals)
        y_min = max(0.0, float(np.floor(np.nanmin(vv) / 10.0) * 10.0))
        # Clamp upper bound at 100% for probability-based metrics.
        y_max = 100.0
        if y_max <= y_min:
            y_min, y_max = 0.0, 100.0
    else:
        y_min, y_max = 0.0, 100.0
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.set_xticks(x)

    if clean:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([""] * len(methods))
        ax.tick_params(axis="x", labelbottom=False, length=4)
        ax.tick_params(axis="y", labelleft=False, length=4)
        ax.grid(False)
    else:
        ax.set_title(f"4-Method Multi-Metric Profile ({setting})", fontsize=13, pad=14)
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_xticklabels(methods, fontsize=10, rotation=0, ha="center")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(False)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Metric",
            fontsize=10,
            title_fontsize=11,
            ncol=1,
            handlelength=1.7,
            handletextpad=0.6,
        )

    ax.tick_params(axis="both", direction="out", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    global FIG_SIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--setting", default=DEFAULT_SETTING)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics, e.g. ACC,AUROC,AUPRC,F1,Specificity,Precision,Recall",
    )
    parser.add_argument("--fig_side", type=float, default=FIG_SIDE)
    args = parser.parse_args()

    FIG_SIDE = float(args.fig_side)

    metric_order = parse_metric_list(args.metrics)
    ensure_dir(args.out_dir)
    df = build_metrics_table(args.benchmark_root, args.setting, metric_order)

    out_csv = os.path.join(args.out_dir, f"Method_Compare_Metrics_{args.setting}_values.csv")
    df.to_csv(out_csv, index=False)

    out_no_ext = os.path.join(args.out_dir, f"Method_Compare_Metrics_{args.setting}")
    draw_bar(df, metric_order=metric_order, out_no_ext=out_no_ext, setting=args.setting, clean=False)
    draw_bar(df, metric_order=metric_order, out_no_ext=out_no_ext, setting=args.setting, clean=True)

    out_lines_no_ext = os.path.join(args.out_dir, f"Method_Compare_MetricLines_{args.setting}")
    draw_metric_lines(df, metric_order=metric_order, out_no_ext=out_lines_no_ext, setting=args.setting, clean=False)
    draw_metric_lines(df, metric_order=metric_order, out_no_ext=out_lines_no_ext, setting=args.setting, clean=True)

    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] metrics: {', '.join(metric_order)}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")
    print(f"[DONE] line full: {out_lines_no_ext}_full.png/.pdf")
    print(f"[DONE] line clean: {out_lines_no_ext}_clean.png/.pdf")


if __name__ == "__main__":
    main()
