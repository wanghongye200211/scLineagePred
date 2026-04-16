#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-plot GSE99915 4-method ROC curves (clean/full) with 140802 color style.

Input:
  classification/GSE99915/roc_click/benchmark_all_timepoints_official_plus/<SETTING>/benchmark_probs.npz

Output:
  classification/GSE99915/roc_click/official_methods_plot_v3/
    - ROC_Comparison_<SETTING>_4Methods_full.png/.pdf
    - ROC_Comparison_<SETTING>_4Methods_clean.png/.pdf
    - ROC_Comparison_4Methods_values.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/benchmark_all_timepoints_official_plus"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/official_methods_plot_v3"

METHOD_ORDER = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
METHOD_COLORS = {
    "scLineagetracer": "#d62728",
    "CellRank": "#1f77b4",
    "WOT": "#2ca02c",
    "CoSpar": "#9467bd",
}
KEY_MAP = {
    "scLineagetracer": "p_scLineagetracer",
    "CellRank": "p_CellRank",
    "WOT": "p_WOT",
    "CoSpar": "p_CoSpar",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def style_axes_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def roc_endpoints_clean(fpr, tpr):
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
    return fpr, tpr


def to_pos_prob(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1].astype(np.float64)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    raise ValueError(f"Unsupported probability array shape: {arr.shape}")


def read_setting_curves(setting_dir):
    npz_path = os.path.join(setting_dir, "benchmark_probs.npz")
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    if "y_true" not in data.files:
        return None
    y_true = np.asarray(data["y_true"]).astype(np.int64)

    curves = {}
    for m in METHOD_ORDER:
        k = KEY_MAP[m]
        if k not in data.files:
            continue
        p = to_pos_prob(data[k])
        fpr, tpr, _ = roc_curve(y_true, p)
        fpr, tpr = roc_endpoints_clean(fpr, tpr)
        curves[m] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    return y_true, curves


def draw_roc(curves, setting, out_no_ext, clean):
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color="#555555", alpha=0.55)

    for m in METHOD_ORDER:
        if m not in curves:
            continue
        c = curves[m]
        label = None if clean else f"{m} (AUC={c['auc']:.3f})"
        ax.plot(
            c["fpr"],
            c["tpr"],
            lw=2.7,
            color=METHOD_COLORS.get(m, "#444444"),
            label=label,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)
    ax.grid(False)

    if clean:
        style_axes_clean(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    else:
        ax.tick_params(axis="both", direction="out", top=False, right=False)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(f"ROC Comparison (4 Methods) - {setting}", fontsize=13, pad=10)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rows = []

    setting_dirs = []
    for name in os.listdir(args.benchmark_root):
        p = os.path.join(args.benchmark_root, name)
        if os.path.isdir(p):
            setting_dirs.append((name, p))
    setting_dirs = sorted(setting_dirs)

    for setting, sdir in setting_dirs:
        parsed = read_setting_curves(sdir)
        if parsed is None:
            continue
        _, curves = parsed
        if len(curves) == 0:
            continue

        out_no_ext = os.path.join(args.out_dir, f"ROC_Comparison_{setting}_4Methods")
        draw_roc(curves, setting, out_no_ext, clean=False)
        draw_roc(curves, setting, out_no_ext, clean=True)

        for m in METHOD_ORDER:
            if m in curves:
                rows.append(
                    {
                        "setting": setting,
                        "method": m,
                        "auc": curves[m]["auc"],
                    }
                )

    out_csv = os.path.join(args.out_dir, "ROC_Comparison_4Methods_values.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] figures dir: {args.out_dir}")


if __name__ == "__main__":
    main()
