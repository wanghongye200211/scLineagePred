# -*- coding: utf-8 -*-
"""
GSE132188: selected macro-ROC comparison (scLineagetracer) across timepoints.

Input:
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/roc_curve_points_macro.csv

Output:
  classification/GSE132188/roc/official_methods_plot_v3/
    - ROC_Selected_Comparison_scLineagetracer_full|clean.(png|pdf)
    - ROC_Selected_Comparison_scLineagetracer_values.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/benchmark_all_timepoints_official_plus"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/official_methods_plot_v3"

SETTING_ORDER = ["UpTo_12.5", "UpTo_13.5", "UpTo_14.5", "All_15.5"]
SETTING_LABEL = {
    "UpTo_12.5": "<=12.5d",
    "UpTo_13.5": "<=13.5d",
    "UpTo_14.5": "<=14.5d",
    "All_15.5": "All(15.5d)",
}

METHOD = "scLineagetracer"
LINE_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
GREY = "#444444"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def load_curve(benchmark_root: str, setting: str):
    p = os.path.join(benchmark_root, setting, "roc_curve_points_macro.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    sub = df[df["Method"].astype(str) == METHOD].copy()
    if len(sub) == 0:
        return None
    fpr = sub["fpr_macro"].to_numpy(dtype=float)
    tpr = sub["tpr_macro"].to_numpy(dtype=float)
    order = np.argsort(fpr)
    fpr = np.clip(fpr[order], 0.0, 1.0)
    tpr = np.clip(tpr[order], 0.0, 1.0)
    return fpr, tpr, float(auc(fpr, tpr))


def draw(curves, out_no_ext: str, clean: bool):
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=GREY, alpha=0.55)

    for i, (setting, (fpr, tpr, auc_val)) in enumerate(curves.items()):
        lab = SETTING_LABEL.get(setting, setting)
        ax.plot(
            fpr,
            tpr,
            lw=2.7,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            label=f"{lab} (AUC={auc_val:.3f})",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.02, y=0.02)
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.grid(False)

    if clean:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(labelbottom=False, labelleft=False)
    else:
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title("Selected Macro ROC (scLineagetracer)", fontsize=13, pad=10)
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

    curves = {}
    rows = []
    for s in SETTING_ORDER:
        c = load_curve(args.benchmark_root, s)
        if c is None:
            continue
        curves[s] = c
        rows.append({"Setting": s, "Time": SETTING_LABEL.get(s, s), "Method": METHOD, "AUC_macro": c[2]})

    if len(curves) == 0:
        raise RuntimeError(f"No valid curves found in {args.benchmark_root}")

    out_csv = os.path.join(args.out_dir, "ROC_Selected_Comparison_scLineagetracer_values.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_no_ext = os.path.join(args.out_dir, "ROC_Selected_Comparison_scLineagetracer")
    draw(curves, out_no_ext=out_no_ext, clean=False)
    draw(curves, out_no_ext=out_no_ext, clean=True)

    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] full/clean: {out_no_ext}_*.png/.pdf")


if __name__ == "__main__":
    main()
