# -*- coding: utf-8 -*-
"""
GSE132188 regression-view ROC summary:
Single figure that shows 4 cell-type ROC curves for each of 4 timepoint settings.

Layout:
- 2x2 subplots
- each subplot corresponds to one setting (UpTo_12.5 / UpTo_13.5 / UpTo_14.5 / All_15.5)
- each subplot overlays 4 class OvR ROC curves (Alpha/Beta/Delta/Epsilon)

Data source:
- classification benchmark probabilities (scLineagetracer) from:
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/<SETTING>/benchmark_probs.npz
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/benchmark_all_timepoints_official_plus"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE132188/Reg_15.5_from_12.5_13.5_14.5/plots_template_v1"

SETTINGS = ["UpTo_12.5", "UpTo_13.5", "UpTo_14.5", "All_15.5"]
CLASSES = ["Alpha", "Beta", "Delta", "Epsilon"]

CLASS_COLORS = {
    "Alpha": "#4E79A7",
    "Beta": "#F28E2B",
    "Delta": "#59A14F",
    "Epsilon": "#E15759",
}

LINE_GREY = "#777777"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def roc_endpoints_clean(fpr: np.ndarray, tpr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    fpr = np.clip(fpr, 0.0, 1.0)
    tpr = np.clip(tpr, 0.0, 1.0)
    return fpr, tpr


def load_setting_probs(benchmark_root: str, setting: str) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = os.path.join(benchmark_root, setting, "benchmark_probs.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing benchmark_probs.npz for setting: {setting}")
    z = np.load(npz_path, allow_pickle=True)
    if "y_true" not in z.files or "p_scLineagetracer" not in z.files:
        raise KeyError(f"Missing keys in {npz_path}. Required: y_true, p_scLineagetracer")
    y_true = np.asarray(z["y_true"], dtype=np.int64)
    prob = np.asarray(z["p_scLineagetracer"], dtype=np.float64)
    return y_true, prob


def build_curves(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    y_bin = label_binarize(y_true, classes=np.arange(len(CLASSES)))
    curves = {}
    for i, cls_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        fpr, tpr = roc_endpoints_clean(fpr, tpr)
        curves[cls_name] = (fpr, tpr, float(auc(fpr, tpr)))
    return curves


def draw_2x2(curves_by_setting: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]], out_no_ext: str, clean: bool):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, setting in zip(axes, SETTINGS):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=LINE_GREY, alpha=0.5)
        curves = curves_by_setting[setting]
        for cls_name in CLASSES:
            fpr, tpr, auc_val = curves[cls_name]
            ax.plot(
                fpr,
                tpr,
                lw=2.2,
                color=CLASS_COLORS.get(cls_name, "#666666"),
                label=f"{cls_name} (AUC={auc_val:.3f})",
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="both", direction="out", top=False, right=False)
        ax.grid(False)

        if not clean:
            ax.set_title(setting, fontsize=12, pad=7)
            ax.legend(loc="lower right", frameon=False, fontsize=8)
        else:
            ax.set_title("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if not clean:
        axes[0].set_ylabel("True Positive Rate", fontsize=11)
        axes[2].set_ylabel("True Positive Rate", fontsize=11)
        axes[2].set_xlabel("False Positive Rate", fontsize=11)
        axes[3].set_xlabel("False Positive Rate", fontsize=11)
        fig.suptitle("GSE132188: 4-Class ROC Across 4 Timepoints", fontsize=14, y=0.995)

    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default=BENCHMARK_ROOT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    curves_by_setting = {}
    auc_rows: List[dict] = []

    for setting in SETTINGS:
        y_true, prob = load_setting_probs(args.benchmark_root, setting)
        curves = build_curves(y_true, prob)
        curves_by_setting[setting] = curves
        for cls_name in CLASSES:
            auc_rows.append(
                {
                    "setting": setting,
                    "class": cls_name,
                    "auc": curves[cls_name][2],
                }
            )

    out_csv = os.path.join(args.out_dir, "ROC_4Classes_4Timepoints_auc_table.csv")
    pd.DataFrame(auc_rows).to_csv(out_csv, index=False)

    out_no_ext = os.path.join(args.out_dir, "ROC_4Classes_4Timepoints")
    draw_2x2(curves_by_setting, out_no_ext=out_no_ext, clean=False)
    draw_2x2(curves_by_setting, out_no_ext=out_no_ext, clean=True)

    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")


if __name__ == "__main__":
    main()
