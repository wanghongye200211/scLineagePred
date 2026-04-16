# -*- coding: utf-8 -*-
"""
GSE132188: macro-ROC comparison (4 methods only) across four timepoints.

Input:
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/roc_curve_points_macro.csv
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/metrics_summary.csv

Output:
  classification/GSE132188/roc/official_methods_plot_v3/
    - ROC_Comparison_4Methods_4Timepoints_Journal_full|clean.(png|pdf)
    - ROC_Comparison_4Methods_4Timepoints_values.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "font.size": 10.5,
    }
)


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/benchmark_all_timepoints_official_plus"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/official_methods_plot_v3"

SETTING_ORDER = ["UpTo_12.5", "UpTo_13.5", "UpTo_14.5", "All_15.5"]
SETTING_LABEL = {
    "UpTo_12.5": "<=12.5d",
    "UpTo_13.5": "<=13.5d",
    "UpTo_14.5": "<=14.5d",
    "All_15.5": "All(15.5d)",
}

METHODS = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
METHOD_COLORS = {
    "scLineagetracer": "#1F77B4",
    "CellRank": "#E15759",
    "WOT": "#59A14F",
    "CoSpar": "#F28E2B",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=320, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=320, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _load_one_setting_curves(benchmark_root: str, setting: str):
    p_curve = os.path.join(benchmark_root, setting, "roc_curve_points_macro.csv")
    if not os.path.isfile(p_curve):
        return {}
    df = pd.read_csv(p_curve)
    out = {}
    for m in METHODS:
        sub = df[df["Method"].astype(str) == m]
        if len(sub) == 0:
            continue
        fpr = sub["fpr_macro"].to_numpy(dtype=float)
        tpr = sub["tpr_macro"].to_numpy(dtype=float)
        order = np.argsort(fpr)
        fpr = np.clip(fpr[order], 0.0, 1.0)
        tpr = np.clip(tpr[order], 0.0, 1.0)
        out[m] = (fpr, tpr, float(auc(fpr, tpr)))
    return out


def _load_auc_table(benchmark_root: str):
    rows = []
    for s in SETTING_ORDER:
        p = os.path.join(benchmark_root, s, "metrics_summary.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        for m in METHODS:
            sub = df[df["Method"].astype(str) == m]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "Setting": s,
                    "Time": SETTING_LABEL.get(s, s),
                    "Method": m,
                    "AUC_macro": float(sub["AUC_macro"].iloc[0]),
                    "Accuracy": float(sub["Accuracy"].iloc[0]) if "Accuracy" in sub.columns else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _draw_panel(curves_by_setting, use_settings, out_no_ext: str, clean: bool, title: str):
    use_settings = [s for s in use_settings if s in curves_by_setting and len(curves_by_setting[s]) > 0]
    if len(use_settings) == 0:
        raise RuntimeError("No ROC curves found for the selected settings.")

    n = len(use_settings)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    flat_axes = axes.ravel()

    for i, s in enumerate(use_settings):
        ax = flat_axes[i]
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#666666", alpha=0.6)

        curves = curves_by_setting[s]
        for m in METHODS:
            if m not in curves:
                continue
            fpr, tpr, auc_val = curves[m]
            label = f"{m} (AUC={auc_val:.3f})" if not clean else None
            ax.plot(
                fpr,
                tpr,
                lw=2.35,
                color=METHOD_COLORS.get(m, "#444444"),
                alpha=0.98,
                label=label,
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="both", direction="out", top=False, right=False)
        ax.grid(False)

        if clean:
            ax.set_title("")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(SETTING_LABEL.get(s, s), fontsize=12, pad=7)
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.legend(loc="lower right", frameon=False, fontsize=8)

    # Hide extra subplot slots when len(settings) is not 2 or 4.
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    if not clean:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    curves_by_setting = {}
    for s in SETTING_ORDER:
        curves = _load_one_setting_curves(args.benchmark_root, s)
        if len(curves) > 0:
            curves_by_setting[s] = curves

    if len(curves_by_setting) == 0:
        raise RuntimeError(f"No valid ROC curves found under: {args.benchmark_root}")

    auc_table = _load_auc_table(args.benchmark_root)
    out_csv = os.path.join(args.out_dir, "ROC_Comparison_4Methods_4Timepoints_values.csv")
    auc_table.to_csv(out_csv, index=False)

    out_no_ext = os.path.join(args.out_dir, "ROC_Comparison_4Methods_4Timepoints_Journal")
    _draw_panel(
        curves_by_setting,
        use_settings=SETTING_ORDER,
        out_no_ext=out_no_ext,
        clean=False,
        title="Macro ROC Comparison (4 Methods, 4 Timepoints)",
    )
    _draw_panel(
        curves_by_setting,
        use_settings=SETTING_ORDER,
        out_no_ext=out_no_ext,
        clean=True,
        title="",
    )

    out_no_ext_first2 = os.path.join(args.out_dir, "ROC_Comparison_4Methods_First2Days_Journal")
    _draw_panel(
        curves_by_setting,
        use_settings=["UpTo_12.5", "UpTo_13.5"],
        out_no_ext=out_no_ext_first2,
        clean=False,
        title="Macro ROC Comparison (4 Methods, First Two Timepoints)",
    )
    _draw_panel(
        curves_by_setting,
        use_settings=["UpTo_12.5", "UpTo_13.5"],
        out_no_ext=out_no_ext_first2,
        clean=True,
        title="",
    )

    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")
    print(f"[DONE] first2 full: {out_no_ext_first2}_full.png/.pdf")
    print(f"[DONE] first2 clean: {out_no_ext_first2}_clean.png/.pdf")


if __name__ == "__main__":
    main()
