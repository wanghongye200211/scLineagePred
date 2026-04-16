# -*- coding: utf-8 -*-
"""
GSE175634: 4-method accuracy bar plot across timepoints.

Input:
  classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_all_timepoints_official/*/metrics_summary.csv

Output:
  classification/GSE175634/GSE175634_CMvsCF/roc_click/official_methods_plot_v3/
    - Accuracy_Bar_4Methods_full.png/.pdf
    - Accuracy_Bar_4Methods_clean.png/.pdf
    - Accuracy_Bar_4Methods_values.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_all_timepoints_official"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/official_methods_plot_v3"

SETTING_ORDER = ["Obs_Day1", "Obs_Day3", "Obs_Day5", "Obs_Day7", "Obs_Day11"]
METHODS = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]

METHOD_COLORS = {
    # Match ROC line colors in compare_roc_official_remaining3.py
    "scLineagetracer": "#d62728",
    "CellRank": "#1f77b4",
    "WOT": "#2ca02c",
    "CoSpar": "#9467bd",
}

BAR_WIDTH = 0.14
Y_LIM = (0.0, 1.05)
Y_TICK_STEP = 0.1
FIG_ASPECT = 1.1
FIG_HEIGHT = 5.6
TOP_WHITESPACE = 0.86


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def setting_to_day_label(setting: str) -> str:
    m = re.search(r"Day(\d+)", str(setting))
    if m is None:
        return str(setting)
    return f"d{m.group(1)}"


def read_accuracy_table(benchmark_root: str) -> pd.DataFrame:
    rows = []
    for setting in SETTING_ORDER:
        csv_path = os.path.join(benchmark_root, setting, "metrics_summary.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if ("Method" not in df.columns) or ("Accuracy" not in df.columns):
            continue

        rec = {"Setting": setting, "Time": setting_to_day_label(setting)}
        for method in METHODS:
            sub = df[df["Method"].astype(str) == method]
            if len(sub) == 0:
                rec[method] = np.nan
            else:
                rec[method] = float(sub["Accuracy"].iloc[0])
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError(f"No valid metrics_summary.csv found under: {benchmark_root}")
    out["Setting"] = pd.Categorical(out["Setting"], categories=SETTING_ORDER, ordered=True)
    out = out.sort_values("Setting").reset_index(drop=True)
    return out


def draw_bar(df: pd.DataFrame, out_no_ext: str, clean: bool):
    x = np.arange(len(df))
    n_methods = len(METHODS)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * BAR_WIDTH

    fig_w = FIG_HEIGHT * FIG_ASPECT
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))

    for i, method in enumerate(METHODS):
        vals = df[method].to_numpy(dtype=float)
        ax.bar(
            x + offsets[i],
            vals,
            width=BAR_WIDTH * 0.88,
            color=METHOD_COLORS.get(method, "#777777"),
            edgecolor="black",
            linewidth=0.6,
            label=method,
        )

    ax.set_ylim(*Y_LIM)
    ax.set_yticks(np.arange(Y_LIM[0], Y_LIM[1] + 1e-9, Y_TICK_STEP))
    ax.set_xticks(x)
    ax.set_xticklabels(df["Time"].astype(str).tolist(), fontsize=11)

    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.grid(False)

    if clean:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
    else:
        ax.set_title("Accuracy Comparison Across Timepoints", fontsize=13, pad=10)
        ax.set_xlabel("Observed Timepoint", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, TOP_WHITESPACE))
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = read_accuracy_table(args.benchmark_root)
    out_csv = os.path.join(args.out_dir, "Accuracy_Bar_4Methods_values.csv")
    df.to_csv(out_csv, index=False)

    out_no_ext = os.path.join(args.out_dir, "Accuracy_Bar_4Methods")
    draw_bar(df, out_no_ext, clean=False)
    draw_bar(df, out_no_ext, clean=True)

    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")


if __name__ == "__main__":
    main()
