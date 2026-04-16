# -*- coding: utf-8 -*-
"""
GSE132188: 4-method first-two-timepoint comparison heatmap (large) for uptoday folder.

Output:
  /Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/uptoday
    - Model_Compare_First2Days_JournalHeatmap_uptoday_full.(png|pdf)
    - Model_Compare_First2Days_JournalHeatmap_uptoday_clean.(png|pdf)
    - Model_Compare_First2Days_values_uptoday.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "font.size": 11,
    }
)


BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/benchmark_all_timepoints_official_plus"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/uptoday"

SETTING_ORDER = ["UpTo_12.5", "UpTo_13.5"]
SETTING_LABEL = {
    "UpTo_12.5": "12.5d",
    "UpTo_13.5": "13.5d",
}

METHODS = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
METHOD_COLORS = {
    "scLineagetracer": "#1F77B4",
    "CellRank": "#E15759",
    "WOT": "#59A14F",
    "CoSpar": "#F28E2B",
}
UNIFIED_CMAP = "YlGnBu"
UNIFIED_VMIN = 0.50
UNIFIED_VMAX = 1.00


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    # Keep exact 1:1 canvas ratio at export time (no tight cropping).
    fig.savefig(out_no_ext + ".png", dpi=360)
    fig.savefig(out_no_ext + ".pdf", dpi=360)
    plt.close(fig)


def _read_metric(benchmark_root: str, metric_name: str) -> pd.DataFrame:
    rows = []
    for setting in SETTING_ORDER:
        csv_path = os.path.join(benchmark_root, setting, "metrics_summary.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if ("Method" not in df.columns) or (metric_name not in df.columns):
            continue

        rec = {"Setting": setting, "Time": SETTING_LABEL.get(setting, setting)}
        for m in METHODS:
            sub = df[df["Method"].astype(str) == m]
            rec[m] = float(sub[metric_name].iloc[0]) if len(sub) > 0 else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError(f"No valid metrics found for {metric_name} under: {benchmark_root}")
    out["Setting"] = pd.Categorical(out["Setting"], categories=SETTING_ORDER, ordered=True)
    out = out.sort_values("Setting").reset_index(drop=True)
    return out


def _method_order(df_auc: pd.DataFrame):
    return sorted(METHODS, key=lambda m: float(np.nanmean(df_auc[m].to_numpy(dtype=float))), reverse=True)


def _compute_big_text_fs(fig, ax, nrows: int, ncols: int, sample_text: str = "0.704", cover_ratio: float = 0.70) -> float:
    # Estimate fontsize so text occupies roughly 70% of one cell width/height.
    bbox = ax.get_position()
    fw, fh = fig.get_size_inches()
    cell_w_in = (bbox.width * fw) / max(ncols, 1)
    cell_h_in = (bbox.height * fh) / max(nrows, 1)

    nch = max(len(sample_text), 1)
    fs_w = (cover_ratio * cell_w_in * 72.0) / (0.58 * nch)
    fs_h = (cover_ratio * cell_h_in * 72.0) / 1.35
    return float(max(10.0, min(fs_w, fs_h)))


def _draw_one_heatmap(ax, fig, df_metric: pd.DataFrame, method_order, title: str, clean: bool):
    times = df_metric["Time"].tolist()
    mat = df_metric[method_order].to_numpy(dtype=float).T  # [n_methods, n_times]
    im = ax.imshow(mat, aspect="equal", cmap=UNIFIED_CMAP, vmin=UNIFIED_VMIN, vmax=UNIFIED_VMAX)

    # clean 版本也保留文字（数值注释 + 轴刻度）
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels(times, fontsize=(13 if clean else 12))
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=(12 if clean else 11))
    for lab in ax.get_yticklabels():
        lab.set_color(METHOD_COLORS.get(lab.get_text(), "#333333"))

    if not clean:
        ax.set_title(title, fontsize=15, pad=9)
    else:
        ax.set_title("")

    ax.set_xticks(np.arange(-0.5, len(times), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(method_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fs = _compute_big_text_fs(fig, ax, mat.shape[0], mat.shape[1], sample_text="0.704", cover_ratio=0.70)
    span = max(UNIFIED_VMAX - UNIFIED_VMIN, 1e-8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                txt, color = "NA", "#555555"
            else:
                txt = f"{v:.3f}"
                frac = (v - UNIFIED_VMIN) / span
                color = "white" if frac >= 0.58 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color=color, fontweight="bold")

    ax.tick_params(axis="both", direction="out", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_large_heatmap(df_acc: pd.DataFrame, df_auc: pd.DataFrame, out_no_ext: str, clean: bool):
    method_order = _method_order(df_auc)
    # Force whole figure to 1:1 aspect.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 12.0), sharey=True)

    im0 = _draw_one_heatmap(
        axes[0],
        fig=fig,
        df_metric=df_acc,
        method_order=method_order,
        title="Accuracy",
        clean=clean,
    )
    im1 = _draw_one_heatmap(
        axes[1],
        fig=fig,
        df_metric=df_auc,
        method_order=method_order,
        title="Macro-AUC",
        clean=clean,
    )

    if not clean:
        axes[0].set_ylabel("Method", fontsize=13)
        axes[0].set_xlabel("Observed timepoint", fontsize=13)
        axes[1].set_xlabel("Observed timepoint", fontsize=13)
        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label("Accuracy", fontsize=12)
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Macro-AUC", fontsize=12)
        fig.suptitle("4-Method Benchmark (First Two Timepoints)", fontsize=17, y=1.03)
    else:
        axes[0].set_ylabel("Method", fontsize=13)
        axes[0].set_xlabel("Observed timepoint", fontsize=13)
        axes[1].set_xlabel("Observed timepoint", fontsize=13)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df_acc = _read_metric(args.benchmark_root, "Accuracy")
    df_auc = _read_metric(args.benchmark_root, "AUC_macro")

    out_values = os.path.join(args.out_dir, "Model_Compare_First2Days_values_uptoday.csv")
    acc_long = df_acc.melt(id_vars=["Setting", "Time"], value_vars=METHODS, var_name="Method", value_name="Accuracy")
    auc_long = df_auc.melt(id_vars=["Setting", "Time"], value_vars=METHODS, var_name="Method", value_name="AUC_macro")
    merged = acc_long.merge(auc_long, on=["Setting", "Time", "Method"], how="outer")
    merged.to_csv(out_values, index=False)

    out_no_ext = os.path.join(args.out_dir, "Model_Compare_First2Days_JournalHeatmap_uptoday")
    draw_large_heatmap(df_acc, df_auc, out_no_ext=out_no_ext, clean=False)
    draw_large_heatmap(df_acc, df_auc, out_no_ext=out_no_ext, clean=True)

    print(f"[DONE] values: {out_values}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")


if __name__ == "__main__":
    main()
