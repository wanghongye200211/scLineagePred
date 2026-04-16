# -*- coding: utf-8 -*-
"""
GSE132188: 4-method comparison on the first two timepoints (journal-style, non-linear).

This script uses an annotated benchmark heatmap (not bars, not slope lines):
- Left block: Accuracy
- Right block: Macro-F1

Input:
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/metrics_summary.csv
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/benchmark_probs.npz

Output:
  classification/GSE132188/roc/official_methods_plot_v3/
    - Model_Compare_First2Days_JournalHeatmap_full|clean.(png|pdf)
    - Model_Compare_First2Days_values.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/official_methods_plot_v3"

# 只画“前两天”
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

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
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


def _read_macro_f1(benchmark_root: str) -> pd.DataFrame:
    rows = []
    for setting in SETTING_ORDER:
        npz_path = os.path.join(benchmark_root, setting, "benchmark_probs.npz")
        if not os.path.isfile(npz_path):
            continue
        dat = np.load(npz_path, allow_pickle=True)
        if "y_true" not in dat.files:
            continue

        y_true = np.asarray(dat["y_true"], dtype=np.int64)
        rec = {"Setting": setting, "Time": SETTING_LABEL.get(setting, setting)}
        for m in METHODS:
            key = f"p_{m}"
            if key not in dat.files:
                rec[m] = np.nan
                continue
            prob = np.asarray(dat[key], dtype=np.float64)
            if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.shape[1] == 0:
                rec[m] = np.nan
                continue
            y_pred = np.argmax(prob, axis=1)
            rec[m] = float(f1_score(y_true, y_pred, average="macro"))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError(f"No valid benchmark_probs.npz found under: {benchmark_root}")
    out["Setting"] = pd.Categorical(out["Setting"], categories=SETTING_ORDER, ordered=True)
    out = out.sort_values("Setting").reset_index(drop=True)
    return out


def _method_order(df_acc: pd.DataFrame, df_f1: pd.DataFrame):
    def _score(m):
        acc_mean = float(np.nanmean(df_acc[m].to_numpy(dtype=float)))
        f1_mean = float(np.nanmean(df_f1[m].to_numpy(dtype=float)))
        return 0.5 * (acc_mean + f1_mean)

    return sorted(METHODS, key=_score, reverse=True)


def _nice_ticks(vmin: float, vmax: float, n_target: int = 6) -> np.ndarray:
    span = max(float(vmax) - float(vmin), 1e-9)
    raw_step = span / max(n_target - 1, 1)
    exp10 = np.floor(np.log10(raw_step))
    frac = raw_step / (10 ** exp10)

    if frac <= 1.0:
        nice_frac = 1.0
    elif frac <= 2.0:
        nice_frac = 2.0
    elif frac <= 2.5:
        nice_frac = 2.5
    elif frac <= 5.0:
        nice_frac = 5.0
    else:
        nice_frac = 10.0

    step = nice_frac * (10 ** exp10)
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step

    ticks = np.arange(lo, hi + 0.5 * step, step)
    ticks = ticks[(ticks >= -1e-9) & (ticks <= 1.0 + 1e-9)]
    if ticks.size < 2:
        ticks = np.array([vmin, vmax], dtype=float)
    return ticks.astype(float)


def _draw_journal_heatmap(df_acc: pd.DataFrame, df_f1: pd.DataFrame, out_no_ext: str, clean: bool):
    method_order = _method_order(df_acc, df_f1)
    times = df_acc["Time"].tolist()
    mat_acc = df_acc[method_order].to_numpy(dtype=float).T
    mat_f1 = df_f1[method_order].to_numpy(dtype=float).T
    mat = np.concatenate([mat_acc, mat_f1], axis=1)  # [n_methods, 2*n_times]
    col_labels = times + times

    # Use data-driven range, but align to rounded "nice" ticks.
    finite = mat[np.isfinite(mat)]
    if finite.size > 0:
        data_min = float(np.nanmin(finite))
        data_max = float(np.nanmax(finite))
        tick_vals = _nice_ticks(data_min, data_max, n_target=6)
        vmin = float(tick_vals[0])
        vmax = float(tick_vals[-1])
    else:
        vmin = 0.0
        vmax = 1.0
        tick_vals = np.linspace(0.0, 1.0, 6)

    # Near-square canvas + equal aspect -> square cells.
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 10.2))
    im = ax.imshow(mat, aspect="equal", cmap="YlGnBu", vmin=vmin, vmax=vmax)

    if not clean:
        # Big-font style requested by user.
        fs_axis = 24
        fs_tick = 22
        fs_group = 30
        fs_cbar = 22

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=fs_tick)
        ax.set_yticks(np.arange(len(method_order)))
        ax.set_yticklabels(method_order, fontsize=fs_tick)
        for lab in ax.get_yticklabels():
            lab.set_color(METHOD_COLORS.get(lab.get_text(), "#333333"))

        ax.set_xlabel("Observed timepoint", fontsize=fs_axis)
        ax.set_ylabel("Method", fontsize=fs_axis)
        ax.set_title("")

        # Cell boundaries + separator between Accuracy and F1 blocks.
        ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(method_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        split_x = len(times) - 0.5
        ax.axvline(split_x, color="white", linewidth=4.0)

        # Top block labels.
        acc_center = (len(times) - 1) / 2.0
        f1_center = len(times) + (len(times) - 1) / 2.0
        ax.text(acc_center, -0.72, "Accuracy", ha="center", va="bottom", fontsize=fs_group, fontweight="semibold", clip_on=False)
        ax.text(f1_center, -0.72, "F1", ha="center", va="bottom", fontsize=fs_group, fontweight="semibold", clip_on=False)

        # Compute the largest in-cell font that still fits a single square.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        p00 = ax.transData.transform((0, 0))
        p10 = ax.transData.transform((1, 0))
        p01 = ax.transData.transform((0, 1))
        cell_w_px = float(abs(p10[0] - p00[0]))
        cell_h_px = float(abs(p01[1] - p00[1]))

        probe = ax.text(
            0,
            0,
            "0.568",
            ha="center",
            va="center",
            fontweight="semibold",
            alpha=0.0,
        )
        fs_cell = 12.0
        for cand in range(64, 9, -1):
            probe.set_fontsize(cand)
            bb = probe.get_window_extent(renderer=renderer)
            if bb.width <= cell_w_px * 0.72 and bb.height <= cell_h_px * 0.72:
                fs_cell = float(cand)
                break
        probe.remove()

        span = max(vmax - vmin, 1e-8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt, color = "NA", "#555555"
                else:
                    txt = f"{v:.3f}"
                    frac = (v - vmin) / span
                    color = "white" if frac >= 0.58 else "#222222"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs_cell, color=color, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("Score", fontsize=fs_cbar)
        cbar.set_ticks(tick_vals)
        cbar.ax.set_yticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in tick_vals])
        cbar.ax.tick_params(labelsize=fs_tick - 2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")

    ax.tick_params(axis="both", direction="out", top=False, right=False, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.91, top=0.90, bottom=0.15)
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df_acc = _read_metric(args.benchmark_root, "Accuracy")
    df_f1 = _read_macro_f1(args.benchmark_root)

    out_values = os.path.join(args.out_dir, "Model_Compare_First2Days_values.csv")
    acc_long = df_acc.melt(id_vars=["Setting", "Time"], value_vars=METHODS, var_name="Method", value_name="Accuracy")
    f1_long = df_f1.melt(id_vars=["Setting", "Time"], value_vars=METHODS, var_name="Method", value_name="F1_macro")
    merged = acc_long.merge(f1_long, on=["Setting", "Time", "Method"], how="outer")
    merged.to_csv(out_values, index=False)

    out_no_ext = os.path.join(args.out_dir, "Model_Compare_First2Days_JournalHeatmap")
    _draw_journal_heatmap(df_acc, df_f1, out_no_ext=out_no_ext, clean=False)
    _draw_journal_heatmap(df_acc, df_f1, out_no_ext=out_no_ext, clean=True)

    print(f"[DONE] values: {out_values}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")


if __name__ == "__main__":
    main()
