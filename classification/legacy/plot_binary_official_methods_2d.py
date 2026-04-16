# -*- coding: utf-8 -*-
"""
2D comparison plots for binary datasets from benchmark_probs.npz.

For each setting under benchmark root:
- draw full plot (labels/legend/ticks)
- draw clean plot (no labels/legend/title/tick numbers)
- use x=0.5 and y=0.5 separators (quadrant split)

Only official methods are used (no proxy methods).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GREY = "#444444"
COLOR_NEG = "#1f77b4"
COLOR_POS = "#d62728"
MARK_NEG = "s"
MARK_POS = "o"

POINT_SIZE_NEG = 26
POINT_SIZE_POS = 28
ALPHA_2D = 0.70

N_POINTS_2D_DEFAULT = 6000
N_POINTS_2D_BY_DATASET = {
    "GSE175634": 3000,
}
SAMPLE_SEED_2D = 2026


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_fig(fig, out_no_ext):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def sanitize_key(s):
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def style_axes_full(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)


def style_axes_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def plot_2d_pair(xv, yv, y_true, out_no_ext, xlab, ylab, neg_label, pos_label, n_points_2d):
    xv = np.clip(np.asarray(xv, dtype=np.float64), 0.0, 1.0)
    yv = np.clip(np.asarray(yv, dtype=np.float64), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=np.int64)

    if n_points_2d > 0 and len(y_true) > n_points_2d:
        rng = np.random.default_rng(SAMPLE_SEED_2D)
        idx = rng.choice(np.arange(len(y_true)), size=n_points_2d, replace=False)
        xv, yv, y_true = xv[idx], yv[idx], y_true[idx]

    m0 = y_true == 0
    m1 = y_true == 1

    def _draw(ax, clean):
        ax.axvline(0.5, lw=1.6, color=GREY, ls="--", alpha=0.60, zorder=1)
        ax.axhline(0.5, lw=1.6, color=GREY, ls="--", alpha=0.60, zorder=1)

        ax.scatter(
            xv[m0], yv[m0], s=POINT_SIZE_NEG, alpha=ALPHA_2D, c=COLOR_NEG, marker=MARK_NEG,
            edgecolors="none", label=neg_label, zorder=2,
        )
        ax.scatter(
            xv[m1], yv[m1], s=POINT_SIZE_POS, alpha=ALPHA_2D, c=COLOR_POS, marker=MARK_POS,
            edgecolors="none", label=pos_label, zorder=3,
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        if clean:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            style_axes_clean(ax)
        else:
            ax.set_xlabel(xlab, fontsize=13)
            ax.set_ylabel(ylab, fontsize=13)
            ax.legend(loc="upper left", frameon=False, fontsize=10)
            style_axes_full(ax)

    fig_full, ax_full = plt.subplots(figsize=(6.4, 6.4))
    _draw(ax_full, clean=False)
    fig_full.tight_layout()
    save_fig(fig_full, out_no_ext + "_full")

    fig_clean, ax_clean = plt.subplots(figsize=(6.4, 6.4))
    _draw(ax_clean, clean=True)
    fig_clean.tight_layout()
    save_fig(fig_clean, out_no_ext + "_clean")


def load_prob_dict(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    y_true = np.asarray(z["y_true"], dtype=np.int64)

    key_map = {
        "p_scLineagetracer": "scLineagetracer",
        "p_CellRank": "CellRank",
        "p_WOT": "WOT",
        "p_CoSpar": "CoSpar",
    }

    probs = {}
    for k, m in key_map.items():
        if k not in z:
            continue
        arr = np.asarray(z[k], dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] < 2:
                continue
            p = arr[:, 1]
        elif arr.ndim == 1:
            p = arr
        else:
            continue
        probs[m] = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return y_true, probs


def load_allowed_methods(setting_dir):
    """
    Restrict plotting to methods that are not marked as fallback/proxy/failed
    in metrics_summary.csv (if available).
    """
    ms = os.path.join(setting_dir, "metrics_summary.csv")
    if not os.path.isfile(ms):
        return None
    try:
        df = pd.read_csv(ms)
        if "Method" not in df.columns:
            return None
        allow = set(df["Method"].astype(str).tolist())
        if "MethodNote" in df.columns:
            for _, r in df.iterrows():
                m = str(r["Method"])
                note = str(r["MethodNote"]).lower()
                if any(k in note for k in ["fallback", "proxy", "failed", "unavailable", "no_result"]):
                    allow.discard(m)
        allow = {m for m in allow if "proxy" not in m.lower()}
        return allow
    except Exception:
        return None


def run_dataset(dataset):
    if dataset == "GSE175634":
        benchmark_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_all_timepoints_official"
        if not os.path.isdir(benchmark_root):
            benchmark_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_all_timepoints_official_plus"
        out_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/official_methods_plot_v3/2d"
        neg_label = "CM"
        pos_label = "CF"
    elif dataset == "GSE99915":
        benchmark_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/benchmark_all_timepoints_official_plus"
        out_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/official_methods_plot_v2/2d"
        neg_label = "Failed"
        pos_label = "Reprogrammed"
    else:
        benchmark_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus"
        out_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/official_methods_plot_v2/2d"
        neg_label = "Neutrophil"
        pos_label = "Monocyte"

    ensure_dir(out_root)
    n_points_2d = int(N_POINTS_2D_BY_DATASET.get(dataset, N_POINTS_2D_DEFAULT))

    settings = [d for d in sorted(os.listdir(benchmark_root)) if os.path.isdir(os.path.join(benchmark_root, d))]
    pair_candidates = [
        ("scLineagetracer", "CellRank"),
        ("scLineagetracer", "WOT"),
        ("scLineagetracer", "CoSpar"),
    ]

    for setting in settings:
        setting_dir = os.path.join(benchmark_root, setting)
        npz = os.path.join(setting_dir, "benchmark_probs.npz")
        if not os.path.isfile(npz):
            continue
        y_true, probs = load_prob_dict(npz)
        allowed_methods = load_allowed_methods(setting_dir)
        for x_name, y_name in pair_candidates:
            if (x_name not in probs) or (y_name not in probs):
                continue
            if (allowed_methods is not None) and ((x_name not in allowed_methods) or (y_name not in allowed_methods)):
                continue
            tag = f"{sanitize_key(x_name)}vs{sanitize_key(y_name)}"
            out_no_ext = os.path.join(out_root, f"Pred2D_{setting}_{tag}")
            plot_2d_pair(
                probs[x_name],
                probs[y_name],
                y_true,
                out_no_ext,
                f"Predicted probability of {pos_label} ({x_name})",
                f"Predicted probability of {pos_label} ({y_name})",
                neg_label,
                pos_label,
                n_points_2d,
            )
        print(f"[DONE] {dataset} {setting}")

    print(f"[DONE] output: {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GSE175634", "GSE99915", "GSE140802"])
    args = parser.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
