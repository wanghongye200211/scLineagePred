# -*- coding: utf-8 -*-
"""
GSE132188: per-setting OvR ROC (4 classes in one figure), with full + clean outputs.

Input:
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/benchmark_probs.npz

Output:
  classification/GSE132188/roc/uptoday/
    - ROC_<setting>_AllClasses_full|clean.(png|pdf)
    - ROC_AllClasses_scLineagetracer_values.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

SETTING_ORDER = ["UpTo_12.5", "UpTo_13.5", "UpTo_14.5", "All_15.5"]
CLASS_NAMES = ["Alpha", "Beta", "Delta", "Epsilon"]
METHOD_KEY_MAP = {
    "scLineagetracer": "p_scLineagetracer",
    "CellRank": "p_CellRank",
    "WOT": "p_WOT",
    "CoSpar": "p_CoSpar",
}
CLASS_COLOR = {
    "Alpha": "#1f77b4",
    "Beta": "#ff7f0e",
    "Delta": "#2ca02c",
    "Epsilon": "#d62728",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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

    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)

    mask_one = (fpr == 1.0)
    if np.any(mask_one):
        t1 = float(np.max(tpr[mask_one]))
        keep = ~mask_one
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]
    return fpr, tpr


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=320, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=320, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def draw_one_setting(y_true, prob, setting: str, method_name: str, out_no_ext: str, clean: bool):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n_classes = len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(7.4, 6.7))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color="#555555", alpha=0.55)

    auc_rows = []
    for i, cname in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, prob[:, i])
        fpr, tpr = _roc_endpoints_clean(fpr, tpr)
        auc_i = float(auc(fpr, tpr))
        auc_rows.append({"Setting": setting, "Method": method_name, "Class": cname, "AUC": auc_i})
        label = f"{cname} (AUC={auc_i:.3f})" if not clean else None
        ax.plot(
            fpr,
            tpr,
            lw=2.8,
            color=CLASS_COLOR.get(cname, "#333333"),
            label=label,
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
        ax.set_title(f"Stacking ROC (OvR) - {setting}", fontsize=13, pad=10)
        ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)
    return auc_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--method", choices=list(METHOD_KEY_MAP.keys()), default="scLineagetracer")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    key = METHOD_KEY_MAP[args.method]

    all_rows = []
    for setting in SETTING_ORDER:
        npz_path = os.path.join(args.benchmark_root, setting, "benchmark_probs.npz")
        if not os.path.isfile(npz_path):
            print(f"[WARN] missing npz: {npz_path}")
            continue
        z = np.load(npz_path, allow_pickle=True)
        if ("y_true" not in z) or (key not in z):
            print(f"[WARN] missing y_true or {key}: {npz_path}")
            continue

        y_true = np.asarray(z["y_true"], dtype=int)
        prob = np.asarray(z[key], dtype=float)
        if prob.ndim != 2 or prob.shape[1] != len(CLASS_NAMES):
            print(f"[WARN] bad shape for {setting}/{args.method}: {prob.shape}")
            continue

        out_base = os.path.join(args.out_dir, f"ROC_{setting}_AllClasses")
        all_rows.extend(draw_one_setting(y_true, prob, setting, args.method, out_base, clean=False))
        all_rows.extend(draw_one_setting(y_true, prob, setting, args.method, out_base, clean=True))

    if len(all_rows) > 0:
        out_csv = os.path.join(args.out_dir, f"ROC_AllClasses_{args.method}_values.csv")
        pd.DataFrame(all_rows).drop_duplicates(["Setting", "Method", "Class", "AUC"]).to_csv(out_csv, index=False)
        print(f"[DONE] values: {out_csv}")

    print(f"[DONE] outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()

