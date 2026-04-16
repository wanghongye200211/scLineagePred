# -*- coding: utf-8 -*-
"""
GSE132188: large single-panel confusion matrices for 4 methods x 4 settings.

Goal:
- Export one large figure per (setting, method), not small multi-panel montage.
- Highlight the known discrepancy: low Epsilon recall vs relatively high Epsilon AUC.

Input:
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/benchmark_probs.npz
  classification/GSE132188/roc/benchmark_all_timepoints_official_plus/*/metrics_auc_per_class.csv

Output:
  classification/GSE132188/roc/official_methods_plot_v3/confusion_4methods_large/
    - CM_<setting>_<method>_rowNorm_full.(png|pdf)
    - CM_<setting>_<method>_counts_full.(png|pdf)
    - Confusion_4Methods_Epsilon_vs_AUC_summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

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
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/official_methods_plot_v3/confusion_4methods_large"

SETTING_ORDER = ["UpTo_12.5", "UpTo_13.5", "UpTo_14.5", "All_15.5"]
SETTING_LABEL = {
    "UpTo_12.5": "<=12.5d",
    "UpTo_13.5": "<=13.5d",
    "UpTo_14.5": "<=14.5d",
    "All_15.5": "All(15.5d)",
}

METHODS = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
METHOD_KEY_IN_NPZ = {
    "scLineagetracer": "p_scLineagetracer",
    "CellRank": "p_CellRank",
    "WOT": "p_WOT",
    "CoSpar": "p_CoSpar",
}

CLASS_NAMES = ["Alpha", "Beta", "Delta", "Epsilon"]
EPS_CLASS = "Epsilon"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_no_ext + ".pdf", dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _load_auc_per_class(benchmark_root: str, setting: str):
    p = os.path.join(benchmark_root, setting, "metrics_auc_per_class.csv")
    if not os.path.isfile(p):
        return {}
    df = pd.read_csv(p)
    out = {}
    for m in METHODS:
        sub = df[df["Method"].astype(str) == m]
        out[m] = {str(r["Class"]): float(r["AUC"]) for _, r in sub.iterrows()}
    return out


def _draw_cm_row_norm(cm, setting: str, method: str, eps_recall: float, eps_auc: float, out_no_ext: str):
    row_sum = cm.sum(axis=1, keepdims=True)
    cmn = cm / np.clip(row_sum, 1e-12, None)

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized ratio", fontsize=11)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(
        f"Confusion (row-normalized) | {SETTING_LABEL.get(setting, setting)} | {method}\n"
        f"Epsilon recall={eps_recall:.3f}, Epsilon AUC={eps_auc:.3f}",
        fontsize=13,
        pad=12,
    )

    # Highlight epsilon diagonal cell.
    eps_idx = CLASS_NAMES.index(EPS_CLASS)
    rect = plt.Rectangle((eps_idx - 0.5, eps_idx - 0.5), 1.0, 1.0, fill=False, ec="#D62728", lw=2.2)
    ax.add_patch(rect)

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            v = float(cmn[i, j])
            txt = f"{v:.3f}\n(n={int(cm[i, j])})"
            color = "white" if v >= 0.58 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color=color, fontweight="semibold")

    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", direction="out", top=False, right=False)

    fig.tight_layout()
    save_fig(fig, out_no_ext)


def _draw_cm_counts(cm, setting: str, method: str, out_no_ext: str):
    vmax = float(np.max(cm)) if np.size(cm) > 0 else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    im = ax.imshow(cm, cmap="Greens", vmin=0.0, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell count", fontsize=11)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Confusion (counts) | {SETTING_LABEL.get(setting, setting)} | {method}", fontsize=13, pad=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            frac = float(v) / vmax
            color = "white" if frac >= 0.58 else "#222222"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11, color=color, fontweight="semibold")

    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", direction="out", top=False, right=False)

    fig.tight_layout()
    save_fig(fig, out_no_ext)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=BENCHMARK_ROOT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    rows = []
    labels = np.arange(len(CLASS_NAMES), dtype=int)
    eps_idx = CLASS_NAMES.index(EPS_CLASS)

    for setting in SETTING_ORDER:
        p_npz = os.path.join(args.benchmark_root, setting, "benchmark_probs.npz")
        if not os.path.isfile(p_npz):
            print(f"[WARN] missing npz: {p_npz}")
            continue
        z = np.load(p_npz, allow_pickle=True)
        if "y_true" not in z:
            print(f"[WARN] missing y_true in {p_npz}")
            continue
        y_true = np.asarray(z["y_true"], dtype=int)
        auc_map = _load_auc_per_class(args.benchmark_root, setting)

        for method in METHODS:
            key = METHOD_KEY_IN_NPZ[method]
            if key not in z:
                print(f"[WARN] missing key {key} in {p_npz}")
                continue

            prob = np.asarray(z[key], dtype=float)
            if prob.ndim != 2 or prob.shape[1] != len(CLASS_NAMES):
                print(f"[WARN] unexpected prob shape for {setting} {method}: {prob.shape}")
                continue

            y_pred = np.argmax(prob, axis=1).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            acc = float(accuracy_score(y_true, y_pred))
            row_sum = cm.sum(axis=1)
            rec = np.divide(np.diag(cm), np.clip(row_sum, 1e-12, None))
            eps_recall = float(rec[eps_idx])
            eps_auc = float(auc_map.get(method, {}).get(EPS_CLASS, np.nan))

            rows.append(
                {
                    "Setting": setting,
                    "Time": SETTING_LABEL.get(setting, setting),
                    "Method": method,
                    "Accuracy_from_argmax": acc,
                    "Alpha_recall": float(rec[0]),
                    "Beta_recall": float(rec[1]),
                    "Delta_recall": float(rec[2]),
                    "Epsilon_recall": eps_recall,
                    "Epsilon_AUC": eps_auc,
                }
            )

            safe_method = method.replace(" ", "_")
            out_row = os.path.join(args.out_dir, f"CM_{setting}_{safe_method}_rowNorm_full")
            _draw_cm_row_norm(cm, setting, method, eps_recall, eps_auc, out_row)

            out_cnt = os.path.join(args.out_dir, f"CM_{setting}_{safe_method}_counts_full")
            _draw_cm_counts(cm, setting, method, out_cnt)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No confusion matrices generated. Please check inputs.")

    # Rank by Epsilon recall to expose the gap with AUC.
    df = df.sort_values(["Setting", "Epsilon_recall", "Epsilon_AUC"], ascending=[True, False, False]).reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "Confusion_4Methods_Epsilon_vs_AUC_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"[DONE] output dir: {args.out_dir}")
    print(f"[DONE] summary: {out_csv}")
    print(f"[DONE] generated matrices: {len(df)} methods-settings pairs x 2 styles (rowNorm/counts)")


if __name__ == "__main__":
    main()

