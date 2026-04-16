#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requested GSE99915 figures (clean/full):
1) Day15 ROC comparison of 4 methods (scLineagetracer, CellRank, WOT, CoSpar).
2) Our-method ROC in one panel (Obs_Day21/Obs_Day15/Obs_Day12).
3) Our-method trend plot like reference image (Accuracy + Crossentropy Loss vs Day12/15/21/28).
4) Official 4-method accuracy-vs-time comparison (Day12/15/21).

Notes:
- Our-method values are computed on the selected mixed dataset:
  GSE99915_DeepLineage_Mixed_Natural200k_Masked40k.
- Four-method Day15 ROC / accuracy-vs-time use official benchmark outputs.
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss


# ---------- Paths ----------
MIX_H5 = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Mixed_Natural200k_Masked40k_sequences.h5"
MIX_CSV = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Mixed_Natural200k_Masked40k_index.csv"
BENCHMARK_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/benchmark_all_timepoints_official_plus"

OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/requested_plot_v1"

# Match official remaining-datasets plotting style (scLineagetracer in red)
METHOD_COLORS = {
    "scLineagetracer": "#d62728",
    "CellRank": "#1f77b4",
    "WOT": "#2ca02c",
    "CoSpar": "#9467bd",
}

MY_ROC_COLORS = {
    "Obs_Day12": "#59A14F",
    "Obs_Day15": "#F28E2B",
    "Obs_Day21": "#4E79A7",
}
BAR_COLORS = {
    "scLineagetracer": "#d62728",
    "OurMethod_mixed240k": "#ff7f0e",
    "CellRank": "#1f77b4",
    "WOT": "#2ca02c",
    "CoSpar": "#9467bd",
}

SETTINGS = {
    "All_Days": {"seed": 2026, "mask_idx": []},       # Day28 full
    "Obs_Day21": {"seed": 2024, "mask_idx": [5]},
    "Obs_Day15": {"seed": 42, "mask_idx": [4, 5]},
    "Obs_Day12": {"seed": 123, "mask_idx": [3, 4, 5]},
}
OFFICIAL_TIME_SETTINGS = ["Obs_Day12", "Obs_Day15", "Obs_Day21"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def style_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def clean_endpoints(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    # roc_curve already returns points ordered by threshold. Re-sorting by FPR
    # can scramble equal-FPR segments and introduce visual backtracking lines.
    if len(fpr) == 0 or (not np.isclose(fpr[0], 0.0)) or (not np.isclose(tpr[0], 0.0)):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (not np.isclose(fpr[-1], 1.0)) or (not np.isclose(tpr[-1], 1.0)):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    return np.clip(fpr, 0, 1), np.clip(tpr, 0, 1)


def load_mix_data():
    with h5py.File(MIX_H5, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        M = np.array(f["mask"], dtype=np.float32)
    y = (pd.read_csv(MIX_CSV)["label_str"].astype(str).values == "Reprogrammed").astype(np.int64)
    return X, M, y


def run_our_method_probs(X, M, y, seed, mask_idx):
    Xs = X.copy()
    Ms = M.copy()
    for i in mask_idx:
        Xs[:, i, :] = 0.0
        Ms[:, i] = 0.0
    Xf = np.concatenate([Xs.reshape(len(Xs), -1), Ms], axis=1)

    all_idx = np.arange(len(y))
    tr, tmp = train_test_split(all_idx, test_size=0.2, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])

    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed,
    )
    model.fit(Xf[tr], y[tr])
    p_te = model.predict_proba(Xf[te])[:, 1]

    fpr, tpr, _ = roc_curve(y[te], p_te)
    fpr, tpr = clean_endpoints(fpr, tpr)
    a = float(auc(fpr, tpr))
    acc = float(accuracy_score(y[te], p_te >= 0.5))
    loss = float(log_loss(y[te], np.clip(p_te, 1e-6, 1 - 1e-6)))

    return {
        "y_true": y[te],
        "y_prob": p_te,
        "fpr": fpr,
        "tpr": tpr,
        "auc": a,
        "acc": acc,
        "logloss": loss,
    }


def plot_day15_4methods(our_day15=None):
    bench_day15_npz = os.path.join(BENCHMARK_ROOT, "Obs_Day15", "benchmark_probs.npz")
    data = np.load(bench_day15_npz, allow_pickle=True)
    y_true = np.asarray(data["y_true"]).astype(np.int64)

    curves = {}
    key_map = {
        "scLineagetracer": "p_scLineagetracer",
        "CellRank": "p_CellRank",
        "WOT": "p_WOT",
        "CoSpar": "p_CoSpar",
    }
    for m, k in key_map.items():
        if m == "scLineagetracer" and our_day15 is not None:
            curves[m] = {
                "fpr": np.asarray(our_day15["fpr"], dtype=np.float64),
                "tpr": np.asarray(our_day15["tpr"], dtype=np.float64),
                "auc": float(our_day15["auc"]),
                "source": "mixed240k_local_eval",
            }
            continue
        if k not in data.files:
            continue
        arr = np.asarray(data[k])
        p = arr[:, 1] if arr.ndim == 2 else arr
        fpr, tpr, _ = roc_curve(y_true, p)
        fpr, tpr = clean_endpoints(fpr, tpr)
        curves[m] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)), "source": "official_benchmark"}

    def draw(clean):
        fig, ax = plt.subplots(figsize=(7.2, 6.3))
        for m in ["scLineagetracer", "CellRank", "WOT", "CoSpar"]:
            if m not in curves:
                continue
            c = curves[m]
            label = None if clean else f"{m} (AUC={c['auc']:.3f})"
            ax.plot(c["fpr"], c["tpr"], lw=2.7, color=METHOD_COLORS[m], label=label)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        ax.grid(False)
        if clean:
            style_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.set_title("ROC Comparison (4 Methods) - Obs Day15", fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.tight_layout()
        suffix = "_clean" if clean else "_full"
        save_fig(fig, os.path.join(OUT_DIR, f"ROC_Obs_Day15_4Methods{suffix}"))

    draw(clean=False)
    draw(clean=True)

    rows = [{"method": m, "auc": curves[m]["auc"], "source": curves[m].get("source", "")} for m in curves]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "ROC_Obs_Day15_4Methods_values.csv"), index=False)


def plot_official_4methods_accuracy_vs_time(results):
    method_order = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
    day_labels = {"Obs_Day12": "Day12", "Obs_Day15": "Day15", "Obs_Day21": "Day21"}

    rows = []
    for setting in OFFICIAL_TIME_SETTINGS:
        p = os.path.join(BENCHMARK_ROOT, setting, "metrics_summary.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        # Keep CellRank/WOT/CoSpar from official benchmark.
        for m in ["CellRank", "WOT", "CoSpar"]:
            hit = df[df["Method"] == m]
            if len(hit) == 0:
                continue
            rows.append(
                {
                    "setting": setting,
                    "day": day_labels[setting],
                    "method": m,
                    "accuracy": float(hit.iloc[0]["Accuracy"]),
                    "source": "official_benchmark",
                }
            )

        # Use our mixed240k accuracy for scLineagetracer to stay consistent
        # with ROC_OurMethod_ObsDay12_15_21 and Day15 4-method ROC.
        if setting in results:
            rows.append(
                {
                    "setting": setting,
                    "day": day_labels[setting],
                    "method": "scLineagetracer",
                    "accuracy": float(results[setting]["acc"]),
                    "source": "mixed240k_local_eval",
                }
            )
        else:
            hit_sc = df[df["Method"] == "scLineagetracer"]
            if len(hit_sc) > 0:
                rows.append(
                    {
                        "setting": setting,
                        "day": day_labels[setting],
                        "method": "scLineagetracer",
                        "accuracy": float(hit_sc.iloc[0]["Accuracy"]),
                        "source": "official_benchmark",
                    }
                )

    acc_df = pd.DataFrame(rows)
    if len(acc_df) == 0:
        return
    acc_df["day_order"] = acc_df["day"].map({"Day12": 12, "Day15": 15, "Day21": 21})
    acc_df = acc_df.sort_values(["day_order", "method"]).reset_index(drop=True)

    def draw(clean):
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        x_days = ["Day12", "Day15", "Day21"]
        x = np.arange(len(x_days), dtype=np.float64)
        width = 0.2
        offsets = (np.arange(len(method_order)) - (len(method_order) - 1) / 2.0) * width

        for i, m in enumerate(method_order):
            vals = []
            for d in x_days:
                hit = acc_df[(acc_df["day"] == d) & (acc_df["method"] == m)]
                vals.append(float(hit.iloc[0]["accuracy"]) if len(hit) > 0 else np.nan)
            label = None if clean else m
            ax.bar(
                x + offsets[i],
                vals,
                width=width * 0.9,
                color=METHOD_COLORS[m],
                edgecolor="none",
                alpha=0.95,
                label=label,
            )

        ymin = float(acc_df["accuracy"].min()) - 0.03
        # Increase top headroom for both full/clean figures.
        ymax = 1.03
        ax.set_ylim(max(0.0, ymin), ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(x_days)
        ax.grid(False)

        if clean:
            style_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("Fate Prediction Using Data Until Day", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=13)
            ax.set_title("Official Methods Accuracy Comparison (Day12/15/21)", fontsize=13, pad=10)
            ax.legend(loc="best", frameon=False, fontsize=10)

        fig.tight_layout()
        suffix = "_clean" if clean else "_full"
        save_fig(fig, os.path.join(OUT_DIR, f"Accuracy_4Methods_Day12_15_21{suffix}"))

    draw(clean=False)
    draw(clean=True)
    acc_df.drop(columns=["day_order"]).to_csv(
        os.path.join(OUT_DIR, "Accuracy_4Methods_Day12_15_21_values.csv"), index=False
    )


def plot_auc_grouped_bar(results):
    """
    Grouped bar chart like requested example:
    x-axis groups = Day12/Day15/Day21
    bars = official 4 methods + our mixed240k method.
    """
    method_order = ["scLineagetracer", "OurMethod_mixed240k", "CellRank", "WOT", "CoSpar"]
    day_labels = {"Obs_Day12": "Day12", "Obs_Day15": "Day15", "Obs_Day21": "Day21"}
    day_order_map = {"Day12": 12, "Day15": 15, "Day21": 21}

    rows = []
    # official 4 methods
    for setting in OFFICIAL_TIME_SETTINGS:
        p = os.path.join(BENCHMARK_ROOT, setting, "metrics_summary.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        for m in ["scLineagetracer", "CellRank", "WOT", "CoSpar"]:
            hit = df[df["Method"] == m]
            if len(hit) == 0:
                continue
            rows.append(
                {
                    "setting": setting,
                    "day": day_labels[setting],
                    "method": m,
                    "auc": float(hit.iloc[0]["AUC"]),
                    "source": "official_benchmark",
                }
            )

    # our method from selected mixed dataset (same values used in ROC_OurMethod panel)
    for setting in OFFICIAL_TIME_SETTINGS:
        if setting not in results:
            continue
        rows.append(
            {
                "setting": setting,
                "day": day_labels[setting],
                "method": "OurMethod_mixed240k",
                "auc": float(results[setting]["auc"]),
                "source": "mixed240k_local_eval",
            }
        )

    auc_df = pd.DataFrame(rows)
    if len(auc_df) == 0:
        return
    auc_df["day_order"] = auc_df["day"].map(day_order_map)
    auc_df = auc_df.sort_values(["day_order", "method"]).reset_index(drop=True)

    x_days = ["Day12", "Day15", "Day21"]
    x = np.arange(len(x_days), dtype=np.float64)
    n_methods = len(method_order)
    width = 0.14
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    def draw(clean):
        fig, ax = plt.subplots(figsize=(7.9, 5.8))
        for i, m in enumerate(method_order):
            y = []
            for d in x_days:
                hit = auc_df[(auc_df["day"] == d) & (auc_df["method"] == m)]
                y.append(float(hit.iloc[0]["auc"]) if len(hit) > 0 else np.nan)
            label = None if clean else m
            ax.bar(
                x + offsets[i],
                y,
                width=width * 0.92,
                color=BAR_COLORS.get(m, "#666666"),
                edgecolor="none",
                label=label,
                alpha=0.95,
            )

        valid = auc_df["auc"].to_numpy(dtype=np.float64)
        ymin = max(0.0, float(np.nanmin(valid)) - 0.05)
        ymax = min(1.0, float(np.nanmax(valid)) + 0.02)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(x_days)
        ax.grid(False)

        if clean:
            style_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("Fate Prediction Using Data Until Day", fontsize=12)
            ax.set_ylabel("AUC", fontsize=13)
            ax.set_title("AUC Comparison (Official 4 Methods + Our Mixed240k)", fontsize=13, pad=10)
            ax.legend(loc="best", frameon=False, fontsize=9)

        fig.tight_layout()
        suffix = "_clean" if clean else "_full"
        save_fig(fig, os.path.join(OUT_DIR, f"AUC_5Methods_Day12_15_21{suffix}"))

    draw(clean=False)
    draw(clean=True)
    auc_df.drop(columns=["day_order"]).to_csv(
        os.path.join(OUT_DIR, "AUC_5Methods_Day12_15_21_values.csv"), index=False
    )


def plot_our_roc_panel(results):
    use_settings = ["Obs_Day21", "Obs_Day15", "Obs_Day12"]

    def draw(clean):
        fig, ax = plt.subplots(figsize=(7.2, 6.3))
        for s in use_settings:
            c = results[s]
            label = None if clean else f"{s} (AUC={c['auc']:.3f})"
            ax.plot(c["fpr"], c["tpr"], lw=2.8, color=MY_ROC_COLORS[s], label=label)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        ax.grid(False)
        if clean:
            style_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.set_title("Our Method ROC (Obs Day12/15/21)", fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.tight_layout()
        suffix = "_clean" if clean else "_full"
        save_fig(fig, os.path.join(OUT_DIR, f"ROC_OurMethod_ObsDay12_15_21{suffix}"))

    draw(clean=False)
    draw(clean=True)


def plot_our_trend(results):
    days = ["Day12", "Day15", "Day21", "Day28"]
    setting_map = {
        "Day12": "Obs_Day12",
        "Day15": "Obs_Day15",
        "Day21": "Obs_Day21",
        "Day28": "All_Days",
    }
    acc = [results[setting_map[d]]["acc"] for d in days]
    loss = [results[setting_map[d]]["logloss"] for d in days]

    def draw(clean):
        fig, ax1 = plt.subplots(figsize=(7.6, 5.8))
        c_acc = "#D83A3A"
        c_loss = "#3B4FA1"

        ax1.plot(
            days, acc, color=c_acc, lw=3, marker="o", markersize=9,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor=c_acc,
            label="Accuracy"
        )
        ax1.set_ylim(min(acc) - 0.01, min(1.0, max(acc) + 0.01))
        ax2 = ax1.twinx()
        ax2.plot(
            days, loss, color=c_loss, lw=3, marker="^", markersize=9,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor=c_loss,
            label="Crossentropy Loss"
        )
        ax2.set_ylim(max(0.0, min(loss) - 0.02), max(loss) + 0.02)

        if clean:
            style_clean(ax1)
            style_clean(ax2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax1.set_title("")
        else:
            ax1.set_xlabel("Fate Prediction Using Data Until Day", fontsize=12)
            ax1.set_ylabel("Accuracy", fontsize=13)
            ax2.set_ylabel("Crossentropy Loss", fontsize=13)
            ax1.tick_params(axis="both", direction="out", top=False, right=False)
            ax2.tick_params(axis="both", direction="out", top=False, right=False)
            ax1.legend(loc="upper left", frameon=False, fontsize=10)
            ax2.legend(loc="upper right", frameon=False, fontsize=10)

        ax1.grid(False)
        ax2.grid(False)
        fig.tight_layout()
        suffix = "_clean" if clean else "_full"
        save_fig(fig, os.path.join(OUT_DIR, f"Trend_OurMethod_AccLogLoss{suffix}"))

    draw(clean=False)
    draw(clean=True)

    rows = []
    for d in days:
        s = setting_map[d]
        rows.append(
            {
                "day": d,
                "setting": s,
                "accuracy": results[s]["acc"],
                "crossentropy_loss": results[s]["logloss"],
                "auc": results[s]["auc"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "Trend_OurMethod_values.csv"), index=False)


def main():
    ensure_dir(OUT_DIR)
    X, M, y = load_mix_data()

    # compute our-method curves/metrics on selected mixed dataset
    results = {}
    for s, cfg in SETTINGS.items():
        results[s] = run_our_method_probs(X, M, y, seed=cfg["seed"], mask_idx=cfg["mask_idx"])

    # save summary for quick check
    summary_rows = []
    for s in ["Obs_Day21", "Obs_Day15", "Obs_Day12", "All_Days"]:
        summary_rows.append(
            {
                "setting": s,
                "seed": SETTINGS[s]["seed"],
                "acc_05": results[s]["acc"],
                "auc": results[s]["auc"],
                "logloss": results[s]["logloss"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "OurMethod_mixed240k_metrics_summary.csv"), index=False
    )

    # requested figures
    plot_day15_4methods(our_day15=results.get("Obs_Day15"))
    plot_official_4methods_accuracy_vs_time(results)
    plot_our_roc_panel(results)
    plot_our_trend(results)

    print(f"[DONE] outputs in: {OUT_DIR}")
    print("[DONE] key metrics:")
    for r in summary_rows:
        print(
            f"  {r['setting']}: acc={r['acc_05']:.6f}, auc={r['auc']:.6f}, logloss={r['logloss']:.6f}"
        )


if __name__ == "__main__":
    main()
