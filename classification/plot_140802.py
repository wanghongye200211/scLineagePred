# -*- coding: utf-8 -*-
"""
GSE140802 - plotting from official benchmark probabilities.

Requires benchmark files:
  <BENCHMARK_ROOT>/<setting>/benchmark_probs.npz
with keys:
  y_true, p_scLineagetracer, p_CellRank, p_WOT, p_CoSpar, ...

Outputs:
  <OUT_DIR>/roc_click/official_methods_plot_v2/ROC_<setting>_scLineagetracer_full|clean.(png|pdf)
  <OUT_DIR>/roc_click/official_methods_plot_v2/2d/Pred2D_<setting>_<MethodA>vs<MethodB>_full|clean.(png|pdf)
  <OUT_DIR>/roc_click/plot_summary_official_methods.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss, precision_recall_curve, confusion_matrix


# ============================================================
# ✅ 配置区（你一般只改这里）
# ============================================================
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7"
BENCHMARK_ROOT = os.path.join(OUT_DIR, "roc_click", "benchmark_all_timepoints_official_plus")

POS_LABEL = "Monocyte"
NEG_LABEL = "Neutrophil"

SEEDS = {
    "Day2_Only": 42,
    "Day2_Day4": 2024,
    "All_Days": 2024,
}

# 逐个输出 ROC + 2D 的 setting
SELECT_SETTINGS = ["Day2_Only", "Day2_Day4", "All_Days"]

# ✅ 合并 ROC：你自己选哪些（空列表=不画）
PLOT_SELECTED_ROC = True
SELECTED_ROC_SETTINGS = ["Day2_Only", "Day2_Day4", "All_Days"]  # <- 改这里

# 输出目录
ROC_DIR = os.path.join(OUT_DIR, "roc_click", "official_methods_plot_v2")
D2_DIR  = os.path.join(ROC_DIR, "2d")

# -------- 2D 平面：用 scLineagetracer 对比其他方法--------
PLANE_PAIRS = [
    ("scLineagetracer", "CellRank"),
    ("scLineagetracer", "WOT"),
    ("scLineagetracer", "CoSpar"),
    ("scLineagetracer", "GAN-based OT"),
]

# 2D 抽样点数（<=0=全量）
N_POINTS_2D = 6000
SAMPLE_SEED_2D = 2026

# 点更大（你当前统一格式）
POINT_SIZE_NEG = 26   # 蓝色方块
POINT_SIZE_POS = 28   # 红色圆圈
ALPHA_2D = 0.70

# 坐标轴文字
AXIS_LABEL_MODE = "paper"  # "paper" / "symbol"

# 是否显示 ROC 标题（114412 默认显示）
SHOW_ROC_TITLE = True
# ============================================================


# ============================================================
# 114412 ROC style
# ============================================================
GREY = "#444444"
LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e", "#7f7f7f"]

COLOR_NEG = "#1f77b4"
COLOR_POS = "#d62728"
MARK_NEG = "s"
MARK_POS = "o"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _style_axes_114412(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)


def _style_axes_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    # start (0,0)
    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)

    # end (1,1)
    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)

    # compress x==1 duplicates (avoid right vertical)
    mask_one = (fpr == 1.0)
    if np.any(mask_one):
        t1 = float(np.max(tpr[mask_one]))
        keep = ~mask_one
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]

    return fpr, tpr


def _safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return np.nan
    return float(num) / den


def _auprc_trapz(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # sklearn returns recall from 1 -> 0; reverse to make x-axis increasing for trapezoid AUC.
    return float(auc(recall[::-1], precision[::-1]))


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    y_pred = (p >= float(threshold)).astype(np.int64)

    fpr, tpr, _ = roc_curve(y_true, p)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    auroc = float(auc(fpr, tpr))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = np.nan

    acc = float(accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, p))
    auprc = _auprc_trapz(y_true, p)

    return {
        "AUC": auroc,            # backward compatibility
        "AUROC": auroc,
        "Accuracy": acc,         # backward compatibility
        "ACC": acc,
        "AUPRC": float(auprc),
        "F1": float(f1) if np.isfinite(f1) else np.nan,
        "Specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "Precision": float(precision) if np.isfinite(precision) else np.nan,
        "Recall": float(recall) if np.isfinite(recall) else np.nan,
        "LogLoss": ll,
        "Threshold": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def plot_binary_roc_114412(y_true, y_prob, title, out_no_ext, color="#1f77b4"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr, tpr = _roc_endpoints_clean(fpr, tpr)
    a = auc(fpr, tpr)

    def _draw(ax, clean):
        ax.plot(fpr, tpr, lw=2.8, color=color, label=f"AUC={a:.3f}")
        ax.plot([0, 1], [0, 1], lw=1.5, color=GREY, ls="--", alpha=0.55)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        if clean:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            _style_axes_clean(ax)
        else:
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            if SHOW_ROC_TITLE:
                ax.set_title(title, fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=10)
            _style_axes_114412(ax)
        ax.grid(False)

    fig_full, ax_full = plt.subplots(figsize=(6.8, 6.1))
    _draw(ax_full, clean=False)
    fig_full.tight_layout()
    save_fig(fig_full, out_no_ext + "_full")

    fig_clean, ax_clean = plt.subplots(figsize=(6.8, 6.1))
    _draw(ax_clean, clean=True)
    fig_clean.tight_layout()
    save_fig(fig_clean, out_no_ext + "_clean")
    return float(a), fpr, tpr


def plot_selected_rocs_114412(ordered_curve_dict, title, out_no_ext):
    def _draw(ax, clean):
        ax.plot([0, 1], [0, 1], lw=1.4, color=GREY, ls="--", alpha=0.5)
        for i, (s, (fpr, tpr, a)) in enumerate(ordered_curve_dict.items()):
            ax.plot(fpr, tpr, lw=2.7, color=LINE_COLORS[i % len(LINE_COLORS)], label=f"{s} (AUC={a:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        if clean:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            _style_axes_clean(ax)
        else:
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            if SHOW_ROC_TITLE:
                ax.set_title(title, fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
            _style_axes_114412(ax)
        ax.grid(False)

    fig_full, ax_full = plt.subplots(figsize=(7.2, 6.3))
    _draw(ax_full, clean=False)
    fig_full.tight_layout()
    save_fig(fig_full, out_no_ext + "_full")

    fig_clean, ax_clean = plt.subplots(figsize=(7.2, 6.3))
    _draw(ax_clean, clean=True)
    fig_clean.tight_layout()
    save_fig(fig_clean, out_no_ext + "_clean")


# ============================================================
# 2D prob-plane (two models' p(pos), dashed y=x)
# ============================================================
def axis_label(tag: str) -> str:
    if AXIS_LABEL_MODE == "paper":
        return f"Predicted probability of {POS_LABEL} ({tag})"
    return f"{tag} p({POS_LABEL})"


def pick_axis_values(axis_name, probs: dict):
    if axis_name not in probs:
        raise ValueError(f"[ERROR] axis '{axis_name}' not in {list(probs.keys())}")
    tag = axis_name
    return probs[axis_name], axis_label(tag)


def plot_2d_prob_plane(xv, yv, y_true, out_no_ext, xlab, ylab):
    xv = np.clip(np.asarray(xv, dtype=np.float64), 0.0, 1.0)
    yv = np.clip(np.asarray(yv, dtype=np.float64), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=np.int64)

    # subsample
    if N_POINTS_2D is not None and int(N_POINTS_2D) > 0 and len(y_true) > int(N_POINTS_2D):
        rng = np.random.default_rng(SAMPLE_SEED_2D)
        idx = rng.choice(np.arange(len(y_true)), size=int(N_POINTS_2D), replace=False)
        xv, yv, y_true = xv[idx], yv[idx], y_true[idx]

    m0 = (y_true == 0)  # NEG
    m1 = (y_true == 1)  # POS

    def _draw(ax, clean):
        # quadrant separators to emphasize 0.5 decision boundary
        ax.axvline(0.5, lw=1.6, color=GREY, ls="--", alpha=0.60, zorder=1)
        ax.axhline(0.5, lw=1.6, color=GREY, ls="--", alpha=0.60, zorder=1)
        ax.scatter(xv[m0], yv[m0], s=POINT_SIZE_NEG, alpha=ALPHA_2D, c=COLOR_NEG, marker=MARK_NEG, edgecolors="none", label=NEG_LABEL, zorder=2)
        ax.scatter(xv[m1], yv[m1], s=POINT_SIZE_POS, alpha=ALPHA_2D, c=COLOR_POS, marker=MARK_POS, edgecolors="none", label=POS_LABEL, zorder=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        ax.set_aspect("equal", adjustable="box")
        if clean:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            _style_axes_clean(ax)
        else:
            ax.set_xlabel(xlab, fontsize=13)
            ax.set_ylabel(ylab, fontsize=13)
            ax.legend(loc="upper left", frameon=False, fontsize=10)
            _style_axes_114412(ax)
        ax.grid(False)

    fig_full, ax_full = plt.subplots(figsize=(6.4, 6.4))
    _draw(ax_full, clean=False)
    fig_full.tight_layout()
    save_fig(fig_full, out_no_ext + "_full")

    fig_clean, ax_clean = plt.subplots(figsize=(6.4, 6.4))
    _draw(ax_clean, clean=True)
    fig_clean.tight_layout()
    save_fig(fig_clean, out_no_ext + "_clean")


# ============================================================
# Benchmark IO
# ============================================================
def sanitize_key(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out).strip("_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def load_benchmark_probs(setting: str):
    npz_path = os.path.join(BENCHMARK_ROOT, setting, "benchmark_probs.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"[ERROR] missing benchmark probs: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    if "y_true" not in z:
        raise KeyError(f"[ERROR] y_true not found in {npz_path}")
    y_true = z["y_true"].astype(np.int64)

    method_order = [
        "scLineagetracer",
        "CellRank",
        "WOT",
        "CoSpar",
        "GAN-based OT",
    ]
    probs = {}
    for m in method_order:
        k = "p_" + sanitize_key(m)
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
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        if len(p) != len(y_true):
            raise ValueError(f"[ERROR] benchmark length mismatch: {setting} {m} len={len(p)} != y_true len={len(y_true)}")
        probs[m] = p

    if "scLineagetracer" not in probs:
        raise KeyError(f"[ERROR] scLineagetracer probs not found in {npz_path}")

    return npz_path, y_true, probs


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(ROC_DIR)
    ensure_dir(D2_DIR)

    selected_rocs = {}
    summary_rows = []

    for i, setting in enumerate(SELECT_SETTINGS):
        if setting not in SEEDS:
            raise ValueError(f"[ERROR] missing seed for setting: {setting}")
        seed = int(SEEDS[setting])

        bench_path, y_true, probs = load_benchmark_probs(setting)
        p_sc = probs["scLineagetracer"]
        target_methods = ["CellRank", "WOT", "CoSpar", "GAN-based OT"]
        missing_methods = [m for m in target_methods if m not in probs]

        # metrics for scLineagetracer
        metrics = compute_binary_metrics(y_true, p_sc, threshold=0.5)

        # ROC (scLineagetracer)
        roc_out = os.path.join(ROC_DIR, f"ROC_{setting}_scLineagetracer")
        auc_val, fpr, tpr = plot_binary_roc_114412(
            y_true=y_true,
            y_prob=p_sc,
            title=f"scLineagetracer ROC — {setting}",
            out_no_ext=roc_out,
            color=LINE_COLORS[i % len(LINE_COLORS)]
        )
        metrics["AUC"] = float(auc_val)
        metrics["AUROC"] = float(auc_val)

        # 2D pairs (scLineagetracer vs other methods)
        for x_name, y_name in PLANE_PAIRS:
            if (x_name not in probs) or (y_name not in probs):
                print(f"[WARN] {setting} skip 2D pair: {x_name} vs {y_name} (missing method probs)")
                continue
            xv, xlab = pick_axis_values(x_name, probs)
            yv, ylab = pick_axis_values(y_name, probs)
            pair_tag = f"{sanitize_key(x_name)}vs{sanitize_key(y_name)}"
            d2_out = os.path.join(D2_DIR, f"Pred2D_{setting}_{pair_tag}")
            plot_2d_prob_plane(xv, yv, y_true, d2_out, xlab, ylab)

        print(f"[INFO] benchmark={os.path.basename(bench_path)}")
        print(
            f"[RESULT] {setting} | AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | "
            f"ACC={metrics['ACC']:.4f} | F1={metrics['F1']:.4f} | "
            f"Spec={metrics['Specificity']:.4f} | Prec={metrics['Precision']:.4f} | "
            f"Rec={metrics['Recall']:.4f} | LogLoss={metrics['LogLoss']:.4f}"
        )

        summary_rows.append({
            "Setting": setting,
            "Seed": seed,
            "AUC": float(metrics["AUC"]),
            "AUROC": float(metrics["AUROC"]),
            "Accuracy": float(metrics["Accuracy"]),
            "ACC": float(metrics["ACC"]),
            "AUPRC": float(metrics["AUPRC"]),
            "F1": float(metrics["F1"]),
            "Specificity": float(metrics["Specificity"]),
            "Precision": float(metrics["Precision"]),
            "Recall": float(metrics["Recall"]),
            "LogLoss": float(metrics["LogLoss"]),
            "Threshold": float(metrics["Threshold"]),
            "TP": int(metrics["TP"]),
            "TN": int(metrics["TN"]),
            "FP": int(metrics["FP"]),
            "FN": int(metrics["FN"]),
            "PlanePairs": ";".join([f"{a} vs {b}" for a, b in PLANE_PAIRS]),
            "AvailableMethods": ";".join(sorted(list(probs.keys()))),
            "MissingMethods": ";".join(missing_methods),
            "NPoints2D": int(N_POINTS_2D),
            "BenchmarkFile": os.path.basename(bench_path),
        })

        if PLOT_SELECTED_ROC and (setting in set(SELECTED_ROC_SETTINGS)):
            selected_rocs[setting] = (fpr, tpr, float(auc_val))

    # Selected ROC (keep your specified order)
    if PLOT_SELECTED_ROC and SELECTED_ROC_SETTINGS:
        ordered = {}
        for s in SELECTED_ROC_SETTINGS:
            if s in selected_rocs:
                ordered[s] = selected_rocs[s]
        if len(ordered) > 0:
            out_no_ext = os.path.join(ROC_DIR, "ROC_Selected_Comparison_scLineagetracer")
            plot_selected_rocs_114412(ordered, "Selected ROC Curves", out_no_ext)

    out_csv = os.path.join(ROC_DIR, "plot_summary_official_methods.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"[DONE] Summary saved: {out_csv}")
    print(f"[DONE] Outputs: {ROC_DIR}")


if __name__ == "__main__":
    main()
