from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve


def save_plot(fig, path_no_ext: str) -> None:
    fig.savefig(f"{path_no_ext}.pdf", format="pdf", bbox_inches=None, pad_inches=0.0, dpi=300)
    fig.savefig(f"{path_no_ext}.png", format="png", bbox_inches=None, pad_inches=0.0, dpi=300)
    print(f"   [Plot] Saved: {path_no_ext}.png")


def ensure_roc_endpoints(fpr: np.ndarray, tpr: np.ndarray):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    if (fpr[0] > 0.0) or (tpr[0] > 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (fpr[-1] < 1.0) or (tpr[-1] < 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    return np.clip(fpr, 0.0, 1.0), np.clip(tpr, 0.0, 1.0)


def plot_setting_roc_ovr_macro_style(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    setting: str,
    out_dir: str,
):
    color_map = {
        "Alpha": "#4C72B0",
        "Beta": "#DD8452",
        "Delta": "#55A868",
        "Epsilon": "#C44E52",
    }
    fallback = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    for idx, class_name in enumerate(class_names):
        y_bin = (y_true == idx).astype(np.int64)
        if np.unique(y_bin).size < 2:
            print(f"[WARN] Skip ROC for {setting}/{class_name}: test split contains only one class.")
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, idx])
        fpr, tpr = ensure_roc_endpoints(fpr, tpr)
        auc_value = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=3, color=color_map.get(class_name, fallback[idx % len(fallback)]), label=f"AUC={auc_value:.4f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(False)
        ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
        ax.set_title(f"ROC - {setting} - {class_name}", fontsize=14, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=12, frameon=False)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
        save_plot(fig, os.path.join(out_dir, f"ROC_{setting}_{class_name}"))
        plt.close(fig)

    fpr_list = []
    tpr_list = []
    for idx in range(len(class_names)):
        y_bin = (y_true == idx).astype(np.int64)
        if np.unique(y_bin).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, idx])
        fpr, tpr = ensure_roc_endpoints(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    if not fpr_list:
        raise ValueError(f"Unable to compute ROC curves for setting {setting}: every class is missing in the test split.")

    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for idx in range(len(fpr_list)):
        mean_tpr += np.interp(all_fpr, fpr_list[idx], tpr_list[idx])
    mean_tpr /= float(len(class_names))
    all_fpr, mean_tpr = ensure_roc_endpoints(all_fpr, mean_tpr)
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(all_fpr, mean_tpr, lw=3, color="#777777", label=f"Macro AUC={macro_auc:.4f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title(f"ROC - {setting} - Macro", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
    save_plot(fig, os.path.join(out_dir, f"ROC_{setting}_Macro"))
    plt.close(fig)

    pd.DataFrame({"setting": setting, "fpr": all_fpr, "tpr": mean_tpr, "auc": macro_auc}).to_csv(
        os.path.join(out_dir, f"ROC_{setting}_Macro_points.csv"),
        index=False,
    )
    return all_fpr, mean_tpr, macro_auc


def plot_macro_roc_all_settings_style(
    macro_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    order: List[str],
    out_dir: str,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#E24A33", "#348ABD", "#988ED5", "#55A868", "#DD8452", "#777777"]

    for idx, setting in enumerate(order):
        if setting not in macro_curves:
            continue
        fpr, tpr, auc_value = macro_curves[setting]
        fpr, tpr = ensure_roc_endpoints(fpr, tpr)
        ax.plot(fpr, tpr, lw=3, color=colors[idx % len(colors)], label=f"{setting} (AUC={auc_value:.3f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title("Macro-average ROC (Ensemble) Across Timepoints", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
    save_plot(fig, os.path.join(out_dir, "ROC_Macro_AllTimepoints_Ensemble"))
    plt.close(fig)

    rows = []
    for setting in order:
        if setting not in macro_curves:
            continue
        fpr, tpr, auc_value = macro_curves[setting]
        for fpr_value, tpr_value in zip(fpr, tpr):
            rows.append(
                {
                    "setting": setting,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "auc": float(auc_value),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ROC_Macro_AllTimepoints_Ensemble_points.csv"), index=False)


def plot_performance_trend(results_buffer: Dict[str, dict], order: List[str], x_labels: List[str], out_dir: str):
    xs = []
    accs = []
    losses = []
    for setting, x_label in zip(order, x_labels):
        if setting in results_buffer:
            xs.append(x_label)
            accs.append(results_buffer[setting]["acc"])
            losses.append(results_buffer[setting]["loss"])

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(xs, accs, marker="o", markersize=10, lw=3)
    ax2 = ax1.twinx()
    ax2.plot(xs, losses, marker="^", markersize=10, lw=3)

    ax1.set_xlabel("Observation End Point", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Log Loss", fontsize=14, fontweight="bold")
    ax1.tick_params(direction="in", top=True)
    ax2.tick_params(direction="in", right=True)
    ax1.grid(False)
    ax2.grid(False)

    plt.title("Performance Trend (Ensemble)", fontsize=16, fontweight="bold", pad=15)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.90)
    save_plot(fig, os.path.join(out_dir, "Performance_Trend"))
    plt.close(fig)


def plot_3d_pred_points_pca_true_coded(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    out_prefix: str,
    max_points: int,
    seed: int,
    alpha: float,
    size: float,
):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float32)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    idx = np.arange(len(y_true), dtype=np.int64)
    if len(idx) > max_points:
        keep = []
        for class_idx in range(len(class_names)):
            class_idx_values = idx[y_true == class_idx]
            if len(class_idx_values) == 0:
                continue
            quota = int(round(max_points * (len(class_idx_values) / float(len(idx)))))
            quota = max(1, min(quota, len(class_idx_values)))
            keep.append(rng.choice(class_idx_values, size=quota, replace=False))
        idx = np.unique(np.concatenate(keep)) if keep else rng.choice(idx, size=max_points, replace=False)
        y_true = y_true[idx]
        probs = probs[idx]

    pca = PCA(n_components=max(1, min(3, probs.shape[1], probs.shape[0])), random_state=seed)
    embedding = pca.fit_transform(probs)
    if embedding.shape[1] < 3:
        embedding = np.pad(embedding, ((0, 0), (0, 3 - embedding.shape[1])), mode="constant")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    markers = ["o", "^", "s", "D", "P", "X"]

    fig = plt.figure(figsize=(9, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=size,
            c=colors[class_idx % len(colors)],
            marker=markers[class_idx % len(markers)],
            alpha=alpha,
            edgecolors="k",
            linewidths=0.15,
            label=f"True={class_name} (n={int(mask.sum())})",
        )

    ax.set_xlabel("PC1", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("PC2", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_zlabel("PC3", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor("black")
        except Exception:
            pass
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    save_plot(fig, out_prefix)
    plt.close(fig)

    table = pd.DataFrame(
        {
            "y_true": y_true.astype(np.int64),
            "label_true": np.array([class_names[idx] for idx in y_true], dtype=object),
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "pc3": embedding[:, 2],
            "p_max": probs.max(axis=1),
            "pred_label": np.array([class_names[int(idx)] for idx in probs.argmax(axis=1)], dtype=object),
        }
    )
    for class_idx, class_name in enumerate(class_names):
        table[f"p_{class_name}"] = probs[:, class_idx]
    table.to_csv(out_prefix + "_pred3d_table.csv", index=False)
    print(f"   [Table] Saved: {out_prefix}_pred3d_table.csv")
