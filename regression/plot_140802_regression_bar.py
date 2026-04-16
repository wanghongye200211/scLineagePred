# -*- coding: utf-8 -*-
"""
GSE140802 regression bar summary (R2).

Reads per-task scatter summaries and draws a single 6-bar chart:
  D4-Monocyte, D4-Neutrophil, D4-Mean, D6-Monocyte, D6-Neutrophil, D6-Mean

Outputs:
  - Regression_R2_Bar_140802_values.csv
  - Regression_R2_Bar_140802_full.png/.pdf
  - Regression_R2_Bar_140802_clean.png/.pdf
"""

import os
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE140802"
OUT_DIR = os.path.join(ROOT_DIR, "regression_bar_v1")

TASKS = [
    ("Reg_D4_from_D2_D6", "D4"),
    ("Reg_D6_from_D2_D4", "D6"),
]

CLASS_ORDER = ["Monocyte", "Neutrophil"]
BAR_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_no_ext: str):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _read_one_task(root_dir: str, task: str, day_label: str) -> List[Dict]:
    csv_path = os.path.join(root_dir, task, "plots_template_v1", "scatter_r2_summary.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing scatter summary: {csv_path}")

    df = pd.read_csv(csv_path)
    if ("class" not in df.columns) or ("r2" not in df.columns):
        raise KeyError(f"CSV must contain columns: class, r2 -> {csv_path}")

    rows: List[Dict] = []
    vals = []
    for cls in CLASS_ORDER:
        sub = df[df["class"].astype(str) == cls]
        if len(sub) == 0:
            raise ValueError(f"Class '{cls}' not found in {csv_path}")
        r2 = float(sub["r2"].iloc[0])
        vals.append(r2)
        rows.append(
            {
                "task": task,
                "day": day_label,
                "class": cls,
                "group_label": f"{day_label}-{cls}",
                "r2": r2,
            }
        )

    rows.append(
        {
            "task": task,
            "day": day_label,
            "class": "Mean",
            "group_label": f"{day_label}-Mean",
            "r2": float(np.mean(vals)),
        }
    )
    return rows


def read_r2_table(root_dir: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for task, day in TASKS:
        rows.extend(_read_one_task(root_dir, task, day))

    df = pd.DataFrame(rows)
    order = [f"{d}-{c}" for _, d in TASKS for c in ["Monocyte", "Neutrophil", "Mean"]]
    df["group_label"] = pd.Categorical(df["group_label"], categories=order, ordered=True)
    df = df.sort_values("group_label").reset_index(drop=True)
    df["r2_percent"] = df["r2"] * 100.0
    return df


def draw_bar(df: pd.DataFrame, out_no_ext: str, clean: bool):
    labels = df["group_label"].astype(str).tolist()
    y = df["r2_percent"].to_numpy(dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    bars = ax.bar(
        x,
        y,
        width=0.76,
        color=BAR_COLORS[: len(labels)],
        edgecolor="white",
        linewidth=0.8,
    )

    y_min = max(0.0, float(np.floor(np.nanmin(y) - 1.0)))
    y_max = min(100.0, float(np.ceil(np.nanmax(y) + 1.2)))
    if y_max <= y_min:
        y_min, y_max = 0.0, 100.0
    ax.set_ylim(y_min, y_max)

    if clean:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    else:
        ax.set_title("GSE140802 Regression R2 Summary", fontsize=13, pad=10)
        ax.set_xlabel("Task-Class", fontsize=11)
        ax.set_ylabel("R2 (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

        for rect, val in zip(bars, y):
            ax.text(
                rect.get_x() + rect.get_width() * 0.5,
                rect.get_height() + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#2b2b2b",
            )

    ax.tick_params(axis="both", direction="out", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    fig.tight_layout()
    suffix = "_clean" if clean else "_full"
    save_fig(fig, out_no_ext + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=ROOT_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = read_r2_table(args.root_dir)

    out_csv = os.path.join(args.out_dir, "Regression_R2_Bar_140802_values.csv")
    df.to_csv(out_csv, index=False)

    out_no_ext = os.path.join(args.out_dir, "Regression_R2_Bar_140802")
    draw_bar(df, out_no_ext=out_no_ext, clean=False)
    draw_bar(df, out_no_ext=out_no_ext, clean=True)

    print(f"[DONE] values: {out_csv}")
    print(f"[DONE] full: {out_no_ext}_full.png/.pdf")
    print(f"[DONE] clean: {out_no_ext}_clean.png/.pdf")


if __name__ == "__main__":
    main()
