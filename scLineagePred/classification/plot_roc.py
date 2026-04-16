# -*- coding: utf-8 -*-
"""
Unified macro-ROC plotting for scLineagePred.

Dataset-specific handling:
1. Run `classification/train.py` first so each result directory contains
   `ROC_<setting>_Macro_points.csv`.
2. If a dataset needs special paths or selected settings, edit DATASET_PRESETS
   below at the top of this file.
3. You can also skip presets and pass `--result Name=/path/to/result_dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATASET_PRESETS = {
    "GSE114412": {
        "result_dir": "",
        "settings": [],
    },
    "GSE132188": {
        "result_dir": "",
        "settings": [],
    },
    "GSE140802": {
        "result_dir": "",
        "settings": [],
    },
    "GSE175634": {
        "result_dir": "",
        "settings": [],
    },
    "GSE99915": {
        "result_dir": "",
        "settings": [],
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setting_from_filename(path: Path) -> str:
    name = path.stem
    prefix = "ROC_"
    suffix = "_Macro_points"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def discover_curve_files(result_dir: Path, settings: list[str]) -> list[Path]:
    if settings:
        files = [result_dir / f"ROC_{setting}_Macro_points.csv" for setting in settings]
    else:
        files = sorted(result_dir.glob("ROC_*_Macro_points.csv"))
    return [path for path in files if path.exists()]


def load_curves(dataset_name: str, result_dir: Path, settings: list[str]) -> list[dict]:
    curve_files = discover_curve_files(result_dir, settings)
    if not curve_files:
        raise FileNotFoundError(f"No macro ROC CSV files found in {result_dir}")

    curves = []
    for path in curve_files:
        df = pd.read_csv(path)
        required = {"fpr", "tpr", "auc"}
        if not required.issubset(df.columns):
            missing = ", ".join(sorted(required - set(df.columns)))
            raise ValueError(f"{path} is missing required columns: {missing}")
        setting = str(df["setting"].iloc[0]) if "setting" in df.columns and not df.empty else setting_from_filename(path)
        curves.append(
            {
                "dataset": dataset_name,
                "setting": setting,
                "fpr": df["fpr"].astype(float).to_numpy(),
                "tpr": df["tpr"].astype(float).to_numpy(),
                "auc": float(df["auc"].iloc[0]),
            }
        )
    return curves


def parse_result_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --result value: {spec!r}. Use NAME=/path/to/result_dir")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"Invalid --result value: {spec!r}. Use NAME=/path/to/result_dir")
    return name, Path(raw_path).expanduser().resolve()


def build_sources(args: argparse.Namespace) -> list[tuple[str, Path, list[str]]]:
    sources: list[tuple[str, Path, list[str]]] = []

    for preset_name in args.preset:
        preset = DATASET_PRESETS[preset_name]
        result_dir = str(preset.get("result_dir", "")).strip()
        if not result_dir:
            raise ValueError(
                f"Preset {preset_name} has no result_dir yet. Fill DATASET_PRESETS at the top of plot_roc.py first."
            )
        settings = [str(x) for x in preset.get("settings", [])]
        sources.append((preset_name, Path(result_dir).expanduser().resolve(), settings))

    for spec in args.result:
        name, result_dir = parse_result_spec(spec)
        sources.append((name, result_dir, []))

    if not sources:
        raise ValueError("Provide at least one --preset or --result.")
    return sources


def plot_curves(curves: list[dict], out_dir: Path, out_name: str, title: str) -> None:
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    colors = [
        "#E24A33",
        "#348ABD",
        "#988ED5",
        "#55A868",
        "#DD8452",
        "#8C8C8C",
        "#C44E52",
        "#8172B3",
    ]

    rows = []
    for idx, curve in enumerate(curves):
        color = colors[idx % len(colors)]
        label = f"{curve['dataset']} | {curve['setting']} (AUC={curve['auc']:.3f})"
        ax.plot(curve["fpr"], curve["tpr"], lw=2.5, color=color, label=label)
        for fpr_v, tpr_v in zip(curve["fpr"], curve["tpr"]):
            rows.append(
                {
                    "dataset": curve["dataset"],
                    "setting": curve["setting"],
                    "auc": curve["auc"],
                    "fpr": float(fpr_v),
                    "tpr": float(tpr_v),
                }
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
    fig.savefig(out_dir / f"{out_name}.png", dpi=300)
    fig.savefig(out_dir / f"{out_name}.pdf")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / f"{out_name}_values.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot unified macro ROC curves across one or more datasets")
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        choices=sorted(DATASET_PRESETS),
        help="Dataset preset name defined at the top of this file. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Ad-hoc result directory in NAME=/path/to/result_dir format. Repeat for multiple datasets.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for the combined ROC figure")
    parser.add_argument("--out-name", default="ROC_Comparison", help="Base name of the output figure")
    parser.add_argument("--title", default="Macro ROC Comparison", help="Figure title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves: list[dict] = []
    for dataset_name, result_dir, settings in build_sources(args):
        curves.extend(load_curves(dataset_name, result_dir, settings))
    plot_curves(curves, Path(args.out_dir).expanduser().resolve(), args.out_name, args.title)


if __name__ == "__main__":
    main()
