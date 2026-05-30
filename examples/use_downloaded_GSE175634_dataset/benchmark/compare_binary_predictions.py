#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare binary endpoint prediction files for a GSE175634 setting.")
    parser.add_argument("--setting", required=True, help="Observation setting, for example Obs_Day5")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method prediction file in METHOD=/path/to/predictions.csv format. Repeat for multiple methods.",
    )
    parser.add_argument("--positive-label", default="CF", help="Positive endpoint label used to infer probability column names")
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def parse_method_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Method spec must be METHOD=FILE, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name.strip():
        raise ValueError(f"Missing method name in {spec!r}")
    return name.strip(), Path(path).expanduser().resolve()


def infer_probability_column(frame: pd.DataFrame, positive_label: str) -> str:
    candidates = [
        f"prob_{positive_label}",
        "prob_1",
        "prob_positive",
        "score",
        "probability",
    ]
    for col in candidates:
        if col in frame.columns:
            return col
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    if len(prob_cols) == 1:
        return prob_cols[0]
    raise KeyError(
        "Could not infer positive-class probability column. "
        f"Tried {candidates}; available columns are {list(frame.columns)}"
    )


def read_predictions(path: Path, positive_label: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if "y_true" not in frame.columns:
        raise KeyError(f"{path} must contain a y_true column")
    prob_col = infer_probability_column(frame, positive_label)
    y_true = frame["y_true"].to_numpy()
    if y_true.dtype.kind not in "biufc":
        if "true_label" in frame.columns:
            y_true = (frame["true_label"].astype(str) == positive_label).astype(int).to_numpy()
        else:
            y_true = pd.to_numeric(frame["y_true"], errors="raise").astype(int).to_numpy()
    else:
        y_true = y_true.astype(int)
    y_prob = frame[prob_col].astype(float).to_numpy()
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return y_true, y_prob


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.method:
        method, path = parse_method_spec(spec)
        y_true, y_prob = read_predictions(path, args.positive_label)
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append(
            {
                "Setting": args.setting,
                "Method": method,
                "AUC": roc_auc_score(y_true, y_prob),
                "Accuracy": accuracy_score(y_true, y_pred),
                "LogLoss": log_loss(y_true, np.column_stack([1 - y_prob, y_prob]), labels=[0, 1]),
                "N_test": int(len(y_true)),
                "Source": str(path),
            }
        )

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
