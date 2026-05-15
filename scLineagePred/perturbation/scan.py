from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import ood_rate_1d
from .models import predict_proba_stack


def scan_one_scenario(cfg: Config, X_all: np.ndarray, class_names, time_labels, models, stacker, device: str, scenario: dict):
    keep_len = int(scenario["keep_len"])
    if keep_len < 1 or keep_len > X_all.shape[1]:
        raise ValueError(f"Invalid keep_len={keep_len} for T={X_all.shape[1]}")

    target_t_list = [int(t) for t in scenario["perturb_t_indices"] if 0 <= int(t) < keep_len]
    if not target_t_list:
        raise ValueError(f"Scenario has empty perturb targets after filtering: {scenario}")

    X_base = np.asarray(X_all[:, :keep_len, :], dtype=np.float32)
    base_prob = predict_proba_stack(models, stacker, X_base, device=device, batch_size=cfg.batch_size)
    base_pred = np.argmax(base_prob, axis=1).astype(np.int64)
    base_mean = base_prob.mean(axis=0)
    base_frac = np.array([(base_pred == idx).mean() for idx in range(len(class_names))], dtype=np.float32)

    rows = []
    X_work = X_base.copy()
    for target_t in target_t_list:
        target_label = str(time_labels[target_t])
        for dim in range(X_all.shape[2]):
            ref = X_base[:, target_t, dim].copy()
            for fold in cfg.folds:
                new_values = ref * float(fold)
                X_work[:, target_t, dim] = new_values

                prob = predict_proba_stack(models, stacker, X_work, device=device, batch_size=cfg.batch_size)
                pred = np.argmax(prob, axis=1).astype(np.int64)
                mean_prob = prob.mean(axis=0)
                delta = mean_prob - base_mean
                frac = np.array([(pred == idx).mean() for idx in range(len(class_names))], dtype=np.float32)
                delta_frac = frac - base_frac

                push_idx = int(np.argmax(delta))
                row = {
                    "scenario_id": str(scenario["scenario_id"]),
                    "scenario_label": str(scenario["scenario_label"]),
                    "setting": str(scenario["setting"]),
                    "keep_len": keep_len,
                    "target_t": int(target_t),
                    "target_label": target_label,
                    "dim": int(dim),
                    "fold": float(fold),
                    "flip_rate": float((pred != base_pred).mean()),
                    "ood_rate": float(ood_rate_1d(ref, new_values)),
                    "delta_l1": float(np.abs(delta).sum()),
                    "delta_maxabs": float(np.abs(delta).max()),
                    "push_class": str(class_names[push_idx]),
                    "push_delta": float(delta[push_idx]),
                }
                for idx, class_name in enumerate(class_names):
                    row[f"mean_prob_{class_name}"] = float(mean_prob[idx])
                    row[f"delta_mean_prob_{class_name}"] = float(delta[idx])
                    row[f"pred_frac_{class_name}"] = float(frac[idx])
                    row[f"delta_pred_frac_{class_name}"] = float(delta_frac[idx])
                rows.append(row)

            X_work[:, target_t, dim] = ref

    df = pd.DataFrame(rows)
    summary_rows = []
    for (target_t, dim), group in df.groupby(["target_t", "dim"], sort=True):
        group_in = group[group["ood_rate"] <= cfg.max_ood_rate_for_ranking]
        if len(group_in) == 0:
            group_in = group
        best = group_in.sort_values(["flip_rate", "delta_l1", "delta_maxabs"], ascending=[False, False, False]).iloc[0]
        row = {
            "scenario_id": str(scenario["scenario_id"]),
            "scenario_label": str(scenario["scenario_label"]),
            "setting": str(scenario["setting"]),
            "keep_len": keep_len,
            "target_t": int(target_t),
            "target_label": str(best["target_label"]),
            "dim": int(dim),
            "best_fold": float(best["fold"]),
            "best_flip_rate": float(best["flip_rate"]),
            "best_ood_rate": float(best["ood_rate"]),
            "best_delta_l1": float(best["delta_l1"]),
            "best_delta_maxabs": float(best["delta_maxabs"]),
            "best_push_class": str(best["push_class"]),
            "best_push_delta": float(best["push_delta"]),
        }
        for class_name in class_names:
            row[f"best_delta_mean_prob_{class_name}"] = float(best[f"delta_mean_prob_{class_name}"])
            row[f"best_delta_pred_frac_{class_name}"] = float(best[f"delta_pred_frac_{class_name}"])
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows).sort_values(
        ["target_t", "best_flip_rate", "best_delta_l1", "best_delta_maxabs"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return df, df_sum
