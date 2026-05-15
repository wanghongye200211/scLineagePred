# -*- coding: utf-8 -*-
"""
Unified sequence classification training for scLineagePred.

This entry point keeps one dataset-agnostic workflow. Dataset-specific label
selection is handled through repeated `--target-label` flags, while shared
helpers live in sibling modules to keep the training script readable.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader

try:
    from .config import Config, config_from_args, parse_args
    from .data import (
        SeqDataset,
        build_time_settings,
        collate_pad,
        ensure_dir,
        load_data,
        pick_device,
        set_all_seeds,
        stratified_split,
    )
    from .models import BiLSTMModel, RNNModel, TransformerModel
    from .plots import (
        plot_3d_pred_points_pca_true_coded,
        plot_macro_roc_all_settings_style,
        plot_performance_trend,
        plot_setting_roc_ovr_macro_style,
    )
except ImportError:
    from config import Config, config_from_args, parse_args
    from data import (
        SeqDataset,
        build_time_settings,
        collate_pad,
        ensure_dir,
        load_data,
        pick_device,
        set_all_seeds,
        stratified_split,
    )
    from models import BiLSTMModel, RNNModel, TransformerModel
    from plots import (
        plot_3d_pred_points_pca_true_coded,
        plot_macro_roc_all_settings_style,
        plot_performance_trend,
        plot_setting_roc_ovr_macro_style,
    )


def train_base_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    name: str,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if cfg.label_smoothing > 0:
        try:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        except TypeError:
            print("[WARN] Current torch version does not support label_smoothing; fallback to 0.")
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.min_lr,
            verbose=False,
        )

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    print(f"\n--- Training {name} ---")
    print(f"{'Epoch':<5} | {'TrLoss':<10} | {'VaLoss':<10} | {'VaAcc':<8} | {'LR':<10} | {'Pat':<4}")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, y, lengths in train_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()
            train_loss_sum += float(loss.item()) * x.size(0)
            train_n += x.size(0)

        train_loss = train_loss_sum / max(1, train_n)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        correct = 0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)

                logits = model(x, lengths)
                loss = loss_fn(logits, y)
                val_loss_sum += float(loss.item()) * x.size(0)
                val_n += x.size(0)
                correct += int((logits.argmax(dim=1) == y).sum().item())

        val_loss = val_loss_sum / max(1, val_n)
        val_acc = correct / max(1, val_n)

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < (best_val_loss - cfg.min_delta)
        if improved:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"{epoch + 1:03d}   | {train_loss:<10.4f} | {val_loss:<10.4f} | "
            f"{val_acc:<8.4f} | {current_lr:<10.2e} | {bad_epochs:<4d}"
        )

        if bad_epochs >= cfg.patience:
            print(f"[EarlyStop] {name} stopped at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def get_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs = []
    targets = []
    for x, y, lengths in loader:
        logits = model(x.to(device), lengths.to(device))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(targets, axis=0)


def build_models(input_dim: int, cfg: Config, num_classes: int) -> Dict[str, nn.Module]:
    return {
        "BiLSTM": BiLSTMModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, num_classes),
        "RNN": RNNModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, num_classes),
        "Trans": TransformerModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.nhead, num_classes),
    }


def run(cfg: Config):
    if not cfg.model_dir:
        cfg.model_dir = os.path.join(cfg.out_dir, "saved_models")
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model_dir)

    X, y, clones, class_names, time_labels = load_data(cfg)
    device = pick_device(cfg)
    print(f"[INFO] Device: {device}")

    settings, setting_order, x_labels = build_time_settings(time_labels)
    print(
        f"[INFO] Version={cfg.version} | random-split | "
        f"hidden={cfg.hidden_dim} dropout={cfg.dropout} ls={cfg.label_smoothing}"
    )

    results_buffer: Dict[str, dict] = {}
    macro_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    print("\n" + "=" * 60)
    print("PHASE 1: Train base models + stacking for each timepoint setting")
    print("=" * 60)

    for setting in setting_order:
        keep_len = int(settings[setting])
        seed = cfg.base_seed
        set_all_seeds(seed)

        print(f"\n>>> Setting: {setting} | seed={seed} | keep_len={keep_len}")
        train_idx, val_idx, test_idx = stratified_split(y, seed, cfg.test_frac, cfg.val_frac)

        train_loader = DataLoader(
            SeqDataset(X, y, train_idx, keep_len),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_pad,
        )
        val_loader = DataLoader(
            SeqDataset(X, y, val_idx, keep_len),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_pad,
        )
        test_loader = DataLoader(
            SeqDataset(X, y, test_idx, keep_len),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_pad,
        )

        models = build_models(X.shape[2], cfg, len(class_names))
        val_features = []
        test_features = []

        for name, model in models.items():
            trained = train_base_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                cfg=cfg,
                name=name,
            )

            p_val, _ = get_probs(trained, val_loader, device)
            p_test, _ = get_probs(trained, test_loader, device)
            val_features.append(p_val)
            test_features.append(p_test)

            torch.save(trained.state_dict(), os.path.join(cfg.model_dir, f"{setting}_{name}_s{seed}.pth"))
            models[name] = trained

        X_val = np.concatenate(val_features, axis=1)
        X_test = np.concatenate(test_features, axis=1)
        y_val = y[val_idx]
        y_test = y[test_idx]

        print("\n--- Training Stacking (LogReg) ---")
        stacker = LogisticRegression(
            random_state=seed,
            max_iter=cfg.stack_max_iter,
            C=cfg.stack_C,
            multi_class="auto",
            solver="lbfgs",
        )
        stacker.fit(X_val, y_val)
        p_stack = stacker.predict_proba(X_test)

        y_pred = np.argmax(p_stack, axis=1)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, p_stack, labels=list(range(len(class_names))))
        print(f"   [Result] {setting} | Acc={acc:.4f} | LogLoss={loss:.4f}")

        with open(os.path.join(cfg.model_dir, f"{setting}_Stacking_s{seed}.pkl"), "wb") as handle:
            pickle.dump(stacker, handle)

        results_buffer[setting] = {
            "seed": seed,
            "y_true": y_test,
            "y_prob": p_stack,
            "acc": acc,
            "loss": loss,
            "clones": clones[test_idx],
        }

    print("\n" + "=" * 60)
    print("PHASE 2: ROC + 3D + Summary")
    print("=" * 60)

    for setting in setting_order:
        result = results_buffer[setting]
        macro_curves[setting] = plot_setting_roc_ovr_macro_style(
            result["y_true"],
            result["y_prob"],
            class_names,
            setting,
            cfg.out_dir,
        )

    plot_macro_roc_all_settings_style(macro_curves, setting_order, cfg.out_dir)
    plot_performance_trend(results_buffer, setting_order, x_labels, cfg.out_dir)

    for setting in setting_order:
        result = results_buffer[setting]
        plot_3d_pred_points_pca_true_coded(
            y_true=result["y_true"],
            y_prob=result["y_prob"],
            class_names=class_names,
            title=f"{setting} | 3D Prob PCA (color+marker = TRUE label)",
            out_prefix=os.path.join(cfg.out_dir, f"Pred3D_{setting}_truecoded"),
            max_points=cfg.max_points_3d,
            seed=cfg.base_seed,
            alpha=cfg.alpha_3d,
            size=cfg.size_3d,
        )

    summary_rows = []
    for setting in setting_order:
        result = results_buffer[setting]
        summary_rows.append(
            {
                "Setting": setting,
                "Seed": result["seed"],
                "Accuracy": result["acc"],
                "LogLoss": result["loss"],
                "N_test": int(len(result["y_true"])),
            }
        )
    pd.DataFrame(summary_rows).to_csv(os.path.join(cfg.out_dir, "ensemble_summary.csv"), index=False)
    print(f"   [CSV] Saved: {os.path.join(cfg.out_dir, 'ensemble_summary.csv')}")
    print(f"\n[DONE] All outputs saved to: {cfg.out_dir}")


def main():
    run(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
