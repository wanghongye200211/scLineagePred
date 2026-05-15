# -*- coding: utf-8 -*-
"""
Unified regression training in gene-expression space for scLineagePred.

This module keeps one main regression workflow and delegates data alignment,
task construction, and sequence models to sibling modules.
"""

from __future__ import annotations

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .config import Config, config_from_args, parse_args
    from .data import (
        build_loaders,
        build_tasks_from_timepoints,
        compute_clone_means,
        corr_cols,
        ensure_dir,
        get_expr_matrix,
        infer_reverse_from_samples_order,
        load_h5_sequences,
        maybe_flip_time_labels,
        pick_device,
        read_gene_list,
        read_time_labels_from_h5,
        set_seed,
    )
    from .models import DirectPredictor
except ImportError:
    from config import Config, config_from_args, parse_args
    from data import (
        build_loaders,
        build_tasks_from_timepoints,
        compute_clone_means,
        corr_cols,
        ensure_dir,
        get_expr_matrix,
        infer_reverse_from_samples_order,
        load_h5_sequences,
        maybe_flip_time_labels,
        pick_device,
        read_gene_list,
        read_time_labels_from_h5,
        set_seed,
    )
    from models import DirectPredictor


def train_one(model, train_loader, val_loader, device, cfg, save_path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for x, kpm, y, *_ in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = None if kpm is None else kpm.to(device)
            pred = model(x, mask)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, kpm, y, *_ in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = None if kpm is None else kpm.to(device)
                val_losses.append(loss_fn(model(x, mask), y).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"  [ep {epoch:03d}] tr={train_loss:.5f} va={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  [EarlyStop] patience reached at ep={epoch}, best_va={best_val_loss:.5f}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    metas = {"clone": [], "label": [], "tgt": []}
    for x, kpm, _, clone_id, label, tgt in loader:
        x = x.to(device)
        mask = None if kpm is None else kpm.to(device)
        preds.append(model(x, mask).detach().cpu().numpy())
        metas["clone"].append(np.array(clone_id, dtype=object))
        metas["label"].append(np.array(label, dtype=object))
        metas["tgt"].append(np.array(tgt, dtype=np.int64))
    return np.concatenate(preds, axis=0), {key: np.concatenate(value, axis=0) for key, value in metas.items()}


def fit_stacking(preds_val_list, y_val, alpha):
    model_count = len(preds_val_list)
    sample_count, gene_count = y_val.shape
    ones = np.ones((sample_count, 1), dtype=np.float32)

    weights = np.zeros((model_count, gene_count), dtype=np.float32)
    bias = np.zeros((gene_count,), dtype=np.float32)

    for gene_idx in tqdm(range(gene_count), desc="Stacking (per gene)"):
        design = np.stack([pred[:, gene_idx] for pred in preds_val_list], axis=1).astype(np.float32)
        design = np.concatenate([design, ones], axis=1)
        gram = design.T @ design + np.eye(model_count + 1, dtype=np.float32) * alpha
        gram[-1, -1] = 0.0
        solution = np.linalg.solve(gram, design.T @ y_val[:, gene_idx].astype(np.float32))
        weights[:, gene_idx] = solution[:model_count]
        bias[gene_idx] = solution[model_count]
    return weights, bias


def run(cfg: Config):
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    genes = read_gene_list(cfg.ae_result_dir)
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(genes)}

    print(f"[Data] Loading adata: {cfg.adata_h5ad}")
    adata = ad.read_h5ad(cfg.adata_h5ad)
    adata = adata[:, genes].copy()
    X_expr = get_expr_matrix(adata, cfg.adata_expr_source)
    print(
        f"[Expr] source={cfg.adata_expr_source} shape={X_expr.shape} "
        f"min={float(X_expr.min()):.4f} max={float(X_expr.max()):.4f}"
    )

    print(f"[Data] Loading sequences: {cfg.time_series_h5}")
    X_seq, indices, mask, label_str_h5 = load_h5_sequences(cfg.time_series_h5)
    if indices is None:
        raise ValueError(
            "Your sequences.h5 does NOT contain `indices`, so we cannot fetch real gene expression from adata by index.\n"
            "This unified regression pipeline requires indices for expression alignment.\n"
            "Please regenerate sequences with indices saved."
        )
    print(f"[Seq] X_seq={X_seq.shape} indices={indices.shape} mask={'None' if mask is None else mask.shape}")

    timepoints = read_time_labels_from_h5(cfg.time_series_h5, X_seq.shape[1])
    print(f"[Time] T={X_seq.shape[1]} timepoints={timepoints}")

    df_idx = pd.read_csv(cfg.index_csv)
    if len(df_idx) != len(X_seq):
        raise ValueError(f"index_csv rows ({len(df_idx)}) != sequences ({len(X_seq)}). Use matching files.")

    if infer_reverse_from_samples_order(df_idx, X_seq.shape[1]):
        X_seq = X_seq[:, ::-1, :].copy()
        indices = indices[:, ::-1].copy()
        if mask is not None:
            mask = mask[:, ::-1].copy()
        flipped_labels, changed = maybe_flip_time_labels(timepoints)
        if changed:
            timepoints = flipped_labels
            print("[Time] Detected descending time labels; flipped to ascending after sequence flip.")
        print("[Seq] Detected reverse samples_order; flipped X_seq/indices/mask to chronological order.")

    if cfg.index_clone_col not in df_idx.columns:
        raise KeyError(f"Missing clone id column in index_csv: {cfg.index_clone_col}")
    clone_ids_seq = df_idx[cfg.index_clone_col].astype(str).values

    if cfg.csv_label_col in df_idx.columns:
        label_str_seq = df_idx[cfg.csv_label_col].astype(str).values
    else:
        if label_str_h5 is None:
            raise KeyError(f"Missing {cfg.csv_label_col} in CSV and label_str in H5.")
        label_str_seq = np.array(
            [
                item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                for item in label_str_h5
            ],
            dtype=object,
        )

    if cfg.tasks is None:
        cfg.tasks = build_tasks_from_timepoints(timepoints, cfg.tasks_mode)
    print(f"[Tasks] {len(cfg.tasks)} task(s). mode={cfg.tasks_mode}")

    for task_name, input_positions, target_pos in cfg.tasks:
        if target_pos >= X_seq.shape[1]:
            raise ValueError(f"tgt_pos={target_pos} out of range for T={X_seq.shape[1]}")
        if any(position >= X_seq.shape[1] for position in input_positions):
            raise ValueError(f"in_pos has out-of-range index for T={X_seq.shape[1]}: {input_positions}")

        print(f"\n=== Task: {task_name} | in_pos={input_positions} -> tgt_pos={target_pos} ({timepoints[target_pos]}) ===")
        task_dir = os.path.join(cfg.out_dir, task_name)
        ensure_dir(task_dir)
        ensure_dir(os.path.join(task_dir, "ckpt"))

        valid_mask = indices[:, target_pos] >= 0
        if mask is not None:
            valid_mask &= mask[:, target_pos] == 1
            if cfg.require_all_inputs_present:
                valid_mask &= mask[:, input_positions].sum(axis=1) == len(input_positions)
            else:
                valid_mask &= mask[:, input_positions].sum(axis=1) > 0
        else:
            if cfg.require_all_inputs_present:
                valid_mask &= (indices[:, input_positions] >= 0).all(axis=1)
            else:
                valid_mask &= (indices[:, input_positions] >= 0).any(axis=1)

        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            print("[Skip] no valid sequences for this task (check indices/mask).")
            continue

        keep_labels = np.array(cfg.keep_labels, dtype=object)
        if keep_labels.size > 0:
            keep = np.isin(label_str_seq[valid_idx], keep_labels)
        else:
            keep = np.ones(valid_idx.shape[0], dtype=bool)
        valid_idx = valid_idx[keep]
        if valid_idx.size == 0:
            counts = pd.Series(label_str_seq[np.where(valid_mask)[0]]).value_counts().to_dict()
            print(f"[Skip] 0 sequences after label filter. keep={cfg.keep_labels} | label counts={counts}")
            continue

        X_in = X_seq[valid_idx][:, input_positions, :]
        tgt_cell_idx = indices[valid_idx, target_pos].astype(np.int64)
        labels_tgt = label_str_seq[valid_idx]
        clone_ids = clone_ids_seq[valid_idx]
        mask_in = (
            mask[valid_idx][:, input_positions].astype(np.int8)
            if mask is not None
            else (indices[valid_idx][:, input_positions] >= 0).astype(np.int8)
        )

        rng = np.random.default_rng(cfg.seed)
        split_indices = np.arange(valid_idx.size)
        rng.shuffle(split_indices)
        train_n = int(len(split_indices) * cfg.split_train)
        val_n = int(len(split_indices) * cfg.split_val)
        idx_tr = split_indices[:train_n]
        idx_va = split_indices[train_n : train_n + val_n]
        idx_te = split_indices[train_n + val_n :]
        print(f"[Split-Random] train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} (n={len(split_indices)})")

        train_loader, val_loader, test_loader, mu, sd = build_loaders(
            X_in,
            mask_in,
            tgt_cell_idx,
            labels_tgt,
            clone_ids,
            X_expr,
            idx_tr,
            idx_va,
            idx_te,
            cfg.batch_size,
        )
        np.save(os.path.join(task_dir, "norm_mu.npy"), mu)
        np.save(os.path.join(task_dir, "norm_sd.npy"), sd)

        base_names = ["RNN", "BiLSTM", "Trans"]
        preds_val_list = []
        models = {}

        for name in base_names:
            print(f"[Train] {name}")
            model = DirectPredictor(name, in_dim=X_in.shape[-1], out_dim=X_expr.shape[1], hidden=cfg.hidden, dropout=cfg.dropout)
            ckpt_path = os.path.join(task_dir, "ckpt", f"{name}.pt")
            model = train_one(model, train_loader, val_loader, device, cfg, ckpt_path)
            models[name] = model
            preds_val, _ = predict(model, val_loader, device)
            preds_val_list.append(preds_val.astype(np.float32))

        y_val = X_expr[tgt_cell_idx[idx_va]]
        print("[Stacking] fitting per-gene ridge stacking...")
        W, b = fit_stacking(preds_val_list, y_val, alpha=cfg.stack_alpha)

        report = {"task": task_name, "sign_fix_enable": cfg.sign_fix_enable}
        if cfg.sign_fix_enable:
            p_stack_val = np.zeros_like(preds_val_list[0], dtype=np.float32)
            for idx, pred in enumerate(preds_val_list):
                p_stack_val += pred * W[idx][None, :]
            p_stack_val += b[None, :]

            clone_val = clone_ids[idx_va]
            pred_clone_means, _ = compute_clone_means(p_stack_val, clone_val)
            true_clone_means, _ = compute_clone_means(y_val, clone_val)
            r_stack = corr_cols(pred_clone_means, true_clone_means)

            base_corr = []
            for pred in preds_val_list:
                pred_clone, _ = compute_clone_means(pred, clone_val)
                base_corr.append(corr_cols(pred_clone, true_clone_means))
            base_corr = np.stack(base_corr, axis=0)

            fixed = []
            for gene_idx in np.where(r_stack < cfg.sign_fix_r_threshold)[0]:
                candidates = np.where(base_corr[:, gene_idx] > cfg.sign_fix_min_pos_r)[0]
                if candidates.size == 0:
                    continue
                best_model_idx = candidates[np.argmax(base_corr[candidates, gene_idx])]
                W[:, gene_idx] = 0.0
                W[best_model_idx, gene_idx] = 1.0
                b[gene_idx] = 0.0
                fixed.append((int(gene_idx), int(best_model_idx), float(r_stack[gene_idx]), float(base_corr[best_model_idx, gene_idx])))

            report.update(
                {
                    "sign_fix_threshold": cfg.sign_fix_r_threshold,
                    "sign_fix_min_pos_r": cfg.sign_fix_min_pos_r,
                    "fixed_genes_n": len(fixed),
                }
            )

            if fixed:
                fixed = sorted(fixed, key=lambda item: item[2])
                print(f"[SignFix] fixed {len(fixed)} genes on VAL. Top cases:")
                for gene_idx, model_idx, r_stack_value, r_model_value in fixed[: cfg.sign_fix_report_topn]:
                    print(f"  {genes[gene_idx]}  r_stack={r_stack_value:.3f} -> base[{base_names[model_idx]}] r={r_model_value:.3f}")

            for gene_name in cfg.watch_genes:
                if gene_name in gene_to_idx:
                    gene_idx = gene_to_idx[gene_name]
                    print(
                        f"[Watch][VAL] {gene_name}: r_stack={float(r_stack[gene_idx]):.3f}, "
                        f"r_base={[(base_names[i], float(base_corr[i, gene_idx])) for i in range(len(base_names))]} "
                        f"W={W[:, gene_idx].round(3).tolist()} b={float(b[gene_idx]):.3f}"
                    )

        np.save(os.path.join(task_dir, "stacking_W.npy"), W)
        np.save(os.path.join(task_dir, "stacking_b.npy"), b)
        with open(os.path.join(task_dir, "signfix_report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        print("[Test] predicting stacked output...")
        test_preds = []
        meta = None
        for name in base_names:
            preds, pred_meta = predict(models[name], test_loader, device)
            test_preds.append(preds.astype(np.float32))
            meta = pred_meta if meta is None else meta

        final_pred = np.zeros_like(test_preds[0], dtype=np.float32)
        for idx, pred in enumerate(test_preds):
            final_pred += pred * W[idx][None, :]
        final_pred += b[None, :]
        y_true = X_expr[meta["tgt"]]

        out_npz = os.path.join(task_dir, "test_outputs.npz")
        np.savez_compressed(
            out_npz,
            pred_log=final_pred.astype(np.float32),
            true_log=y_true.astype(np.float32),
            tgt_cell_idx=meta["tgt"].astype(np.int64),
            clone_id=meta["clone"].astype(object),
            label=meta["label"].astype(object),
            gene_names=genes.astype(object),
            task=np.array([task_name], dtype=object),
            timepoints=np.array(timepoints, dtype=object),
            in_pos=np.array(input_positions, dtype=np.int64),
            tgt_pos=np.array([target_pos], dtype=np.int64),
        )
        print(f"[Saved] {out_npz}")

    print("\nDone.")


def main():
    run(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
