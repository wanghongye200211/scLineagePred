# -*- coding: utf-8 -*-
"""
Unified latent-space perturbation scan for scLineagePred.

The public workflow stays in one place, while data loading, ensemble handling,
scan logic, and cell-state transition marker analysis are split into small
support modules.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from .config import Config, config_from_args, parse_args
    from .data import build_strategies, ensure_dir, load_sequences, pick_device, sanitize_name
    from .markers import (
        build_local_marker_rank,
        build_marker_master_rrf,
        integrate_marker_genes_for_class,
        load_decoder,
        load_hvgs,
        pick_dims_for_class,
        top_genes_by_dim,
        transition_genes,
    )
    from .models import load_ensemble
    from .scan import scan_one_scenario
except ImportError:
    from config import Config, config_from_args, parse_args
    from data import build_strategies, ensure_dir, load_sequences, pick_device, sanitize_name
    from markers import (
        build_local_marker_rank,
        build_marker_master_rrf,
        integrate_marker_genes_for_class,
        load_decoder,
        load_hvgs,
        pick_dims_for_class,
        top_genes_by_dim,
        transition_genes,
    )
    from models import load_ensemble
    from scan import scan_one_scenario


def run(cfg: Config):
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "perturbation"))
    ensure_dir(os.path.join(cfg.out_dir, "downstream"))

    device = pick_device(cfg)
    print(f"[INFO] Device: {device}")

    X, y, class_names, time_labels = load_sequences(cfg)
    _, time_count, latent_dim = X.shape

    strategies = build_strategies(cfg, time_labels)
    print("[INFO] Active endpoint strategies:")
    for strategy in strategies:
        print(
            f"  - {strategy['scenario_id']}: setting={strategy['setting']} "
            f"keep_len={strategy['keep_len']} targets={strategy['perturb_t_indices']}"
        )

    ensemble_cache: Dict[str, Tuple[dict, object]] = {}
    all_summaries = []
    decoder_tags = []

    for strategy in strategies:
        setting = str(strategy["setting"])
        if setting not in ensemble_cache:
            ensemble_cache[setting] = load_ensemble(
                cfg=cfg,
                input_dim=latent_dim,
                n_classes=len(class_names),
                device=device,
                setting=setting,
            )
        models, stacker = ensemble_cache[setting]

        scenario_tag = sanitize_name(strategy["scenario_label"])
        scenario_dir = os.path.join(cfg.out_dir, "perturbation", scenario_tag)
        ensure_dir(scenario_dir)

        print(
            f"\n[Perturb] scenario={strategy['scenario_id']} setting={setting} "
            f"keep_len={strategy['keep_len']} target_t={strategy['perturb_t_indices']}"
        )
        df, df_sum = scan_one_scenario(
            cfg=cfg,
            X_all=X,
            class_names=class_names,
            time_labels=time_labels,
            models=models,
            stacker=stacker,
            device=device,
            scenario=strategy,
        )

        df.to_csv(os.path.join(scenario_dir, "dose_response_all_dims.csv"), index=False)
        df_sum.to_csv(os.path.join(scenario_dir, "dim_summary.csv"), index=False)

        for target_t in sorted(df_sum["target_t"].unique().tolist()):
            target_t = int(target_t)
            target_tag = f"t{target_t}_{sanitize_name(time_labels[target_t])}"
            target_dir = os.path.join(scenario_dir, target_tag)
            ensure_dir(target_dir)

            df_t = df[df["target_t"] == target_t].copy()
            df_sum_t = df_sum[df_sum["target_t"] == target_t].copy()
            df_t.to_csv(os.path.join(target_dir, "dose_response_all_dims.csv"), index=False)
            df_sum_t.to_csv(os.path.join(target_dir, "dim_summary.csv"), index=False)

            top_dims = df_sum_t["dim"].head(int(cfg.top_k_dims)).astype(int).tolist()
            with open(os.path.join(target_dir, "top_dims.txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(map(str, top_dims)))

            decoder_tag = f"{scenario_tag}__{target_tag}"
            decoder_tags.append(decoder_tag)
            all_summaries.append(df_sum_t.assign(scenario_tag=scenario_tag, target_tag=target_tag, decoder_tag=decoder_tag))

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            os.path.join(cfg.out_dir, "perturbation", "dim_summary_all_targets.csv"),
            index=False,
        )

    hvgs = load_hvgs(cfg)
    genes, Zg = load_decoder(cfg, latent_dim=latent_dim, hvgs=hvgs)
    downstream_dir = os.path.join(cfg.out_dir, "downstream")
    ensure_dir(downstream_dir)

    union_genes = set()
    for idx in range(time_count - 1):
        left_label = sanitize_name(time_labels[idx])
        right_label = sanitize_name(time_labels[idx + 1])

        delta_mean = (X[:, idx + 1, :] - X[:, idx, :]).mean(axis=0)
        df_transition = transition_genes(genes, Zg, delta_mean, topk=max(200, cfg.top_union_transition))
        df_transition.to_csv(
            os.path.join(downstream_dir, f"genes_transition_{left_label}to{right_label}_mean.csv"),
            index=False,
            float_format="%.10f",
        )
        union_genes.update(df_transition["gene"].astype(str).head(int(cfg.top_union_transition)).tolist())

        for class_idx, class_name in enumerate(class_names):
            class_mask = y == class_idx
            other_mask = ~class_mask
            if class_mask.sum() == 0 or other_mask.sum() == 0:
                continue
            delta_class = (X[class_mask, idx + 1, :] - X[class_mask, idx, :]).mean(axis=0)
            delta_rest = (X[other_mask, idx + 1, :] - X[other_mask, idx, :]).mean(axis=0)
            df_transition = transition_genes(genes, Zg, delta_class - delta_rest, topk=max(200, cfg.top_union_transition))
            df_transition.to_csv(
                os.path.join(downstream_dir, f"genes_transition_{left_label}to{right_label}_{class_name}_vs_rest.csv"),
                index=False,
                float_format="%.10f",
            )
            union_genes.update(df_transition["gene"].astype(str).head(int(cfg.top_union_transition)).tolist())

    for strategy in strategies:
        scenario_tag = sanitize_name(strategy["scenario_label"])
        scenario_dir = os.path.join(cfg.out_dir, "perturbation", scenario_tag)

        for target_t in strategy["perturb_t_indices"]:
            target_t = int(target_t)
            target_tag = f"t{target_t}_{sanitize_name(time_labels[target_t])}"
            target_dir = os.path.join(scenario_dir, target_tag)
            dim_summary_path = os.path.join(target_dir, "dim_summary.csv")
            if not os.path.exists(dim_summary_path):
                continue

            dim_summary = pd.read_csv(dim_summary_path)
            top_dims = dim_summary["dim"].head(int(cfg.top_k_dims)).astype(int).tolist()
            dim_all_set = set(int(value) for value in dim_summary["dim"].astype(int).tolist())

            decoder_tag = f"{scenario_tag}__{target_tag}"
            decoder_dir = os.path.join(downstream_dir, f"decoder_{decoder_tag}")
            ensure_dir(decoder_dir)

            per_dim_tables = [top_genes_by_dim(genes, Zg, int(dim), int(cfg.top_k_genes_per_dim)) for dim in top_dims]
            per_dim_df = pd.concat(per_dim_tables, ignore_index=True) if per_dim_tables else pd.DataFrame()
            if not per_dim_df.empty:
                per_dim_df.to_csv(os.path.join(decoder_dir, "top_genes_per_dim.csv"), index=False)
                union_genes.update(per_dim_df["gene"].astype(str).tolist())

            for class_name in class_names:
                dims_for_class = pick_dims_for_class(
                    dim_sum=dim_summary,
                    top_dims=top_dims,
                    class_name=class_name,
                    mode=cfg.marker_dim_mode,
                )
                if not dims_for_class and cfg.marker_fallback_pos_only and str(cfg.marker_dim_mode).lower() == "push_class":
                    dims_for_class = pick_dims_for_class(
                        dim_sum=dim_summary,
                        top_dims=top_dims,
                        class_name=class_name,
                        mode="pos_only",
                    )
                if not dims_for_class:
                    dims_for_class = [int(dim) for dim in top_dims if int(dim) in dim_all_set]

                marker_rank = integrate_marker_genes_for_class(genes, Zg, dim_summary, dims_for_class, class_name)
                marker_rank.to_csv(os.path.join(decoder_dir, f"marker_genes_ranked_{class_name}.csv"), index=False)

                local_marker_rank = build_local_marker_rank(
                    df_rank=marker_rank,
                    class_name=class_name,
                    rrf_k=int(cfg.rrf_k),
                    top_n=int(cfg.top_union_marker),
                )
                local_marker_rank.to_csv(
                    os.path.join(decoder_dir, f"cell_state_transition_markers_ranked_{class_name}.csv"),
                    index=False,
                )
                union_genes.update(marker_rank["gene"].astype(str).head(int(cfg.top_union_marker)).tolist())

    pd.DataFrame({"gene": sorted(union_genes)}).to_csv(
        os.path.join(downstream_dir, "marker_gene_candidates_union.csv"),
        index=False,
    )

    if cfg.save_marker_master:
        for class_name in class_names:
            marker_master = build_marker_master_rrf(cfg, downstream_dir=downstream_dir, decoder_tags=decoder_tags, class_name=class_name)
            marker_master.to_csv(
                os.path.join(downstream_dir, f"cell_state_transition_markers_master_{class_name}.csv"),
                index=False,
            )
            marker_master[["gene", "marker_score", "direction", "sources", "rank"]].to_csv(
                os.path.join(downstream_dir, f"cell_state_transition_markers_{class_name}.csv"),
                index=False,
            )
            marker_master.to_csv(
                os.path.join(cfg.out_dir, f"cell_state_transition_markers_master_{class_name}.csv"),
                index=False,
            )
            marker_master[["gene", "marker_score"]].to_csv(
                os.path.join(cfg.out_dir, f"cell_state_transition_markers_{class_name}.csv"),
                index=False,
            )

        union_markers = set()
        for class_name in class_names:
            path = os.path.join(downstream_dir, f"cell_state_transition_markers_master_{class_name}.csv")
            if os.path.exists(path):
                union_markers.update(pd.read_csv(path)["gene"].astype(str).head(300).tolist())
        pd.DataFrame({"gene": sorted(union_markers)}).to_csv(
            os.path.join(downstream_dir, "cell_state_transition_markers_union_top300_eachclass.csv"),
            index=False,
        )

    run_meta = cfg.__dict__.copy()
    for key, value in list(run_meta.items()):
        if isinstance(value, tuple):
            run_meta[key] = list(value)
    run_meta["time_labels"] = list(time_labels)
    run_meta["active_strategies"] = strategies

    with open(os.path.join(cfg.out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2)

    print(f"\n[Done] Outputs at: {cfg.out_dir}")


def main():
    run(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
