from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    time_series_h5: str = ""
    index_csv: str = ""
    target_labels: Tuple[str, ...] = ()
    csv_label_col: str = "label_str"

    model_dir: str = ""
    base_seed: int = 2026
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    nhead: int = 4

    folds: Tuple[float, ...] = (
        0.0,
        0.05,
        0.1,
        0.2,
        0.5,
        0.8,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
    )
    max_ood_rate_for_ranking: float = 0.10
    top_k_dims: int = 30
    top_k_genes_per_dim: int = 50
    max_sequences_scan: Optional[int] = 200000

    decoder_dir: str = ""
    genes_txt: str = ""
    z_genes_npy: str = ""
    hvg_h5ad: str = ""

    marker_dim_mode: str = "push_class"
    marker_fallback_pos_only: bool = True
    normalize_gene_vectors: bool = True

    out_dir: str = "./outputs/perturbation"
    scenario_mode: str = "last_two"

    rrf_k: int = 50
    top_union_marker: int = 800
    top_union_transition: int = 300
    save_marker_master: bool = True

    device: str = "auto"
    batch_size: int = 2048

    def __post_init__(self):
        self.time_series_h5 = os.path.expanduser(self.time_series_h5)
        self.index_csv = os.path.expanduser(self.index_csv)
        self.model_dir = os.path.expanduser(self.model_dir)
        self.decoder_dir = os.path.expanduser(self.decoder_dir)
        self.hvg_h5ad = os.path.expanduser(self.hvg_h5ad)
        self.out_dir = os.path.expanduser(self.out_dir)
        if self.decoder_dir:
            if not self.genes_txt:
                self.genes_txt = os.path.join(self.decoder_dir, "genes.txt")
            if not self.z_genes_npy:
                self.z_genes_npy = os.path.join(self.decoder_dir, "Z_genes.npy")
        self.genes_txt = os.path.expanduser(self.genes_txt)
        self.z_genes_npy = os.path.expanduser(self.z_genes_npy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified perturbation scan for scLineagePred")
    parser.add_argument("--time-series-h5", required=True, help="Sequence H5 produced by trajectory reconstruction")
    parser.add_argument("--index-csv", required=True, help="Index CSV paired with the sequence H5")
    parser.add_argument("--model-dir", required=True, help="Directory containing classification ensemble weights")
    parser.add_argument("--decoder-dir", required=True, help="Directory containing decoder outputs such as genes.txt and Z_genes.npy")
    parser.add_argument("--hvg-h5ad", required=True, help="H5AD file used for HVG overlap validation")
    parser.add_argument("--out-dir", required=True, help="Output directory for perturbation results")
    parser.add_argument("--target-label", action="append", dest="target_labels", default=None, help="Endpoint label to keep; repeat for multiple labels")
    parser.add_argument("--csv-label-col", default="label_str", help="Label column in index CSV")
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--folds", nargs="+", type=float, help="Perturbation scale factors")
    parser.add_argument("--max-ood-rate-for-ranking", type=float, default=0.10)
    parser.add_argument("--top-k-dims", type=int, default=30)
    parser.add_argument("--top-k-genes-per-dim", type=int, default=50)
    parser.add_argument("--max-sequences-scan", type=int, default=200000)
    parser.add_argument("--marker-dim-mode", choices=["push_class", "pos_only", "all"], default="push_class")
    parser.add_argument("--scenario-mode", choices=["last_two"], default="last_two")
    parser.add_argument("--rrf-k", type=int, default=50)
    parser.add_argument("--top-union-marker", type=int, default=800)
    parser.add_argument("--top-union-transition", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--genes-txt", help="Optional explicit genes.txt path")
    parser.add_argument("--z-genes-npy", help="Optional explicit Z_genes.npy path")
    parser.add_argument("--no-marker-fallback-pos-only", action="store_true", help="Disable pos_only fallback when push_class yields no dims")
    parser.add_argument("--no-normalize-gene-vectors", action="store_true", help="Disable decoder gene vector normalization")
    parser.add_argument(
        "--no-save-marker-master",
        action="store_true",
        help="Skip aggregated cell-state transition marker exports",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    kwargs = {
        "time_series_h5": args.time_series_h5,
        "index_csv": args.index_csv,
        "csv_label_col": args.csv_label_col,
        "model_dir": args.model_dir,
        "base_seed": args.base_seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "nhead": args.nhead,
        "max_ood_rate_for_ranking": args.max_ood_rate_for_ranking,
        "top_k_dims": args.top_k_dims,
        "top_k_genes_per_dim": args.top_k_genes_per_dim,
        "max_sequences_scan": args.max_sequences_scan,
        "decoder_dir": args.decoder_dir,
        "genes_txt": args.genes_txt or "",
        "z_genes_npy": args.z_genes_npy or "",
        "hvg_h5ad": args.hvg_h5ad,
        "marker_dim_mode": args.marker_dim_mode,
        "marker_fallback_pos_only": not args.no_marker_fallback_pos_only,
        "normalize_gene_vectors": not args.no_normalize_gene_vectors,
        "out_dir": args.out_dir,
        "scenario_mode": args.scenario_mode,
        "rrf_k": args.rrf_k,
        "top_union_marker": args.top_union_marker,
        "top_union_transition": args.top_union_transition,
        "save_marker_master": not args.no_save_marker_master,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if args.target_labels:
        kwargs["target_labels"] = tuple(args.target_labels)
    if args.folds:
        kwargs["folds"] = tuple(args.folds)
    return Config(**kwargs)
