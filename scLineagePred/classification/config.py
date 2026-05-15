from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    version: str = "unified-v1"

    time_series_h5: str = ""
    index_csv: str = ""
    out_dir: str = "./outputs/classification"
    model_dir: str = ""

    label_col: str = "label_str"
    clone_col: str = "clone_id"
    target_labels: Tuple[str, ...] = ()

    base_seed: int = 2026
    batch_size: int = 512
    epochs: int = 600
    patience: int = 30
    min_delta: float = 1e-4
    lr: float = 1e-3

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    nhead: int = 4

    grad_clip_norm: float = 1.0
    use_scheduler: bool = True
    lr_patience: int = 6
    lr_factor: float = 0.5
    min_lr: float = 1e-5

    label_smoothing: float = 0.0
    num_workers: int = 0

    stack_max_iter: int = 4000
    stack_C: float = 0.8

    test_frac: float = 0.10
    val_frac: float = 0.10

    device_prefer: str = "auto"

    max_points_3d: int = 8000
    alpha_3d: float = 0.75
    size_3d: float = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified sequence classification training")
    parser.add_argument("--time-series-h5", required=True, help="Sequence H5 file with X and optional labels")
    parser.add_argument("--index-csv", default="", help="Optional sequence metadata CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--model-dir", default="", help="Model checkpoint directory (default: <out-dir>/saved_models)")
    parser.add_argument("--label-col", default="label_str", help="Label column name in index CSV")
    parser.add_argument("--clone-col", default="clone_id", help="Clone column name in index CSV")
    parser.add_argument(
        "--target-label",
        dest="target_labels",
        action="append",
        default=[],
        help="Endpoint label to keep. Repeat the flag to provide multiple labels.",
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--use-scheduler", action="store_true", default=True)
    parser.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stack-max-iter", type=int, default=4000)
    parser.add_argument("--stack-c", type=float, default=0.8)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--device", dest="device_prefer", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-points-3d", type=int, default=8000)
    parser.add_argument("--alpha-3d", type=float, default=0.75)
    parser.add_argument("--size-3d", type=float, default=24.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        time_series_h5=args.time_series_h5,
        index_csv=args.index_csv,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        label_col=args.label_col,
        clone_col=args.clone_col,
        target_labels=tuple(args.target_labels),
        base_seed=args.base_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nhead=args.nhead,
        grad_clip_norm=args.grad_clip_norm,
        use_scheduler=args.use_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        stack_max_iter=args.stack_max_iter,
        stack_C=args.stack_c,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        device_prefer=args.device_prefer,
        max_points_3d=args.max_points_3d,
        alpha_3d=args.alpha_3d,
        size_3d=args.size_3d,
    )
