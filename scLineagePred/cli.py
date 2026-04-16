from __future__ import annotations

import argparse
from typing import Any

from .legacy import (
    available_legacy_scripts,
    available_scripts,
    run_archived_script,
    run_embedding_training,
    run_legacy_script,
    run_trajectory_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sclineagepred")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List primary bundled scripts")
    list_parser.set_defaults(handler=handle_list_all)

    trajectory = subparsers.add_parser("trajectory", help="DeepRUOT trajectory reconstruction")
    trajectory_sub = trajectory.add_subparsers(dest="trajectory_command", required=True)

    trajectory_list = trajectory_sub.add_parser("list", help="List DeepRUOT scripts")
    trajectory_list.set_defaults(handler=handle_list_category, category="trajectory")

    trajectory_train = trajectory_sub.add_parser("train", help="Run wrapped DeepRUOT training")
    trajectory_train.add_argument("--config", required=True, help="Path to DeepRUOT YAML config")
    trajectory_train.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    trajectory_train.set_defaults(handler=handle_trajectory_train)

    trajectory_run = trajectory_sub.add_parser("run", help="Run a legacy DeepRUOT script")
    trajectory_run.add_argument("script", help="Script name, with or without .py")
    trajectory_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the legacy script")
    trajectory_run.set_defaults(handler=handle_run_legacy, category="trajectory")

    embedding = subparsers.add_parser("embedding", help="Embedding training")
    embedding_sub = embedding.add_subparsers(dest="embedding_command", required=True)

    embedding_list = embedding_sub.add_parser("list", help="List embedding scripts")
    embedding_list.set_defaults(handler=handle_list_category, category="embedding")

    embedding_train = embedding_sub.add_parser("train", help="Run wrapped embedding training")
    embedding_train.add_argument("--expr-h5")
    embedding_train.add_argument("--expr-h5ad")
    embedding_train.add_argument("--gene-names-txt")
    embedding_train.add_argument("--h5-key")
    embedding_train.add_argument("--net-tsv")
    embedding_train.add_argument("--out-dir", required=True)
    embedding_train.add_argument(
        "--do-log1p",
        dest="do_log1p",
        action="store_true",
        default=True,
        help="Apply log1p preprocessing",
    )
    embedding_train.add_argument(
        "--no-log1p",
        dest="do_log1p",
        action="store_false",
        help="Disable log1p preprocessing",
    )
    embedding_train.add_argument("--hidden", type=int, default=256)
    embedding_train.add_argument("--emb-dim", type=int, default=128)
    embedding_train.add_argument("--heads", type=int, default=4)
    embedding_train.add_argument("--tau", type=float, default=0.25)
    embedding_train.add_argument("--dropout", type=float, default=0.2)
    embedding_train.add_argument("--lr", type=float, default=1e-3)
    embedding_train.add_argument("--weight-decay", type=float, default=5e-4)
    embedding_train.add_argument("--batch-size", type=int, default=256)
    embedding_train.add_argument("--epochs-pretrain-dgi", type=int, default=200)
    embedding_train.add_argument("--early-stopping-delta", type=float, default=5e-3)
    embedding_train.add_argument("--epochs-pretrain", type=int, default=0)
    embedding_train.add_argument("--epochs-joint", type=int, default=400)
    embedding_train.add_argument("--early-stopping-patience", type=int, default=80)
    embedding_train.add_argument("--lambda-rec", type=float, default=1.0)
    embedding_train.add_argument("--lambda-dgi", type=float, default=0.3)
    embedding_train.add_argument("--verbose-every", type=int, default=10)
    embedding_train.add_argument("--max-cells-for-gene-features", type=int)
    embedding_train.add_argument("--recon-space", default="log")
    embedding_train.add_argument("--rec-loss-kind", default="mse_weighted")
    embedding_train.add_argument("--decoder-kind", default="grn")
    embedding_train.add_argument("--dec-hidden1", type=int, default=512)
    embedding_train.add_argument("--dec-hidden2", type=int, default=256)
    embedding_train.add_argument("--dec-activation", default="gelu")
    embedding_train.add_argument("--dec-dropout", type=float, default=0.1)
    embedding_train.set_defaults(handler=handle_embedding_train)

    embedding_run = embedding_sub.add_parser("run", help="Run a legacy embedding script")
    embedding_run.add_argument("script", help="Script name, with or without .py")
    embedding_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the legacy script")
    embedding_run.set_defaults(handler=handle_run_legacy, category="embedding")

    classification = subparsers.add_parser("classification", help="Unified classification pipeline")
    classification_sub = classification.add_subparsers(dest="classification_command", required=True)
    classification_train = classification_sub.add_parser("train", help="Run unified classification training")
    classification_train.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to classification/train.py")
    classification_train.set_defaults(handler=handle_run_legacy, category="classification", script="train")
    classification_list = classification_sub.add_parser("list", help="List classification scripts")
    classification_list.set_defaults(handler=handle_list_category, category="classification")
    classification_run = classification_sub.add_parser("run", help="Run a classification script")
    classification_run.add_argument("script", help="Script name, with or without .py")
    classification_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the legacy script")
    classification_run.set_defaults(handler=handle_run_legacy, category="classification")

    regression = subparsers.add_parser("regression", help="Unified regression pipeline")
    regression_sub = regression.add_subparsers(dest="regression_command", required=True)
    regression_train = regression_sub.add_parser("train", help="Run unified regression training")
    regression_train.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to regression/train.py")
    regression_train.set_defaults(handler=handle_run_legacy, category="regression", script="train")
    regression_list = regression_sub.add_parser("list", help="List regression scripts")
    regression_list.set_defaults(handler=handle_list_category, category="regression")
    regression_run = regression_sub.add_parser("run", help="Run a regression script")
    regression_run.add_argument("script", help="Script name, with or without .py")
    regression_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the legacy script")
    regression_run.set_defaults(handler=handle_run_legacy, category="regression")

    perturbation = subparsers.add_parser("perturbation", help="Unified perturbation pipeline")
    perturbation_sub = perturbation.add_subparsers(dest="perturbation_command", required=True)
    perturbation_train = perturbation_sub.add_parser("train", help="Run unified perturbation training")
    perturbation_train.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to perturbation/train.py")
    perturbation_train.set_defaults(handler=handle_run_legacy, category="perturbation", script="train")
    perturbation_list = perturbation_sub.add_parser("list", help="List perturbation scripts")
    perturbation_list.set_defaults(handler=handle_list_category, category="perturbation")
    perturbation_run = perturbation_sub.add_parser("run", help="Run a perturbation script")
    perturbation_run.add_argument("script", help="Script name, with or without .py")
    perturbation_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the legacy script")
    perturbation_run.set_defaults(handler=handle_run_legacy, category="perturbation")

    legacy = subparsers.add_parser("legacy", help="List or run archived dataset-specific scripts")
    legacy_sub = legacy.add_subparsers(dest="legacy_command", required=True)
    legacy_list = legacy_sub.add_parser("list", help="List archived scripts in a category")
    legacy_list.add_argument("category", choices=["trajectory", "embedding", "classification", "regression", "perturbation"])
    legacy_list.set_defaults(handler=handle_list_legacy)
    legacy_run = legacy_sub.add_parser("run", help="Run an archived script")
    legacy_run.add_argument("category", choices=["trajectory", "embedding", "classification", "regression", "perturbation"])
    legacy_run.add_argument("script", help="Archived script name, with or without .py")
    legacy_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the archived script")
    legacy_run.set_defaults(handler=handle_run_archived)

    return parser


def handle_list_all(_: argparse.Namespace) -> None:
    for category in ("trajectory", "embedding", "classification", "regression", "perturbation"):
        print(f"[{category}]")
        for script in available_scripts(category):
            print(script)


def handle_list_category(args: argparse.Namespace) -> None:
    for script in available_scripts(args.category):
        print(script)


def handle_run_legacy(args: argparse.Namespace) -> None:
    run_legacy_script(args.category, args.script, args.script_args)


def handle_list_legacy(args: argparse.Namespace) -> None:
    for script in available_legacy_scripts(args.category):
        print(script)


def handle_run_archived(args: argparse.Namespace) -> None:
    run_archived_script(args.category, args.script, args.script_args)


def handle_trajectory_train(args: argparse.Namespace) -> None:
    summary = run_trajectory_training(config_path=args.config, evaluate=args.evaluate)
    print(f"Trajectory training finished: {summary}")


def handle_embedding_train(args: argparse.Namespace) -> None:
    config: dict[str, Any] = {
        "expr_h5": args.expr_h5,
        "expr_h5ad": args.expr_h5ad,
        "gene_names_txt": args.gene_names_txt,
        "h5_key": args.h5_key,
        "net_tsv": args.net_tsv,
        "do_log1p": args.do_log1p,
        "hidden": args.hidden,
        "emb_dim": args.emb_dim,
        "heads": args.heads,
        "tau": args.tau,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_pretrain_dgi": args.epochs_pretrain_dgi,
        "early_stopping_delta": args.early_stopping_delta,
        "epochs_pretrain": args.epochs_pretrain,
        "epochs_joint": args.epochs_joint,
        "early_stopping_patience": args.early_stopping_patience,
        "lambda_rec": args.lambda_rec,
        "lambda_dgi": args.lambda_dgi,
        "verbose_every": args.verbose_every,
        "out_dir": args.out_dir,
        "max_cells_for_gene_features": args.max_cells_for_gene_features,
        "recon_space": args.recon_space,
        "rec_loss_kind": args.rec_loss_kind,
        "decoder_kind": args.decoder_kind,
        "dec_hidden1": args.dec_hidden1,
        "dec_hidden2": args.dec_hidden2,
        "dec_activation": args.dec_activation,
        "dec_dropout": args.dec_dropout,
    }
    run_embedding_training(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
