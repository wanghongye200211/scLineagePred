from __future__ import annotations

import argparse


class Config:
    ae_result_dir = ""
    time_series_h5 = ""
    index_csv = ""
    index_clone_col = "clone_id"
    csv_label_col = "label_str"
    adata_h5ad = ""
    out_dir = "./outputs/regression"

    adata_expr_source = "X"
    keep_labels = tuple()
    tasks_mode = "all_prev_only"
    tasks = None
    require_all_inputs_present = False

    seed = 42
    split_train = 0.80
    split_val = 0.10
    split_test = 0.10

    device = "auto"
    batch_size = 256
    lr = 1e-3
    epochs = 80
    patience = 12
    hidden = 512
    dropout = 0.2

    stack_alpha = 5.0

    sign_fix_enable = False
    sign_fix_r_threshold = -0.01
    sign_fix_min_pos_r = 0.05
    sign_fix_report_topn = 25
    watch_genes = tuple()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified gene-space regression training")
    parser.add_argument("--ae-result-dir", required=True, help="Directory containing genes.txt from embedding training")
    parser.add_argument("--time-series-h5", required=True, help="Sequence H5 file")
    parser.add_argument("--index-csv", required=True, help="Sequence metadata CSV")
    parser.add_argument("--adata-h5ad", required=True, help="Integrated AnnData with matched cells")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--index-clone-col", default="clone_id")
    parser.add_argument("--csv-label-col", default="label_str")
    parser.add_argument("--adata-expr-source", default="X")
    parser.add_argument(
        "--keep-label",
        dest="keep_labels",
        action="append",
        default=[],
        help="Endpoint label to keep. Repeat the flag to provide multiple labels.",
    )
    parser.add_argument("--tasks-mode", default="all_prev_only", choices=["all_prev_only", "each_prefix"])
    parser.add_argument("--require-all-inputs-present", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-train", type=float, default=0.80)
    parser.add_argument("--split-val", type=float, default=0.10)
    parser.add_argument("--split-test", type=float, default=0.10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stack-alpha", type=float, default=5.0)
    parser.add_argument("--sign-fix-enable", action="store_true")
    parser.add_argument("--sign-fix-r-threshold", type=float, default=-0.01)
    parser.add_argument("--sign-fix-min-pos-r", type=float, default=0.05)
    parser.add_argument("--sign-fix-report-topn", type=int, default=25)
    parser.add_argument(
        "--watch-gene",
        dest="watch_genes",
        action="append",
        default=[],
        help="Gene to print in sign-fix diagnostics. Repeat the flag to provide multiple genes.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.ae_result_dir = args.ae_result_dir
    cfg.time_series_h5 = args.time_series_h5
    cfg.index_csv = args.index_csv
    cfg.index_clone_col = args.index_clone_col
    cfg.csv_label_col = args.csv_label_col
    cfg.adata_h5ad = args.adata_h5ad
    cfg.out_dir = args.out_dir
    cfg.adata_expr_source = args.adata_expr_source
    cfg.keep_labels = tuple(args.keep_labels)
    cfg.tasks_mode = args.tasks_mode
    cfg.require_all_inputs_present = args.require_all_inputs_present
    cfg.seed = args.seed
    cfg.split_train = args.split_train
    cfg.split_val = args.split_val
    cfg.split_test = args.split_test
    cfg.device = args.device
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.hidden = args.hidden
    cfg.dropout = args.dropout
    cfg.stack_alpha = args.stack_alpha
    cfg.sign_fix_enable = args.sign_fix_enable
    cfg.sign_fix_r_threshold = args.sign_fix_r_threshold
    cfg.sign_fix_min_pos_r = args.sign_fix_min_pos_r
    cfg.sign_fix_report_topn = args.sign_fix_report_topn
    cfg.watch_genes = tuple(args.watch_genes)
    return cfg
