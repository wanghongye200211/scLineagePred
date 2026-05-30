#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

for env_name, dirname in (("MPLCONFIGDIR", "sclineagepred_mpl"), ("NUMBA_CACHE_DIR", "sclineagepred_numba")):
    cache_dir = Path(os.environ.setdefault(env_name, str(Path(tempfile.gettempdir()) / dirname)))
    cache_dir.mkdir(parents=True, exist_ok=True)

import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GSE175634 and export scLineagePred inputs.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--time-col", default="diffday")
    parser.add_argument("--state-col", default="type")
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--min-genes", type=int, default=500)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-hvg", type=int, default=1000)
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--max-cells-per-time", type=int, default=0, help="0 keeps all cells; positive values downsample per time point")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_time(value) -> float:
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
    return float(matches[0]) if matches else 0.0


def downsample_per_time(adata: ad.AnnData, time_col: str, max_cells: int, seed: int) -> ad.AnnData:
    if max_cells <= 0:
        return adata
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in adata.obs.groupby(time_col, observed=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) > max_cells:
            idx = rng.choice(idx, size=max_cells, replace=False)
        keep.append(idx)
    selected = np.sort(np.concatenate(keep))
    return adata[selected].copy()


def main() -> None:
    args = parse_args()
    input_h5ad = Path(args.input_h5ad).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)
    if args.time_col not in adata.obs:
        raise KeyError(f"Missing time column {args.time_col!r}; available columns: {list(adata.obs.columns)}")

    adata = downsample_per_time(adata, args.time_col, args.max_cells_per_time, args.seed)
    print(f"[input] {adata.n_obs} cells x {adata.n_vars} genes")

    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_genes=args.min_genes)

    has_negative = bool(adata.X.data.min() < 0) if sp.issparse(adata.X) and adata.X.nnz else bool(np.min(adata.X) < 0)
    if not has_negative:
        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat", subset=True)

    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    processed_h5ad = out_dir / f"processed_norm_log_hvg{args.n_hvg}.h5ad"
    adata.write_h5ad(processed_h5ad, compression="gzip")

    gene_txt = out_dir / f"hvg{args.n_hvg}_genes.txt"
    gene_txt.write_text("\n".join(map(str, adata.var_names)) + "\n", encoding="utf-8")

    raw_times = adata.obs[args.time_col].astype(str).to_numpy()
    parsed = np.array([parse_time(value) for value in raw_times], dtype=float)
    unique_times = sorted(np.unique(parsed).tolist())
    time_to_index = {time: index for index, time in enumerate(unique_times)}
    samples = np.array([time_to_index[value] for value in parsed], dtype=np.int64)

    X = adata.X
    if sp.issparse(X):
        reducer = TruncatedSVD(n_components=args.pca_dim, random_state=args.seed)
        X_pca = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=args.pca_dim, random_state=args.seed, svd_solver="randomized")
        X_pca = reducer.fit_transform(np.asarray(X))
    X_pca = StandardScaler().fit_transform(X_pca).astype(np.float32)

    mapping = pd.DataFrame(
        {
            "row_id": np.arange(adata.n_obs, dtype=np.int64),
            "cell_id": adata.obs_names.astype(str),
            "original_time": raw_times,
            "samples": samples,
            "parsed_time": parsed,
        }
    )
    if args.state_col in adata.obs:
        mapping["state"] = adata.obs[args.state_col].astype(str).to_numpy()
    else:
        mapping["state"] = "unknown"

    pca_cols = [f"x{i + 1}" for i in range(args.pca_dim)]
    pca_frame = pd.DataFrame(X_pca, columns=pca_cols)
    pca_frame.insert(0, "samples", samples)

    direction = "forward"
    ruot_csv = out_dir / f"ruot_input_pca{args.pca_dim}_{direction}.csv"
    ruot_mapping = out_dir / f"ruot_mapping_pca{args.pca_dim}_{direction}.tsv"
    ruot_meta = out_dir / f"ruot_meta_pca{args.pca_dim}_{direction}.json"

    pca_frame.to_csv(ruot_csv, index=False)
    mapping.to_csv(ruot_mapping, sep="\t", index=False)
    ruot_meta.write_text(
        json.dumps(
            {
                "source": str(input_h5ad),
                "processed_h5ad": str(processed_h5ad),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "pca_dim": int(args.pca_dim),
                "time_col": args.time_col,
                "state_col": args.state_col,
                "time_map": {str(i): float(t) for i, t in enumerate(unique_times)},
                "files": {"csv": ruot_csv.name, "mapping": ruot_mapping.name},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[saved] {processed_h5ad}")
    print(f"[saved] {ruot_csv}")
    print(f"[saved] {ruot_mapping}")


if __name__ == "__main__":
    main()
