#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread


REQUIRED_FILES = {
    "matrix": "GSE175634_cell_counts.mtx.gz",
    "genes": "GSE175634_gene_indices_counts.tsv.gz",
    "cells": "GSE175634_cell_indices.tsv.gz",
    "metadata": "GSE175634_cell_metadata.tsv.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AnnData file from downloaded GSE175634 GEO files.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing downloaded GSE175634 supplementary files")
    parser.add_argument("--out-h5ad", required=True, help="Output .h5ad path")
    return parser.parse_args()


def read_tsv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip")


def safe_to_string(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def coerce_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    categorical_columns = {"cell", "exp.grp", "sample", "diffday", "individual", "leiden", "type"}

    for col in categorical_columns:
        if col in meta.columns:
            meta[col] = meta[col].map(safe_to_string)

    for col in meta.columns:
        if col in categorical_columns:
            continue
        if meta[col].dtype == "object":
            numeric = pd.to_numeric(meta[col], errors="coerce")
            if numeric.notna().mean() >= 0.95:
                meta[col] = numeric.astype("float32")

    for col in ("exp.grp", "diffday", "type", "leiden", "sample", "individual"):
        if col in meta.columns:
            meta[col] = meta[col].astype("category")
    return meta


def sanitize_obs(adata: ad.AnnData, category_threshold: int = 2000) -> None:
    for col in list(adata.obs.columns):
        if adata.obs[col].dtype == "object":
            adata.obs[col] = adata.obs[col].map(safe_to_string).astype("string")
    for col in list(adata.obs.columns):
        if str(adata.obs[col].dtype) == "string" and adata.obs[col].nunique(dropna=True) <= category_threshold:
            adata.obs[col] = adata.obs[col].astype("category")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_h5ad = Path(args.out_h5ad).expanduser().resolve()

    paths = {key: raw_dir / name for key, name in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing downloaded GSE175634 files:\n" + "\n".join(missing))

    print(f"[matrix] reading {paths['matrix']}")
    matrix = mmread(paths["matrix"]).tocsr().astype("float32")

    genes = read_tsv_gz(paths["genes"])
    cells = read_tsv_gz(paths["cells"])
    if not {"gene_index", "gene_name"}.issubset(genes.columns):
        raise ValueError(f"Unexpected gene index columns: {list(genes.columns)}")
    if not {"cell_index", "cell_name"}.issubset(cells.columns):
        raise ValueError(f"Unexpected cell index columns: {list(cells.columns)}")

    gene_names = genes["gene_name"].astype(str).tolist()
    cell_names = cells["cell_name"].astype(str).tolist()

    if matrix.shape == (len(gene_names), len(cell_names)):
        matrix = matrix.T.tocsr()
    elif matrix.shape != (len(cell_names), len(gene_names)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match cells x genes "
            f"({len(cell_names)}, {len(gene_names)}) or genes x cells."
        )

    obs = pd.DataFrame(index=pd.Index(cell_names, name="cell"))
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    if obs.index.has_duplicates:
        raise ValueError("Duplicate cell names detected in GSE175634 cell index file.")
    if var.index.has_duplicates:
        raise ValueError("Duplicate gene names detected in GSE175634 gene index file.")

    adata = ad.AnnData(X=matrix, obs=obs, var=var)

    meta = read_tsv_gz(paths["metadata"])
    if "cell" not in meta.columns:
        raise ValueError(f"Metadata must contain a 'cell' column, got: {list(meta.columns)}")
    meta = coerce_metadata(meta)
    meta["cell"] = meta["cell"].astype(str)
    adata.obs = adata.obs.join(meta.set_index("cell"), how="left")
    sanitize_obs(adata)

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad, compression="gzip")
    print(f"[saved] {out_h5ad}")
    print(f"[summary] cells={adata.n_obs} genes={adata.n_vars}")


if __name__ == "__main__":
    main()
