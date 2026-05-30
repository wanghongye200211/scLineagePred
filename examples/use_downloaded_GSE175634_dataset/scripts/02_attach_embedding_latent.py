#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach scLineagePred embedding output to a processed AnnData file.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--z-cells-npy", required=True)
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--latent-key", default="X_latent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(Path(args.input_h5ad).expanduser())
    z_cells = np.asarray(np.load(Path(args.z_cells_npy).expanduser()), dtype=np.float32)
    if z_cells.shape[0] != adata.n_obs:
        raise ValueError(f"Latent rows ({z_cells.shape[0]}) do not match cells ({adata.n_obs}).")
    adata.obsm[args.latent_key] = z_cells
    adata.uns["latent_key"] = args.latent_key

    out_h5ad = Path(args.out_h5ad).expanduser()
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad, compression="gzip")
    print(f"[saved] {out_h5ad}")


if __name__ == "__main__":
    main()
