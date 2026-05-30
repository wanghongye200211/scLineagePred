#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequence H5 and index CSV for GSE175634 endpoint prediction.")
    parser.add_argument("--sequence-csv", required=True)
    parser.add_argument("--sequence-clone-npy", required=True)
    parser.add_argument("--ruot-mapping-tsv", required=True)
    parser.add_argument("--latent-h5ad", required=True)
    parser.add_argument("--out-prefix", required=True, help="Output prefix; writes <prefix>_sequences.h5 and <prefix>_index.csv")
    parser.add_argument("--keep-class", action="append", default=["CM", "CF"], help="Endpoint class to keep; repeat in desired label order")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--state-col", default="state")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--write-chunk", type=int, default=4096)
    return parser.parse_args()


def sorted_idx_cols(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in frame.columns if col.startswith("idx_t")]
    cols.sort(key=lambda col: int(col[5:]) if col[5:].isdigit() else 10**9)
    return cols


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sequence_csv = Path(args.sequence_csv).expanduser().resolve()
    sequence_clone_npy = Path(args.sequence_clone_npy).expanduser().resolve()
    mapping_tsv = Path(args.ruot_mapping_tsv).expanduser().resolve()
    latent_h5ad = Path(args.latent_h5ad).expanduser().resolve()
    out_prefix = Path(args.out_prefix).expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    seq = pd.read_csv(sequence_csv)
    if "endpoint_type" not in seq.columns:
        raise KeyError("Sequence CSV must contain endpoint_type.")
    if "clone_root" not in seq.columns:
        raise KeyError("Sequence CSV must contain clone_root.")

    keep_classes = list(dict.fromkeys(args.keep_class))
    seq["endpoint_type"] = seq["endpoint_type"].astype(str).str.strip()
    seq = seq[seq["endpoint_type"].isin(keep_classes)].reset_index(drop=True)
    seq["label_str"] = seq["endpoint_type"]

    idx_cols = sorted_idx_cols(seq)
    if not idx_cols:
        raise KeyError("Sequence CSV has no idx_t* columns.")
    row_indices = seq[idx_cols].to_numpy(dtype=np.int64)
    n_sequences, n_times = row_indices.shape

    if "clone_id" in seq.columns:
        seq_clone = seq["clone_id"].to_numpy(dtype=np.int64)
    elif sequence_clone_npy.exists():
        seq_clone = np.load(sequence_clone_npy).astype(np.int64)
        if len(seq_clone) != n_sequences:
            raise ValueError(f"Clone array length {len(seq_clone)} != sequence rows {n_sequences}.")
    else:
        seq_clone, _ = pd.factorize(seq["clone_root"].astype(str), sort=True)
        seq_clone = seq_clone.astype(np.int64)

    mapping = pd.read_csv(mapping_tsv, sep="\t").sort_values("row_id").reset_index(drop=True)
    required = {"row_id", "cell_id", "samples", args.state_col}
    missing = required - set(mapping.columns)
    if missing:
        raise KeyError(f"Mapping is missing columns: {sorted(missing)}")

    row_id = mapping["row_id"].to_numpy(np.int64)
    if not np.array_equal(row_id, np.arange(len(row_id))):
        raise ValueError("Mapping row_id must be contiguous and sorted.")
    if row_indices.min() < 0 or row_indices.max() >= len(mapping):
        raise ValueError("Sequence idx_t values are outside mapping row bounds.")

    adata = ad.read_h5ad(latent_h5ad)
    if args.latent_key not in adata.obsm:
        raise KeyError(f"{latent_h5ad} is missing obsm/{args.latent_key}")
    latent = np.asarray(adata.obsm[args.latent_key], dtype=np.float32)

    if "cell_id" in adata.obs:
        adata_cell_id = adata.obs["cell_id"].astype(str).to_numpy()
    else:
        adata_cell_id = adata.obs_names.astype(str).to_numpy()
    map_cell_id = mapping["cell_id"].astype(str).to_numpy()
    row_to_adata = pd.Index(adata_cell_id).get_indexer(map_cell_id)
    missing_rows = np.where(row_to_adata < 0)[0]
    if len(missing_rows):
        examples = map_cell_id[missing_rows[:5]].tolist()
        raise ValueError(f"{len(missing_rows)} mapping cells are missing from latent h5ad, examples={examples}")

    adata_indices = row_to_adata[row_indices].astype(np.int64)
    class_to_y = {label: index for index, label in enumerate(keep_classes)}
    y = np.array([class_to_y[label] for label in seq["label_str"].astype(str)], dtype=np.int64)

    perm = rng.permutation(n_sequences)
    n_train = int(0.8 * n_sequences)
    n_val = int(0.1 * n_sequences)
    train_idx = perm[:n_train].astype(np.int64)
    val_idx = perm[n_train : n_train + n_val].astype(np.int64)
    test_idx = perm[n_train + n_val :].astype(np.int64)

    out_index_csv = out_prefix.with_name(out_prefix.name + "_index.csv")
    out_h5 = out_prefix.with_name(out_prefix.name + "_sequences.h5")

    out_df = seq.copy()
    out_df["y"] = y
    out_df["split"] = "train"
    out_df.loc[val_idx, "split"] = "val"
    out_df.loc[test_idx, "split"] = "test"
    out_df["seq_clone"] = seq_clone
    out_df.to_csv(out_index_csv, index=False)

    unique_samples = sorted(mapping["samples"].unique().tolist())
    time_values = np.array(unique_samples, dtype=np.int64)
    if "original_time" in mapping:
        time_labels = []
        for sample in unique_samples:
            labels = mapping.loc[mapping["samples"] == sample, "original_time"].astype(str).value_counts()
            time_labels.append(labels.index[0])
        time_labels = np.array(time_labels, dtype=object)
    else:
        time_labels = np.array([str(value) for value in unique_samples], dtype=object)

    _, latent_dim = latent.shape
    with h5py.File(out_h5, "w") as handle:
        x_ds = handle.create_dataset(
            "X",
            shape=(n_sequences, n_times, latent_dim),
            dtype=np.float32,
            compression="gzip",
            chunks=(min(args.write_chunk, n_sequences), n_times, latent_dim),
        )
        handle.create_dataset("y", data=y)
        handle.create_dataset("indices", data=adata_indices)
        handle.create_dataset("label_str", data=seq["label_str"].astype(str).to_numpy().astype("S"))
        handle.create_dataset("classes", data=np.array(keep_classes, dtype="S"))
        handle.create_dataset("train_idx", data=train_idx)
        handle.create_dataset("val_idx", data=val_idx)
        handle.create_dataset("test_idx", data=test_idx)
        handle.create_dataset("seq_clone", data=seq_clone.astype(np.int64))
        handle.create_dataset("time_values", data=time_values)
        handle.create_dataset("time_labels", data=time_labels.astype("S"))

        for start in range(0, n_sequences, args.write_chunk):
            end = min(start + args.write_chunk, n_sequences)
            x_ds[start:end] = latent[adata_indices[start:end]]

    print(f"[saved] {out_index_csv}")
    print(f"[saved] {out_h5}")
    print(f"[summary] sequences={n_sequences} times={n_times} latent_dim={latent_dim}")


if __name__ == "__main__":
    main()
