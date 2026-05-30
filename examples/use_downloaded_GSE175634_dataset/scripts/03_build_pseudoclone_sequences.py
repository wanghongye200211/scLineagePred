#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


RUN_PATTERN = re.compile(r"sde_point_(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble pseudo-clonal sequences from trajectory SDE outputs.")
    parser.add_argument("--ruot-input-csv", required=True)
    parser.add_argument("--ruot-mapping-tsv", required=True)
    parser.add_argument("--trajectory-dir", required=True, help="Directory containing sde_point_*.npy and sde_weight_*.npy")
    parser.add_argument("--latent-h5ad", required=True, help="AnnData file with obsm/X_latent and expression matrix")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--keep-endpoint-type", action="append", default=["CM", "CF"])
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--seq-per-particle", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--knn-batch", type=int, default=20000)
    return parser.parse_args()


def sorted_x_cols(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in frame.columns if col.startswith("x")]
    cols.sort(key=lambda col: int(col[1:]) if col[1:].isdigit() else 10**9)
    return cols


def list_runs(trajectory_dir: Path) -> list[int]:
    runs = []
    for path in sorted(glob.glob(str(trajectory_dir / "sde_point_*.npy"))):
        match = RUN_PATTERN.search(path)
        if not match:
            continue
        run_id = int(match.group(1))
        if (trajectory_dir / f"sde_weight_{run_id}.npy").exists():
            runs.append(run_id)
    return sorted(set(runs))


def load_sde_point(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"sde_point must be 3D, got {array.shape} from {path}")
    return array


def load_sde_weight(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"sde_weight must be 3D, got {array.shape} from {path}")
    return array


def soft_prob_from_dist(distances: np.ndarray, tau: float) -> np.ndarray:
    dist2 = distances.astype(np.float64) ** 2
    probs = np.exp(-dist2 / (tau**2 + 1e-12))
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    return probs.astype(np.float32)


def kneighbors_batched(nn: NearestNeighbors, queries: np.ndarray, k: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    n = queries.shape[0]
    distances = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        dist, ind = nn.kneighbors(queries[start:end], n_neighbors=k, return_distance=True)
        distances[start:end] = dist.astype(np.float32)
        indices[start:end] = ind.astype(np.int64)
    return distances, indices


def path_hash(rows: np.ndarray) -> int:
    digest = hashlib.blake2b(rows.astype(np.int32, copy=False).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def write_h5ad_with_clone(latent_h5ad: Path, mapping: pd.DataFrame, out_h5ad: Path, latent_key: str) -> None:
    adata = ad.read_h5ad(latent_h5ad)
    if latent_key not in adata.obsm:
        raise KeyError(f"{latent_h5ad} is missing obsm/{latent_key}")

    map_cell_id = mapping["cell_id"].astype(str).to_numpy()
    if "cell_id" in adata.obs:
        adata_cell_id = adata.obs["cell_id"].astype(str).to_numpy()
    else:
        adata_cell_id = adata.obs_names.astype(str).to_numpy()

    order = pd.Index(adata_cell_id).get_indexer(map_cell_id)
    missing = np.where(order < 0)[0]
    if len(missing):
        examples = map_cell_id[missing[:5]].tolist()
        raise ValueError(f"{len(missing)} mapping cells are missing from latent h5ad, examples={examples}")

    adata = adata[order].copy()
    adata.obs["row_id"] = mapping["row_id"].to_numpy(np.int64)
    adata.obs["cell_id"] = map_cell_id
    adata.obs["samples"] = mapping["samples"].to_numpy(np.int32)
    adata.obs["state"] = mapping["state"].astype(str).to_numpy()
    if "original_time" in mapping:
        adata.obs["original_time"] = mapping["original_time"].astype(str).to_numpy()
    adata.obs["clone_root"] = map_cell_id
    clone_id, _ = pd.factorize(map_cell_id, sort=True)
    adata.obs["clone_id"] = clone_id.astype(np.int64)
    adata.uns["latent_key"] = latent_key
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad, compression="gzip")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ruot_input_csv = Path(args.ruot_input_csv).expanduser().resolve()
    ruot_mapping_tsv = Path(args.ruot_mapping_tsv).expanduser().resolve()
    trajectory_dir = Path(args.trajectory_dir).expanduser().resolve()
    latent_h5ad = Path(args.latent_h5ad).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ruot_input_csv)
    mapping = pd.read_csv(ruot_mapping_tsv, sep="\t").sort_values("row_id").reset_index(drop=True)
    required = {"row_id", "cell_id", "samples", "state"}
    missing_cols = required - set(mapping.columns)
    if missing_cols:
        raise KeyError(f"Mapping file is missing columns: {sorted(missing_cols)}")

    if len(df) != len(mapping):
        raise ValueError(f"Input/mapping row mismatch: {len(df)} != {len(mapping)}")
    if not np.array_equal(mapping["row_id"].to_numpy(np.int64), np.arange(len(mapping))):
        raise ValueError("mapping row_id must be contiguous and sorted from 0 to n-1.")

    x_cols = sorted_x_cols(df)
    if not x_cols:
        raise KeyError("RUOT input CSV has no x1..xK columns.")
    real = df[x_cols].to_numpy(np.float32)
    times = df["samples"].to_numpy(np.int32)
    unique_times = sorted(np.unique(times).tolist())
    rows_by_time = {time: np.where(times == time)[0] for time in unique_times}

    neighbors = {}
    for time in unique_times:
        rows = rows_by_time[time]
        model = NearestNeighbors(n_neighbors=min(args.knn_k, len(rows)), metric="euclidean")
        model.fit(real[rows])
        neighbors[time] = (model, rows)

    write_h5ad_with_clone(
        latent_h5ad=latent_h5ad,
        mapping=mapping,
        out_h5ad=out_dir / "GSE175634_with_latent_and_clone.h5ad",
        latent_key=args.latent_key,
    )

    runs = list_runs(trajectory_dir)
    if not runs:
        raise FileNotFoundError(f"No trajectory runs found in {trajectory_dir}; expected sde_point_*.npy and sde_weight_*.npy")

    cell_id = mapping["cell_id"].astype(str).to_numpy()
    state = mapping["state"].astype(str).to_numpy()
    time_label = mapping["original_time"].astype(str).to_numpy() if "original_time" in mapping else None
    keep_endpoint_types = set(args.keep_endpoint_type)

    sequence_csv = out_dir / "pseudoclone_sequences.csv"
    sequence_clone_npy = out_dir / "pseudoclone_seq_clone.npy"
    clone_map_tsv = out_dir / "pseudoclone_clone_map.tsv"

    header = (
        ["seq_id", "run_id", "particle_id", "rep_id", "clone_root", "clone_id", "endpoint", "endpoint_type", "endpoint_time", "w_end"]
        + [f"w_t{i}" for i in range(len(unique_times))]
        + [f"idx_t{i}" for i in range(len(unique_times))]
        + [f"id_t{i}" for i in range(len(unique_times))]
    )

    clone_root_to_id: dict[str, int] = {}
    clone_id_to_root: list[str] = []
    clone_counts: list[int] = []
    sequence_clones: list[int] = []
    seen_paths: set[int] = set()
    total = 0

    with sequence_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        for run_id in runs:
            points = load_sde_point(trajectory_dir / f"sde_point_{run_id}.npy")
            weights = load_sde_weight(trajectory_dir / f"sde_weight_{run_id}.npy")
            if points.shape[0] != len(unique_times) and points.shape[1] == len(unique_times):
                points = np.transpose(points, (1, 0, 2))
            if weights.shape[0] != len(unique_times) and weights.shape[1] == len(unique_times):
                weights = np.transpose(weights, (1, 0, 2))
            if points.shape[0] != len(unique_times):
                raise ValueError(f"Run {run_id} has {points.shape[0]} time points, expected {len(unique_times)}")

            n_particles = points.shape[1]
            root_file = trajectory_dir / f"sde_t0_row_id_{run_id}.npy"
            if root_file.exists():
                root_rows = np.asarray(np.load(root_file), dtype=np.int64)
                if root_rows.shape[0] != n_particles:
                    root_rows = None
            else:
                root_rows = None
            if root_rows is None:
                nn0, rows0 = neighbors[unique_times[0]]
                _, ind0 = kneighbors_batched(nn0, points[0], 1, args.knn_batch)
                root_rows = rows0[ind0[:, 0]]

            candidate_rows = []
            candidate_probs = []
            for time_index, time in enumerate(unique_times):
                nn_t, rows_t = neighbors[time]
                k = min(args.knn_k, len(rows_t))
                distances, local_idx = kneighbors_batched(nn_t, points[time_index], k, args.knn_batch)
                candidate_rows.append(rows_t[local_idx])
                candidate_probs.append(soft_prob_from_dist(distances, args.tau))

            for particle_id in range(n_particles):
                w_t = weights[:, particle_id, 0].astype(np.float32)
                for rep_id in range(args.seq_per_particle):
                    path_rows = np.empty(len(unique_times), dtype=np.int64)
                    path_rows[0] = int(root_rows[particle_id])
                    for time_index in range(1, len(unique_times)):
                        path_rows[time_index] = int(
                            rng.choice(candidate_rows[time_index][particle_id], p=candidate_probs[time_index][particle_id])
                        )

                    end_row = int(path_rows[-1])
                    endpoint_type = str(state[end_row])
                    if endpoint_type not in keep_endpoint_types:
                        continue

                    hashed = path_hash(path_rows)
                    if hashed in seen_paths:
                        continue
                    seen_paths.add(hashed)

                    clone_root = str(cell_id[int(path_rows[0])])
                    if clone_root in clone_root_to_id:
                        clone_id = clone_root_to_id[clone_root]
                        clone_counts[clone_id] += 1
                    else:
                        clone_id = len(clone_id_to_root)
                        clone_root_to_id[clone_root] = clone_id
                        clone_id_to_root.append(clone_root)
                        clone_counts.append(1)

                    row = {
                        "seq_id": total,
                        "run_id": run_id,
                        "particle_id": particle_id,
                        "rep_id": rep_id,
                        "clone_root": clone_root,
                        "clone_id": clone_id,
                        "endpoint": str(cell_id[end_row]),
                        "endpoint_type": endpoint_type,
                        "endpoint_time": str(time_label[end_row]) if time_label is not None else "",
                        "w_end": float(w_t[-1]),
                    }
                    for time_index in range(len(unique_times)):
                        row[f"w_t{time_index}"] = float(w_t[time_index])
                        row[f"idx_t{time_index}"] = int(path_rows[time_index])
                        row[f"id_t{time_index}"] = str(cell_id[int(path_rows[time_index])])
                    writer.writerow(row)
                    sequence_clones.append(clone_id)
                    total += 1

    np.save(sequence_clone_npy, np.array(sequence_clones, dtype=np.int64))
    pd.DataFrame(
        {
            "clone_id": np.arange(len(clone_id_to_root), dtype=np.int64),
            "clone_root": np.array(clone_id_to_root, dtype=object),
            "n_sequences": np.array(clone_counts, dtype=np.int64),
        }
    ).sort_values("n_sequences", ascending=False).to_csv(clone_map_tsv, sep="\t", index=False)

    print(f"[saved] {sequence_csv} rows={total}")
    print(f"[saved] {sequence_clone_npy}")
    print(f"[saved] {clone_map_tsv}")


if __name__ == "__main__":
    main()
