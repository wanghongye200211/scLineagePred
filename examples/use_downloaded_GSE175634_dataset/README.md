# Use the Downloaded GSE175634 Dataset

This folder gives a complete, reviewable path for running `scLineagePred` on the downloaded GSE175634 human cardiac differentiation dataset. It starts from the GEO supplementary files, builds the processed inputs, reconstructs trajectories, learns the latent representation, and then runs endpoint classification, gene-expression regression, and perturbation-based cell-state transition marker analysis.

The large GEO files and generated model outputs are not stored in this repository. The commands below assume that the downloaded files are available locally.

## 1. Download Data

Create a raw-data directory and download the required GEO supplementary files:

```bash
mkdir -p examples/use_downloaded_GSE175634_dataset/data/raw
cd examples/use_downloaded_GSE175634_dataset/data/raw

base=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE175nnn/GSE175634/suppl
curl -L -O ${base}/GSE175634_cell_counts.mtx.gz
curl -L -O ${base}/GSE175634_gene_indices_counts.tsv.gz
curl -L -O ${base}/GSE175634_cell_indices.tsv.gz
curl -L -O ${base}/GSE175634_cell_metadata.tsv.gz
```

The same files are listed on the GEO record for GSE175634:
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175634>.

## 2. Install

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

GPU acceleration is optional but recommended for the full workflow. The commands are valid on CPU, CUDA, or Apple MPS, although runtime differs substantially.
For trajectory reconstruction, set `TRAJECTORY_DEVICE=mps` or `TRAJECTORY_DEVICE=cuda` if that backend is available; the wrapper defaults to `cpu` for portability.
The wrapper also sets local `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` paths under `work/` so Scanpy and Matplotlib do not need write access to user-level cache directories.

## 3. Run the Pipeline

The wrapper script records the full command sequence. By default it uses the full downloaded dataset and manuscript-scale settings.

```bash
bash examples/use_downloaded_GSE175634_dataset/run_gse175634_pipeline.sh
```

For a quick installation check, run a downsampled smoke path with shorter training:

```bash
SMOKE=1 bash examples/use_downloaded_GSE175634_dataset/run_gse175634_pipeline.sh
```

The smoke run verifies the file flow and command interfaces, but it is not expected to reproduce the benchmark values below.

## 4. Workflow Stages

The wrapper performs these stages:

| Stage | Script or command | Main outputs |
|---|---|---|
| Raw GEO to AnnData | `scripts/00_build_h5ad_from_geo.py` | `work/GSE175634_counts_with_metadata.h5ad` |
| Preprocessing | `scripts/01_preprocess_for_sclineagepred.py` | `work/preprocess/processed_norm_log_hvg1000.h5ad`, `ruot_input_pca30_forward.csv`, `ruot_mapping_pca30_forward.tsv` |
| Trajectory reconstruction | `python -m scLineagePred trajectory train --evaluate` | `work/trajectory/GSE175634/sde_point_*.npy`, `sde_weight_*.npy` |
| Dimensionality reduction / latent representation | `python -m scLineagePred embedding train` | `work/embedding/Z_cells.npy`, `Z_genes.npy`, `genes.txt` |
| Attach latent space | `scripts/02_attach_embedding_latent.py` | `work/processed/GSE175634_with_latent.h5ad` |
| Pseudo-clonal sequence assembly | `scripts/03_build_pseudoclone_sequences.py` | `work/processed/pseudoclone_sequences.csv`, `GSE175634_with_latent_and_clone.h5ad` |
| Sequence H5 construction | `scripts/04_build_sequence_h5.py` | `work/processed/GSE175634_CMvsCF_all_generated_sequences.h5`, `GSE175634_CMvsCF_all_generated_index.csv` |
| Classification | `python -m scLineagePred classification train` | `work/classification/ensemble_summary.csv`, `predictions_*.csv`, saved models |
| Regression | `python -m scLineagePred regression train` | `work/regression/*/test_outputs.npz`, checkpoints |
| Perturbation | `python -m scLineagePred perturbation train` | `work/perturbation/cell_state_transition_markers_*`, latent-dimension sensitivity outputs |

## 5. Benchmark Comparison

Benchmarking compares endpoint classification for D15 CM/CF outcomes using observation windows ending at D1, D3, D5, D7, and D11. The benchmark uses the same held-out endpoint labels for all methods and reports AUROC, accuracy, and log loss.

The representative manuscript-scale results are included in:

- `benchmark/benchmark_metrics_gse175634.csv`
- `benchmark/sclineagepred_classification_summary.csv`
- `benchmark/regression_summary_gse175634.csv`

To recompute metrics from prediction files:

```bash
python examples/use_downloaded_GSE175634_dataset/benchmark/compare_binary_predictions.py \
  --setting Obs_Day5 \
  --method scLineagePred=work/classification/predictions_Obs_Day5.csv \
  --method CellRank=/path/to/cellrank_obs_day5_predictions.csv \
  --method WOT=/path/to/wot_obs_day5_predictions.csv \
  --method CoSpar=/path/to/cospar_obs_day5_predictions.csv \
  --positive-label CF \
  --out-csv work/benchmark/Obs_Day5_metrics.csv
```

Each prediction CSV should contain a `y_true` column and one positive-class probability column. Accepted probability column names include `prob_CF`, `prob_1`, `prob_positive`, `score`, or `probability`.

## 6. Expected Full-Run Results

The included benchmark table summarizes the full-run endpoint prediction comparison:

| Setting | scLineagePred AUROC | CellRank AUROC | WOT AUROC | CoSpar AUROC |
|---|---:|---:|---:|---:|
| Obs_Day1 | 0.8509 | 0.5828 | 0.4274 | 0.3218 |
| Obs_Day3 | 0.8934 | 0.5930 | 0.5174 | 0.3544 |
| Obs_Day5 | 0.9182 | 0.7890 | 0.7432 | 0.6666 |
| Obs_Day7 | 0.9422 | 0.8461 | 0.8391 | 0.8458 |
| Obs_Day11 | 0.9647 | 0.9218 | 0.9297 | 0.9103 |

Regression on the D15 endpoint yielded clone-level agreement of `R2=0.9977` for CF and `R2=0.9957` for CM in the representative full run.

## 7. Notes for Reviewers

- This example intentionally does not commit large `.h5ad`, `.h5`, checkpoint, or figure files.
- The full GSE175634 run is large; use `SMOKE=1` to validate the workflow quickly.
- The benchmark CSV files are small, committed summaries of the manuscript-scale run and can be used to verify reported numbers independently of the large intermediate files.
