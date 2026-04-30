# scLineagePred

`scLineagePred` is a source-only GitHub project prepared from the original research workspace.

It keeps three top-level code blocks:

- `DeepRUOT/`: trajectory reconstruction code, wrapped for reusable CLI execution
- `autoencoder/`: embedding training code
- `scLineagePred/`: one unified downstream package with `classification/`, `regression/`, and `perturbation/`

This repository intentionally excludes datasets, checkpoints, figures, and intermediate results.
Old per-dataset archive scripts have been removed from the main repository so the structure stays focused on the unified pipeline only.

## Dataset Adaptation

Different datasets are handled through the same three downstream files:

1. Classification: pass dataset-specific endpoint labels with repeated `--target-label`.
2. Regression: pass the endpoint labels to keep with repeated `--keep-label`.
3. Perturbation: pass the same endpoint labels with repeated `--target-label`; the script derives perturbation scenarios from the last two timepoints automatically.

If a dataset has special paths or selected ROC settings, put those details at the top of `scLineagePred/classification/plot_roc.py`.

## Project Layout

```text
scLineagePred/
├── DeepRUOT/
├── autoencoder/
└── scLineagePred/
    ├── classification/
    │   ├── train.py
    │   └── plot_roc.py
    ├── regression/
    │   └── train.py
    └── perturbation/
        └── train.py
```

## Wrapped Entry Points

List available scripts:

```bash
python -m scLineagePred list
```

Run wrapped DeepRUOT training:

```bash
python -m scLineagePred trajectory train --config /path/to/config.yaml
python -m scLineagePred trajectory train --config /path/to/config.yaml --evaluate
```

Run wrapped embedding training:

```bash
python -m scLineagePred embedding train \
  --expr-h5ad /path/to/data.h5ad \
  --gene-names-txt /path/to/genes.txt \
  --net-tsv /path/to/network.tsv \
  --out-dir /path/to/output
```

Run unified classification:

```bash
python -m scLineagePred classification train -- \
  --time-series-h5 /path/to/sequences.h5 \
  --index-csv /path/to/index.csv \
  --out-dir /path/to/output \
  --target-label Alpha \
  --target-label Beta
```

Plot unified macro ROC curves:

```bash
python -m scLineagePred classification plot-roc -- \
  --result DatasetA=/path/to/run_a \
  --result DatasetB=/path/to/run_b \
  --out-dir /path/to/roc_output
```

Run unified regression:

```bash
python -m scLineagePred regression train -- \
  --ae-result-dir /path/to/embedding_output \
  --time-series-h5 /path/to/sequences.h5 \
  --index-csv /path/to/index.csv \
  --adata-h5ad /path/to/integrated.h5ad \
  --out-dir /path/to/output \
  --keep-label Alpha \
  --keep-label Beta
```

Run unified perturbation:

```bash
python -m scLineagePred perturbation train -- \
  --time-series-h5 /path/to/sequences.h5 \
  --index-csv /path/to/index.csv \
  --model-dir /path/to/classification_models \
  --decoder-dir /path/to/embedding_output \
  --hvg-h5ad /path/to/with_latent.h5ad \
  --out-dir /path/to/output \
  --target-label Alpha \
  --target-label Beta
```

Notes:

- `scLineagePred/classification/train.py`, `scLineagePred/classification/plot_roc.py`, `scLineagePred/regression/train.py`, and `scLineagePred/perturbation/train.py` are the main downstream entry points.
- The wrapped commands are focused on reusable interfaces instead of dataset-specific filenames.
