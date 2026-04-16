# scLineagePred

`scLineagePred` is a source-only GitHub project prepared from the original research workspace.

It keeps three top-level code blocks:

- `DeepRUOT/`: trajectory reconstruction code, wrapped for reusable CLI execution
- `autoencoder/`: embedding training code
- `scLineagePred/`: one unified downstream package with `classification/`, `regression/`, and `perturbation/`

This repository intentionally excludes datasets, checkpoints, figures, and intermediate results.
Dataset-specific historical scripts are archived under each module's `legacy/` directory and are no longer the main interface.

## Project Layout

```text
scLineagePred/
├── DeepRUOT/
├── autoencoder/
└── scLineagePred/
    ├── classification/
    ├── regression/
    └── perturbation/
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

- `scLineagePred/classification/train.py`, `scLineagePred/regression/train.py`, and `scLineagePred/perturbation/train.py` are now the primary downstream entry points.
- Archived per-dataset scripts are still available through `python -m scLineagePred legacy list <category>` and `python -m scLineagePred legacy run <category> <script>`.
- The wrapped commands are focused on reusable interfaces instead of dataset-specific filenames.
