# scLineagePred

`scLineagePred` is a source-only GitHub project prepared from the original research workspace.

It keeps four integrated code blocks:

- `DeepRUOT/`: trajectory reconstruction code, wrapped for reusable CLI execution
- `autoencoder/`: embedding training code
- `classification/`: one unified classification pipeline
- `regression/`: one unified regression pipeline

This repository intentionally excludes datasets, checkpoints, figures, and intermediate results.
Dataset-specific historical scripts are archived under each module's `legacy/` directory and are no longer the main interface.

## Project Layout

```text
scLineagePred/
├── DeepRUOT/
├── autoencoder/
├── classification/
├── regression/
└── sclineagepred/
```

## Wrapped Entry Points

List available scripts:

```bash
python -m sclineagepred list
```

Run wrapped DeepRUOT training:

```bash
python -m sclineagepred trajectory train --config /path/to/config.yaml
python -m sclineagepred trajectory train --config /path/to/config.yaml --evaluate
```

Run wrapped embedding training:

```bash
python -m sclineagepred embedding train \
  --expr-h5ad /path/to/data.h5ad \
  --gene-names-txt /path/to/genes.txt \
  --net-tsv /path/to/network.tsv \
  --out-dir /path/to/output
```

Run unified classification:

```bash
python -m sclineagepred classification train -- \
  --time-series-h5 /path/to/sequences.h5 \
  --index-csv /path/to/index.csv \
  --out-dir /path/to/output \
  --target-label Alpha \
  --target-label Beta
```

Run unified regression:

```bash
python -m sclineagepred regression train -- \
  --ae-result-dir /path/to/embedding_output \
  --time-series-h5 /path/to/sequences.h5 \
  --index-csv /path/to/index.csv \
  --adata-h5ad /path/to/integrated.h5ad \
  --out-dir /path/to/output \
  --keep-label Alpha \
  --keep-label Beta
```

Notes:

- `classification/train.py` and `regression/train.py` are now the primary training entry points.
- Archived per-dataset scripts are still available through `python -m sclineagepred legacy list <category>` and `python -m sclineagepred legacy run <category> <script>`.
- The wrapped commands are focused on reusable interfaces instead of dataset-specific filenames.
