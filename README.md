# scLineagePred

`scLineagePred` is a source-only GitHub project prepared from the original research workspace.

It keeps four code blocks:

- `DeepRUOT/`: trajectory reconstruction code, wrapped for reusable CLI execution
- `autoencoder/`: embedding training code
- `classification/`: classification scripts
- `regression/`: regression scripts

This repository intentionally excludes datasets, checkpoints, figures, and intermediate results.

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

List or launch legacy classification / regression scripts:

```bash
python -m sclineagepred classification list
python -m sclineagepred classification run class_140802

python -m sclineagepred regression list
python -m sclineagepred regression run regression_140802
```

Notes:

- `classification/` and `regression/` remain legacy research scripts with their original logic.
- The wrapped commands are focused on the reusable parts: trajectory reconstruction and embedding training.
- Most legacy scripts still expect the user to edit paths or configs before running them on new data.

