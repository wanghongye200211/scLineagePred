# scLineagePred

`scLineagePred` is a Python codebase for three connected tasks in single-cell lineage analysis:

1. trajectory reconstruction
2. embedding training
3. downstream prediction with classification, regression, and perturbation

This repository keeps reusable source code only. Datasets, checkpoints, figures, and intermediate outputs are intentionally excluded.

## Repository Layout

```text
scLineagePred/
├── trajectory_reconstruction/
│   ├── config.yaml
│   ├── train.py
│   └── core/
├── autoencoder/
│   ├── dataio.py
│   ├── netmodel.py
│   ├── train_model.py
│   └── utils.py
└── scLineagePred/
    ├── classification/
    │   ├── config.py
    │   ├── data.py
    │   ├── models.py
    │   ├── plots.py
    │   ├── plot_roc.py
    │   └── train.py
    ├── regression/
    │   ├── config.py
    │   ├── data.py
    │   ├── models.py
    │   └── train.py
    └── perturbation/
        ├── config.py
        ├── data.py
        ├── drivers.py
        ├── models.py
        ├── scan.py
        └── train.py
```

The top level follows a simple GitHub-friendly pattern: one folder per stage, one main runnable entry per stage, and small helper modules next to it.

## Quick Start

List public entry points:

```bash
python -m scLineagePred list
```

Run trajectory reconstruction:

```bash
python -m scLineagePred trajectory train \
  --config trajectory_reconstruction/config.yaml
```

Run embedding training:

```bash
python -m scLineagePred embedding train \
  --expr-h5ad /path/to/data.h5ad \
  --gene-names-txt /path/to/genes.txt \
  --net-tsv /path/to/network.tsv \
  --out-dir /path/to/output
```

Run classification:

```bash
python -m scLineagePred classification train -- \
  --time-series-h5 /path/to/sequences.h5 \
  --index-csv /path/to/index.csv \
  --out-dir /path/to/output \
  --target-label Alpha \
  --target-label Beta
```

Plot macro ROC curves:

```bash
python -m scLineagePred classification plot-roc -- \
  --result DatasetA=/path/to/run_a \
  --result DatasetB=/path/to/run_b \
  --out-dir /path/to/roc_output
```

Run regression:

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

Run perturbation:

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

## Dataset Adaptation

The repository no longer keeps one script per dataset. Instead, special cases are handled in a few stable places:

- `classification/train.py`: select endpoint classes with repeated `--target-label`.
- `regression/train.py`: keep endpoint classes with repeated `--keep-label`.
- `perturbation/train.py`: reuse repeated `--target-label` and scan the final two observation windows.
- `classification/plot_roc.py`: if one dataset needs a custom ROC comparison preset, define it near the top of that file.

## Notes

- `trajectory_reconstruction/config.yaml` is the only user-facing YAML template for the trajectory stage.
- Internal defaults for trajectory reconstruction stay in Python code so the public config file can remain short.
- The downstream folders now expose one main workflow each, but the helper code is split into `config`, `data`, `models`, `plots`, `drivers`, or `scan` modules where that makes the code easier to maintain.
