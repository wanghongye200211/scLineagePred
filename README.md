# scLineagePred

`scLineagePred` is a Python codebase for connected tasks in time-resolved single-cell lineage analysis:

1. trajectory reconstruction
2. embedding training
3. downstream prediction with classification, regression, and perturbation

This repository keeps reusable source code only. Datasets, trained checkpoints, figures, and intermediate outputs are intentionally excluded.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/wanghongye200211/scLineagePred.git
cd scLineagePred
python -m pip install -r requirements.txt
python -m pip install -e .
```

PyTorch and PyTorch Geometric installation can depend on the CUDA version available on the target machine. If GPU acceleration is required, install the matching `torch` and `torch-geometric` builds following their official instructions before running the workflows below.

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

## Data Availability

All datasets used in the manuscript are publicly available. The lineage-resolved hematopoiesis dataset (GSE140802) and the direct lineage reprogramming dataset (GSE99915) were obtained from the processed datasets distributed with CoSpar:

- CoSpar documentation: <https://cospar.readthedocs.io/en/latest/>
- Hematopoiesis tutorial: <https://cospar.readthedocs.io/en/latest/20210121_all_hematopoietic_data_v3.html>
- Reprogramming tutorial: <https://cospar.readthedocs.io/en/latest/20210121_reprogramming_static_barcoding_v2.html>

The pancreatic endocrine specification dataset (GSE114412), human cardiac differentiation dataset (GSE175634), and late pancreatic endocrinogenesis dataset (GSE132188) were downloaded from the Gene Expression Omnibus (GEO):

- GSE114412: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412>
- GSE175634: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175634>
- GSE132188: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188>

No new sequencing data were generated for this study. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for a manuscript-ready data and code availability statement.

## Code Availability

The source code and analysis scripts for scLineagePred are available in this repository. The repository contains reusable code for trajectory reconstruction, embedding training, classification, regression, and perturbation analyses. Large datasets, trained checkpoints, figures, and intermediate outputs are not included.

## Reproducibility Notes

- Install dependencies from `requirements.txt` and install the package with `python -m pip install -e .`.
- Prepare dataset-specific input files outside the repository to avoid committing large public datasets or generated artifacts.
- Use `trajectory_reconstruction/config.yaml` as the editable template for trajectory reconstruction.
- Use the command-line entry points in the Quick Start section to run the main workflow stages.

## Notes

- `trajectory_reconstruction/config.yaml` is the only user-facing YAML template for the trajectory stage.
- Internal defaults for trajectory reconstruction stay in Python code so the public config file can remain short.
- The downstream folders now expose one main workflow each, but the helper code is split into `config`, `data`, `models`, `plots`, `drivers`, or `scan` modules where that makes the code easier to maintain.

## License

No open-source license file is currently included. Add a license before publication if public reuse beyond manuscript review is intended.
