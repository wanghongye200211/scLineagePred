#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXAMPLE_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
RAW_DIR="${RAW_DIR:-${EXAMPLE_DIR}/data/raw}"
WORK_DIR="${WORK_DIR:-${EXAMPLE_DIR}/work}"
SMOKE="${SMOKE:-0}"
TRAJECTORY_DEVICE="${TRAJECTORY_DEVICE:-cpu}"

COUNTS_H5AD="${WORK_DIR}/GSE175634_counts_with_metadata.h5ad"
PREPROCESS_DIR="${WORK_DIR}/preprocess"
PROCESSED_DIR="${WORK_DIR}/processed"
EMBEDDING_DIR="${WORK_DIR}/embedding"
TRAJECTORY_DIR="${WORK_DIR}/trajectory"
CLASSIFICATION_DIR="${WORK_DIR}/classification"
REGRESSION_DIR="${WORK_DIR}/regression"
PERTURBATION_DIR="${WORK_DIR}/perturbation"
BENCHMARK_DIR="${WORK_DIR}/benchmark"

mkdir -p "${WORK_DIR}" "${PREPROCESS_DIR}" "${PROCESSED_DIR}" "${BENCHMARK_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_DIR}/.matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${WORK_DIR}/.numba_cache}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

if [[ "${SMOKE}" == "1" ]]; then
  MAX_CELLS_PER_TIME=300
  TRAJ_SAMPLE_SIZE=100
  TRAJ_PRETRAIN_EPOCHS=3
  TRAJ_SCORE_EPOCHS=3
  TRAJ_TRAIN_EPOCHS=3
  EMB_PRETRAIN_EPOCHS=3
  EMB_JOINT_EPOCHS=3
  CLS_EPOCHS=3
  REG_EPOCHS=3
else
  MAX_CELLS_PER_TIME=0
  TRAJ_SAMPLE_SIZE=500
  TRAJ_PRETRAIN_EPOCHS=500
  TRAJ_SCORE_EPOCHS=3001
  TRAJ_TRAIN_EPOCHS=500
  EMB_PRETRAIN_EPOCHS=200
  EMB_JOINT_EPOCHS=400
  CLS_EPOCHS=600
  REG_EPOCHS=80
fi

cd "${REPO_ROOT}"

echo "[1/9] Build AnnData from downloaded GEO files"
"${PYTHON}" "${EXAMPLE_DIR}/scripts/00_build_h5ad_from_geo.py" \
  --raw-dir "${RAW_DIR}" \
  --out-h5ad "${COUNTS_H5AD}"

echo "[2/9] Preprocess and prepare trajectory input"
"${PYTHON}" "${EXAMPLE_DIR}/scripts/01_preprocess_for_sclineagepred.py" \
  --input-h5ad "${COUNTS_H5AD}" \
  --out-dir "${PREPROCESS_DIR}" \
  --time-col diffday \
  --state-col type \
  --n-hvg 1000 \
  --pca-dim 30 \
  --max-cells-per-time "${MAX_CELLS_PER_TIME}"

TRAJECTORY_CONFIG="${WORK_DIR}/trajectory_gse175634.local.yaml"
cat > "${TRAJECTORY_CONFIG}" <<YAML
use_pinn: false
sample_with_replacement: false
device: "${TRAJECTORY_DEVICE}"
sample_size: ${TRAJ_SAMPLE_SIZE}
use_mass: false

exp:
  name: "GSE175634"
  output_dir: "${TRAJECTORY_DIR}"

data:
  file_path: "${PREPROCESS_DIR}/ruot_input_pca30_forward.csv"
  dim: 30
  hold_one_out: false
  hold_out: 6

model:
  in_out_dim: 30
  hidden_dim: 400
  n_hiddens: 2
  activation: "leakyrelu"
  score_hidden_dim: 128

pretrain:
  epochs: ${TRAJ_PRETRAIN_EPOCHS}
  lr: 0.0001
  lambda_ot: 1.0
  lambda_mass: 0.0
  lambda_energy: 0.0

score_train:
  epochs: ${TRAJ_SCORE_EPOCHS}
  lr: 0.0001
  lambda_penalty: 1
  sigma: 0.05
  score_batch_size: 50

train:
  epochs: ${TRAJ_TRAIN_EPOCHS}
  lr: 0.0001
  lambda_ot: 10
  lambda_mass: 0.0
  lambda_energy: 0.01
  lambda_pinn: 100
  lambda_initial: 0.1
  scheduler_step_size: 100
  scheduler_gamma: 0.8
YAML

echo "[3/9] Reconstruct trajectories"
"${PYTHON}" -m scLineagePred trajectory train \
  --config "${TRAJECTORY_CONFIG}" \
  --evaluate

echo "[4/9] Learn latent representation"
"${PYTHON}" -m scLineagePred embedding train \
  --expr-h5ad "${PREPROCESS_DIR}/processed_norm_log_hvg1000.h5ad" \
  --out-dir "${EMBEDDING_DIR}" \
  --epochs-pretrain-dgi "${EMB_PRETRAIN_EPOCHS}" \
  --epochs-joint "${EMB_JOINT_EPOCHS}" \
  --max-cells-for-gene-features 5000

echo "[5/9] Attach latent space"
"${PYTHON}" "${EXAMPLE_DIR}/scripts/02_attach_embedding_latent.py" \
  --input-h5ad "${PREPROCESS_DIR}/processed_norm_log_hvg1000.h5ad" \
  --z-cells-npy "${EMBEDDING_DIR}/Z_cells.npy" \
  --out-h5ad "${PROCESSED_DIR}/GSE175634_with_latent.h5ad"

echo "[6/9] Assemble pseudo-clonal sequences"
"${PYTHON}" "${EXAMPLE_DIR}/scripts/03_build_pseudoclone_sequences.py" \
  --ruot-input-csv "${PREPROCESS_DIR}/ruot_input_pca30_forward.csv" \
  --ruot-mapping-tsv "${PREPROCESS_DIR}/ruot_mapping_pca30_forward.tsv" \
  --trajectory-dir "${TRAJECTORY_DIR}/GSE175634" \
  --latent-h5ad "${PROCESSED_DIR}/GSE175634_with_latent.h5ad" \
  --out-dir "${PROCESSED_DIR}" \
  --keep-endpoint-type CM \
  --keep-endpoint-type CF

echo "[7/9] Build sequence H5"
"${PYTHON}" "${EXAMPLE_DIR}/scripts/04_build_sequence_h5.py" \
  --sequence-csv "${PROCESSED_DIR}/pseudoclone_sequences.csv" \
  --sequence-clone-npy "${PROCESSED_DIR}/pseudoclone_seq_clone.npy" \
  --ruot-mapping-tsv "${PREPROCESS_DIR}/ruot_mapping_pca30_forward.tsv" \
  --latent-h5ad "${PROCESSED_DIR}/GSE175634_with_latent.h5ad" \
  --out-prefix "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated" \
  --keep-class CM \
  --keep-class CF

echo "[8/9] Run classification, regression, and perturbation"
"${PYTHON}" -m scLineagePred classification train -- \
  --time-series-h5 "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated_sequences.h5" \
  --index-csv "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated_index.csv" \
  --out-dir "${CLASSIFICATION_DIR}" \
  --target-label CM \
  --target-label CF \
  --epochs "${CLS_EPOCHS}"

"${PYTHON}" -m scLineagePred regression train -- \
  --ae-result-dir "${EMBEDDING_DIR}" \
  --time-series-h5 "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated_sequences.h5" \
  --index-csv "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated_index.csv" \
  --adata-h5ad "${PROCESSED_DIR}/GSE175634_with_latent_and_clone.h5ad" \
  --out-dir "${REGRESSION_DIR}" \
  --keep-label CM \
  --keep-label CF \
  --epochs "${REG_EPOCHS}"

"${PYTHON}" -m scLineagePred perturbation train -- \
  --time-series-h5 "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated_sequences.h5" \
  --index-csv "${PROCESSED_DIR}/GSE175634_CMvsCF_all_generated_index.csv" \
  --model-dir "${CLASSIFICATION_DIR}/saved_models" \
  --decoder-dir "${EMBEDDING_DIR}" \
  --hvg-h5ad "${PROCESSED_DIR}/GSE175634_with_latent_and_clone.h5ad" \
  --out-dir "${PERTURBATION_DIR}" \
  --target-label CM \
  --target-label CF

echo "[9/9] Benchmark metrics from scLineagePred predictions"
for setting in Obs_Day1 Obs_Day3 Obs_Day5 Obs_Day7 Obs_Day11; do
  "${PYTHON}" "${EXAMPLE_DIR}/benchmark/compare_binary_predictions.py" \
    --setting "${setting}" \
    --method "scLineagePred=${CLASSIFICATION_DIR}/predictions_${setting}.csv" \
    --positive-label CF \
    --out-csv "${BENCHMARK_DIR}/${setting}_metrics.csv"
done

echo "[DONE] Outputs written under ${WORK_DIR}"
