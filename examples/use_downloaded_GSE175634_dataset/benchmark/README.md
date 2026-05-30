# GSE175634 Benchmark

This directory contains small, committed benchmark summaries for the downloaded GSE175634 example. The large prediction arrays, model checkpoints, and baseline intermediate files are intentionally excluded from GitHub.

## Files

- `benchmark_metrics_gse175634.csv`: endpoint classification comparison for D15 CM/CF prediction.
- `sclineagepred_classification_summary.csv`: scLineagePred-only classification performance across observation windows.
- `regression_summary_gse175634.csv`: endpoint gene-expression regression agreement for CF and CM.
- `compare_binary_predictions.py`: utility to recompute AUROC, accuracy, and log loss from prediction CSV files.

## Prediction CSV Contract

For each method and setting, provide a CSV with:

- `y_true`: binary label encoded as 0/1, where `1` is the positive endpoint.
- one positive-class probability column. Accepted names are `prob_<positive_label>`, `prob_1`, `prob_positive`, `score`, or `probability`.

Example:

```bash
python compare_binary_predictions.py \
  --setting Obs_Day5 \
  --method scLineagePred=/path/to/predictions_Obs_Day5.csv \
  --method CellRank=/path/to/cellrank_obs_day5_predictions.csv \
  --method WOT=/path/to/wot_obs_day5_predictions.csv \
  --method CoSpar=/path/to/cospar_obs_day5_predictions.csv \
  --positive-label CF \
  --out-csv Obs_Day5_metrics.csv
```
