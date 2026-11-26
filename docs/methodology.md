# Methodology — ECG-Pacemaker-Detection

## Preprocessing
- 12-lead ECG resampled to 500 Hz.
- 15-second windows (7500 samples).
- Z-score normalization per lead.

## Model
- Conv1d encoder + Transformer encoder + classification head.
- Trained with cross-entropy on synthetic paced/non-paced labels.

## Evaluation
- 5-fold stratified CV at patient level.
- Metrics: ROC-AUC, accuracy, F1.

## Limitations
- Synthetic data only; no clinical validation.
- Original research performance (>0.95 AUC) is from the upstream study, not reproduced here.
