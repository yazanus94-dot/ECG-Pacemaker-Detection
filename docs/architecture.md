# Architecture — ECG-Pacemaker-Detection

## Model: Convolutional Transformer Network (CTN)

```
Input: 12-lead ECG (12 x 7500 samples @ 500 Hz, 15 s)
  -> Conv1d encoder (6 layers, 12->256, stride downsampling x20)
  -> Positional encoding
  -> Transformer encoder (multi-head self-attention)
  -> Pooling
  -> Classification head (paced vs non-paced)
```

## Data
- Synthetic ECG generator (no real patient data).
- 12-lead, 500 Hz, 15 s segments.

## Training
- PyTorch, Adam optimizer, cross-entropy loss.
- 5-fold stratified patient-level splits (synthetic).

## Inference
- Single forward pass -> paced probability.

## Note on provenance
Portfolio adaptation of a research collaboration (Sheba Medical Center AI Center for Cardiology + Philips Research North America). Real patient data removed; synthetic generator provided. Original published results are not independently reproduced here.
