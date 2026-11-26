# ECG Pacemaker Detection — Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-CPU--only-blue.svg)](Dockerfile)

> **Portfolio Adaptation** — This repository is a modernized, sanitized reimplementation of a research collaboration between the **AI Center for Cardiology at Sheba Medical Center** and **Philips Research North America**. All real patient data has been removed; a synthetic data generator is provided for demonstration.

---

## 🎯 Project Overview

Automated detection of cardiac pacemakers from 12-lead ECG signals using a **Convolutional Transformer Network (CTN)**. The model identifies pacing spikes and classifies ECG recordings as paced vs. non-paced.

### Clinical Context
- Pacemaker patients require specialized ECG interpretation
- Pacing spikes can mimic or obscure pathological patterns
- Automated detection supports triage and reduces specialist workload
- Originally developed for the **PhysioNet 2020 Challenge** framework

### Key Results (from Original Research)
| Metric | Performance |
|--------|-------------|
| **ROC-AUC** | >0.95 on held-out test folds |
| **Challenge Metric** | Competitive with PhysioNet 2020 top entries |
| **Cross-validation** | 5-fold stratified, patient-level splits |

---

## 🧠 Model Architecture: CTN (Convolutional Transformer Network)

```
Input: 12-lead ECG (12 × 7500 samples @ 500Hz = 15 seconds)
         │
         ▼
┌─────────────────────────────────────┐
│  Conv1d Encoder (6 layers)          │
│  12 → 128 → 256 → 256 → 256 → 256  │
│  Strides: 3,3,2,2,1,1 (×20 downsample) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Transformer Encoder (8 layers)     │
│  d_model=256, 8 heads, d_ff=2048   │
│  Positional Encoding + Dropout(0.1) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Global Pooling → FC(256→64) → FC  │
│  Multi-label output (27 SNOMED codes) │
└─────────────────────────────────────┘
```

**Key Design Choices:**
- **Conv1d front-end** learns translation-invariant ECG features
- **Transformer** captures long-range temporal dependencies
- **QRS-aligned windowing** option for beat-level attention
- **Over-sampling** handles severe class imbalance (paced ≪ non-paced)

---

## 📊 Data Pipeline

### Original Data Sources (Not Included)
- **PhysioNet 2020 Challenge** — Public ECG databases
- **PTB-XL** — Large public 12-lead ECG database
- **Sheba Medical Center** — Private annotated cohort (used with permission)

### Portfolio Version — Synthetic Data
```bash
# Generate demo dataset
python src/generate_synthetic_data.py ./demo_data 50
```
Generates WFDB-format records (`.hea` + `.dat`) with realistic:
- P/QRS/T wave morphology
- Baseline wander, muscle noise, powerline interference
- **Pacemaker spikes** (2ms, high-amplitude, pre-R peak)
- Configurable paced/non-paced ratio

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CPU-only works; a CUDA 11.8+ GPU is optional for faster training
- 8GB+ RAM

### Installation
```bash
# Clone
git clone https://github.com/yazanus94-dot/ECG-Pacemaker-Detection.git
cd ECG-Pacemaker-Detection

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic demo data
python src/generate_synthetic_data.py ./demo_data 30
```

### Docker (Recommended)
```bash
# Build (CPU-only image: pinned CPU PyTorch wheels, Python 3.11)
docker build -t ecg-pacemaker .

# Verify the model imports and report the PyTorch build
docker run --rm ecg-pacemaker

# Run the test suite inside the image
docker run --rm -v "$PWD/tests:/app/tests" ecg-pacemaker \
  sh -c "pip install -q pytest && python -m pytest tests/ -q"
```

> The image is CPU-only on purpose: the previous CUDA/devel base image was
> multi-gigabyte and made CI builds slow and unreliable. Train on a GPU by
> installing CUDA PyTorch wheels in your own environment.

---

## 🏃 Training

```bash
# Train on synthetic data (demo)
python src/train_model.py ./demo_data ./output_models

# Original training script (requires real data)
python src/train_12ECG_classifier.py /path/to/physionet_data ./results
```

**Training Features:**
- 5-fold stratified cross-validation (pre-defined folds)
- Noam learning rate schedule (Transformer-style)
- Early stopping (patience=10, monitor=val AUROC)
- TensorBoard logging
- Class-weighted BCE loss for imbalance

### Expected Outputs
```
output_models/
└── ctn/
    ├── fold_0/
    │   ├── ctn.tar          # Best model checkpoint
    │   ├── thrs.txt         # Optimal thresholds
    │   ├── log_fold_0.csv   # Training logs
    │   └── val_0_results.csv
    ├── fold_1/ ...
    └── ...
```

---

## 🔬 Inference

```bash
# Single record inference
python src/driver.py ./model_checkpoint ./input_record_dir ./output_predictions

# Batch inference via run_12ECG_classifier
python src/run_12ECG_classifier.py ./model ./test_data ./predictions
```

**Output Format (PhysioNet Challenge):**
```csv
# Classes
10370003,426783006
# Labels (binary)
0,1
# Scores (probabilities)
0.123,0.987
```

---

## 📁 Project Structure

```
ECG-Pacemaker-Detection/
├── src/
│   ├── model.py                    # CTN architecture
│   ├── train_12ECG_classifier.py   # Main training pipeline
│   ├── train_model.py              # Training entry point
│   ├── driver.py                   # Inference driver
│   ├── dataloader.py               # QRS-aligned & padding datasets
│   ├── utils.py                    # Config, constants, data loading
│   ├── optimizer.py                # Noam optimizer
│   ├── over_sampling.py            # SMOTE-style over-sampling
│   ├── under_sampling.py           # Under-sampling
│   ├── generate_synthetic_data.py  # 🆕 Synthetic ECG generator
│   ├── StratifiedKFold.py          # Custom CV splits
│   ├── run_12ECG_classifier.py     # Challenge submission format
│   ├── eval/                       # PhysioNet evaluation metrics
│   └── feats/                      # Feature engineering modules
├── demo_data/                      # Generated synthetic data (gitignored)
├── requirements.txt                # Modern dependencies
├── Dockerfile                      # CPU-only image (Python 3.11)
├── LICENSE                         # BSD-2-Clause + attribution
└── README.md
```

---

## 🛠 Technical Details

### Preprocessing Pipeline
1. **Resampling** → 500 Hz uniform
2. **Bandpass filter** → 3–45 Hz FIR (order=150)
3. **Polarity correction** → Per-lead template median
4. **Normalization** → Per-sequence [-1, 1]
5. **Windowing** → 15s windows (QRS-aligned or random)

### Class Imbalance Handling
| Technique | Implementation |
|-----------|----------------|
| Over-sampling | Replicate minority class records in training folds |
| Class weights | `BCEWithLogitsLoss(pos_weight=...)` |
| Threshold optimization | Per-class sweep on validation set |

### Evaluation Metrics
- **Primary:** Macro AUROC, Challenge Metric (PhysioNet 2020)
- **Secondary:** Fβ (β=2), Gβ, Geometric Mean
- **Per-class:** AUROC, AUPRC, Sensitivity, Specificity

### Continuous Integration
`.github/workflows/ci.yml` runs four mandatory jobs on every push and pull
request: Ruff + Black, the `pytest` suite, a Docker build that must import
`CTN`, and a security scan (`pip-audit` + TruffleHog). No step is masked with
`continue-on-error` or `|| true`. See [`docs/CI.md`](docs/CI.md) for the exact
commands, the Python 3.11 requirement and how to reproduce every gate locally.

---

## ⚠️ Known Limitations

| Limitation | Mitigation |
|------------|------------|
| Synthetic data ≠ real clinical distribution | Use for architecture validation only |
| Legacy code comments in Hebrew | Preserved for attribution |
| 27 SNOMED codes but only 2 target classes | Full multi-label ready, demo uses binary |
| CUDA 10.1 → 12.1 migration untested on original weights | Retrain with modern stack |

---

## 📜 License & Attribution

**Code:** BSD 2-Clause (PhysioNet 2020) — see [LICENSE](LICENSE)

**Research Collaboration:**
- **Sheba Medical Center** — AI Center for Cardiology
- **Philips Research North America**

**Original Researchers:**
> Dvora Rothman, Elisheva Tufik, Sigal Sina, Gal Goshen, Raizy Kellerman, Michal Cohen-Shelly, Avi Sabbag (Sheba)
> Sara Mariani, Annamalai Natarajan, Asif Rahman, Yale Chang, Jonathan Rubin, Shruti Vij, Gregory Boverman (Philips)

**Data Sources:**
- PhysioNet 2020 Challenge (BSD-2-Clause)
- PTB-XL Database
- Sheba Medical Center private cohort (**not redistributed**)

**Conference:** Israel Cardiology Association Conference, May 2025 (poster)

---

## 👤 Portfolio Adaptation

This repository was adapted for portfolio demonstration by **Yazan Al-Jabari** ([@yazanus94-dot](https://github.com/yazanus94-dot)).

**Changes from Original:**
- ✅ Removed all patient data (CSV, feature zips, demographics)
- ✅ Added synthetic ECG data generator
- ✅ Modernized dependencies (PyTorch 2.1+, NumPy 1.24+, Python 3.11+)
- ✅ Reproducible CPU-only Docker image (Python 3.11, pinned CPU PyTorch wheels)
- ✅ Added comprehensive documentation
- ✅ Preserved all original attribution and licensing

**Disclaimer:** This is a **technical demonstration** of the ML pipeline architecture. It does not contain real patient data and should not be used for clinical decision-making.

---

## 🔗 Related Work

- PhysioNet 2020 Challenge: *Classification of 12-lead ECGs*
- PTB-XL: *A large publicly available ECG dataset*
- "Attention is All You Need" — Transformer architecture
- ECG signal processing with `wfdb` and `biosppy`

---

*Built with ❤️ for AI in Healthcare — Portfolio Project by Yazan Al-Jabari*