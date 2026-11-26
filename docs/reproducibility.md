# Reproducibility — ECG-Pacemaker-Detection

## Environment
- Python 3.11+, PyTorch 2.1+. Install: `pip install -r requirements.txt`.

## Run tests
```
python -m pytest -q
```
Result: 18 passed (verified 2026-11-26).

## Docker
```
docker build -t ecg-pacemaker .
```
