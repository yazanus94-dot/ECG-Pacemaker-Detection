# Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request
targeting `main`. Every job is mandatory — there is no `continue-on-error`, no
`|| true` and no skipped test step.

| Job | What it proves | Gate |
|---|---|---|
| **Lint & Format** | `ruff check src/ tests/` and `black --check src/ tests/` | mandatory |
| **Run Tests** | `pytest tests/ -v` with coverage and JUnit artefacts | mandatory |
| **Build Docker Image** | the CPU image builds and `from src.model import CTN` works inside it | mandatory |
| **Security Scan** | `pip-audit -r requirements.txt` and TruffleHog | mandatory |
| **Notify on Failure** | aggregate warning so a red pipeline is never silent | informational |

## Why things are pinned the way they are

* **Python 3.11.** `torch` pulls `filelock`, and the releases that carry the
  fixes for the filelock advisories (CVE-2026-112906 … CVE-2026-112941) require
  Python ≥ 3.11. `requirements.txt` states this floor and the Dockerfile and
  every workflow job use `python:3.11`.
* **CPU-only image.** The previous CUDA/devel base image was several gigabytes
  and the extra toolchain added nothing to CI. The image installs pinned CPU
  PyTorch wheels from `download.pytorch.org/whl/cpu`.
* **No `torchvision`.** Nothing in `src/` imports it; the wheel was only
  inflating install time and the audit surface.
* **TruffleHog range vs history.** Pull requests scan
  `base.sha..head.sha`; pushes scan the full history. Giving the action
  `base: main` + `head: HEAD` on a push compares a commit with itself
  (`BASE and HEAD commits are the same`), which is what broke the old pipeline.
  `--fail` is not repeated in `extra_args` because the action passes it itself.

## Reproducing the gates locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements.txt -r requirements-dev.txt
pip install ruff==0.8.6 black==26.10.1 pip-audit

ruff check src/ tests/
black --check src/ tests/
pytest tests/ -q
pip-audit -r requirements.txt

docker build -t ecg-pacemaker-detection:local .
docker run --rm ecg-pacemaker-detection:local \
  python -c "from src.model import CTN; print('CTN import OK')"
```

## Test suite scope

The 18 tests in `tests/` are synthetic and structural: CTN construction, the
positional encoding, transformer pooling, forward shape/finiteness, evaluation
determinism, loss/backprop, one optimizer step, and the synthetic ECG generator
(shape, determinism, seed independence, physiology bounds, pacemaker spike
injection, WFDB layout).

They prove the code runs and the model contract holds. They are **not**
evidence of clinical accuracy, and no patient data is used or required.
