"""Tests for the synthetic ECG generator.

The generator produces the only data this repository can create without
credentialed PhysioNet/BDSP access. These tests pin its contract: shape,
determinism, physiological sanity and the on-disk WFDB demo layout.
"""

import numpy as np
import pytest

from src.generate_synthetic_data import (
    create_demo_dataset,
    generate_pacemaker_spikes,
    generate_synthetic_ecg,
)


def test_generate_synthetic_ecg_shape_and_dtype():
    ecg = generate_synthetic_ecg(n_samples=2000, n_leads=12, fs=500, seed=7)

    assert isinstance(ecg, np.ndarray)
    assert ecg.shape == (12, 2000)
    assert ecg.dtype == np.float64


def test_generate_synthetic_ecg_is_reproducible():
    first = generate_synthetic_ecg(n_samples=1200, seed=42)
    second = generate_synthetic_ecg(n_samples=1200, seed=42)

    np.testing.assert_array_equal(first, second)


def test_generate_synthetic_ecg_differs_across_seeds():
    first = generate_synthetic_ecg(n_samples=1200, seed=1)
    second = generate_synthetic_ecg(n_samples=1200, seed=2)

    assert not np.array_equal(first, second)


def test_generate_synthetic_ecg_is_finite_and_plausible():
    ecg = generate_synthetic_ecg(n_samples=3000, seed=3)

    assert np.isfinite(ecg).all()
    # QRS amplitude is drawn from [0.8, 1.5] with small noise on top; a value
    # above 5 mV would indicate a generation bug rather than a real waveform.
    assert np.abs(ecg).max() < 5.0
    # Every lead must carry a non-constant signal.
    assert all(np.ptp(lead) > 0.1 for lead in ecg)


def test_pacemaker_spikes_preserve_shape_and_raise_peak():
    np.random.seed(0)
    base = generate_synthetic_ecg(n_samples=3000, seed=11)
    np.random.seed(0)
    paced = generate_pacemaker_spikes(base, fs=500, spike_prob=1.0)

    assert paced.shape == base.shape
    assert np.isfinite(paced).all()
    # Spikes are strictly additive, so the absolute peak cannot decrease.
    assert np.abs(paced).max() >= np.abs(base).max()


def test_pacemaker_spikes_with_zero_probability_are_a_no_op():
    np.random.seed(0)
    base = generate_synthetic_ecg(n_samples=2000, seed=5)
    paced = generate_pacemaker_spikes(base, fs=500, spike_prob=0.0)

    np.testing.assert_allclose(paced, base)


def test_create_demo_dataset_writes_wfdb_records(tmp_path):
    create_demo_dataset(tmp_path, n_records=10, paced_ratio=0.3)

    headers = sorted(tmp_path.glob("SYNTH_*.hea"))
    data_files = sorted(tmp_path.glob("SYNTH_*.dat"))
    labels = tmp_path / "demo_labels.csv"

    assert len(headers) == 10
    assert len(data_files) == 10
    assert labels.is_file()

    rows = labels.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "Patient,pacemaker"
    assert len(rows) == 11
    # 30% of 10 records are labelled as paced.
    assert sum(1 for row in rows[1:] if row.endswith(",1")) == 3

    header = headers[0].read_text(encoding="utf-8").splitlines()
    assert header[0].split()[1] == "12"


@pytest.mark.parametrize("n_leads", [1, 3, 12])
def test_leads_are_independently_generated(n_leads):
    ecg = generate_synthetic_ecg(n_samples=800, n_leads=n_leads, seed=9)

    assert ecg.shape == (n_leads, 800)
    if n_leads > 1:
        assert not np.array_equal(ecg[0], ecg[1])
