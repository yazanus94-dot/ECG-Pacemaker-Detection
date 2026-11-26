#!/usr/bin/env python
"""
Synthetic ECG Data Generator for Pacemaker Detection Demo
Generates realistic-looking synthetic ECG data for demonstration purposes.
No real patient data is used or stored.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import wfdb


def generate_synthetic_ecg(n_samples=5000, n_leads=12, fs=500, seed=42):
    """
    Generate synthetic ECG signal with realistic morphology.

    Args:
        n_samples: Number of time samples
        n_leads: Number of ECG leads (standard 12-lead)
        fs: Sampling frequency in Hz
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (n_leads, n_samples)
    """
    np.random.seed(seed)
    t = np.arange(n_samples) / fs

    # Base ECG components
    ecg = np.zeros((n_leads, n_samples))

    # Heart rate ~60-100 bpm
    hr = np.random.uniform(60, 100)
    rr_interval = 60 / hr  # seconds per beat
    n_beats = int(n_samples / fs / rr_interval)

    for lead in range(n_leads):
        signal = np.zeros(n_samples)

        # Generate beats
        for i in range(n_beats):
            beat_center = int((i + 0.5) * rr_interval * fs)
            if beat_center >= n_samples:
                break

            # P wave
            p_width = int(0.08 * fs)
            p_amp = np.random.uniform(0.05, 0.15)
            p_start = beat_center - int(0.2 * fs)
            if p_start >= 0:
                p_t = np.arange(p_width) / fs
                signal[p_start : p_start + p_width] += p_amp * np.sin(
                    np.pi * p_t / (p_width / fs)
                )

            # QRS complex
            qrs_width = int(0.08 * fs)
            qrs_amp = np.random.uniform(0.8, 1.5)
            qrs_start = beat_center - qrs_width // 2
            if 0 <= qrs_start < n_samples - qrs_width:
                qrs_t = np.arange(qrs_width) / fs
                # Triphasic QRS
                signal[qrs_start : qrs_start + qrs_width] += qrs_amp * (
                    -0.3 * np.exp(-(((qrs_t - 0.02) / 0.01) ** 2))  # Q
                    + 1.0 * np.exp(-(((qrs_t - 0.04) / 0.008) ** 2))  # R
                    + -0.4 * np.exp(-(((qrs_t - 0.06) / 0.012) ** 2))  # S
                )

            # T wave
            t_width = int(0.16 * fs)
            t_amp = np.random.uniform(0.2, 0.5)
            t_start = beat_center + int(0.15 * fs)
            if t_start >= 0 and t_start + t_width < n_samples:
                t_t = np.arange(t_width) / fs
                signal[t_start : t_start + t_width] += t_amp * np.sin(
                    np.pi * t_t / (t_width / fs)
                )

        # Add baseline wander
        bw_freq = np.random.uniform(0.1, 0.5)
        bw_amp = np.random.uniform(0.02, 0.05)
        signal += bw_amp * np.sin(2 * np.pi * bw_freq * t)

        # Add muscle noise
        noise = np.random.normal(0, 0.01, n_samples)
        signal += noise

        # Add powerline interference (50/60 Hz)
        pl_freq = np.random.choice([50, 60])
        pl_amp = np.random.uniform(0.01, 0.03)
        signal += pl_amp * np.sin(2 * np.pi * pl_freq * t)

        ecg[lead] = signal

    return ecg


def generate_pacemaker_spikes(ecg, fs=500, spike_prob=0.3):
    """
    Add pacemaker spikes to ECG (simulating pacing).

    Args:
        ecg: ECG signal array (n_leads, n_samples)
        fs: Sampling frequency
        spike_prob: Probability of pacing per beat

    Returns:
        np.ndarray: ECG with pacemaker spikes
    """
    n_leads, n_samples = ecg.shape
    ecg_paced = ecg.copy()

    # Pacemaker spike characteristics
    spike_width = int(0.002 * fs)  # 2ms
    spike_amp = np.random.uniform(2.0, 5.0)  # Large amplitude

    # Estimate beat positions from R peaks (lead II)
    lead_ii = ecg[1]  # Lead II typically shows clear R peaks
    # Simple peak detection
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(lead_ii, distance=int(0.4 * fs), prominence=0.5)

    for peak in peaks:
        if np.random.random() < spike_prob:
            # Add spike before R peak
            spike_pos = peak - int(0.01 * fs)  # 10ms before R
            if 0 <= spike_pos < n_samples - spike_width:
                for lead in range(n_leads):
                    # Spike appears in all leads but varies
                    lead_spike_amp = spike_amp * np.random.uniform(0.5, 1.5)
                    spike_shape = lead_spike_amp * np.exp(
                        -np.arange(spike_width) ** 2 / (spike_width / 3) ** 2
                    )
                    ecg_paced[lead, spike_pos : spike_pos + spike_width] += spike_shape

    return ecg_paced


def create_demo_dataset(output_dir, n_records=20, paced_ratio=0.3):
    """
    Create synthetic demo dataset with WFDB format files.

    Args:
        output_dir: Output directory path
        n_records: Number of records to generate
        paced_ratio: Fraction of records with pacemaker
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    n_paced = int(n_records * paced_ratio)

    for i in range(n_records):
        record_id = f"SYNTH_{i:04d}"
        has_pacemaker = i < n_paced

        # Generate base ECG
        ecg = generate_synthetic_ecg(seed=1000 + i)

        # Add pacemaker spikes if needed
        if has_pacemaker:
            ecg = generate_pacemaker_spikes(ecg)
            label = 1
        else:
            label = 0

        # Save as WFDB format
        # .hea header
        header_lines = [
            f"{record_id} 12 {len(ecg[0])} 500",
        ]
        for lead_idx in range(12):
            header_lines.append(
                f"{record_id}.dat 16+24 1000/mv 16 0 0 -32768 0 0 0 0 0"
            )

        with open(output_dir / f"{record_id}.hea", "w") as f:
            f.write("\n".join(header_lines))

        # .dat binary data (16-bit integers)
        # Scale to 16-bit range
        ecg_scaled = (ecg * 1000).astype(np.int16)  # mV * 1000
        ecg_scaled.T.tofile(output_dir / f"{record_id}.dat")

        records.append({"Patient": record_id, "pacemaker": label})

    # Save labels CSV
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "demo_labels.csv", index=False)

    print(f"Generated {n_records} synthetic records in {output_dir}")
    print(f"  - {n_paced} with pacemaker")
    print(f"  - {n_records - n_paced} without pacemaker")
    print(f"Labels saved to {output_dir / 'demo_labels.csv'}")


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./demo_data"
    n_records = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    create_demo_dataset(output_dir, n_records=n_records)
