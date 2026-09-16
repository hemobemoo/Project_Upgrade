import logging
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CUSTOM KERAS COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3):
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def se_block(inputs, ratio=16):
    filters = inputs.shape[-1]
    se = keras.layers.GlobalAveragePooling1D()(inputs)
    se = keras.layers.Dense(filters // ratio, activation="relu")(se)
    se = keras.layers.Dense(filters, activation="sigmoid")(se)
    se = keras.layers.Reshape((1, filters))(se)
    return keras.layers.multiply([inputs, se])


def focal_loss_v3(y_true, y_pred_logits, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_true = y_true * (1.0 - 0.1) + 0.5 * 0.1
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
    y_pred = tf.nn.sigmoid(y_pred_logits)
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    loss = alpha * tf.pow((1 - p_t), gamma) * bce
    return tf.reduce_mean(loss)


CUSTOM_OBJECTS = {
    "focal_loss_v3": focal_loss_v3,
    "se_block": se_block,
    "transformer_encoder": transformer_encoder,
}

TARGET_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL LOADER & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_ecg_model(checkpoint_path) -> keras.Model | None:
    """
    Load the trained hybrid ECG model.

    Supports two formats:
      1. A single .keras file (Keras v3 native format).
      2. A config.json + model.h5 pair (architecture + weights).

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the .keras file, or to the directory containing config.json & model.h5.

    Returns
    -------
    keras.Model | None
        The loaded model, or None if no valid model is found (triggers Demo Mode).
    """
    path = Path(checkpoint_path)

    # ── Strategy 1: Native .keras file ───────────────────────────────────────
    if path.suffix == ".keras" and path.exists():
        try:
            model = keras.models.load_model(
                str(path),
                custom_objects=CUSTOM_OBJECTS,
                safe_mode=False,
            )
            log.info("Model loaded from %s", path)
            return model
        except Exception as exc:
            log.error("Failed to load .keras model: %s", exc)

    # ── Strategy 2: config.json + model.h5 ───────────────────────────────────
    search_dir = path if path.is_dir() else path.parent
    config_path = search_dir / "config.json"
    weights_path = search_dir / "model.h5"

    if config_path.exists() and weights_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_json = f.read()

            # Reconstruct architecture; custom layers (se_block, transformer_encoder)
            # are resolved via CUSTOM_OBJECTS.
            model = keras.models.model_from_json(config_json, custom_objects=CUSTOM_OBJECTS)

            # Attach weights. compile() is NOT required for predict().
            model.load_weights(str(weights_path))

            log.info("Model architecture loaded from %s, weights from %s", config_path, weights_path)
            return model
        except Exception as exc:
            log.error("Failed to load model from JSON + H5: %s", exc)
            return None

    log.warning(
        "No model found at %s (expected .keras file or a directory containing "
        "config.json + model.h5). Demo mode will use mock predictions.", checkpoint_path
    )
    return None


def get_signal_from_files(hea_bytes, hea_name, dat_bytes, dat_name, tmp_dir):
    import wfdb
    hea_path = os.path.join(tmp_dir, hea_name)
    dat_path = os.path.join(tmp_dir, dat_name)
    with open(hea_path, "wb") as f:
        f.write(hea_bytes)
    with open(dat_path, "wb") as f:
        f.write(dat_bytes)
    record_path = hea_path.replace(".hea", "")
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.fs


def load_sample_signal(sample_name: str = "13003_hr"):
    """
    Load a sample ECG signal from the bundled "Testing Samples" directory,
    for use by the Live Monitor demo/simulator.

    Reuses get_signal_from_files so the loading logic (wfdb read) is not
    duplicated.

    Parameters
    ----------
    sample_name : str
        Record name without extension (e.g. "13003_hr"), matching a
        <sample_name>.hea / <sample_name>.dat pair in "Testing Samples".

    Returns
    -------
    (signal_raw_data, fs) — same shape/format as get_signal_from_files.
    """
    import tempfile
    samples_dir = Path(__file__).parent / "Testing Samples"
    hea_path = samples_dir / f"{sample_name}.hea"
    dat_path = samples_dir / f"{sample_name}.dat"

    if not hea_path.exists():
        raise FileNotFoundError(f"Sample {hea_path.name} not found in {samples_dir}")
    if not dat_path.exists():
        raise FileNotFoundError(f"Sample {dat_path.name} not found in {samples_dir}")

    hea_bytes = hea_path.read_bytes()
    dat_bytes = dat_path.read_bytes()

    with tempfile.TemporaryDirectory() as tmp_dir:
        return get_signal_from_files(hea_bytes, hea_path.name, dat_bytes, dat_path.name, tmp_dir)


def list_sample_signals() -> list[str]:
    """Return available sample record names (without extension) for the Live Monitor."""
    samples_dir = Path(__file__).parent / "Testing Samples"
    if not samples_dir.exists():
        return []
    return sorted({p.stem for p in samples_dir.glob("*.hea")})


def build_demo_stream(sample_name: str = "13003_hr", repeats: int = 4):
    """
    Build a longer, continuous demo ECG stream by tiling a real sample record.

    This gives the Live Monitor simulator enough duration to slice several
    sliding windows (via slice_signal_windows) without writing a new signal
    loader — the record itself is still loaded through load_sample_signal /
    get_signal_from_files.

    Parameters
    ----------
    sample_name : str
        Record name in "Testing Samples" (without extension).
    repeats : int
        How many times to tile the record to build a longer stream.

    Returns
    -------
    (continuous_signal, fs)
    """
    signal, fs = load_sample_signal(sample_name)
    tiled = np.tile(signal, (max(1, repeats), 1))
    return tiled, fs


# ══════════════════════════════════════════════════════════════════════════════
# 3. HEART RATE & RHYTHM ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_heart_rate(signal: np.ndarray, fs: float) -> int:
    lead0 = signal[:, 0] if signal.ndim > 1 else signal
    peaks, _ = find_peaks(lead0, distance=int(fs * 0.5), prominence=0.3)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        return int(60 / np.mean(rr_intervals))
    return 70


def estimate_rhythm_regularity(signal: np.ndarray, fs: float) -> tuple[str, str]:
    lead0 = signal[:, 0] if signal.ndim > 1 else signal
    peaks, _ = find_peaks(lead0, distance=int(fs * 0.5), prominence=0.3)

    if len(peaks) < 3:
        return "Undetermined", "Undetermined"

    rr = np.diff(peaks) / fs
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    cv = std_rr / mean_rr
    hr = 60 / mean_rr

    regularity = "Regular" if cv < 0.05 else "Mostly Regular" if cv < 0.15 else "Irregular"
    rhythm = ("Sinus Bradycardia" if hr < 60 else
              "Normal Sinus Rhythm" if hr <= 100 else
              "Sinus Tachycardia" if hr <= 150 else "Tachycardia")
    if cv > 0.20:
        rhythm = "Possible Atrial Fibrillation"
    return rhythm, regularity


# ══════════════════════════════════════════════════════════════════════════════
# 4. STREAMING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def slice_signal_windows(signal: np.ndarray, fs: float, window_sec: float = 10.0, overlap_sec: float = 5.0):
    """
    Slice a continuous signal into overlapping windows.
    Returns list of (window_array, start_sample, end_sample).
    """
    window_size = int(fs * window_sec)
    step_size = int(fs * (window_sec - overlap_sec))
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        windows.append((signal[start:end], start, end))
    return windows


def simulate_continuous_signal(duration_sec: float = 60.0, fs: float = 500.0, n_leads: int = 12, seed: int = 42):
    """Generate a synthetic continuous ECG-like signal for demo streaming."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * fs)) / fs
    signal = np.zeros((len(t), n_leads))
    for i in range(n_leads):
        base = 0.3 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.normal(size=len(t))
        peak_interval = int(fs / 1.2)
        for pk in range(peak_interval, len(t), peak_interval):
            if pk < len(t) - 10:
                base[pk-2:pk+3] += np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        signal[:, i] = base
    return signal


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION (AUTO-ADAPTIVE LEAD COUNT)
# ══════════════════════════════════════════════════════════════════════════════

_STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_ecg_signal(signal: np.ndarray, fs: float = 500,
                    n_leads: int | None = None, duration_s: float = 10.0):
    total_leads = signal.shape[1] if signal.ndim > 1 else 1
    n_leads = total_leads if n_leads is None else min(n_leads, total_leads)
    n_samples = min(signal.shape[0], int(fs * duration_s))
    t = np.linspace(0, n_samples / fs, n_samples)

    fig, axes = plt.subplots(n_leads, 1,
                             figsize=(14, 2.0 * n_leads),
                             facecolor="#fefcf0")
    if n_leads == 1:
        axes = [axes]

    for ax, i in zip(axes, range(n_leads)):
        ax.set_facecolor("#fefcf0")
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.grid(which="minor", color="#f4b8b8", linewidth=0.4, alpha=0.7)
        ax.grid(which="major", color="#e88888", linewidth=0.7, alpha=0.5)
        ax.set_xticks(np.arange(0, n_samples / fs + 0.2, 0.2))
        ax.set_yticks(np.arange(-2, 2.5, 0.5))
        sig_slice = signal[:n_samples, i] if signal.ndim > 1 else signal[:n_samples]
        ax.plot(t, sig_slice, color="#1a1a2e", linewidth=0.9)
        ax.set_xlim(0, n_samples / fs)
        ax.set_ylim(-1.8, 1.8)
        lead_label = _STANDARD_LEADS[i] if i < len(_STANDARD_LEADS) else f"L{i+1}"
        ax.set_ylabel(lead_label, fontsize=8, color="#1d4ed8", fontweight="600")
        ax.tick_params(labelsize=7, colors="#888")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. BINARY PRE-FILTER MODEL (config.json + model.h5)
# ══════════════════════════════════════════════════════════════════════════════

def load_binary_model(dir_path) -> keras.Model | None:
    """
    Load binary Normal/Non-normal classifier from config.json + model.h5.
    Returns None if files are missing.
    """
    path = Path(dir_path)
    config_path = path / "config.json"
    weights_path = path / "model.h5"

    if not config_path.exists() or not weights_path.exists():
        log.warning("Binary model files not found in %s (expected config.json + model.h5)", path)
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = f.read()
        # config.json uses only standard Keras layers; passing CUSTOM_OBJECTS is harmless.
        model = keras.models.model_from_json(config_json, custom_objects=CUSTOM_OBJECTS)
        model.load_weights(str(weights_path))
        log.info("Binary pre-filter model loaded from %s", path)
        return model
    except Exception as exc:
        log.error("Failed to load binary model: %s", exc)
        return None


def run_binary_inference(model, signal_arr) -> tuple[int, float]:
    """
    Run binary pre-filter on a pre-processed signal window.

    Parameters
    ----------
    model : keras.Model
        Binary classifier (output sigmoid, shape (batch, 1)).
    signal_arr : np.ndarray
        Shape (1, 1000, 12) as returned by SignalAPIProcessor.

    Returns
    -------
    (prediction, confidence) where prediction is 0 (Normal) or 1 (Non-normal).
    """
    prob = float(model.predict(signal_arr, verbose=0)[0][0])
    prediction = 1 if prob >= 0.5 else 0
    return prediction, prob
