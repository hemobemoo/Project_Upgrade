import io
import tempfile
import threading
from collections import Counter
from pathlib import PurePath

import numpy as np
import tensorflow as tf
import logging

from Metadata_Processor import MetadataAPIProcessor
from Signal_Processor import SignalAPIProcessor
from settings import (
    MODEL_PATH, BEST_THRESHOLDS, DEMO_MODE,
    OPENAI_API_KEY, FINE_TUNED_MODEL, REPORT_TEMPERATURE,
    BINARY_MODEL_DIR,
)
from ecg_engine import (
    load_ecg_model, load_binary_model, run_binary_inference,
    get_signal_from_files,
    estimate_heart_rate, estimate_rhythm_regularity,
    slice_signal_windows,
)
from gpt_report import generate_ecg_report
from xai_engine import compute_integrated_gradients

log = logging.getLogger(__name__)

# ── Chunked-batch preprocessing dependencies ───────────────────────────────────
try:
    from scipy.signal import butter, filtfilt, resample_poly
except ImportError as _imp_exc:  # pragma: no cover
    butter = filtfilt = resample_poly = None
    log.warning(
        "scipy is not installed — bandpass filtering / resampling in the chunked "
        "batch pipeline will be skipped (raw signal used as-is): %s", _imp_exc
    )

# Model's expected sampling rate for 10-second windows. Override in settings.py
# with a `TARGET_FS = <int>` constant if it differs from this fallback.
try:
    from settings import TARGET_FS
except ImportError:
    TARGET_FS = 100  # NOTE: confirm this matches the rate the model was trained on
    log.warning(
        "settings.TARGET_FS is not defined - falling back to %s Hz. If the model was trained "
        "on a different rate (e.g. 500 Hz) add `TARGET_FS = <int>` to settings.py, otherwise "
        "batch inference will fail with an input-shape mismatch.", TARGET_FS
    )

CHUNK_SEC = 10.0            # model was trained on fixed 10-second windows
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0

# ── Thread-safe model singleton ────────────────────────────────────────────────
_MODEL = None
_MODEL_LOCK = threading.Lock()

# ── Binary pre-filter model singleton ──────────────────────────────────────────
_BINARY_MODEL = None
_BINARY_MODEL_LOCK = threading.Lock()

def _get_binary_model():
    global _BINARY_MODEL
    with _BINARY_MODEL_LOCK:
        if _BINARY_MODEL is None:
            _BINARY_MODEL = load_binary_model(BINARY_MODEL_DIR)
    return _BINARY_MODEL

def get_binary_model():
    return _get_binary_model()

SIG_PROC = SignalAPIProcessor()
META_PROC = MetadataAPIProcessor()

def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = load_ecg_model(MODEL_PATH)
    return _MODEL

def get_model():
    return _get_model()


LABELS_LIST = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def run_inference(model, metadata_arr, signal_arr):
    """
    Run model inference and return predictions, best_idx, probabilities.
    If model is None, returns realistic mock data.
    """
    thresholds_dict = dict(zip(LABELS_LIST, BEST_THRESHOLDS))

    if model is None:
        probabilities = np.array([0.12, 0.84, 0.09, 0.41, 0.07])
    else:
        logits = model.predict([metadata_arr, signal_arr], verbose=0)
        probabilities = tf.nn.sigmoid(logits).numpy()[0]

    detected_indices_raw = [i for i, p in enumerate(probabilities) if p >= thresholds_dict[LABELS_LIST[i]]]
    raw_detected_labels = [LABELS_LIST[i] for i in detected_indices_raw]
    has_pathology = any(lbl != 'NORM' for lbl in raw_detected_labels)

    final_detected_indices = []
    if has_pathology and 'NORM' in raw_detected_labels:
        final_detected_indices = [i for i in detected_indices_raw if LABELS_LIST[i] != 'NORM']
    else:
        final_detected_indices = detected_indices_raw

    predictions = []
    for i, label in enumerate(LABELS_LIST):
        prob = float(probabilities[i])
        is_positive = i in final_detected_indices
        predictions.append((label, prob, is_positive))

    if not final_detected_indices:
        best_idx = int(np.argmax(probabilities))
    else:
        det_probs = [probabilities[i] for i in final_detected_indices]
        best_idx = final_detected_indices[np.argmax(det_probs)]

    return predictions, best_idx, probabilities



# ── Chunked-batch: input parsing (.npy / .mat / .csv / WFDB) ───────────────────

N_LEADS = 12
SUPPORTED_ARRAY_EXTS = (".npy", ".mat", ".csv")


def _to_samples_by_leads(arr: np.ndarray, source: str, n_leads: int = N_LEADS) -> np.ndarray:
    """
    Normalise any parsed array to float64 (samples, n_leads).
    Accepts (samples, 12), (12, samples) and singleton-batch shapes such as
    (1, samples, 12). Raises a descriptive ValueError otherwise, so the UI can
    show exactly what was wrong with the file.
    """
    arr = np.asarray(arr)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{source}: array is not numeric (dtype={arr.dtype}).")

    arr = np.squeeze(arr) if arr.ndim > 2 else arr
    if arr.ndim != 2:
        raise ValueError(f"{source}: expected a 2-D (samples x {n_leads}) array, got shape {arr.shape}.")

    if arr.shape[1] == n_leads:
        pass
    elif arr.shape[0] == n_leads:
        arr = arr.T
    else:
        raise ValueError(f"{source}: expected {n_leads} leads, got array of shape {arr.shape}.")

    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] < 2:
        raise ValueError(f"{source}: signal has no samples.")
    return np.ascontiguousarray(arr)


def _parse_npy(data: bytes, name: str, fs_hint):
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:                       # corrupt file, wrong extension, pickled object arrays...
        raise ValueError(f"{name}: not a valid numeric .npy file ({type(exc).__name__}).") from exc
    return _to_samples_by_leads(arr, name), fs_hint


def _parse_mat(data: bytes, name: str, fs_hint):
    from scipy.io import loadmat
    mat = loadmat(io.BytesIO(data), squeeze_me=True)

    fs = fs_hint
    for key in ("fs", "Fs", "FS", "sampling_rate", "sampling_frequency", "sfreq", "freq"):
        if key in mat:
            try:
                candidate = float(np.asarray(mat[key]).reshape(-1)[0])
                if candidate > 0:
                    fs = candidate
                    break
            except (TypeError, ValueError):
                pass

    candidates = []
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(val)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number) and N_LEADS in arr.shape:
            candidates.append((arr.size, key, arr))
    if not candidates:
        raise ValueError(f"{name}: no numeric 2-D array with {N_LEADS} leads found in the .mat file "
                         f"(variables: {[k for k in mat if not k.startswith('__')]}).")
    candidates.sort(key=lambda t: t[0], reverse=True)   # biggest array = the signal
    _, key, arr = candidates[0]
    return _to_samples_by_leads(arr, f"{name}[{key}]"), fs


def _parse_csv(data: bytes, name: str, fs_hint):
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data), header=None, sep=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")   # drops a text header row / empty cols
    arr = df.to_numpy(dtype=np.float64)
    # Optional leading time column: N_LEADS + 1 columns whose first one is strictly increasing.
    if arr.ndim == 2 and arr.shape[1] == N_LEADS + 1 and np.all(np.diff(arr[:, 0]) > 0):
        arr = arr[:, 1:]
    return _to_samples_by_leads(arr, name), fs_hint


def load_signal_from_bytes(file_bytes: bytes, file_name: str, fs: float | None = None):
    """
    Parse ONE raw upload (.npy / .mat / .csv) straight from memory - exactly what
    NiceGUI hands over in the upload event. Returns (signal (samples, 12), fs).

    .npy and .csv carry no sampling rate, so `fs` must be supplied by the caller;
    .mat files may embed one (keys such as 'fs'), which then takes precedence.
    """
    ext = PurePath(file_name).suffix.lower()
    parsers = {".npy": _parse_npy, ".mat": _parse_mat, ".csv": _parse_csv}
    if ext not in parsers:
        raise ValueError(f"{file_name}: unsupported file type '{ext}' "
                         f"(supported: {', '.join(SUPPORTED_ARRAY_EXTS)}, or WFDB .hea + .dat).")
    if not file_bytes:
        raise ValueError(f"{file_name}: file is empty.")

    signal, fs_out = parsers[ext](file_bytes, file_name, fs)
    if fs_out is None or float(fs_out) <= 0:
        raise ValueError(f"{file_name}: sampling rate is required for {ext} files.")
    return signal, float(fs_out)


# ── Chunked-batch: preprocessing helpers ───────────────────────────────────────

def _bandpass_filter(signal_arr: np.ndarray, fs: float, low: float = BANDPASS_LOW_HZ,
                      high: float = BANDPASS_HIGH_HZ, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass (0.5-45 Hz default), applied per lead.
    signal_arr: (samples,) or (samples, leads). No-op if scipy is unavailable
    or the requested band is invalid for the given fs.
    """
    if butter is None or filtfilt is None:
        return signal_arr

    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.999)
    if low_n <= 0 or low_n >= high_n:
        log.warning("Skipping bandpass filter: invalid band for fs=%s", fs)
        return signal_arr

    b, a = butter(order, [low_n, high_n], btype="band")
    if signal_arr.ndim == 1:
        return filtfilt(b, a, signal_arr)
    return np.stack([filtfilt(b, a, signal_arr[:, ch]) for ch in range(signal_arr.shape[1])], axis=1)


def _resample_signal(signal_arr: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Polyphase resample from orig_fs to target_fs. No-op if rates already match."""
    if resample_poly is None or orig_fs == target_fs:
        return signal_arr

    from math import gcd
    g = gcd(int(orig_fs), int(target_fs))
    up, down = int(target_fs) // g, int(orig_fs) // g

    if signal_arr.ndim == 1:
        return resample_poly(signal_arr, up, down)
    return np.stack([resample_poly(signal_arr[:, ch], up, down) for ch in range(signal_arr.shape[1])], axis=1)


def preprocess_and_chunk(signal_raw_data: np.ndarray, fs: float,
                          target_fs: float = TARGET_FS, chunk_sec: float = CHUNK_SEC):
    """
    Denoise (bandpass), resample to the model's target rate, and slice a
    (possibly long, possibly short) raw ECG signal into consecutive fixed-length
    chunks of `chunk_sec` seconds each.

    Returns:
        chunks:        list of np.ndarray, each (chunk_samples, leads)
        time_windows:  list of (start_sec, end_sec) tuples, in the ORIGINAL signal's timeline
        used_fs:       the sampling rate the chunks are expressed in (== target_fs)

    Signals shorter than `chunk_sec` are zero-padded to a single full-length
    chunk rather than being dropped, so the model always sees its expected
    input shape; the reported time window reflects the real (unpadded) duration.
    """
    filtered = _bandpass_filter(signal_raw_data, fs)
    resampled = _resample_signal(filtered, fs, target_fs)

    chunk_samples = int(round(target_fs * chunk_sec))
    n_samples = resampled.shape[0]

    if n_samples <= 0:
        raise ValueError("Signal is empty after preprocessing.")

    if n_samples < chunk_samples:
        pad_width = chunk_samples - n_samples
        pad_spec = (0, pad_width) if resampled.ndim == 1 else ((0, pad_width), (0, 0))
        padded = np.pad(resampled, pad_spec)
        return [padded], [(0.0, n_samples / target_fs)], target_fs

    # Reuse the existing (non-overlapping) windowing utility used by the
    # streaming pipeline — keeps chunk-slicing behavior consistent across modes.
    windows = slice_signal_windows(resampled, target_fs, window_sec=chunk_sec, overlap_sec=0.0)

    chunks, time_windows = [], []
    for chunk_signal, start, end in windows:
        if chunk_signal.shape[0] < chunk_samples:
            pad_width = chunk_samples - chunk_signal.shape[0]
            pad_spec = (0, pad_width) if chunk_signal.ndim == 1 else ((0, pad_width), (0, 0))
            chunk_signal = np.pad(chunk_signal, pad_spec)
        chunks.append(chunk_signal)
        time_windows.append((start / target_fs, end / target_fs))

    return chunks, time_windows, target_fs


# ── Chunked-batch: vectorized batch inference (no LLM calls) ──────────────────

def _check_model_input_shape(model, signal_batch: np.ndarray) -> None:
    """
    Fail early with an actionable message if the chunk tensor does not match the
    model's signal input (typically a TARGET_FS / chunk-length mismatch), instead
    of surfacing a cryptic Keras error deep inside predict().
    """
    if model is None:
        return
    try:
        expected = model.input_shape[1][1:]          # inputs are [metadata, signal]
        actual = tuple(signal_batch.shape[1:])
    except Exception:
        return                                       # subclassed / unknown model: let Keras decide
    if len(expected) == len(actual) and any(
        e is not None and e != a for e, a in zip(expected, actual)
    ):
        raise ValueError(
            f"Chunk tensor shape {actual} does not match the model's signal input {tuple(expected)}. "
            f"Check TARGET_FS (currently {TARGET_FS} Hz) and CHUNK_SEC ({CHUNK_SEC}s) against the "
            f"sampling rate / window length the model was trained on."
        )


def run_chunked_inference(model, metadata_arr: np.ndarray, chunks: list):
    """
    ONE vectorized batch forward-pass across all chunks of a single record.
    Signal-processing + model inference ONLY — no LLM report generation here,
    by design (see `generate_single_report` for the on-demand report step).

    metadata_arr: (1, n_meta_features) for the record (repeated per chunk).
    Returns: (per_chunk_predictions, probabilities_matrix (n_chunks, n_labels))
    """
    n_chunks = len(chunks)
    thresholds_dict = dict(zip(LABELS_LIST, BEST_THRESHOLDS))

    # SIG_PROC.process_for_api returns a batch-of-1 array per call (see its use
    # in run_full_pipeline / run_streaming_pipeline) — concatenate to build the
    # (Num_Chunks, samples, leads) batch for a single model.predict() call.
    signal_batch = np.concatenate([SIG_PROC.process_for_api(c) for c in chunks], axis=0)
    metadata_batch = np.repeat(metadata_arr, n_chunks, axis=0)
    _check_model_input_shape(model, signal_batch)

    if model is None:
        rng = np.random.default_rng(0)
        probabilities_matrix = rng.uniform(0.05, 0.5, size=(n_chunks, len(LABELS_LIST)))
    else:
        logits = model.predict([metadata_batch, signal_batch], verbose=0)
        probabilities_matrix = tf.nn.sigmoid(logits).numpy()

    per_chunk_predictions = []
    for row in probabilities_matrix:
        detected = [i for i, p in enumerate(row) if p >= thresholds_dict[LABELS_LIST[i]]]
        raw_labels = [LABELS_LIST[i] for i in detected]
        has_pathology = any(lbl != 'NORM' for lbl in raw_labels)
        final = ([i for i in detected if LABELS_LIST[i] != 'NORM']
                 if (has_pathology and 'NORM' in raw_labels) else detected)
        preds = [(LABELS_LIST[i], float(row[i]), i in final) for i in range(len(LABELS_LIST))]
        per_chunk_predictions.append(preds)

    return per_chunk_predictions, probabilities_matrix


def _generate_fallback_report(predictions, age, sex, hr, heart_rhythm, rhythm_reg):
    """Generate a basic markdown report when LLM is unavailable."""
    active = [name for name, prob, is_pos in predictions if is_pos]
    if not active:
        active = [max(predictions, key=lambda x: x[1])[0]]

    report = f"""## Clinical Interpretation Report

**Patient:** {age}-year-old {sex}
**Vitals:** Heart Rate {hr} bpm | Rhythm: {heart_rhythm} | Regularity: {rhythm_reg}

### Diagnostic Summary
The ECG analysis indicates the following condition(s): **{', '.join(active)}**.

### Clinical Correlation
Based on the automated pattern recognition, the model has identified features consistent with {active[0]}. 
Further clinical evaluation and correlation with patient symptoms and history are recommended.

### Recommendations
- Consult cardiology if this is an acute presentation.
- Compare with prior ECGs if available.
- Consider additional cardiac biomarkers (troponin, BNP) as clinically indicated.

*This report was generated by CardioInsight AI (Demo Mode).*
"""
    return report


# ── Full pipeline (single record) ─────────────────────────────────────────────

def run_full_pipeline(
    hea_bytes: bytes, hea_name: str,
    dat_bytes: bytes, dat_name: str,
    age: int,
    sex: str,
    height: float | None = None,
    weight: float | None = None,
) -> dict:

    with tempfile.TemporaryDirectory() as tmp_dir:
        signal_raw_data, fs = get_signal_from_files(hea_bytes, hea_name, dat_bytes, dat_name, tmp_dir)
        signal_arr = SIG_PROC.process_for_api(signal_raw_data)

    sex_num = 0 if str(sex).lower() in ["male", "0", "0.0"] else 1
    metadata_arr = META_PROC.process_metadata_for_api(age, sex_num, height, weight)

    model = _get_model()
    predictions, best_idx, probabilities = run_inference(model, metadata_arr, signal_arr)

    hr = estimate_heart_rate(signal_raw_data, fs)
    heart_rhythm, rhythm_reg = estimate_rhythm_regularity(signal_raw_data, fs)

    # Report generation with fallback
    try:
        report = generate_ecg_report(
            predictions=predictions,
            api_key=OPENAI_API_KEY,
            model_id=FINE_TUNED_MODEL,
            temperature=REPORT_TEMPERATURE,
            meta={"age": age, "sex": sex, "hr": hr, "heart_rhythm": heart_rhythm, "rhythm_regularity": rhythm_reg},
        )
    except Exception as exc:
        log.warning("LLM report generation failed: %s. Using fallback.", exc)
        report = _generate_fallback_report(predictions, age, sex, hr, heart_rhythm, rhythm_reg)

    # XAI - Integrated Gradients for importance
    ig_signal = np.zeros_like(signal_arr[0])
    ig_meta = np.zeros(metadata_arr.shape[1])

    try:
        if model is not None:
            ig_meta, ig_signal = compute_integrated_gradients(
                model, metadata_arr, signal_arr, class_index=best_idx, n_steps=50
            )
            lead_importance = np.mean(np.abs(ig_signal), axis=0).tolist()
            meta_importance = ig_meta.tolist()
        else:
            raise RuntimeError("Model not available: skipping IG computation.")
    except Exception as e:
        log.warning("Error computing IG for importance: %s", e)
        lead_importance = [0.0] * signal_arr.shape[-1]
        meta_importance = [0.0] * len(metadata_arr[0])

    return {
        "predictions": predictions,
        "report": report,
        "signal": signal_raw_data,
        "signal_arr": signal_arr,
        "metadata_arr": metadata_arr,
        "fs": fs,
        "ig_signal": ig_signal,
        "meta": {
            "age": age,
            "sex": sex,
            "height": height,
            "weight": weight,
            "hr": hr,
            "heart_rhythm": heart_rhythm,
            "rhythm_regularity": rhythm_reg,
            "best_idx": int(best_idx),
            "lead_importance": lead_importance,
            "meta_importance": meta_importance,
        },
    }


# ── Batch pipeline ────────────────────────────────────────────────────────────

def run_batch_pipeline(records: list[dict]) -> list[dict]:
    """
    Process multiple ECG records.
    records: list of dicts with keys: hea_bytes, hea_name, dat_bytes, dat_name, age, sex, height, weight
    Returns: list of result dicts
    """
    results = []
    for rec in records:
        try:
            result = run_full_pipeline(**rec)
            result["status"] = "success"
            result["filename"] = rec["hea_name"]
        except Exception as exc:
            result = {
                "status": "error",
                "filename": rec["hea_name"],
                "error": str(exc),
                "predictions": [("NORM", 0.12, False), ("MI", 0.84, True), ("STTC", 0.09, False), ("CD", 0.41, False), ("HYP", 0.07, False)],
                "report": f"Error processing {rec['hea_name']}: {exc}",
            }
        results.append(result)
    return results


# ── Chunked batch pipeline (long/variable-length recordings) ──────────────────
#
# Token-cost optimization: this pipeline performs signal processing + model
# inference for every record/chunk automatically, but deliberately does NOT
# call the LLM here. Reports are generated on demand via `generate_single_report`
# — call it only when the user picks a specific record or segment in the UI.

def run_chunked_record(
    hea_bytes: bytes | None = None, hea_name: str | None = None,
    dat_bytes: bytes | None = None, dat_name: str | None = None,
    age: int = 50,
    sex: str = "Male",
    height: float | None = None,
    weight: float | None = None,
    target_fs: float = TARGET_FS,
    chunk_sec: float = CHUNK_SEC,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
    fs: float | None = None,
) -> dict:
    """
    Process ONE (possibly long) ECG record by:
      1. Denoising (0.5-45 Hz bandpass) and resampling to `target_fs`.
      2. Slicing into consecutive `chunk_sec`-second windows (0-10s, 10-20s, ...).
      3. Running a single vectorized batch forward-pass across all chunks.

    Input is EITHER a WFDB pair (`hea_bytes`/`hea_name` + `dat_bytes`/`dat_name`)
    OR a single raw-array upload (`file_bytes`/`file_name` ending in .npy, .mat
    or .csv, plus its sampling rate `fs` unless the .mat embeds one).

    Does NOT generate an LLM report — `report` is returned as None on both the
    record and every segment. Call `generate_single_report(...)` separately,
    on demand, for a chosen record or segment.
    """
    if file_bytes is not None:
        display_name = file_name or "uploaded_signal"
        signal_raw_data, fs = load_signal_from_bytes(file_bytes, display_name, fs)
    elif hea_bytes is not None and dat_bytes is not None:
        display_name = hea_name or "record.hea"
        with tempfile.TemporaryDirectory() as tmp_dir:
            signal_raw_data, fs = get_signal_from_files(hea_bytes, hea_name, dat_bytes, dat_name, tmp_dir)
    else:
        raise ValueError("Record has neither a WFDB (.hea + .dat) pair nor a .npy/.mat/.csv file.")

    chunks, time_windows, used_fs = preprocess_and_chunk(
        signal_raw_data, fs, target_fs=target_fs, chunk_sec=chunk_sec
    )

    sex_num = 0 if str(sex).lower() in ["male", "0", "0.0"] else 1
    metadata_arr = META_PROC.process_metadata_for_api(age, sex_num, height, weight)

    model = _get_model()
    per_chunk_predictions, probabilities_matrix = run_chunked_inference(model, metadata_arr, chunks)

    segments = []
    for i, (chunk_signal, (start_s, end_s), preds) in enumerate(zip(chunks, time_windows, per_chunk_predictions)):
        hr = estimate_heart_rate(chunk_signal, used_fs)
        heart_rhythm, rhythm_reg = estimate_rhythm_regularity(chunk_signal, used_fs)
        is_abnormal = any(is_pos and label != 'NORM' for label, _, is_pos in preds)
        segments.append({
            "chunk_index": i,
            "time_window": f"{start_s:.0f}s-{end_s:.0f}s",
            "predictions": preds,
            "is_abnormal": is_abnormal,
            "heart_rate": hr,
            "heart_rhythm": heart_rhythm,
            "rhythm_regularity": rhythm_reg,
            "report": None,   # populated on demand via generate_single_report
        })

    # Record-level aggregation: max probability per label across all chunks, so
    # a single abnormal segment still surfaces the finding at the record level.
    agg_probabilities = probabilities_matrix.max(axis=0)
    thresholds_dict = dict(zip(LABELS_LIST, BEST_THRESHOLDS))
    agg_detected = [i for i, p in enumerate(agg_probabilities) if p >= thresholds_dict[LABELS_LIST[i]]]
    agg_raw_labels = [LABELS_LIST[i] for i in agg_detected]
    agg_has_pathology = any(lbl != 'NORM' for lbl in agg_raw_labels)
    agg_final = ([i for i in agg_detected if LABELS_LIST[i] != 'NORM']
                 if (agg_has_pathology and 'NORM' in agg_raw_labels) else agg_detected)
    record_predictions = [(LABELS_LIST[i], float(agg_probabilities[i]), i in agg_final)
                           for i in range(len(LABELS_LIST))]

    valid_hrs = [s["heart_rate"] for s in segments if s["heart_rate"]]
    overall_hr = float(np.mean(valid_hrs)) if valid_hrs else None
    n_abnormal_chunks = sum(1 for s in segments if s["is_abnormal"])

    # Most frequent rhythm descriptors across chunks (used as context for a record-level report).
    common_rhythm = Counter(s["heart_rhythm"] for s in segments).most_common(1)
    common_reg = Counter(s["rhythm_regularity"] for s in segments).most_common(1)

    return {
        "filename": display_name,
        "status": "success",
        "predictions": record_predictions,   # record-level (max-across-chunks), for summary display
        "report": None,                      # NOT generated automatically — see generate_single_report
        "n_chunks": len(segments),
        "n_abnormal_chunks": n_abnormal_chunks,
        "segments": segments,                # per-10s-chunk: chunk_index, time_window, predictions, is_abnormal, heart_rate, report
        "meta": {
            "age": age, "sex": sex, "height": height, "weight": weight,
            "hr": overall_hr,
            "heart_rhythm": common_rhythm[0][0] if common_rhythm else "—",
            "rhythm_regularity": common_reg[0][0] if common_reg else "—",
            "duration_s": float(signal_raw_data.shape[0] / fs),
            "fs_used": used_fs,
            "original_fs": fs,
        },
    }


def run_chunked_batch_pipeline(
    records: list[dict],
    chunk_sec: float = CHUNK_SEC,
    target_fs: float = TARGET_FS,
) -> list[dict]:
    """
    Batch entry point for long/variable-length ECG recordings. Each record is
    denoised, resampled, sliced into consecutive `chunk_sec`-second chunks, and
    run through ONE vectorized batch forward-pass per record — signal
    processing + model inference only, no LLM calls (see module docstring above
    `run_chunked_record`).

    records: list of dicts. Each record carries patient fields (age, sex, height,
    weight) plus EITHER a WFDB pair (hea_bytes, hea_name, dat_bytes, dat_name)
    OR a raw-array file (file_bytes, file_name, fs) for .npy / .mat / .csv.

    Returns: one result dict per input record, in order. Failures never raise —
    they come back as {"status": "error", "error": "...", ...} with the same keys
    the UI reads, so a single bad file cannot blank the whole batch.
    """
    results = []
    for rec in records:
        name = rec.get("file_name") or rec.get("hea_name") or "unknown"
        try:
            result = run_chunked_record(
                hea_bytes=rec.get("hea_bytes"), hea_name=rec.get("hea_name"),
                dat_bytes=rec.get("dat_bytes"), dat_name=rec.get("dat_name"),
                file_bytes=rec.get("file_bytes"), file_name=rec.get("file_name"),
                fs=rec.get("fs"),
                age=rec["age"], sex=rec["sex"],
                height=rec.get("height"), weight=rec.get("weight"),
                target_fs=target_fs, chunk_sec=chunk_sec,
            )
        except Exception as exc:
            if isinstance(exc, ValueError):
                log.warning("Skipping %s: %s", name, exc)          # bad/incompatible user data - message goes to the UI
            else:
                log.exception("Chunked batch processing failed for %s", name)   # unexpected: keep the traceback
            result = {
                "status": "error",
                "filename": name,
                "error": f"{type(exc).__name__}: {exc}",
                "predictions": [],
                "segments": [],
                "n_chunks": 0,
                "n_abnormal_chunks": 0,
                "meta": {"hr": None, "age": rec.get("age"), "sex": rec.get("sex")},
                "report": None,
            }
        results.append(result)
    return results


# ── On-demand LLM report generation (token-cost optimization) ─────────────────

def generate_single_report(
    predictions: list,
    age: int,
    sex: str,
    hr: float | None,
    heart_rhythm: str = "—",
    rhythm_regularity: str = "—",
) -> str:
    """
    Generate ONE clinical narrative report on demand for a chosen record or a
    chosen abnormal segment. Intentionally NOT called automatically anywhere
    in the chunked batch pipeline — wire this to a UI button so LLM tokens are
    only spent when a user actually asks to see a report.
    """
    meta = {"age": age, "sex": sex, "hr": hr, "heart_rhythm": heart_rhythm, "rhythm_regularity": rhythm_regularity}
    try:
        return generate_ecg_report(
            predictions=predictions,
            api_key=OPENAI_API_KEY,
            model_id=FINE_TUNED_MODEL,
            temperature=REPORT_TEMPERATURE,
            meta=meta,
        )
    except Exception as exc:
        log.warning("On-demand LLM report generation failed: %s. Using fallback.", exc)
        return _generate_fallback_report(predictions, age, sex, hr, heart_rhythm, rhythm_regularity)


# ── Streaming pipeline ────────────────────────────────────────────────────────

def run_streaming_pipeline(
    continuous_signal: np.ndarray,
    fs: float,
    age: int,
    sex: str,
    height: float | None = None,
    weight: float | None = None,
    window_sec: float = 10.0,
    overlap_sec: float = 5.0,
):
    """
    Real-time streaming pipeline:
      1. Slice continuous signal into overlapping windows.
      2. Run lightweight binary pre-filter (signal only).
      3. If abnormal (1), run full multi-label model with metadata for detailed diagnosis.

    Yields a dict per window for real-time display.
    """
    sex_num = 0 if str(sex).lower() in ["male", "0", "0.0"] else 1
    metadata_arr = META_PROC.process_metadata_for_api(age, sex_num, height, weight)

    multi_model = _get_model()
    binary_model = _get_binary_model()

    windows = slice_signal_windows(continuous_signal, fs, window_sec, overlap_sec)

    for win_idx, (window_signal, start, end) in enumerate(windows):
        signal_arr = SIG_PROC.process_for_api(window_signal)

        # Step 1: Binary pre-filter (fast, no metadata required)
        if binary_model is not None:
            binary_pred, binary_conf = run_binary_inference(binary_model, signal_arr)
        else:
            # If binary model is missing, force detailed analysis so the demo still works
            binary_pred = 1
            binary_conf = 1.0

        # Step 2: Detailed multi-label analysis only if binary flags abnormal
        if binary_pred == 1:
            predictions, best_idx, probabilities = run_inference(multi_model, metadata_arr, signal_arr)
        else:
            # Normal window — skip heavy model, return clean NORM
            predictions = [
                ("NORM", float(binary_conf), True),
                ("MI", 0.05, False),
                ("STTC", 0.04, False),
                ("CD", 0.03, False),
                ("HYP", 0.02, False),
            ]
            best_idx = 0

        hr = estimate_heart_rate(window_signal, fs)
        heart_rhythm, rhythm_reg = estimate_rhythm_regularity(window_signal, fs)

        yield {
            "window_idx": win_idx,
            "start_sample": start,
            "end_sample": end,
            "start_sec": start / fs,
            "end_sec": end / fs,
            "fs": fs,
            "window_signal": window_signal,
            "binary_pred": binary_pred,
            "binary_conf": binary_conf,
            "predictions": predictions,
            "best_idx": best_idx,
            "hr": hr,
            "heart_rhythm": heart_rhythm,
            "rhythm_regularity": rhythm_reg,
            "anomaly_flag": binary_pred == 1,
        }


# ── Full-result analysis for a flagged streaming window ───────────────────────

def run_streaming_window_analysis(
    window_signal: np.ndarray,
    fs: float,
    age: int,
    sex: str,
    height: float | None = None,
    weight: float | None = None,
) -> dict:
    """
    Run the same full-result pipeline used for Manual Upload (multi-label
    inference + LLM report + Integrated-Gradients importance) on a single
    ~10s window flagged abnormal by the streaming binary pre-filter.

    Returns a dict with the exact same shape as run_full_pipeline's return
    value, so the caller can reuse the Manual Upload results renderer as-is.
    """
    signal_arr = SIG_PROC.process_for_api(window_signal)

    sex_num = 0 if str(sex).lower() in ["male", "0", "0.0"] else 1
    metadata_arr = META_PROC.process_metadata_for_api(age, sex_num, height, weight)

    model = _get_model()
    predictions, best_idx, probabilities = run_inference(model, metadata_arr, signal_arr)

    hr = estimate_heart_rate(window_signal, fs)
    heart_rhythm, rhythm_reg = estimate_rhythm_regularity(window_signal, fs)

    try:
        report = generate_ecg_report(
            predictions=predictions,
            api_key=OPENAI_API_KEY,
            model_id=FINE_TUNED_MODEL,
            temperature=REPORT_TEMPERATURE,
            meta={"age": age, "sex": sex, "hr": hr, "heart_rhythm": heart_rhythm, "rhythm_regularity": rhythm_reg},
        )
    except Exception as exc:
        log.warning("LLM report generation failed: %s. Using fallback.", exc)
        report = _generate_fallback_report(predictions, age, sex, hr, heart_rhythm, rhythm_reg)

    ig_signal = np.zeros_like(signal_arr[0])
    ig_meta = np.zeros(metadata_arr.shape[1])

    try:
        if model is not None:
            ig_meta, ig_signal = compute_integrated_gradients(
                model, metadata_arr, signal_arr, class_index=best_idx, n_steps=50
            )
            lead_importance = np.mean(np.abs(ig_signal), axis=0).tolist()
            meta_importance = ig_meta.tolist()
        else:
            raise RuntimeError("Model not available: skipping IG computation.")
    except Exception as e:
        log.warning("Error computing IG for importance: %s", e)
        lead_importance = [0.0] * signal_arr.shape[-1]
        meta_importance = [0.0] * len(metadata_arr[0])

    return {
        "predictions": predictions,
        "report": report,
        "signal": window_signal,
        "signal_arr": signal_arr,
        "metadata_arr": metadata_arr,
        "fs": fs,
        "ig_signal": ig_signal,
        "meta": {
            "age": age,
            "sex": sex,
            "height": height,
            "weight": weight,
            "hr": hr,
            "heart_rhythm": heart_rhythm,
            "rhythm_regularity": rhythm_reg,
            "best_idx": int(best_idx),
            "lead_importance": lead_importance,
            "meta_importance": meta_importance,
        },
    }
