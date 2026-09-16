import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

log = logging.getLogger(__name__)

LEAD_NAMES  = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}


def _get_layer_output_shape(layer) -> tuple | None:
    """
    Robustly get a layer's output shape.
    Tries multiple strategies for Keras 2/3 compatibility.
    Returns a tuple or None if shape cannot be determined.
    """
    # Strategy 1: layer.output.shape  (most reliable for Keras 3 Functional API)
    try:
        out = layer.output
        if isinstance(out, (list, tuple)):
            out = out[0]
        return tuple(out.shape)
    except Exception:
        pass

    # Strategy 2: layer.output_shape  (works in Keras 2 and some Keras 3 models)
    try:
        shape = layer.output_shape
        if isinstance(shape, list):
            shape = shape[0]
        if hasattr(shape, '__len__'):
            return tuple(shape)
    except Exception:
        pass

    # Strategy 3: inspect inbound nodes for shape hints
    try:
        if hasattr(layer, '_inbound_nodes') and layer._inbound_nodes:
            node = layer._inbound_nodes[0]
            if hasattr(node, 'output_shapes') and node.output_shapes:
                s = node.output_shapes
                if isinstance(s, list):
                    s = s[0]
                if hasattr(s, '__len__'):
                    return tuple(s)
    except Exception:
        pass

    return None


def _find_gradcam_layer(model) -> str:
    """
    Find the best layer for Grad-CAM: the last Conv1D layer (or any layer
    with 3D output) before any global pooling or flattening.
    """
    conv3d_candidates = []   # Conv1D / Conv2D layers with 3D output
    other3d_candidates = []  # Any other non-input layer with 3D output

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        shape = _get_layer_output_shape(layer)
        if shape is None or len(shape) != 3:
            continue

        is_conv = isinstance(layer, (keras.layers.Conv1D, keras.layers.Conv2D))
        entry = (layer.name, shape)

        if is_conv:
            conv3d_candidates.append(entry)
        else:
            other3d_candidates.append(entry)

    # Prefer the last Conv1D layer — standard Grad-CAM best practice
    if conv3d_candidates:
        chosen = conv3d_candidates[-1]
        log.info("Selected Grad-CAM layer (Conv): %s with shape %s", chosen[0], chosen[1])
        return chosen[0]

    # Fall back to the last 3D-output layer that isn't an input
    if other3d_candidates:
        chosen = other3d_candidates[-1]
        log.info("Selected Grad-CAM layer (non-Conv): %s with shape %s", chosen[0], chosen[1])
        return chosen[0]

    # Debug: log what we found (or didn't)
    log.error("Grad-CAM: no 3D-output layer found among %d layers.", len(model.layers))
    for i, layer in enumerate(model.layers):
        shape = _get_layer_output_shape(layer)
        log.debug("  L%02d %s shape=%s type=%s", i, layer.name, shape, type(layer).__name__)

    raise ValueError(
        "No suitable layer found for Grad-CAM. "
        "The model may lack convolutional layers with temporal/spatial dimensions."
    )


def get_focus_class(predictions: list) -> int:
    if not predictions:
        return 0
    positives = [(i, prob) for i, (_, prob, is_pos) in enumerate(predictions) if is_pos]
    if positives:
        return max(positives, key=lambda x: x[1])[0]
    return max(range(len(predictions)), key=lambda i: predictions[i][1])


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRAD-CAM (ROBUST MULTI-INPUT & ADAPTIVE LEADS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gradcam(model, metadata_arr, signal_arr, class_index):
    layer_name = _find_gradcam_layer(model)
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(f"Layer {layer_name} not found in model.") from exc

    # Build gradient model — prefer Keras 3 functional construction
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
    except Exception:
        # Fallback: use tf.keras.models.Model for older TF versions
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

    with tf.GradientTape() as tape:
        last_conv_out, preds = grad_model([metadata_arr, signal_arr])
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        class_output = preds[:, class_index]

    grads = tape.gradient(class_output, last_conv_out)
    if grads is None:
        raise RuntimeError(
            "Gradients are None. The target layer may not be connected to the output "
            "via differentiable operations, or the model may be frozen."
        )

    reduce_axes = tuple(range(grads.shape.rank - 1))
    weights = tf.reduce_mean(grads, axis=reduce_axes)

    last_conv_out = tf.squeeze(last_conv_out, axis=0)
    cam = tf.reduce_sum(tf.multiply(weights, last_conv_out), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-10)

    sig_input = np.squeeze(signal_arr, axis=0) if signal_arr.ndim == 3 else signal_arr
    input_length = sig_input.shape[0]
    cam_resized = np.interp(
        np.linspace(0, 1, input_length),
        np.linspace(0, 1, len(cam)),
        cam.numpy(),
    )

    n_leads = sig_input.shape[1] if sig_input.ndim > 1 else 1
    final_heatmaps = []
    for i in range(n_leads):
        lead_sig = sig_input[:, i] if sig_input.ndim > 1 else sig_input
        lead_sig = np.abs(lead_sig)
        lead_sig = (lead_sig - lead_sig.min()) / (lead_sig.max() - lead_sig.min() + 1e-10)
        combined = cam_resized * (lead_sig * 0.5 + 0.5)
        final_heatmaps.append(combined)

    return cam_resized, final_heatmaps


def plot_gradcam(signal_raw, signal_arr, metadata_arr, model, predictions, class_index=None):
    if class_index is None:
        class_index = get_focus_class(predictions)
    class_label = CLASS_NAMES.get(class_index, f"Class {class_index}")
    _, final_heatmaps = compute_gradcam(model, metadata_arr, signal_arr, class_index)

    sig_raw = np.squeeze(signal_raw) if signal_raw.ndim == 3 else signal_raw
    if sig_raw.ndim == 1:
        sig_raw = sig_raw[:, np.newaxis]
    n_leads = sig_raw.shape[1]
    T = sig_raw.shape[0]

    fig, axes = plt.subplots(n_leads, 1, figsize=(18, max(4, 2.0 * n_leads)), sharex=True, facecolor="#fff")
    if n_leads == 1:
        axes = [axes]

    fig.suptitle(f"Grad-CAM — Focus: {class_label}", fontsize=16, fontweight="600", color="#1e3a8a", y=0.995)

    for i in range(n_leads):
        ax = axes[i]
        sig_raw_i = sig_raw[:, i]
        heatmap = final_heatmaps[i] if i < len(final_heatmaps) else np.zeros(T)

        if len(heatmap) != T:
            heatmap = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(heatmap)), heatmap)

        extent = [0, T, sig_raw_i.min(), sig_raw_i.max()]
        ax.plot(sig_raw_i, color="#1a1a2e", linewidth=0.9, zorder=2)
        ax.imshow(heatmap[np.newaxis, :], cmap="jet", aspect="auto", extent=extent, alpha=0.45, zorder=1)
        lead_label = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i+1}"
        ax.set_ylabel(lead_label, rotation=0, labelpad=25, fontweight="bold", fontsize=8, color="#1d4ed8")
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#fff")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. SALIENCY MAPS (ADAPTIVE LEADS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_saliency(model, metadata_arr, signal_arr, class_index) -> np.ndarray:
    signal_tensor = tf.convert_to_tensor(signal_arr, dtype=tf.float32)
    meta_tensor   = tf.convert_to_tensor(metadata_arr, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(signal_tensor)
        preds = model([meta_tensor, signal_tensor])
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        target_logit = preds[:, class_index]

    grads = tape.gradient(target_logit, signal_tensor)
    if grads is None:
        raise RuntimeError("Saliency gradients are None. Check model differentiability.")

    saliency = tf.abs(grads)[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def plot_saliency(signal_raw, signal_arr, metadata_arr, model, predictions, class_index=None):
    if class_index is None:
        class_index = get_focus_class(predictions)

    class_label = CLASS_NAMES.get(class_index, f"Class {class_index}")
    saliency = compute_saliency(model, metadata_arr, signal_arr, class_index)

    sig_raw = np.squeeze(signal_raw) if signal_raw.ndim == 3 else signal_raw
    if sig_raw.ndim == 1:
        sig_raw = sig_raw[:, np.newaxis]
    n_leads = sig_raw.shape[1]
    T = sig_raw.shape[0]

    fig, axes = plt.subplots(n_leads, 1, figsize=(18, max(4, 2.0 * n_leads)), sharex=True, facecolor="#fff")
    if n_leads == 1:
        axes = [axes]

    fig.suptitle(f"Saliency Map — Focus: {class_label}", fontsize=16, fontweight="600", color="#1e3a8a", y=0.995)

    for i in range(n_leads):
        ax = axes[i]
        sig_raw_i = sig_raw[:, i]
        sal = saliency[:, i] if i < saliency.shape[1] else np.zeros(T)

        if len(sal) != T:
            sal = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(sal)), sal)

        ax.plot(sig_raw_i, color="#1a1a2e", linewidth=0.9, zorder=2)
        ax.fill_between(np.arange(T), sig_raw_i.min(), sig_raw_i.max(), where=(sal > 0.3), color="#ef4444", alpha=0.35, zorder=1)
        lead_label = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i+1}"
        ax.set_ylabel(lead_label, rotation=0, labelpad=25, fontweight="bold", fontsize=8, color="#1d4ed8")
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#fff")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTEGRATED GRADIENTS (BATCHED, ROBUST)
# ══════════════════════════════════════════════════════════════════════════════

def compute_integrated_gradients(
    model, metadata_arr, signal_arr,
    class_index: int,
    n_steps: int = 50,
    batch_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    meta_baseline   = np.zeros_like(metadata_arr)
    signal_baseline = np.zeros_like(signal_arr)

    grad_sum_meta   = np.zeros_like(metadata_arr[0], dtype=np.float64)
    grad_sum_signal = np.zeros_like(signal_arr[0],   dtype=np.float64)

    alphas = np.linspace(0.0, 1.0, n_steps + 1)
    n_batches = int(np.ceil(len(alphas) / batch_size))

    for start in range(0, len(alphas), batch_size):
        batch_alphas = alphas[start : start + batch_size]
        
        m_batch = tf.constant(
            np.stack([meta_baseline[0] + a * (metadata_arr[0] - meta_baseline[0]) for a in batch_alphas], axis=0),
            dtype=tf.float32,
        )
        s_batch = tf.constant(
            np.stack([signal_baseline[0] + a * (signal_arr[0] - signal_baseline[0]) for a in batch_alphas], axis=0),
            dtype=tf.float32,
        )

        with tf.GradientTape() as tape:
            tape.watch(m_batch)
            tape.watch(s_batch)
            preds = model([m_batch, s_batch])
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            target = preds[:, class_index]

        grads = tape.gradient(target, [m_batch, s_batch])
        if grads is None or grads[0] is None or grads[1] is None:
            raise RuntimeError("Integrated Gradients returned None gradients.")

        grad_sum_meta   += tf.reduce_mean(grads[0], axis=0).numpy()
        grad_sum_signal += tf.reduce_mean(grads[1], axis=0).numpy()

    ig_meta   = (metadata_arr[0] - meta_baseline[0]) * (grad_sum_meta / n_batches)
    ig_signal = (signal_arr[0]   - signal_baseline[0]) * (grad_sum_signal / n_batches)

    return ig_meta.astype(np.float32), ig_signal.astype(np.float32)


def plot_ig_leads_final(ig_signal: np.ndarray):
    """Plot lead importance bar chart. Handles any lead count and shape orientation."""
    if ig_signal.ndim == 1:
        ig_signal = ig_signal[np.newaxis, :]

    # Determine orientation: if first dim <= 12 and <= second dim, likely (leads, time)
    if ig_signal.shape[0] <= ig_signal.shape[1] and ig_signal.shape[0] <= 12:
        lead_importance = np.mean(np.abs(ig_signal), axis=1)
    else:
        lead_importance = np.mean(np.abs(ig_signal), axis=0)

    n_leads = len(lead_importance)
    lead_names = LEAD_NAMES[:n_leads] if n_leads <= len(LEAD_NAMES) else [f"L{i+1}" for i in range(n_leads)]

    sorted_idx = np.argsort(lead_importance)
    vals = lead_importance[sorted_idx]
    names = [lead_names[i] for i in sorted_idx]
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(vals)))

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.3 * n_leads)), facecolor="#fff")
    ax.barh(names, vals, color=colors, height=0.6)
    ax.set_title("Lead Importance (Integrated Gradients)", fontsize=9, fontweight="600")
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    plt.tight_layout()
    return fig


def plot_ig_metadata_final(ig_meta):
    """Plot metadata feature importance. Adapts to actual feature count."""
    n_features = len(np.atleast_1d(ig_meta))
    feature_names = ["Age", "Sex", "Height", "Weight"][:n_features]
    if n_features > len(feature_names):
        feature_names += [f"Meta {i+1}" for i in range(n_features - len(feature_names))]

    colors = ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"][:n_features]
    fig, ax = plt.subplots(figsize=(5.5, max(2.0, 0.5 * n_features)), facecolor="#fff")
    ax.barh(feature_names, np.atleast_1d(ig_meta), color=colors, height=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.3)
    ax.set_title("Metadata Impact (Integrated Gradients)", fontsize=9, fontweight="600")
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    plt.tight_layout()
    return fig


def plot_ig_signal(signal_raw: np.ndarray, ig_signal: np.ndarray):
    """Plot IG attribution overlaid on raw ECG. Adapts to any lead count."""
    ig_norm = np.abs(ig_signal)
    ig_norm = (ig_norm - ig_norm.min()) / (ig_norm.max() - ig_norm.min() + 1e-10)

    sig_raw = np.squeeze(signal_raw) if signal_raw.ndim == 3 else signal_raw
    if sig_raw.ndim == 1:
        sig_raw = sig_raw[:, np.newaxis]
    n_leads = sig_raw.shape[1]
    T = sig_raw.shape[0]

    fig, axes = plt.subplots(n_leads, 1, figsize=(18, max(4, 2.0 * n_leads)), sharex=True, facecolor="#fff")
    if n_leads == 1:
        axes = [axes]

    fig.suptitle("Integrated Gradients — Signal Attribution", fontsize=16, fontweight="600", color="#1e3a8a", y=0.995)

    for i in range(n_leads):
        ax = axes[i]
        sig_raw_i = sig_raw[:, i]
        ig = ig_norm[:, i] if i < ig_norm.shape[1] else np.zeros(T)

        if len(ig) != T:
            ig = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(ig)), ig)

        ax.plot(sig_raw_i, color="#1a1a2e", linewidth=0.9, zorder=2)
        ax.fill_between(np.arange(T), sig_raw_i.min(), sig_raw_i.max(), where=(ig > 0.2), color="#22c55e", alpha=0.35, zorder=1)
        lead_label = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i+1}"
        ax.set_ylabel(lead_label, rotation=0, labelpad=25, fontweight="bold", fontsize=8, color="#1d4ed8")
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#fff")
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("Time (samples)", fontsize=8, color="#555")
    plt.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_xai(method: str, model, signal_raw, signal_arr, metadata_arr, predictions) -> dict:
    """Run the selected XAI method and return a dict of figures."""
    class_index = get_focus_class(predictions)
    class_label = CLASS_NAMES.get(class_index, f"Class {class_index}")
    figures = {}

    try:
        if method == "Grad-CAM":
            figures[f"Grad-CAM ({class_label})"] = plot_gradcam(
                signal_raw, signal_arr, metadata_arr, model, predictions, class_index
            )
        elif method == "Saliency Maps":
            figures[f"Saliency Map ({class_label})"] = plot_saliency(
                signal_raw, signal_arr, metadata_arr, model, predictions, class_index
            )
        elif method == "Integrated Grads (IG)":
            ig_meta, ig_signal = compute_integrated_gradients(
                model, metadata_arr, signal_arr, class_index
            )
            figures[f"IG — Signal ({class_label})"] = plot_ig_signal(signal_raw, ig_signal)
            figures[f"IG — Metadata ({class_label})"] = plot_ig_metadata_final(ig_meta)
    except Exception as exc:
        log.error("XAI method %s failed: %s", method, exc)
        raise

    return figures
