# CardioInsight — project memory
> Created after the Live ECG Signal Simulator task (NiceGUI Live Monitor). Dense; re-read before resuming.

## 1. Project snapshot
- **What**: 12-lead ECG classifier (5 labels) + Explainable AI (Grad-CAM / Saliency / Integrated Gradients) + LLM clinical report generation.
- **UI**: NiceGUI (migrated from Streamlit; Streamlit files are reference-only, untouched). Single-page app, `@ui.page("/")` in `cardioinsight_nicegui.py`.
- **Stack**: Python, TensorFlow/Keras (model), wfdb (signal I/O), pandas/matplotlib, OpenAI-style LLM API (tenacity retry).
- **Models**: CNN-SE-Transformer hybrid `utils/best_hybrid_v4_transformer.keras` (input: 1000-sample × 12-lead + 3 metadata features); lightweight binary pre-filter in `utils/`.
- **Labels**: `LABELS_LIST = ['NORM','MI','STTC','CD','HYP']` (api_bassam.py:57); multi-label via per-class thresholds `BEST_THRESHOLDS` (settings.py).
- **Default rates/geometry**: 100 Hz nominal; windows = 10 s = 1000 samples; overlap 5 s (ecg_engine.py:247).

## 2. Run & validate
- Run: `python cardioinsight_nicegui.py` → `ui.run(title="CardioInsight 🫀", host="0.0.0.0", port=8501, reload=False, show=False)`. Open http://localhost:8501. Venv `.venv` already active.
- Syntax check: `python -m py_compile cardioinsight_nicegui.py api_bassam.py ecg_engine.py xai_engine.py gpt_report.py settings.py`
- Lint check: `python -m pyflakes <files>` — project standard is zero warnings.
- Server smoke test: start app → GET http://localhost:8501 → expect 200, no server-side tracebacks.
- E2E pattern (used 2026-09-19): temp script → `load_sample_signal` → `build_demo_stream(repeats=4)` → `run_streaming_pipeline` (via `run.io_bound`) → find abnormal window → `run_streaming_window_analysis` → assert predictions/report/IG present; **delete temp script after**.
- `app_nicegui.py` is a legacy alternate NiceGUI entry; main canonical app is `cardioinsight_nicegui.py`. Confirm before editing either.

## 3. File map & edit policy
| File | Role | Policy |
|---|---|---|
| `cardioinsight_nicegui.py` | Main NiceGUI app (~770 lines). All Live Monitor code. | ✅ Edit (Live scope) |
| `api_bassam.py` | Inference pipelines (binary, full, batch, streaming). | ✅ Edit streaming helpers only |
| `ecg_engine.py` | Model arch + loaders + HR/rhythm estimators + plotting. | ✅ Edit streaming helpers only |
| `xai_engine.py` | Grad-CAM/Saliency/IG + `run_xai`. | ⚠️ Touch only for Live integration |
| `gpt_report.py` | `generate_ecg_report` (LLM + fallback). | ⚠️ Touch only if report contract changes |
| `settings.py` | API keys, model paths, `BEST_THRESHOLDS`. | 🚫 DO NOT MODIFY values |
| `Signal_Processor.py` | `SignalAPIProcessor(fs=100, lowcut=0.5, highcut=45, order=4, target_len=1000)` — Butter bandpass filtfilt + resize to 1000. | 🚫 Do not touch |
| `Metadata_Processor.py` | `MetadataAPIProcessor` — scaler_mean=[59.63,167.55,71.74], scaler_std=[16.91,7.84,11.60]; 3 features (age, height, weight). | 🚫 Do not touch |
| `live_monitor.py` | Old Streamlit live monitor (synthetic ECG generators). | 🚫 Reference only; do not use/restyle |
| `app_nicegui.py` | Legacy alternate NiceGUI entry. | ⚠️ Avoid unless confirmed canonical |
| `Testing Samples/` | Bundled `.hea`/`.dat` records (default `"13003_hr"`). | ✅ Source data |
| `utils/` | Model weights, config.json, binary model. | 🚫 Do not touch |
| Streamlit files (parent `Project_V_Kimi/`) | Original app + references. | 🚫 Do not touch |

## 4. Pipeline architecture (data flow)
Shared chain (api_bassam.py / ecg_engine.py):
`raw (T,12) → SignalAPIProcessor.process_for_api → bandpass(0.5–45 Hz) → resize (1000,12) → model input`
`metadata (age, height, weight) → MetadataAPIProcessor.process_metadata_for_api → scaled (1,3)`

Invocation hierarchy:
- `run_inference(model, metadata_arr, signal_arr)` → 5-class predictions + probabilities + best_idx.
- `run_binary_inference(model, signal_arr)` → `(0/1, conf)` fast pre-filter (ecg_engine.py:349).
- `run_full_pipeline(hea_bytes, hea_name, dat_bytes, dat_name, ...)` → **full result dict** (Manual Upload & Live abnormal).
- `run_batch_pipeline(records)` → list of full results (+ `status`/`filename` each).
- `run_streaming_pipeline(signal, fs, age, sex, ...)` → **generator** of window dicts (two-stage: binary → full only if abnormal).
- `run_streaming_window_analysis(window_signal, fs, age, sex, ...)` → **one full result dict** for a flagged window (same shape as `run_full_pipeline`).

<!-- next: section 5 -->
## 5. Verified function inventory (signature, file:line)
**ecg_engine.py**
- `transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3)` L18 — Keras block.
- `se_block(inputs, ratio=16)` L33 — Squeeze-Excitation.
- `focal_loss_v3(y_true, y_pred_logits, gamma=2.0, alpha=0.25)` L42.
- `get_signal_from_files(hea_bytes, hea_name, dat_bytes, dat_name, tmp_dir)` L127 — wfdb reader → temp dir.
- `load_sample_signal(sample_name="13003_hr")` L140 — NEW: bundled `.hea/.dat` → `(signal, fs)`.
- `list_sample_signals() -> list[str]` L175 — NEW: record names in `Testing Samples/`.
- `build_demo_stream(sample_name="13003_hr", repeats=4)` L183 — NEW: tiles real record into longer stream; reuses loader above.
- `slice_signal_windows(signal, fs, window_sec=10.0, overlap_sec=5.0)` L247 — yields `(window_signal, start_sample, end_sample)`.
- `estimate_heart_rate(signal, fs) -> int` L212; `estimate_rhythm_regularity(signal, fs) -> tuple[str,str]` L221 — lead-0 peak analysis.
- `plot_ecg_signal(signal, fs=500, n_leads=None, duration_s=10.0)` L284 — matplotlib multi-lead plot.
- `run_binary_inference(model, signal_arr) -> tuple[int,float]` L349 — 0/1 pre-filter.

**api_bassam.py** — globals: `SIG_PROC = SignalAPIProcessor()` L43; `META_PROC = MetadataAPIProcessor()` L44; `LABELS_LIST` L57; `_BINARY_MODEL_LOCK`/`_MODEL_LOCK`.
- `_get_binary_model()` L33 / `get_binary_model()` L40; `_get_model()` L46 / `get_model()` L53 — lazy-loaded, thread-locked.
- `run_inference(model, metadata_arr, signal_arr)` L59 → predictions/best_idx/probabilities.
- `_generate_fallback_report(predictions, age, sex, hr, heart_rhythm, rhythm_reg)` L97 — markdown fallback when LLM unavailable.
- `run_full_pipeline(hea_bytes, hea_name, dat_bytes, dat_name, ...)` L127.
- `run_batch_pipeline(records: list[dict]) -> list[dict]` L205 — adds `status`/`filename`.
- `run_streaming_pipeline(continuous_signal, fs, age, sex, height=None, weight=None, window_sec=10.0, overlap_sec=5.0)` L231 — generator of window dicts; two-stage (binary → full if abnormal).
- `run_streaming_window_analysis(window_signal, fs, ...)` L306 — NEW: full single-window analysis = same shape as `run_full_pipeline`.

**xai_engine.py**
- `get_focus_class(predictions)` L102; `_find_gradcam_layer(model)` L54; `compute_gradcam(model, metadata_arr, signal_arr, class_index)` L115.
- `plot_gradcam(signal_raw, signal_arr, metadata_arr, model, predictions, class_index=None)` L176.
- `compute_saliency(...)` L220; `plot_saliency(...)` L240.
- `compute_integrated_gradients(model, metadata_arr, signal_arr, class_index, ...)` L284.
- `plot_ig_leads_final(ig_signal)` L332; `plot_ig_metadata_final(ig_meta)` L361; `plot_ig_signal(signal_raw, ig_signal)` L380.
- `run_xai(method, model, signal_raw, signal_arr, metadata_arr, predictions) -> dict of figures` L423.

**gpt_report.py**: `generate_ecg_report(predictions, api_key, ...)` L75 — tenacity-retried LLM, falls back to markdown.

## 6. Shared result-dict contract
Both `run_full_pipeline` and `run_streaming_window_analysis` return a dict with (UI consumes these in `render_results`, cardioinsight_nicegui.py:336-412):
- `predictions` — list of `(name, prob, is_pos)` tuples; e.g. NORM-only → `[("NORM", conf, True), ("MI", .05, False), ...]`.
- `report` — markdown string (LLM or `_generate_fallback_report`).
- `signal` — raw window/original ndarray.
- `signal_arr` — preprocessed input `(1, 1000, 12)`.
- `metadata_arr` — scaled `(1, 3)`.
- `fs`, `meta` (incl. `meta_importance`), `ig_signal` (IG lead importance).
`run_batch_pipeline` adds `status="success"` and `filename`.

## 7. NiceGUI architecture patterns
- **Per-client state**: `state = {...}` plain dict inside `main_page()` (L127) — replaces Streamlit `session_state`. NOT shared across clients.
- **Page**: LEFT column (mode controls + per-mode content) + RIGHT `content` column. `ui.timer(0.1, render_content, once=True)` L763 drives async render.
- **Mode dispatch**: `render_content()` L429 switches on `state["analysis_mode"]` (`"Manual Upload"` | `"📁 Batch Upload"` | `"🔁 Live Monitor"`); switched by inline-dense `ui.radio` L683 with `on_change=on_mode_change` L479 (stops streaming/timer, closes dialog, re-renders).
- **Heavy work** must use `await run.io_bound(...)` (model inference, `run_streaming_pipeline(...)`, `run_xai`, `fig_to_b64`, file loads, npy load) — blocking the event loop freezes UI.
- **Results rendering**: `render_results(data, viz_override)` L321 → prediction badges → ECG plot (or `viz_override`) → XAI figures (`run_xai`, `state["viz_type"]`, default `"None"`) → LLM report → metadata importance; figures → `fig_to_b64(fig, dpi=100)` L46 → `ui.html <img data:image/png;base64,...>`.
- **Tables/HTML**: `df_html(df)` L54 (pandas → `ci-table`); `_grab_upload(e)` L58 → `{name, bytes}`.
- **Dialogs**: `ui.dialog(...)` for Live Monitor popup; store inner box in `state` (e.g. `live_dialog_box`, `monitor_box`) to clear/update later.

<!-- next: sections 8-12 -->
## 8. Live ECG Signal Simulator (built in this task)
Scope: Live Monitor only. Touched files: `ecg_engine.py`, `api_bassam.py`, `cardioinsight_nicegui.py`. No changes to Manual Upload / Batch / settings / models / Streamlit files.

**Simulation chain** (reuses existing loaders; no synthetic generator in app path):
1. User picks a bundled sample (dropdown bound to `list_sample_signals()`, L601) OR uploads a continuous `.npy` (per-channel sampling-rate input, default 500 Hz, min 100 / max 1000).
2. Sample path: `load_sample_into_stream()` L599 → `await run.io_bound(lambda: build_demo_stream(sample_name, repeats=4))` → set `monitor_fs` → `_begin_stream(signal, fs)`.
3. Upload path: `start_stream()` L619 → `np.load(io.BytesIO(bytes))` → `_begin_stream(signal, monitor_fs)`.
4. `_begin_stream(signal, fs)` L633: pre-slices **all** windows upfront via `list(run_streaming_pipeline(signal, fs, age, sex, height, weight))` (10 s / 5 s overlap; uses `age_input`/`sex_input`/`height_input`/`weight_input` widgets) → stores `stream_signal`, `stream_windows`, `stream_idx=0`, `stream_abnormal=False`, `stream_abnormal_result=None`, `stream_abnormal_window=None` → opens live dialog → activates `stream_timer` (0.3 s tick) → `advance_stream`.

**Timer loop** `advance_stream()` L203:
- If `stream_running=False` → skip. Past last window → **finish**: build `stream_rows` (Window, Time, HR, Rhythm, Anomaly) → close dialog → banner `ci-banner ci-banner-ok` "✅ Streaming analysis complete." → `Final Stream Report` table + **Download Stream Report** CSV (`stream_report.csv`) → return.
- Per tick: `window_slice = stream_signal[start:end]`; render live ECG via `plot_ecg_signal(window_slice, fs, duration_s=10.0)` → `fig_to_b64` + `run.io_bound` into `live_dialog_box`.
- **Abnormal**: if `r["anomaly_flag"]` → `stream_running=False`, timer off, `stream_abnormal="pending"` → dialog shows `ci-banner ci-banner-bad` "🚨 Anomaly detected at Xs–Ys! Stream stopped." + ECG snapshot + spinner → main panel banner same + "Running full AI pipeline (XAI + LLM report)…" → `await run.io_bound(run_streaming_window_analysis, window_signal=window_slice, fs=fs, age, sex, height, weight)` → success: `stream_abnormal=True`, `stream_abnormal_result=full_result`, `stream_abnormal_window=r` → close dialog → `render_content()`. Failure: notify negative, reset `stream_abnormal=False`, re-render.
- Normal tick: small ECG preview + binary prediction/confidence row.

**Abnormal result display** (`render_content` Live branch, L454): red banner "🚨 Anomaly detected" with window time range + `🔁 Reset Live Monitor` (outline) → `await render_results(state["stream_abnormal_result"])` — exact Manual Upload layout (prediction badges → ECG + XAI figures → LLM report → metadata importance). `reset_stream_alert()` L664 zeroes `stream_abnormal/stream_abnormal_result/stream_abnormal_window/stream_running`.

**Live Monitor state keys**: `monitor_file`, `monitor_fs` (default 500), `selected_sample`, `sample_repeats` (4), `stream_signal`, `stream_windows`, `stream_idx`, `stream_running`, `stream_rows`, `stream_abnormal` (False | "pending" | True), `stream_abnormal_result`, `stream_abnormal_window`, `monitor_box`, `live_dialog_box`.

## 9. Visual style (NiceGUI)
Classes: `ci-table`, `ci-banner`, `ci-banner-bad` (red), `ci-banner-ok` (green), `ci-right`/left, `w-full`, `gap-2/3`, `no-wrap`, `flex-1`, `q-mb-md/sm/md`, `text-h6`, `text-blue-900/800`, `bordered`, `flat`, `outline`, `dense`, `inline`, `lg`. Buttons: primary vs `props("outline")`; iconography ▶ ⏹ 🔁 ⬇️ 🚨 ✅ 📁. Toasts: `ui.notify("msg", type="warning"|"negative"|"positive")`. Busy: `ui.spinner(size="lg", color="primary"|"negative")` + `spinner.delete()` after. Figures: `data:image/png;base64,...` via `fig_to_b64`, `width:100%;background:white;border-radius:8px`.

## 10. Constraints (non-negotiable)
- `settings.py` values (API keys, model paths, thresholds) MUST NOT be edited — override via `.env` if needed.
- Do NOT modify Batch mode code (NiceGUI migration pending), model weights, or training code.
- Do NOT touch or restyle Streamlit files — logic references only.
- Reuse existing functions (`get_signal_from_files`, `slice_signal_windows`, `run_streaming_pipeline`, `run_streaming_window_analysis`, `run_xai`, `generate_ecg_report`, `fig_to_b64`, …) — no duplicate logic.
- If a change outside Live Monitor scope seems necessary → STOP and ask.
- Keep `pyflakes` clean across touched files.

## 11. Common gotchas
- `state` is per-client (dict inside `main_page`); never cross-client state.
- Blocking TF/LLM/file ops MUST be in `await run.io_bound(...)` or UI freezes.
- `render_results` has optional `data`/`viz_override` params (added for Live abnormal reuse) — backward compatible; callers pass `None` for defaults.
- Binary model may be missing → `run_streaming_pipeline` forces `binary_pred=1` (forces full analysis) so demo still runs.
- Streaming pre-computes ALL windows at `_begin_stream` time; timer only plays back. Large signals → many windows.
- `ci-banner-bad`/`ci-banner-ok` banners must be created inside a cleared box each render to avoid stacking duplicates.
- Confirm which entry (`app_nicegui.py` vs `cardioinsight_nicegui.py`) is canonical before editing either.

## 12. Task history
- Completed (this task): Live ECG Signal Simulator — streaming simulation (sample + .npy), real-time popup, two-stage pipeline (binary pre-filter → full CNN-SE-Transformer only on abnormal), abnormal stop + alert + full XAI + LLM result displayed in Manual Upload layout.
- Pending (not started): Batch mode migration from Streamlit to NiceGUI — out of scope.


