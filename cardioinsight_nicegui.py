"""
CardioInsight — NiceGUI rewrite (drop-in replacement for the Streamlit app).
Run:  python cardioinsight_app.py
Requires the same project modules as before: api_bassam, xai_engine, ecg_engine, settings.
"""

import base64
import io
import traceback
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless backend — required outside Streamlit
import matplotlib.pyplot as plt

from nicegui import ui, run                # run = nicegui's async executor (io_bound / cpu_bound)

from api_bassam import (
    run_full_pipeline, get_model, run_batch_pipeline, run_streaming_pipeline,
    run_streaming_window_analysis,
)
from xai_engine import run_xai, plot_ig_leads_final, plot_ig_metadata_final
from settings import ECG_DISPLAY_DURATION, DEMO_MODE
from ecg_engine import plot_ecg_signal, build_demo_stream, list_sample_signals


# ── Helpers ────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _logo_b64() -> str:
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


@lru_cache(maxsize=1)
def _cached_model():
    return get_model()


def fig_to_b64(fig, dpi: int = 100) -> str:
    """Render a matplotlib figure to a base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def df_html(df: pd.DataFrame) -> None:
    ui.html(df.to_html(index=False, border=0, classes="ci-table"))


async def _grab_upload(e) -> dict:
    """Resolve a NiceGUI upload event to {'name', 'bytes'}.

    Works for both SmallFileUpload (in-memory) and LargeFileUpload (spooled
    temp file) — reading is async, so do it here in the upload handler.
    """
    return {"name": e.file.name, "bytes": await e.file.read()}


# ── Global styles (once, at import time) ───────────────────────────────────────
ui.add_head_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

body, .q-card, .q-tabs { font-family: 'DM Sans', sans-serif; }
.nicegui-content { padding: 0 2rem 3rem; }

.ci-header { background: #ffffff; color: #1e3a8a; padding: 0.5rem 1rem; border-bottom: 1px solid #dbeafe; }
.ci-logo-img { width: 40px; height: 40px; object-fit: contain; border-radius: 8px; }
.ci-logo-text { font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: #1e3a8a; letter-spacing: -0.5px; line-height: 1.1; }
.ci-subtitle { font-size: 0.78rem; color: #64748b; font-weight: 400; }
.ci-hamburger { font-size: 1.4rem; color: #3b82f6; }

.ci-disclaimer { background: #fff7ed; border: 1px solid #fdba74; border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.78rem; color: #92400e; text-align: center; margin: 0.6rem 0 1rem; }
.ci-demo { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #1e40af; margin-bottom: 0.8rem; }

.ci-ecg-box { border: 2px solid #fbbf24; border-radius: 12px; padding: 0.8rem 1rem; background: #fffdf0; margin-bottom: 1.2rem; }
.ci-ecg-title { color: #1d4ed8; font-weight: 600; font-size: 1rem; margin-bottom: 0.6rem; }
.ci-report-box { border: 2px solid #38bdf8; border-radius: 12px; padding: 1.2rem 1.4rem; background: #f0f9ff; margin-bottom: 1.2rem; }
.ci-report-title { color: #1d4ed8; font-weight: 600; font-size: 1rem; margin-bottom: 0.8rem; }
.ci-meta-box { border: 2px solid #2563eb; border-radius: 12px; padding: 1.2rem 1.4rem; background: #eff6ff; margin-bottom: 1.2rem; }
.ci-meta-title { color: #1d4ed8; font-weight: 600; font-size: 1rem; margin-bottom: 0.8rem; }
.ci-rhythm-row { display: flex; gap: 0.8rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
.ci-rhythm-card { flex: 1; min-width: 150px; background: #fff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 0.7rem 1rem; }
.ci-rhythm-label { font-size: 0.72rem; color: #64748b; font-weight: 500; margin-bottom: 0.2rem; }
.ci-rhythm-value { font-size: 0.95rem; color: #1e40af; font-weight: 600; }

.ci-banner { border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.88rem; margin-bottom: 0.8rem; }
.ci-banner-ok { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.ci-banner-bad { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

.ci-welcome { background: #fff; border-radius: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); padding: 3rem 2rem; text-align: center; margin-top: 1rem; }
.ci-welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.ci-welcome h2 { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #1e3a8a; margin: 0 0 0.6rem; }
.ci-welcome p { color: #64748b; font-size: 0.92rem; line-height: 1.6; max-width: 440px; margin: 0 auto; }

.ci-about { background: #fff; border-radius: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); padding: 1.2rem 1.6rem; margin-top: 1.2rem; }
.ci-about-header { display: flex; align-items: center; gap: 0.5rem; font-weight: 600; font-size: 0.95rem; color: #1e3a8a; margin-bottom: 0.8rem; }
.ci-about-icon { background: #2563eb; color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 0.8rem; font-weight: 700; }

.pred-badge { display: inline-block; padding: 0.22rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin: 0.2rem 0.25rem; }
.pred-positive { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.pred-normal { background: #dcfce7; color: #166534; border: 1px solid #86efac; }

.ci-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; background: #fff; }
.ci-table th { background: #eff6ff; color: #1e40af; padding: 6px 10px; text-align: left; }
.ci-table td { padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }

.q-btn.bg-primary { background: #2563eb !important; }
.ci-left { width: 31%; position: sticky; top: 90px; max-height: calc(100vh - 110px); overflow-y: auto; }
.ci-right { width: 67%; }
</style>
""", shared=True)


# ── Page ───────────────────────────────────────────────────────────────────────
@ui.page("/")
def main_page() -> None:
    # Per-client state (plain dict — replaces Streamlit session_state).
    state = {
        "analysis_mode": "Manual Upload",
        "viz_type": "None",
        "analysed": False,
        "predictions": None, "report": None, "signal": None, "signal_arr": None,
        "metadata_arr": None, "fs": None, "meta": None, "ig_signal_raw": None,
        "results": None, "batch_results": None, "batch_rows": None,
        "manual_hea": None, "manual_dat": None,
        "batch_files": [], "batch_zip": None,
        "monitor_file": None, "monitor_fs": 500,
        "stream_signal": None, "stream_windows": None,
        "stream_idx": 0, "stream_running": False, "stream_rows": None,
        "monitor_box": None,
        # Live-monitor simulator / abnormal-alert additions
        "live_dialog_box": None,
        "stream_abnormal": False,       # False | "pending" | True
        "stream_abnormal_result": None,  # full-pipeline result dict for the flagged window
        "stream_abnormal_window": None,  # the raw window dict that triggered the alert
        "selected_sample": (list_sample_signals() or [None])[0],
        "sample_repeats": 4,
    }
    refs = {}  # late-bound UI references (mode_controls refreshable, content column)

    # ── Sticky header + disclaimer ─────────────────────────────────────────────
    with ui.header().classes("ci-header w-full"):
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.image(f"data:image/png;base64,{_logo_b64()}").classes("ci-logo-img")
            with ui.column().classes("gap-0"):
                ui.label("CardioInsight").classes("ci-logo-text")
                ui.label("AI-Powered Cardiac Rhythm Classification").classes("ci-subtitle")
        ui.label("☰").classes("ci-hamburger")
    ui.html('<div class="ci-disclaimer">⚠️ <strong>Research use only.</strong> CardioInsight is not a '
            'certified medical device and must not be used as a substitute for clinical diagnosis. '
            'Always consult a qualified physician for medical decisions.</div>')

    # ── Shared About block ─────────────────────────────────────────────────────
    def render_about() -> None:
        with ui.element("div").classes("ci-about w-full"):
            ui.html('<div class="ci-about-header"><span class="ci-about-icon">i</span> About CardioInsight</div>')
            with ui.tabs().classes("w-full") as tabs:
                t1 = ui.tab("CardioInsight")
                t2 = ui.tab("System function")
            with ui.tab_panels(tabs, value=t1).classes("w-full"):
                with ui.tab_panel(t1):
                    ui.markdown("CardioInsight is an ECG signal analysis application that detects potential "
                                "heart rhythm abnormalities. It uses a hybrid CNN-SE-Transformer deep learning "
                                "model trained on the PTB-XL dataset to classify ECG recordings into five "
                                "diagnostic categories: Normal, Myocardial Infarction, ST/T-Wave Change, "
                                "Conduction Disturbance, and Hypertrophy. Model predictions are then passed "
                                "to a fine-tuned GPT-4o-mini to generate structured clinical interpretation reports.")
                with ui.tab_panel(t2):
                    ui.markdown("""Pipeline overview:
- **Input** — Upload a WFDB-format ECG record (`.hea` + `.dat`) and enter patient metadata.
- **Preprocessing** — Signal loaded via `wfdb`; normalised per-lead; heart rate and rhythm estimated from R-peaks.
- **Model inference** — Hybrid CNN-SE-Transformer → multi-label probabilities for 5 classes.
- **Threshold tuning** — Per-class F1-optimal thresholds determine positive labels.
- **LLM Report** — Positive predictions → fine-tuned GPT-4o-mini → structured clinical report.
- **Visualisation** — ECG waveform (all leads), lead contribution, metadata chart, rhythm cards.
- **Streaming** — Continuous signal monitoring with real-time windowed analysis.
- **Batch** — Process multiple ECG records simultaneously.

Configuration: Edit `.env` to set model path, API key, and thresholds.""")

    def welcome(icon: str, title: str, text: str) -> None:
        ui.html(f'<div class="ci-welcome w-full"><div class="ci-welcome-icon">{icon}</div>'
                f'<h2>{title}</h2><p>{text}</p></div>')

    # ── Live-signal popup dialog (created once, reused across streams) ─────────
    with ui.dialog() as live_dialog, ui.card().classes("q-pa-md").style("min-width:600px;max-width:900px;"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            ui.label("📡 Live ECG Stream").classes("text-h6 text-blue-900")
            ui.button(icon="close", on_click=lambda: live_dialog.close()).props("flat round dense")
        live_dialog_content = ui.column().classes("w-full gap-2")
        state["live_dialog_box"] = live_dialog_content

    # ── Streaming timer (created once, activated on demand) ────────────────────
    async def advance_stream() -> None:
        if not state["stream_running"]:
            stream_timer.active = False
            return
        windows = state["stream_windows"]
        idx = state["stream_idx"]
        box = state.get("monitor_box")
        dialog_box = state.get("live_dialog_box")
        if box is None:
            return
        if idx >= len(windows):
            state["stream_running"] = False
            stream_timer.active = False
            rows = [{"Window": w["window_idx"] + 1,
                     "Time": f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s",
                     "HR": w["hr"], "Rhythm": w["heart_rhythm"],
                     "Anomaly": "Yes" if w["anomaly_flag"] else "No"} for w in windows]
            state["stream_rows"] = rows
            live_dialog.close()
            box.clear()
            with box:
                ui.html('<div class="ci-banner ci-banner-ok">✅ Streaming analysis complete.</div>')
                ui.label("Final Stream Report").classes("text-h6 text-blue-900 q-mb-sm")
                df_html(pd.DataFrame(rows))
                ui.button("⬇️ Download Stream Report",
                          on_click=lambda: ui.download(pd.DataFrame(rows).to_csv(index=False).encode(),
                                                       "stream_report.csv")).props("outline").classes("q-mt-md")
            return

        r = windows[idx]
        state["stream_idx"] = idx + 1
        fs = state["monitor_fs"]
        sig = state["stream_signal"]
        window_slice = sig[r["start_sample"]:r["end_sample"]]
        b64 = await run.io_bound(
            lambda: fig_to_b64(plot_ecg_signal(window_slice, fs=fs, duration_s=10.0))
        )

        # ── Abnormal window: halt the stream, alert, then run the full pipeline ─
        if r["anomaly_flag"]:
            state["stream_running"] = False
            stream_timer.active = False
            state["stream_abnormal"] = "pending"

            if dialog_box is not None:
                dialog_box.clear()
                with dialog_box:
                    ui.html(f'<div class="ci-banner ci-banner-bad">🚨 Anomaly detected at {r["start_sec"]:.1f}s '
                            f'- {r["end_sec"]:.1f}s! Stream stopped.</div>')
                    ui.html(f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;border-radius:8px;">')
                    ui.spinner(size="lg", color="negative").classes("q-mt-md")

            box.clear()
            with box:
                ui.html(f'<div class="ci-banner ci-banner-bad">🚨 Anomaly detected at {r["start_sec"]:.1f}s '
                        f'- {r["end_sec"]:.1f}s! Stream stopped for detailed analysis.</div>')
                ui.spinner(size="lg", color="negative")
                ui.label("Running full AI pipeline (XAI + LLM report)…").classes("text-blue-800")

            try:
                full_result = await run.io_bound(
                    run_streaming_window_analysis,
                    window_signal=window_slice, fs=fs,
                    age=int(age_input.value) if age_input.value is not None else 50,
                    sex=sex_input.value, height=height_input.value, weight=weight_input.value,
                )
            except Exception as exc:
                ui.notify(f"Full analysis failed: {exc}", type="negative")
                print(traceback.format_exc())
                state["stream_abnormal"] = False
                live_dialog.close()
                await render_content()
                return

            state["stream_abnormal"] = True
            state["stream_abnormal_result"] = full_result
            state["stream_abnormal_window"] = r
            live_dialog.close()
            await render_content()
            return

        box.clear()
        with box:
            ui.label(f"🔴 Live Analysis — Window {idx + 1}/{len(windows)}").classes("text-h6 text-blue-900 q-mb-sm")
            ui.linear_progress(value=(idx + 1) / len(windows)).classes("q-mb-md")
            ui.html(f'<div class="ci-banner ci-banner-ok">✅ Normal rhythm at {r["start_sec"]:.1f}s '
                    f'- {r["end_sec"]:.1f}s</div>')
            badges = " ".join(
                f'<span class="pred-badge {"pred-positive" if is_pos else "pred-normal"}">{name} — {prob:.0%}</span>'
                for name, prob, is_pos in r["predictions"]
            )
            ui.html(f"<div><strong>Detailed Predictions:</strong> {badges}</div>").classes("q-mb-md")
            ui.html('<div class="ci-ecg-box"><div class="ci-ecg-title">📈 Window ECG</div>'
                    f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;border-radius:8px;"></div>')
            if idx > 0:
                hist = pd.DataFrame([
                    {"Window": i + 1, "Time": f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s",
                     "HR": w["hr"], "Rhythm": w["heart_rhythm"],
                     "Anomaly": "🚨" if w["anomaly_flag"] else "✅"}
                    for i, w in enumerate(windows[:idx + 1])
                ])
                ui.label("Analysis History").classes("text-subtitle1 text-blue-900 q-mt-sm")
                df_html(hist)

        # ── Mirror the same window into the live popup ──────────────────────────
        if dialog_box is not None:
            dialog_box.clear()
            with dialog_box:
                ui.label(f"Window {idx + 1}/{len(windows)} — {r['start_sec']:.1f}s to {r['end_sec']:.1f}s") \
                    .classes("text-subtitle1 text-blue-900")
                ui.linear_progress(value=(idx + 1) / len(windows)).classes("q-mb-sm")
                ui.html(f'<div class="ci-banner ci-banner-ok">✅ Normal rhythm — HR {r["hr"]} bpm, '
                        f'{r["heart_rhythm"]}</div>')
                ui.html(f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;border-radius:8px;">')

    stream_timer = ui.timer(0.3, advance_stream, active=False)

    # ── RIGHT COLUMN renderers ─────────────────────────────────────────────────
    async def render_results(data: dict | None = None, viz_override: str | None = None) -> None:
        """Render prediction badges + ECG/XAI + LLM report + metadata outputs.

        Defaults to the shared `state` dict (Manual Upload / Batch behaviour,
        unchanged). Live Monitor's abnormal-window alert passes an explicit
        `data` dict (same shape as run_full_pipeline's return value) so it can
        reuse this exact layout without touching Manual Upload's state.
        """
        if data is None:
            data = {
                "predictions": state["predictions"], "report": state["report"],
                "signal": state["signal"], "signal_arr": state["signal_arr"],
                "metadata_arr": state["metadata_arr"], "fs": state["fs"],
                "ig_signal": state["ig_signal_raw"], "meta": state["meta"],
            }
        preds = data["predictions"]
        meta = data["meta"]
        report = data["report"]
        fs = data["fs"] if data["fs"] else 500
        viz = viz_override if viz_override is not None else state["viz_type"]

        badges = " ".join(
            f'<span class="pred-badge {"pred-positive" if is_pos else "pred-normal"}">{name} — {prob:.0%}</span>'
            for name, prob, is_pos in preds
        )
        ui.html(f"<div class='q-mb-md'><strong>Predictions:</strong> {badges}</div>")

        if viz == "None":
            if data["signal"] is not None:
                sig = data["signal"]
                n_leads = sig.shape[1] if getattr(sig, "ndim", 1) > 1 else 1
                b64 = await run.io_bound(
                    lambda: fig_to_b64(plot_ecg_signal(sig, fs=fs, duration_s=ECG_DISPLAY_DURATION))
                )
                ui.html(
                    '<div class="ci-ecg-box"><div class="ci-ecg-title">📈 ECG Signal (Raw Waveform)'
                    f'<span style="font-size:0.75rem;font-weight:400;color:#64748b;margin-left:0.5rem;">'
                    f'{n_leads} leads · scroll to view all</span></div>'
                    f'<div style="height:420px;overflow-y:auto;border-radius:8px;">'
                    f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;"></div></div>'
                )
            else:
                ui.label("No signal data found to display.").classes("text-orange-700")
        else:
            spinner = ui.spinner(size="lg", color="primary")
            ui.label(f"Computing {viz}…").classes("text-blue-800")
            try:
                xai_figs = await run.io_bound(
                    run_xai, method=viz, model=_cached_model(),
                    signal_raw=data["signal"], signal_arr=data["signal_arr"],
                    metadata_arr=data["metadata_arr"], predictions=data["predictions"],
                )
                spinner.delete()
                if xai_figs:
                    b64 = await run.io_bound(lambda: fig_to_b64(list(xai_figs.values())[0]))
                    ui.html(
                        f'<div class="ci-ecg-box"><div class="ci-ecg-title">🔬 Interpretation: {viz}</div>'
                        f'<div style="height:420px;overflow-y:auto;border-radius:8px;background:white;">'
                        f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;"></div></div>'
                    )
                else:
                    ui.label("Generating visualization...").classes("text-blue-800")
            except Exception as exc:
                spinner.delete()
                ui.notify(f"❌ XAI Error: {exc}", type="negative")
                ui.label("💡 Tip: Switch to 'None' to view raw ECG while debugging.").classes("text-grey-600")

        # LLM Report
        with ui.element("div").classes("ci-report-box w-full"):
            ui.html('<div class="ci-report-title">📋 LLM Report</div>')
            ui.markdown(report)
        ui.button("⬇️ Download Report",
                  on_click=lambda: ui.download(report.encode("utf-8"), "cardioinsight_report.md")
                  ).props("outline").classes("q-mb-lg")

        # Metadata outputs
        with ui.element("div").classes("ci-meta-box w-full"):
            ui.html('<div class="ci-meta-title">📊 Metadata Outputs</div>')
            ui.html(
                '<div class="ci-rhythm-row">'
                f'<div class="ci-rhythm-card"><div class="ci-rhythm-label">Heart Rhythm</div>'
                f'<div class="ci-rhythm-value">{meta.get("heart_rhythm", "—")}</div></div>'
                f'<div class="ci-rhythm-card"><div class="ci-rhythm-label">Rhythm Regularity</div>'
                f'<div class="ci-rhythm-value">{meta.get("rhythm_regularity", "—")}</div></div>'
                '</div>'
            )
            with ui.row().classes("w-full items-start no-wrap gap-4"):
                with ui.column().classes("w-1/2 gap-2"):
                    try:
                        b64 = await run.io_bound(lambda: fig_to_b64(plot_ig_leads_final(data["ig_signal"])))
                        ui.html(f'<img src="data:image/png;base64,{b64}" style="width:100%;background:white;border-radius:8px;">')
                    except Exception as exc:
                        ui.label(f"Lead importance plot failed: {exc}").classes("text-orange-700")
                with ui.column().classes("w-1/2 gap-2"):
                    try:
                        b64 = await run.io_bound(
                            lambda: fig_to_b64(plot_ig_metadata_final(meta["meta_importance"]))
                        )
                        ui.html(f'<img src="data:image/png;base64,{b64}" style="width:100%;background:white;border-radius:8px;">')
                    except Exception as exc:
                        ui.label(f"Metadata plot failed: {exc}").classes("text-orange-700")
                    df_html(pd.DataFrame({
                        "Feature": ["Age", "Sex", "Heart Rate (bpm)"],
                        "Value": [str(meta["age"]), meta["sex"], f"{meta['hr']} bpm"],
                    }))

        render_about()

    async def render_content() -> None:
        content.clear()
        with content:
            mode = state["analysis_mode"]
            if mode == "Manual Upload":
                if state["analysed"]:
                    await render_results()
                else:
                    welcome("🩺", "Welcome to ECG Analysis",
                            "Upload an ECG signal file to get started. The system will visualise the "
                            "waveform and provide classification of potential cardiac conditions.")
                    render_about()
            elif mode == "📁 Batch Upload":
                if state.get("batch_rows"):
                    ui.label("Batch Summary").classes("text-h6 text-blue-900 q-mb-sm")
                    df_html(pd.DataFrame(state["batch_rows"]))
                    ui.button("⬇️ Download CSV",
                              on_click=lambda: ui.download(
                                  pd.DataFrame(state["batch_rows"]).to_csv(index=False).encode(),
                                  "cardioinsight_batch.csv")).props("outline").classes("q-mt-md")
                else:
                    welcome("📁", "Batch ECG Analysis",
                            "Upload multiple .hea/.dat pairs or a ZIP archive, then run the batch analysis.")
                render_about()
            else:  # 🔁 Live Monitor
                if state["stream_abnormal"] is True and state["stream_abnormal_result"] is not None:
                    # Abnormal window detected: show alert + full pipeline results
                    # in the same layout/format as Manual Upload.
                    w = state["stream_abnormal_window"]
                    ui.html(
                        '<div class="ci-banner ci-banner-bad" style="font-size:1rem;font-weight:600;">'
                        f'🚨 STREAM STOPPED — Abnormal rhythm detected at {w["start_sec"]:.1f}s-{w["end_sec"]:.1f}s '
                        f'(binary pre-filter confidence: {w.get("binary_conf", 0):.2%})</div>'
                    )
                    ui.button("🔁 Reset Live Monitor", on_click=reset_stream_alert).props("outline").classes("q-mb-md")
                    await render_results(state["stream_abnormal_result"])
                    return
                box = ui.column().classes("w-full gap-2")
                state["monitor_box"] = box
                if state["stream_running"] or state["stream_abnormal"] == "pending":
                    with box:
                        ui.spinner(size="lg", color="primary")
                        ui.label("Streaming…").classes("text-blue-800")
                else:
                    welcome("📡", "Live ECG Monitor",
                            "Upload a continuous .npy signal — or load a bundled sample — and start "
                            "monitoring to analyze sliding windows in real time.")
                    render_about()

    # ── Handlers ───────────────────────────────────────────────────────────────
    async def on_mode_change(e) -> None:
        state["analysis_mode"] = e.value
        state["stream_running"] = False
        stream_timer.active = False
        live_dialog.close()
        if refs.get("mode_controls") is not None:
            refs["mode_controls"].refresh()
        await render_content()

    async def on_viz_change(e) -> None:
        state["viz_type"] = e.value
        if state["analysed"]:
            await render_content()

    async def on_manual_hea(e):   state["manual_hea"] = await _grab_upload(e)
    async def on_manual_dat(e):   state["manual_dat"] = await _grab_upload(e)
    async def on_batch_file(e):   state["batch_files"].append(await _grab_upload(e))
    async def on_batch_zip(e):    state["batch_zip"] = await _grab_upload(e)
    async def on_monitor_file(e): state["monitor_file"] = await _grab_upload(e)

    async def run_manual() -> None:
        errors = []
        if state["manual_hea"] is None or state["manual_dat"] is None:
            errors.append("Please upload both the .hea and .dat ECG files.")
        if age_input.value is None:
            errors.append("Please enter patient age.")
        for e in errors:
            ui.notify(e, type="negative")
        if errors:
            return

        spinner = ui.spinner(size="lg", color="primary")
        try:
            result = await run.io_bound(
                run_full_pipeline,
                hea_bytes=state["manual_hea"]["bytes"], hea_name=state["manual_hea"]["name"],
                dat_bytes=state["manual_dat"]["bytes"], dat_name=state["manual_dat"]["name"],
                age=int(age_input.value), sex=sex_input.value,
                height=height_input.value, weight=weight_input.value,
            )
        except Exception as exc:
            ui.notify(f"Analysis failed: {exc}", type="negative")
            print(traceback.format_exc())
            return
        finally:
            spinner.delete()

        state.update({
            "predictions": result["predictions"],
            "report": result["report"],
            "signal": result["signal"],
            "signal_arr": result["signal_arr"],
            "metadata_arr": result["metadata_arr"],
            "fs": result["fs"],
            "meta": result["meta"],
            "ig_signal_raw": result["ig_signal"],
            "results": {"meta": {"meta_importance": result["meta"]["meta_importance"],
                                 "lead_importance": result["ig_signal"]}},
            "analysed": True,
        })
        await render_content()

    async def run_batch() -> None:
        records = []
        file_map = {}
        for f in state["batch_files"]:
            stem = Path(f.name).stem
            ext = Path(f.name).suffix.lower()
            file_map.setdefault(stem, {})[ext] = f
        for stem, pair in file_map.items():
            if ".hea" in pair and ".dat" in pair:
                records.append({
                    "hea_bytes": pair[".hea"]["bytes"], "hea_name": pair[".hea"]["name"],
                    "dat_bytes": pair[".dat"]["bytes"], "dat_name": pair[".dat"]["name"],
                    "age": int(age_input.value) if age_input.value is not None else 50,
                    "sex": sex_input.value,
                    "height": height_input.value, "weight": weight_input.value,
                })
        if state["batch_zip"] is not None:
            with zipfile.ZipFile(io.BytesIO(state["batch_zip"]["bytes"])) as zf:
                names = zf.namelist()
                for hea_name in [n for n in names if n.lower().endswith(".hea")]:
                    dat_name = hea_name[:-4] + ".dat"
                    if dat_name in names:
                        records.append({
                            "hea_bytes": zf.read(hea_name), "hea_name": Path(hea_name).name,
                            "dat_bytes": zf.read(dat_name), "dat_name": Path(dat_name).name,
                            "age": int(age_input.value) if age_input.value is not None else 50,
                            "sex": sex_input.value,
                            "height": height_input.value, "weight": weight_input.value,
                        })
        if not records:
            ui.notify("No valid .hea/.dat pairs found.", type="negative")
            return

        spinner = ui.spinner(size="lg", color="primary")
        try:
            batch_results = await run.io_bound(run_batch_pipeline, records)
        except Exception as exc:
            ui.notify(f"Batch processing failed: {exc}", type="negative")
            print(traceback.format_exc())
            return
        finally:
            spinner.delete()

        rows = []
        for r in batch_results:
            if r["status"] == "success":
                preds = ", ".join([n for n, _, p in r["predictions"] if p]) or "NORM"
                rows.append({"File": r["filename"], "Prediction": preds,
                             "HR": r["meta"]["hr"], "Status": "✅"})
            else:
                rows.append({"File": r["filename"], "Prediction": "Error", "HR": "-", "Status": "❌"})
        state["batch_results"] = batch_results
        state["batch_rows"] = rows
        await render_content()

    async def on_sample_change(e) -> None:
        state["selected_sample"] = e.value

    async def load_sample_into_stream() -> None:
        """Simulate a continuous ECG stream from a bundled sample record."""
        sample_name = state.get("selected_sample")
        if not sample_name:
            ui.notify("No sample signals found in 'Testing Samples'.", type="warning")
            return
        spinner = ui.spinner(size="lg", color="primary")
        try:
            signal, fs = await run.io_bound(
                lambda: build_demo_stream(sample_name, repeats=state.get("sample_repeats", 4))
            )
        except Exception as exc:
            ui.notify(f"Failed to load sample: {exc}", type="negative")
            return
        finally:
            spinner.delete()

        state["monitor_fs"] = int(fs)
        await _begin_stream(signal, fs)

    async def start_stream() -> None:
        if state["monitor_file"] is None:
            ui.notify("Upload a continuous .npy signal, or load a sample, first.", type="warning")
            return
        spinner = ui.spinner(size="lg", color="primary")
        try:
            signal = await run.io_bound(lambda: np.load(io.BytesIO(state["monitor_file"]["bytes"])))
        except Exception as exc:
            ui.notify(f"Failed to read signal: {exc}", type="negative")
            return
        finally:
            spinner.delete()
        await _begin_stream(signal, state["monitor_fs"])

    async def _begin_stream(signal: np.ndarray, fs: float) -> None:
        spinner = ui.spinner(size="lg", color="primary")
        try:
            windows = await run.io_bound(lambda: list(run_streaming_pipeline(
                signal, fs,
                age=int(age_input.value) if age_input.value is not None else 50,
                sex=sex_input.value, height=height_input.value, weight=weight_input.value,
                window_sec=10.0, overlap_sec=5.0,
            )))
        except Exception as exc:
            ui.notify(f"Failed to start stream: {exc}", type="negative")
            return
        finally:
            spinner.delete()

        state.update({
            "stream_signal": signal, "stream_windows": windows,
            "stream_idx": 0, "stream_running": True,
            "stream_abnormal": False, "stream_abnormal_result": None,
            "stream_abnormal_window": None,
        })
        await render_content()
        live_dialog.open()
        stream_timer.active = True

    async def stop_stream() -> None:
        state["stream_running"] = False
        stream_timer.active = False
        live_dialog.close()
        await render_content()

    async def reset_stream_alert() -> None:
        state.update({
            "stream_abnormal": False, "stream_abnormal_result": None,
            "stream_abnormal_window": None, "stream_running": False,
            "stream_windows": None, "stream_signal": None, "stream_idx": 0,
        })
        stream_timer.active = False
        await render_content()

    # ── Layout: fixed left | scrollable right ──────────────────────────────────
    with ui.row().classes("w-full items-start no-wrap q-mt-md gap-lg"):
        # ── LEFT: control panel ──────────────────────────────────────────────────
        with ui.column().classes("ci-left gap-3"):
            with ui.card().classes("w-full q-pa-md"):
                ui.label("🎛️ Control Panel").classes("text-h6 text-blue-900 q-mb-sm")

                if DEMO_MODE:
                    ui.html('<div class="ci-demo">🛡️ <strong>Demo Mode Active</strong> — Mock predictions enabled.</div>')

                ui.radio(["Manual Upload", "📁 Batch Upload", "🔁 Live Monitor"],
                         value="Manual Upload", on_change=on_mode_change).props("inline dense")

                ui.separator().classes("q-my-sm")
                ui.label("Patient Information").classes("text-subtitle2 text-blue-900")
                age_input = ui.number(label="Age", min=1, max=120, value=None,
                                      placeholder="Enter age").classes("w-full")
                sex_input = ui.select(["Male", "Female"], value="Male", label="Gender").classes("w-full")
                height_input = ui.number(label="Height (cm)", min=50, max=250, value=None,
                                         placeholder="Enter height").classes("w-full")
                weight_input = ui.number(label="Weight (kg)", min=2, max=300, value=None,
                                         placeholder="Enter weight").classes("w-full")

                ui.separator().classes("q-my-sm")
                ui.label("Explainability Method").classes("text-subtitle2 text-blue-900")
                ui.select(["None", "Grad-CAM", "Saliency Maps", "Integrated Grads (IG)"],
                          value="None", label="Visualization Type",
                          on_change=on_viz_change).classes("w-full")

                ui.separator().classes("q-my-sm")

                # Dynamic mode-specific section (refreshed when the mode changes)
                @ui.refreshable
                def mode_controls() -> None:
                    mode = state["analysis_mode"]
                    if mode == "Manual Upload":
                        ui.label("ECG Signal Upload").classes("text-subtitle2 text-blue-900")
                        ui.upload(label="Upload .hea file", auto_upload=True,
                                  on_upload=on_manual_hea
                                  ).props('accept=.hea flat bordered').classes("w-full")
                        ui.upload(label="Upload .dat file", auto_upload=True,
                                  on_upload=on_manual_dat
                                  ).props('accept=.dat flat bordered').classes("w-full")
                        ui.button("🔍 Analyse ECG", on_click=run_manual,
                                  color="primary").classes("w-full")
                    elif mode == "📁 Batch Upload":
                        ui.label("Batch ECG Upload").classes("text-subtitle2 text-blue-900")
                        ui.label("Upload multiple .hea/.dat pairs or a ZIP archive.").classes("text-grey-600 text-xs")
                        ui.upload(label="Select .hea & .dat files", multiple=True, auto_upload=True,
                                  on_upload=on_batch_file
                                  ).props('accept=.hea,.dat flat bordered').classes("w-full")
                        ui.upload(label="Or upload ZIP archive", auto_upload=True,
                                  on_upload=on_batch_zip
                                  ).props('accept=.zip flat bordered').classes("w-full")
                        ui.button("🔍 Analyse Batch", on_click=run_batch,
                                  color="primary").classes("w-full")
                    else:
                        ui.label("📡 Continuous ECG Monitoring").classes("text-subtitle2 text-blue-900")
                        ui.label("Real-time ingestion → binary pre-filter → auto-routing to "
                                 "CardioInsight AI pipeline.").classes("text-grey-600 text-xs")

                        ui.label("Simulate from a bundled sample record").classes("text-caption text-grey-700 q-mt-sm")
                        with ui.row().classes("w-full no-wrap gap-2 items-end"):
                            sample_options = list_sample_signals()
                            ui.select(sample_options, value=state["selected_sample"],
                                      label="Sample record", on_change=on_sample_change,
                                      ).classes("flex-1")
                            ui.button("📂 Load Sample", on_click=load_sample_into_stream,
                                       color="primary").props("outline")
                        ui.separator().classes("q-my-sm")
                        ui.upload(label="Upload continuous .npy signal", auto_upload=True,
                                  on_upload=on_monitor_file
                                  ).props('accept=.npy flat bordered').classes("w-full")
                        ui.number(label="Sampling Rate (Hz)", min=100, max=1000, value=500,
                                  on_change=lambda e: state.update(monitor_fs=int(e.value or 500))
                                  ).classes("w-full")
                        with ui.row().classes("w-full no-wrap gap-2"):
                            ui.button("▶ Start Stream", on_click=start_stream,
                                      color="primary").classes("flex-1")
                            ui.button("⏹ Stop", on_click=stop_stream).props("outline").classes("flex-1")

                refs["mode_controls"] = mode_controls
                mode_controls()

        # ── RIGHT: results panel ─────────────────────────────────────────────────
        with ui.column().classes("ci-right gap-3"):
            content = ui.column().classes("w-full gap-3")
            refs["content"] = content

    # Initial render of the right panel (one-shot timer; render_content is async)
    ui.timer(0.1, render_content, once=True)


# ── Bootstrap ──────────────────────────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="CardioInsight 🫀", host="0.0.0.0", port=8501, reload=False, show=False)
