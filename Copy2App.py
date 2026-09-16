import base64
import io
import traceback
import zipfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from api_bassam import run_full_pipeline, get_model, run_batch_pipeline
from xai_engine import run_xai, plot_ig_leads_final, plot_ig_metadata_final
from settings import ECG_DISPLAY_DURATION, DEMO_MODE
from ecg_engine import plot_ecg_signal

# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def _logo_b64() -> str:
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

@st.cache_resource
def _cached_model():
    return get_model()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioInsight",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { padding: 0 0 3rem 0; max-width: 100%; }

.ci-header { position: sticky; top: 0; z-index: 999; background: #fff; display: flex; align-items: center; justify-content: space-between; padding: 1rem 2.5rem 0.8rem 2.5rem; border-bottom: 1px solid #dbeafe; }
.ci-logo-row { display: flex; align-items: center; gap: 0.6rem; }
.ci-logo-img { width: 40px; height: 40px; object-fit: contain; border-radius: 8px; padding: 2px; }
.ci-logo-text { font-family: 'DM Serif Display', serif; font-size: 1.75rem; color: #1e3a8a; letter-spacing: -0.5px; }
.ci-subtitle { font-size: 0.78rem; color: #64748b; font-weight: 400; margin-top: -2px; }
.ci-hamburger { font-size: 1.4rem; color: #3b82f6; cursor: pointer; }

.ci-disclaimer { background: #fff7ed; border: 1px solid #fdba74; border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.78rem; color: #92400e; text-align: center; margin: 0.6rem 2.5rem 0 2.5rem; }
.ci-ecg-box { border: 2px solid #fbbf24; border-radius: 12px; padding: 0.8rem 1rem; background: #fffdf0; margin-bottom: 1.2rem; }
.ci-ecg-title { color: #1d4ed8; font-weight: 600; font-size: 1rem; margin-bottom: 0.6rem; }
.ci-report-box { border: 2px solid #38bdf8; border-radius: 12px; padding: 1.2rem 1.4rem; background: #f0f9ff; margin-bottom: 1.2rem; }
.ci-report-title { color: #1d4ed8; font-weight: 600; font-size: 1rem; margin-bottom: 0.8rem; }
.ci-meta-box { border: 2px solid #2563eb; border-radius: 12px; padding: 1.2rem 1.4rem; background: #eff6ff; margin-bottom: 1.2rem; }
.ci-meta-title { color: #1d4ed8; font-weight: 600; font-size: 1rem; margin-bottom: 0.8rem; }
.ci-rhythm-row { display: flex; gap: 0.8rem; margin-bottom: 0.8rem; }
.ci-rhythm-card { flex: 1; background: #fff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 0.7rem 1rem; }
.ci-rhythm-label { font-size: 0.72rem; color: #64748b; font-weight: 500; margin-bottom: 0.2rem; }
.ci-rhythm-value { font-size: 0.95rem; color: #1e40af; font-weight: 600; }
.ci-welcome { background: #fff; border-radius: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); padding: 3.5rem 2rem; text-align: center; margin-top: 2rem; }
.ci-welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.ci-welcome h2 { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #1e3a8a; margin-bottom: 0.6rem; }
.ci-welcome p { color: #64748b; font-size: 0.92rem; line-height: 1.6; max-width: 420px; margin: 0 auto 1.6rem; }
.pred-badge { display: inline-block; padding: 0.22rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin: 0.2rem 0.25rem; }
.pred-positive { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.pred-normal { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.ci-about { background: #fff; border-radius: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); padding: 1.2rem 1.6rem; margin-top: 1.2rem; margin-bottom: 1rem; }
.ci-about-header { display: flex; align-items: center; gap: 0.5rem; font-weight: 600; font-size: 0.95rem; color: #1e3a8a; margin-bottom: 0.8rem; }
.ci-about-icon { background: #2563eb; color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 0.8rem; font-weight: 700; }

div[data-testid="stButton"] > button[kind="primary"] { background: #2563eb !important; border-color: #2563eb !important; }
div[data-testid="stButton"] > button[kind="primary"]:hover { background: #1d4ed8 !important; border-color: #1d4ed8 !important; }
[data-testid="stHorizontalBlock"] > div:first-child { position: sticky; top: 75px; max-height: calc(100vh - 90px); overflow-y: auto; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label, div[data-testid="stFileUploader"] label { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Sticky header ──────────────────────────────────────────────────────────────
logo_b64 = _logo_b64()
st.markdown(f"""
<div class="ci-header">
 <div>
   <div class="ci-logo-row">
     <img src="data:image/png;base64,{logo_b64}" class="ci-logo-img" />
     <span class="ci-logo-text">CardioInsight</span>
   </div>
   <div class="ci-subtitle">AI-Powered Cardiac Rhythm Classification</div>
 </div>
 <span class="ci-hamburger">☰</span>
</div>
<div class="ci-disclaimer">
⚠️ <strong>Research use only.</strong> CardioInsight is not a certified medical device and must not be used as a substitute for clinical diagnosis. Always consult a qualified physician for medical decisions.
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULT_KEYS = [
    "predictions", "report", "signal", "signal_arr", "metadata_arr", "fs",
    "meta", "analysed", "viz_type", "batch_results",
    "stream_signal", "stream_windows", "stream_idx", "stream_running",
    "ig_signal_raw", "results",
]
for _k in DEFAULT_KEYS:
    if _k not in st.session_state:
        if _k == "viz_type":
            st.session_state[_k] = "None"
        elif _k in ("stream_idx",):
            st.session_state[_k] = 0
        elif _k in ("stream_running",):
            st.session_state[_k] = False
        else:
            st.session_state[_k] = None

# ── Shared About block ────────────────────────────────────────────────────────
def _render_about():
    st.markdown("""
 <div class="ci-about">
   <div class="ci-about-header">
     <span class="ci-about-icon">i</span> About CardioInsight
   </div>
 """, unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["CardioInsight", "System function"])
    with tab1:
        st.markdown("CardioInsight is an ECG signal analysis application that detects potential heart rhythm abnormalities. It uses a hybrid CNN-SE-Transformer deep learning model trained on the PTB-XL dataset to classify ECG recordings into five diagnostic categories: Normal, Myocardial Infarction, ST/T-Wave Change, Conduction Disturbance, and Hypertrophy. Model predictions are then passed to a fine-tuned GPT-4o-mini to generate structured clinical interpretation reports.")
    with tab2:
        st.markdown("""Pipeline overview:
- **Input** — Upload a WFDB-format ECG record (`.hea` + `.dat`) and enter patient metadata.
- **Preprocessing** — Signal loaded via `wfdb`; normalised per-lead; heart rate and rhythm estimated from R-peaks.
- **Model inference** — Hybrid CNN-SE-Transformer → multi-label probabilities for 5 classes.
- **Threshold tuning** — Per-class F1-optimal thresholds determine positive labels.
- **LLM Report** — Positive predictions → fine-tuned GPT-4o-mini → structured clinical report.
- **Visualisation** — ECG waveform (all leads), lead contribution, metadata chart, rhythm cards.
- **Streaming** — Continuous signal monitoring with real-time windowed analysis.
- **Batch** — Process multiple ECG records simultaneously.

Configuration: Edit `.env` to set model path, API key, and thresholds.""")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Layout: fixed left | scrollable right ─────────────────────────────────────
left_col, right_col = st.columns([1, 2.4], gap="large")

# ── LEFT COLUMN ───────────────────────────────────────────────────────────────
with left_col:
    with st.container(border=True):
        st.markdown("### 🎛️ Control Panel")

        if DEMO_MODE:
            st.info("🛡️ **Demo Mode Active** — Mock predictions enabled.")

        analysis_mode = st.radio("Analysis Mode", ["Manual Upload", "📁 Batch Upload", "🔁 Live Monitor"], horizontal=True, key="analysis_mode")

        st.markdown("**Patient Information**")
        age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age", key="pat_age")
        sex = st.selectbox("Gender", ["Male", "Female"], key="pat_sex")
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=None, placeholder="Enter height", key="pat_height")
        weight = st.number_input("Weight (kg)", min_value=2, max_value=300, value=None, placeholder="Enter weight", key="pat_weight")

        st.markdown("**Explainability Method**")
        viz_type = st.selectbox(
            "Visualization Type",
            ["None", "Grad-CAM", "Saliency Maps", "Integrated Grads (IG)"],
            key="viz_type"
        )

        st.markdown("---")

        if analysis_mode == "Manual Upload":
            st.markdown("**ECG Signal Upload**")
            uploaded_hea = st.file_uploader("Upload .hea file", type=["hea"], key="manual_hea")
            uploaded_dat = st.file_uploader("Upload .dat file", type=["dat"], key="manual_dat")
            run_btn = st.button("🔍 Analyse ECG", use_container_width=True, type="primary", key="manual_run")

        elif analysis_mode == "📁 Batch Upload":
            st.markdown("**Batch ECG Upload**")
            st.caption("Upload multiple .hea/.dat pairs or a ZIP archive.")
            uploaded_files = st.file_uploader("Select .hea & .dat files", type=["hea", "dat"], accept_multiple_files=True, key="batch_files")
            uploaded_zip = st.file_uploader("Or upload ZIP archive", type=["zip"], key="batch_zip")
            run_batch_btn = st.button("🔍 Analyse Batch", use_container_width=True, type="primary", key="batch_run")

        else:
            st.markdown("**📡 Continuous ECG Monitoring**")
            st.caption("Real-time ingestion → binary pre-filter → auto-routing to CardioInsight AI pipeline.")

            monitor_file = st.file_uploader("Upload continuous .npy signal", type=["npy"], key="monitor_file")
            fs_monitor = st.number_input("Sampling Rate (Hz)", min_value=100, max_value=1000, value=500, key="monitor_fs")

            c1, c2 = st.columns(2)
            with c1:
                start_monitor = st.button("▶ Start Stream", use_container_width=True, type="primary", key="mon_start")
            with c2:
                stop_monitor = st.button("⏹ Stop", use_container_width=True, key="mon_stop")

            if stop_monitor:
                st.session_state["stream_running"] = False
                st.rerun()

            if start_monitor and monitor_file is not None:
                try:
                    from api_bassam import run_streaming_pipeline

                    with st.spinner("Initializing stream pipeline..."):
                        signal = np.load(monitor_file)
                        windows = list(run_streaming_pipeline(
                            signal, fs_monitor,
                            age=(age if age is not None else 50),
                            sex=sex,
                            height=height,
                            weight=weight,
                            window_sec=10.0,
                            overlap_sec=5.0
                        ))

                    st.session_state["stream_signal"] = signal
                    st.session_state["stream_windows"] = windows
                    st.session_state["stream_running"] = True
                    st.session_state["stream_idx"] = 0
                    # monitor_fs is managed by the widget with key="monitor_fs";
                    # no need to write it explicitly (avoids Streamlit key conflict).
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to start stream: {exc}")

# ── RIGHT COLUMN ──────────────────────────────────────────────────────────────
with right_col:
    # ── MANUAL MODE ───────────────────────────────────────────────────────────
    if analysis_mode == "Manual Upload" and run_btn:
        errors = []
        if not uploaded_hea or not uploaded_dat:
            errors.append("Please upload both the .hea and .dat ECG files.")
        if age is None:
            errors.append("Please enter patient age.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Running analysis & AI Interpretation…"):
                try:
                    result = run_full_pipeline(
                        hea_bytes=uploaded_hea.read(), hea_name=uploaded_hea.name,
                        dat_bytes=uploaded_dat.read(), dat_name=uploaded_dat.name,
                        age=age, sex=sex, height=height, weight=weight
                    )

                    st.session_state.update({
                        "predictions": result["predictions"],
                        "report": result["report"],
                        "signal": result["signal"],
                        "signal_arr": result["signal_arr"],
                        "metadata_arr": result["metadata_arr"],
                        "fs": result["fs"],
                        "meta": result["meta"],
                        "analysed": True,
                        "ig_signal_raw": result["ig_signal"],
                        "results": {"meta": {"meta_importance": result["meta"]["meta_importance"],
                                              "lead_importance": result["ig_signal"]}}
                    })
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.code(traceback.format_exc())

    # ── BATCH MODE ────────────────────────────────────────────────────────────
    elif analysis_mode == "📁 Batch Upload" and run_batch_btn:
        records = []

        # Parse multiple files
        if uploaded_files:
            file_map = {}
            for f in uploaded_files:
                stem = Path(f.name).stem
                ext = Path(f.name).suffix.lower()
                file_map.setdefault(stem, {})[ext] = f
            for stem, pair in file_map.items():
                if '.hea' in pair and '.dat' in pair:
                    records.append({
                        "hea_bytes": pair['.hea'].read(), "hea_name": pair['.hea'].name,
                        "dat_bytes": pair['.dat'].read(), "dat_name": pair['.dat'].name,
                        "age": age or 50, "sex": sex, "height": height, "weight": weight,
                    })

        # Parse ZIP
        if uploaded_zip:
            with zipfile.ZipFile(uploaded_zip) as zf:
                names = zf.namelist()
                hea_files = [n for n in names if n.lower().endswith(".hea")]
                for hea_name in hea_files:
                    base = hea_name[:-4]
                    dat_name = base + ".dat"
                    if dat_name in names:
                        records.append({
                            "hea_bytes": zf.read(hea_name), "hea_name": Path(hea_name).name,
                            "dat_bytes": zf.read(dat_name), "dat_name": Path(dat_name).name,
                            "age": age or 50, "sex": sex, "height": height, "weight": weight,
                        })

        if not records:
            st.error("No valid .hea/.dat pairs found.")
        else:
            with st.spinner(f"Processing {len(records)} records…"):
                try:
                    batch_results = run_batch_pipeline(records)
                    st.session_state["batch_results"] = batch_results

                    rows = []
                    for r in batch_results:
                        if r["status"] == "success":
                            preds = ", ".join([name for name, _, is_pos in r["predictions"] if is_pos]) or "NORM"
                            rows.append({"File": r["filename"], "Prediction": preds, "HR": r["meta"]["hr"], "Status": "✅"})
                        else:
                            rows.append({"File": r["filename"], "Prediction": "Error", "HR": "-", "Status": "❌"})

                    st.markdown("### Batch Summary")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    csv = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download CSV", csv, "cardioinsight_batch.csv", "text/csv")

                except Exception as exc:
                    st.error(f"Batch processing failed: {exc}")
                    st.code(traceback.format_exc())

    # ── LIVE MONITOR MODE ─────────────────────────────────────────────────────
    elif analysis_mode == "🔁 Live Monitor":
        # --- Active streaming with pre-computed windows ---
        if st.session_state.get("stream_running") and st.session_state.get("stream_windows"):
            windows = st.session_state["stream_windows"]
            idx = st.session_state.get("stream_idx", 0)
            signal = st.session_state["stream_signal"]
            fs = st.session_state.get("monitor_fs", 500)

            if idx < len(windows):
                result = windows[idx]
                st.session_state["stream_idx"] = idx + 1

                st.markdown(f"### 🔴 Live Analysis — Window {idx + 1}/{len(windows)}")
                st.progress((idx + 1) / len(windows), text=f"{result['start_sec']:.1f}s - {result['end_sec']:.1f}s")

                if result["anomaly_flag"]:
                    st.error(f"🚨 Anomaly detected at {result['start_sec']:.1f}s - {result['end_sec']:.1f}s! (Binary confidence: {result.get('binary_conf', 0):.2%})")
                else:
                    st.success(f"✅ Normal rhythm at {result['start_sec']:.1f}s - {result['end_sec']:.1f}s")

                badges = " ".join(
                    f'<span class="pred-badge {"pred-positive" if is_pos else "pred-normal"}">{name} — {prob:.0%}</span>'
                    for name, prob, is_pos in result["predictions"]
                )
                st.markdown(f"**Detailed Predictions:** {badges}", unsafe_allow_html=True)

                # Mini ECG of this window
                fig = plot_ecg_signal(signal[result["start_sample"]:result["end_sample"]], fs=fs, duration_s=10.0)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                plt.close(fig)
                st.image(buf, use_container_width=True)

                # History table
                if idx > 0:
                    hist_df = pd.DataFrame([
                        {"Window": i+1, "Time": f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s",
                         "HR": w["hr"], "Rhythm": w["heart_rhythm"], "Anomaly": "🚨" if w["anomaly_flag"] else "✅"}
                        for i, w in enumerate(windows[:idx+1])
                    ])
                    st.markdown("**Analysis History**")
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)

                time.sleep(0.3)
                st.rerun()
            else:
                st.success("✅ Streaming analysis complete.")
                st.session_state["stream_running"] = False

                df = pd.DataFrame([
                    {"Window": w["window_idx"]+1, "Time": f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s",
                     "HR": w["hr"], "Rhythm": w["heart_rhythm"], "Anomaly": "Yes" if w["anomaly_flag"] else "No"}
                    for w in windows
                ])
                st.markdown("### Final Stream Report")
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Stream Report", csv, "stream_report.csv", "text/csv")

        # --- Welcome message when not streaming ---
        elif not st.session_state.get("stream_running"):
            st.markdown("""
             <div class="ci-welcome">
               <div class="ci-welcome-icon">📡</div>
               <h2>Live ECG Monitor</h2>
               <p>Upload a continuous .npy signal and start monitoring to analyze sliding windows in real time.</p>
             </div>
            """, unsafe_allow_html=True)

    # ── RESULTS DISPLAY (Manual mode) ─────────────────────────────────────────
    if analysis_mode == "Manual Upload" and st.session_state.get("analysed"):
        preds = st.session_state["predictions"]
        sig = st.session_state["signal"]
        fs = st.session_state.get("fs", 500)
        meta = st.session_state["meta"]
        report = st.session_state["report"]
        viz_type = st.session_state.get("viz_type", "None")

        badges = " ".join(
            f'<span class="pred-badge {"pred-positive" if is_pos else "pred-normal"}">{name} — {prob:.0%}</span>'
            for name, prob, is_pos in preds
        )
        st.markdown(f"**Predictions:** {badges}", unsafe_allow_html=True)
        st.markdown(" ")

        st.markdown(f'<div class="ci-meta-box"><div class="ci-meta-title">🔬 Interpretation & Visualization — {viz_type}</div>', unsafe_allow_html=True)

        current_sig = st.session_state.get("signal")

        if viz_type == "None":
            if current_sig is not None:
                n_leads_actual = current_sig.shape[1] if current_sig.ndim > 1 else 1
                fig_ecg = plot_ecg_signal(current_sig, fs=fs, duration_s=ECG_DISPLAY_DURATION)
                buf = io.BytesIO()
                fig_ecg.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                ecg_b64 = base64.b64encode(buf.read()).decode()
                plt.close(fig_ecg)
                st.markdown(
                    f'<div class="ci-ecg-box"><div class="ci-ecg-title">📈 ECG Signal (Raw Waveform)'
                    f'<span style="font-size:0.75rem;font-weight:400;color:#64748b;margin-left:0.5rem;">{n_leads_actual} leads · scroll to view all</span></div>'
                    f'<div style="height:420px;overflow-y:auto;border-radius:8px;"><img src="data:image/png;base64,{ecg_b64}" style="width:100%;display:block;"></div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("No signal data found to display.")
        else:
            with st.spinner(f"Computing {viz_type}…"):
                try:
                    plt.close('all')
                    xai_figs = run_xai(
                        method=viz_type, model=get_model(),
                        signal_raw=st.session_state.get("signal"),
                        signal_arr=st.session_state.get("signal_arr"),
                        metadata_arr=st.session_state.get("metadata_arr"),
                        predictions=st.session_state.get("predictions")
                    )
                    if xai_figs:
                        fig_to_plot = list(xai_figs.values())[0]
                        buf = io.BytesIO()
                        fig_to_plot.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                        buf.seek(0)
                        xai_b64 = base64.b64encode(buf.read()).decode()
                        plt.close(fig_to_plot)
                        st.markdown(
                            f'<div class="ci-ecg-box"><div class="ci-ecg-title">🔬 Interpretation: {viz_type}</div>'
                            f'<div style="height:420px;overflow-y:auto;border-radius:8px; background: white;"><img src="data:image/png;base64,{xai_b64}" style="width:100%; display:block;"></div></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("Generating visualization...")
                except Exception as xai_exc:
                    st.error(f"❌ XAI Error: {xai_exc}")
                    st.caption("💡 Tip: Switch to 'None' to view raw ECG while debugging.")

        # LLM Report
        st.markdown('<div class="ci-report-box"> <div class="ci-report-title">📋 LLM Report </div>', unsafe_allow_html=True)
        st.markdown(report)
        st.markdown("</div>", unsafe_allow_html=True)

        report_bytes = report.encode('utf-8')
        st.download_button("⬇️ Download Report", report_bytes, "cardioinsight_report.md", "text/markdown")

        # Metadata Outputs
        st.markdown('<div class="ci-meta-box"> <div class="ci-meta-title">📊 Metadata Outputs </div>', unsafe_allow_html=True)
        st.markdown(f"""
         <div class="ci-rhythm-row">
         <div class="ci-rhythm-card"><div class="ci-rhythm-label">Heart Rhythm</div><div class="ci-rhythm-value">{meta.get("heart_rhythm", "—")}</div></div>
         <div class="ci-rhythm-card"><div class="ci-rhythm-label">Rhythm Regularity</div><div class="ci-rhythm-value">{meta.get("rhythm_regularity", "—")}</div></div>
         </div>
        """, unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        ig_results = st.session_state["results"]["meta"]
        with mc1:
            try:
                fig_leads = plot_ig_leads_final(st.session_state["ig_signal_raw"])
                st.pyplot(fig_leads, use_container_width=True)
                plt.close(fig_leads)
            except Exception as exc:
                st.warning(f"Lead importance plot failed: {exc}")
        with mc2:
            try:
                fig_meta = plot_ig_metadata_final(ig_results["meta_importance"])
                st.pyplot(fig_meta, use_container_width=True)
                plt.close(fig_meta)
            except Exception as exc:
                st.warning(f"Metadata plot failed: {exc}")
            st.dataframe(pd.DataFrame({
                "Feature": ["Age", "Sex", "Heart Rate (bpm)"],
                "Value": [str(meta["age"]), meta["sex"], f"{meta['hr']} bpm"]
            }), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        _render_about()

    elif analysis_mode == "Manual Upload" and not st.session_state.get("analysed"):
        st.markdown("""
         <div class="ci-welcome">
           <div class="ci-welcome-icon">🩺</div>
           <h2>Welcome to ECG Analysis</h2>
           <p>Upload an ECG signal file to get started. The system will visualise the waveform and provide classification of potential cardiac conditions.</p>
         </div>
        """, unsafe_allow_html=True)
        _render_about()
