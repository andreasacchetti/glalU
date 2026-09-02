import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse (Plotly Box-Zoom)")

if "selected_peaks" not in st.session_state:
    st.session_state.selected_peaks = []
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    st.session_state.audio_bytes = audio_file.getvalue()
    st.session_state.selected_peaks = []

if st.session_state.audio_bytes is not None:
    data, fs = sf.read(io.BytesIO(st.session_state.audio_bytes))
    if len(data.shape) > 1:
        data = data[:, 0]
    t = np.arange(len(data)) / fs

    # 1. Signal im Zeitbereich
    st.subheader("1. Signal im Zeitbereich & FFT-Bereich")
    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01
    )

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t, y=data, mode="lines", name="Signal"))
    fig_time.add_vrect(
        x0=t_min, x1=t_max, fillcolor="orange", opacity=0.3,
        layer="below", line_width=0, annotation_text="FFT Window"
    )
    fig_time.update_layout(
        xaxis_title="Zeit [s]", yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20), height=250,
        dragmode="zoom"  # Box zoom enabled by default
    )
    st.plotly_chart(fig_time, use_container_width=True)

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]

    # 2. FFT Calculation & Interactive Peak Selection
    if len(xfft) > 0:
        L = len(xfft)
        m = int(2 ** np.ceil(np.log2(L)))
        Z = fft.fft(xfft, m)

        ReZ = np.real(Z[: L // 2 + 1])
        ImZ = np.imag(Z[: L // 2 + 1])
        P = np.abs(Z / L)[: L // 2 + 1]
        P[1:-1] *= 2
        f = np.arange(len(P)) * fs / m

        st.subheader("2. FFT Spektrum — Nutze die Plotly-Werkzeuge oben rechts für Box-Zoom")

        fig_fft = go.Figure()
        fig_fft.add_trace(
            go.Scatter(x=f[f <= 5000], y=P[f <= 5000], mode="lines", name="|FFT|")
        )

        # Plot selected peak markers
        if st.session_state.selected_peaks:
            p_x = st.session_state.selected_peaks
            p_y = [P[np.argmin(np.abs(f - px))] for px in p_x]
            fig_fft.add_trace(
                go.Scatter(x=p_x, y=p_y, mode="markers", marker=dict(color="red", size=10, symbol="x"))
            )

        fig_fft.update_layout(
            xaxis_title="Frequenz [Hz]", yaxis_title="|FFT|",
            margin=dict(l=20, r=20, t=30, b=20), height=400,
            dragmode="zoom",          # Enables click-and-drag box zoom
            clickmode="event+select"  # Enables single-click point selection
        )
        # Streamlit 1.35+ Native Plotly Event Listener
        plotly_event = st.plotly_chart(
            fig_fft, 
            use_container_width=True, 
            on_select="rerun", 
            selection_mode="points"
        )

        # Handle point selection
        if plotly_event and "selection" in plotly_event and plotly_event["selection"]["points"]:
            clicked_freq = plotly_event["selection"]["points"][0]["x"]
            
            # Snap to local peak
            df_max = 50
            idx_search = np.abs(f - clicked_freq) < df_max
            if np.any(idx_search):
                exact_peak_f = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak_f = float(clicked_freq)

            if exact_peak_f not in st.session_state.selected_peaks:
                st.session_state.selected_peaks.append(exact_peak_f)
                st.rerun()

        # 3. Fourier Coefficients Output
        st.subheader("3. Ausgewählte Peaks & Fourierkoeffizienten")

        if st.button("🗑️ Peaks zurücksetzen"):
            st.session_state.selected_peaks = []
            st.rerun()

        if st.session_state.selected_peaks:
            df_max = 100
            selected_freqs = sorted(st.session_state.selected_peaks)
            a_coeffs, b_coeffs = [], []

            for sf_freq in selected_freqs:
                idx = np.abs(f - sf_freq) < df_max
                if np.any(idx):
                    a_coeffs.append(np.max(ReZ[idx]) + np.min(ReZ[idx]))
                    b_coeffs.append(-(np.max(ImZ[idx]) + np.min(ImZ[idx])))
                else:
                    a_coeffs.append(0.0)
                    b_coeffs.append(0.0)

            export_str = "f(Hz)\ta_k\tb_k\n"
            for f_val, a_val, b_val in zip(selected_freqs, a_coeffs, b_coeffs):
                export_str += f"{f_val:.2f}\t{a_val:.5f}\t{b_val:.5f}\n"

            st.text_area("Fourierkoeffizienten", export_str, height=150)
            st.download_button("💾 Fourierdaten exportieren (.txt)", export_str, "Fourierkoeffizienten.txt", "text/plain")
