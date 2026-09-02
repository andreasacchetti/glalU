import io
import numpy as np
import plotly.graph_objects as go
import scipy.fft as fft
import soundfile as sf
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse (Interactive)")

# Initialize session state for persistent data across reruns
if "selected_peaks" not in st.session_state:
    st.session_state.selected_peaks = []
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "fs" not in st.session_state:
    st.session_state.fs = None

# Audio Recording Input
audio_file = st.audio_input("Record your audio")

# Store new recording into session state if available
if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]  # Mono channel
    st.session_state.audio_data = data
    st.session_state.fs = fs
    # Reset peaks on fresh recording
    st.session_state.selected_peaks = []

# Continue only if we have stored audio data
if st.session_state.audio_data is not None:
    data = st.session_state.audio_data
    fs = st.session_state.fs
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 1. Signal im Zeitbereich & Window Range
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich & FFT-Bereich")

    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01,
        key="time_slider"
    )

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]

    # Time domain plot
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(x=t, y=data, mode="lines", name="Audio Signal"))
    fig_signal.add_vrect(
        x0=t_min, x1=t_max, fillcolor="LightSalmon", opacity=0.3,
        layer="below", line_width=0, annotation_text="FFT Window"
    )
    fig_signal.update_layout(
        xaxis_title="Zeit [s]", yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20), height=250
    )
    st.plotly_chart(fig_signal, use_container_width=True)

    # ----------------------------------------------------
    # 2. FFT Calculation & Interactive Peak Selection
    # ----------------------------------------------------
    if len(xfft) > 0:
        L = len(xfft)
        m = int(2 ** np.ceil(np.log2(L)))
        Z = fft.fft(xfft, m)

        ReZ = np.real(Z[: L // 2 + 1])
        ImZ = np.imag(Z[: L // 2 + 1])
        P = np.abs(Z / L)[: L // 2 + 1]
        P[1:-1] *= 2
        f = np.arange(len(P)) * fs / m

        st.subheader("2. FFT Spektrum — Klicke auf den Plot, um Peaks auszuwählen")

        # Interactive Plotly Spectrum
        fig_fft = go.Figure()
        fig_fft.add_trace(
            go.Scatter(
                x=f, y=P, mode="lines", name="|FFT|",
                hovertemplate="Frequenz: %{x:.2f} Hz<br>Betrag: %{y:.5f}<extra></extra>"
            )
        )

        # Draw existing markers for selected peaks
        if st.session_state.selected_peaks:
            peak_x = st.session_state.selected_peaks
            peak_y = [P[np.argmin(np.abs(f - px))] for px in peak_x]
            fig_fft.add_trace(
                go.Scatter(
                    x=peak_x, y=peak_y, mode="markers",
                    marker=dict(color="red", size=10, symbol="x"),
                    name="Ausgewählte Peaks"
                )
            )

        fig_fft.update_layout(
            xaxis_title="Frequenz [Hz]", yaxis_title="|FFT|",
            margin=dict(l=20, r=20, t=30, b=20), height=450,
            xaxis=dict(range=[0, 5000])  # Default view 0-5kHz
        )

        # Capture clicks on the chart
        selected_points = plotly_events(fig_fft, click_event=True, key="fft_plot")

        if selected_points:
            clicked_freq = selected_points[0]["x"]
            
            # Snap to highest peak within 50 Hz of the click
            df_max = 50  
            idx_search = np.abs(f - clicked_freq) < df_max
            if np.any(idx_search):
                local_f = f[idx_search]
                local_P = P[idx_search]
                exact_peak_f = float(local_f[np.argmax(local_P)])
            else:
                exact_peak_f = float(clicked_freq)

            # Prevent double-clicking the same peak
            if not any(np.isclose(exact_peak_f, existing, atol=1.0) for existing in st.session_state.selected_peaks):
                st.session_state.selected_peaks.append(exact_peak_f)
                st.rerun()

        # ----------------------------------------------------
        # 3. Fourier Coefficients Output
        # ----------------------------------------------------
        st.subheader("3. Ausgewählte Peaks & Fourierkoeffizienten")

        if st.button("🗑️ Peaks zurücksetzen"):
            st.session_state.selected_peaks = []
            st.rerun()

        if st.session_state.selected_peaks:
            df_max = 100
            selected_freqs = sorted(st.session_state.selected_peaks)
            a_coeffs = []
            b_coeffs = []

            for sf_freq in selected_freqs:
                idx = np.abs(f - sf_freq) < df_max
                if np.any(idx):
                    Re_plus = np.max(ReZ[idx])
                    Re_minus = np.min(ReZ[idx])
                    Im_plus = np.max(ImZ[idx])
                    Im_minus = np.min(ImZ[idx])

                    a_coeffs.append(Re_plus + Re_minus)
                    b_coeffs.append(-(Im_plus + Im_minus))
                else:
                    a_coeffs.append(0.0)
                    b_coeffs.append(0.0)

            # Build text export format
            export_str = "f(Hz)\ta_k\tb_k\n"
            for f_val, a_val, b_val in zip(selected_freqs, a_coeffs, b_coeffs):
                export_str += f"{f_val:.2f}\t{a_val:.5f}\t{b_val:.5f}\n"

            st.text_area("Fourierkoeffizienten", export_str, height=150)

            st.download_button(
                label="💾 Fourierdaten exportieren (.txt)",
                data=export_str,
                file_name="Fourierkoeffizienten.txt",
                mime="text/plain",
            )
