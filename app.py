import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse (Interactive)")

# Initialize session state variables
if "selected_peaks" not in st.session_state:
    st.session_state.selected_peaks = []
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# Audio Recording Input
audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    st.session_state.audio_bytes = audio_file.getvalue()
    st.session_state.selected_peaks = []  # Reset peaks on new recording

if st.session_state.audio_bytes is not None:
    data, fs = sf.read(io.BytesIO(st.session_state.audio_bytes))
    if len(data.shape) > 1:
        data = data[:, 0]  # Mono channel
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 1. Signal im Zeitbereich & Window Range
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich & FFT-Bereich")
    st.caption("💡 **Bedienung Plot:** Scrollen = Zoomen | Klicken & Ziehen = Verschieben (Pan)")

    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01
    )

    # Downsample slightly if audio is long for smooth rendering
    step_time = max(1, len(data) // 5000)
    df_time = pd.DataFrame({
        "Zeit": t[::step_time],
        "Amplitude": data[::step_time]
    })

    # Time Domain Plot via Altair
    time_chart = alt.Chart(df_time).mark_line(color="#1f77b4").encode(
        x=alt.X("Zeit:Q", title="Zeit [s]"),
        y=alt.Y("Amplitude:Q", title="Amplitude")
    ).properties(height=250)

    # Highlight box for the FFT Window
    window_rect = alt.Chart(pd.DataFrame({
        "t_min": [t_min],
        "t_max": [t_max]
    })).mark_rect(opacity=0.3, color="orange").encode(
        x="t_min:Q",
        x2="t_max:Q"
    )

    # .interactive() enables scroll zoom & drag pan
    st.altair_chart((time_chart + window_rect).interactive(), use_container_width=True)

    # Slice the selected region for FFT calculation
    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]

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

        # Filter plot up to 5kHz for default view
        valid_idx = f <= 5000
        step_fft = max(1, np.sum(valid_idx) // 4000)
        
        df_fft = pd.DataFrame({
            "Frequenz": f[valid_idx][::step_fft],
            "Betrag": P[valid_idx][::step_fft]
        })

        st.subheader("2. FFT Spektrum — Klicke auf den Plot, um Peaks auszuwählen")

        # Clickable FFT Chart setup
        selection = alt.selection_point(fields=["Frequenz"], nearest=True, on="click")

        fft_line = alt.Chart(df_fft).mark_line().encode(
            x=alt.X("Frequenz:Q", title="Frequenz [Hz]"),
            y=alt.Y("Betrag:Q", title="|FFT|")
        ).properties(height=350)

        fft_points = alt.Chart(df_fft).mark_circle(size=60).encode(
            x="Frequenz:Q",
            y="Betrag:Q",
            color=alt.condition(selection, alt.value("red"), alt.value("transparent")),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.0))
        ).add_params(selection)

        # .interactive() enables scroll zoom & drag pan on FFT spectrum
        fft_chart_interactive = (fft_line + fft_points).interactive()

        event_data = st.altair_chart(fft_chart_interactive, on_select="rerun", use_container_width=True)

        # Handle plot click
        if event_data and "selection" in event_data and event_data["selection"]:
            points_selected = event_data["selection"].get("param_1", [])
            if points_selected:
                clicked_freq = points_selected[0]["Frequenz"]
                
                # Snap to highest local peak in a 50Hz vicinity
                df_max = 50
                idx_search = np.abs(f - clicked_freq) < df_max
                if np.any(idx_search):
                    local_f = f[idx_search]
                    local_P = P[idx_search]
                    exact_peak_f = float(local_f[np.argmax(local_P)])
                else:
                    exact_peak_f = float(clicked_freq)

                if exact_peak_f not in st.session_state.selected_peaks:
                    st.session_state.selected_peaks.append(exact_peak_f)

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
