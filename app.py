import io
import numpy as np
import pandas as pd
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import altair as alt

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# 1. Audio Aufnahme
audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]  # Mono
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 1. Time Domain Signal
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich")
    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01
    )

    df_time = pd.DataFrame({"Zeit [s]": t, "Amplitude": data})
    
    # Downsample time plot if audio is long to ensure fast rendering
    if len(df_time) > 10000:
        step = len(df_time) // 10000
        df_time_plot = df_time.iloc[::step]
    else:
        df_time_plot = df_time

    chart_time = alt.Chart(df_time_plot).mark_line().encode(
        x=alt.X("Zeit [s]:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("Amplitude:Q")
    ).properties(height=200).interactive()

    st.altair_chart(chart_time, use_container_width=True)

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

    # ----------------------------------------------------
    # 2. FFT Spectrum & Interactive Peak Selection
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

        st.subheader("2. FFT Spektrum & Peaks")
        st.caption("💡 **Klicke direkt auf Peaks im Chart**, um Frequenzen auszuwählen, oder gib sie manuell unten ein.")

        # Prepare FFT DataFrame (capped at 5kHz)
        valid_mask = f <= 5000
        df_fft = pd.DataFrame({
            "Frequenz [Hz]": f[valid_mask],
            "|FFT|": P[valid_mask]
        })

        # Manual Number Inputs for precise tweaking
        with st.expander("🎯 Peak-Frequenzen manuell eingeben (Hz)", expanded=True):
            cols = st.columns(5)
            user_freqs = []
            for i in range(10):
                with cols[i % 5]:
                    val = st.number_input(
                        f"Peak {i+1} (Hz):", 
                        min_value=0.0, 
                        max_value=float(fs/2), 
                        value=0.0, 
                        step=10.0,
                        key=f"peak_in_{i}"
                    )
                    if val > 0:
                        user_freqs.append(val)

        # Snap manual input frequencies to nearest local maximum within 50Hz
        snapped_peaks = []
        df_max = 50
        for u_freq in user_freqs:
            idx_search = np.abs(f - u_freq) < df_max
            if np.any(idx_search):
                exact_peak = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak = float(u_freq)
            snapped_peaks.append(exact_peak)

        # Build Interactive Altair FFT Plot
        brush_selection = alt.selection_point(on="click", nearest=True, fields=["Frequenz [Hz]"])

        base_spectrum = alt.Chart(df_fft).mark_line(color="#1f77b4").encode(
            x=alt.X("Frequenz [Hz]:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("|FFT|:Q"),
            tooltip=["Frequenz [Hz]", "|FFT|"]
        ).properties(height=350)

        # Clickable hit-targets on spectrum
        points = alt.Chart(df_fft).mark_point(size=30, opacity=0).encode(
            x="Frequenz [Hz]:Q",
            y="|FFT|:Q"
        ).add_params(brush_selection)

        # Red dashed lines for entered peaks
        if len(snapped_peaks) > 0:
            df_peaks = pd.DataFrame({"Peak": snapped_peaks})
            peak_lines = alt.Chart(df_peaks).mark_rule(
                color="red", 
                strokeDash=[4, 4], 
                strokeWidth=2
            ).encode(
                x="Peak:Q"
            )
            fft_chart = (base_spectrum + points + peak_lines).interactive()
        else:
            fft_chart = (base_spectrum + points).interactive()

        # Render Chart and catch click events
        chart_event = st.altair_chart(fft_chart, use_container_width=True, on_select="rerun")

        # Extract clicked frequency directly from chart interaction
        if chart_event and "selection" in chart_event and "param_1" in chart_event["selection"]:
            clicked_data = chart_event["selection"]["param_1"]
            if clicked_data and len(clicked_data) > 0:
                clicked_freq = clicked_data[0].get("Frequenz [Hz]")
                if clicked_freq and clicked_freq not in snapped_peaks:
                    snapped_peaks.append(clicked_freq)

        # ----------------------------------------------------
        # 3. Fourier Coefficients & Audio Synthesis
        # ----------------------------------------------------
        if len(snapped_peaks) > 0:
            st.subheader("3. Fourierkoeffizienten & Audio-Synthese")
            col_left, col_right = st.columns([1, 1])

            a_coeffs, b_coeffs = [], []
            for sf_freq in snapped_peaks:
                idx = np.abs(f - sf_freq) < 100
                if np.any(idx):
                    a_coeffs.append(float(np.max(ReZ[idx]) + np.min(ReZ[idx])))
                    b_coeffs.append(float(-(np.max(ImZ[idx]) + np.min(ImZ[idx]))))
                else:
                    a_coeffs.append(0.0)
                    b_coeffs.append(0.0)

            with col_left:
                export_str = "f(Hz)\ta_k\tb_k\n"
                for f_val, a_val, b_val in zip(snapped_peaks, a_coeffs, b_coeffs):
                    export_str += f"{f_val:.2f}\t{a_val:.5f}\t{b_val:.5f}\n"

                st.text_area("Berechnete Koeffizienten", export_str, height=160)
                st.download_button(
                    "💾 Fourierdaten exportieren (.txt)", 
                    export_str, 
                    "Fourierkoeffizienten.txt", 
                    "text/plain"
                )

            with col_right:
                st.markdown("**🔊 Audio-Wiedergabe**")
                st.audio(audio_file.getvalue(), format="audio/wav")
                st.caption("Originale Audioaufnahme")

                xsynth = np.zeros_like(tfft)
                for i in range(len(snapped_peaks)):
                    xsynth += a_coeffs[i] * np.cos(2 * np.pi * snapped_peaks[i] * tfft)
                    xsynth += b_coeffs[i] * np.sin(2 * np.pi * snapped_peaks[i] * tfft)

                if np.max(np.abs(xsynth)) != 0:
                    xsynth *= (np.max(np.abs(xfft)) / np.max(np.abs(xsynth)))

                synth_buffer = io.BytesIO()
                sf.write(synth_buffer, xsynth, fs, format="WAV")
                st.audio(synth_buffer.getvalue(), format="audio/wav")
                st.caption(f"Synthetisiertes Signal ({len(snapped_peaks)} Peaks)")
