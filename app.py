import io
import numpy as np
import pandas as pd
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import altair as alt

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 1. Signal im Zeitbereich
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich")
    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01
    )

    # Downsample time domain data for stable rendering
    max_pts = 5000
    step = max(1, len(data) // max_pts)
    df_time = pd.DataFrame({
        "Zeit": np.round(t[::step], 4),
        "Amplitude": np.round(data[::step], 4)
    })

    # Line chart for full time signal
    line_time = alt.Chart(df_time).mark_line(color="#1f77b4").encode(
        x=alt.X("Zeit:Q", title="Zeit [s]"),
        y=alt.Y("Amplitude:Q", title="Amplitude")
    )

    # Highlight rectangle for selected time slice
    df_window = pd.DataFrame([{"t0": t_min, "t1": t_max}])
    highlight_window = alt.Chart(df_window).mark_rect(
        color="orange", opacity=0.3
    ).encode(
        x="t0:Q",
        x2="t1:Q"
    )

    chart_time = (line_time + highlight_window).properties(height=220).interactive()
    st.altair_chart(chart_time, use_container_width=True)

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

    # ----------------------------------------------------
    # 2. FFT Spektrum
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

        st.subheader("2. FFT Spektrum")
        st.caption("💡 **Zoom & Pan:** Nutze Mausrad/Touchpad. Für Box-Zoom **Shift-Taste gedrückt halten und Rechteck ziehen**.")

        valid_mask = f <= 5000
        f_sub = f[valid_mask]
        P_sub = P[valid_mask]

        step_fft = max(1, len(f_sub) // max_pts)
        df_fft = pd.DataFrame({
            "Frequenz": np.round(f_sub[::step_fft], 2),
            "Magnitude": np.round(P_sub[::step_fft], 5)
        })

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

        snapped_peaks = []
        df_max = 50
        for u_freq in user_freqs:
            idx_search = np.abs(f - u_freq) < df_max
            if np.any(idx_search):
                exact_peak = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak = float(u_freq)
            snapped_peaks.append(exact_peak)

        # Main spectrum line
        base_fft = alt.Chart(df_fft).mark_line(color="#1f77b4").encode(
            x=alt.X("Frequenz:Q", title="Frequenz [Hz]"),
            y=alt.Y("Magnitude:Q", title="|FFT|"),
            tooltip=["Frequenz", "Magnitude"]
        )

        # Red vertical dashed rules for entered peaks
        if len(snapped_peaks) > 0:
            df_peaks = pd.DataFrame({"Peak": snapped_peaks})
            peak_rules = alt.Chart(df_peaks).mark_rule(
                color="red", 
                strokeDash=[4, 4], 
                strokeWidth=2
            ).encode(x="Peak:Q")
            final_fft = (base_fft + peak_rules).properties(height=380).interactive(bind_y=True)
        else:
            final_fft = base_fft.properties(height=380).interactive(bind_y=True)

        st.altair_chart(final_fft, use_container_width=True)

        # ----------------------------------------------------
        # 3. Fourierkoeffizienten & Synthese
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
                st.download_button("💾 Fourierdaten exportieren (.txt)", export_str, "Fourierkoeffizienten.txt", "text/plain")

            with col_right:
                st.audio(audio_file.getvalue(), format="audio/wav")
                xsynth = np.zeros_like(tfft)
                for i in range(len(snapped_peaks)):
                    xsynth += a_coeffs[i] * np.cos(2 * np.pi * snapped_peaks[i] * tfft)
                    xsynth += b_coeffs[i] * np.sin(2 * np.pi * snapped_peaks[i] * tfft)

                if np.max(np.abs(xsynth)) != 0:
                    xsynth *= (np.max(np.abs(xfft)) / np.max(np.abs(xsynth)))

                synth_buffer = io.BytesIO()
                sf.write(synth_buffer, xsynth, fs, format="WAV")
                st.audio(synth_buffer.getvalue(), format="audio/wav")
