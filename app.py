import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# ----------------------------------------------------
# 1. Audio Aufnahme
# ----------------------------------------------------
audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]  # Mono channel
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 2. Signal im Zeitbereich & Window Range
    # ----------------------------------------------------
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
        margin=dict(l=20, r=20, t=30, b=20), height=220,
        dragmode="zoom",
        uirevision="time_chart_state"  # Preserves time-domain zoom state
    )
    st.plotly_chart(fig_time, use_container_width=True, key="time_domain_chart")

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

    # ----------------------------------------------------
    # 3. FFT Berechnung & Peak-Eingabe
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
        st.caption("💡 Trage unten bis zu 10 Peak-Frequenzen ein. Die gezoomte Ansicht bleibt beim Ändern erhalten.")

        # 10 manual input boxes
        with st.expander("🎯 bis zu 10 Peak-Frequenzen manuell eingeben (Hz)", expanded=True):
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

        # Process entered frequencies (Snap to local peak max within 50Hz)
        snapped_peaks = []
        df_max = 50
        for u_freq in user_freqs:
            idx_search = np.abs(f - u_freq) < df_max
            if np.any(idx_search):
                exact_peak = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak = float(u_freq)
            snapped_peaks.append(exact_peak)

        # Build FFT Plot
        valid_mask = f <= 5000
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=f[valid_mask], y=P[valid_mask], mode="lines", name="|FFT|"))

        # Add vertical red lines for each entered peak
        for peak_f in snapped_peaks:
            fig_fft.add_vline(
                x=peak_f, 
                line_width=2, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"{peak_f:.1f} Hz",
                annotation_position="top right"
            )

        fig_fft.update_layout(
            xaxis_title="Frequenz [Hz]", yaxis_title="|FFT|",
            margin=dict(l=20, r=20, t=30, b=20), height=380,
            dragmode="zoom",
            uirevision="fft_chart_state"  # Preserves FFT zoom state during reruns
        )
        # Static key ensures Streamlit doesn't recreate the canvas element
        st.plotly_chart(fig_fft, use_container_width=True, key="fft_spectrum_chart")

        # ----------------------------------------------------
        # 4. Koeffizienten & Synthese Audio (wenn ≥ 1 Peak)
        # ----------------------------------------------------
        if len(snapped_peaks) > 0:
            st.subheader("3. Fourierkoeffizienten & Audio-Synthese")
            col_left, col_right = st.columns([1, 1])

            # Calculate Fourier coefficients
            df_max = 100
            a_coeffs, b_coeffs = [], []

            for sf_freq in snapped_peaks:
                idx = np.abs(f - sf_freq) < df_max
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
