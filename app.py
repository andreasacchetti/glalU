import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st

from bokeh.plotting import figure
from bokeh.models import Span, Label

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# 1. Audio Aufnahme
audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]
    t = np.arange(len(data)) / fs

    # 2. Time-Domain Signal
    st.subheader("1. Signal im Zeitbereich & FFT-Bereich")
    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01
    )

    p_time = figure(height=220, sizing_mode="stretch_width", x_axis_label="Zeit [s]", y_axis_label="Amplitude")
    p_time.line(t, data, color="#1f77b4")
    
    # Orange FFT window box
    from bokeh.models import BoxAnnotation
    p_time.add_layout(BoxAnnotation(left=t_min, right=t_max, fill_color="orange", fill_alpha=0.3))
    st.bokeh_chart(p_time, use_container_width=True)

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

    # 3. FFT Spectrum
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

        # 10 Peak Inputs
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

        # Snap to local peak
        snapped_peaks = []
        df_max = 50
        for u_freq in user_freqs:
            idx_search = np.abs(f - u_freq) < df_max
            if np.any(idx_search):
                exact_peak = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak = float(u_freq)
            snapped_peaks.append(exact_peak)

        # Build Bokeh FFT Chart
        valid_mask = f <= 5000
        p_fft = figure(
            height=380, 
            sizing_mode="stretch_width", 
            x_axis_label="Frequenz [Hz]", 
            y_axis_label="|FFT|",
            tools="pan,wheel_zoom,box_zoom,reset" # Standard zoom/pan tools
        )
        p_fft.line(f[valid_mask], P[valid_mask], color="#1f77b4", line_width=1.5)

        # Add vertical red lines for each peak
        for peak_f in snapped_peaks:
            vh_line = Span(location=peak_f, dimension='height', line_color='red', line_dash='dashed', line_width=2)
            p_fft.add_layout(vh_line)
            p_fft.add_layout(Label(x=peak_f, y=max(P[valid_mask])*0.8, text=f"{peak_f:.1f}Hz", text_color="red"))

        st.bokeh_chart(p_fft, use_container_width=True)

        # 4. Coefficients & Synthesis
        if len(snapped_peaks) > 0:
            st.subheader("3. Fourierkoeffizienten & Audio-Synthese")
            col_left, col_right = st.columns([1, 1])

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
                st.download_button("💾 Fourierdaten exportieren (.txt)", export_str, "Fourierkoeffizienten.txt", "text/plain")

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
