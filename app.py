import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# Initialize persistent session state
if "selected_peaks" not in st.session_state:
    st.session_state.selected_peaks = []
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    st.session_state.audio_bytes = audio_file.getvalue()
    st.session_state.selected_peaks = []  # Reset on new recording

if st.session_state.audio_bytes is not None:
    data, fs = sf.read(io.BytesIO(st.session_state.audio_bytes))
    if len(data.shape) > 1:
        data = data[:, 0]  # Mono channel
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 1. Signal im Zeitbereich
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
        margin=dict(l=20, r=20, t=30, b=20), height=230,
        dragmode="zoom"
    )
    st.plotly_chart(fig_time, use_container_width=True)

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

    # ----------------------------------------------------
    # 2. FFT Calculation & Working Click Peak Selection
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

        st.subheader("2. FFT Spektrum — Klicke direkt auf eine Spitze, um Peaks auszuwählen")
        st.caption("💡 Nutze Drag-to-Box-Zoom zum Vergrößern. Klicke direkt auf die Linie/Spitze zum Auswählen.")

        valid_mask = f <= 5000
        f_sub = f[valid_mask]
        P_sub = P[valid_mask]

        fig_fft = go.Figure()
        # mode='lines+markers' with tiny invisible markers makes individual points clickable
        fig_fft.add_trace(
            go.Scatter(
                x=f_sub, y=P_sub, 
                mode="lines+markers", 
                marker=dict(size=4, opacity=0),
                name="|FFT|"
            )
        )

        # Draw selected peak markers
        if st.session_state.selected_peaks:
            p_x = st.session_state.selected_peaks
            p_y = [P[np.argmin(np.abs(f - px))] for px in p_x]
            fig_fft.add_trace(
                go.Scatter(
                    x=p_x, y=p_y, 
                    mode="markers", 
                    marker=dict(color="red", size=12, symbol="x"),
                    name="Ausgewählte Peaks"
                )
            )

        fig_fft.update_layout(
            xaxis_title="Frequenz [Hz]", yaxis_title="|FFT|",
            margin=dict(l=20, r=20, t=30, b=20), height=380,
            dragmode="zoom",
            clickmode="event+select"
        )

        plotly_event = st.plotly_chart(
            fig_fft, 
            use_container_width=True, 
            on_select="rerun", 
            selection_mode="points"
        )

        # Capture click event
        if plotly_event and "selection" in plotly_event and plotly_event["selection"]["points"]:
            clicked_freq = plotly_event["selection"]["points"][0]["x"]
            
            # Snap to highest local peak within 50 Hz
            df_max = 50
            idx_search = np.abs(f - clicked_freq) < df_max
            if np.any(idx_search):
                exact_peak_f = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak_f = float(clicked_freq)

            if not any(np.isclose(exact_peak_f, ex, atol=0.5) for ex in st.session_state.selected_peaks):
                st.session_state.selected_peaks.append(exact_peak_f)
                st.rerun()

        # ----------------------------------------------------
        # 3. Fourier Coefficients & Export
        # ----------------------------------------------------
        st.subheader("3. Ausgewählte Peaks & Fourierkoeffizienten")

        col_left, col_right = st.columns([1, 1])

        with col_left:
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
                        a_coeffs.append(float(np.max(ReZ[idx]) + np.min(ReZ[idx])))
                        b_coeffs.append(float(-(np.max(ImZ[idx]) + np.min(ImZ[idx]))))
                    else:
                        a_coeffs.append(0.0)
                        b_coeffs.append(0.0)

                export_str = "f(Hz)\ta_k\tb_k\n"
                for f_val, a_val, b_val in zip(selected_freqs, a_coeffs, b_coeffs):
                    export_str += f"{f_val:.2f}\t{a_val:.5f}\t{b_val:.5f}\n"

                st.text_area("Fourierkoeffizienten", export_str, height=140)
                st.download_button("💾 Fourierdaten exportieren (.txt)", export_str, "Fourierkoeffizienten.txt", "text/plain")

        # ----------------------------------------------------
        # 4. Fourier Synthese & Audio Playback (Restored)
        # ----------------------------------------------------
        with col_right:
            st.markdown("**🔊 Audiovergleich & Fourier-Synthese**")
            
            # Original Audio Player
            st.audio(st.session_state.audio_bytes, format="audio/wav")
            st.caption("Originale Audioaufnahme")

            if st.session_state.selected_peaks:
                # Compute Fourier Synthesis x_synth(t) = sum(a*cos) + sum(b*sin)
                xsynth = np.zeros_like(tfft)
                for i in range(len(selected_freqs)):
                    xsynth += a_coeffs[i] * np.cos(2 * np.pi * selected_freqs[i] * tfft)
                    xsynth += b_coeffs[i] * np.sin(2 * np.pi * selected_freqs[i] * tfft)

                # Normalize to match original segment amplitude
                if np.max(np.abs(xsynth)) != 0:
                    xsynth *= (np.max(np.abs(xfft)) / np.max(np.abs(xsynth)))

                # Write synthetic audio to memory buffer
                synth_buffer = io.BytesIO()
                sf.write(synth_buffer, xsynth, fs, format="WAV")
                
                # Synthesized Audio Player
                st.audio(synth_buffer.getvalue(), format="audio/wav")
                st.caption("Synthetisiertes Signal (basierend auf ausgewählten Peaks)")
