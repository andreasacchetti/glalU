import io
import altair as alt
import numpy as np
import pandas as pd
import scipy.fft as fft
import soundfile as sf
import streamlit as st

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# ----------------------------------------------------
# Session State Setup (Coordinates Memory)
# ----------------------------------------------------
if "time_x_range" not in st.session_state:
    st.session_state.time_x_range = None
if "time_y_range" not in st.session_state:
    st.session_state.time_y_range = None

if "fft_x_range" not in st.session_state:
    st.session_state.fft_x_range = None
if "fft_y_range" not in st.session_state:
    st.session_state.fft_y_range = None

audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]
    t = np.arange(len(data)) / fs

    # ----------------------------------------------------
    # 1. Signal im Zeitbereich (Altair Box-Zoom)
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich")

    t_start_val = st.session_state.time_x_range[0] if st.session_state.time_x_range else float(t[0])
    t_end_val = st.session_state.time_x_range[1] if st.session_state.time_x_range else float(t[-1])

    y_time_min = st.session_state.time_y_range[0] if st.session_state.time_y_range else float(np.min(data))
    y_time_max = st.session_state.time_y_range[1] if st.session_state.time_y_range else float(np.max(data))

    # Indicators & Dedicated Reset Button
    col_t1, col_t2, col_t3, col_t4, col_t_btn = st.columns([2, 2, 2, 2, 2])
    with col_t1:
        st.metric("⏱️ Startzeit (s)", f"{t_start_val:.4f} s")
    with col_t2:
        st.metric("⏱️ Endzeit (s)", f"{t_end_val:.4f} s")
    with col_t3:
        st.metric("📈 Y Min", f"{y_time_min:.4f}")
    with col_t4:
        st.metric("📈 Y Max", f"{y_time_max:.4f}")
    with col_t_btn:
        st.write("")
        if st.button("🔄 Zeit-Zoom zurücksetzen", key="reset_time_zoom"):
            st.session_state.time_x_range = None
            st.session_state.time_y_range = None
            st.rerun()

    # Downsample time data for smooth rendering
    step_t = max(1, len(t) // 5000)
    df_time = pd.DataFrame({"Zeit": t[::step_t], "Amplitude": data[::step_t]})

    # Explicit Altair Box-Selection interval
    brush_time = alt.selection_interval(encodings=["x", "y"], name="time_brush")

    x_scale_time = alt.Scale(domain=st.session_state.time_x_range) if st.session_state.time_x_range else alt.Scale()
    y_scale_time = alt.Scale(domain=st.session_state.time_y_range) if st.session_state.time_y_range else alt.Scale()

    chart_time = (
        alt.Chart(df_time)
        .mark_line(color="#1f77b4", strokeWidth=1)
        .encode(
            x=alt.X("Zeit:Q", title="Zeit [s]", scale=x_scale_time),
            y=alt.Y("Amplitude:Q", title="Amplitude", scale=y_scale_time)
        )
        .add_params(brush_time)
        .properties(height=220)
    )

    event_time = st.altair_chart(chart_time, use_container_width=True, on_select="rerun", key="time_chart")

    # Capture and retain box selection coordinates
    if event_time and "selection" in event_time and "time_brush" in event_time["selection"]:
        bounds = event_time["selection"]["time_brush"]
        if "Zeit" in bounds and len(bounds["Zeit"]) == 2:
            st.session_state.time_x_range = [float(bounds["Zeit"][0]), float(bounds["Zeit"][1])]
        if "Amplitude" in bounds and len(bounds["Amplitude"]) == 2:
            st.session_state.time_y_range = [float(bounds["Amplitude"][0]), float(bounds["Amplitude"][1])]

    # Mask signal using active zoom slice
    mask = (t >= t_start_val) & (t <= t_end_val)
    xfft = data[mask]
    tfft = t[mask]

    # ----------------------------------------------------
    # 2. FFT Spektrum (Altair Box-Zoom)
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

        col_f_hdr, col_f_btn = st.columns([4, 1])
        with col_f_btn:
            if st.button("🔄 FFT-Zoom zurücksetzen", key="reset_fft_zoom"):
                st.session_state.fft_x_range = None
                st.session_state.fft_y_range = None
                st.rerun()

        valid_mask = f <= 5000
        f_sub = f[valid_mask]
        P_sub = P[valid_mask]
        step_f = max(1, len(f_sub) // 5000)

        # 10 Peak inputs
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
                        step=0.1,
                        key=f"peak_in_{i}"
                    )
                    if val > 0:
                        user_freqs.append(val)

        df_fft = pd.DataFrame({"Frequenz": f_sub[::step_f], "FFT": P_sub[::step_f]})

        brush_fft = alt.selection_interval(encodings=["x", "y"], name="fft_brush")

        x_scale_fft = alt.Scale(domain=st.session_state.fft_x_range) if st.session_state.fft_x_range else alt.Scale()
        y_scale_fft = alt.Scale(domain=st.session_state.fft_y_range) if st.session_state.fft_y_range else alt.Scale()

        base_fft = (
            alt.Chart(df_fft)
            .mark_line(color="#1f77b4", strokeWidth=1.5)
            .encode(
                x=alt.X("Frequenz:Q", title="Frequenz [Hz]", scale=x_scale_fft),
                y=alt.Y("FFT:Q", title="|FFT|", scale=y_scale_fft)
            )
        )

        # Red dashed peak lines
        rules = []
        if len(user_freqs) > 0:
            df_peaks = pd.DataFrame({"Peak": user_freqs})
            rule_chart = (
                alt.Chart(df_peaks)
                .mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2)
                .encode(x="Peak:Q")
            )
            rules.append(rule_chart)

        chart_fft = alt.layer(base_fft, *rules).add_params(brush_fft).properties(height=380)

        event_fft = st.altair_chart(chart_fft, use_container_width=True, on_select="rerun", key="fft_chart")

        # Capture and retain box selection coordinates
        if event_fft and "selection" in event_fft and "fft_brush" in event_fft["selection"]:
            bounds_f = event_fft["selection"]["fft_brush"]
            if "Frequenz" in bounds_f and len(bounds_f["Frequenz"]) == 2:
                st.session_state.fft_x_range = [float(bounds_f["Frequenz"][0]), float(bounds_f["Frequenz"][1])]
            if "FFT" in bounds_f and len(bounds_f["FFT"]) == 2:
                st.session_state.fft_y_range = [float(bounds_f["FFT"][0]), float(bounds_f["FFT"][1])]

        # ----------------------------------------------------
        # 3. Fourierkoeffizienten & Audio-Synthese
        # ----------------------------------------------------
        if len(user_freqs) > 0:
            st.subheader("3. Fourierkoeffizienten & Audio-Synthese")
            col_left, col_right = st.columns([1, 1])

            a_coeffs, b_coeffs = [], []
            for u_freq in user_freqs:
                idx = np.abs(f - u_freq) < (fs / m)
                if np.any(idx):
                    a_coeffs.append(float(np.max(ReZ[idx]) + np.min(ReZ[idx])))
                    b_coeffs.append(float(-(np.max(ImZ[idx]) + np.min(ImZ[idx]))))
                else:
                    a_coeffs.append(0.0)
                    b_coeffs.append(0.0)

            with col_left:
                export_str = "f(Hz)\ta_k\tb_k\n"
                for f_val, a_val, b_val in zip(user_freqs, a_coeffs, b_coeffs):
                    export_str += f"{f_val:.2f}\t{a_val:.5f}\t{b_val:.5f}\n"

                st.text_area("Berechnete Koeffizienten", export_str, height=160)
                st.download_button("💾 Fourierdaten exportieren (.txt)", export_str, "Fourierkoeffizienten.txt", "text/plain")

            with col_right:
                st.audio(audio_file.getvalue(), format="audio/wav")
                xsynth = np.zeros_like(tfft)
                for i in range(len(user_freqs)):
                    xsynth += a_coeffs[i] * np.cos(2 * np.pi * user_freqs[i] * tfft)
                    xsynth += b_coeffs[i] * np.sin(2 * np.pi * user_freqs[i] * tfft)

                if np.max(np.abs(xsynth)) != 0:
                    xsynth *= (np.max(np.abs(xfft)) / np.max(np.abs(xsynth)))

                synth_buffer = io.BytesIO()
                sf.write(synth_buffer, xsynth, fs, format="WAV")
                st.audio(synth_buffer.getvalue(), format="audio/wav")
