import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------
if "time_x_range" not in st.session_state:
    st.session_state.time_x_range = None
if "fft_x_range" not in st.session_state:
    st.session_state.fft_x_range = None

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

    # Time Bounds Indicators
    t_start_val = st.session_state.time_x_range[0] if st.session_state.time_x_range else float(t[0])
    t_end_val = st.session_state.time_x_range[1] if st.session_state.time_x_range else float(t[-1])

    col_t1, col_t2, col_t3 = st.columns([2, 2, 1])
    with col_t1:
        st.metric("⏱️ Startzeit (s)", f"{t_start_val:.4f} s")
    with col_t2:
        st.metric("⏱️ Endzeit (s)", f"{t_end_val:.4f} s")
    with col_t3:
        if st.button("🔄 Zeit-Zoom zurücksetzen", key="reset_time_zoom"):
            st.session_state.time_x_range = None
            st.rerun()

    step_t = max(1, len(t) // 5000)
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=t[::step_t], y=data[::step_t],
        mode='lines', line=dict(color='#1f77b4', width=1),
        name="Audio"
    ))

    layout_time = dict(
        height=220,
        margin=dict(l=20, r=20, t=20, b=30),
        xaxis_title="Zeit [s]",
        yaxis_title="Amplitude",
        dragmode="zoom"
    )

    if st.session_state.time_x_range is not None:
        layout_time["xaxis"] = dict(range=st.session_state.time_x_range)

    fig_time.update_layout(**layout_time)

    # Render Chart and catch layout changes
    event_time = st.plotly_chart(
        fig_time, 
        use_container_width=True, 
        on_select="rerun",
        key="time_plot"
    )

    # Update state only if a box zoom occurred
    if event_time and "selection" in event_time and "box" in event_time["selection"]:
        box_t = event_time["selection"]["box"]
        if len(box_t) > 0 and "x" in box_t[0]:
            x_min, x_max = min(box_t[0]["x"]), max(box_t[0]["x"])
            if st.session_state.time_x_range != [x_min, x_max]:
                st.session_state.time_x_range = [x_min, x_max]
                st.rerun()

    # Apply active slice mask
    mask = (t >= t_start_val) & (t <= t_end_val)
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

        valid_mask = f <= 5000
        f_sub = f[valid_mask]
        P_sub = P[valid_mask]
        step_f = max(1, len(f_sub) // 5000)

        # FFT Bounds Indicators
        f_start_val = st.session_state.fft_x_range[0] if st.session_state.fft_x_range else 0.0
        f_end_val = st.session_state.fft_x_range[1] if st.session_state.fft_x_range else 2000.0

        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        with col_f1:
            st.metric("📊 Min Frequenz (Hz)", f"{f_start_val:.2f} Hz")
        with col_f2:
            st.metric("📊 Max Frequenz (Hz)", f"{f_end_val:.2f} Hz")
        with col_f3:
            if st.button("🔄 FFT-Zoom zurücksetzen", key="reset_fft_zoom"):
                st.session_state.fft_x_range = None
                st.rerun()

        # Peak Inputs (NO SNAP ROUNDING)
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
                        step=0.1,  # Exact decimal precision
                        key=f"peak_in_{i}"
                    )
                    if val > 0:
                        user_freqs.append(val)

        # Plotly FFT Chart
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=f_sub[::step_f], y=P_sub[::step_f],
            mode='lines', line=dict(color='#1f77b4', width=1.5),
            name="|FFT|"
        ))

        # Add vertical lines for exact user frequency inputs (NO forced rounding)
        for peak_f in user_freqs:
            fig_fft.add_vline(x=peak_f, line_width=2, line_dash="dash", line_color="red")

        layout_fft = dict(
            height=380,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Frequenz [Hz]",
            yaxis_title="|FFT|",
            dragmode="zoom"
        )

        if st.session_state.fft_x_range is not None:
            layout_fft["xaxis"] = dict(range=st.session_state.fft_x_range)

        fig_fft.update_layout(**layout_fft)

        # Render FFT Chart and catch zoom events
        event_fft = st.plotly_chart(
            fig_fft, 
            use_container_width=True, 
            on_select="rerun",
            key="fft_plot"
        )

        # Update FFT range in state only when a new box zoom is drawn
        if event_fft and "selection" in event_fft and "box" in event_fft["selection"]:
            box_f = event_fft["selection"]["box"]
            if len(box_f) > 0 and "x" in box_f[0]:
                x_min_f, x_max_f = min(box_f[0]["x"]), max(box_f[0]["x"])
                if st.session_state.fft_x_range != [x_min_f, x_max_f]:
                    st.session_state.fft_x_range = [x_min_f, x_max_f]
                    st.rerun()

        # ----------------------------------------------------
        # 3. Fourierkoeffizienten & Audio-Synthese
        # ----------------------------------------------------
        if len(user_freqs) > 0:
            st.subheader("3. Fourierkoeffizienten & Audio-Synthese")
            col_left, col_right = st.columns([1, 1])

            a_coeffs, b_coeffs = [], []
            for u_freq in user_freqs:
                idx = np.abs(f - u_freq) < (fs / m)  # Exact bin evaluation
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
