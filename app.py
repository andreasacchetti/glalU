import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

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


def capture_chart_zoom(widget_key, x_state_key, y_state_key):
    """Extracts explicit box-zoom coordinates from Plotly event payload and updates state."""
    if widget_key in st.session_state and st.session_state[widget_key]:
        state = st.session_state[widget_key]
        if "selection" in state and "box" in state["selection"]:
            box_data = state["selection"]["box"]
            if len(box_data) > 0:
                box = box_data[0]
                if "x" in box and len(box["x"]) == 2:
                    st.session_state[x_state_key] = [min(box["x"]), max(box["x"])]
                if "y" in box and len(box["y"]) == 2:
                    st.session_state[y_state_key] = [min(box["y"]), max(box["y"])]


audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]
    t = np.arange(len(data)) / fs

    # Process pending events from previous run
    capture_chart_zoom("time_plot", "time_x_range", "time_y_range")
    capture_chart_zoom("fft_plot", "fft_x_range", "fft_y_range")

    # ----------------------------------------------------
    # 1. Signal im Zeitbereich
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich")

    # Display indicator values based on captured zoom
    t_min_disp = st.session_state.time_x_range[0] if st.session_state.time_x_range else float(t[0])
    t_max_disp = st.session_state.time_x_range[1] if st.session_state.time_x_range else float(t[-1])

    y_min_disp = st.session_state.time_y_range[0] if st.session_state.time_y_range else float(np.min(data))
    y_max_disp = st.session_state.time_y_range[1] if st.session_state.time_y_range else float(np.max(data))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("⏱️ X Min (s)", f"{t_min_disp:.4f} s")
    with c2:
        st.metric("⏱️ X Max (s)", f"{t_max_disp:.4f} s")
    with c3:
        st.metric("📈 Y Min", f"{y_min_disp:.4f}")
    with c4:
        st.metric("📈 Y Max", f"{y_max_disp:.4f}")

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
        dragmode="select"
    )

    # Force range back whenever chart is rendered
    if st.session_state.time_x_range is not None:
        layout_time["xaxis"] = dict(range=st.session_state.time_x_range)
    if st.session_state.time_y_range is not None:
        layout_time["yaxis"] = dict(range=st.session_state.time_y_range)

    fig_time.update_layout(**layout_time)

    # Render Time Plot and catch events
    st.plotly_chart(
        fig_time, 
        use_container_width=True, 
        on_select="rerun",
        selection_mode=["box"],
        key="time_plot"
    )

    # Apply active slice mask for FFT calculation
    mask = (t >= t_min_disp) & (t <= t_max_disp)
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

        # Plotly FFT Chart
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=f_sub[::step_f], y=P_sub[::step_f],
            mode='lines', line=dict(color='#1f77b4', width=1.5),
            name="|FFT|"
        ))

        # Vertical peak lines
        for peak_f in user_freqs:
            fig_fft.add_vline(x=peak_f, line_width=2, line_dash="dash", line_color="red")

        layout_fft = dict(
            height=380,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Frequenz [Hz]",
            yaxis_title="|FFT|",
            dragmode="select"
        )

        # Force range back whenever peaks update or script reruns
        if st.session_state.fft_x_range is not None:
            layout_fft["xaxis"] = dict(range=st.session_state.fft_x_range)
        if st.session_state.fft_y_range is not None:
            layout_fft["yaxis"] = dict(range=st.session_state.fft_y_range)

        fig_fft.update_layout(**layout_fft)

        # Render FFT Chart
        st.plotly_chart(
            fig_fft, 
            use_container_width=True, 
            on_select="rerun",
            selection_mode=["box"],
            key="fft_plot"
        )

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
