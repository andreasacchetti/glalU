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
# Session State & History Stack Setup
# ----------------------------------------------------
if "time_history" not in st.session_state:
    st.session_state.time_history = []  # Stack of (x_range, y_range)
if "time_x_range" not in st.session_state:
    st.session_state.time_x_range = None
if "time_y_range" not in st.session_state:
    st.session_state.time_y_range = None

if "fft_history" not in st.session_state:
    st.session_state.fft_history = []  # Stack of (x_range, y_range)
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
    # 1. Signal im Zeitbereich
    # ----------------------------------------------------
    st.subheader("1. Signal im Zeitbereich")

    t_start_val = st.session_state.time_x_range[0] if st.session_state.time_x_range else float(t[0])
    t_end_val = st.session_state.time_x_range[1] if st.session_state.time_x_range else float(t[-1])

    # Header Row with Indicators and Icon Buttons
    col_t_ind1, col_t_ind2, col_t_empty, col_t_back, col_t_reset = st.columns([2, 2, 4, 1, 1])
    with col_t_ind1:
        st.metric("⏱️ Startzeit (s)", f"{t_start_val:.4f} s")
    with col_t_ind2:
        st.metric("⏱️ Endzeit (s)", f"{t_end_val:.4f} s")
    with col_t_back:
        st.write("")
        has_time_history = len(st.session_state.time_history) > 0
        if st.button("⏪", key="back_time_zoom", help="Ein Schritt zurück", disabled=not has_time_history):
            prev_x, prev_y = st.session_state.time_history.pop()
            st.session_state.time_x_range = prev_x
            st.session_state.time_y_range = prev_y
            st.rerun()
    with col_t_reset:
        st.write("")
        has_time_zoom = st.session_state.time_x_range is not None
        if st.button("🔄", key="reset_time_zoom", help="Zoom vollständig zurücksetzen", disabled=not has_time_zoom):
            st.session_state.time_history.clear()
            st.session_state.time_x_range = None
            st.session_state.time_y_range = None
            st.rerun()

    # Downsample time data safely for chart rendering
    step_t = max(1, len(t) // 10000)
    df_time = pd.DataFrame({"Zeit": t[::step_t], "Amplitude": data[::step_t]})

    x_scale_time = alt.Scale(domain=st.session_state.time_x_range, zero=False) if st.session_state.time_x_range else alt.Scale(zero=False)
    y_scale_time = alt.Scale(domain=st.session_state.time_y_range, zero=False) if st.session_state.time_y_range else alt.Scale(zero=False)

    brush_time = alt.selection_interval(encodings=["x", "y"], name="time_brush")

    chart_time = (
        alt.Chart(df_time)
        .mark_line(color="#1f77b4", strokeWidth=1, clip=True)  # clip=True prevents overflowing chart boundaries
        .encode(
            x=alt.X("Zeit:Q", title="Zeit [s]", scale=x_scale_time),
            y=alt.Y("Amplitude:Q", title="Amplitude", scale=y_scale_time)
        )
        .add_params(brush_time)
        .properties(height=240)
    )

    event_time = st.altair_chart(chart_time, use_container_width=True, on_select="rerun", key="time_chart")

    # Capture zoom upon button release with boundary updates
    if event_time and "selection" in event_time and "time_brush" in event_time["selection"]:
        bounds = event_time["selection"]["time_brush"]
        if "Zeit" in bounds and "Amplitude" in bounds and len(bounds["Zeit"]) == 2 and len(bounds["Amplitude"]) == 2:
            new_tx = [float(bounds["Zeit"][0]), float(bounds["Zeit"][1])]
            new_ty = [float(bounds["Amplitude"][0]), float(bounds["Amplitude"][1])]

            # Avoid re-triggering if bounds haven't changed
            if st.session_state.time_x_range != new_tx or st.session_state.time_y_range != new_ty:
                # Push current bounds to history stack
                st.session_state.time_history.append((st.session_state.time_x_range, st.session_state.time_y_range))
                st.session_state.time_x_range = new_tx
                st.session_state.time_y_range = new_ty
                st.rerun()

    # Apply precise range mask (handles micro-range precision safely)
    eps = 1e-9
    mask = (t >= (t_start_val - eps)) & (t <= (t_end_val + eps))
    xfft = data[mask]
    tfft = t[mask]

    # Fallback to single point or tiny slice if mask yields empty set on micro zooms
    if len(xfft) == 0:
        idx = np.searchsorted(t, t_start_val)
        idx = min(max(0, idx), len(t) - 1)
        xfft = data[idx:idx+2]
        tfft = t[idx:idx+2]

    # ----------------------------------------------------
    # 2. FFT Spektrum
    # ----------------------------------------------------
    if len(xfft) > 0:
        L = len(xfft)
        m = int(2 ** np.ceil(np.log2(max(L, 16))))
        Z = fft.fft(xfft, m)

        ReZ = np.real(Z[: L // 2 + 1])
        ImZ = np.imag(Z[: L // 2 + 1])
        P = np.abs(Z / L)[: L // 2 + 1]
        P[1:-1] *= 2
        f = np.arange(len(P)) * fs / m

        st.subheader("2. FFT Spektrum")

        col_f_hdr, col_f_empty, col_f_back, col_f_reset = st.columns([6, 2, 1, 1])
        with col_f_back:
            has_fft_history = len(st.session_state.fft_history) > 0
            if st.button("⏪", key="back_fft_zoom", help="Ein Schritt zurück", disabled=not has_fft_history):
                prev_fx, prev_fy = st.session_state.fft_history.pop()
                st.session_state.fft_x_range = prev_fx
                st.session_state.fft_y_range = prev_fy
                st.rerun()
        with col_f_reset:
            has_fft_zoom = st.session_state.fft_x_range is not None
            if st.button("🔄", key="reset_fft_zoom", help="Zoom vollständig zurücksetzen", disabled=not has_fft_zoom):
                st.session_state.fft_history.clear()
                st.session_state.fft_x_range = None
                st.session_state.fft_y_range = None
                st.rerun()

        valid_mask = f <= 5000
        f_sub = f[valid_mask]
        P_sub = P[valid_mask]
        step_f = max(1, len(f_sub) // 10000)

        df_fft = pd.DataFrame({"Frequenz": f_sub[::step_f], "FFT": P_sub[::step_f]})

        x_scale_fft = alt.Scale(domain=st.session_state.fft_x_range, zero=False) if st.session_state.fft_x_range else alt.Scale(zero=False)
        y_scale_fft = alt.Scale(domain=st.session_state.fft_y_range, zero=False) if st.session_state.fft_y_range else alt.Scale(zero=False)

        brush_fft = alt.selection_interval(encodings=["x", "y"], name="fft_brush")

        base_fft = (
            alt.Chart(df_fft)
            .mark_line(color="#1f77b4", strokeWidth=1.5, clip=True)  # Clipping applied
            .encode(
                x=alt.X("Frequenz:Q", title="Frequenz [Hz]", scale=x_scale_fft),
                y=alt.Y("FFT:Q", title="|FFT|", scale=y_scale_fft)
            )
        )

        # Temporary dummy rendering layer for user peaks
        # Will render vertical rules once frequencies are populated below
        user_freq_keys = [f"peak_in_{i}" for i in range(10)]
        active_peaks = [st.session_state[k] for k in user_freq_keys if k in st.session_state and st.session_state[k] > 0]

        rules = []
        if len(active_peaks) > 0:
            df_peaks = pd.DataFrame({"Peak": active_peaks})
            rule_chart = (
                alt.Chart(df_peaks)
                .mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2, clip=True)
                .encode(x="Peak:Q")
            )
            rules.append(rule_chart)

        chart_fft = alt.layer(base_fft, *rules).add_params(brush_fft).properties(height=340)

        event_fft = st.altair_chart(chart_fft, use_container_width=True, on_select="rerun", key="fft_chart")

        # Capture FFT box zoom upon release
        if event_fft and "selection" in event_fft and "fft_brush" in event_fft["selection"]:
            bounds_f = event_fft["selection"]["fft_brush"]
            if "Frequenz" in bounds_f and "FFT" in bounds_f and len(bounds_f["Frequenz"]) == 2 and len(bounds_f["FFT"]) == 2:
                new_fx = [float(bounds_f["Frequenz"][0]), float(bounds_f["Frequenz"][1])]
                new_fy = [float(bounds_f["FFT"][0]), float(bounds_f["FFT"][1])]

                if st.session_state.fft_x_range != new_fx or st.session_state.fft_y_range != new_fy:
                    st.session_state.fft_history.append((st.session_state.fft_x_range, st.session_state.fft_y_range))
                    st.session_state.fft_x_range = new_fx
                    st.session_state.fft_y_range = new_fy
                    st.rerun()

        # ----------------------------------------------------
        # Peak Selection (Relocated Below FFT Chart)
        # ----------------------------------------------------
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
