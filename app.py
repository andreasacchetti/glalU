import io
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, Label, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

# Initialize persistent zoom memory
if "fft_x_start" not in st.session_state:
    st.session_state.fft_x_start = 0.0
if "fft_x_end" not in st.session_state:
    st.session_state.fft_x_end = 2000.0

audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]
    t = np.arange(len(data)) / fs

    # 1. Signal im Zeitbereich
    st.subheader("1. Signal im Zeitbereich")
    t_min, t_max = st.slider(
        "FFT-Analysebereich anpassen [s]:",
        min_value=0.0,
        max_value=float(t[-1]),
        value=(0.0, float(t[-1])),
        step=0.01
    )

    # Time Domain Bokeh Plot
    step_t = max(1, len(t) // 5000)
    p_time = figure(height=200, sizing_mode="stretch_width", x_axis_label="Zeit [s]", y_axis_label="Amplitude")
    p_time.line(t[::step_t], data[::step_t], line_width=1, color="#1f77b4")
    
    # Highlight region span
    time_span = Span(location=t_min, dimension='height', line_color='orange', line_width=2)
    time_span2 = Span(location=t_max, dimension='height', line_color='orange', line_width=2)
    p_time.add_layout(time_span)
    p_time.add_layout(time_span2)
    st.bokeh_chart(p_time)

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

    # 2. FFT Spektrum
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
        st.caption("🔍 **Nutze das Box-Zoom-Werkzeug** in der Bokeh-Toolbar (oben rechts am Graph), um ein Rechteck zu ziehen.")

        valid_mask = f <= 5000
        f_sub = f[valid_mask]
        P_sub = P[valid_mask]
        step_f = max(1, len(f_sub) // 5000)

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

        # Build Bokeh Plot with Box-Zoom Tools
        p_fft = figure(
            height=380, 
            sizing_mode="stretch_width",
            title="FFT Spektrum",
            x_axis_label="Frequenz [Hz]",
            y_axis_label="|FFT|",
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_drag="box_zoom",
            x_range=(st.session_state.fft_x_start, st.session_state.fft_x_end)
        )

        source_fft = ColumnDataSource(data=dict(x=f_sub[::step_f], y=P_sub[::step_f]))
        p_fft.line('x', 'y', source=source_fft, line_width=1.5, color="#1f77b4")

        # Draw Vertical Lines for Peak Inputs
        for peak_f in snapped_peaks:
            vline = Span(location=peak_f, dimension='height', line_color='red', line_dash='dashed', line_width=2)
            p_fft.add_layout(vline)

        # JavaScript callback to update session_state bounds when zoomed/panned
        source_event = ColumnDataSource(data=dict(x_range=[]))
        callback = CustomJS(args=dict(x_range=p_fft.x_range, source=source_event), code="""
            var start = x_range.start;
            var end = x_range.end;
            source.data = {'x_range': [start, end]};
            source.change.emit();
        """)
        p_fft.x_range.js_on_change('start', callback)
        p_fft.x_range.js_on_change('end', callback)

        # Render Bokeh chart with event tracking
        res = streamlit_bokeh_events(
            p_fft,
            events="x_range",
            key="fft_bokeh_chart",
            refresh_on_update=False
        )

        # Preserve Zoom Range across inputs
        if res and "x_range" in res:
            bounds = res["x_range"]
            if len(bounds) == 2:
                st.session_state.fft_x_start = bounds[0]
                st.session_state.fft_x_end = bounds[1]

        # 3. Fourierkoeffizienten & Synthese
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
