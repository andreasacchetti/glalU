import io
import json
import numpy as np
import scipy.fft as fft
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("Audio Fourier Analyse & Synthese")

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

    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    tfft = t[mask]

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

        snapped_peaks = []
        df_max = 50
        for u_freq in user_freqs:
            idx_search = np.abs(f - u_freq) < df_max
            if np.any(idx_search):
                exact_peak = float(f[idx_search][np.argmax(P[idx_search])])
            else:
                exact_peak = float(u_freq)
            snapped_peaks.append(exact_peak)

        valid_mask = f <= 5000
        x_data = f[valid_mask].tolist()
        y_data = P[valid_mask].tolist()
        max_p = float(np.max(P[valid_mask])) if len(P[valid_mask]) > 0 else 1.0

        # Build shapes and annotations with explicit y-coordinates
        shapes = []
        annotations = []
        for peak_f in snapped_peaks:
            shapes.append({
                'type': 'line',
                'x0': peak_f, 'x1': peak_f,
                'y0': 0, 'y1': max_p * 1.1,
                'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
            })
            annotations.append({
                'x': peak_f, 'y': max_p * 1.05,
                'text': f"{peak_f:.1f} Hz",
                'showarrow': False,
                'font': {'color': 'red', 'size': 12}
            })

        plot_html = f"""
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <div id="fft_chart" style="width:100%;height:380px;"></div>
        <script>
            (function() {{
                var xData = {json.dumps(x_data)};
                var yData = {json.dumps(y_data)};
                var shapes = {json.dumps(shapes)};
                var annotations = {json.dumps(annotations)};

                var trace = {{
                    x: xData,
                    y: yData,
                    type: 'scatter',
                    mode: 'lines',
                    name: '|FFT|'
                }};

                var layout = {{
                    margin: {{ l: 40, r: 20, t: 30, b: 40 }},
                    xaxis: {{ title: 'Frequenz [Hz]' }},
                    yaxis: {{ title: '|FFT|' }},
                    shapes: shapes,
                    annotations: annotations
                }};

                // Read cached zoom bounds
                var savedX = localStorage.getItem('fft_x_range');
                var savedY = localStorage.getItem('fft_y_range');

                if (savedX) {{
                    try {{ layout.xaxis.range = JSON.parse(savedX); }} catch(e) {{}}
                }}
                if (savedY) {{
                    try {{ layout.yaxis.range = JSON.parse(savedY); }} catch(e) {{}}
                }}

                var chartDiv = document.getElementById('fft_chart');
                Plotly.react(chartDiv, [trace], layout, {{responsive: true}});

                // Save or reset zoom state on relayout
                chartDiv.on('plotly_relayout', function(eventdata) {{
                    if (eventdata['xaxis.range[0]'] !== undefined) {{
                        localStorage.setItem('fft_x_range', JSON.stringify([eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']]));
                        localStorage.setItem('fft_y_range', JSON.stringify([eventdata['yaxis.range[0]'], eventdata['yaxis.range[1]']]));
                    }} else if (eventdata['xaxis.autorange'] || eventdata['autosize']) {{
                        localStorage.removeItem('fft_x_range');
                        localStorage.removeItem('fft_y_range');
                    }}
                }});
            }})();
        </script>
        """

        components.html(plot_html, height=400)

        # 3. Fourier synthesis
        if len(snapped_peaks) > 0:
            st.subheader("3. Fourierkoeffizienten & Synthese")
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
