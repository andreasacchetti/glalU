import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import io
import soundfile as sf

st.title("Audio Fourier Analyse")

# 1. Audio Recording (Handles Browser Microphone Permission & Cloud Upload)
audio_file = st.audio_input("Record your audio")

if audio_file:
    # Read raw audio bytes into Numpy array
    data, fs = sf.read(io.BytesIO(audio_file.getvalue()))
    if len(data.shape) > 1:
        data = data[:, 0]  # Mono channel
        
    t = np.arange(len(data)) / fs

    # 2. Time-Domain Signal Plot
    st.subheader("1. Signal im Zeitbereich")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, data)
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # 3. FFT Time Window Selection (Replaces ginput selection)
    st.subheader("2. FFT Analysebereich auswählen")
    t_min, t_max = st.slider("Zeitfenster [s]", 0.0, float(t[-1]), (0.0, float(t[-1])))
    
    mask = (t >= t_min) & (t <= t_max)
    xfft = data[mask]
    
    if len(xfft) > 0:
        # FFT Computation
        L = len(xfft)
        m = int(2**np.ceil(np.log2(L)))
        Z = fft(xfft, m)
        
        ReZ = np.real(Z[:L//2+1])
        ImZ = np.imag(Z[:L//2+1])
        P = np.abs(Z / L)[:L//2+1]
        P[1:-1] *= 2
        f = np.arange(len(P)) * fs / m

        # 4. Plot FFT Spectrum
        st.subheader("3. FFT Spektrum")
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6))
        ax2[0].plot(f, P)
        ax2[0].set_title("Betragsspektrum (linear)")
        ax2[0].set_xlabel("Frequenz [Hz]")
        
        ax2[1].semilogy(f, P)
        ax2[1].set_title("Betragsspektrum (logarithmisch)")
        ax2[1].set_xlabel("Frequenz [Hz]")
        plt.tight_layout()
        st.pyplot(fig2)

        # 5. Peak Frequency Selection (Replaces Matplotlib ginput)
        st.subheader("4. Peaks auswählen & Fourierkoeffizienten")
        # Find local maxima to present as selectable peaks
        peak_indices = np.where((P[1:-1] > P[:-2]) & (P[1:-1] > P[2:]))[0] + 1
        top_freqs = np.round(f[peak_indices], 2)
        
        selected_freqs = st.multiselect("Select Peak Frequencies [Hz]", top_freqs)

        if selected_freqs:
            df_max = 100
            a_coeffs = []
            b_coeffs = []
            
            for sf_freq in selected_freqs:
                idx = np.abs(f - sf_freq) < df_max
                if np.any(idx):
                    Re_plus = np.max(ReZ[idx])
                    Re_minus = np.min(ReZ[idx])
                    Im_plus = np.max(ImZ[idx])
                    Im_minus = np.min(ImZ[idx])
                    
                    a_coeffs.append(Re_plus + Re_minus)
                    b_coeffs.append(-(Im_plus + Im_minus))
                else:
                    a_coeffs.append(0)
                    b_coeffs.append(0)

            # Export Data View & CSV Download
            export_str = "f(Hz)\ta_k\tb_k\n"
            for f_val, a_val, b_val in zip(selected_freqs, a_coeffs, b_coeffs):
                export_str += f"{f_val}\t{a_val:.5f}\t{b_val:.5f}\n"
                
            st.text_area("Fourierkoeffizienten", export_str, height=150)
            
            # Browser File Download (Replaces local filedialog save)
            st.download_button(
                label="💾 Fourierdaten exportieren",
                data=export_str,
                file_name="Fourierkoeffizienten.txt",
                mime="text/plain"
            )
