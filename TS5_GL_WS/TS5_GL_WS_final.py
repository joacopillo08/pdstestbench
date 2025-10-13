# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 12:08:29 2025

@author: JGL y MWS
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal as sig
import mne as mne


#ECG CON RUIDO analizado con Welch

fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()
N_ecg = len(ecg_one_lead)

cant_promedio = 15
nperseg = N_ecg // cant_promedio

#print(nperseg)
nfft = 2 * nperseg
win = "flattop"

f_ecg, PSD_ECG_W = sig.welch(ecg_one_lead, fs = fs_ecg, window=win, nperseg = nperseg, nfft = nfft )
PSD_ECG_dB = 10 * np.log10(PSD_ECG_W)

plt.figure(figsize=(10,5))
plt.plot(f_ecg, PSD_ECG_dB)
plt.title('PSD del ECG con Ruido (Método de Welch)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)
plt.xlim(0,50)
plt.tight_layout()

#Pletismografia con Blackman-tukey

def blackman_tukey(x,  M = None):    
    
    #N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;

fs_ppg = 400 # Hz

ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
N_ppg = len(ppg)

ppg = ppg - np.mean(ppg)

PSD_ppg_BT = blackman_tukey(ppg)

f_ppg = np.fft.fftfreq(N_ppg, d = 1/fs_ppg)

PSD_ppg_dB = 10 * np.log10(PSD_ppg_BT)

plt.figure(figsize=(10,5))
plt.plot(f_ppg[:N_ppg // 2], PSD_ppg_dB[:N_ppg // 2])
plt.title('PSD del PPG con Ruido (Método de Blackman-Tukey)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)
plt.xlim(0,50)
plt.ylim(bottom=-20)
plt.tight_layout()
plt.show()

#Audios con periodograma

def periodograma_ventaneado (audio, fs, N_audios):
    win_audios = sig.windows.hann(N_audios)
    f_win, Periodograma_win = sig.periodogram(audio, fs = fs, window = win_audios, scaling='density')
    Periodograma_win_dB = 10 * np.log10(Periodograma_win)
    
    return f_win, Periodograma_win, Periodograma_win_dB

fs_cuca, wav_data_cuca = sio.wavfile.read('lacucaracha.wav')
audio_cuca = wav_data_cuca.astype(float)
audio_cuca = audio_cuca - np.mean(audio_cuca)
N_cuca = len(audio_cuca)

f_cuca, PSD_cuca, PSD_cuca_dB = periodograma_ventaneado(audio_cuca, fs = fs_cuca, N_audios = N_cuca)

plt.figure(figsize=(10,5))
plt.plot(f_cuca, PSD_cuca_dB)
plt.title('PSD del audio "La cucaracha" (Método de periodograma ventaneado)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)
plt.xlim(0,4000)
plt.tight_layout()
plt.show()

#prueba

fs_prueba, wav_data_prueba = sio.wavfile.read('prueba psd.wav')
audio_prueba = wav_data_prueba.astype(float)
audio_prueba = audio_prueba - np.mean(audio_prueba)
N_prueba = len(audio_prueba)

f_prueba, PSD_prueba, PSD_prueba_dB = periodograma_ventaneado(audio_prueba, fs = fs_prueba, N_audios = N_prueba)

plt.figure(figsize=(10,5))
plt.plot(f_prueba, PSD_prueba_dB)
plt.title('PSD del audio "prueba" (Método de periodograma ventaneado)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)
plt.xlim(0,4000)
plt.tight_layout()
plt.show()

#silbido

fs_silbido, wav_data_silbido = sio.wavfile.read('silbido.wav')
audio_silbido = wav_data_silbido.astype(float)
audio_silbido = audio_silbido - np.mean(audio_silbido)
N_silbido = len(audio_silbido)

f_silbido, PSD_silbido, PSD_silbido_dB = periodograma_ventaneado(audio_silbido, fs = fs_silbido, N_audios = N_silbido)

plt.figure(figsize=(10,5))
plt.plot(f_silbido, PSD_silbido_dB)
plt.title('PSD del audio "silbido" (Método de periodograma ventaneado)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)
plt.xlim(0,4000)
plt.tight_layout()
plt.show()

## estimacion de ancho de banda

def estimacion_bw (freq, PSD, porcentaje):
    df = freq[1] - freq[0]
    en_acum = np.cumsum(PSD * df)
    en_tot = en_acum[-1] #la energia total es el ultimo valor del acum
    en_corte = en_tot * porcentaje
    
    indice_corte = np.where(en_acum >= en_corte)[0][0]
    freq_BW = freq[indice_corte]
    
    return freq_BW
#para BT hay que calcular en el eje positivo solamente
f_ppg_positivo = f_ppg[:N_ppg // 2]
PSD_ppg_BT_positivo = PSD_ppg_BT[:N_ppg // 2]

BW_ECG = estimacion_bw (freq = f_ecg, PSD = PSD_ECG_W , porcentaje = 0.95)
BW_PPG = estimacion_bw (freq = f_ppg_positivo, PSD = PSD_ppg_BT_positivo , porcentaje = 0.95)
BW_cuca = estimacion_bw (freq = f_cuca, PSD = PSD_cuca , porcentaje = 0.95)
BW_prueba = estimacion_bw (freq = f_prueba, PSD = PSD_prueba , porcentaje = 0.95)
BW_silbido = estimacion_bw (freq = f_silbido, PSD = PSD_silbido , porcentaje = 0.95)

print("BW ECG:", BW_ECG)
print("BW PPG:", BW_PPG)
print("BW La cucaracha:", BW_cuca)
print("BW prueba:", BW_prueba)
print("BW silbido:", BW_silbido)


#BONUS

raw = mne.io.read_raw_edf('brux1.edf', preload = True)

emg = raw.copy().pick_channels(['EMG1-EMG2'])

emg_data = emg.get_data()[0]
fs_emg = int(emg.info['sfreq'])
N_emg = len(emg_data)

cant_promedio_emg = 15
nperseg_emg = N_emg // cant_promedio_emg
#nfft_emg = 2 * nperseg
win_emg = 'hann'

#calculo PSD con WELCH
f_emg, PSD_EMG_W = sig.welch(emg_data, fs = fs_emg, window = win_emg, nperseg = nperseg_emg)
PSD_EMG_dB = 10 * np.log10(PSD_EMG_W)

plt.figure(figsize=(10,5))
plt.plot(f_emg, PSD_EMG_dB)
plt.title('PSD de la Señal EMG (Método de Welch)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)
plt.xlim(0,500)  # EMG tiene energía hasta ~450 Hz
plt.tight_layout()
plt.show()

#calculo BW
BW_EMG = estimacion_bw (freq = f_emg, PSD = PSD_EMG_W , porcentaje = 0.95)

print("El ancho de banda esencial del EMG es: ", BW_EMG)