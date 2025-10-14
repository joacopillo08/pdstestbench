#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:46:57 2025

@author: joacomillo12
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from scipy import signal

#Aca uso todo lo mismo que en la ts1. Tambien podria importar el archivo pero se le va a complicar a los profes.
fs = 100000
N = 1000
f = 2000
Ts = 1/fs
tt = np.arange(N) * Ts           # vector de tiempo

deltaF = fs/N
def mi_funcion_sen(vmax, dc, f, fase, N, fs):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    xx = dc + vmax * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    return tt, xx

#Para hacer modulacion hay que multplicar una señal contra otra señal.
def modulacion(vmax, dc, f, fase, N, fs):
    tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
    tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f/2, fase=0, N = N, fs=fs)
    x2 = xx * x1
    return x2


tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f, fase=np.pi/2, N = N, fs=fs)
x2 = modulacion(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)

#Forma de hacer el clipeo para recortar la amplitud de la señal. 
x3 = np.clip(xx,-0.75,0.75,out=None)

x4 = signal.square(2*np.pi*4000*tt)
x4 = x4 - np.mean(x4)
#este print esta para ver si le saque a la cuadrada la media. remoción de DC
#print("mean x4:", np.mean(x4))      # ~ 0.0
#print("sum x4:", np.sum(x4))        # ~ 0

# aca hago el del puslo. Como Npulso = Tpulso . fs. Voy a tener 200 muestras
# como mi N lo tengo fijo en 500. voy a tener 300 muestras que estan en 0. Si yo aumento N por ejemplo, siempre voy a tener fijas 200muestras que valen 1 y las N-200=0.
#Si yo cambio fs, me cambia el Npulso entonces ahi ya se modifican las muestras que valen 1. 
T_pulso = 0.01    # 10 ms 
N_pulso = int(T_pulso * fs)  # 500 muestras

N1 = 2000   # señal total de 40 ms
pulso = np.zeros(N1)
pulso[:N_pulso] = 1
tt_pulso = np.arange(N1) * Ts   # vector de tiempo consistente con N

potencia_xx = np.mean(xx**2)
potencia_x1 = np.mean(x1**2)
potencia_x2 = np.mean(x2**2)
potencia_x3 = np.mean(x3**2)
potencia_x4 = np.mean(x4**2)
energia_pulso = np.sum(pulso**2) * Ts #aca uso energia porque, Señales no periódicas o de duración finita. energía en unidades tiempo·amplitud^2
print("Señal principal, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_xx)
print("Señal desfada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x1)
print("Señal modulada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x2)
print("Señal recortada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x3)
print("Señal cuadrada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x4)
print("Señal pulso, Ts: ",Ts, " N: ", N, "y energia:", energia_pulso)
print("\n")
# =============================================================================
# Ejercicio 1
# HECHO --> Graficar la señal de salida para cada una de las señales de entrada que generó en el TS1. Considere que las mismas son causales.
# HECHO --> Hallar la respuesta al impulso y usando la misma, repetir la generación de la señal de salida para alguna de las señales de entrada consideradas en el punto anterior.
# En cada caso indique la frecuencia de muestreo, el tiempo de simulación y la potencia o energía de la señal de salida.
# =============================================================================

def en_diferencias(N,x):
    y = np.zeros(N)
    for n in range (N):
        x0 = x[n]
        x1 = x[n-1] if n-1 >= 0 else 0
        x2 = x[n-2] if n-2 >= 0 else 0
        y1 = y[n-1] if n-1 >= 0 else 0
        y2 = y[n-2] if n-2 >= 0 else 0
        y[n] = 3* 10**(-2)*x0 + 5 * 10**(-2)*x1 +  3 * 10**(-2)*x2 + 1.5*y1-0.5*y2
    return y

entradas = [
    (xx,   "Seno principal"),
    (x1,   "Seno desf. π/2"),
    (x2,   "AM (f/2)"),
    (x3,   "AM clipeada 75%"),
    (x4,   "Cuadrada 4 kHz"),
    (pulso,"Pulso 10 ms"),
]

# convolucion de la entrada xx con delta
delta = np.zeros(len(xx))
delta[0] = 1
h = en_diferencias(N = N, x = delta)
y_conv = np.convolve(xx, h)[:N]

fig, axs = plt.subplots(3, 2, figsize=(12, 8))
axs = axs.ravel()  # aplanar para poder usar axs[i]

for i, (x, nombre) in enumerate(entradas):
    y = en_diferencias(len(x), x)
    t = np.arange(len(x)) * Ts

    ax = axs[i]
    ax.plot(t, y, label="Salida", linewidth=1.5)
    ax.plot(t, x, '--', label="Entrada", linewidth=1.0)
    if i == 0:  # primer subplot corresponde a xx
        ax.plot(t, y_conv, linestyle='none', marker='o', markersize=2.5,label="h*xx (convolución)")

    ax.set_title(nombre)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)

fig.supxlabel("Tiempo [s]")
fig.supylabel("Amplitud")
fig.tight_layout()
plt.show()


t_h = np.arange(len(h)) * Ts

plt.figure(figsize=(10,4))
plt.stem(t_h[:200], h[:200], basefmt=" ")
plt.title("Respuesta al impulso h[n] — 200 primeras muestras")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


#%%
# =========================
# Ejercicio 2
# =========================
# Sistema A (FIR): y[n] = x[n] + 3 x[n-10]

b2 = np.zeros(11)
b2[0] = 1.0
b2[10] = 3.0
a2 = np.array([1.0])
y2  = lfilter(b2, a2, xx)

delta = np.zeros(len(xx))
delta[0] = 1
h2  = lfilter(b2, a2, delta)               
y2_conv = np.convolve(xx, h2, mode='full')[:N]  # causal
#y2_conv = y2_conv1[:len(xx)]  


plt.figure(figsize=(10,4))
plt.plot(tt, y2, '--', label="y (lfilter)", linewidth=1.5)
plt.plot(tt, y2_conv, linestyle='none', marker='o', markersize=2.5, label="y (conv con h2)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.title("Sistema A (FIR): y[n] = x[n] + 3·x[n−10]")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

# Sistema B (IIR): y[n] = x[n] + 3 y[n-10]  (INESTABLE)

b3 = np.array([1.0])
a3 = np.zeros(11)
a3[0]  = 1.0   # y[n]
a3[10] = -3.0  # -3 y[n-10]
y3  = lfilter(b3, a3, xx)

h3 = lfilter(b3, a3, delta)     # [1, 0.., 3, 0.., 9, ...] hasta len(xx)

y3_conv = np.convolve(xx, h3, mode='full')[:N]    # mismo largo que xx

# --- Plot comparación --------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(tt, y3, '--', label="y (lfilter IIR)", linewidth=1.5)
plt.plot(tt, y3_conv, linestyle='none', marker='o', markersize=2.5, label="y (conv con h3 trunc.)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.title("Sistema B (IIR, inestable): y[n] = x[n] + 3·y[n−10]")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()










