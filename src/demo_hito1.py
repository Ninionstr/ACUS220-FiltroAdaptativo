"""Hito 1: LMS cancelando un zumbido de 50 Hz sobre una voz sintética."""
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(__file__))
from adaptive import atenuacion_db, lms, nlms, rls
from metricas import curva_aprendizaje_db, tiempo_convergencia
from signals import ruta_acustica, voz_sintetica, zumbido

fs, dur, L = 8000, 4, 32
s = voz_sintetica(dur, fs)
x = zumbido(dur, fs)                  # referencia del zumbido (50 Hz + armónicos)
ruido, _ = ruta_acustica(x)           # el ruido llega al micrófono distorsionado
d = s + ruido

ALGORITMOS = [
    ("LMS", lms, dict(mu=0.005)),
    ("NLMS", nlms, dict(mu=0.05)),
    ("RLS", rls, dict(lam=0.999)),
]

raiz = os.path.join(os.path.dirname(__file__), "..")
dir_fig = os.path.join(raiz, "figures")
dir_res = os.path.join(raiz, "results")
dir_audio = os.path.join(dir_res, "audio")
os.makedirs(dir_audio, exist_ok=True)

t = np.arange(len(d)) / fs
m = slice(fs, None)
salidas, filas = {}, []
for nombre, f, kw in ALGORITMOS:
    t0 = time.perf_counter()
    e, _ = f(x, d, L=L, **kw)
    t_comp = time.perf_counter() - t0
    residuo = e - s
    curva = curva_aprendizaje_db(residuo, fs)
    att = atenuacion_db(ruido[m], residuo[m])
    t_conv = tiempo_convergencia(ruido, residuo, fs)
    salidas[nombre] = dict(e=e, residuo=residuo, curva=curva)
    filas.append(dict(algoritmo=nombre, L=L, **kw, atenuacion_db=round(att, 2),
                      convergencia_s=round(t_conv, 3), computo_s=round(t_comp, 3)))
    print(f"{nombre:5s} atenuación = {att:5.1f} dB | convergencia = {t_conv:.3f} s"
          f" | cómputo = {t_comp:.2f} s")

campos = ["algoritmo", "L", "mu", "lam", "atenuacion_db", "convergencia_s", "computo_s"]
with open(os.path.join(dir_res, "hito1_metricas.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=campos, restval="")
    w.writeheader()
    w.writerows(filas)

escala = 0.9 / np.max(np.abs(d))
sf.write(os.path.join(dir_audio, "hito1_contaminada.wav"), d * escala, fs)
sf.write(os.path.join(dir_audio, "hito1_voz_original.wav"), s * escala, fs)
for nombre, r in salidas.items():
    sf.write(os.path.join(dir_audio, f"hito1_salida_{nombre.lower()}.wav"),
             np.clip(r["e"] * escala, -1, 1), fs)

colores = {"LMS": "C0", "NLMS": "C1", "RLS": "C2"}

z = (t >= 2.15) & (t < 2.25)
fig, ax = plt.subplots(len(ALGORITMOS) + 1, 1, figsize=(9, 8), sharex=True, sharey=True)
ax[0].plot(t[z], d[z], "C3", lw=0.8, label="voz + zumbido (micrófono)")
ax[0].plot(t[z], s[z], "k", lw=0.8, alpha=0.5, label="voz original")
for a, nombre in zip(ax[1:], salidas):
    a.plot(t[z], salidas[nombre]["e"][z], colores[nombre], lw=0.8, label=f"salida {nombre}")
    a.plot(t[z], s[z], "k", lw=0.8, alpha=0.5, label="voz original")
for a in ax:
    a.set_ylabel("amplitud")
    a.legend(loc="upper right", fontsize=8)
    a.grid(alpha=0.3)
ax[-1].set_xlabel("tiempo [s]")
fig.suptitle(f"Cancelación de zumbido de 50 Hz (L = {L}, fs = {fs} Hz) — zoom 2,15–2,25 s")
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "hito1_tiempo.png"), dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
f_w, p_antes = welch(ruido[m], fs, nperseg=4096)
ax.semilogy(f_w, p_antes, "C3", lw=1, label="ruido en el micrófono (antes)")
for nombre, r in salidas.items():
    _, p = welch(r["residuo"][m], fs, nperseg=4096)
    ax.semilogy(f_w, p, colores[nombre], lw=1, label=f"residuo {nombre}")
for k in range(1, 6):
    ax.axvline(50 * k, color="gray", ls=":", lw=0.7)
ax.set_xlim(0, 400)
ax.set_xlabel("frecuencia [Hz]")
ax.set_ylabel("PSD [unidades²/Hz]")
ax.set_title("Espectro del ruido antes y después del filtrado (líneas: 50 Hz y armónicos)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "hito1_espectro.png"), dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, curva_aprendizaje_db(ruido, fs), "C3", lw=1, label="ruido sin filtrar")
for nombre, r in salidas.items():
    ax.plot(t, r["curva"], colores[nombre], lw=1, label=nombre)
ax.set_xlabel("tiempo [s]")
ax.set_ylabel("potencia del residuo [dB]")
ax.set_title("Curva de aprendizaje (media móvil causal de 50 ms)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "hito1_aprendizaje.png"), dpi=120)
plt.close(fig)
