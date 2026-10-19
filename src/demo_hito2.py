"""Hito 2: cancelación de ruido sobre voz real (Open Speech Repository)."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from adaptive import atenuacion_db, lms, nlms, rls
from datos import cargar_voz
from metricas import tiempo_convergencia
from signals import escenario, ruido_banda_ancha, zumbido

fs, L = 8000, 16
ALGORITMOS = [
    ("LMS", lms, dict(mu=0.005)),
    ("NLMS", nlms, dict(mu=0.05)),
    ("RLS", rls, dict(lam=0.999)),
]
COLORES = {"LMS": "C0", "NLMS": "C1", "RLS": "C2"}

raiz = os.path.join(os.path.dirname(__file__), "..")
dir_fig = os.path.join(raiz, "figures")
dir_res = os.path.join(raiz, "results")
dir_audio = os.path.join(dir_res, "audio")
os.makedirs(dir_audio, exist_ok=True)
m = slice(fs, None)


def ruidos(dur):
    return {"zumbido 50 Hz": zumbido(dur, fs), "banda ancha (< 1,5 kHz)": ruido_banda_ancha(dur, fs)}


def guardar_csv(nombre, filas):
    with open(os.path.join(dir_res, nombre), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


s = cargar_voz()
dur = len(s) / fs
x, ruido = escenario(s, zumbido(dur, fs))
d = s + ruido

filas, salidas = [], {}
for nombre, f, kw in ALGORITMOS:
    e, _ = f(x, d, L=L, **kw)
    salidas[nombre] = e
    filas.append(dict(algoritmo=nombre, L=L,
                      atenuacion_db=round(atenuacion_db(ruido[m], (e - s)[m]), 2),
                      convergencia_s=round(tiempo_convergencia(ruido, e - s, fs), 3)))
    print(f"{nombre:5s} atenuación = {filas[-1]['atenuacion_db']:5.1f} dB"
          f" | convergencia = {filas[-1]['convergencia_s']:.3f} s")
guardar_csv("hito2_voz_real.csv", filas)

escala = 0.9 / np.max(np.abs(d))
sf.write(os.path.join(dir_audio, "hito2_contaminada.wav"), d * escala, fs)
for nombre, e in salidas.items():
    sf.write(os.path.join(dir_audio, f"hito2_salida_{nombre.lower()}.wav"),
             np.clip(e * escala, -1, 1), fs)

paneles = [("voz original", s), ("micrófono: voz + zumbido", d),
           ("salida NLMS", salidas["NLMS"]), ("salida RLS", salidas["RLS"])]
fig, ax = plt.subplots(len(paneles), 1, figsize=(10, 9), sharex=True)
for a, (titulo, senal) in zip(ax, paneles):
    a.specgram(senal[: 8 * fs], NFFT=512, Fs=fs, noverlap=384, cmap="magma", vmin=-110, vmax=-20)
    a.set_ylim(0, 1000)
    a.set_ylabel("frecuencia [Hz]")
    a.set_title(titulo, fontsize=10)
ax[-1].set_xlabel("tiempo [s]")
fig.suptitle(f"Espectrogramas con voz real (primeros 8 s, L = {L})")
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "hito2_espectrogramas.png"), dpi=120)
plt.close(fig)

s10 = s[: 10 * fs]
NIVELES = [-10, -5, 0, 5, 10, 20]
filas_snr = []
for nombre_ruido, x0 in ruidos(10).items():
    for nivel in NIVELES:
        x, ruido = escenario(s10, x0, ruido_sobre_voz_db=nivel)
        d = s10 + ruido
        for nombre, f, kw in ALGORITMOS:
            e, _ = f(x, d, L=L, **kw)
            filas_snr.append(dict(ruido=nombre_ruido, ruido_sobre_voz_db=nivel, algoritmo=nombre,
                                  atenuacion_db=round(atenuacion_db(ruido[m], (e - s10)[m]), 2)))
guardar_csv("hito2_snr.csv", filas_snr)

REF_SNR = [np.inf, 40, 30, 20, 10]
rng = np.random.default_rng(4)
x, ruido = escenario(s10, ruidos(10)["banda ancha (< 1,5 kHz)"])
d = s10 + ruido
filas_ref = []
for snr_ref in REF_SNR:
    sigma = 0 if np.isinf(snr_ref) else np.sqrt(np.mean(x**2) / 10 ** (snr_ref / 10))
    x_ref = x + sigma * rng.standard_normal(len(x))
    for nombre, f, kw in ALGORITMOS:
        e, _ = f(x_ref, d, L=L, **kw)
        filas_ref.append(dict(snr_referencia_db=snr_ref, algoritmo=nombre,
                              atenuacion_db=round(atenuacion_db(ruido[m], (e - s10)[m]), 2)))
guardar_csv("hito2_referencia.csv", filas_ref)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, nombre_ruido in zip(ax[:2], ruidos(1)):
    for nombre, _, _ in ALGORITMOS:
        datos = [r for r in filas_snr if r["ruido"] == nombre_ruido and r["algoritmo"] == nombre]
        a.plot(NIVELES, [r["atenuacion_db"] for r in datos], "o-", color=COLORES[nombre], label=nombre)
    a.set_xlabel("ruido sobre la voz [dB]")
    a.set_title(f"Voz real + {nombre_ruido}", fontsize=10)
    a.grid(alpha=0.3)
ax[0].set_ylabel("atenuación del ruido [dB]")
ax[0].legend(fontsize=8)
etiquetas = ["sin ruido" if np.isinf(v) else f"{v:g}" for v in REF_SNR]
for nombre, _, _ in ALGORITMOS:
    datos = [r for r in filas_ref if r["algoritmo"] == nombre]
    ax[2].plot(range(len(REF_SNR)), [r["atenuacion_db"] for r in datos], "o-",
               color=COLORES[nombre], label=nombre)
ax[2].set_xticks(range(len(REF_SNR)), etiquetas)
ax[2].set_xlabel("SNR de la referencia [dB]")
ax[2].set_title("Referencia ruidosa (banda ancha, +10 dB)", fontsize=10)
ax[2].grid(alpha=0.3)
fig.suptitle(f"Sensibilidad con voz real (L = {L})")
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "hito2_sensibilidad.png"), dpi=120)
plt.close(fig)
