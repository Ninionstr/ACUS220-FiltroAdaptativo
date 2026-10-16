import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from adaptive import atenuacion_db, lms, nlms, rls
from metricas import tiempo_convergencia
from signals import ruido_banda_ancha, ruta_acustica, voz_sintetica, zumbido

fs, dur = 8000, 4
LONGITUDES = [4, 8, 16, 32, 64]
N_TAPS_RUTA = 16
ALGORITMOS = [
    ("LMS", lms, dict(mu=0.005)),
    ("NLMS", nlms, dict(mu=0.05)),
    ("RLS", rls, dict(lam=0.999)),
]
RUIDOS = {
    "zumbido 50 Hz": zumbido(dur, fs),
    "banda ancha (< 1,5 kHz)": ruido_banda_ancha(dur, fs),
}

raiz = os.path.join(os.path.dirname(__file__), "..")
dir_fig = os.path.join(raiz, "figures")
dir_res = os.path.join(raiz, "results")
os.makedirs(dir_res, exist_ok=True)

s = voz_sintetica(dur, fs)
m = slice(fs, None)


def escenario(x):
    ruido, _ = ruta_acustica(x, n_taps=N_TAPS_RUTA)
    g = np.sqrt(10 * np.mean(s**2) / np.mean(ruido**2))
    return g * x, g * ruido


filas = []
for nombre_ruido, x0 in RUIDOS.items():
    x, ruido = escenario(x0)
    d = s + ruido
    for nombre, f, kw in ALGORITMOS:
        for L in LONGITUDES:
            t0 = time.perf_counter()
            e, _ = f(x, d, L=L, **kw)
            t_comp = time.perf_counter() - t0
            att = atenuacion_db(ruido[m], (e - s)[m])
            filas.append(dict(ruido=nombre_ruido, algoritmo=nombre, L=L,
                              atenuacion_db=round(att, 2), computo_s=round(t_comp, 3)))
            print(f"{nombre_ruido:24s} {nombre:5s} L={L:3d}  {att:6.1f} dB  {t_comp:.2f} s")

with open(os.path.join(dir_res, "barrido_L.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)

colores = {"LMS": "C0", "NLMS": "C1", "RLS": "C2"}
fig, ax = plt.subplots(1, len(RUIDOS), figsize=(10, 4), sharey=True)
for a, nombre_ruido in zip(ax, RUIDOS):
    for nombre, _, _ in ALGORITMOS:
        datos = [r for r in filas if r["ruido"] == nombre_ruido and r["algoritmo"] == nombre]
        a.plot([r["L"] for r in datos], [r["atenuacion_db"] for r in datos], "o-",
               color=colores[nombre], label=nombre)
    a.axvline(N_TAPS_RUTA, color="gray", ls=":", lw=1)
    a.annotate("largo de la ruta", (N_TAPS_RUTA, 1), xytext=(3, -3), textcoords="offset points",
               xycoords=("data", "axes fraction"), color="gray", fontsize=8, rotation=90, va="top")
    a.set_xscale("log", base=2)
    a.set_xticks(LONGITUDES, [str(L) for L in LONGITUDES])
    a.set_xlabel("longitud del filtro L [coeficientes]")
    a.set_title(nombre_ruido)
    a.grid(alpha=0.3)
ax[0].set_ylabel("atenuación del ruido [dB]")
ax[0].legend(fontsize=8)
fig.suptitle("Atenuación según algoritmo, tipo de ruido y longitud del filtro (ruido 10 dB sobre la voz)")
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "barrido_L.png"), dpi=120)
plt.close(fig)

x, ruido = escenario(RUIDOS["zumbido 50 Hz"])
d = s + ruido
filas_mu = []
for mu in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    e, _ = nlms(x, d, L=32, mu=mu)
    r = e - s
    att = atenuacion_db(ruido[m], r[m])
    t_conv = tiempo_convergencia(ruido, r, fs)
    filas_mu.append(dict(mu=mu, atenuacion_db=round(att, 2), convergencia_s=round(t_conv, 3)))
    print(f"NLMS mu={mu:<6} {att:6.1f} dB  convergencia {t_conv:.3f} s")

with open(os.path.join(dir_res, "barrido_mu.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas_mu[0]))
    w.writeheader()
    w.writerows(filas_mu)

mus = [r["mu"] for r in filas_mu]
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(mus, [r["atenuacion_db"] for r in filas_mu], "o-", color="C1", label="atenuación")
ax1.set_xscale("log")
ax1.set_xlabel("paso de adaptación μ (NLMS)")
ax1.set_ylabel("atenuación [dB]", color="C1")
ax2 = ax1.twinx()
ax2.plot(mus, [r["convergencia_s"] * 1000 for r in filas_mu], "s--", color="C4")
ax2.set_ylabel("tiempo de convergencia [ms]", color="C4")
ax1.set_title("NLMS: μ grande converge rápido pero deja más ruido residual")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(dir_fig, "barrido_mu.png"), dpi=120)
plt.close(fig)
