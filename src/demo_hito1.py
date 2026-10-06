"""Hito 1: LMS cancelando un zumbido de 50 Hz sobre una voz sintética."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from adaptive import atenuacion_db, lms, nlms
from signals import ruta_acustica, voz_sintetica, zumbido

fs, dur = 8000, 4
s = voz_sintetica(dur, fs)
x = zumbido(dur, fs)                  # referencia del zumbido (50 Hz + armónicos)
ruido, _ = ruta_acustica(x)           # el ruido llega al micrófono distorsionado
d = s + ruido

raiz = os.path.join(os.path.dirname(__file__), "..")
t = np.arange(len(d)) / fs
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for nombre, f, kw in [("LMS", lms, dict(mu=0.005)), ("NLMS", nlms, dict(mu=0.05))]:
    e, _ = f(x, d, L=32, **kw)
    # se mide el error contra la voz limpia una vez convergido
    m = slice(fs, None)
    print(f"{nombre}: atenuación = {atenuacion_db(d[m] - s[m], e[m] - s[m]):.1f} dB")
    ax[1].plot(t, e, lw=0.6, label=nombre)
ax[0].plot(t, d, lw=0.6, label="voz + zumbido")
ax[0].legend(); ax[1].plot(t, s, "k", lw=0.6, alpha=0.5, label="voz original")
ax[1].legend(); ax[1].set_xlabel("tiempo [s]")
plt.tight_layout()
plt.savefig(os.path.join(raiz, "figures", "hito1_lms.png"), dpi=120)
