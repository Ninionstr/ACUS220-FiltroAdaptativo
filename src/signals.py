"""Generación de señales sintéticas para probar la cancelación de ruido."""
import numpy as np


def tono(f, dur, fs, amp=1.0):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t)


def voz_sintetica(dur, fs, seed=0):
    """Señal tipo voz: tono modulado en amplitud con armónicos."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    fase = 2 * np.pi * np.cumsum(f0) / fs
    s = sum(np.sin(k * fase) / k for k in range(1, 6))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6)))
    return 0.3 * s * env


def ruido_banda_ancha(dur, fs, fc=1500, seed=1):
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    b, a = butter(4, fc / (fs / 2))
    return lfilter(b, a, rng.standard_normal(int(dur * fs)))


def ruta_acustica(x, n_taps=16, seed=2):
    """Pasa el ruido de referencia por una ruta acústica corta desconocida."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(n_taps) * np.exp(-np.arange(n_taps) / 4)
    return np.convolve(x, h)[: len(x)], h


def zumbido(dur, fs, f0=50, n_arm=5, seed=3):
    """Zumbido eléctrico: fundamental + armónicos + algo de ruido de fondo."""
    rng = np.random.default_rng(seed)
    x = sum(tono(f0 * k, dur, fs, 1.0 / k) for k in range(1, n_arm + 1))
    return x + 0.05 * rng.standard_normal(len(x))


def escenario(s, x, ruido_sobre_voz_db=10, n_taps=16):
    ruido, _ = ruta_acustica(x, n_taps=n_taps)
    g = np.sqrt(10 ** (ruido_sobre_voz_db / 10) * np.mean(s**2) / np.mean(ruido**2))
    return g * x, g * ruido
