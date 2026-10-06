"""Filtros adaptativos: LMS, NLMS y RLS.

Todas las funciones reciben la referencia de ruido x y la señal deseada d
(señal + ruido) y devuelven (e, w_final): e es la salida limpia.
"""
import numpy as np


def _ventana(x, n, L):
    """Vector [x[n], x[n-1], ..., x[n-L+1]] con ceros antes del inicio."""
    v = np.zeros(L)
    m = min(L, n + 1)
    v[:m] = x[n::-1][:m] if n >= 0 else 0
    return v


def lms(x, d, L=32, mu=0.01):
    w = np.zeros(L)
    e = np.zeros(len(d))
    for n in range(len(d)):
        u = _ventana(x, n, L)
        e[n] = d[n] - w @ u
        w += mu * e[n] * u
    return e, w


def nlms(x, d, L=32, mu=0.5, eps=1e-8):
    w = np.zeros(L)
    e = np.zeros(len(d))
    for n in range(len(d)):
        u = _ventana(x, n, L)
        e[n] = d[n] - w @ u
        w += mu * e[n] * u / (eps + u @ u)
    return e, w


def rls(x, d, L=32, lam=0.999, delta=1.0):
    w = np.zeros(L)
    P = np.eye(L) / delta
    e = np.zeros(len(d))
    for n in range(len(d)):
        u = _ventana(x, n, L)
        k = P @ u / (lam + u @ P @ u)
        e[n] = d[n] - w @ u
        w += k * e[n]
        P = (P - np.outer(k, u @ P)) / lam
    return e, w


def atenuacion_db(ruido_antes, ruido_despues):
    """Reducción de potencia del ruido en dB (positivo = mejora)."""
    return 10 * np.log10(np.mean(ruido_antes**2) / np.mean(ruido_despues**2))
