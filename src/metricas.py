import numpy as np


def curva_aprendizaje_db(residuo, fs, ventana_s=0.05):
    n = max(1, int(ventana_s * fs))
    suma = np.convolve(residuo**2, np.ones(n))[: len(residuo)]
    p = suma / np.minimum(np.arange(1, len(residuo) + 1), n)
    return 10 * np.log10(p + 1e-12)


def tiempo_convergencia(ruido, residuo, fs, margen_db=3.0, ventana_s=0.01):
    att = (curva_aprendizaje_db(ruido, fs, ventana_s)
           - curva_aprendizaje_db(residuo, fs, ventana_s))
    final = np.median(att[int(0.75 * len(att)):])
    idx = np.flatnonzero(att >= final - margen_db)
    return idx[0] / fs if idx.size else np.nan
