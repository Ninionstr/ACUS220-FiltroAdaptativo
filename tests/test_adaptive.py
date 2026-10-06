import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from adaptive import atenuacion_db, lms, nlms, rls
from signals import ruta_acustica, voz_sintetica, zumbido


def _caso():
    fs = 8000
    s = voz_sintetica(3, fs)
    x = zumbido(3, fs)
    r, _ = ruta_acustica(x)
    return s, x, s + r, r


def test_nlms_cancela_zumbido():
    s, x, d, r = _caso()
    e, _ = nlms(x, d, L=32, mu=0.05)
    assert atenuacion_db(r[8000:], (e - s)[8000:]) > 10


def test_lms_reduce_ruido():
    s, x, d, r = _caso()
    e, _ = lms(x, d, L=32, mu=0.005)
    assert atenuacion_db(r[8000:], (e - s)[8000:]) > 10


def test_rls_cancela_zumbido():
    s, x, d, r = _caso()
    e, _ = rls(x, d, L=16)
    assert atenuacion_db(r[8000:], (e - s)[8000:]) > 20
