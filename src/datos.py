import os
import urllib.request

import soundfile as sf

URL_OSR = "https://www.voiptroubleshooter.com/open_speech/american/"
ARCHIVOS_OSR = ["OSR_us_000_0010_8k.wav", "OSR_us_000_0030_8k.wav"]
DIR_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def descargar_osr():
    os.makedirs(DIR_RAW, exist_ok=True)
    for nombre in ARCHIVOS_OSR:
        ruta = os.path.join(DIR_RAW, nombre)
        if not os.path.exists(ruta):
            urllib.request.urlretrieve(URL_OSR + nombre, ruta)


def cargar_voz(nombre=ARCHIVOS_OSR[0], dur=None, fs=8000):
    descargar_osr()
    s, fs_archivo = sf.read(os.path.join(DIR_RAW, nombre))
    if fs_archivo != fs:
        raise ValueError(f"{nombre}: fs = {fs_archivo} Hz, se esperaba {fs} Hz")
    return s[: int(dur * fs)] if dur else s
