# Cancelación activa de ruido con filtros adaptativos

Proyecto del curso **ACUS220 — Acústica Computacional con Python**.
Integrante: Ninoska Toledo.

## Idea del proyecto
Implementar desde cero filtros adaptativos (LMS, NLMS y RLS) que cancelan un ruido
(zumbido de 50 Hz, ventilador, motor) usando una señal de referencia del ruido,
y comparar su desempeño.

## Pregunta principal
¿Qué algoritmo adaptativo (LMS, NLMS o RLS) logra la mejor cancelación de ruido
según el tipo de ruido y la longitud del filtro?

## Motivación
La cancelación activa de ruido (audífonos, cabinas, vehículos) se basa en filtros
adaptativos. Implementarlos sin librerías de IA permite ver cómo aprende el filtro y
medir cuánto ruido se elimina en dB.

## Datos
- Señales sintéticas: voz/tono + ruido tonal (50 Hz) y ruido de banda ancha.
- Grabaciones reales (hito 3): micrófono de referencia + micrófono de error
  (dos celulares). Se guardan en `data/raw/` (no se versionan).

## Método
```
d[n] = s[n] + (h * x)[n]     señal deseada + ruido filtrado por la ruta acústica
y[n] = w^T x[n]              estimación del ruido
e[n] = d[n] - y[n]           señal limpia (salida)
w   <- w + mu * e[n] * x[n]  actualización LMS (NLMS normaliza por ||x||^2)
```
Métricas: atenuación de ruido en dB, tiempo de convergencia, costo computacional.

## Hitos
| Hito | Objetivo |
|------|----------|
| 1 | LMS sobre un tono de 50 Hz sintético. |
| 2 | NLMS y RLS con ruido de banda ancha; comparación de métricas. |
| 3 | Grabaciones reales con dos micrófonos y demo con audio. |

## Uso
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/demo_hito1.py      # genera figures/hito1_lms.png y results/
pytest                        # pruebas
```

## Estructura
```
src/        algoritmos (adaptive.py), señales sintéticas (signals.py), demos
tests/      pruebas
notebooks/  exploración
figures/    gráficos
results/    audios y tablas de resultados
references/ bibliografía
```

## Referencias
- Haykin, *Adaptive Filter Theory*.
- Kuo & Morgan, *Active Noise Control Systems*.
