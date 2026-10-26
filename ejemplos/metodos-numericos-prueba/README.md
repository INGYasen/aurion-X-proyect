# Prueba: métodos numéricos (Excel + Octave)

Demostración de cómo se entregaría una práctica con **hoja de cálculo** y **evidencias en GNU Octave**.

## Ejercicio de ejemplo

Raíz de \(f(x) = x^3 - x - 2\) en \([1, 2]\) con el **método de bisección** y tolerancia \(\varepsilon = 10^{-6}\).

## Contenido

| Ruta | Descripción |
|------|-------------|
| `excel/ejemplo_metodos_numericos_biseccion.xlsx` | Tabla con fórmulas en Excel |
| `octave/biseccion_ejemplo.m` | Script reproducible |
| `evidencias/salida_consola.txt` | Log de ejecución (tabla de iteraciones) |
| `evidencias/iteraciones_biseccion.csv` | Mismos datos en CSV |
| `evidencias/biseccion_fx_raiz.png` | Gráfica \(f(x)\) y raíz aproximada |

## Cómo reproducir (Octave)

Desde la raíz de este ejemplo:

```bash
cd ejemplos/metodos-numericos-prueba
octave --no-gui --quiet octave/biseccion_ejemplo.m
```

O:

```bash
bash octave/ejecutar_demo.sh
```

## Nota

Esta carpeta es solo **muestra de formato**; no forma parte del sistema de trading del repositorio principal.
