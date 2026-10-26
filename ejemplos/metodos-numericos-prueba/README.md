# Simulación entrega U2 — Excel multihoja + Octave

Ejemplo de entrega **completa**: una **hoja por ejercicio**, **imagen de Octave incrustada** en cada hoja y filas de **prueba** (Excel vs Octave).

Inspirado en el examen modelo U2 (ejercicios 4 y 5 del PDF: parábola por mínimos cuadrados e integración).

## Archivo principal

`excel/simulacion_examen_u2_multihoja.xlsx`

| Hoja | Contenido |
|------|-----------|
| **Indice** | Lista de ejercicios y enlace conceptual al PDF modelo |
| **Ejercicio_1** | Bisección + tabla + prueba + gráfica Octave |
| **Ejercicio_2** | Newton-Raphson + prueba + gráfica |
| **Ejercicio_3** | Gauss 3×3 + residual + gráfica |
| **Ejercicio_4** | Mínimos cuadrados (parábola) + ŷ Excel vs Octave |
| **Ejercicio_5** | Trapecio + errores + gráfica |

## Regenerar todo (Octave → Excel)

```bash
cd ejemplos/metodos-numericos-prueba
bash octave/ejecutar_todos.sh
python3 scripts/generar_entrega_simulacion.py
```

Evidencias por ejercicio en `evidencias/ejercicio_XX_*/` (`resultado.txt`, `octave_grafica.png`).

## Nota

Simulación con datos de ejemplo; al recibir tu enunciado real se sustituyen datos y se mantiene el mismo formato (1 hoja = 1 ejercicio).
