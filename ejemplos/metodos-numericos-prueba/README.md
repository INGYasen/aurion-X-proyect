# Examen modelo U2 — Excel con despeje y evidencia Octave

Entrega del examen modelo de Métodos Numéricos (Mg. Jessica Perez Rivera). Una hoja de Excel por ejercicio. En cada hoja está el enunciado, el desarrollo para copiar (despejes y fracciones), las fórmulas que arman cada número, las gráficas de Octave y el log completo.

## Archivo para abrir

`excel/examen_modelo_u2_completo.xlsx`

| Hoja | Ejercicio del PDF |
|------|-------------------|
| `00_Indice` | Qué hay en cada hoja |
| `00_Validacion` | **15 pruebas enlazadas**; debe decir **INTEGRADO SIN FALLAS** al abrir en Excel |
| `Ej1_No_lineal` | Sistema no lineal: Newton, punto fijo, sustitución |
| `Ej2_Jacobi_Seidel` | Reordenar a E.D.D., Jacobi y Gauss-Seidel |
| `Ej3_Newton_interp` | Catenaria, polinomio de Newton, flecha en el punto medio |
| `Ej4_Minimos` | Tubo de Pitot, parábola por mínimos cuadrados |
| `Ej5_Lagrange` | Interpolación inversa, Lagrange de grado 2 |

Las celdas en amarillo son el desarrollo a mano. Las naranjas son la respuesta. Las verdes son la corrida de Octave.

## Regenerar

```bash
cd ejemplos/metodos-numericos-prueba
bash octave/ejecutar_todos.sh
python3 scripts/generar_examen_u2.py
```

Las imágenes y los logs quedan en `evidencias/ejercicio_0X_*/`.
