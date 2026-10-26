# Unidad 2 — teoría y prácticas de Excel

Material de clase de Métodos Numéricos (Mg. Jessica Pérez Rivera). Sirve para copiar la **forma** de la tabla y elegir el **método** según el enunciado.

Aquí solo hay guías teóricas en PDF y libros de Excel. No hay Octave, scripts ni capturas.

La forma se mira en el Excel (columnas, títulos, celdas en amarillo, fórmulas). El PDF dice qué problema es. Si el enunciado pide mínimos cuadrados, el molde es C, b y a, aunque una hoja se llame interpolación.

## Mapa

| Práctica | Hojas | Guía | Qué método muestra |
|---|---|---|---|
| `excel/01-sesion-6-metodos-directos.xlsx` | `directo;cramer`, `regresiva`, `inversa`, `gauss`, `factorizacion lu`, `cholesky` | `teoria/01-sesion-6-metodos-directos.pdf` | Cramer, sustitución regresiva e inversa con fórmulas. Gauss, LU y Cholesky en este libro quedaron como conclusión. El desarrollo con fórmulas de esos tres está en el taller. |
| `excel/11-taller-2.1.xlsx` | `ejercicio 1` a `ejercicio 4` | `teoria/06-taller-2.1-metodos-directos.pdf` | 1: inversa y Cramer. 2: Gauss. 3: LU. 4: Cholesky. |
| `excel/02-sesion-7-iterativos.xlsx` | `normas`, `condicional`, `Jacobi` | `teoria/02-sesion-7-metodos-iterativos.pdf` | Jacobi con fórmulas (B, C, iteraciones). Normas y condicionamiento como conclusión. |
| `excel/03-sesion-8-no-lineales.xlsx` | `Seidel`, `punto fijo`, `newton raphson` | `teoria/03-sesion-8-sistemas-no-lineales.pdf` | Seidel, punto fijo y Newton. Primero el despeje (o F y J) en la hoja. |
| `excel/04-sesion-8-newton-punto-fijo-ejercicios.xlsx` | `ejercico1A`, `Ejercicio1B`, `ejercicio1C`, `ejercicio2_metodo`, `ejercico2B`, `Ejercicio2C` | misma guía de la sesión 8 | Ejercicios extra de Newton y punto fijo. Los números están escritos; el molde con fórmulas vivas es el de la sesión 8 y el examen. |
| `excel/05-sesion-9-clase.xlsx` | `polinomial`, `lineal multiple`, `c a b` | `teoria/04-sesion-9-regresion-polinomial-multiple.pdf` | Regresión polinomial y lineal múltiple. C, b y a. |
| `excel/06-sesion-9-ajuste.xlsx` | `AJUSTE`, `REGRESION_POLINOMIAL`, `c a b`, `Regresion multiple` | misma guía de regresión | Ajuste por mínimos cuadrados. No pasa por todos los puntos. |
| `excel/07-sesion-9-ejercicio-dieta.xlsx` | `recta`, `parabola`, `comparacion` | misma guía de regresión | Misma forma de ajuste, otro enunciado. |
| `excel/08-sesion-9-regresion-tarea.xlsx` | `Ejercicio1` … `Ejercicio10` | misma guía de regresión | Diez ejercicios de regresión con el mismo molde. |
| `excel/09-sesion-10-lagrange-newton.xlsx` | `Ejemplo 1`, `Ejemplo 2`, `newton`, `ejercicio4.2` | `teoria/05-sesion-10-interpolacion.pdf` | Lagrange y Newton interpolante. La portada del PDF dice «Ajuste de Curvas»; el contenido de esta guía es interpolación. |
| `excel/10-sesion-10-tarea-semana5.xlsx` | `ejercicio 1`, `ejercicio4.2`, `ejercicio4.3` | misma guía de la sesión 10 | Lagrange y Newton. En 4.3, grado 3: Lagrange con los cuatro nodos y Newton regresivo (el x pedido está más cerca del último nodo). |
| `excel/12-examen-modelo-unidad-2.xlsx` | `Ejercicio 1` … `Ejercicio 5` | `teoria/07-examen-modelo-unidad-2.pdf` | 1 no lineal (Newton y punto fijo). 2 Jacobi y Seidel, reordenado a E.D.D. 3 Newton interpolante. 4 parábola por mínimos cuadrados. 5 Lagrange inverso (se intercambian x e y). |

## Cómo se distingue el método

- Varias ecuaciones lineales y pide inversa, Cramer, Gauss, LU o Cholesky: sesión 6 o taller 2.1.
- Jacobi, Seidel, normas o condicionamiento: sesión 7. Seidel de sistemas no lineales está en la sesión 8.
- Sistema con x², y² o productos: sesión 8. Newton usa F y J. Punto fijo usa el despeje G.
- Puntos (x, y) y la curva no tiene que pasar por todos: regresión, sesión 9. Columnas de potencias, fila suma, C, b, `a = inv(C)*b`, Q = (P − yi)².
- Pide el valor en un x y el polinomio pasa por los nodos elegidos: sesión 10. Lagrange con Lj, o Newton con diferencias. Los nodos son los más cercanos a ese x. El grado es el que pide el enunciado.
- Pide el x de un y dado: Lagrange inverso. Las y pasan a la columna x y las x a la columna y.

Amarillo donde va la entrada o la respuesta que ella marca (`x=`, `P`, `a`). Los anchos de número tienen que verse; si no, Excel muestra `######`.
