# Contexto 1: Resolución de Desafíos - Modelos de Ensamblaje
**Fecha de registro:** 2026-10-15

## 1. Instrucciones
* Completar únicamente el archivo `submission.py` resolviendo los 8 desafíos planteados en las 6 partes de la tarea.
* No modificar las firmas de las funciones ni los nombres de las constantes.
* Utilizar únicamente la biblioteca estándar de Python y `numpy`.
* Desarrollar soluciones generales y robustas, no adaptadas exclusivamente a los casos visibles del autograder (`grader.py`), considerando casos reservados, casos límite y de escala.
* Correr `grader.py` para validar las pruebas visibles (60/60 puntos).
* Ejecutar `experimento.py` para reproducir los cuatro experimentos y llenar las constantes de la sección de respuestas al inicio de `submission.py` (40 puntos restantes).
* Garantizar que no se generen archivos `.pyc` ni se modifiquen `util.py`, `experimento.py` ni `grader.py`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar las ocho funciones de modelos de ensamblaje en `submission.py` y completar la sección de respuestas con los resultados de los experimentos.
* **Fases/Hitos principales:**
  1. **Parte 1 (Gini):** Implementar `impureza_gini(y)` manejando casos puros, mezclas, empates y arrays vacíos.
  2. **Parte 2 (Mejor División):** Implementar `mejor_division(X, y, predictores)` evaluando umbrales candidatos de util.py con tolerancia numérica a empates (1e-12) y ganancia estrictamente positiva.
  3. **Parte 3 (Árbol de Decisión):** Implementar `entrenar_arbol(X, y, ...)` con 4 condiciones de parada, subconjuntos de predictores ordenados y manejo cuidadoso del generador `rng`.
  4. **Parte 4 (Bagging y Bosque Aleatorio):** Implementar `predecir_ensamble(arboles, X)` y `entrenar_bosque(X, y, B, m, semilla, ...)` usando `rng.integers(0, n, size=n)` para bootstrap con reemplazo.
  5. **Parte 5 (Error Fuera de Bolsa):** Implementar `error_oob(bosque, X, y)` acumulando predicciones solo de árboles que excluyeron cada observación y manejando casos sin observaciones evaluables (`NaN`).
  6. **Parte 6 (AdaBoost):** Implementar `entrenar_adaboost(X, y, T)` y `predecir_adaboost(modelo, X)` con actualización y normalización de pesos, recorte numérico de epsilon y parada por `eps >= 0.5`.
  7. **Ejecución y Verificación:** Validar con `grader.py` (60/60 puntos), ejecutar `experimento.py` y actualizar las constantes de la sección de respuestas.
* **Entregables:**
  * `submission.py` con las 8 funciones implementadas y las 8 constantes de experimentos completadas.

## 3. Referencias
* **Path del plan original:** `submission.py`
