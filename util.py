"""Utilidades de la tarea «Modelos de Ensamblaje». No hay que modificar este archivo.

Aquí viven los conjuntos de datos (todos generados localmente con semilla
fija: la tarea no descarga nada) y las piezas de apoyo que ya vienen
escritas:

* ``predecir_arbol``: recorre un árbol como el que construyes en la Parte 3.
  Fija el contrato de la estructura, así que conviene leerlo antes de
  escribir ``entrenar_arbol``.
* ``mejor_tocon`` y ``predecir_tocon``: el clasificador simple sobre el que
  se apoya AdaBoost en la Parte 6. Te lo damos hecho para que esa parte se
  concentre en la mecánica del refuerzo (pesos, error ponderado y alfa) y no
  en volver a buscar umbrales.

Convenciones de toda la tarea:

* ``X`` es un arreglo de numpy de forma ``(n, J)``: una fila por
  observación, una columna por predictor.
* ``y`` es un arreglo de forma ``(n,)`` con valores en ``{-1, +1}``.
* Una división se escribe siempre ``x[j] <= t``: lo que cumple la condición
  cae en la rama izquierda y el resto en la derecha.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "clase_mayoritaria",
    "datos_ejemplo",
    "datos_lunas",
    "datos_tabla",
    "dividir",
    "exactitud",
    "mejor_tocon",
    "predecir_arbol",
    "predecir_tocon",
    "umbrales_candidatos",
]


# ----------------------------------------------------------------------
# Piezas de apoyo
# ----------------------------------------------------------------------
def clase_mayoritaria(y):
    """Clase más frecuente en ``y``, con ``+1`` como desempate.

    Con etiquetas en ``{-1, +1}`` basta mirar el signo de la suma: si hay
    tantas positivas como negativas la suma es cero y devolvemos ``+1``.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 1
    return 1 if float(np.sum(y)) >= 0.0 else -1


def umbrales_candidatos(columna):
    """Umbrales que producen particiones distintas sobre una columna.

    Son los puntos medios entre valores consecutivos **distintos**. Si la
    columna es constante no hay ninguno y se devuelve un arreglo vacío.
    """
    valores = np.unique(np.asarray(columna, dtype=float))
    if valores.size < 2:
        return np.empty(0, dtype=float)
    return (valores[:-1] + valores[1:]) / 2.0


def predecir_arbol(arbol, X):
    """Etiqueta que el árbol asigna a cada fila de ``X``.

    La estructura de un árbol es un diccionario. Una **hoja** trae::

        {"hoja": True, "clase": +1 o -1, "n": cantidad de observaciones}

    y un **nodo interno** trae::

        {"hoja": False, "predictor": j, "umbral": t,
         "izquierda": subárbol de x[j] <= t,
         "derecha":   subárbol de x[j] >  t}

    Devuelve un arreglo de forma ``(n,)`` con valores en ``{-1, +1}``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    salida = np.empty(X.shape[0], dtype=int)
    for i, fila in enumerate(X):
        nodo = arbol
        while not nodo["hoja"]:
            if fila[nodo["predictor"]] <= nodo["umbral"]:
                nodo = nodo["izquierda"]
            else:
                nodo = nodo["derecha"]
        salida[i] = int(nodo["clase"])
    return salida


def mejor_tocon(X, y, pesos):
    """Tocón de decisión que minimiza el error ponderado.

    Un **tocón** es un árbol de una sola división: mira un predictor ``j``,
    lo compara con un umbral ``t`` y responde ``signo`` a la izquierda y
    ``-signo`` a la derecha. El ``signo`` es lo que le permite atacar
    también los casos en que la clase positiva está al otro lado del corte.

    Se recorren todos los predictores, todos los umbrales candidatos y los
    dos signos, y se devuelve el de menor error ponderado
    ``sum(pesos[predicción != y])``. Los empates se resuelven por el primero
    que aparece, así que la función es determinista.

    Devuelve ``{"predictor": j, "umbral": t, "signo": s}``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    pesos = np.asarray(pesos, dtype=float)

    mejor = {"predictor": 0, "umbral": 0.0, "signo": 1}
    mejor_error = np.inf
    for j in range(X.shape[1]):
        columna = X[:, j]
        for t in umbrales_candidatos(columna):
            izquierda = columna <= t
            for signo in (1, -1):
                prediccion = np.where(izquierda, signo, -signo)
                error = float(np.sum(pesos[prediccion != y]))
                if error < mejor_error - 1e-12:
                    mejor_error = error
                    mejor = {"predictor": int(j), "umbral": float(t),
                             "signo": int(signo)}
    return mejor


def predecir_tocon(tocon, X):
    """Etiqueta que el tocón asigna a cada fila de ``X``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    izquierda = X[:, tocon["predictor"]] <= tocon["umbral"]
    signo = int(tocon["signo"])
    return np.where(izquierda, signo, -signo).astype(int)


def exactitud(y_verdadero, y_predicho):
    """Fracción de aciertos, como ``float`` entre 0 y 1."""
    y_verdadero = np.asarray(y_verdadero).astype(int)
    y_predicho = np.asarray(y_predicho).astype(int)
    return float(np.mean(y_verdadero == y_predicho))


# ----------------------------------------------------------------------
# Conjuntos de datos
# ----------------------------------------------------------------------
def datos_ejemplo():
    """Las diez observaciones del ejemplo trabajado a mano del enunciado.

    Dos predictores. El primero ordena las observaciones de 1 a 10 y separa
    bastante bien las clases; el segundo alterna 0 y 1 y no informa nada.

    Devuelve ``(X, y)`` con ``X`` de forma ``(10, 2)`` y ``y`` de forma
    ``(10,)``.
    """
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [6.0, 1.0],
            [7.0, 0.0],
            [8.0, 1.0],
            [9.0, 0.0],
            [10.0, 1.0],
        ]
    )
    y = np.array([1, 1, 1, 1, -1, 1, -1, -1, 1, -1])
    return X, y


def datos_lunas(n=200, ruido=0.20, semilla=0):
    """Dos medias lunas entrelazadas en el plano: no separables por una recta.

    Es el conjunto donde se ve para qué sirve un árbol: la frontera es
    curva, así que ninguna recta la resuelve, pero una escalera de cortes
    verticales y horizontales sí se le acerca.

    Devuelve ``(X, y)`` con ``X`` de forma ``(n, 2)`` y ``y`` de forma
    ``(n,)`` con valores en ``{-1, +1}``.
    """
    rng = np.random.default_rng(semilla)
    n_positivas = n // 2
    n_negativas = n - n_positivas

    angulo = np.linspace(0.0, np.pi, n_positivas)
    luna_alta = np.column_stack([np.cos(angulo), np.sin(angulo)])
    angulo = np.linspace(0.0, np.pi, n_negativas)
    luna_baja = np.column_stack([1.0 - np.cos(angulo), 0.5 - np.sin(angulo)])

    X = np.vstack([luna_alta, luna_baja])
    X = X + rng.normal(0.0, float(ruido), size=X.shape)
    y = np.concatenate([np.ones(n_positivas, dtype=int),
                        -np.ones(n_negativas, dtype=int)])

    orden = rng.permutation(n)
    return X[orden], y[orden]


def datos_tabla(n=300, semilla=1):
    """Conjunto tabular pequeño con un predictor claramente dominante.

    Son ocho predictores normales estándar e independientes. La etiqueta
    depende con fuerza del predictor 0, algo del 1 y del 2, y nada de los
    cinco restantes, que son ruido puro. Esa asimetría es justamente la
    situación en que el sorteo de predictores del bosque tiene algo que
    aportar: sin él, casi todos los árboles parten cortando por el
    predictor 0 y terminan pareciéndose demasiado.

    Devuelve ``(X, y)`` con ``X`` de forma ``(n, 8)`` y ``y`` de forma
    ``(n,)`` con valores en ``{-1, +1}``.
    """
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(int(n), 8))
    puntaje = 2.5 * X[:, 0] + 0.9 * X[:, 1] - 0.7 * X[:, 2]
    probabilidad = 1.0 / (1.0 + np.exp(-puntaje))
    y = np.where(rng.random(int(n)) < probabilidad, 1, -1).astype(int)
    return X, y


def dividir(X, y, proporcion_reservada=0.3, semilla=0):
    """Separa los datos en una parte de entrenamiento y una reservada.

    La permutación depende solo de ``semilla``, así que la partición es la
    misma en todas las corridas.

    Devuelve ``(X_entrena, y_entrena, X_reservado, y_reservado)``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n = y.shape[0]
    orden = np.random.default_rng(semilla).permutation(n)
    corte = int(round(n * (1.0 - float(proporcion_reservada))))
    entrena, reservado = orden[:corte], orden[corte:]
    return X[entrena], y[entrena], X[reservado], y[reservado]
