#!/usr/bin/env python3
"""Tarea: Modelos de Ensamblaje — este es el único archivo que debes completar.

Convenciones que se usan en toda la tarea (las mismas del enunciado):

* ``X`` es un arreglo de numpy de forma ``(n, J)``: una fila por
  observación, una columna por predictor.
* ``y`` es un arreglo de forma ``(n,)`` con valores en ``{-1, +1}``.
* Una división se escribe siempre ``x[j] <= t``: lo que cumple la condición
  cae en la rama **izquierda** y el resto en la **derecha**.
* Un árbol es un diccionario anidado. Una hoja trae
  ``{"hoja": True, "clase": c, "n": n}`` y un nodo interno trae
  ``{"hoja": False, "predictor": j, "umbral": t, "izquierda": ..., "derecha": ...}``.
  La función ``predecir_arbol`` de ``util.py`` recorre exactamente esa
  estructura: conviene leerla antes de escribir la Parte 3.

El archivo trae dos cosas. Primero la **sección de respuestas**, donde anotas
los números que imprime ``python experimento.py`` una vez que tu código está
listo. Debajo van las seis partes con las ocho funciones. Las dos cosas se
corrigen de forma automática y las dos viajan en este mismo archivo, que es lo
único que se entrega.

Solo se permiten la biblioteca estándar y numpy. No modifiques las firmas de
las funciones ni los nombres de las constantes: el corrector automático los
busca tal como están escritos.
"""
from __future__ import annotations

import numpy as np

from util import (
    clase_mayoritaria,
    mejor_tocon,
    predecir_arbol,
    predecir_tocon,
    umbrales_candidatos,
)


########################################################################
########################################################################
##
##  SECCIÓN DE RESPUESTAS
##
##  Acá van los números que produce tu propio código. No se calculan a
##  mano ni se copian del enunciado: se leen de las tablas que imprime
##
##      python experimento.py
##
##  una vez que las seis partes están completas. Cada constante lleva
##  encima el experimento y la columna de la que sale.
##
##  Cómo se comparan las respuestas:
##
##  * los números se comparan con TRES decimales, así que da lo mismo
##    anotar 0,1234 o 0,123;
##  * las claves de cada diccionario tienen que ser exactamente las que
##    dice el comentario, ni una más ni una menos.
##
##  `grader.py` no revisa esta sección: sus puntos los pone el corrector
##  del curso al momento de entregar. Debajo empieza el código, que es
##  lo que sí revisan las pruebas de `grader.py`.
##
########################################################################
########################################################################


############################################################
# Experimento 1. Un árbol solo: profundidad contra error
#
# Las columnas «hojas» y «err. reservado» de la primera tabla. Las
# claves son la primera columna tal cual: las cinco profundidades como
# texto y la última fila como 'sin tope'.
############################################################

# Cuántas hojas tiene el árbol de cada profundidad.
# BEGIN_TU_CODIGO
EXP1_HOJAS = {'1': 2, '2': 4, '3': 6, '5': 14, '8': 23, 'sin tope': 26}
# END_TU_CODIGO

# Error sobre el conjunto reservado de cada uno de esos árboles.
# BEGIN_TU_CODIGO
EXP1_ERROR_RESERVADO = {'1': 0.189, '2': 0.056, '3': 0.056, '5': 0.100, '8': 0.111, 'sin tope': 0.122}
# END_TU_CODIGO


############################################################
# Experimento 2. Bagging: la variabilidad del error al aumentar B
#
# Las columnas «err. reservado medio» y «desv. estándar» de la segunda
# tabla, con la cantidad de árboles como clave (entera).
############################################################

# Error reservado medio de las seis semillas, para cada B.
# BEGIN_TU_CODIGO
EXP2_ERROR_MEDIO = {1: 0.119, 3: 0.106, 9: 0.094, 27: 0.085}
# END_TU_CODIGO

# Desviación estándar de esos mismos seis errores.
# BEGIN_TU_CODIGO
EXP2_DESVIACION = {1: 0.0319, 3: 0.0190, 9: 0.0140, 27: 0.0052}
# END_TU_CODIGO


############################################################
# Experimento 3. Bagging contra bosque aleatorio
#
# De la tercera tabla se piden dos cosas: el error del árbol suelto y
# la fila de bagging, la de «todos (8)». Las filas con sorteo no se
# piden, porque dependen del orden en que cada implementación consume
# el generador al recorrer las ramas y no admiten una respuesta única.
############################################################

# Error reservado del árbol suelto, la línea que va sobre la tabla.
# BEGIN_TU_CODIGO
EXP3_ERROR_ARBOL_SOLO = 0.244
# END_TU_CODIGO

# La fila «todos (8)»: su error fuera de bolsa y su error reservado.
# BEGIN_TU_CODIGO
EXP3_BAGGING = {'oob': 0.214, 'reservado': 0.233}
# END_TU_CODIGO


############################################################
# Experimento 4. AdaBoost: rondas, pesos y voto final
#
# Las dos columnas de error de la cuarta tabla, con la cantidad de
# rondas como clave (entera).
############################################################

# Error de entrenamiento del ensamble de T rondas.
# BEGIN_TU_CODIGO
EXP4_ERROR_ENTRENAMIENTO = {1: 0.167, 3: 0.095, 10: 0.095, 30: 0.062, 80: 0.043}
# END_TU_CODIGO

# Error reservado de esos mismos ensambles.
# BEGIN_TU_CODIGO
EXP4_ERROR_RESERVADO = {1: 0.200, 3: 0.078, 10: 0.078, 30: 0.089, 80: 0.100}
# END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 1: la impureza de Gini
# ----------------------------------------------------------------------
def impureza_gini(y):
    """Impureza de Gini de un conjunto de etiquetas.

    Con ``p_k`` la proporción de la clase ``k`` dentro de ``y``::

        Gini(y) = 1 - sum_k p_k^2

    Vale ``0`` cuando todas las etiquetas son iguales (región pura) y llega
    a su máximo, ``0,5`` en el caso binario, cuando las dos clases están
    empatadas.

    Sobre un conjunto vacío devuelve ``0.0``: no hay incertidumbre que
    medir. Devuelve siempre un ``float`` de Python, no un arreglo.

    Ejemplo::

        >>> round(impureza_gini(np.array([1, 1, 1, -1])), 4)
        0.375
    """
    # BEGIN_TU_CODIGO
    y = np.asarray(y)
    n = y.size
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    proporciones = counts / n
    gini = 1.0 - float(np.sum(proporciones ** 2))
    return float(max(0.0, gini))
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 2: la mejor división
# ----------------------------------------------------------------------
def mejor_division(X, y, predictores=None):
    """Busca la división ``x[j] <= t`` que más reduce la impureza de Gini.

    La **ganancia** de una división que parte ``y`` en ``y_izq`` (con
    ``n_izq`` observaciones) e ``y_der`` (con ``n_der``) es::

        ganancia = Gini(y) - (n_izq/n) * Gini(y_izq) - (n_der/n) * Gini(y_der)

    Se prueban, para cada predictor candidato, todos los umbrales que
    producen particiones distintas: los puntos medios entre valores
    consecutivos distintos de esa columna. ``umbrales_candidatos`` de
    ``util.py`` ya los calcula.

    Argumentos:
        X, y: los datos, con las convenciones del módulo.
        predictores: secuencia de índices de columna que se pueden usar. Si
            es ``None`` se consideran todas. Este argumento es el que hace
            posible el bosque aleatorio de la Parte 4: ahí se le pasa un
            subconjunto sorteado.

    Devuelve la tupla ``(j, t, ganancia)``. Si ninguna división consigue
    ganancia **estrictamente positiva** (porque el nodo ya es puro, porque
    todas las columnas candidatas son constantes o porque ningún corte
    mejora nada), devuelve ``(-1, 0.0, 0.0)``.

    Recorre los predictores en el orden en que vienen y los umbrales de
    menor a mayor; ante un empate gana el primero que aparece. Así la
    función es determinista.
    """
    # BEGIN_TU_CODIGO
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = y.size
    if n < 2:
        return (-1, 0.0, 0.0)

    gini_nodo = impureza_gini(y)
    if gini_nodo == 0.0:
        return (-1, 0.0, 0.0)

    if predictores is None:
        columnas = range(X.shape[1])
    else:
        columnas = predictores

    mejor_j = -1
    mejor_t = 0.0
    mejor_ganancia = 0.0

    for j in columnas:
        columna = X[:, j]
        umbrales = umbrales_candidatos(columna)
        for t in umbrales:
            izq = columna <= t
            n_izq = int(np.sum(izq))
            n_der = n - n_izq
            if n_izq == 0 or n_der == 0:
                continue
            y_izq = y[izq]
            y_der = y[~izq]
            ganancia = gini_nodo - (n_izq / n) * impureza_gini(y_izq) - (n_der / n) * impureza_gini(y_der)
            if ganancia > mejor_ganancia + 1e-12:
                mejor_ganancia = float(ganancia)
                mejor_j = int(j)
                mejor_t = float(t)

    if mejor_j == -1:
        return (-1, 0.0, 0.0)
    return (mejor_j, mejor_t, mejor_ganancia)
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 3: el árbol de decisión
# ----------------------------------------------------------------------
def entrenar_arbol(X, y, profundidad_max=None, min_muestras_division=2,
                   m=None, rng=None):
    """Construye recursivamente un árbol de decisión.

    En cada nodo: si corresponde detenerse, se devuelve una **hoja** con la
    clase mayoritaria del nodo (``clase_mayoritaria`` de ``util.py`` la
    calcula, con ``+1`` como desempate). Si no, se busca la mejor división
    con ``mejor_division`` y se recurre sobre las dos mitades.

    Hay cuatro condiciones de parada, y basta que se cumpla una:

    1. el nodo es **puro** (todas sus etiquetas son iguales);
    2. el nodo tiene **menos de** ``min_muestras_division`` observaciones,
       es decir, es demasiado pequeño para que valga la pena partirlo;
    3. se agotó la **profundidad**: ``profundidad_max`` llegó a ``0``
       (``None`` significa sin límite);
    4. ninguna división da ganancia positiva, o sea ``mejor_division``
       devolvió ``j = -1``.

    Argumentos:
        profundidad_max: cuántos niveles de división quedan por debajo de
            este nodo. La recursión lo pasa disminuido en uno a cada hijo.
        min_muestras_division: tamaño mínimo de un nodo para intentar
            dividirlo.
        m: cuántos predictores se sortean en **cada** división. ``None`` (o
            un valor mayor o igual que ``J``) significa considerarlos todos,
            que es el árbol de siempre y también el de la Parte 4 cuando se
            usa como bagging. Con ``m < J`` se sortean ``m`` índices
            distintos y se le pasan ordenados a ``mejor_division``.
        rng: generador de numpy (``np.random.default_rng(...)``) para ese
            sorteo. Si es ``None`` se crea uno con semilla ``0``.

    Devuelve el árbol como diccionario, con la estructura que documenta
    ``predecir_arbol``.
    """
    # BEGIN_TU_CODIGO
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n = y.size

    if n == 0:
        return {"hoja": True, "clase": 1, "n": 0}

    es_puro = (len(np.unique(y)) <= 1)
    muy_pequeno = (n < min_muestras_division)
    limite_alcanzado = (profundidad_max is not None and profundidad_max <= 0)

    if es_puro or muy_pequeno or limite_alcanzado:
        return {
            "hoja": True,
            "clase": int(clase_mayoritaria(y)),
            "n": int(n),
        }

    J = X.shape[1]
    if m is not None and m < J:
        if rng is None:
            rng = np.random.default_rng(0)
        predictores = np.sort(rng.choice(J, size=m, replace=False))
    else:
        predictores = None

    j, t, ganancia = mejor_division(X, y, predictores=predictores)
    if j == -1:
        return {
            "hoja": True,
            "clase": int(clase_mayoritaria(y)),
            "n": int(n),
        }

    izq_idx = X[:, j] <= t
    der_idx = ~izq_idx

    prof_hijo = None if profundidad_max is None else profundidad_max - 1

    sub_izq = entrenar_arbol(
        X[izq_idx], y[izq_idx],
        profundidad_max=prof_hijo,
        min_muestras_division=min_muestras_division,
        m=m,
        rng=rng,
    )
    sub_der = entrenar_arbol(
        X[der_idx], y[der_idx],
        profundidad_max=prof_hijo,
        min_muestras_division=min_muestras_division,
        m=m,
        rng=rng,
    )

    return {
        "hoja": False,
        "predictor": int(j),
        "umbral": float(t),
        "izquierda": sub_izq,
        "derecha": sub_der,
    }
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 4: bagging, bosque aleatorio y voto de mayoría
# ----------------------------------------------------------------------
def predecir_ensamble(arboles, X):
    """Voto de mayoría de una lista de árboles sobre cada fila de ``X``.

    Cada árbol responde ``+1`` o ``-1``; gana la etiqueta más votada. Con
    etiquetas simétricas eso es el signo de la suma de los votos, y el
    empate (posible cuando la cantidad de árboles es par) se resuelve a
    favor de ``+1``.

    Devuelve un arreglo de forma ``(n,)`` con valores en ``{-1, +1}``. Con
    la lista vacía devuelve todo ``+1``.
    """
    # BEGIN_TU_CODIGO
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if len(arboles) == 0:
        return np.ones(n, dtype=int)

    votos = np.zeros(n, dtype=int)
    for arbol in arboles:
        votos += predecir_arbol(arbol, X)

    return np.where(votos >= 0, 1, -1).astype(int)
    # END_TU_CODIGO


def entrenar_bosque(X, y, B=25, m=None, semilla=0, profundidad_max=None,
                    min_muestras_division=2):
    """Entrena ``B`` árboles sobre ``B`` muestras bootstrap.

    Una **muestra bootstrap** de un conjunto de ``n`` observaciones se
    obtiene sorteando ``n`` índices en ``[0, n)`` **con reemplazo**: algunas
    observaciones salen repetidas y otras quedan fuera de la bolsa. Con
    ``m = None`` esto es exactamente bagging; con ``m < J`` es un bosque
    aleatorio, porque cada división de cada árbol solo puede mirar ``m``
    predictores sorteados.

    Usa un único generador ``rng = np.random.default_rng(semilla)`` para
    todo, y en cada vuelta primero sortea los índices de la bolsa y después
    entrena el árbol pasándole ese mismo ``rng``. Así toda la corrida queda
    determinada por ``semilla``.

    Devuelve un diccionario con exactamente estas cuatro claves:

    ==============  ==================================================
    ``"arboles"``   lista con los ``B`` árboles entrenados
    ``"bolsas"``    lista con los ``B`` arreglos de índices sorteados,
                    cada uno de forma ``(n,)``; son los que la Parte 5
                    necesita para saber qué observación vio cada árbol
    ``"B"``         entero: cuántos árboles trae el ensamble
    ``"m"``         el ``m`` con que se entrenó (puede ser ``None``)
    ==============  ==================================================
    """
    # BEGIN_TU_CODIGO
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n = y.size

    rng = np.random.default_rng(semilla)
    arboles = []
    bolsas = []

    for _ in range(B):
        bolsa = rng.integers(0, n, size=n)
        arbol = entrenar_arbol(
            X[bolsa], y[bolsa],
            profundidad_max=profundidad_max,
            min_muestras_division=min_muestras_division,
            m=m,
            rng=rng,
        )
        arboles.append(arbol)
        bolsas.append(bolsa)

    return {
        "arboles": arboles,
        "bolsas": bolsas,
        "B": int(B),
        "m": m,
    }
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 5: el error fuera de bolsa
# ----------------------------------------------------------------------
def error_oob(bosque, X, y):
    """Error fuera de bolsa del ensamble sobre los datos con que se entrenó.

    Para cada observación ``i`` se vota **solo** entre los árboles que no la
    tuvieron en su muestra bootstrap: son los que nunca la vieron, así que
    para ellos ``i`` es un dato nuevo. Se compara ese voto con ``y[i]`` y se
    devuelve la fracción de desaciertos.

    Las observaciones que quedaron dentro de la bolsa de **todos** los
    árboles no se pueden evaluar y simplemente no entran en el promedio. Si
    no queda ninguna observación evaluable, devuelve ``float("nan")``.

    El voto se resuelve igual que en ``predecir_ensamble``: signo de la suma
    de los votos, con ``+1`` en caso de empate.

    Devuelve un ``float`` entre 0 y 1.
    """
    # BEGIN_TU_CODIGO
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y).astype(int)
    n = y.size

    arboles = bosque["arboles"]
    bolsas = bosque["bolsas"]

    votos = np.zeros(n, dtype=int)
    conteo = np.zeros(n, dtype=int)

    for arbol, bolsa in zip(arboles, bolsas):
        fuera = np.ones(n, dtype=bool)
        fuera[bolsa] = False
        indices_fuera = np.where(fuera)[0]
        if indices_fuera.size == 0:
            continue
        pred = predecir_arbol(arbol, X[indices_fuera])
        votos[indices_fuera] += pred
        conteo[indices_fuera] += 1

    evaluables = conteo > 0
    if not np.any(evaluables):
        return float("nan")

    prediccion_oob = np.where(votos[evaluables] >= 0, 1, -1)
    desaciertos = np.sum(prediccion_oob != y[evaluables])
    return float(desaciertos / np.sum(evaluables))
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 6: AdaBoost con tocones de decisión
# ----------------------------------------------------------------------
def entrenar_adaboost(X, y, T=10):
    """AdaBoost sobre tocones de decisión.

    En cada una de las ``T`` rondas:

    1. se ajusta un tocón a los datos **ponderados** por ``pesos``, con
       ``mejor_tocon(X, y, pesos)`` de ``util.py``;
    2. se calcula su **error ponderado**
       ``eps = sum(pesos[predicción != y])`` (los pesos suman 1, así que
       ``eps`` es una fracción);
    3. se calcula el peso de la ronda::

           alfa = (1/2) * ln((1 - eps) / eps)

    4. se actualizan los pesos y se renormalizan para que vuelvan a sumar
       uno::

           pesos <- pesos * exp(-alfa * y * predicción)
           pesos <- pesos / suma(pesos)

    Los pesos parten uniformes, en ``1/n``. Un ejemplo bien clasificado
    tiene ``y * predicción = +1`` y su peso se multiplica por
    ``exp(-alfa) < 1``: se encoge. Uno mal clasificado engorda. Por eso la
    ronda siguiente se concentra en lo que la anterior no resolvió.

    Dos resguardos numéricos:

    * si ``eps >= 0.5`` el tocón no es mejor que lanzar una moneda: el bucle
      se detiene sin agregarlo (``alfa`` sería negativo o cero);
    * ``eps`` se recorta al intervalo ``[1e-10, 1 - 1e-10]`` antes de entrar
      en el logaritmo, para que un tocón perfecto no divida por cero.

    Devuelve un diccionario con exactamente estas tres claves, las tres
    listas del mismo largo (que puede ser menor que ``T`` si el bucle se
    detuvo antes):

    ===============  ================================================
    ``"tocones"``    los tocones ajustados, en orden
    ``"alfas"``      el ``alfa`` de cada ronda, como ``float``
    ``"errores"``    el error ponderado de cada ronda, como ``float``
    ===============  ================================================
    """
    # BEGIN_TU_CODIGO
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n = y.size

    if T <= 0 or n == 0:
        return {"tocones": [], "alfas": [], "errores": []}

    pesos = np.full(n, 1.0 / n, dtype=float)
    tocones = []
    alfas = []
    errores = []

    for _ in range(T):
        tocon = mejor_tocon(X, y, pesos)
        pred = predecir_tocon(tocon, X)
        eps = float(np.sum(pesos[pred != y]))

        if eps >= 0.5:
            break

        eps_acotado = float(np.clip(eps, 1e-10, 1.0 - 1e-10))
        alfa = float(0.5 * np.log((1.0 - eps_acotado) / eps_acotado))

        tocones.append(tocon)
        alfas.append(alfa)
        errores.append(eps)

        pesos = pesos * np.exp(-alfa * y * pred)
        pesos = pesos / np.sum(pesos)

    return {
        "tocones": tocones,
        "alfas": alfas,
        "errores": errores,
    }
    # END_TU_CODIGO


def predecir_adaboost(modelo, X):
    """Voto ponderado del ensamble de AdaBoost sobre cada fila de ``X``.

    El puntaje de una fila es ``sum_t alfa_t * tocon_t(x)`` y la etiqueta es
    su signo, con ``+1`` en caso de empate. Un ensamble sin tocones responde
    todo ``+1``.

    Devuelve un arreglo de forma ``(n,)`` con valores en ``{-1, +1}``.
    """
    # BEGIN_TU_CODIGO
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    tocones = modelo["tocones"]
    alfas = modelo["alfas"]

    if len(tocones) == 0:
        return np.ones(n, dtype=int)

    puntaje = np.zeros(n, dtype=float)
    for tocon, alfa in zip(tocones, alfas):
        puntaje += alfa * predecir_tocon(tocon, X)

    return np.where(puntaje >= 0.0, 1, -1).astype(int)
    # END_TU_CODIGO


if __name__ == "__main__":
    print("Este archivo solo define constantes y funciones.")
    print("Para revisar tu código:           python grader.py")
    print("Para obtener las cuatro tablas:   python experimento.py")
