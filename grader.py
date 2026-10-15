#!/usr/bin/env python3
"""Autograder de la tarea «Modelos de Ensamblaje».

Se ejecuta sin conexión a internet y sin más dependencias que numpy::

    python grader.py

Imprime el detalle de cada prueba, el subtotal por parte y el puntaje final.

Estas son las pruebas visibles: las que puedes correr mientras trabajas.
Suman 60 de los 100 puntos de la tarea y revisan el código de
``submission.py``, con al menos un caso básico de cada función. Los 40
restantes los pone el corrector del curso, que lee la sección de respuestas
del comienzo de ``submission.py`` y vuelve a probar esas mismas ocho
funciones en casos límite y de escala. Toda la tarea se corrige de forma
automática.
"""
from __future__ import annotations

import os
import signal
import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from util import (  # noqa: E402
    datos_ejemplo,
    datos_lunas,
    datos_tabla,
    dividir,
    exactitud,
    predecir_arbol,
    predecir_tocon,
)

import submission as sub  # noqa: E402


# ----------------------------------------------------------------------
# Infraestructura de pruebas
# ----------------------------------------------------------------------
PARTES = [
    (1, "La impureza de Gini", 8),
    (2, "La mejor división", 13),
    (3, "El árbol de decisión", 17),
    (4, "Bagging, bosque aleatorio y voto de mayoría", 17),
    (5, "El error fuera de bolsa", 13),
    (6, "AdaBoost con tocones de decisión", 17),
    (7, "La sección de respuestas", 15),
]
PUNTAJE_TOTAL = sum(puntos for _, _, puntos in PARTES)

PRUEBAS = []


def prueba(parte, descripcion, puntos, extra=False):
    """Decorador que registra una función de prueba."""

    def decorador(funcion):
        PRUEBAS.append(
            {
                "parte": parte,
                "descripcion": descripcion,
                "puntos": puntos,
                "funcion": funcion,
                "extra": extra,
            }
        )
        return funcion

    return decorador


def cerca(obtenido, esperado, tol=1e-9):
    """Compara arreglos o escalares con tolerancia absoluta."""
    return np.allclose(
        np.asarray(obtenido, dtype=float),
        np.asarray(esperado, dtype=float),
        rtol=0.0,
        atol=tol,
    )


def afirmar(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)


def hoja(clase, n=1):
    """Un árbol de una sola hoja, para armar ensambles de juguete."""
    return {"hoja": True, "clase": int(clase), "n": int(n)}


def profundidad(arbol):
    """Cuántos niveles de división tiene el árbol (una hoja sola vale 0)."""
    if arbol["hoja"]:
        return 0
    return 1 + max(profundidad(arbol["izquierda"]), profundidad(arbol["derecha"]))


def hojas(arbol):
    """Lista de las hojas del árbol, de izquierda a derecha."""
    if arbol["hoja"]:
        return [arbol]
    return hojas(arbol["izquierda"]) + hojas(arbol["derecha"])


def predictores_usados(arbol):
    """Conjunto de predictores que aparecen en las divisiones del árbol."""
    if arbol["hoja"]:
        return set()
    return ({int(arbol["predictor"])}
            | predictores_usados(arbol["izquierda"])
            | predictores_usados(arbol["derecha"]))


def revisar_estructura(arbol, ruta="raíz"):
    """Comprueba que el árbol cumpla el contrato del diccionario."""
    afirmar(isinstance(arbol, dict),
            f"cada nodo debe ser un diccionario; en {ruta} vino "
            f"{type(arbol).__name__}")
    afirmar("hoja" in arbol, f"al nodo {ruta} le falta la clave «hoja»")
    if arbol["hoja"]:
        afirmar(set(arbol) == {"hoja", "clase", "n"},
                f"una hoja lleva exactamente las claves hoja, clase y n; "
                f"en {ruta} trajo {sorted(arbol)}")
        afirmar(int(arbol["clase"]) in (-1, 1),
                f"la clase de una hoja es +1 o -1; en {ruta} es "
                f"{arbol['clase']}")
        return
    afirmar(
        set(arbol) == {"hoja", "predictor", "umbral", "izquierda", "derecha"},
        f"un nodo interno lleva exactamente las claves hoja, predictor, "
        f"umbral, izquierda y derecha; en {ruta} trajo {sorted(arbol)}",
    )
    revisar_estructura(arbol["izquierda"], f"{ruta}.izquierda")
    revisar_estructura(arbol["derecha"], f"{ruta}.derecha")


# ----------------------------------------------------------------------
# Parte 1: la impureza de Gini
# ----------------------------------------------------------------------
@prueba(1, "impureza_gini: región pura, empate y mezcla desbalanceada", 4)
def p1_valores():
    casos = [
        (np.array([1, 1, 1, 1]), 0.0, "una región pura tiene impureza 0"),
        (np.array([-1, -1]), 0.0, "una región pura tiene impureza 0"),
        (np.array([1, -1]), 0.5,
         "con las dos clases empatadas la impureza binaria llega a su máximo, 0,5"),
        (np.array([1, 1, 1, -1]), 0.375, "con 3 de 4 positivas es 1 - 0,75^2 - 0,25^2"),
    ]
    for y, esperado, explicacion in casos:
        obtenido = sub.impureza_gini(y)
        afirmar(
            cerca(obtenido, esperado),
            f"con y = {list(y)} la impureza vale {esperado} ({explicacion}); "
            f"dio {obtenido}",
        )
    afirmar(
        isinstance(sub.impureza_gini(np.array([1, -1])), float),
        "impureza_gini debe devolver un float de Python, no un arreglo de numpy",
    )


# ----------------------------------------------------------------------
# Parte 2: la mejor división
# ----------------------------------------------------------------------
@prueba(2, "mejor_division: el corte obvio de un conjunto ordenado", 4)
def p2_corte_obvio():
    # Conjunto propio del corrector: seis observaciones separadas por x[0] = 3,5.
    X = np.array([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0],
                  [4.0, 7.0], [5.0, 8.0], [6.0, 9.0]])
    y = np.array([1, 1, 1, -1, -1, -1])
    j, t, ganancia = sub.mejor_division(X, y)
    afirmar(j == 0,
            f"el predictor 0 separa perfectamente las clases: se esperaba j = 0 "
            f"y dio {j}")
    afirmar(
        cerca(t, 3.5),
        f"el umbral es el punto medio entre 3 y 4, o sea 3,5; dio {t}",
    )
    afirmar(
        cerca(ganancia, 0.5),
        f"la impureza pasa de 0,5 a 0: la ganancia es 0,5 y dio {ganancia}",
    )


@prueba(2, "mejor_division: el ejemplo trabajado del enunciado", 3)
def p2_ejemplo():
    X, y = datos_ejemplo()
    copia_X, copia_y = X.copy(), y.copy()
    j, t, ganancia = sub.mejor_division(X, y)
    afirmar(
        j == 0 and cerca(t, 4.5),
        f"sobre el ejemplo del enunciado la mejor división es x[0] <= 4,5; "
        f"dio x[{j}] <= {t}",
    )
    afirmar(
        cerca(ganancia, 0.48 - 0.4 * 0.0 - 0.6 * (4.0 / 9.0), 1e-9),
        f"la ganancia del ejemplo es 0,48 - 0,4·0 - 0,6·(4/9) = 0,2133…; "
        f"dio {ganancia}",
    )
    afirmar(cerca(X, copia_X, 0.0) and cerca(y, copia_y, 0.0),
            "mejor_division no debe modificar X ni y")


@prueba(2, "mejor_division: cuándo no hay división que valga la pena", 3)
def p2_sin_division():
    X, y = datos_ejemplo()
    j, t, ganancia = sub.mejor_division(X, y, predictores=[1])
    afirmar(
        j == -1 and cerca(ganancia, 0.0),
        "el predictor 1 del ejemplo alterna 0 y 1 sin informar nada: ninguna "
        f"división gana algo, así que corresponde (-1, 0.0, 0.0) y dio "
        f"({j}, {t}, {ganancia})",
    )

    puro = np.array([[1.0], [2.0], [3.0]])
    j, _, ganancia = sub.mejor_division(puro, np.array([1, 1, 1]))
    afirmar(j == -1 and cerca(ganancia, 0.0),
            f"un nodo puro no admite ninguna división con ganancia positiva; "
            f"dio j = {j} con ganancia {ganancia}")

    constante = np.array([[5.0], [5.0], [5.0], [5.0]])
    j, _, _ = sub.mejor_division(constante, np.array([1, 1, -1, -1]))
    afirmar(j == -1,
            "con una columna constante no hay ningún umbral que produzca dos "
            f"lados no vacíos: corresponde j = -1 y dio {j}")


# ----------------------------------------------------------------------
# Parte 3: el árbol de decisión
# ----------------------------------------------------------------------
@prueba(3, "entrenar_arbol: el tocón del ejemplo, nodo por nodo", 4)
def p3_tocon():
    X, y = datos_ejemplo()
    arbol = sub.entrenar_arbol(X, y, profundidad_max=1)
    revisar_estructura(arbol)
    afirmar(not arbol["hoja"], "con profundidad_max = 1 la raíz debe dividir")
    afirmar(
        int(arbol["predictor"]) == 0 and cerca(arbol["umbral"], 4.5),
        f"la raíz debe cortar por x[0] <= 4,5; dio x[{arbol['predictor']}] <= "
        f"{arbol['umbral']}",
    )
    izquierda, derecha = arbol["izquierda"], arbol["derecha"]
    afirmar(
        izquierda["hoja"] and derecha["hoja"],
        "con profundidad_max = 1 los dos hijos deben ser hojas: la recursión "
        "pasa profundidad_max - 1 y a los hijos les toca 0",
    )
    afirmar(
        int(izquierda["clase"]) == 1 and int(izquierda["n"]) == 4,
        f"la rama x[0] <= 4,5 recoge 4 observaciones, todas positivas; dio "
        f"clase {izquierda['clase']} con n = {izquierda['n']}",
    )
    afirmar(
        int(derecha["clase"]) == -1 and int(derecha["n"]) == 6,
        f"la rama x[0] > 4,5 recoge 6 observaciones con mayoría negativa; dio "
        f"clase {derecha['clase']} con n = {derecha['n']}",
    )


@prueba(3, "entrenar_arbol: sin límites llega a hojas puras", 3)
def p3_completo():
    X, y = datos_ejemplo()
    copia_X, copia_y = X.copy(), y.copy()
    arbol = sub.entrenar_arbol(X, y)
    revisar_estructura(arbol)
    prediccion = predecir_arbol(arbol, X)
    afirmar(
        cerca(prediccion, y, 0.0),
        "sin límite de profundidad el árbol debe clasificar sin errores los "
        f"datos con que se entrenó; se equivocó en "
        f"{int(np.sum(prediccion != y))} de {len(y)}",
    )
    afirmar(cerca(X, copia_X, 0.0) and cerca(y, copia_y, 0.0),
            "entrenar_arbol no debe modificar X ni y")

    puro = sub.entrenar_arbol(np.array([[1.0], [2.0], [3.0]]), np.array([-1, -1, -1]))
    afirmar(
        puro["hoja"] and int(puro["clase"]) == -1,
        "un nodo puro se convierte en hoja de inmediato, sin buscar división",
    )


@prueba(3, "entrenar_arbol: las condiciones de parada", 3)
def p3_parada():
    X, y = datos_lunas(120, semilla=3)

    solo_hoja = sub.entrenar_arbol(X, y, profundidad_max=0)
    afirmar(
        solo_hoja["hoja"] and int(solo_hoja["n"]) == len(y),
        "con profundidad_max = 0 no queda ningún nivel por dividir: el árbol "
        "es una sola hoja con las n observaciones",
    )

    for limite in (1, 2, 3):
        arbol = sub.entrenar_arbol(X, y, profundidad_max=limite)
        obtenida = profundidad(arbol)
        afirmar(
            obtenida <= limite,
            f"con profundidad_max = {limite} ningún camino puede tener más de "
            f"{limite} división(es); el árbol llegó a {obtenida}",
        )
        afirmar(
            obtenida == limite,
            f"sobre estos datos hay ganancia de sobra: con profundidad_max = "
            f"{limite} el árbol debería llegar justo a {limite} y llegó a "
            f"{obtenida}",
        )

    grande = sub.entrenar_arbol(X, y, min_muestras_division=40)
    for nodo in hojas(grande):
        afirmar(
            int(nodo["n"]) >= 1,
            "una hoja no puede quedar vacía: revisa cómo repartes las filas",
        )
    afirmar(
        profundidad(grande) <= profundidad(sub.entrenar_arbol(X, y)),
        "subir min_muestras_division solo puede detener antes la recursión",
    )
    sin_dividir = sub.entrenar_arbol(X, y, min_muestras_division=len(y) + 1)
    afirmar(
        sin_dividir["hoja"],
        f"con min_muestras_division mayor que n = {len(y)} el nodo raíz ya es "
        "demasiado pequeño para dividirse: el árbol es una sola hoja",
    )


@prueba(3, "entrenar_arbol: el sorteo de m predictores por división", 3)
def p3_sorteo():
    X, y = datos_tabla(150, semilla=5)

    todos = sub.entrenar_arbol(X, y, profundidad_max=3, m=None,
                               rng=np.random.default_rng(0))
    otro = sub.entrenar_arbol(X, y, profundidad_max=3, m=None,
                              rng=np.random.default_rng(7))
    afirmar(
        cerca(predecir_arbol(todos, X), predecir_arbol(otro, X), 0.0),
        "con m = None no hay sorteo: el árbol no puede depender del rng",
    )

    # Cuando no hay nada que sortear, el generador tiene que quedar tal como
    # llegó. No basta con que el árbol salga igual: en entrenar_bosque el rng
    # es compartido, así que un sorteo de más acá corre todas las bolsas que
    # vienen después y cambia la corrida entera.
    siguiente = np.random.default_rng(31).integers(0, 10 ** 6)
    for cuantos in (None, X.shape[1], X.shape[1] + 3):
        gastado = np.random.default_rng(31)
        sub.entrenar_arbol(X, y, profundidad_max=3, m=cuantos, rng=gastado)
        afirmar(
            int(gastado.integers(0, 10 ** 6)) == int(siguiente),
            f"con m = {cuantos} sobre {X.shape[1]} predictores no queda nada "
            "que sortear, así que entrenar_arbol no debe consumir ningún "
            "número del rng: sortear igual y después descartar el resultado "
            "deja el generador movido",
        )

    uno = sub.entrenar_arbol(X, y, profundidad_max=3, m=1,
                             rng=np.random.default_rng(0))
    revisar_estructura(uno)
    distinto = sub.entrenar_arbol(X, y, profundidad_max=3, m=1,
                                  rng=np.random.default_rng(99))
    afirmar(
        not cerca(predecir_arbol(uno, X), predecir_arbol(distinto, X), 0.0),
        "con m = 1 cada división mira un solo predictor sorteado: dos rng "
        "distintos tienen que dar árboles distintos",
    )
    repetido = sub.entrenar_arbol(X, y, profundidad_max=3, m=1,
                                  rng=np.random.default_rng(0))
    afirmar(
        cerca(predecir_arbol(uno, X), predecir_arbol(repetido, X), 0.0),
        "con el mismo rng el árbol debe salir idéntico: el sorteo es lo único "
        "aleatorio y la semilla lo fija",
    )


# ----------------------------------------------------------------------
# Parte 4: bagging, bosque aleatorio y voto de mayoría
# ----------------------------------------------------------------------
@prueba(4, "predecir_ensamble: voto de mayoría, empate y ensamble vacío", 4)
def p4_voto():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    mayoria_positiva = [hoja(1), hoja(1), hoja(-1)]
    afirmar(
        cerca(sub.predecir_ensamble(mayoria_positiva, X), [1, 1, 1], 0.0),
        "dos votos por +1 contra uno por -1: gana +1",
    )
    mayoria_negativa = [hoja(-1), hoja(-1), hoja(-1), hoja(1), hoja(1)]
    afirmar(
        cerca(sub.predecir_ensamble(mayoria_negativa, X), [-1, -1, -1], 0.0),
        "tres votos por -1 contra dos por +1: gana -1",
    )
    empate = [hoja(1), hoja(-1)]
    afirmar(
        cerca(sub.predecir_ensamble(empate, X), [1, 1, 1], 0.0),
        "con la votación empatada la convención de la tarea es responder +1",
    )
    afirmar(
        cerca(sub.predecir_ensamble([], X), [1, 1, 1], 0.0),
        "sin árboles la suma de votos es 0 y por la misma convención sale +1",
    )
    salida = sub.predecir_ensamble(mayoria_positiva, X)
    afirmar(
        np.asarray(salida).shape == (3,),
        f"predecir_ensamble debe devolver un arreglo de forma (3,); dio "
        f"{np.asarray(salida).shape}",
    )


@prueba(4, "entrenar_bosque: contrato del diccionario y muestreo bootstrap", 3)
def p4_contrato():
    X, y = datos_lunas(80, semilla=2)
    copia_X, copia_y = X.copy(), y.copy()
    bosque = sub.entrenar_bosque(X, y, B=6, semilla=0, profundidad_max=3)

    afirmar(isinstance(bosque, dict),
            f"entrenar_bosque debe devolver un diccionario; devolvió "
            f"{type(bosque).__name__}")
    afirmar(
        set(bosque) == {"arboles", "bolsas", "B", "m"},
        f"las claves deben ser exactamente arboles, bolsas, B y m; trajo "
        f"{sorted(bosque)}",
    )
    afirmar(len(bosque["arboles"]) == 6 and len(bosque["bolsas"]) == 6,
            f"con B = 6 deben venir 6 árboles y 6 bolsas; vinieron "
            f"{len(bosque['arboles'])} y {len(bosque['bolsas'])}")

    n = len(y)
    hubo_repetido = False
    hubo_fuera = False
    for bolsa in bosque["bolsas"]:
        indices = np.asarray(bolsa)
        afirmar(
            indices.shape == (n,),
            f"cada bolsa tiene tantos índices como observaciones, o sea "
            f"({n},); dio {indices.shape}",
        )
        afirmar(
            int(indices.min()) >= 0 and int(indices.max()) < n,
            "los índices de la bolsa tienen que caer en [0, n)",
        )
        distintos = len(set(int(i) for i in indices))
        hubo_repetido = hubo_repetido or distintos < n
        hubo_fuera = hubo_fuera or distintos < n
    afirmar(
        hubo_repetido and hubo_fuera,
        "el muestreo bootstrap es CON reemplazo: alguna observación tiene que "
        "salir repetida y otras quedar fuera de la bolsa. Si todas las bolsas "
        "traen las n observaciones distintas, estás permutando en vez de "
        "sortear con reemplazo",
    )

    # El sorteo está fijado hasta el detalle: la primera bolsa son los primeros
    # n índices que entrega el generador de esa semilla. Así dos corridas de
    # dos personas distintas producen exactamente el mismo bosque.
    primera = np.random.default_rng(0).integers(0, n, size=n)
    afirmar(
        cerca(np.asarray(bosque["bolsas"][0]), primera, 0.0),
        "la primera bolsa debe ser rng.integers(0, n, size=n) del generador "
        "recién creado con la semilla: otro sorteo con reemplazo (por ejemplo "
        "rng.choice) da una bolsa distinta y con eso cambian todos los "
        "números que anotes en la sección de respuestas",
    )

    afirmar(cerca(X, copia_X, 0.0) and cerca(y, copia_y, 0.0),
            "entrenar_bosque no debe modificar X ni y")


@prueba(4, "entrenar_bosque: la semilla fija toda la corrida", 3)
def p4_semilla():
    X, y = datos_lunas(80, semilla=2)
    uno = sub.entrenar_bosque(X, y, B=5, m=1, semilla=4, profundidad_max=3)
    otro = sub.entrenar_bosque(X, y, B=5, m=1, semilla=4, profundidad_max=3)
    afirmar(
        cerca(sub.predecir_ensamble(uno["arboles"], X),
              sub.predecir_ensamble(otro["arboles"], X), 0.0),
        "dos bosques con la misma semilla deben predecir exactamente igual",
    )
    afirmar(
        cerca(np.asarray(uno["bolsas"]), np.asarray(otro["bolsas"]), 0.0),
        "dos bosques con la misma semilla deben sortear las mismas bolsas",
    )
    tercero = sub.entrenar_bosque(X, y, B=5, m=1, semilla=123, profundidad_max=3)
    afirmar(
        not cerca(np.asarray(uno["bolsas"]), np.asarray(tercero["bolsas"]), 0.0),
        "con otra semilla las bolsas tienen que cambiar: si no, el bootstrap "
        "no está usando el generador",
    )
    afirmar(uno["m"] == 1, "la clave «m» debe guardar el m con que se entrenó")


# ----------------------------------------------------------------------
# Parte 5: el error fuera de bolsa
# ----------------------------------------------------------------------
@prueba(5, "error_oob: un caso con bolsas escritas a mano", 4)
def p5_a_mano():
    y = np.array([1, 1, 1, -1, -1, -1])
    X = np.arange(6.0).reshape(6, 1)
    bosque = {
        "B": 3,
        "m": None,
        "arboles": [hoja(1), hoja(-1), hoja(1)],
        "bolsas": [
            np.array([0, 1, 2, 0, 1, 2]),   # fuera: 3, 4, 5
            np.array([3, 4, 5, 3, 4, 5]),   # fuera: 0, 1, 2
            np.array([0, 1, 3, 4, 0, 1]),   # fuera: 2, 5
        ],
    }
    obtenido = sub.error_oob(bosque, X, y)
    afirmar(
        cerca(obtenido, 5.0 / 6.0, 1e-9),
        "cada observación se vota solo con los árboles que no la vieron: las "
        "0, 1 y 2 reciben -1 del segundo árbol (la 2 además +1 del tercero, y "
        "el empate sale +1) y las 3, 4 y 5 reciben +1. Quedan 5 errores de 6, "
        f"o sea 0,8333; dio {obtenido}",
    )
    afirmar(isinstance(obtenido, float),
            "error_oob debe devolver un float de Python")


@prueba(5, "error_oob: sin observaciones fuera de bolsa", 3)
def p5_sin_oob():
    y = np.array([1, -1, 1, -1])
    X = np.arange(4.0).reshape(4, 1)
    completo = {
        "B": 2, "m": None,
        "arboles": [hoja(1), hoja(-1)],
        "bolsas": [np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0])],
    }
    obtenido = sub.error_oob(completo, X, y)
    afirmar(
        isinstance(obtenido, float) and np.isnan(obtenido),
        "si toda observación quedó dentro de la bolsa de todos los árboles no "
        f"hay nada que evaluar y corresponde float('nan'); dio {obtenido}",
    )

    mixto = {
        "B": 2, "m": None,
        "arboles": [hoja(1), hoja(1)],
        "bolsas": [np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])],
    }
    obtenido = sub.error_oob(mixto, X, y)
    afirmar(
        cerca(obtenido, 0.5, 1e-9),
        "el primer árbol no deja a nadie fuera y no vota por nadie; el segundo "
        "deja fuera a las observaciones 2 y 3, y responde +1 a las dos, así "
        f"que acierta una de dos: 0,5. Dio {obtenido}",
    )


@prueba(5, "error_oob: estima el error del conjunto reservado", 3)
def p5_estimacion():
    X, y = datos_lunas(200, ruido=0.25, semilla=0)
    X_ent, y_ent, X_res, y_res = dividir(X, y, 0.3, semilla=0)
    bosque = sub.entrenar_bosque(X_ent, y_ent, B=20, semilla=0)

    oob = sub.error_oob(bosque, X_ent, y_ent)
    afirmar(
        0.0 <= oob <= 1.0,
        f"el error fuera de bolsa es una fracción entre 0 y 1; dio {oob}",
    )
    reservado = 1.0 - exactitud(y_res, sub.predecir_ensamble(bosque["arboles"], X_res))
    afirmar(
        abs(oob - reservado) <= 0.10,
        f"el error fuera de bolsa debería quedar cerca del error del conjunto "
        f"reservado: OOB {oob:.3f} contra {reservado:.3f}",
    )
    entrenamiento = 1.0 - exactitud(
        y_ent, sub.predecir_ensamble(bosque["arboles"], X_ent))
    afirmar(
        oob > entrenamiento + 1e-9,
        f"el error fuera de bolsa tiene que ser mayor que el error sobre los "
        f"datos de entrenamiento: si te dan casi iguales ({oob:.3f} contra "
        f"{entrenamiento:.3f}), probablemente estás votando con todos los "
        f"árboles y no solo con los que dejaron fuera cada observación",
    )


# ----------------------------------------------------------------------
# Parte 6: AdaBoost con tocones de decisión
# ----------------------------------------------------------------------
@prueba(6, "entrenar_adaboost: la primera ronda del ejemplo del enunciado", 4)
def p6_primera_ronda():
    X, y = datos_ejemplo()
    copia_X, copia_y = X.copy(), y.copy()
    modelo = sub.entrenar_adaboost(X, y, T=1)

    afirmar(isinstance(modelo, dict),
            f"entrenar_adaboost debe devolver un diccionario; devolvió "
            f"{type(modelo).__name__}")
    afirmar(
        set(modelo) == {"tocones", "alfas", "errores"},
        f"las claves deben ser exactamente tocones, alfas y errores; trajo "
        f"{sorted(modelo)}",
    )
    afirmar(
        len(modelo["tocones"]) == 1 and len(modelo["alfas"]) == 1
        and len(modelo["errores"]) == 1,
        "con T = 1 las tres listas tienen que traer un solo elemento",
    )
    afirmar(
        cerca(modelo["errores"][0], 0.2, 1e-9),
        "con pesos uniformes de 1/10, el mejor tocón falla en 2 de las 10 "
        f"observaciones: el error ponderado es 0,2 y dio {modelo['errores'][0]}",
    )
    afirmar(
        cerca(modelo["alfas"][0], 0.5 * np.log(4.0), 1e-9),
        f"con eps = 0,2 el alfa es (1/2)·ln(0,8/0,2) = (1/2)·ln 4 = 0,6931…; "
        f"dio {modelo['alfas'][0]}",
    )
    afirmar(cerca(X, copia_X, 0.0) and cerca(y, copia_y, 0.0),
            "entrenar_adaboost no debe modificar X ni y")


@prueba(6, "predecir_adaboost: voto ponderado por los alfas", 3)
def p6_voto_ponderado():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    fuerte = {"predictor": 0, "umbral": 1.5, "signo": 1}    # +1, +1, -1, -1
    debil = {"predictor": 0, "umbral": 1.5, "signo": -1}    # -1, -1, +1, +1

    modelo = {"tocones": [fuerte, debil, debil],
              "alfas": [2.0, 0.4, 0.4], "errores": [0.1, 0.4, 0.4]}
    afirmar(
        cerca(sub.predecir_adaboost(modelo, X), [1, 1, -1, -1], 0.0),
        "el tocón con alfa 2,0 debe imponerse a dos tocones con alfa 0,4: el "
        "voto es ponderado, no por cabeza",
    )

    volteado = {"tocones": [fuerte, debil], "alfas": [0.4, 2.0],
                "errores": [0.4, 0.1]}
    afirmar(
        cerca(sub.predecir_adaboost(volteado, X), [-1, -1, 1, 1], 0.0),
        "al mover el alfa grande al otro tocón, la predicción debe darse "
        "vuelta entera",
    )

    empatado = {"tocones": [fuerte, debil], "alfas": [1.0, 1.0],
                "errores": [0.3, 0.3]}
    afirmar(
        cerca(sub.predecir_adaboost(empatado, X), [1, 1, 1, 1], 0.0),
        "con el puntaje en cero rige la misma convención del voto de mayoría: "
        "sale +1",
    )
    vacio = {"tocones": [], "alfas": [], "errores": []}
    afirmar(
        cerca(sub.predecir_adaboost(vacio, X), [1, 1, 1, 1], 0.0),
        "un ensamble sin tocones tiene puntaje 0 en todas partes: responde +1",
    )


@prueba(6, "entrenar_adaboost: los pesos cambian de una ronda a la otra", 3)
def p6_reponderacion():
    X, y = datos_lunas(120, semilla=1)
    modelo = sub.entrenar_adaboost(X, y, T=6)
    errores = modelo["errores"]
    alfas = modelo["alfas"]

    afirmar(len(errores) == 6 and len(alfas) == 6,
            f"con T = 6 y tocones útiles en cada ronda deben salir 6 rondas; "
            f"salieron {len(errores)}")
    for i, eps in enumerate(errores):
        afirmar(
            0.0 <= eps < 0.5,
            f"el error ponderado de la ronda {i} debe caer en [0, 0,5): un "
            f"tocón peor que eso detiene el bucle. Dio {eps}",
        )
    for i, (eps, alfa) in enumerate(zip(errores, alfas)):
        afirmar(
            cerca(alfa, 0.5 * np.log((1.0 - eps) / eps), 1e-7),
            f"en la ronda {i} el alfa no calza con (1/2)·ln((1-eps)/eps): "
            f"eps = {eps} pide alfa = {0.5 * np.log((1 - eps) / eps)} y dio "
            f"{alfa}",
        )
        afirmar(alfa > 0.0, f"con eps < 0,5 el alfa es positivo; dio {alfa}")

    afirmar(
        errores[1] > errores[0] + 1e-9,
        "tras la primera ronda los ejemplos mal clasificados engordan, así que "
        "el segundo tocón lo tiene más difícil y su error ponderado sube. Si "
        f"te da {errores[1]} contra {errores[0]}, revisa si estás "
        "actualizando y renormalizando los pesos",
    )

    vacio = sub.entrenar_adaboost(X, y, T=0)
    afirmar(
        vacio["tocones"] == [] and vacio["alfas"] == [] and vacio["errores"] == [],
        "con T = 0 no se ajusta ningún tocón: las tres listas quedan vacías",
    )


@prueba(6, "AdaBoost mejora al crecer el número de rondas", 3)
def p6_mejora():
    X, y = datos_lunas(200, ruido=0.25, semilla=0)
    X_ent, y_ent, X_res, y_res = dividir(X, y, 0.3, semilla=0)

    una = sub.entrenar_adaboost(X_ent, y_ent, T=1)
    muchas = sub.entrenar_adaboost(X_ent, y_ent, T=40)

    ent_una = exactitud(y_ent, sub.predecir_adaboost(una, X_ent))
    ent_muchas = exactitud(y_ent, sub.predecir_adaboost(muchas, X_ent))
    afirmar(
        ent_muchas > ent_una,
        f"cuarenta rondas tienen que clasificar mejor el entrenamiento que una "
        f"sola: {ent_muchas:.3f} contra {ent_una:.3f}",
    )
    afirmar(
        ent_muchas >= 0.92,
        f"con 40 tocones el ensamble debería acertar al menos el 92% del "
        f"entrenamiento y acertó {ent_muchas:.3f}",
    )
    reservado = exactitud(y_res, sub.predecir_adaboost(muchas, X_res))
    afirmar(
        reservado >= 0.85,
        f"y al menos el 85% del conjunto reservado; acertó {reservado:.3f}",
    )
    afirmar(
        cerca(predecir_tocon(muchas["tocones"][0], X_ent),
              predecir_tocon(una["tocones"][0], X_ent), 0.0),
        "la primera ronda parte siempre de pesos uniformes, así que el primer "
        "tocón no puede depender de cuántas rondas se pidieron",
    )


# ----------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------
TOPE_SEGUNDOS = 120

# El temporizador se apoya en SIGALRM, que existe en Linux y macOS pero no en
# Windows. Si no está disponible, el corrector funciona igual, solo que sin
# tope de tiempo.
HAY_TEMPORIZADOR = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class TiempoAgotado(Exception):
    """Se levanta cuando una prueba supera el tope de tiempo."""


def _sono_la_alarma(_numero, _marco):
    raise TiempoAgotado


def ejecutar_una(caso):
    """Corre una prueba y devuelve (puntos obtenidos, detalle del error).

    Cada prueba tiene un tope de ``TOPE_SEGUNDOS``: una recursión que nunca
    llega a una hoja dejaría al corrector colgado para siempre, y en una
    corrección por lotes una sola entrega así bloquearía todo lo demás.
    """
    anterior = None
    if HAY_TEMPORIZADOR:
        anterior = signal.signal(signal.SIGALRM, _sono_la_alarma)
        signal.setitimer(signal.ITIMER_REAL, TOPE_SEGUNDOS)
    try:
        caso["funcion"]()
    except TiempoAgotado:
        return 0, (f"la prueba superó los {TOPE_SEGUNDOS} s: ¿alguna condición "
                   "de parada del árbol quedó sin escribir?")
    except NotImplementedError:
        return 0, "sin implementar todavía"
    except RecursionError:
        return 0, ("recursión sin fondo: revisa las condiciones de parada de "
                   "entrenar_arbol, en particular el nodo puro")
    except AssertionError as exc:
        return 0, str(exc)
    except Exception as exc:  # noqa: BLE001 — el grader nunca debe caerse
        return 0, f"{type(exc).__name__}: {exc}"
    finally:
        if HAY_TEMPORIZADOR:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, anterior)
    return caso["puntos"], ""


def main():
    print("=" * 70)
    print(" Autograder — Tarea: Modelos de Ensamblaje")
    print("=" * 70)

    obtenido_total = 0
    maximo_total = 0
    hay_extra = any(c["extra"] for c in PRUEBAS)

    for numero, titulo, _ in PARTES:
        casos = [c for c in PRUEBAS if c["parte"] == numero]
        if not casos:
            continue
        print(f"\nParte {numero}: {titulo}")
        obtenido_parte = maximo_parte = 0
        for caso in casos:
            puntos, detalle = ejecutar_una(caso)
            obtenido_parte += puntos
            maximo_parte += caso["puntos"]
            marca = "OK   " if puntos == caso["puntos"] else "FALLA"
            sufijo = "  (adicional)" if caso["extra"] else ""
            print(f"  [{marca}] {caso['descripcion']}{sufijo}"
                  f"  ->  {puntos}/{caso['puntos']}")
            if detalle:
                print(f"           {detalle}")
        print(f"  Subtotal parte {numero}: {obtenido_parte}/{maximo_parte}")
        obtenido_total += obtenido_parte
        maximo_total += maximo_parte

    print("\n" + "=" * 70)
    if hay_extra:
        print(f" PUNTAJE TOTAL: {obtenido_total}/{maximo_total}")
    else:
        print(f" PUNTAJE EN LAS PRUEBAS VISIBLES: {obtenido_total}/{maximo_total}")
        print(f" La tarea vale {PUNTAJE_TOTAL} puntos: los "
              f"{PUNTAJE_TOTAL - maximo_total} restantes los pone el corrector")
        print(" del curso, que revisa la sección de respuestas del comienzo de")
        print(" submission.py y prueba esas mismas ocho funciones en casos")
        print(" límite y de escala. Los 100 puntos se corrigen solos.")
    print("=" * 70)
    return 0 if obtenido_total == maximo_total else 1


if __name__ == "__main__":
    sys.exit(main())
