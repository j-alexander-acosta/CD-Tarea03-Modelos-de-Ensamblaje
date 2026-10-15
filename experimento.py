#!/usr/bin/env python3
"""Cierre de la tarea «Modelos de Ensamblaje». No hay que modificar este archivo.

Corre los cuatro experimentos que cierran la tarea y deja las tablas en la
terminal. Necesita las ocho funciones de ``submission.py`` completas::

    python experimento.py

Todo se genera localmente con semillas fijas: no descarga nada y dos
corridas seguidas dan exactamente lo mismo.
"""
from __future__ import annotations

import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True

import numpy as np

from util import (
    datos_lunas,
    datos_tabla,
    dividir,
    exactitud,
    predecir_arbol,
)

from submission import (
    entrenar_adaboost,
    entrenar_arbol,
    entrenar_bosque,
    error_oob,
    predecir_adaboost,
    predecir_ensamble,
)


def titulo(texto):
    print()
    print("=" * 70)
    print(f" {texto}")
    print("=" * 70)


# ----------------------------------------------------------------------
def experimento_1_un_arbol():
    """Un árbol solo: qué le hace la profundidad al error."""
    titulo("1. Un árbol solo: profundidad contra error")
    X, y = datos_lunas(300, ruido=0.35, semilla=0)
    X_ent, y_ent, X_res, y_res = dividir(X, y, 0.3, semilla=0)

    print(f"{'profundidad':>12} {'hojas':>7} {'err. entren.':>13} "
          f"{'err. reservado':>15}")
    for limite in (1, 2, 3, 5, 8, None):
        arbol = entrenar_arbol(X_ent, y_ent, profundidad_max=limite)
        err_ent = 1.0 - exactitud(y_ent, predecir_arbol(arbol, X_ent))
        err_res = 1.0 - exactitud(y_res, predecir_arbol(arbol, X_res))
        etiqueta = "sin tope" if limite is None else str(limite)
        print(f"{etiqueta:>12} {contar_hojas(arbol):>7} {err_ent:>13.3f} "
              f"{err_res:>15.3f}")
    print("\nAl crecer la profundidad el error de entrenamiento baja hasta cero")
    print("y el del conjunto reservado deja de mejorar: es la varianza del")
    print("árbol profundo, la materia prima del ensamble.")


def contar_hojas(arbol):
    if arbol["hoja"]:
        return 1
    return contar_hojas(arbol["izquierda"]) + contar_hojas(arbol["derecha"])


# ----------------------------------------------------------------------
def experimento_2_bagging():
    """Bagging: cómo cae la variabilidad del error al aumentar B."""
    titulo("2. Bagging: la variabilidad del error al aumentar B")
    X, y = datos_lunas(300, ruido=0.35, semilla=0)
    X_ent, y_ent, X_res, y_res = dividir(X, y, 0.3, semilla=0)

    print(f"{'B':>4} {'err. reservado medio':>22} {'desv. estándar':>16} "
          f"{'peor corrida':>14}")
    for B in (1, 3, 9, 27):
        errores = []
        for semilla in range(6):
            bosque = entrenar_bosque(X_ent, y_ent, B=B, semilla=semilla)
            errores.append(
                1.0 - exactitud(y_res, predecir_ensamble(bosque["arboles"], X_res))
            )
        errores = np.array(errores)
        print(f"{B:>4} {errores.mean():>22.3f} {errores.std():>16.4f} "
              f"{errores.max():>14.3f}")
    print("\nCada fila repite el mismo experimento con seis semillas distintas.")
    print("Lo que se estabiliza al crecer B no es solo el error medio: es la")
    print("dispersión entre corridas. Eso es bajar la varianza.")


# ----------------------------------------------------------------------
def experimento_3_bosque():
    """Bagging contra bosque aleatorio sobre los mismos datos."""
    titulo("3. Bagging contra bosque aleatorio, y el error fuera de bolsa")
    X, y = datos_tabla(300, semilla=1)
    X_ent, y_ent, X_res, y_res = dividir(X, y, 0.3, semilla=0)

    arbol = entrenar_arbol(X_ent, y_ent, min_muestras_division=5)
    err_arbol = 1.0 - exactitud(y_res, predecir_arbol(arbol, X_res))
    print(f"Un árbol solo: error reservado {err_arbol:.3f}\n")

    print(f"{'m':>10} {'error OOB':>11} {'err. reservado':>16} "
          f"{'diferencia':>12}")
    for m in (None, 4, 3, 2, 1):
        bosque = entrenar_bosque(X_ent, y_ent, B=25, m=m, semilla=0,
                                 min_muestras_division=5)
        oob = error_oob(bosque, X_ent, y_ent)
        res = 1.0 - exactitud(y_res, predecir_ensamble(bosque["arboles"], X_res))
        etiqueta = "todos (8)" if m is None else str(m)
        print(f"{etiqueta:>10} {oob:>11.3f} {res:>16.3f} {oob - res:>12.3f}")
    print("\nLa primera fila es bagging: cada división ve los ocho predictores.")
    print("Las siguientes son bosques aleatorios. La comparación es honesta")
    print("porque los cinco ensambles usan los mismos datos, el mismo B y la")
    print("misma semilla: lo único que cambia es m.")


# ----------------------------------------------------------------------
def experimento_4_adaboost():
    """AdaBoost: qué pasa al agregar rondas."""
    titulo("4. AdaBoost: rondas, pesos y voto final")
    X, y = datos_lunas(300, ruido=0.35, semilla=0)
    X_ent, y_ent, X_res, y_res = dividir(X, y, 0.3, semilla=0)

    print(f"{'T':>4} {'err. entren.':>13} {'err. reservado':>15}")
    for T in (1, 3, 10, 30, 80):
        modelo = entrenar_adaboost(X_ent, y_ent, T=T)
        err_ent = 1.0 - exactitud(y_ent, predecir_adaboost(modelo, X_ent))
        err_res = 1.0 - exactitud(y_res, predecir_adaboost(modelo, X_res))
        print(f"{T:>4} {err_ent:>13.3f} {err_res:>15.3f}")

    modelo = entrenar_adaboost(X_ent, y_ent, T=8)
    print("\nLas primeras ocho rondas, una por una:")
    print(f"{'ronda':>6} {'predictor':>10} {'umbral':>9} {'signo':>6} "
          f"{'eps':>8} {'alfa':>8}")
    for i, (tocon, eps, alfa) in enumerate(
        zip(modelo["tocones"], modelo["errores"], modelo["alfas"]), start=1
    ):
        print(f"{i:>6} {tocon['predictor']:>10} {tocon['umbral']:>9.3f} "
              f"{tocon['signo']:>6} {eps:>8.3f} {alfa:>8.3f}")
    print("\nEl error ponderado sube tras la primera ronda y se acerca a 0,5:")
    print("los pesos van concentrándose en lo difícil, así que cada tocón")
    print("nuevo enfrenta un problema más duro y entra con menos alfa.")


# ----------------------------------------------------------------------
def main():
    np.set_printoptions(precision=3, suppress=True)
    experimento_1_un_arbol()
    experimento_2_bagging()
    experimento_3_bosque()
    experimento_4_adaboost()
    print()
    print("=" * 70)
    print(" Estas cuatro tablas son las que se anotan en la sección de")
    print(" respuestas, al comienzo de submission.py. Cada constante lleva")
    print(" encima el experimento y la columna de la que sale.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
