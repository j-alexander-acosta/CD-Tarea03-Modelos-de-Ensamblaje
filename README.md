# Tarea: Modelos de Ensamblaje

Ciencia de Datos — Universidad Católica del Norte

## Archivos

| Archivo | Qué es |
|---|---|
| `enunciado.pdf` | El enunciado completo. Léelo primero. |
| `submission.py` | **El único archivo que debes editar y entregar.** Trae la sección de respuestas y las ocho funciones. |
| `util.py` | Conjuntos de datos y piezas de apoyo ya escritas. No se modifica. |
| `experimento.py` | Cierre de la tarea: corre los cuatro experimentos y deja en la terminal las tablas de las que sale la sección de respuestas. |
| `grader.py` | Corrector automático. Corre sin conexión a internet. |

## Antes de empezar

`COMO-TRABAJAR.pdf` trae las instrucciones generales: qué instalar, cómo abrir
una terminal en esta carpeta, qué significa el puntaje del corrector y los
errores más frecuentes. Sirve para todas las tareas del curso.

Un detalle que conviene saber de entrada: en Windows el intérprete se llama
`python` (o `py`), y en Linux y macOS suele ser `python3`. Los comandos de acá
usan `python`: si tu sistema responde «command not found», prueba con
`python3`.

## Cómo trabajar

Basta con Python 3 y numpy:

```
pip install -r requirements.txt
```

Cada vez que completes una parte, corre el corrector desde esta carpeta:

```
python grader.py
```

El corrector imprime, prueba por prueba, qué pasó y por qué. Está pensado para
leerse: cuando algo falla, el mensaje dice qué valor se esperaba y cuál
llegó. Las seis partes son independientes en el puntaje, pero no en el código:
la Parte 3 llama a la Parte 2, que llama a la Parte 1, y las Partes 4 y 5 se
apoyan en la 3. Conviene completarlas en orden.

Las pruebas visibles suman 60 de los 100 puntos. Los 40 restantes los pone el
corrector del curso, que revisa la sección de respuestas y prueba esas mismas
ocho funciones en casos límite y de escala. Los 100 puntos se corrigen solos,
sin revisión a mano.

Al terminar, ejecuta el cierre:

```
python experimento.py
```

Demora alrededor de medio minuto y deja en la terminal las cuatro tablas de
las que sale la sección de respuestas: la profundidad de un árbol solo, la
caída de la dispersión del error al aumentar `B`, la comparación entre bagging
y bosque aleatorio con su error fuera de bolsa, y las rondas de AdaBoost.
Anota esos números en las ocho constantes que abren `submission.py`.

## Entrega

Un solo archivo: `submission.py`, con las ocho funciones completadas y la
sección de respuestas llena. No se entrega nada más. No hay que subir datos:
todo se genera localmente con semillas fijas, así que el corrector reproduce
exactamente los mismos conjuntos en cualquier computador y sin conexión a
internet. No incluyas la carpeta `__pycache__`.
