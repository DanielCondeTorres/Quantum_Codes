print('The famous $N$ queens problem consists of placing $N$ queens on a $N \times N$ chessboard in such a way that none of them attacks any other. Model in QUBO the problem using `qubovert` and solve it for the case N=8.')






"""
Este script resuelve el problema de las 8 reinas usando un modelo QUBO (Quadratic Unconstrained Binary Optimization)
y el paquete qubovert. El objetivo es colocar 8 reinas en un tablero de ajedrez de 8x8 de forma que no se ataquen entre sí.

- Cada variable binaria x_{i}_{j} indica si hay una reina en la posición (i, j).
- La función objetivo penaliza posiciones no permitidas (reinas que se atacan).
- Se utiliza recocido simulado para encontrar una solución óptima.
"""
import qubovert

size = 8  # Tamaño del tablero (8x8)
lagrange = 1  # Penalización para las restricciones

# Creamos las variables del modelo QUBO, una por cada casilla del tablero
Q = qubovert.QUBO()
for i in range(size):
    for j in range(size):
        Q.create_var(f"x_{i}_{j}")  # Variable binaria: 1 si hay reina en (i, j), 0 si no

# Añadimos el primer bloque de la función objetivo: maximizar el número de reinas
for i in range(size):
    for j in range(size):
        Q[(f"x_{i}_{j}",)] = -1  # Queremos tantas reinas como sea posible por eso -1, con un +1 añadiria menos reinas

# Incluimos las restricciones para que no se ataquen entre sí
for i1 in range(size):
    for i2 in range(size):
        for i3 in range(size):
            for i4 in range(size):
                # Si están en la misma fila, columna o diagonal, penalizamos
                if i1 == i3 or i2 == i4 or i1 - i3 == i2 - i4 or i1 - i3 == i4 - i2:
                    if not (i1 == i3 and i2 == i4):  # Excepto la misma casilla, esta no la entiendo demasiado bien.
                        Q[(f"x_{i1}_{i2}", f"x_{i3}_{i4}")] = lagrange

# Ejecutamos el recocido simulado para resolver el QUBO
anneal_res = qubovert.sim.anneal_pubo(Q, num_anneals=100)

# Mostramos la solución encontrada en formato de tablero
for i in range(size):
    for j in range(size):
        if anneal_res.best.state[f"x_{i}_{j}"] == 0:
            print("O", end=" ")  # Casilla vacía
        else:
            print("X", end=" ")  # Reina
    print()  # Nueva línea para la siguiente fila
