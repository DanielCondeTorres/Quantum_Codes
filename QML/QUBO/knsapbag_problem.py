"""
Este script resuelve el problema de la mochila (knapsack) usando un modelo QUBO (Quadratic Unconstrained Binary Optimization)
y el paquete qubovert. El objetivo es seleccionar objetos para maximizar el valor total sin exceder el peso máximo permitido.

- value: lista de valores de los objetos
- weight: lista de pesos de los objetos
- L: peso máximo permitido
"""

from qubovert import QUBO

# Define inputs
value = [1, 4, 2, 2, 5, 2]
weight = [1, 3, 3, 3, 2, 2]
L = 7
n = len(value)

# Penalty coefficient (tune if needed)
lagrange = 10

# Initialize QUBO
Q = QUBO()

# Objective part: maximize total value => minimize -value
for i in range(n):
    Q[(i, i)] += -value[i]

# Constraint part: (sum w_i x_i - L)^2
# Expand square: sum_i sum_j w_i w_j x_i x_j - 2L sum_i w_i x_i + L^2
for i in range(n):
    for j in range(n):
        Q[(i, j)] += lagrange * weight[i] * weight[j]
    Q[(i, i)] += -2 * lagrange * L * weight[i]

# L^2 term is constant, can be omitted in optimization
# Solve QUBO
solution = Q.solve_bruteforce()  # exact solution since small problem

# Results
selected_items = [i for i in range(n) if solution[i] == 1]
total_value = sum(value[i] for i in selected_items)
total_weight = sum(weight[i] for i in selected_items)

print("Selected item indices:", selected_items)
print("Total value:", total_value)
print("Total weight:", total_weight)


# Con annealer: 
