"""
Descripción:
Este script resuelve un problema de optimización relacionado con la organización de un evento empresarial.
Queremos invitar a la mayor cantidad de empresas posibles, pero existen conflictos entre algunas de ellas,
por lo que no todas pueden estar juntas en el evento.

Empresas involucradas: e1, e2, e3, e4, e5, e6

Restricciones:
- Si se invita a e3, no se puede invitar a e4
- Si se invita a e1, no se puede invitar ni a e2 ni a e3
- Si se invita a e2, no se puede invitar ni a e1 ni a e6

Objetivo:
Maximizar el número de empresas invitadas sin violar las restricciones.

Método de resolución:
Usamos un optimizador clásico (COBYLA) que puede usarse como aproximación clásica al enfoque cuántico (QAOA o VQE).
Este código es útil para testear antes de pasar a simulación cuántica o implementación real.
"""

# --------------------------------------------
# Importamos qubovert, que nos permite definir
# el problema como un QUBO (modelo binario).
# --------------------------------------------
import qubovert
import matplotlib.pyplot as plt

# Creamos un objeto QUBO vacío
qubo = qubovert.QUBO()

# Creamos las variables binarias e1, e2, ..., e6
# Cada variable representa si la empresa ei es invitada (1) o no (0)
for i in range(1, 7):
    qubo.create_var(f"e{i}")

# --------------------------------------------------
# Parte 1: OBJETIVO -> Maximizar número de empresas
# Como qubovert minimiza por defecto, sumamos +1 por empresa
# (equivalente a minimizar -cantidad de empresas invitadas)
# --------------------------------------------------
for i in range(1, 7):
    qubo[(f"e{i}",)] = 1

# --------------------------------------------------
# Parte 2: RESTRICCIONES (conflictos entre empresas)
# Penalizamos combinaciones conflictivas
# --------------------------------------------------

# Si invitas a e3 y e4 juntos, es un conflicto → penalizamos
qubo[(f"e{3}", f"e{4}")] = -1

# Si invitas a e1, no puedes invitar a e2 ni e3 → penalizaciones
qubo[(f"e{1}", f"e{2}")] = -2
qubo[(f"e{1}", f"e{3}")] = -1

# Si invitas a e2, no puedes invitar a e6 → penalización
qubo[(f"e{2}", f"e{6}")] = -1

print('Opcion 1: Quantum annealer')
anneal_res = qubovert.sim.anneal_pubo(-qubo, num_anneals=1000) # Signo menos porque no lo estoy penalizando
print('Solucion annealing: ',anneal_res.best.state)
# Convertimos el QUBO a formato Ising (QUso = QUBO to Spin Operator)
# Esto es necesario para usarlo como un Hamiltoniano cuántico
print('Opción 2: Ising model')
ising = qubo.to_quso()
print("ising", ising)


import pennylane as qml

# --------------------------------------------------
# Función para convertir el diccionario Ising en un
# objeto Hamiltonian compatible con PennyLane
# --------------------------------------------------
def create_hamiltonian(ising):
    coeffs = []
    ops = []

    for term in ising:
        if len(term) == 0:
            ops.append(qml.Identity(0))  # término constante
        elif len(term) == 1:
            ops.append(qml.PauliZ(term[0]))  # término lineal
        elif len(term) == 2:
            ops.append(qml.PauliZ(term[0]) @ qml.PauliZ(term[1]))  # término de interacción
        elif len(term) == 3:
            ops.append(qml.PauliZ(term[0]) @ qml.PauliZ(term[1])@ qml.PauliZ(term[2]))  # término de interacción
        else:
            ops.append(qml.PauliZ(term[0]) @ qml.PauliZ(term[1])@ qml.PauliZ(term[2])@ qml.PauliZ(term[3])) 
        coeffs.append(ising[term])
    return qml.Hamiltonian(coeffs, ops)

# Creamos el Hamiltoniano H a partir del QUBO convertido
H = create_hamiltonian(ising)

import numpy as np

# Dispositivo cuántico simulado con 6 qubits
dev = qml.device("default.qubit", wires=6)

# -------------------------------------------
# Definimos el circuito variacional (ansatz)
# Aplicamos rotaciones RY y conexiones CNOT
# Devuelve el valor esperado del Hamiltoniano
# -------------------------------------------
@qml.qnode(dev)
def circuit(params):
    for i in range(6):
        qml.RY(params[i], wires=i)
    for i in range(5):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(H)

# Generamos parámetros iniciales aleatorios
params = np.random.rand(6)
# 🎯 Dibuja el circuito como un gráfico desplegable
fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(params)
plt.show()
# Ejecutamos el circuito una vez para verificar
circuit(params)

from scipy.optimize import minimize

# Definimos la función de coste (negativo del valor esperado)
def cost(params):
    return -circuit(params)

# Ejecutamos el optimizador COBYLA
sol = minimize(cost, params, method="COBYLA")

# Obtenemos los parámetros óptimos
optimal_params = sol.x

# Creamos un nuevo dispositivo que toma solo 1 muestra (shot único)
dev = qml.device("default.qubit", wires=6, shots=1)

# Definimos un circuito para generar una muestra (bitstring) según los parámetros óptimos
@qml.qnode(dev)
def circuit_sample(params):
    for i in range(6):
        qml.RY(params[i], wires=i)
    for i in range(5):
        qml.CNOT(wires=[i, i+1])
    return qml.sample(wires=range(6))

# Mostramos el valor esperado del Hamiltoniano optimizado
print("assistant:", np.round(circuit(optimal_params)))

# Mostramos qué empresas fueron invitadas (bitstring de 6 bits)
print("invitations:", circuit_sample(optimal_params))
