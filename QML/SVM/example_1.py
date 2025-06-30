# Ejercicio:
# El objetivo de este ejercicio es calcular la función kernel K(xi, xj)
# donde las entradas x serán vectores de dimensión 2 y el mapa de características será el ZZ feature map.
# Además, dibujaremos el circuito usando PennyLane y matplotlib.

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def feature_map(x, wires):
    # ZZ-map: Mapa de características que introduce correlaciones entre los qubits
    for i in range(2):
        qml.Hadamard(wires=wires[i])  # Aplica Hadamard a cada qubit para crear superposición
    qml.RZ(2*x[0], wires=wires[0])   # Rotación alrededor de Z en el qubit 0, parametrizada por x[0]
    qml.RZ(2*x[1], wires=wires[1])   # Rotación alrededor de Z en el qubit 1, parametrizada por x[1]
    qml.CNOT(wires=[wires[0], wires[1]])    # CNOT para entrelazar los qubits
    qml.RZ(2*(np.pi - x[0])*(np.pi - x[1]), wires=wires[1])  # Rotación ZZ parametrizada
    qml.CNOT(wires=[wires[0], wires[1]])    # Segundo CNOT para completar la interacción ZZ

# Definimos un dispositivo de PennyLane para 2 qubits
dev = qml.device('default.qubit', wires=2)


# Usando nuestro feature map podemos generar F(x)|0>. Nuestro objetivo es calcular Re(<0|F(xj)adjF(xi)|0>).
# Esto es equivalente a aplicar el Hadamard Test sobre el operador U := F(xj)adjF(xi).
# El Hadamard Test nos permite calcular la parte real del elemento de matriz <0|U|0> usando un qubit ancilla.

import numpy as np

# Definimos dos vectores de entrada de ejemplo
x_i = [1, 2]
x_j = [1, -3]

# Definimos el operador U = F(xj) adjoint(F(xi))
def U(x_i, x_j, wires):
    """
    Aplica el feature map F(xj) seguido de la adjunta de F(xi) sobre los qubits indicados.
    wires: lista de los wires donde se aplica el feature map (deben ser 2 wires)
    """
    feature_map(x_j, wires)
    qml.adjoint(feature_map)(x_i, wires=wires)

# Creamos un dispositivo de PennyLane con 3 qubits:
# - wire 0: qubit ancilla para el Hadamard test
# - wires 1 y 2: qubits de datos
#
dev = qml.device("default.qubit", wires=3)

# Definimos el circuito Hadamard Test
def circuit(x_i, x_j):
    """
    Realiza el Hadamard Test para calcular Re(<0|F(xj)adjF(xi)|0>).
    x_i, x_j: vectores de entrada para el feature map
    """
    qml.Hadamard(wires=0)  # Prepara el qubit ancilla en superposición
    # Aplica el operador controlado U = F(xj) adjoint(F(xi)) sobre los wires 1 y 2, controlado por el ancilla (wire 0)
    qml.ctrl(U, control=0)(x_i, x_j, [1, 2])
    qml.Hadamard(wires=0)  # Hadamard final para interferencia
    return qml.expval(qml.PauliZ(wires=0))  # Mide la expectativa de Z en el ancilla

# Convertimos la función en un QNode para poder dibujar el circuito
draw_circuit = qml.draw_mpl(qml.QNode(circuit, dev))

# Dibujamos el circuito para los valores de ejemplo
draw_circuit(x_i, x_j)
plt.show()

# --- DIBUJO DEL FEATURE MAP SOLO ---
# Definimos un QNode para visualizar solo el feature map en los wires 0 y 1
feature_map_dev = qml.device('default.qubit', wires=2)

@qml.qnode(feature_map_dev)
def feature_map_circuit(x):
    feature_map(x, wires=[0, 1])
    return qml.state()

# Dibuja el circuito del feature map para un ejemplo
x_example = np.array([0.1, 0.2])
draw_feature_map = qml.draw_mpl(feature_map_circuit)
draw_feature_map(x_example)
plt.show()
# --- FIN DIBUJO FEATURE MAP ---
