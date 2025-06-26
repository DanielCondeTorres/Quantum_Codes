import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ========================= QAOA Layer-wise Training =========================
# Este script implementa el entrenamiento progresivo (layer-wise) de QAOA para encontrar
# la energía máxima del siguiente Hamiltoniano en 5 qubits:
# H = Z1 - 0.3*Z2*Z3 + 0.4*Z0*Z3 + Z4 - 0.3*Z3*Z4

# Definimos el Hamiltoniano del problema
H = (
    qml.PauliZ(1)
    - 0.3 * qml.PauliZ(2) @ qml.PauliZ(3)
    + 0.4 * qml.PauliZ(0) @ qml.PauliZ(3)
    + qml.PauliZ(4)
    - 0.3 * qml.PauliZ(3) @ qml.PauliZ(4)
)

# Número de qubits y capas máximas
n_qubits = 5
p_max = 30  # Número máximo de capas QAOA (puedes ajustar este valor)

# Creamos el dispositivo cuántico
dev = qml.device("default.qubit", wires=list(range(n_qubits)))

def mixer_layer(beta, wires):
    """
    Aplica la capa mezcladora RX(2*beta) en cada qubit.
    Args:
        beta (float): Ángulo de rotación para la capa mezcladora.
        wires (list[int]): Lista de qubits sobre los que aplicar la rotación.
    """
    for i in wires:
        qml.RX(2 * beta, wires=i)

def qaoa_circuit(H, gammas, betas):
    """
    Circuito QAOA general para cualquier número de capas y qubits.
    Args:
        H (qml.Hamiltonian): Hamiltoniano del problema.
        gammas (list[float]): Ángulos para las capas de coste.
        betas (list[float]): Ángulos para las capas mezcladoras.
    Returns:
        float: Valor esperado del Hamiltoniano (energía esperada).
    """
    # Inicializamos en superposición uniforme
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    # Aplicamos las capas de QAOA
    for gamma, beta in zip(gammas, betas):
        qml.templates.ApproxTimeEvolution(H, gamma, 1)
        mixer_layer(beta, wires=range(n_qubits))
    # Devolvemos el valor esperado del Hamiltoniano
    return qml.expval(H)

# Creamos el QNode para entrenamiento
qnode = qml.QNode(qaoa_circuit, dev)

# Entrenamiento progresivo por capas (layer-wise)
gammas = []  # Lista de parámetros gamma óptimos por capa
betas = []   # Lista de parámetros beta óptimos por capa
energy_progress = []  # Guarda la energía máxima tras cada etapa

for p in range(1, p_max + 1):
    # Inicializamos los parámetros: los anteriores + uno nuevo para cada capa
    params_init = np.array(gammas + [0.1] + betas + [0.1])
    def cost(params):
        # Separamos los parámetros en gammas y betas
        g = params[:p]
        b = params[p:]
        # Negativo porque buscamos el máximo
        return -qnode(H, g, b)
    # Optimizamos los parámetros actuales
    sol = minimize(cost, params_init, method="COBYLA")
    # Guardamos los parámetros óptimos encontrados
    gammas = list(sol.x[:p])
    betas = list(sol.x[p:])
    # Calculamos la energía máxima alcanzada hasta ahora
    max_energy = qnode(H, gammas, betas)
    energy_progress.append(max_energy)
    print(f"Capas: {p}, Energía máxima: {max_energy:.6f}")

# Graficamos la evolución de la energía máxima con el número de capas
plt.figure()
plt.plot(range(1, p_max + 1), energy_progress, marker='o')
plt.xlabel('Número de capas QAOA (p)')
plt.ylabel('Energía máxima encontrada')
plt.title('Entrenamiento progresivo (layer-wise) de QAOA')
plt.grid()
plt.show()

# =========================
# Comentarios finales:
# - El entrenamiento por capas (layer-wise) consiste en optimizar primero una capa, luego añadir otra y optimizar solo los nuevos parámetros, manteniendo los anteriores.
# - Esto suele acelerar la convergencia y mejorar la calidad de la solución final.
# - El código imprime la energía máxima encontrada tras cada etapa y muestra su evolución.
