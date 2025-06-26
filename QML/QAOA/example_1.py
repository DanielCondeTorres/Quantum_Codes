import pennylane as qml
import matplotlib.pyplot as plt

# Definimos el Hamiltoniano del problema para QAOA
# H = 0.1*Z0 - 2*Z1*Z0
H = 0.1 * qml.PauliZ(0) - 2 * qml.PauliZ(1) @ qml.PauliZ(0)

# Parámetros iniciales para QAOA (ángulos para las capas de coste y mezclador)
gammas = [0.3, 0.2]
betas = [0.2, 0.4]

# Configuramos un dispositivo cuántico de 2 qubits
dev = qml.device("default.qubit", wires=[0, 1])

def mixer_layer(beta, wires):
    """
    Aplica la capa mezcladora estándar de QAOA: una rotación RX(2*beta) en cada qubit.
    Args:
        beta (float): Ángulo de rotación para la capa mezcladora.
        wires (list[int]): Lista de qubits sobre los que aplicar la rotación.
    """
    for i in wires:
        qml.RX(2 * beta, wires=i)

@qml.qnode(dev)
def qaoa(H, gammas, betas):
    """
    Circuito QAOA para un Hamiltoniano H y listas de parámetros gammas y betas.
    Args:
        H (qml.Hamiltonian): Hamiltoniano de coste.
        gammas (list[float]): Ángulos para la unidad de coste.
        betas (list[float]): Ángulos para la unidad mezcladora.
    Returns:
        array: Probabilidades de medir cada estado base computacional.
    """
    # Inicializamos en superposición uniforme
    for i in range(2):
        qml.Hadamard(wires=i)
    # Aplicamos capas alternas de coste y mezclador
    for gamma, beta in zip(gammas, betas):
        # Capa de coste: evolución temporal bajo H durante gamma
        qml.templates.ApproxTimeEvolution(H, gamma, 1)
        # Capa mezcladora: rotaciones RX
        mixer_layer(beta, wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Ejecutamos el circuito QAOA y obtenemos las probabilidades de salida
output = qaoa(H, gammas, betas)

# Graficamos la distribución de salida
plt.bar(range(len(output)), output)
plt.xlabel('Estado base')
plt.ylabel('Probabilidad')
plt.title('Distribución de salida de QAOA')
plt.show()

# Ahora entrenamos los parámetros gamma y beta para mejorar la solución.
def value_solution(params):
    """
    Calcula el valor esperado del Hamiltoniano para un conjunto dado de parámetros.
    Args:
        params (list[float]): Lista de parámetros [gamma1, gamma2, beta1, beta2].
    Returns:
        float: Valor esperado del Hamiltoniano (energía esperada).
    """
    gammas = params[0:2]
    betas = params[2:]

    @qml.qnode(dev)
    def qaoa(H, gammas, betas):
        # Inicialización en superposición
        for i in range(2):
            qml.Hadamard(wires=i)
        # Capas alternas de coste y mezclador
        for gamma, beta in zip(gammas, betas):
            qml.templates.ApproxTimeEvolution(H, gamma, 1)
            mixer_layer(beta, wires=[0, 1])
        # Devolvemos el valor esperado del Hamiltoniano
        return qml.expval(H)

    return qaoa(H, gammas, betas)

from scipy.optimize import minimize

def cost(params):
    """
    Función de coste para la optimización: el negativo del valor esperado del Hamiltoniano.
    Args:
        params (list[float]): Parámetros del circuito QAOA.
    Returns:
        float: Negativo de la energía esperada (para minimizar).
    """
    return -value_solution(params)

# Unimos los parámetros iniciales en una sola lista
params = [*gammas, *betas]

# Ejecutamos la optimización usando el método COBYLA
sol = minimize(cost, params, method="COBYLA")
optimal_params = sol.x

# Ejecutamos el circuito QAOA con los parámetros óptimos encontrados
output = qaoa(H, optimal_params[:2], optimal_params[2:])

# Graficamos la nueva distribución de salida óptima
plt.bar(range(len(output)), output)
plt.xlabel('Estado base')
plt.ylabel('Probabilidad')
plt.title('Distribución óptima de salida de QAOA tras entrenamiento')
plt.show()
