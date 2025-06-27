# =============================
# IMPORTACIÓN DE LIBRERÍAS
# =============================
import pennylane as qml  # Librería principal para computación cuántica híbrida
import matplotlib.pyplot as plt  # Para graficar el circuito cuántico
from pennylane import numpy as np  # Versión de numpy compatible con PennyLane para derivadas automáticas

# =============================
# DEFINICIÓN DE FUNCIONES CUÁNTICAS
# =============================
def angle_embedding(x, wires):
    """
    Codifica los datos clásicos de entrada (x) en el estado cuántico usando rotaciones RX.
    Cada valor de x rota un qubit diferente.
    x: lista o array de números (uno por qubit)
    wires: lista de índices de qubits
    
    IMPORTANTE:
    - Aquí usamos RX porque queremos transformar los datos clásicos en estados cuánticos.
    - RX(x[i]) toma el valor clásico x[i] y lo "mete" en el qubit i como una rotación.
    - Así, el circuito puede recibir y procesar información clásica.
    """
    for i in range(len(wires)):
        qml.RX(x[i], wires=i)  # Aplica una rotación RX al qubit i

def variational_block(theta, wires):
    """
    Aplica un bloque variacional: primero rota cada qubit con RY usando parámetros theta,
    luego entrelaza los qubits con puertas CNOT.
    theta: lista de parámetros (uno por qubit)
    wires: lista de índices de qubits
    
    IMPORTANTE:
    - Aquí usamos RY porque queremos que el circuito tenga parámetros entrenables.
    - RY(theta[i]) rota el qubit i con un ángulo que el modelo puede ajustar durante el entrenamiento.
    - Así, el circuito puede "aprender" a transformar los datos de entrada para resolver el problema.
    - Después, las CNOT entrelazan los qubits para crear correlaciones cuánticas.
    """
    for i in range(len(wires)):
        qml.RY(theta[i], wires=wires[i])  # Rota cada qubit con RY (parámetro entrenable)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=wires[i:i + 2])  # Entrelaza qubits adyacentes

# =============================
# CONFIGURACIÓN DEL DISPOSITIVO CUÁNTICO SIMULADO
# =============================
dev = qml.device("default.qubit", wires=3)  # Simulador de 3 qubits

# =============================
# DEFINICIÓN DEL CIRCUITO CUÁNTICO COMO QNODE
# =============================
@qml.qnode(dev)
def model(x, theta):
    """
    Define el circuito cuántico completo:
    1. Codifica los datos de entrada x en los qubits.
    2. Aplica el bloque variacional con parámetros theta.
    3. Devuelve el valor esperado del operador PauliZ en el primer qubit (medida cuántica).
    """
    angle_embedding(x, wires=range(3))
    variational_block(theta, wires=range(3))
    return qml.expval(qml.PauliZ(0))  # Medida del primer qubit

# =============================
# EJEMPLO DE USO DEL CIRCUITO
# =============================
x = [0.1, 0.2, 0.3]  # Datos de entrada de ejemplo (uno por qubit)
theta = [1.1, 0.2, -1.4]  # Parámetros variacionales de ejemplo

# Ejecuta el circuito y muestra el resultado de la medida
print("Resultado del circuito con valores de ejemplo:")
print(model(x, theta))

# =============================
# DIBUJO DEL CIRCUITO CUÁNTICO
# =============================
print("\nCircuito cuántico (texto):")
print(qml.draw(model)(x, theta))  # Dibujo en texto

# Dibujo bonito con matplotlib
fig, ax = qml.draw_mpl(model)(x, theta)
plt.show()

# =============================
# ENTRENAMIENTO DEL CIRCUITO CUÁNTICO (APRENDIZAJE AUTOMÁTICO)
# =============================
# Ahora vamos a "entrenar" los parámetros theta para que el circuito aprenda a aproximar una función objetivo.
# Usamos un conjunto de datos de entrenamiento muy pequeño (2 ejemplos) y un método de descenso de gradiente.

# Datos de entrenamiento: dos ejemplos, cada uno con 3 características (para 3 qubits)
X_train = np.array([[0.1, 0.3, 0.2], [0.2, -0.3, 0.5]])  # Datos de entrada
# Etiquetas objetivo: valores que queremos que el circuito aprenda a predecir
y_train = np.array([-1, 1])

# Inicializamos los parámetros theta como un array que permite el cálculo de gradientes
# (esto es necesario para que PennyLane pueda optimizarlos)
theta = np.array([1.1, 0.2, -1.4], requires_grad=True)

def error(theta, X, y):
    """
    Calcula el error cuadrático medio (MSE) entre las predicciones del circuito y las etiquetas reales.
    theta: parámetros variacionales (a optimizar)
    X: datos de entrada (array de ejemplos)
    y: etiquetas reales (array de valores objetivo)
    Devuelve el error promedio sobre todos los ejemplos.
    """
    err = 0
    for i in range(len(y)):
        pred = model(X[i], theta)  # Predicción del circuito para el ejemplo i
        err += (pred - y[i]) ** 2  # Suma el error cuadrático
    return err / len(y)  # Promedia el error

# Mostramos el error inicial antes de entrenar
print("\nError inicial antes de entrenar:", error(theta, X_train, y_train))

# Hiperparámetros del entrenamiento
epochs = 100  # Número de iteraciones de entrenamiento
lr = 0.5      # Tasa de aprendizaje (cuánto se ajustan los parámetros en cada paso)

# Calcula el gradiente de la función de error respecto a theta automáticamente
grad_function = qml.grad(error, argnum=0)

# Bucle de entrenamiento: ajusta theta para minimizar el error
for epoch in range(epochs):
    grad = grad_function(theta, X_train, y_train)  # Calcula el gradiente
    theta = theta - lr * grad  # Actualiza los parámetros en la dirección que reduce el error
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, error: {error(theta, X_train, y_train)}")  # Muestra el error cada 10 épocas

# Mostramos los parámetros finales tras el entrenamiento
print("\nTheta entrenada (parámetros finales): ", theta)

# =============================
# EVALUACIÓN SOBRE DATOS DE TEST
# =============================
# Definimos un pequeño conjunto de test (puedes cambiar estos valores)
X_test = np.array([[0.0, 0.0, 0.0], [0.5, -0.2, 0.1]])
y_test = np.array([-1, 1])  # Valores reales esperados (puedes cambiarlos según el problema)

print("\nEvaluación en datos de test:")
for i in range(len(X_test)):
    pred = model(X_test[i], theta)
    print(f"Test {i+1}: Entrada = {X_test[i]}, Real = {y_test[i]}, Predicción = {pred:.4f}")

# También puedes calcular el error medio en test si lo deseas:
def test_error(theta, X, y):
    err = 0
    for i in range(len(y)):
        pred = model(X[i], theta)
        err += (pred - y[i]) ** 2
    return err / len(y)

print("Error medio en test:", test_error(theta, X_test, y_test))

# =============================
# RESUMEN DEL SCRIPT
# =============================
# 1. Se define un circuito cuántico parametrizado (modelo variacional).
# 2. Se ejecuta el circuito con valores de ejemplo y se dibuja.
# 3. Se entrena el circuito para que aprenda a aproximar una función objetivo simple usando descenso de gradiente.
# 4. Se muestran los resultados y el progreso del entrenamiento.
#
# NOTA: Este es un ejemplo muy simple y didáctico. En la práctica, los circuitos y los conjuntos de datos pueden ser mucho más complejos.
# =============================
