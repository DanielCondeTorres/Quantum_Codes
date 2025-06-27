from pennylane import numpy as np
import matplotlib.pyplot as plt

# =============================
# DATOS DE ENTRADA (xs) Y SALIDA (ys)
# =============================
# xs: valores de entrada (pueden verse como características o variables independientes)
# ys: valores de salida (lo que queremos que el modelo aprenda a predecir)
xs = np.array([-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1,  0.1,  0.3,  0.5, 0.7,  0.9,  1.1,  1.3,  1.5])
ys = np.array([2.33, 2.98, 2.81, 2.17, 1.41, 0.74, 0.27, 0.03, 0.03, 0.27, 0.74, 1.41, 2.17, 2.81, 2.98, 2.33])

# Visualizamos los datos originales
plt.scatter(xs, ys, label="Datos reales")
plt.title("Datos de entrada y salida")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

import pennylane as qml

# =============================
# CONFIGURACIÓN DEL DISPOSITIVO CUÁNTICO
# =============================
# Usamos un solo qubit (wires=1) y el simulador por defecto de PennyLane
# Esto es suficiente para este ejemplo sencillo

dev = qml.device("default.qubit", wires = 1)

# =============================
# DEFINICIÓN DEL CIRCUITO CUÁNTICO
# =============================
# Este circuito toma un parámetro theta0 y un dato de entrada x
# Aplica una rotación RY al qubit, donde el ángulo es theta0 * x
# Devuelve el valor esperado de PauliZ (medida cuántica)
@qml.qnode(dev)
def circuit(theta0, x):
    # Codificamos el dato x en el qubit usando una rotación RY parametrizada
    qml.RY(theta0 * x, wires = 0)
    # Medimos el valor esperado de PauliZ (resultado entre -1 y 1)
    return qml.expval(qml.PauliZ(wires = 0))

# =============================
# MODELO CUÁNTICO COMPLETO
# =============================
# El modelo cuántico toma tres parámetros:
#   - theta0: controla la rotación en el circuito
#   - theta1 y theta2: parámetros clásicos para ajustar la escala y el desplazamiento
# La salida del circuito se multiplica por theta1 y se le suma theta2
# Esto permite que el modelo se ajuste mejor a los datos

def q_model(theta0, theta1, theta2, x):
    return circuit(theta0, x) * theta1 + theta2

# =============================
# FUNCIÓN DE ERROR (PÉRDIDA)
# =============================
# Calcula el error cuadrático medio (RMSE) entre las predicciones del modelo y los valores reales
# Cuanto menor sea este error, mejor se ajusta el modelo a los datos

def q_error(theta0, theta1, theta2):
    er = 0
    for x, y in zip(xs, ys):
        er += (q_model(theta0, theta1, theta2, x) - y) ** 2
    return np.sqrt(er) / len(xs)

# =============================
# ENTRENAMIENTO DEL MODELO
# =============================
# Inicializamos los parámetros de forma aleatoria
# El objetivo es encontrar los valores de theta0, theta1 y theta2 que minimizan el error

theta0, theta1, theta2 = np.random.rand(3) * np.pi  # Inicialización aleatoria

# Calculamos el gradiente de la función de error respecto a cada parámetro
# Esto nos permite saber en qué dirección ajustar los parámetros para reducir el error
#
# IMPORTANTE SOBRE 'argnum':
# - 'argnum = [0,1,2]' le dice a qml.grad que calcule el gradiente respecto a los tres primeros argumentos de la función q_error,
#   es decir, respecto a theta0, theta1 y theta2.
# - Así, podemos actualizar estos parámetros durante el entrenamiento.
#
# SOBRE xs e ys:
# - xs e ys NO están como argumentos de la función q_error, sino que se usan como variables globales dentro de la función.
# - Esto es posible porque en Python, las funciones pueden acceder a variables definidas fuera de ellas (ámbito global).
# - Si quisiéramos usar otros datos, podríamos modificar xs e ys globalmente, o reescribir la función para que los reciba como argumentos.
# - En este ejemplo, se hace así para simplificar el código y el uso de gradientes.

gradient_fn_theta = qml.grad(q_error, argnum = [0,1,2])

lr = 0.9  # Tasa de aprendizaje: controla cuánto se ajustan los parámetros en cada paso

# Bucle de entrenamiento: ajusta los parámetros durante 101 épocas
for epoch in range(101):
    gradiente = gradient_fn_theta(theta0, theta1, theta2)
    # Actualizamos cada parámetro en la dirección opuesta al gradiente (descenso de gradiente)
    theta0 = theta0 - lr*gradiente[0]
    theta1 = theta1 - lr*gradiente[1]
    theta2 = theta2 - lr*gradiente[2]
    # Mostramos el error cada 5 épocas para ver el progreso
    if epoch % 5 == 0:
        print("epoch", epoch, "loss", q_error(theta0, theta1, theta2))

# =============================
# PREDICCIÓN Y VISUALIZACIÓN DE RESULTADOS
# =============================
# Usamos el modelo entrenado para predecir los valores de salida para cada x
# Dibujamos los datos reales y las predicciones para comparar visualmente

preds = [q_model(theta0, theta1, theta2, x) for x in xs]

plt.scatter(xs, ys, label="Datos reales")
plt.scatter(xs, preds, label="Predicción modelo cuántico")
plt.title("Ajuste del modelo cuántico a los datos")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# =============================
# VISUALIZACIÓN DEL CIRCUITO CUÁNTICO
# =============================
# Mostramos el circuito cuántico utilizado en el modelo con un ejemplo de parámetros

fig, ax = qml.draw_mpl(circuit)(theta0, xs[0])
plt.show()

# =============================
# RESUMEN DEL SCRIPT
# =============================
# 1. Se definen datos de entrada y salida (xs, ys).
# 2. Se construye un circuito cuántico sencillo con un solo qubit.
# 3. Se define un modelo cuántico con parámetros entrenables.
# 4. Se entrena el modelo para que aprenda a aproximar los datos.
# 5. Se visualizan los resultados para comparar el ajuste del modelo.
# =============================
