# Dataset para clasificación binaria
# Este código genera un conjunto de datos sintético con dos clases
# distribuidas en cuatro clusters en un espacio 2D

import numpy as np
import matplotlib.pyplot as plt

# Número total de muestras a generar
n_samples = 200

# Generamos cuatro clusters de puntos usando distribuciones normales
# Cada cluster tiene n_samples/4 puntos (50 puntos en este caso)

# Cluster A: centrado en (0,2)
A = np.array([[np.random.normal(loc=0), np.random.normal(loc=2)] 
              for i in range(n_samples//4)])

# Cluster B: centrado en (2,-2)
B = np.array([[np.random.normal(loc=2), np.random.normal(loc=-2)] 
              for i in range(n_samples//4)])

# Cluster C: centrado en (2,2)
C = np.array([[np.random.normal(loc=2), np.random.normal(loc=2)] 
              for i in range(n_samples//4)])

# Cluster D: centrado en (-2,-2)
D = np.array([[np.random.normal(loc=-2), np.random.normal(loc=-2)] 
              for i in range(n_samples//4)])

# Combinamos todos los clusters en un solo array de features
features = np.concatenate([A,B,C,D], axis=0)

# Normalización al rango [-1, 1]
# 1. Encontramos el valor absoluto máximo para cada característica
max_abs_values = np.max(np.abs(features), axis=0)
print("\nValores máximos absolutos por característica:", max_abs_values)

# 2. Normalizamos dividiendo por el valor máximo absoluto
# Esto garantiza que todos los valores estén entre -1 y 1
features_normalized = features / max_abs_values

# Guardamos los parámetros de normalización para futuros datos
normalization_params = {
    'max_abs_values': max_abs_values
}

print("\nEstadísticas de la normalización:")
print(f"Valores máximos absolutos: {max_abs_values}")
print(f"Rango de valores después de normalizar:")
print(f"Min: {np.min(features_normalized, axis=0)}")
print(f"Max: {np.max(features_normalized, axis=0)}")

# Ahora features_normalized contiene nuestros datos normalizados en [-1, 1]
feature = features_normalized

# Creamos las etiquetas:
# -1 para los primeros n_samples/2 puntos (clusters A y B)
# +1 para los últimos n_samples/2 puntos (clusters C y D)
label = np.concatenate([-np.ones(n_samples//2), np.ones(n_samples//2)], axis=0)

# Combinamos features y labels en una lista de tuplas (feature, label)
data = list(zip(feature, label))

# Mezclamos aleatoriamente los datos para evitar sesgos en el entrenamiento
np.random.shuffle(data)

# Visualizamos los clusters normalizados
plt.figure(figsize=(12, 5))

# Subplot 1: Datos originales
plt.subplot(1, 2, 1)
plt.scatter(A[:,0], A[:,1], color='orange', label='Clase -1 (A)')
plt.scatter(B[:,0], B[:,1], color='orange', label='Clase -1 (B)')
plt.scatter(C[:,0], C[:,1], color='blue', label='Clase +1 (C)')
plt.scatter(D[:,0], D[:,1], color='blue', label='Clase +1 (D)')
plt.title('Datos Originales')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)

# Subplot 2: Datos normalizados
plt.subplot(1, 2, 2)
n = n_samples//4
plt.scatter(feature[:n,0], feature[:n,1], color='orange', label='Clase -1 (A)')
plt.scatter(feature[n:2*n,0], feature[n:2*n,1], color='orange', label='Clase -1 (B)')
plt.scatter(feature[2*n:3*n,0], feature[2*n:3*n,1], color='blue', label='Clase +1 (C)')
plt.scatter(feature[3*n:,0], feature[3*n:,1], color='blue', label='Clase +1 (D)')
plt.title('Datos Normalizados')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============= DIVISIÓN DEL DATASET EN ENTRENAMIENTO Y PRUEBA =============

# Porcentaje de datos que usaremos para entrenamiento (50% en este caso)
# Este valor determina cuántos datos usaremos para entrenar vs. probar el modelo
percentage_train = 0.5

# Calculamos el número exacto de muestras para entrenamiento
# m = 100 si n_samples = 200 y percentage_train = 0.5
m = int(percentage_train * n_samples)

# CONJUNTO DE ENTRENAMIENTO
# -----------------------
# x_train: Extraemos las características (coordenadas x,y) de los primeros m datos
# Usamos list comprehension para obtener data[i][0] (features) para i de 0 a m-1
x_train = np.array([data[i][0] for i in range(m)])

# y_train: Extraemos las etiquetas (-1 o +1) de los primeros m datos
# Usamos list comprehension para obtener data[i][1] (label) para i de 0 a m-1
y_train = np.array([data[i][1] for i in range(m)])

# CONJUNTO DE PRUEBA (TEST)
# -----------------------
# x_test: Extraemos las características de los datos restantes (desde m hasta el final)
# Usamos list comprehension para obtener data[i][0] para i de m a n_samples-1
x_test = np.array([data[i][0] for i in range(m, n_samples)])

# y_test: Extraemos las etiquetas de los datos restantes
# Usamos list comprehension para obtener data[i][1] para i de m a n_samples-1
y_test = np.array([data[i][1] for i in range(m, n_samples)])

# Verificación de las dimensiones de los conjuntos
print("\nDimensiones de los conjuntos de datos:")
print(f"Conjunto de entrenamiento (x_train): {x_train.shape}")
print(f"Etiquetas de entrenamiento (y_train): {y_train.shape}")
print(f"Conjunto de prueba (x_test): {x_test.shape}")
print(f"Etiquetas de prueba (y_test): {y_test.shape}")

import pennylane as qml
from pennylane import numpy as np
from time import time
from sklearn import svm

# ============= IMPLEMENTACIÓN DEL KERNEL CUÁNTICO =============

# Dimensión de los datos de entrada
n = len(x_train[0])

def feature_map(x, wires):
    """
    Implementa el ZZ feature map que codifica datos clásicos en estados cuánticos.
    
    Args:
        x (array): Vector de características de dimensión 2
        wires (list): Lista de qubits donde aplicar el feature map
    """
    # Primera capa de Hadamards para crear superposición
    for i in range(2):
        qml.Hadamard(wires=wires[i])
    
    # Rotaciones Z individuales
    qml.RZ(2*x[0], wires=wires[0])
    qml.RZ(2*x[1], wires=wires[1])
    
    # Entrelazamiento y rotación ZZ
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(2*(np.pi - x[0])*(np.pi - x[1]), wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

# Dispositivo cuántico con 2 qubits
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    Circuito que implementa el kernel cuántico usando el feature map.
    
    Args:
        x1, x2 (array): Vectores de características a comparar
        
    Returns:
        float: Valor del kernel entre x1 y x2
    """
    # Aplicamos el feature map y su adjunta
    feature_map(x2, wires=[0, 1])
    qml.adjoint(feature_map)(x1, wires=[0, 1])
    # Medimos la probabilidad del estado |00⟩
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

def quantum_kernel(x1, x2):
    """
    Función kernel que calcula la similitud entre dos vectores de características.
    
    Args:
        x1, x2 (array): Vectores de características a comparar
        
    Returns:
        float: Similitud entre x1 y x2 en el espacio de Hilbert
    """
    # El valor esperado ya viene como un número real
    result = kernel_circuit(x1, x2)
    # Normalizamos el resultado para que esté entre 0 y 1
    return (result + 1) / 2

# ============= ENTRENAMIENTO DEL CLASIFICADOR SVM =============

print('Calculando matriz de Gram para entrenamiento...')
# Usamos la implementación específica de PennyLane para la matriz de Gram
train_kernel_matrix = qml.kernels.square_kernel_matrix(x_train, quantum_kernel)

# Creamos y entrenamos el clasificador SVM
print("Entrenando SVM...")
clf = svm.SVC(kernel='precomputed')
clf.fit(train_kernel_matrix, y_train)

# ============= EVALUACIÓN DEL MODELO =============

print("Calculando matriz de kernel para test...")
# Para el test seguimos usando kernel_matrix ya que queremos comparar test con train
test_kernel_matrix = qml.kernels.kernel_matrix(x_test, x_train, quantum_kernel)

print("Realizando predicciones...")
predictions = clf.predict(test_kernel_matrix)

# Calculamos la precisión
accuracy = np.mean(predictions == y_test) * 100
print(f"Precisión del test: {accuracy:.2f}%")

# Visualizamos la matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()


print('Prediccion: ')

for i in range(len(x_test)):
    color = 'blue'
    if predictions[i] == -1:
        color = 'orange'
    plt.scatter(x_test[i,0],x_test[i,1],color=color)
plt.show()

print('Real: ')
for i in range(len(x_test)):
    color = 'blue'
    if y_test[i] == -1:
        color = 'orange'
    plt.scatter(x_test[i,0],x_test[i,1],color=color)
plt.show()
