# Dataset para clasificación binaria
# Este código genera un conjunto de datos sintético con dos clases
# distribuidas en cuatro clusters en un espacio 2D

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot

# Número total de muestras a generar
n_samples = 200

# Generamos cuatro clusters de puntos usando distribuciones normales
# Cada cluster tiene n_samples/4 puntos (50 puntos en este caso)

# Cluster A: centrado en (0,2)
A = np.array([[np.random.normal(loc=-2), np.random.normal(loc=2)] 
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
max_abs_values = np.max(np.abs(features), axis=0)
features_normalized = features / max_abs_values

feature = features_normalized

# Creamos las etiquetas:
# -1 para los primeros n_samples/2 puntos (clusters A y B)
# +1 para los últimos n_samples/2 puntos (clusters C y D)
label = np.concatenate([-np.ones(n_samples//2), np.ones(n_samples//2)], axis=0)

# Visualización inicial
data_df = pd.DataFrame(feature, columns=['Feature_1', 'Feature_2'])
data_df['outcome'] = label

# Pairplot
sns.pairplot(data_df, hue='outcome', palette="tab10")
plt.suptitle('Pairplot del dataset sintético', y=1.02)
plt.show()

# Visualización de clusters
plt.figure(figsize=(12, 5))

# Subplot 1: Datos originales
plt.subplot(1, 2, 1)
plt.scatter(A[:,0], A[:,1], color='orange', label='Clase -1 (A)', alpha=0.7)
plt.scatter(B[:,0], B[:,1], color='orange', label='Clase -1 (B)', alpha=0.7)
plt.scatter(C[:,0], C[:,1], color='blue', label='Clase +1 (C)', alpha=0.7)
plt.scatter(D[:,0], D[:,1], color='blue', label='Clase +1 (D)', alpha=0.7)
plt.title('Datos Originales')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)

# Subplot 2: Datos normalizados
plt.subplot(1, 2, 2)
n = n_samples//4
plt.scatter(feature[:n,0], feature[:n,1], color='orange', label='Clase -1 (A)', alpha=0.7)
plt.scatter(feature[n:2*n,0], feature[n:2*n,1], color='orange', label='Clase -1 (B)', alpha=0.7)
plt.scatter(feature[2*n:3*n,0], feature[2*n:3*n,1], color='blue', label='Clase +1 (C)', alpha=0.7)
plt.scatter(feature[3*n:,0], feature[3*n:,1], color='blue', label='Clase +1 (D)', alpha=0.7)
plt.title('Datos Normalizados')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================
# CLASIFICACIÓN BINARIA: PREPARACIÓN DE DATOS
# =====================
# Convertimos las etiquetas de {-1, +1} a {0, 1} para one-hot encoding
label_bin = (label == 1).astype(int)

# Split de datos (estratificado)
percentage_train = 0.5
X_train, X_test, y_train, y_test = train_test_split(
    feature, label_bin, test_size=percentage_train, random_state=42, stratify=label_bin
)

# One-hot encoding para DOS clases (arreglado)
trainy = tf.one_hot(y_train, depth=2)
testy = tf.one_hot(y_test, depth=2)

print("\nDimensiones de los conjuntos de datos:")
print(f"Conjunto de entrenamiento (X_train): {X_train.shape}")
print(f"Etiquetas de entrenamiento (y_train): {y_train.shape}")
print(f"Etiquetas one-hot entrenamiento (trainy): {trainy.shape}")
print(f"Conjunto de prueba (X_test): {X_test.shape}")
print(f"Etiquetas de prueba (y_test): {y_test.shape}")
print(f"Etiquetas one-hot prueba (testy): {testy.shape}")

# =====================
# DEFINICIÓN DEL MODELO HÍBRIDO CUÁNTICO-CLÁSICO
# =====================
n_qubits = 2  # Igual que el número de features
layers = 1
output_dim = 2  # Dos clases

dev = qml.device("default.qubit", wires=n_qubits)

# QNode: circuito cuántico parametrizado
@qml.qnode(dev)
def qnode(inputs, weights):
    # Codificamos los datos clásicos en los qubits
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    # Aplicamos capas variacionales
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Medimos la expectativa de PauliZ en cada qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Definimos la forma de los pesos del circuito
weight_shapes = {"weights": (layers, n_qubits, 3)}

# Modelo Keras personalizado
class HybridQuantumClassifier(tf.keras.Model):
    def __init__(self, qnode, weight_shapes, output_dim, n_qubits, **kwargs):
        super().__init__(**kwargs)
        self.classical_input_layer = tf.keras.layers.Dense(n_qubits, activation='relu')
        self.quantum_layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
        self.classical_output_layer = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        x = self.classical_input_layer(inputs)
        x = self.quantum_layer(x)
        return self.classical_output_layer(x)

# Instanciamos el modelo
model = HybridQuantumClassifier(qnode, weight_shapes, output_dim, n_qubits)
model.build(input_shape=(None, X_train.shape[1]))
print(model.summary())

# =====================
# COMPILACIÓN Y ENTRENAMIENTO DEL MODELO
# =====================
# Usar optimizador legacy para evitar el warning en Mac M1/M2
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# Entrenamiento con callback para guardar métricas
history = model.fit(X_train, trainy, validation_data=(X_test, testy), epochs=30, batch_size=5)

# =====================
# VISUALIZACIÓN DE LOSS Y ACCURACY
# =====================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Loss')
plt.plot(history.history['loss'], label='train', linewidth=2)
plt.plot(history.history['val_loss'], label='val', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='val', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# =====================
# VISUALIZACIÓN CON IMSHOW Y COLORBAR - EVOLUCIÓN DE LA CLASIFICACIÓN
# =====================
plt.subplot(1, 3, 3)

# Crear una grilla para evaluar el modelo en todo el espacio
xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predecir en todos los puntos de la grilla
Z = model.predict(grid_points)
Z = Z[:, 1]  # Probabilidad de la clase 1
Z = Z.reshape(xx.shape)

# Crear el mapa de colores
im = plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='RdYlBu', alpha=0.8)
plt.colorbar(im, label='Probabilidad Clase +1')

# Superponer los puntos de datos
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                     edgecolors='black', s=50, alpha=0.9)
plt.title('Clasificación del Modelo Cuántico')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

plt.tight_layout()
plt.show()

# =====================
# MATRIZ DE CONFUSIÓN EN TEST
# =====================
predy = model.predict(X_test)
rounded_labels_pred = np.argmax(predy, axis=1)
rounded_labels_real = np.argmax(testy, axis=1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
cm = confusion_matrix(rounded_labels_real, rounded_labels_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=['Clase 0', 'Clase 1'])
cm_display.plot(cmap='Blues', ax=plt.gca())
plt.title('Matriz de Confusión (Test)')

# =====================
# EVOLUCIÓN DE LA CLASIFICACIÓN DURANTE EL ENTRENAMIENTO
# =====================
plt.subplot(1, 2, 2)

# Crear una representación de cómo mejora la accuracy
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Crear un mapa de colores que muestre la evolución
accuracy_matrix = np.array([train_acc, val_acc])
im = plt.imshow(accuracy_matrix, cmap='viridis', aspect='auto', interpolation='bilinear')
plt.colorbar(im, label='Accuracy')
plt.yticks([0, 1], ['Train', 'Validation'])
plt.xlabel('Época')
plt.title('Evolución de Accuracy')

# Añadir etiquetas de valores
for i in range(len(epochs)):
    if i % 5 == 0:  # Mostrar cada 5 épocas para no saturar
        plt.text(i, 0, f'{train_acc[i]:.2f}', ha='center', va='center', 
                color='white' if train_acc[i] < 0.5 else 'black', fontsize=8)
        plt.text(i, 1, f'{val_acc[i]:.2f}', ha='center', va='center', 
                color='white' if val_acc[i] < 0.5 else 'black', fontsize=8)

plt.tight_layout()
plt.show()

# =====================
# DIBUJO DEL CIRCUITO CUÁNTICO EJEMPLO
# =====================
example_input = X_train[0].astype(np.float32)
example_weights = np.random.uniform(0, 2*np.pi, (layers, n_qubits, 3))

plt.figure(figsize=(10, 6))
fig, ax = qml.draw_mpl(qnode)(example_input, example_weights)
plt.title('Diagrama del Circuito Cuántico')
plt.show()

# =====================
# MÉTRICAS FINALES
# =====================
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\n" + "="*50)
print("MÉTRICAS FINALES DEL MODELO")
print("="*50)
print(f"Accuracy de entrenamiento: {final_train_acc:.4f}")
print(f"Accuracy de validación: {final_val_acc:.4f}")
print(f"Loss de entrenamiento: {final_train_loss:.4f}")
print(f"Loss de validación: {final_val_loss:.4f}")
print("="*50)

# =====================
# VISUALIZACIÓN ADICIONAL: FRONTERAS DE DECISIÓN EN 3D
# =====================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# Plot 2D con fronteras de decisión
ax1 = plt.subplot(1, 2, 1)
xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = Z[:, 1].reshape(xx.shape)

contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
plt.colorbar(contour, label='Probabilidad Clase +1')
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                     edgecolors='black', s=50)
plt.title('Fronteras de Decisión 2D')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

# Plot 3D
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(xx, yy, Z, cmap='RdYlBu', alpha=0.8)
ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, c=y_test, cmap='RdYlBu', s=50)
ax2.set_title('Superficie de Decisión 3D')
ax2.set_xlabel('Característica 1')
ax2.set_ylabel('Característica 2')
ax2.set_zlabel('Probabilidad')

plt.tight_layout()
plt.show()
