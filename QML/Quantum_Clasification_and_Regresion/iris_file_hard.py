import matplotlib.pyplot as plt
import pennylane as qml
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# torch is not used in the TF model, so we can remove it for a cleaner setup
# import torch
import seaborn as sns
import pandas as pd

# ======================================
# DATA LOADING AND PREPROCESSING
# ======================================
iris = load_iris()
X = iris.data
# Normalize each feature to [0, 2π]
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min) * (2 * np.pi)
y = iris.target





# =============================
# VISUALIZACIÓN GLOBAL CON SEABORN.PAIRPLOT
# =============================
# Esta visualización muestra todas las relaciones entre las variables originales del dataset
# y colorea según la clase real (outcome)

# Creamos un DataFrame con los datos originales y la columna de clase
# Usamos los datos sin normalizar para que los ejes sean interpretables
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['outcome'] = iris.target

# Creamos el pairplot
sns.pairplot(iris_df, hue='outcome', palette="tab10")
plt.suptitle('Pairplot de todas las variables del dataset Iris', y=1.02)
plt.savefig('Pic.png', dpi=300, bbox_inches='tight')
plt.show()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42, stratify=y)

# One-hot encoding for multiclass classification, required by categorical_crossentropy loss
trainy = tf.one_hot(y_train, depth=3)
testy = tf.one_hot(y_test, depth=3)

# ======================================
# QUANTUM CIRCUIT AND HYBRID MODEL DEFINITION
# ======================================
n_qubits = 4 # Mismo que features
layers = 1
output_dim = 3  # Number of classes

dev = qml.device("default.qubit", wires=n_qubits)

# QNode: Parameterized quantum circuit
@qml.qnode(dev)
def qnode(inputs, weights):
    # Encode classical data into qubits
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    # Apply variational layers with strong entanglement
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure the expectation value of PauliZ on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



# Define the shape of the quantum circuit's weights
weight_shapes = {"weights": (layers, n_qubits, 3)}

# --- START OF ADAPTATION FOR KERAS 3 ---
# Create a custom Keras Model subclass
class HybridQuantumClassifier(tf.keras.Model):
    def __init__(self, qnode, weight_shapes, output_dim, n_qubits, **kwargs):
        super().__init__(**kwargs)
        # Classical input layer (optional, but good for feature transformation)
        self.classical_input_layer = tf.keras.layers.Dense(n_qubits, activation='relu')
        # Quantum layer using PennyLane's KerasLayer
        # Note: While KerasLayer still has Keras 2 legacy, using it within a custom Model
        # often works better with Keras 3's internal workings compared to Sequential.
        # PennyLane is actively working on native Keras 3 layers.
        self.quantum_layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
        # Classical output layer
        self.classical_output_layer = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        # Pass inputs through classical pre-processing
        x = self.classical_input_layer(inputs)
        # Pass through the quantum layer
        x = self.quantum_layer(x)
        # Pass through classical output layer for classification
        return self.classical_output_layer(x)

# Instantiate the custom model
model = HybridQuantumClassifier(qnode, weight_shapes, output_dim, n_qubits)

# Build the model explicitly to define input shape (important for custom models)
# The input shape is (None, 4) where None is batch size and 4 is the number of features.
model.build(input_shape=(None, X_train.shape[1]))
print(model.summary())
# --- END OF ADAPTATION FOR KERAS 3 ---

# ======================================
# MODEL COMPILATION AND TRAINING
# ======================================
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, trainy, validation_data=(X_test, testy), epochs=30, batch_size=5)

# ======================================
# VISUALIZATION OF LOSS AND ACCURACY
# ======================================
from matplotlib import pyplot
# Plot of loss during training
pyplot.figure(figsize=(10, 6))
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
# Plot of accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='val')
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

# ======================================
# CONFUSION MATRIX ON TEST SET
# ======================================
predy = model.predict(X_test)
rounded_labels_pred = np.argmax(predy, axis=1)
rounded_labels_real = np.argmax(testy, axis=1)
cm = confusion_matrix(rounded_labels_real, rounded_labels_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0,1,2])
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix (Test set)')
plt.show()

# ======================================
# PCA AND 2D VISUALIZATION OF TRAIN AND TEST
# ======================================
pca = PCA(n_components=2)
X_all = np.vstack([X_train, X_test])
X_pca = pca.fit_transform(X_all)
X_train_pca = X_pca[:len(X_train)]
X_test_pca = X_pca[len(X_train):]

colors = ['blue', 'red', 'green']
label_names = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(10, 7))
# Training points: X, real color
for i in range(len(X_train)):
    color = colors[y_train[i]]
    plt.scatter(X_train_pca[i,0], X_train_pca[i,1], color=color, marker='x', s=80, alpha=0.7)
# Test points: circle, predicted color
for i in range(len(X_test)):
    color_pred = colors[rounded_labels_pred[i]]
    # Red edge if misclassified
    edge = 'k' if rounded_labels_pred[i] == y_test[i] else 'red'
    plt.scatter(X_test_pca[i,0], X_test_pca[i,1], color=color_pred, marker='o', s=100, edgecolor=edge, linewidths=2, alpha=0.7)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='x', color='w', label='Train (real)', markerfacecolor='gray', markeredgecolor='k', markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='Test (predicted)', markerfacecolor='gray', markeredgecolor='k', markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='Misclassified', markerfacecolor='gray', markeredgecolor='red', markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='Setosa', markerfacecolor='blue', markeredgecolor='k', markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='Versicolor', markerfacecolor='red', markeredgecolor='k', markersize=10),
                  Line2D([0], [0], marker='o', color='w', label='Virginica', markerfacecolor='green', markeredgecolor='k', markersize=10)]
plt.legend(handles=legend_elements, loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA: X=Train (real color), O=Test (predicted color), red edge=error')
plt.grid(True)
plt.show()

# ======================================
# DRAWING THE PENNYLANE QUANTUM CIRCUIT WITH MATPLOTLIB
# ======================================
example_input = X_norm[0]
example_input = example_input.astype(np.float32)
example_weights = np.random.uniform(0, 2*np.pi, (layers, n_qubits, 3))

fig, ax = qml.draw_mpl(qnode)(example_input, example_weights)
plt.title('Quantum Circuit Diagram')
plt.show()
