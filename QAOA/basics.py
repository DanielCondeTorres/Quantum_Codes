# For the Hamiltonian  XZ + YZ + IZ :
#
# Use diagonalization to compute  U(t,t0) = exp(-i H⋅(t−t0))  for  t0 = 0  and  t = 1.
#
# Implement the unitary using the formula  ∏_{k=1}^n U(t_k, t_{k−1})
# where  t_n = 1 ,  t_0 = 0 ,  t_k = k * t_n / n , and  n = 10 .
#
# Use HamiltonianGate to get the corresponding unitary at each time step.
#
# Verify that the two results you obtain are the same.


# Importamos las bibliotecas necesarias
import numpy as np
from scipy.linalg import eig  # Para diagonalización
from qiskit.circuit.library import HamiltonianGate  # Para evolución por pasos
from qiskit.quantum_info import SparsePauliOp  # Para construir el Hamiltoniano
from qiskit.quantum_info import Operator  # Para obtener la matriz unitaria

# Configuramos numpy para que muestre los números con más claridad
np.set_printoptions(precision=6, suppress=True)

# Definimos el Hamiltoniano H = XZ + YZ + ZI usando una representación esparsa
H = SparsePauliOp(["XZ", "YZ", "ZI"], [1, 1, 1])

# Definimos el tiempo total de evolución
t = 1

# Diagonalizamos el Hamiltoniano: H = P D P^{-1}
eigenvalues, eigenvectors = eig(H)  # Obtenemos autovalores y autovectores
P = eigenvectors                    # Matriz de autovectores
D = np.diag(eigenvalues)           # Matriz diagonal con autovalores
P_inv = np.linalg.inv(P)           # Inversa de la matriz de autovectores

# Calculamos U1 = exp(-i H t) = P exp(-i D t) P^{-1}
expD = np.diag(np.exp(-1j * t * np.diagonal(D)))  # Exponencial de la parte diagonal
U1 = P @ expD @ P_inv  # Transformación de vuelta a la base original

# Implementamos la evolución en n pasos pequeños usando HamiltonianGate
n = 10
U2 = Operator(HamiltonianGate(H, time=1/n))  # Primer paso

# Multiplicamos sucesivamente n-1 veces para completar la evolución total
for i in range(n - 1):
    U2 = Operator(HamiltonianGate(H, time=1/n)) @ U2

# Verificamos que ambas aproximaciones unitarias sean equivalentes
assert np.allclose(U1, U2)
