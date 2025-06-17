import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator
import numpy as np

def myInvQFT_qiskit(n):
    qc = QuantumCircuit(n)
    # Fases y H exactamente igual a como en tu versión
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            angle = -np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
        qc.h(i)
    # Reversión de qubits: añade los swaps al final
    for j in range(n//2):
        qc.swap(j, n - j - 1)
    return qc
n = 4
qc_manual = myInvQFT_qiskit(n)
# Usando Qiskit directamente
qc_builtin = QFT(num_qubits=n, inverse=True, do_swaps=True)
print("my circuit for IQFT:")
print(qc_manual.draw())
print()
print("built-in circuit for IQFT:")
print(qc_builtin.draw())
print()
A = Operator(qc_manual).data
B = Operator(qc_builtin).data
print("Max diff:", np.max(np.abs(A - B)))
print("Coinciden?", np.allclose(A, B, atol=1e-8))
print(A)
