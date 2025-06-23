# Ejercicio:
# 
# Para el Hamiltoniano H = Z, obtén la matriz unitaria correspondiente usando Qiskit de tres formas distintas:
#
# 1. Implementa la compuerta Rz(θ) con el ángulo correspondiente, considerando que la evolución unitaria bajo H = Z
#    por un tiempo t está dada por U = exp(-i * Z * t) = Rz(2t), salvo una fase global.
#
# 2. Usa la clase HamiltonianGate de Qiskit, que toma como entrada un Hamiltoniano y un tiempo de evolución (duration),
#    y devuelve la matriz unitaria U = exp(-i * H * t).
#
# 3. Usa la clase PauliEvolutionGate de Qiskit, que también toma como entrada un Hamiltoniano en forma de operador de Pauli
#    y un tiempo de evolución, y construye la unidad correspondiente.


#1.
from qiskit import QuantumCircuit
from math import pi

qc = QuantumCircuit(1)
qc.rz(2*pi,0)
qcOp = Operator.from_circuit(qc)
qcOp.draw('latex')

#2.
from qiskit.circuit.library import HamiltonianGate
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from math import pi

H = SparsePauliOp(['Z'], coeffs=[1])
qc = QuantumCircuit(1)
qc.append(HamiltonianGate(H, pi),[0])
qcOp = Operator.from_circuit(qc)
qcOp.draw('latex')


#3.
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from math import pi

qc = QuantumCircuit(1)
op = SparsePauliOp(['Z'], coeffs=[1])
evo = PauliEvolutionGate(op, pi)
qc.append(evo, [0])
print(qc.decompose().draw())
qcOp = Operator.from_circuit(qc)
qcOp.draw('latex')
