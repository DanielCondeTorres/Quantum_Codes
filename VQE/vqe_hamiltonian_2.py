# Cálculo de ⟨ψ|H|ψ⟩ para el estado |ψ⟩ y el operador H dado.
# 
# Definimos:
# |ψ⟩ = (1/√3) |0⟩ + (√2/√3) |1⟩
# H = 2Y + Z - X
#
# Paso 1: Escribimos las matrices de Pauli:
# X = [[0, 1],
#      [1, 0]]
#
# Y = [[0, -i],
#      [i,  0]]
#
# Z = [[1,  0],
#      [0, -1]]
#
# Paso 2: Calculamos los términos por separado:
# H₁ = 2Y
# H₂ = Z
# H₃ = -X
#
# Paso 3: Escribimos el estado |ψ⟩ como vector:
# |ψ⟩ = [1/√3,
#        √2/√3]
#
# Paso 4: Calculamos ⟨ψ|H₁|ψ⟩, ⟨ψ|H₂|ψ⟩, ⟨ψ|H₃|ψ⟩ por separado
# usando producto matricial y conjugado transpuesto
#
# Finalmente, sumamos:
# ⟨ψ|H|ψ⟩ = ⟨ψ|H₁|ψ⟩ + ⟨ψ|H₂|ψ⟩ + ⟨ψ|H₃|ψ⟩

from qiskit import  QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
import math
stateprep = StatePreparation([1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)])
shots = 10000



# Calculate 2Y (need sdg, H for Y and 2, for 2Y)
qc = QuantumCircuit(1,1)
qc.append(stateprep, [0])
qc.sdg(0)
qc.h(0)
qc = transpile(qc, basis_gates=['u', 'cx'])
qc.measure(0,0)

job = AerSimulator().run(qc,shots=shots)
counts = job.result().get_counts(qc)
e1 = 2*(-1*counts["1"] / shots + 1*counts["0"] / shots)
print(counts, "Expectation value of H_1 is estimated as: ", e1)

# Calculate Z
qc = QuantumCircuit(1,1)
qc.append(stateprep, [0])
qc = transpile(qc, basis_gates=['u', 'cx'])
qc.measure(0,0)

job = AerSimulator().run(qc,shots=shots)
counts = job.result().get_counts(qc)
e2 = (-1*counts["1"] / shots + 1*counts["0"] / shots)
print(counts, "Expectation value of H_2 is estimated as: ", e2)

# Calculate X (we need H)
qc = QuantumCircuit(1,1)
qc.append(stateprep, [0])
qc.h(0)
qc = transpile(qc, basis_gates=['u', 'cx'])
qc.measure(0,0)

job = AerSimulator().run(qc,shots=shots)
counts = job.result().get_counts(qc)
e3 = -1*(-1*counts["1"] / shots + 1*counts["0"] / shots)
print(counts, "Expectation value of H_3 is estimated as: ", e3)

print("Estimated expectation value for H: ", e1+e2+e3)


# Now, other method is using expectation_value function of Statevector.
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

op = SparsePauliOp(["Y", "Z", "X"], [2,1, -1])
stateprep = StatePreparation([1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)])
statevector = Statevector(stateprep)
# Get the expectation value of the operator
expectation_value = statevector.expectation_value(op)
print("Expectation value:", expectation_value)
