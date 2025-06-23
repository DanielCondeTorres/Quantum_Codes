# Tarea 7: Cálculo del valor esperado de Z₀Z₂Z₃ con respecto al estado |0101⟩
#
# Paso 1: Recordamos que Z actúa como:
# Z|0⟩ = |0⟩ → eigenvalor +1
# Z|1⟩ = -|1⟩ → eigenvalor -1
#
# Paso 2: El operador Z₀Z₂Z₃ significa aplicar la compuerta Z al qubit 0, 2 y 3.
#
# Paso 3: Evaluamos el efecto de cada Zᵢ sobre el estado base |0101⟩:
# Qubit 0: está en estado |0⟩ → Z₀ contribuye con +1
# Qubit 2: está en estado |0⟩ → Z₂ contribuye con +1
# Qubit 3: está en estado |1⟩ → Z₃ contribuye con -1
#
# Paso 4: El valor esperado de Z₀Z₂Z₃ sobre |0101⟩ es simplemente el producto de los eigenvalores:
# ⟨0101|Z₀Z₂Z₃|0101⟩ = (+1) * (+1) * (-1) = -1
#
# Resultado: el valor esperado es -1


#ZOZ1 you need a CX connection between qubits
from qiskit import  QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

qc = QuantumCircuit(4,1)
qc.x(0)
qc.x(2)

# Measurement with Z0Z2Z3
qc.cx(0,2)
qc.cx(2,3)
qc.measure(3,0)

shots = 10000
job = AerSimulator().run(qc,shots=shots)
counts = job.result().get_counts(qc)
e = -1*counts.get("1",0) / shots + 1*counts.get("0",0) / shots
print(counts)
print("Expectation value of Z0Z2Z3 is estimated as: ", e)
