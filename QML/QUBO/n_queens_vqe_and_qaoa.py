from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from docplex.mp.model import Model
from qiskit.circuit.library import RealAmplitudes

size = 8
lagrange = 1

# Crear el modelo con docplex
mdl = Model('eight_queens')

# Variables binarias x[i,j]
x = {(i,j): mdl.binary_var(name=f"x_{i}_{j}") for i in range(size) for j in range(size)}

# Función objetivo: minimizar sum x[i,j]
mdl.minimize(mdl.sum(-x[i,j] for i in range(size) for j in range(size)))

# Restricciones: no se atacan entre sí (fila, columna, diagonales)
for i1 in range(size):
    for i2 in range(size):
        for i3 in range(size):
            for i4 in range(size):
                if (i1 == i3 or i2 == i4 or i1 - i3 == i2 - i4 or i1 - i3 == i4 - i2) and not (i1 == i3 and i2 == i4):
                    mdl.add_constraint(x[i1,i2] + x[i3,i4] <= 1)

# Convertir a QuadraticProgram
qp = QuadraticProgram()
qp.from_docplex(mdl)

# Convertir a operador Ising
operator, offset = qp.to_ising()

# Instanciar el simulador
backend = Aer.get_backend('aer_simulator_statevector')
quantum_instance = QuantumInstance(backend)

# Optimizer clásico
optimizer = COBYLA(maxiter=250)

# --- Ejecutar QAOA ---
qaoa = QAOA(optimizer=optimizer, reps=1, quantum_instance=quantum_instance)
qaoa_solver = MinimumEigenOptimizer(qaoa)
result_qaoa = qaoa_solver.solve(qp)

print("QAOA Solution:")
print(result_qaoa)

# --- Ejecutar VQE con RealAmplitudes ---
ansatz = RealAmplitudes(operator.num_qubits, reps=3)
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
vqe_solver = MinimumEigenOptimizer(vqe)
result_vqe = vqe_solver.solve(qp)

print("VQE Solution:")
print(result_vqe)
