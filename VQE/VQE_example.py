
from qiskit import  QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter


# Funcion a minimizar
def func(params):
    # Define for the PQC
    theta_0 = Parameter('θ0')
    theta_1 = Parameter('θ1')

    # Create a Quantum Circuit with 2 qubits
    qc = QuantumCircuit(2)

    # Apply parameterized gates
    qc.rx(theta_0, 0)  # Apply RX gate with parameter θ₁ on qubit 0
    qc.ry(theta_1, 1)  # Apply RY gate with parameter θ₂ on qubit 1
    qc.cx(0, 1)        # Apply CNOT gate between qubits 0 and 1

    bc = qc.assign_parameters({theta_0: params[0], theta_1: params[1]})
    bc.measure_all()
    job = AerSimulator().run(bc,shots=1024)
    counts = job.result().get_counts(bc)
    prob = counts.get("00", 0) / 1024 # prob que queremos tener
    return 1-prob  # Queremos que la probabilidad de NO obtener el 00 sea minima


# Minimizámos
from scipy.optimize import minimize

initial_guess = [0.5, 0.5]

cost_function = func

result = minimize(cost_function, initial_guess, method='COBYLA')
print("Optimization results:", result)
print("Optimal parameters:", result.x)


# Comprobamos la solucion
from math import pi
from qiskit import  QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter

# Define parameters for the PQC
theta_0 = Parameter('θ0')
theta_1 = Parameter('θ1')

# Create a Quantum Circuit with 2 qubits
qc = QuantumCircuit(2)

# Apply parameterized gates
qc.rx(theta_0, 0)  # Apply RX gate with parameter θ₁ on qubit 0
qc.ry(theta_1, 1)  # Apply RY gate with parameter θ₂ on qubit 1
qc.cx(0, 1)        # Apply CNOT gate between qubits 0 and 1

bc = qc.assign_parameters({theta_0: result.x[0], theta_1: result.x[1]})
bc.measure_all()

job = AerSimulator().run(bc,shots=1024)
counts = job.result().get_counts(bc)
print(counts)
# o bien,
func(result.x)
