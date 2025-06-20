#Without Qiskit
''' Compute expectation values of random single qubit quantum states in Python with respect to hamiltonian  𝑍 .'''


import numpy as np

# Define the function to create a random quantum state
def generate_random_state(num_qubits):
    random_state =np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)  # Random complex numbers
    random_state /= np.linalg.norm(random_state)  # Normalize the state to make it a unit vector
    return random_state

# Set number of qubits
n = 1
# Pauli Z matrix
pauli_z = np.array([[0, 1], [0, -1]])

# Define the function to compute the expectation value of Pauli Z
def compute_expectation_value(state, operator):
    e = np.real(np.conj(state) @ operator @ state)
    return e

for _ in range(10):
    random_state = generate_random_state(n)
    expectation_value = compute_expectation_value(random_state, pauli_z)
    print("Expectation value", expectation_value)



# Without Qiskit
''' Repeat Task 1 for the Hamiltonian  𝑋𝑋+𝑌𝑌+𝑍𝑍 , this time using Qiskit Statevector functionalities to compute the expectation value. '''
from qiskit.quantum_info import Statevector

# Set number of qubits
n = 2
op = SparsePauliOp(["XX", "YY", "ZZ"], [1.0, 1.0, 1.0])

for _ in range(10):
    random_state = generate_random_state(n)
    # Convert the state to a Qiskit Statevector
    statevector = Statevector(random_state)
    # Get the expectation value of the operator
    expectation_value = statevector.expectation_value(op)
    print("Expectation value:", expectation_value)
